"""
식품 패키지 디자인 트렌드 키워드 추출 파이프라인
===================================================
소셜미디어 데이터를 분석하여 가공식품 패키지 디자인에
직접 반영 가능한 트렌드 키워드 20개를 도출합니다.

사용법:
    python main.py                  # 일일 크롤링 모드
    python main.py --backfill       # 12개월 백필 (전체)
    python main.py --backfill --day 1  # 백필 1일차 (쿼리 분할)
    python main.py --analyze-only   # 수집 없이 BigQuery 데이터만으로 분석
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# .env 파일 로드 (.env 없으면 .env.example 사용)
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent / ".env.example"
load_dotenv(env_path)

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from collectors.naver_blog import collect_naver_blogs
from collectors.youtube import collect_youtube
from collectors.google_trends import get_google_trends
from collectors.naver_datalab import get_search_trend, get_shopping_keywords_trend
from processors.preprocessor import preprocess_documents
from processors.keyword_extractor import extract_tfidf_keywords, extract_keybert_keywords, extract_extended_keywords
from processors.design_filter import filter_design_keywords
from analyzers.trend_analyzer import analyze_trends
from storage.gcp_client import upload_to_gcs, insert_raw_contents, insert_trend_results, load_all_raw_contents, upload_trend_results_to_gcs
from config import (
    TARGET_PRODUCT, TARGET_CONSUMER, FINAL_KEYWORD_COUNT,
    DESIGN_CATEGORIES, BACKFILL_TOTAL_DAYS, get_search_queries,
    RECENT_WINDOW_DAYS, BLEND_RECENT_WEIGHT, BLEND_FULL_WEIGHT, RECENT_MIN_DOCS,
    KEYBERT_TOP_N,
)


def main(mode: str = "daily", day: int | None = None, analyze_only: bool = False):
    print("=" * 60)
    print("  식품 패키지 디자인 트렌드 키워드 추출 파이프라인")
    print("=" * 60)
    print(f"  타겟 제품군: {TARGET_PRODUCT}")
    print(f"  타겟 소비자: {TARGET_CONSUMER}")
    if analyze_only:
        print(f"  실행 모드: analyze-only (수집 없이 BigQuery 전체 분석)")
    else:
        print(f"  실행 모드: {mode}" + (f" (day {day}/{BACKFILL_TOTAL_DAYS})" if day else ""))
    print(f"  실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not analyze_only:
        # ── Phase 1: 데이터 수집 ──
        print("\n[Phase 1] 데이터 수집")
        print("-" * 40)

        search_queries = get_search_queries(mode, day=day)
        print(f"  검색 쿼리 {len(search_queries)}개 사용 (모드: {mode})")

        naver_data = collect_naver_blogs(search_queries, mode=mode)
        youtube_data = collect_youtube(search_queries, mode=mode)

        all_documents = naver_data + youtube_data
        print(f"\n  총 수집 문서: {len(all_documents)}건")

        # 원본 수집 데이터 저장 (GCP만)
        if all_documents:
            _save_to_gcp(naver_data, youtube_data)

        if not all_documents:
            print("\n[!] 수집된 데이터가 없습니다.")
            print("    .env 파일에 API 키를 설정해주세요.")
            print("    (.env.example 파일을 참고하세요)")
            if mode == "backfill":
                print("\n    backfill 모드: 수집할 데이터가 없어 종료합니다.")
                return
            print("\n    데모 모드로 실행합니다...\n")
            all_documents = _generate_demo_data()

        # ── backfill 모드: 수집 + 저장만 하고 종료 ──
        if mode == "backfill":
            print("\n[backfill] 수집 + 저장 완료. 분석은 --analyze-only로 실행하세요.")
            print("완료!")
            return

    # ── BigQuery에서 전체 데이터 로드 후 분석 ──
    print("\n[Phase 2] BigQuery 전체 데이터 로드")
    print("-" * 40)

    all_documents = load_all_raw_contents()
    if all_documents:
        print(f"  BigQuery 전체 데이터 {len(all_documents)}건으로 분석 실행")
    else:
        print("  [ERR] BigQuery 데이터 로드 실패")
        return

    # ── Phase 3: 전처리 ──
    print("\n[Phase 3] 전처리")
    print("-" * 40)

    processed_docs = preprocess_documents(all_documents)

    if not processed_docs:
        print("[!] 전처리 후 유효한 문서가 없습니다.")
        return

    # ── Phase 4: 키워드 추출 (2트랙 블렌딩) ──
    print("\n[Phase 4] 키워드 추출 (2트랙 블렌딩)")
    print("-" * 40)

    # 4-0. 최근/전체 문서 분리
    recent_docs, full_docs = _split_docs_by_recency(processed_docs, RECENT_WINDOW_DAYS)

    if len(recent_docs) >= RECENT_MIN_DOCS:
        # ── 트랙 1: 최근 7일 데이터 ──
        print(f"\n  [트랙 1] 최근 {RECENT_WINDOW_DAYS}일 데이터 ({len(recent_docs)}건)")
        recent_tfidf = extract_tfidf_keywords(recent_docs, min_df=1)
        recent_candidates = [kw for kw, _ in recent_tfidf]
        recent_keybert = extract_keybert_keywords(recent_docs, recent_candidates)
        print(f"  [트랙 1] KeyBERT 키워드: {[kw for kw, _ in recent_keybert[:10]]}...")

        # ── 트랙 2: 전체 데이터 ──
        print(f"\n  [트랙 2] 전체 데이터 ({len(full_docs)}건)")
        full_tfidf = extract_tfidf_keywords(full_docs)
        full_candidates = [kw for kw, _ in full_tfidf]
        full_keybert = extract_keybert_keywords(full_docs, full_candidates)
        print(f"  [트랙 2] KeyBERT 키워드: {[kw for kw, _ in full_keybert[:10]]}...")

        # ── 블렌딩 ──
        keybert_keywords = _blend_keywords(
            recent_keybert, full_keybert,
            BLEND_RECENT_WEIGHT, BLEND_FULL_WEIGHT,
        )
        print(f"\n  [블렌딩] 최종 키워드 ({BLEND_RECENT_WEIGHT:.0%} 최근 + {BLEND_FULL_WEIGHT:.0%} 전체):")
        print(f"    {[kw for kw, _ in keybert_keywords[:10]]}...")
    else:
        # ── 폴백: 최근 데이터 부족 → 전체 데이터만 사용 ──
        print(f"  [폴백] 최근 {RECENT_WINDOW_DAYS}일 문서 {len(recent_docs)}건 < {RECENT_MIN_DOCS}건 → 전체 데이터만 사용")
        tfidf_keywords = extract_tfidf_keywords(processed_docs)
        candidates = [kw for kw, _ in tfidf_keywords]
        keybert_keywords = extract_keybert_keywords(processed_docs, candidates)
        print(f"  KeyBERT 선정 키워드: {[kw for kw, _ in keybert_keywords[:10]]}...")

    # 4-3. 확장 키워드 추출 (명사구 N-gram) — 전체 데이터로 1회
    extended_keywords = extract_extended_keywords(processed_docs)
    if extended_keywords:
        print(f"  확장 키워드 예시: {[kw for kw, _ in extended_keywords[:10]]}...")

    # 4-4. 디자인 적합성 필터링
    print("\n[Phase 4-4] 디자인 적합성 필터링")
    print("-" * 40)

    design_keywords = filter_design_keywords(keybert_keywords)

    # ── Phase 5: 트렌드 분석 + 교차검증 ──
    print("\n[Phase 5] 트렌드 추이 분석")
    print("-" * 40)

    gt_keywords = [kw["keyword"] for kw in design_keywords[:30]]

    # 5-1. Google Trends 조회
    google_trends_df = None
    if gt_keywords:
        try:
            google_trends_df = get_google_trends(gt_keywords)
        except Exception as e:
            print(f"  [WARN] Google Trends 조회 실패: {e}")

    # 5-2. 네이버 데이터랩 - 검색어트렌드
    naver_search_df = None
    if gt_keywords:
        try:
            keyword_groups = [[kw] for kw in gt_keywords[:20]]
            naver_search_df = get_search_trend(keyword_groups)
        except Exception as e:
            print(f"  [WARN] 네이버 검색어트렌드 조회 실패: {e}")

    # 5-3. 네이버 데이터랩 - 쇼핑인사이트
    naver_shopping_df = None
    if gt_keywords:
        try:
            naver_shopping_df = get_shopping_keywords_trend(gt_keywords[:20])
        except Exception as e:
            print(f"  [WARN] 네이버 쇼핑인사이트 조회 실패: {e}")

    # ── Phase 6: 최종 키워드 20개 선정 ──
    print("\n[Phase 6] 최종 키워드 선정")
    print("-" * 40)

    final_keywords = analyze_trends(
        processed_docs, design_keywords, google_trends_df,
        naver_search_df, naver_shopping_df,
        extended_keywords=extended_keywords,
    )

    # ── 결과 출력 + 저장 ──
    _print_results(final_keywords)
    _save_results(final_keywords)

    # 분석 결과 저장 (BigQuery + GCS)
    print("\n[Phase 7] 분석 결과 저장")
    print("-" * 40)
    try:
        insert_trend_results(final_keywords)
    except Exception as e:
        print(f"  [WARN] BigQuery 결과 저장 실패: {e}")

    try:
        upload_trend_results_to_gcs(final_keywords)
    except Exception as e:
        print(f"  [WARN] GCS 결과 저장 실패: {e}")

    # ── Phase 8: 시각화 차트 생성 + PDF 보고서 + GCS 업로드 ──
    print("\n[Phase 8] 시각화 차트 + PDF 보고서 생성")
    print("-" * 40)

    charts_dir = Path(__file__).parent / "output" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(__file__).parent / "output"

    try:
        from visualize import (
            chart_top_keywords, chart_category_distribution,
            chart_trend_direction, chart_wordcloud, chart_monthly_frequency,
            generate_pdf_report, upload_pdf_to_gcs, _upload_charts_to_gcs,
        )
    except Exception as e:
        print(f"  [ERR] visualize 모듈 로드 실패: {e}")
        import traceback; traceback.print_exc()
        print("\n완료!")
        return

    # 개별 차트 생성 (하나 실패해도 나머지 진행)
    chart_funcs = [
        ("Top20 키워드", lambda: chart_top_keywords(final_keywords, charts_dir)),
        ("카테고리 분포", lambda: chart_category_distribution(final_keywords, charts_dir)),
        ("트렌드 방향", lambda: chart_trend_direction(final_keywords, charts_dir)),
        ("월별 빈도 추이", lambda: chart_monthly_frequency(final_keywords, processed_docs, charts_dir)),
        ("워드클라우드", lambda: chart_wordcloud(final_keywords, charts_dir)),
    ]
    for name, func in chart_funcs:
        try:
            func()
        except Exception as e:
            print(f"  [WARN] {name} 차트 생성 실패: {e}")
            import traceback; traceback.print_exc()

    # 차트 PNG → GCS 업로드
    chart_files = list(charts_dir.glob("*.png"))
    print(f"  [차트] 로컬 PNG {len(chart_files)}개 생성됨: {[f.name for f in chart_files]}")
    try:
        _upload_charts_to_gcs(chart_files)
    except Exception as e:
        print(f"  [ERR] GCS 차트 업로드 실패: {e}")
        import traceback; traceback.print_exc()

    # PDF 보고서 생성 + GCS 업로드
    try:
        pdf_path = generate_pdf_report(final_keywords, charts_dir, output_dir)
        if pdf_path:
            upload_pdf_to_gcs(pdf_path)
    except Exception as e:
        print(f"  [WARN] PDF 생성/업로드 실패: {e}")
        import traceback; traceback.print_exc()

    print("\n완료!")


def _print_results(keywords: list[dict]):
    """최종 결과 콘솔 출력"""
    print("\n")
    print("=" * 70)
    print(f"  최종 트렌드 키워드 TOP {len(keywords)}")
    print("=" * 70)
    print(f"{'순위':>4} | {'키워드':<12} | {'확장 키워드':<24} | {'카테고리':<12} | {'트렌드':>6} | 근거")
    print("-" * 90)

    for kw in keywords:
        ext = kw.get("extended_keyword", kw["keyword"])
        print(
            f"{kw['rank']:>4} | {kw['keyword']:<12} | {ext:<24} | {kw['category']:<12} "
            f"| {kw.get('trend_symbol', '→')} {kw['trend']:<4} | {kw['reason']}"
        )

    print("=" * 90)

    # 카테고리별 분포
    from collections import Counter
    cat_counts = Counter(kw["category"] for kw in keywords)
    print("\n  [카테고리 분포]")
    for cat, count in cat_counts.most_common():
        bar = "#" * count
        print(f"    {cat:<12}: {bar} ({count}개)")


def _save_results(keywords: list[dict]):
    """결과를 파일로 저장"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 저장
    csv_path = output_dir / f"trend_keywords_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "keyword", "extended_keyword", "category", "trend", "reason"])
        writer.writeheader()
        for kw in keywords:
            writer.writerow({
                "rank": kw["rank"],
                "keyword": kw["keyword"],
                "extended_keyword": kw.get("extended_keyword", kw["keyword"]),
                "category": kw["category"],
                "trend": kw["trend"],
                "reason": kw["reason"],
            })
    print(f"\n  CSV 저장: {csv_path}")

    # 텍스트 요약 저장
    txt_path = output_dir / f"trend_keywords_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"식품 패키지 디자인 트렌드 키워드 TOP {len(keywords)}\n")
        f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 50 + "\n\n")
        for kw in keywords:
            ext = kw.get("extended_keyword", kw["keyword"])
            f.write(f"{kw['rank']:>2}. {kw['keyword']} → {ext} [{kw['category']}] - {kw['trend']}\n")
        f.write("\n\n키워드 리스트:\n")
        keyword_list = ", ".join(f'"{kw["keyword"]}"' for kw in keywords)
        f.write(keyword_list + "\n")
        f.write("\n확장 키워드 리스트:\n")
        ext_list = ", ".join(f'"{kw.get("extended_keyword", kw["keyword"])}"' for kw in keywords)
        f.write(keyword_list + "\n")
    print(f"  TXT 저장: {txt_path}")


def _save_raw_data(naver_data: list[dict], youtube_data: list[dict]):
    """수집된 원본 데이터를 파일로 저장"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON으로 전체 원본 저장
    raw_path = output_dir / f"raw_data_{timestamp}.json"
    raw = {
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "naver_blog_count": len(naver_data),
        "youtube_count": len(youtube_data),
        "total_count": len(naver_data) + len(youtube_data),
        "naver_blog": naver_data,
        "youtube": youtube_data,
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    # CSV로도 저장 (엑셀에서 바로 열 수 있게)
    csv_path = output_dir / f"raw_data_{timestamp}.csv"
    all_docs = naver_data + youtube_data
    if all_docs:
        fieldnames = ["source", "title", "description", "date", "link", "query"]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for doc in all_docs:
                row = {
                    "source": doc.get("source", ""),
                    "title": doc.get("title", ""),
                    "description": doc.get("description", ""),
                    "date": doc.get("postdate", doc.get("published_at", "")),
                    "link": doc.get("link", doc.get("video_id", "")),
                    "query": doc.get("query", ""),
                }
                writer.writerow(row)

    print(f"\n  [원본 데이터 저장]")
    print(f"    JSON: {raw_path}")
    print(f"    CSV:  {csv_path}")


def _save_to_gcp(naver_data: list[dict], youtube_data: list[dict]):
    """수집 데이터를 GCS 백업 + BigQuery INSERT"""
    print("\n  [GCP 저장]")

    # GCS 백업
    try:
        if naver_data:
            upload_to_gcs(naver_data, "naver")
        if youtube_data:
            upload_to_gcs(youtube_data, "youtube")
    except Exception as e:
        print(f"  [WARN] GCS 업로드 실패: {e}")

    # BigQuery INSERT (중복 제외)
    try:
        all_docs = naver_data + youtube_data
        insert_raw_contents(all_docs)
    except Exception as e:
        print(f"  [WARN] BigQuery INSERT 실패: {e}")


def _split_docs_by_recency(
    docs: list[dict], days: int = 7
) -> tuple[list[dict], list[dict]]:
    """문서를 최근 N일과 전체로 분리.

    Returns:
        (recent_docs, full_docs) — full_docs는 전체 문서 (recent 포함)
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent = [doc for doc in docs if doc.get("date", "") >= cutoff]
    print(f"  [문서 분리] 전체 {len(docs)}건 중 최근 {days}일: {len(recent)}건")
    return recent, docs


def _blend_keywords(
    recent_kw: list[tuple[str, float]],
    full_kw: list[tuple[str, float]],
    recent_weight: float = 0.6,
    full_weight: float = 0.4,
) -> list[tuple[str, float]]:
    """두 트랙의 키워드 점수를 정규화 후 가중 블렌딩.

    Returns:
        블렌딩된 [(키워드, 점수), ...] 상위 KEYBERT_TOP_N개
    """
    def _normalize(kw_list: list[tuple[str, float]]) -> dict[str, float]:
        if not kw_list:
            return {}
        max_score = max(score for _, score in kw_list) if kw_list else 1.0
        if max_score == 0:
            max_score = 1.0
        return {kw: score / max_score for kw, score in kw_list}

    recent_map = _normalize(recent_kw)
    full_map = _normalize(full_kw)

    all_keywords = set(recent_map.keys()) | set(full_map.keys())

    blended = []
    for kw in all_keywords:
        r_score = recent_map.get(kw, 0.0)
        f_score = full_map.get(kw, 0.0)
        blended_score = r_score * recent_weight + f_score * full_weight
        blended.append((kw, blended_score))

    blended.sort(key=lambda x: x[1], reverse=True)

    # 최근 트랙에만 있는 키워드 수 로깅
    recent_only = set(recent_map.keys()) - set(full_map.keys())
    if recent_only:
        print(f"  [블렌딩] 최근에만 등장한 새 키워드 {len(recent_only)}개: {list(recent_only)[:5]}...")

    return blended[:KEYBERT_TOP_N]


def _generate_demo_data() -> list[dict]:
    """API 키 없을 때 사용할 데모 데이터 (DESIGN_CATEGORIES 기반 동적 생성)"""
    # DESIGN_CATEGORIES에서 샘플 키워드 추출
    all_keywords = []
    for cat, words in DESIGN_CATEGORIES.items():
        all_keywords.extend(words[:8])

    # 데모 포스트 동적 생성
    demo_posts = []
    months = [
        "20250301", "20250315", "20250401", "20250420",
        "20250510", "20250601", "20250715", "20250801",
        "20250910", "20251005", "20251115", "20251201",
        "20260101", "20260110", "20260115", "20260120",
        "20260125", "20260130", "20260201", "20260205",
        "20260208", "20260210", "20260212", "20260214",
        "20260215",
    ]

    for i, date in enumerate(months):
        # 랜덤하게 3~5개 키워드 조합
        sample_size = min(random.randint(3, 5), len(all_keywords))
        selected = random.sample(all_keywords, sample_size)

        demo_posts.append({
            "title": f"{TARGET_PRODUCT} 트렌드 키워드 - {' '.join(selected[:2])}",
            "description": " ".join(selected) + f" {TARGET_PRODUCT} 패키지 디자인 트렌드",
            "postdate": date,
            "source": "naver_blog",
        })

    print(f"  [데모] {len(demo_posts)}건의 샘플 데이터를 사용합니다")
    return demo_posts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="식품 패키지 디자인 트렌드 키워드 추출 파이프라인"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="12개월 백필 모드 (모든 시즌 쿼리 포함)",
    )
    parser.add_argument(
        "--day",
        type=int,
        choices=range(1, BACKFILL_TOTAL_DAYS + 1),
        default=None,
        help=f"백필 분할 실행 일차 (1~{BACKFILL_TOTAL_DAYS}). YouTube 쿼터 분산용.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="수집 없이 BigQuery 전체 데이터만으로 트렌드 분석 실행",
    )
    args = parser.parse_args()

    if args.analyze_only:
        main(mode="daily", analyze_only=True)
    else:
        run_mode = "backfill" if args.backfill else "daily"

        if args.day and not args.backfill:
            parser.error("--day는 --backfill과 함께 사용해야 합니다.")

        main(mode=run_mode, day=args.day)
