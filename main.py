"""
식품 패키지 디자인 트렌드 키워드 추출 파이프라인
===================================================
소셜미디어 데이터를 분석하여 가공식품 패키지 디자인에
직접 반영 가능한 트렌드 키워드 20개를 도출합니다.

사용법:
    python main.py
"""

import csv
import json
import os
import sys
from datetime import datetime
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
from processors.keyword_extractor import extract_tfidf_keywords, extract_keybert_keywords
from processors.design_filter import filter_design_keywords
from analyzers.trend_analyzer import analyze_trends
from config import TARGET_PRODUCT, TARGET_CONSUMER, FINAL_KEYWORD_COUNT


def main():
    print("=" * 60)
    print("  식품 패키지 디자인 트렌드 키워드 추출 파이프라인")
    print("=" * 60)
    print(f"  타겟 제품군: {TARGET_PRODUCT}")
    print(f"  타겟 소비자: {TARGET_CONSUMER}")
    print(f"  실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Phase 1: 데이터 수집 ──
    print("\n[Phase 1] 데이터 수집")
    print("-" * 40)

    naver_data = collect_naver_blogs()
    youtube_data = collect_youtube()

    all_documents = naver_data + youtube_data
    print(f"\n  총 수집 문서: {len(all_documents)}건")

    # 원본 수집 데이터 저장
    if all_documents:
        _save_raw_data(naver_data, youtube_data)

    if not all_documents:
        print("\n[!] 수집된 데이터가 없습니다.")
        print("    .env 파일에 API 키를 설정해주세요.")
        print("    (.env.example 파일을 참고하세요)")
        print("\n    데모 모드로 실행합니다...\n")
        all_documents = _generate_demo_data()

    # ── Phase 2: 전처리 ──
    print("\n[Phase 2] 전처리")
    print("-" * 40)

    processed_docs = preprocess_documents(all_documents)

    if not processed_docs:
        print("[!] 전처리 후 유효한 문서가 없습니다.")
        return

    # ── Phase 3: 키워드 추출 ──
    print("\n[Phase 3] 키워드 추출")
    print("-" * 40)

    # 3-1. TF-IDF
    tfidf_keywords = extract_tfidf_keywords(processed_docs)
    print(f"  TF-IDF 상위 키워드: {[kw for kw, _ in tfidf_keywords[:10]]}...")

    # 3-2. KeyBERT
    candidates = [kw for kw, _ in tfidf_keywords]
    keybert_keywords = extract_keybert_keywords(processed_docs, candidates)
    print(f"  KeyBERT 선정 키워드: {[kw for kw, _ in keybert_keywords[:10]]}...")

    # 3-3. 디자인 적합성 필터링
    print("\n[Phase 3-3] 디자인 적합성 필터링")
    print("-" * 40)

    design_keywords = filter_design_keywords(keybert_keywords)

    # ── Phase 4: 트렌드 분석 + 교차검증 ──
    print("\n[Phase 4] 트렌드 추이 분석")
    print("-" * 40)

    gt_keywords = [kw["keyword"] for kw in design_keywords[:30]]

    # 4-1. Google Trends 조회
    google_trends_df = None
    if gt_keywords:
        try:
            google_trends_df = get_google_trends(gt_keywords)
        except Exception as e:
            print(f"  [WARN] Google Trends 조회 실패: {e}")

    # 4-2. 네이버 데이터랩 - 검색어트렌드
    naver_search_df = None
    if gt_keywords:
        try:
            keyword_groups = [[kw] for kw in gt_keywords[:20]]
            naver_search_df = get_search_trend(keyword_groups)
        except Exception as e:
            print(f"  [WARN] 네이버 검색어트렌드 조회 실패: {e}")

    # 4-3. 네이버 데이터랩 - 쇼핑인사이트
    naver_shopping_df = None
    if gt_keywords:
        try:
            naver_shopping_df = get_shopping_keywords_trend(gt_keywords[:20])
        except Exception as e:
            print(f"  [WARN] 네이버 쇼핑인사이트 조회 실패: {e}")

    # ── Phase 5: 최종 키워드 20개 선정 ──
    print("\n[Phase 5] 최종 키워드 선정")
    print("-" * 40)

    final_keywords = analyze_trends(
        processed_docs, design_keywords, google_trends_df,
        naver_search_df, naver_shopping_df,
    )

    # ── 결과 출력 ──
    _print_results(final_keywords)
    _save_results(final_keywords)

    print("\n완료!")


def _print_results(keywords: list[dict]):
    """최종 결과 콘솔 출력"""
    print("\n")
    print("=" * 70)
    print(f"  최종 트렌드 키워드 TOP {len(keywords)}")
    print("=" * 70)
    print(f"{'순위':>4} | {'키워드':<14} | {'카테고리':<12} | {'트렌드':>6} | 근거")
    print("-" * 70)

    for kw in keywords:
        print(
            f"{kw['rank']:>4} | {kw['keyword']:<14} | {kw['category']:<12} "
            f"| {kw.get('trend_symbol', '→')} {kw['trend']:<4} | {kw['reason']}"
        )

    print("=" * 70)

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
        writer = csv.DictWriter(f, fieldnames=["rank", "keyword", "category", "trend", "reason"])
        writer.writeheader()
        for kw in keywords:
            writer.writerow({
                "rank": kw["rank"],
                "keyword": kw["keyword"],
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
            f.write(f"{kw['rank']:>2}. {kw['keyword']} [{kw['category']}] - {kw['trend']}\n")
        f.write("\n\n키워드 리스트:\n")
        keyword_list = ", ".join(f'"{kw["keyword"]}"' for kw in keywords)
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


def _generate_demo_data() -> list[dict]:
    """API 키 없을 때 사용할 데모 데이터"""
    demo_posts = [
        {"title": "요즘 핫한 말차 과자 패키지 진짜 예쁘다", "description": "말차 그린톤의 미니멀한 디자인이 눈에 띄네요 뉴트로 감성도 살짝 들어가 있고 프리미엄 느낌", "postdate": "20250110", "source": "naver_blog"},
        {"title": "두바이초콜릿 시리즈 신상 과자 리뷰", "description": "두바이초콜릿 컨셉 패키지 고급스러운 골드톤 크래프트지 소재로 되어있어요 프리미엄 라인", "postdate": "20250115", "source": "naver_blog"},
        {"title": "편의점 신상 제로슈거 스낵 모음", "description": "제로슈거 저당 과자 미니사이즈로 소분되어 있고 클린라벨 깔끔한 패키지 디자인 헬시플레저 트렌드 반영", "postdate": "20250120", "source": "naver_blog"},
        {"title": "발렌타인데이 한정판 과자 패키지 모음", "description": "발렌타인데이 시즌 핑크톤 파스텔 일러스트 Y2K 감성 귀여운 패키지 선물용", "postdate": "20250201", "source": "naver_blog"},
        {"title": "비건 과자 패키지 디자인 트렌드", "description": "비건 친환경 재활용 종이포장 어스톤 컬러 미니멀리즘 자연 감성 캠핑 피크닉용", "postdate": "20250105", "source": "naver_blog"},
        {"title": "2025 과자 패키지 트렌드 뉴트로", "description": "뉴트로 레트로 감성 복고풍 디자인 과자 패키지 Y2K 색감 그라데이션 파스텔톤 조합", "postdate": "20250108", "source": "naver_blog"},
        {"title": "프로틴 과자 신상 리뷰", "description": "프로틴 고단백 스낵 운동 후 간식 스탠딩파우치 지퍼백 형태 프리미엄 패키지", "postdate": "20250125", "source": "naver_blog"},
        {"title": "벚꽃 시즌 한정 과자 모음", "description": "벚꽃 시즌 핑크 파스텔 봄 감성 과자 패키지 콜라보 한정판 예쁜 디자인", "postdate": "20250301", "source": "naver_blog"},
        {"title": "홈카페 과자 추천", "description": "홈카페 간식 말차 쿠키 크래프트지 포장 미니사이즈 소분 개별포장 감성 패키지", "postdate": "20250112", "source": "naver_blog"},
        {"title": "흑임자 과자 트렌드", "description": "흑임자 블랙 톤 고급스러운 무광 패키지 한국적 감성 프리미엄 라인 미니멀 디자인", "postdate": "20250118", "source": "naver_blog"},
        {"title": "캠핑 스낵 패키지 리뷰", "description": "캠핑용 과자 이지컷 소분 포장 스탠딩파우치 아웃도어 감성 투명 창이 있어서 좋아요", "postdate": "20250130", "source": "naver_blog"},
        {"title": "글루텐프리 과자 신상", "description": "글루텐프리 건강 스낵 클린라벨 깔끔한 산세리프 폰트 화이트 톤 미니멀 디자인", "postdate": "20250205", "source": "naver_blog"},
        {"title": "마라맛 과자 시리즈", "description": "마라 매운맛 레드톤 강렬한 패키지 디자인 불꽃 그래픽 MZ세대 타겟 플렉스 느낌", "postdate": "20250210", "source": "naver_blog"},
        {"title": "편의점 콜라보 과자 한정판", "description": "콜라보 한정판 과자 특별한 패키지 콜렉터블 디자인 시즌 이벤트 굿즈 느낌", "postdate": "20250215", "source": "naver_blog"},
        {"title": "다이어트 과자 제로칼로리", "description": "다이어트 제로칼로리 스낵 헬시플레저 라이트한 패키지 파스텔 블루 그린 톤 깔끔한 느낌", "postdate": "20250220", "source": "naver_blog"},
        {"title": "크리스마스 한정 과자 패키지", "description": "크리스마스 레드 그린 겨울 시즌 한정판 과자 선물세트 프리미엄 포장 리본", "postdate": "20241225", "source": "naver_blog"},
        {"title": "오트밀 쿠키 신상 리뷰", "description": "오트밀 건강한 간식 크래프트지 자연 감성 홈카페 스타일 소분 패키지", "postdate": "20250128", "source": "naver_blog"},
        {"title": "유광 vs 무광 패키지 비교", "description": "유광 패키지 화려한 느낌 무광 패키지 고급스러운 프리미엄 감성 미니멀 트렌드", "postdate": "20250203", "source": "naver_blog"},
        {"title": "편의점 신상 과자 2025", "description": "편의점 신상 과자 말차 두바이초콜릿 제로슈거 프로틴 미니사이즈 트렌드 키워드 총정리", "postdate": "20250212", "source": "naver_blog"},
        {"title": "혼술 안주 과자 패키지", "description": "혼술 안주용 스낵 블랙톤 시크한 패키지 어스톤 감성 미니멀 지퍼백 재밀봉", "postdate": "20250207", "source": "naver_blog"},
        {"title": "콤부차 맛 과자 출시", "description": "콤부차 과자 새로운 맛 파스텔 그린 패키지 건강 트렌드 프리미엄 라인", "postdate": "20250218", "source": "naver_blog"},
        {"title": "과자 트렌드 2025 총정리", "description": "말차 두바이초콜릿 제로슈거 흑임자 프로틴 글루텐프리 미니사이즈 뉴트로 Y2K 헬시플레저 비건 소분 크래프트지", "postdate": "20250225", "source": "naver_blog"},
        {"title": "아사이 과자 신상 리뷰", "description": "아사이 보라색 퍼플톤 건강한 이미지 프리미엄 패키지 그라데이션 디자인", "postdate": "20250222", "source": "naver_blog"},
        {"title": "제로웨이스트 과자 포장", "description": "제로웨이스트 친환경 종이포장 재활용 비건 감성 어스톤 크래프트지 소재", "postdate": "20250115", "source": "naver_blog"},
        {"title": "빼빼로데이 과자 패키지", "description": "빼빼로데이 특별 패키지 핑크 하트 일러스트 귀여운 디자인 시즌 이벤트", "postdate": "20241111", "source": "naver_blog"},
    ]
    print(f"  [데모] {len(demo_posts)}건의 샘플 데이터를 사용합니다")
    return demo_posts


if __name__ == "__main__":
    main()
