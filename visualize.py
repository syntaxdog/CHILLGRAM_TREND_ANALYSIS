"""트렌드 키워드 분석 결과 시각화 — PPT용 차트 생성

사용법:
    python visualize.py              # BigQuery 최신 결과로 차트 생성
    python visualize.py --csv FILE   # 로컬 CSV 파일로 차트 생성

출력: output/charts/ 폴더에 PNG 파일 4장
"""

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # SSH/헤드리스 환경용 (GUI 없이 PNG 저장)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 한글 폰트 설정 ──
def _setup_korean_font():
    """시스템에서 한글 폰트를 찾아 matplotlib에 설정."""
    candidates = [
        "Malgun Gothic",   # Windows
        "NanumGothic",     # Linux (nanum)
        "AppleGothic",     # macOS
        "NanumBarunGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    print("[WARN] 한글 폰트를 찾을 수 없습니다. 글자가 깨질 수 있습니다.")
    return None

_setup_korean_font()

# ── 색상 팔레트 ──
CATEGORY_COLORS = {
    "트렌드맛·원료": "#FF6B6B",
    "시즌·이벤트": "#FFA94D",
    "비주얼·감성": "#845EF7",
    "건강·가치소비": "#51CF66",
    "소재·구조": "#339AF0",
}
TREND_COLORS = {"상승": "#FF6B6B", "안정": "#868E96", "하강": "#339AF0", "시즌": "#FFA94D"}
DEFAULT_COLOR = "#868E96"


def load_from_bigquery() -> list[dict]:
    """BigQuery에서 최신 분석 결과 로드."""
    from storage.gcp_client import get_bq_client, PROJECT_ID, BQ_DATASET, BQ_TABLE_RESULTS

    client = get_bq_client()
    query = f"""
        SELECT rank, keyword, extended_keyword, category, trend, score, reason
        FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RESULTS}`
        WHERE analysis_date = (
            SELECT MAX(analysis_date) FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RESULTS}`
        )
        ORDER BY rank ASC
    """
    result = client.query(query).result()
    keywords = []
    for row in result:
        keywords.append({
            "rank": row.rank,
            "keyword": row.keyword,
            "extended_keyword": row.extended_keyword or row.keyword,
            "category": row.category,
            "trend": row.trend,
            "score": float(row.score) if row.score else 0.0,
            "reason": row.reason or "",
        })
    return keywords


def load_from_csv(csv_path: str) -> list[dict]:
    """로컬 CSV 파일에서 결과 로드."""
    keywords = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            keywords.append({
                "rank": int(row.get("rank", i)),
                "keyword": row.get("keyword", ""),
                "extended_keyword": row.get("extended_keyword", row.get("keyword", "")),
                "category": row.get("category", ""),
                "trend": row.get("trend", "안정"),
                "score": float(row.get("score", 0)),
                "reason": row.get("reason", ""),
            })
    return keywords


def chart_top_keywords(keywords: list[dict], output_dir: Path):
    """1. Top 20 키워드 점수 수평 바차트."""
    fig, ax = plt.subplots(figsize=(12, 8))

    kw_list = list(reversed(keywords[:20]))
    labels = [f"{kw['keyword']}" for kw in kw_list]
    scores = [kw.get("score", kw.get("final_score", 0)) for kw in kw_list]
    colors = [CATEGORY_COLORS.get(kw["category"], DEFAULT_COLOR) for kw in kw_list]

    bars = ax.barh(labels, scores, color=colors, height=0.7, edgecolor="white", linewidth=0.5)

    # 점수 라벨
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9, color="#495057")

    ax.set_xlabel("최종 점수", fontsize=11)
    ax.set_title("트렌드 키워드 TOP 20", fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 카테고리 범례
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor="white")
        for cat, color in CATEGORY_COLORS.items()
    ]
    ax.legend(legend_handles, CATEGORY_COLORS.keys(), loc="lower right", fontsize=9)

    plt.tight_layout()
    path = output_dir / "01_top20_keywords.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [차트] {path}")


def chart_category_distribution(keywords: list[dict], output_dir: Path):
    """2. 카테고리 분포 도넛 차트."""
    cat_counts = Counter(kw["category"] for kw in keywords)

    labels = list(cat_counts.keys())
    sizes = list(cat_counts.values())
    colors = [CATEGORY_COLORS.get(cat, DEFAULT_COLOR) for cat in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.8,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    for text in autotexts:
        text.set_fontsize(12)
        text.set_fontweight("bold")
    for text in texts:
        text.set_fontsize(11)

    ax.set_title("카테고리 분포", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    path = output_dir / "02_category_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [차트] {path}")


def chart_trend_direction(keywords: list[dict], output_dir: Path):
    """3. 트렌드 방향 분포 차트."""
    trend_counts = Counter(kw["trend"] for kw in keywords)

    labels = list(trend_counts.keys())
    sizes = list(trend_counts.values())
    colors = [TREND_COLORS.get(t, DEFAULT_COLOR) for t in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, sizes, color=colors, width=0.5, edgecolor="white", linewidth=1)

    for bar, count in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{count}개", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("키워드 수", fontsize=11)
    ax.set_title("트렌드 방향 분포", fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(sizes) * 1.3 if sizes else 5)

    plt.tight_layout()
    path = output_dir / "03_trend_direction.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [차트] {path}")


def chart_monthly_frequency(keywords: list[dict], processed_docs: list[dict], output_dir: Path):
    """5. 키워드별 월별 빈도 변화 추이 라인 차트 (TOP 10)."""
    from analyzers.trend_analyzer import compute_monthly_frequency

    monthly_freq = compute_monthly_frequency(processed_docs, keywords)

    # TOP 10 키워드만 표시
    top_keywords = keywords[:10]
    top_kw_names = [kw["keyword"] for kw in top_keywords]

    # 전체 월 목록 (정렬)
    all_months = sorted({m for freq in monthly_freq.values() for m in freq})
    if not all_months:
        print("  [WARN] 월별 빈도 데이터가 없어 차트를 생성할 수 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    for kw_info in top_keywords:
        kw = kw_info["keyword"]
        if kw not in monthly_freq:
            continue
        freq = monthly_freq[kw]
        values = [freq.get(m, 0) for m in all_months]
        color = CATEGORY_COLORS.get(kw_info.get("category", ""), DEFAULT_COLOR)
        ax.plot(all_months, values, marker="o", markersize=4, linewidth=2,
                label=kw, color=color, alpha=0.85)

    ax.set_xlabel("월", fontsize=11)
    ax.set_ylabel("언급 빈도", fontsize=11)
    ax.set_title("키워드별 월별 빈도 변화 추이 (TOP 10)", fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # x축 라벨 회전
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "05_monthly_frequency.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [차트] {path}")


def chart_wordcloud(keywords: list[dict], output_dir: Path):
    """4. 키워드 워드클라우드."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  [WARN] wordcloud 패키지 미설치 — pip install wordcloud")
        return

    # 한글 폰트 경로 찾기
    font_path = None
    if sys.platform == "win32":
        candidate = "C:/Windows/Fonts/malgun.ttf"
        if os.path.exists(candidate):
            font_path = candidate
    if not font_path:
        for f in fm.fontManager.ttflist:
            if "malgun" in f.fname.lower() or "nanum" in f.fname.lower():
                font_path = f.fname
                break

    word_freq = {kw["keyword"]: max(kw.get("score", kw.get("final_score", 0)) * 100, 1) for kw in keywords}

    wc_kwargs = dict(
        width=1200, height=600,
        background_color="white",
        max_words=30,
        colormap="Set2",
        prefer_horizontal=0.7,
        min_font_size=14,
    )
    if font_path:
        wc_kwargs["font_path"] = font_path

    wc = WordCloud(**wc_kwargs).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("트렌드 키워드 워드클라우드", fontsize=16, fontweight="bold", pad=15)

    plt.tight_layout()
    path = output_dir / "04_wordcloud.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [차트] {path}")


def generate_pdf_report(keywords: list[dict], charts_dir: Path, output_dir: Path) -> Path | None:
    """차트 PNG들과 키워드 요약을 합쳐 PDF 보고서를 생성.

    Returns:
        생성된 PDF 파일 경로 (실패 시 None)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = output_dir / "trend_report.pdf"

    try:
        with PdfPages(str(pdf_path)) as pdf:
            # ── 1페이지: 타이틀 + 키워드 테이블 ──
            fig, ax = plt.subplots(figsize=(12, 16))
            ax.axis("off")

            today_str = date.today().isoformat()
            title_text = f"식품 패키지 디자인 트렌드 키워드 분석 보고서\n{today_str}"
            ax.text(0.5, 0.97, title_text, transform=ax.transAxes,
                    fontsize=20, fontweight="bold", ha="center", va="top")

            # 키워드 테이블
            if keywords:
                col_labels = ["순위", "키워드", "확장 키워드", "카테고리", "트렌드", "점수"]
                table_data = []
                for kw in keywords[:20]:
                    table_data.append([
                        str(kw.get("rank", "")),
                        kw.get("keyword", ""),
                        kw.get("extended_keyword", kw.get("keyword", "")),
                        kw.get("category", ""),
                        kw.get("trend", ""),
                        f"{kw.get('score', kw.get('final_score', 0)):.3f}",
                    ])

                table = ax.table(
                    cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.0, 1.4)

                # 헤더 스타일
                for j in range(len(col_labels)):
                    table[0, j].set_facecolor("#4472C4")
                    table[0, j].set_text_props(color="white", fontweight="bold")

                # 행 색상 교대
                for i in range(1, len(table_data) + 1):
                    for j in range(len(col_labels)):
                        if i % 2 == 0:
                            table[i, j].set_facecolor("#D9E2F3")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── 나머지 페이지: 차트 PNG 삽입 ──
            chart_order = [
                "01_top20_keywords.png",
                "02_category_distribution.png",
                "03_trend_direction.png",
                "05_monthly_frequency.png",
                "04_wordcloud.png",
            ]
            for chart_name in chart_order:
                chart_path = charts_dir / chart_name
                if not chart_path.exists():
                    continue
                img = plt.imread(str(chart_path))
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"  [PDF] 보고서 생성 완료: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"  [WARN] PDF 생성 실패: {e}")
        return None


def upload_pdf_to_gcs(pdf_path: Path):
    """PDF 보고서를 GCS에 업로드 (latest + history)."""
    if not pdf_path or not pdf_path.exists():
        return

    try:
        from storage.gcp_client import get_gcs_client, GCS_BUCKET
    except Exception as e:
        print(f"  [WARN] GCS 클라이언트 로드 실패: {e}")
        return

    today = date.today().isoformat()

    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)

        # latest
        latest_blob = bucket.blob("results/latest/trend_report.pdf")
        latest_blob.upload_from_filename(str(pdf_path), content_type="application/pdf")

        # history
        history_blob = bucket.blob(f"results/history/{today}/trend_report.pdf")
        history_blob.upload_from_filename(str(pdf_path), content_type="application/pdf")

        print(f"  [GCS] PDF 업로드 완료")
        print(f"    latest:  gs://{GCS_BUCKET}/results/latest/trend_report.pdf")
        print(f"    history: gs://{GCS_BUCKET}/results/history/{today}/trend_report.pdf")
    except Exception as e:
        print(f"  [WARN] GCS PDF 업로드 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="트렌드 키워드 시각화")
    parser.add_argument("--csv", type=str, help="로컬 CSV 파일 경로 (BigQuery 대신 사용)")
    args = parser.parse_args()

    # 데이터 로드
    if args.csv:
        print(f"[데이터] CSV 로드: {args.csv}")
        keywords = load_from_csv(args.csv)
    else:
        print("[데이터] BigQuery에서 최신 결과 로드")
        try:
            keywords = load_from_bigquery()
        except Exception as e:
            print(f"  [ERR] BigQuery 로드 실패: {e}")
            print("  --csv 옵션으로 로컬 CSV를 지정하세요.")
            return

    if not keywords:
        print("[!] 시각화할 데이터가 없습니다.")
        return

    print(f"[데이터] {len(keywords)}개 키워드 로드 완료\n")

    # 출력 디렉토리
    output_dir = Path(__file__).parent / "output" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 차트 생성
    chart_top_keywords(keywords, output_dir)
    chart_category_distribution(keywords, output_dir)
    chart_trend_direction(keywords, output_dir)
    chart_wordcloud(keywords, output_dir)

    chart_files = list(output_dir.glob("*.png"))
    print(f"\n로컬 저장 완료! 차트 {len(chart_files)}장: {output_dir}")

    # GCS 업로드
    _upload_charts_to_gcs(chart_files)


def _upload_charts_to_gcs(chart_files: list[Path]):
    """생성된 차트 PNG를 GCS에 업로드."""
    if not chart_files:
        return

    try:
        from storage.gcp_client import get_gcs_client, GCS_BUCKET
    except Exception as e:
        print(f"  [WARN] GCS 클라이언트 로드 실패: {e}")
        return

    today = date.today().isoformat()

    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)

        print("\n[GCS] 차트 업로드")
        for f in sorted(chart_files):
            # latest (항상 최신)
            latest_path = f"results/latest/charts/{f.name}"
            blob = bucket.blob(latest_path)
            blob.upload_from_filename(str(f), content_type="image/png")

            # history (날짜별)
            history_path = f"results/history/{today}/charts/{f.name}"
            blob_h = bucket.blob(history_path)
            blob_h.upload_from_filename(str(f), content_type="image/png")

            print(f"  [GCS] {f.name} 업로드 완료")

        print(f"  [GCS] latest: gs://{GCS_BUCKET}/results/latest/charts/")
        print(f"  [GCS] history: gs://{GCS_BUCKET}/results/history/{today}/charts/")
    except Exception as e:
        print(f"  [WARN] GCS 차트 업로드 실패: {e}")


if __name__ == "__main__":
    main()
