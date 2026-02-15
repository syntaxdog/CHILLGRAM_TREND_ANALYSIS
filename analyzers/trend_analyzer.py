"""트렌드 추이 분석 + 최종 키워드 20개 선정 (계절성 보정 포함)"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd

from config import (
    TREND_RISING_THRESHOLD,
    TREND_FALLING_THRESHOLD,
    SEASONAL_TOLERANCE,
    WEIGHT_TREND,
    WEIGHT_RECENCY,
    WEIGHT_DESIGN,
    WEIGHT_DIVERSITY,
    WEIGHT_NOVELTY,
    FINAL_KEYWORD_COUNT,
    DESIGN_CATEGORIES,
    SEASONAL_KEYWORDS,
)


def analyze_trends(
    processed_docs: list[dict],
    design_keywords: list[dict],
    google_trends_df: pd.DataFrame | None = None,
    naver_search_trend_df: pd.DataFrame | None = None,
    naver_shopping_trend_df: pd.DataFrame | None = None,
) -> list[dict]:
    """트렌드 분석 후 최종 키워드 20개 선정."""
    if not design_keywords:
        print("  [트렌드] 분석할 키워드가 없습니다")
        return []

    # 1. 월별 빈도 계산
    monthly_freq = _compute_monthly_frequency(processed_docs, design_keywords)

    # 2. 트렌드 방향 판별 (상승/하강/안정/시즌)
    trend_directions = _classify_trends(monthly_freq)

    # 3. recency 점수 계산
    recency_scores = _compute_recency_scores(processed_docs, design_keywords)

    # 4. Google Trends 교차 검증
    google_verified = _verify_with_google_trends(design_keywords, google_trends_df)

    # 4-1. 네이버 데이터랩 교차 검증
    naver_verified = _verify_with_naver_datalab(
        design_keywords, naver_search_trend_df, naver_shopping_trend_df
    )

    # 5. 가중 점수 계산
    scored_keywords = _compute_final_scores(
        design_keywords, trend_directions, recency_scores,
        google_verified, naver_verified,
    )

    # 6. 카테고리 다양성 보장하며 상위 20개 선정
    final_keywords = _select_diverse_top(scored_keywords, FINAL_KEYWORD_COUNT)

    print(f"  [트렌드] 최종 {len(final_keywords)}개 키워드 선정 완료")
    return final_keywords


def _compute_monthly_frequency(
    docs: list[dict], keywords: list[dict]
) -> dict[str, dict[str, int]]:
    """키워드별 월별 출현 빈도 계산"""
    keyword_set = {kw["keyword"] for kw in keywords}
    monthly = defaultdict(lambda: Counter())

    for doc in docs:
        date_str = doc.get("date", "")
        if len(date_str) >= 10 and date_str[4] == "-":
            month = date_str[:7]  # "2024-01-15" → "2024-01"
        elif len(date_str) >= 8 and date_str[4] != "-":
            month = f"{date_str[:4]}-{date_str[4:6]}"  # "20240115" → "2024-01"
        elif len(date_str) >= 7 and date_str[4] == "-":
            month = date_str[:7]  # "2024-01"
        else:
            continue

        for token in doc["tokens"]:
            if token in keyword_set:
                monthly[token][month] += 1

    return dict(monthly)


def _check_if_seasonal(keyword: str, recent_months: list[str]) -> bool:
    """키워드가 최근 월의 계절 사전에 포함되는지 확인.

    Args:
        keyword: 확인할 키워드
        recent_months: 최근 월 리스트 (예: ["2025-12", "2026-01", "2026-02"])
    """
    for month_str in recent_months:
        month_num = str(int(month_str.split("-")[1]))  # "2026-01" → "1"
        seasonal_list = SEASONAL_KEYWORDS.get(month_num, [])
        if keyword in seasonal_list:
            return True
    return False


def _classify_trends(monthly_freq: dict) -> dict[str, str]:
    """상승/하강/안정/시즌 분류 (계절성 보정 포함)"""
    directions = {}

    for keyword, month_counts in monthly_freq.items():
        if not month_counts:
            directions[keyword] = "안정"
            continue

        sorted_months = sorted(month_counts.keys())
        values = [month_counts[m] for m in sorted_months]
        overall_avg = sum(values) / len(values) if values else 0

        # 최근 3개월 평균
        recent = values[-3:] if len(values) >= 3 else values
        recent_avg = sum(recent) / len(recent) if recent else 0

        if overall_avg == 0:
            directions[keyword] = "안정"
            continue

        # 계절성 보정: 계절 키워드는 더 높은 임계값 적용
        is_seasonal = _check_if_seasonal(keyword, sorted_months[-3:])

        if is_seasonal:
            # 계절 키워드: 1.5 + 0.3 = 1.8배 이상이어야 진짜 상승
            seasonal_threshold = TREND_RISING_THRESHOLD + SEASONAL_TOLERANCE
            if recent_avg > overall_avg * seasonal_threshold:
                directions[keyword] = "상승"  # 시즌 이상의 폭발적 관심
            elif recent_avg < overall_avg * TREND_FALLING_THRESHOLD:
                directions[keyword] = "하강"
            else:
                directions[keyword] = "시즌"  # 정상적인 계절 스파이크
        else:
            # 비계절 키워드: 기존 임계값
            if recent_avg > overall_avg * TREND_RISING_THRESHOLD:
                directions[keyword] = "상승"
            elif recent_avg < overall_avg * TREND_FALLING_THRESHOLD:
                directions[keyword] = "하강"
            else:
                directions[keyword] = "안정"

    return directions


def _compute_recency_scores(
    docs: list[dict], keywords: list[dict]
) -> dict[str, float]:
    """최근 문서 기반 recency 점수 계산. 0.0 ~ 1.0."""
    keyword_set = {kw["keyword"] for kw in keywords}
    now = datetime.now()
    cutoff_30d = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_60d = (now - timedelta(days=60)).strftime("%Y-%m-%d")

    recent_30d = Counter()
    recent_60d = Counter()
    total_count = Counter()

    for doc in docs:
        date_str = doc.get("date", "")
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        doc_keywords = {t for t in doc.get("tokens", []) if t in keyword_set}
        for kw in doc_keywords:
            total_count[kw] += 1
            if date_str >= cutoff_30d:
                recent_30d[kw] += 1
            elif date_str >= cutoff_60d:
                recent_60d[kw] += 1

    scores = {}
    for kw_info in keywords:
        kw = kw_info["keyword"]
        total = total_count.get(kw, 0)
        if total == 0:
            scores[kw] = 0.5
            continue

        r30 = recent_30d.get(kw, 0)
        r60 = recent_60d.get(kw, 0)
        recency_ratio = (r30 * 2.0 + r60 * 1.0) / (total * 2.0)
        scores[kw] = min(recency_ratio * 2.0, 1.0)

    return scores


def _verify_with_google_trends(
    keywords: list[dict], gt_df: pd.DataFrame | None
) -> set[str]:
    """Google Trends 데이터로 검증된 키워드 set 반환"""
    if gt_df is None or gt_df.empty:
        return {kw["keyword"] for kw in keywords}

    verified = set()
    for kw in keywords:
        word = kw["keyword"]
        if word in gt_df.columns:
            series = gt_df[word]
            if series.mean() >= 10:
                verified.add(word)
        else:
            verified.add(word)

    return verified


def _verify_with_naver_datalab(
    keywords: list[dict],
    search_df: pd.DataFrame | None,
    shopping_df: pd.DataFrame | None,
) -> set[str]:
    """네이버 데이터랩 데이터로 검증된 키워드 set 반환"""
    verified = set()

    for kw in keywords:
        word = kw["keyword"]
        found = False

        if search_df is not None and not search_df.empty and word in search_df.columns:
            if search_df[word].mean() >= 5:
                found = True

        if shopping_df is not None and not shopping_df.empty and word in shopping_df.columns:
            if shopping_df[word].mean() >= 5:
                found = True

        if found:
            verified.add(word)
        elif search_df is None and shopping_df is None:
            verified.add(word)

    return verified


def _compute_final_scores(
    keywords: list[dict],
    trends: dict[str, str],
    recency_scores: dict[str, float],
    google_verified: set[str],
    naver_verified: set[str] | None = None,
) -> list[dict]:
    """가중 점수 계산 (계절성 보정 반영)"""
    well_known = set()
    for examples in DESIGN_CATEGORIES.values():
        well_known.update(examples[:3])

    scored = []
    for kw in keywords:
        word = kw["keyword"]
        trend = trends.get(word, "안정")
        recency = recency_scores.get(word, 0.5)

        # 트렌드 점수 — "시즌"은 안정과 상승 사이
        trend_score = {
            "상승": 1.0, "시즌": 0.6, "안정": 0.5, "하강": 0.1
        }.get(trend, 0.5)

        if word not in google_verified:
            trend_score *= 0.7
        if naver_verified and word in naver_verified:
            trend_score = min(trend_score * 1.2, 1.0)

        recency_score = recency
        design_score = min(kw["score"] / 0.1, 1.0) if kw["score"] > 0 else 0.5
        diversity_score = 1.0
        novelty_score = 0.3 if word in well_known else 1.0

        final_score = (
            WEIGHT_TREND * trend_score
            + WEIGHT_RECENCY * recency_score
            + WEIGHT_DESIGN * design_score
            + WEIGHT_DIVERSITY * diversity_score
            + WEIGHT_NOVELTY * novelty_score
        )

        # 이유 생성
        trend_symbol = {
            "상승": "^", "시즌": "~", "안정": "->", "하강": "v"
        }.get(trend, "->")
        reason = f"트렌드 {trend_symbol} {trend}"
        if recency >= 0.7:
            reason += " + 최근급증"
        if word in google_verified and trend in ("상승", "시즌"):
            reason += " + Google검증"
        if naver_verified and word in naver_verified:
            reason += " + 네이버검증"

        scored.append({
            "keyword": word,
            "category": kw["category"],
            "trend": trend,
            "trend_symbol": trend_symbol,
            "final_score": final_score,
            "recency": recency,
            "mention_count": kw.get("mention_count", 0),
            "reason": reason,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


def _select_diverse_top(scored: list[dict], n: int) -> list[dict]:
    """카테고리 다양성을 보장하며 상위 n개 선정"""
    selected = []
    category_counts = Counter()
    max_per_category = max(n // len(DESIGN_CATEGORIES) + 2, 3)

    # 1차: 각 카테고리별 최소 1개씩
    for cat in DESIGN_CATEGORIES:
        for kw in scored:
            if kw["category"] == cat and kw not in selected:
                selected.append(kw)
                category_counts[cat] += 1
                break

    # 2차: 점수 순으로 나머지 채우기
    for kw in scored:
        if len(selected) >= n:
            break
        if kw in selected:
            continue
        if category_counts[kw["category"]] >= max_per_category:
            continue
        selected.append(kw)
        category_counts[kw["category"]] += 1

    # 부족하면 점수 순으로 추가
    for kw in scored:
        if len(selected) >= n:
            break
        if kw not in selected:
            selected.append(kw)

    for i, kw in enumerate(selected[:n], 1):
        kw["rank"] = i

    return selected[:n]
