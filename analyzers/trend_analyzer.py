"""트렌드 추이 분석 + 최종 키워드 20개 선정"""

from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

from config import (
    TREND_RISING_THRESHOLD,
    TREND_FALLING_THRESHOLD,
    WEIGHT_TREND,
    WEIGHT_DESIGN,
    WEIGHT_DIVERSITY,
    WEIGHT_NOVELTY,
    FINAL_KEYWORD_COUNT,
    DESIGN_CATEGORIES,
)


def analyze_trends(
    processed_docs: list[dict],
    design_keywords: list[dict],
    google_trends_df: pd.DataFrame | None = None,
    naver_search_trend_df: pd.DataFrame | None = None,
    naver_shopping_trend_df: pd.DataFrame | None = None,
) -> list[dict]:
    """트렌드 분석 후 최종 키워드 20개 선정.

    Args:
        processed_docs: 전처리된 문서 리스트
        design_keywords: 디자인 필터 통과 키워드 [{"keyword", "score", "category"}, ...]
        google_trends_df: Google Trends 데이터 (선택)
        naver_search_trend_df: 네이버 검색어트렌드 데이터 (선택)
        naver_shopping_trend_df: 네이버 쇼핑인사이트 데이터 (선택)

    Returns:
        최종 키워드 리스트 [{"rank", "keyword", "category", "trend", "score", "reason"}, ...]
    """
    if not design_keywords:
        print("  [트렌드] 분석할 키워드가 없습니다")
        return []

    # 1. 월별 빈도 계산
    monthly_freq = _compute_monthly_frequency(processed_docs, design_keywords)

    # 2. 트렌드 방향 판별 (상승/하강/안정)
    trend_directions = _classify_trends(monthly_freq)

    # 3. Google Trends 교차 검증
    google_verified = _verify_with_google_trends(design_keywords, google_trends_df)

    # 3-1. 네이버 데이터랩 교차 검증
    naver_verified = _verify_with_naver_datalab(
        design_keywords, naver_search_trend_df, naver_shopping_trend_df
    )

    # 4. 가중 점수 계산
    scored_keywords = _compute_final_scores(
        design_keywords, trend_directions, google_verified, naver_verified
    )

    # 5. 카테고리 다양성 보장하며 상위 20개 선정
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
        if len(date_str) >= 7:
            month = date_str[:7]  # "2024-01"
        elif len(date_str) >= 6:
            month = f"{date_str[:4]}-{date_str[4:6]}"
        else:
            continue

        for token in doc["tokens"]:
            if token in keyword_set:
                monthly[token][month] += 1

    return dict(monthly)


def _classify_trends(monthly_freq: dict) -> dict[str, str]:
    """상승/하강/안정 분류"""
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
        elif recent_avg > overall_avg * TREND_RISING_THRESHOLD:
            directions[keyword] = "상승"
        elif recent_avg < overall_avg * TREND_FALLING_THRESHOLD:
            directions[keyword] = "하강"
        else:
            directions[keyword] = "안정"

    return directions


def _verify_with_google_trends(
    keywords: list[dict], gt_df: pd.DataFrame | None
) -> set[str]:
    """Google Trends 데이터로 검증된 키워드 set 반환"""
    if gt_df is None or gt_df.empty:
        # Google Trends 데이터 없으면 전부 검증된 것으로 처리
        return {kw["keyword"] for kw in keywords}

    verified = set()
    for kw in keywords:
        word = kw["keyword"]
        if word in gt_df.columns:
            series = gt_df[word]
            # 평균 검색량이 10 이상이면 유효
            if series.mean() >= 10:
                verified.add(word)
        else:
            # Google Trends에 조회하지 않은 키워드는 통과
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

        # 검색어트렌드에서 확인
        if search_df is not None and not search_df.empty and word in search_df.columns:
            if search_df[word].mean() >= 5:
                found = True

        # 쇼핑인사이트에서 확인
        if shopping_df is not None and not shopping_df.empty and word in shopping_df.columns:
            if shopping_df[word].mean() >= 5:
                found = True

        if found:
            verified.add(word)
        elif search_df is None and shopping_df is None:
            # 데이터랩 데이터가 아예 없으면 전부 통과
            verified.add(word)

    return verified


def _compute_final_scores(
    keywords: list[dict],
    trends: dict[str, str],
    google_verified: set[str],
    naver_verified: set[str] | None = None,
) -> list[dict]:
    """가중 점수 계산"""
    # 이미 보편화된 키워드 (참신성 감점용)
    well_known = set()
    for examples in DESIGN_CATEGORIES.values():
        well_known.update(examples[:3])  # 각 카테고리의 앞 3개는 이미 알려진 것

    scored = []
    for kw in keywords:
        word = kw["keyword"]
        trend = trends.get(word, "안정")

        # 트렌드 점수 (40%)
        trend_score = {"상승": 1.0, "안정": 0.5, "하강": 0.1}.get(trend, 0.5)
        if word not in google_verified:
            trend_score *= 0.7  # Google Trends 미검증 감점

        # 네이버 데이터랩 검증 보너스
        if naver_verified and word in naver_verified:
            trend_score = min(trend_score * 1.2, 1.0)  # 최대 20% 보너스

        # 디자인 반영 가능성 (30%) - TF-IDF/KeyBERT 점수 활용
        design_score = min(kw["score"] / 0.1, 1.0) if kw["score"] > 0 else 0.5

        # 다양성은 select_diverse_top에서 처리 (20%)
        diversity_score = 1.0

        # 참신성 (10%) - 이미 보편화된 키워드는 감점
        novelty_score = 0.3 if word in well_known else 1.0

        final_score = (
            WEIGHT_TREND * trend_score
            + WEIGHT_DESIGN * design_score
            + WEIGHT_DIVERSITY * diversity_score
            + WEIGHT_NOVELTY * novelty_score
        )

        # 이유 생성
        trend_symbol = {"상승": "↑", "안정": "->", "하강": "↓"}.get(trend, "->")
        reason = f"트렌드 {trend_symbol} {trend}"
        if word in google_verified and trend == "상승":
            reason += " + Google 검증"
        if naver_verified and word in naver_verified:
            reason += " + 네이버 검증"

        scored.append({
            "keyword": word,
            "category": kw["category"],
            "trend": trend,
            "trend_symbol": trend_symbol,
            "final_score": final_score,
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

    # 2차: 점수 순으로 나머지 채우기 (카테고리 편중 방지)
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

    # 순위 부여
    for i, kw in enumerate(selected[:n], 1):
        kw["rank"] = i

    return selected[:n]
