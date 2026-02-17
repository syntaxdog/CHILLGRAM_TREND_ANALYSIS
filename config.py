"""프로젝트 설정값 — domain_config.json에서 도메인 지식 로드"""

from datetime import datetime

from config_generator import load_domain_config

# === 도메인 설정 (domain_config.json) ===
_domain = load_domain_config()

TARGET_PRODUCT = _domain["target_product"]
TARGET_CONSUMER = _domain["target_consumer"]
BASE_QUERIES = _domain["base_queries"]
SEASONAL_QUERIES = _domain["seasonal_queries"]
USER_DICTIONARY = [tuple(item) for item in _domain["user_dictionary"]]
STOPWORDS = set(_domain["stopwords"])
NEGATIVE_EXPRESSIONS = _domain["negative_expressions"]
DESIGN_CATEGORIES = _domain["design_categories"]
SEASONAL_KEYWORDS = _domain["seasonal_keywords"]

# === 분석 기간 ===
ANALYSIS_PERIOD_MONTHS = 12

# === 수집 설정 (모드별) ===
COLLECT_LIMITS = {
    "backfill": {"naver": 1000, "youtube": 200, "instagram": 200},
    "daily":    {"naver": 100,  "youtube": 30,  "instagram": 50},
}

# === 백필 분할 (4일) ===
BACKFILL_TOTAL_DAYS = 4

# === 키워드 추출 제한 ===
TFIDF_TOP_N = 150
KEYBERT_TOP_N = 80
FINAL_KEYWORD_COUNT = 20

# === 트렌드 판정 임계값 ===
TREND_RISING_THRESHOLD = 1.5
TREND_FALLING_THRESHOLD = 0.5
SEASONAL_TOLERANCE = 0.3  # 계절 키워드는 1.5 + 0.3 = 1.8 이상이어야 상승

# === 2트랙 블렌딩 설정 ===
RECENT_WINDOW_DAYS = 7       # 최근 윈도우 기간 (일)
BLEND_RECENT_WEIGHT = 0.6    # 최근 트랙 가중치
BLEND_FULL_WEIGHT = 0.4      # 전체 트랙 가중치
RECENT_MIN_DOCS = 10         # 최근 문서 최소 수 (미달 시 전체 데이터만 사용)

# === 스코어링 가중치 ===
WEIGHT_TREND = 0.30
WEIGHT_RECENCY = 0.20
WEIGHT_DESIGN = 0.25
WEIGHT_DIVERSITY = 0.15
WEIGHT_NOVELTY = 0.10


def get_search_queries(mode: str = "daily", day: int | None = None) -> list[str]:
    """모드에 따라 검색 쿼리 반환.

    Args:
        mode: "backfill" 또는 "daily"
        day: backfill 분할 실행 시 일차 (1~BACKFILL_TOTAL_DAYS).
             None이면 전체 쿼리 반환.
    """
    if mode == "backfill":
        # 전체 쿼리 구성: base + 모든 시즌
        all_queries = list(BASE_QUERIES)
        for month_queries in SEASONAL_QUERIES.values():
            all_queries.extend(month_queries)

        if day is not None:
            # day별로 쿼리 균등 분할
            chunk_size = len(all_queries) // BACKFILL_TOTAL_DAYS
            remainder = len(all_queries) % BACKFILL_TOTAL_DAYS
            start = chunk_size * (day - 1) + min(day - 1, remainder)
            end = start + chunk_size + (1 if day <= remainder else 0)
            return all_queries[start:end]

        return all_queries
    else:
        # daily: base + 현재 월 ±1개월의 시즌 쿼리만
        queries = list(BASE_QUERIES)
        current_month = datetime.now().month
        for offset in [-1, 0, 1]:
            m = ((current_month - 1 + offset) % 12) + 1
            month_key = str(m)
            if month_key in SEASONAL_QUERIES:
                queries.extend(SEASONAL_QUERIES[month_key])
        return queries
