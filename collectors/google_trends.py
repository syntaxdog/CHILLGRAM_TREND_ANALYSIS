"""Google Trends 데이터 수집 (pytrends)"""

import time

import pandas as pd

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False


def get_google_trends(keywords: list[str], timeframe: str = "today 12-m") -> pd.DataFrame:
    """키워드별 Google Trends 검색량 추이 수집.

    Args:
        keywords: 검색할 키워드 리스트
        timeframe: 기간 (예: "today 6-m", "today 12-m")

    Returns:
        DataFrame - 컬럼이 키워드, 인덱스가 날짜, 값이 상대 검색량
    """
    if not HAS_PYTRENDS:
        print("[WARN] pytrends 미설치 - Google Trends 수집 스킵")
        return pd.DataFrame()

    if not keywords:
        return pd.DataFrame()

    pytrends = TrendReq(hl="ko", tz=540)  # KST

    all_data = pd.DataFrame()

    # Google Trends는 한 번에 최대 5개 키워드 비교 가능
    for i in range(0, len(keywords), 5):
        batch = keywords[i:i + 5]
        print(f"  [Google Trends] 조회 중: {batch}")
        try:
            pytrends.build_payload(batch, timeframe=timeframe, geo="KR")
            df = pytrends.interest_over_time()
            if not df.empty and "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            all_data = pd.concat([all_data, df], axis=1)
        except Exception as e:
            print(f"    [ERR] Google Trends 조회 실패: {e}")

        time.sleep(1)  # 요청 간격

    print(f"  [Google Trends] {len(all_data.columns)}개 키워드 추이 수집 완료")
    return all_data
