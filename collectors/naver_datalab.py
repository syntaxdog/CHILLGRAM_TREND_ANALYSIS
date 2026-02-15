"""네이버 데이터랩 API - 검색어트렌드 + 쇼핑인사이트"""

import os
import json
from datetime import datetime, timedelta

import requests
import pandas as pd

from config import ANALYSIS_PERIOD_MONTHS


def _get_headers():
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    return {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }


def _date_range():
    end = datetime.now()
    start = end - timedelta(days=ANALYSIS_PERIOD_MONTHS * 30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def get_search_trend(keyword_groups: list[list[str]]) -> pd.DataFrame | None:
    """네이버 검색어트렌드 API 조회.

    Args:
        keyword_groups: 키워드 그룹 리스트 (최대 5개 그룹, 그룹당 키워드 묶음)
            예: [["두바이초콜릿", "두바이 초콜릿"], ["말차"], ["제로슈거", "제로 슈거"]]

    Returns:
        DataFrame(columns=[키워드그룹명...], index=날짜) 또는 None
    """
    headers = _get_headers()
    if not headers:
        print("  [데이터랩] API 키 미설정 - 검색어트렌드 스킵")
        return None

    start_date, end_date = _date_range()

    # 최대 5개 그룹씩 처리
    all_results = {}

    for batch_start in range(0, len(keyword_groups), 5):
        batch = keyword_groups[batch_start:batch_start + 5]

        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": "month",
            "keywordGroups": [
                {
                    "groupName": group[0],  # 첫 번째 키워드를 그룹명으로
                    "keywords": group,
                }
                for group in batch
            ],
        }

        try:
            resp = requests.post(
                "https://openapi.naver.com/v1/datalab/search",
                headers=headers,
                data=json.dumps(body),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [데이터랩] 검색어트렌드 API 오류: {e}")
            continue

        for result in data.get("results", []):
            name = result["title"]
            series = {}
            for item in result.get("data", []):
                series[item["period"]] = item["ratio"]
            all_results[name] = series

    if not all_results:
        return None

    df = pd.DataFrame(all_results)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"  [데이터랩] 검색어트렌드 {len(df.columns)}개 키워드 조회 완료")
    return df


def get_shopping_insight(category_id: str = "50000000", keyword: str = None) -> pd.DataFrame | None:
    """네이버 쇼핑인사이트 API - 분야별 트렌드 조회.

    Args:
        category_id: 쇼핑 카테고리 ID (기본: 50000000 = 식품)
        keyword: 특정 키워드 (None이면 카테고리 전체)

    Returns:
        DataFrame 또는 None
    """
    headers = _get_headers()
    if not headers:
        print("  [데이터랩] API 키 미설정 - 쇼핑인사이트 스킵")
        return None

    start_date, end_date = _date_range()

    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": "month",
        "category": category_id,
        "device": "",    # 전체
        "gender": "",    # 전체
        "ages": [],      # 전체
    }
    if keyword:
        body["keyword"] = keyword

    try:
        resp = requests.post(
            "https://openapi.naver.com/v1/datalab/shopping/category/keywords",
            headers=headers,
            data=json.dumps(body),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [데이터랩] 쇼핑인사이트 API 오류: {e}")
        return None

    results = data.get("results", [])
    if not results:
        return None

    all_series = {}
    for result in results:
        name = result.get("title", "unknown")
        series = {}
        for item in result.get("data", []):
            series[item["period"]] = item["ratio"]
        all_series[name] = series

    df = pd.DataFrame(all_series)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"  [데이터랩] 쇼핑인사이트 조회 완료")
    return df


def get_shopping_keywords_trend(keywords: list[str], category_id: str = "50000000") -> pd.DataFrame | None:
    """쇼핑인사이트 키워드별 트렌드 조회 (최대 5개씩 배치).

    Args:
        keywords: 조회할 키워드 리스트
        category_id: 쇼핑 카테고리 ID (기본: 50000000 = 식품)

    Returns:
        DataFrame(columns=[키워드...], index=날짜) 또는 None
    """
    headers = _get_headers()
    if not headers:
        print("  [데이터랩] API 키 미설정 - 쇼핑인사이트 스킵")
        return None

    start_date, end_date = _date_range()
    all_results = {}

    for batch_start in range(0, len(keywords), 5):
        batch = keywords[batch_start:batch_start + 5]

        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": "month",
            "category": category_id,
            "keyword": batch,
            "device": "",
            "gender": "",
            "ages": [],
        }

        try:
            resp = requests.post(
                "https://openapi.naver.com/v1/datalab/shopping/category/keyword/age",
                headers=headers,
                data=json.dumps(body),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [데이터랩] 쇼핑인사이트 키워드 API 오류: {e}")
            continue

        for result in data.get("results", []):
            name = result.get("title", "unknown")
            series = {}
            for item in result.get("data", []):
                series[item["period"]] = item["ratio"]
            all_results[name] = series

    if not all_results:
        return None

    df = pd.DataFrame(all_results)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"  [데이터랩] 쇼핑인사이트 {len(df.columns)}개 키워드 조회 완료")
    return df
