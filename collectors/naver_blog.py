"""네이버 블로그 검색 API를 통한 데이터 수집"""

import os
import re
import time
from datetime import datetime, timedelta

import requests

from config import SEARCH_QUERIES, NAVER_DISPLAY_COUNT, NAVER_SORT, ANALYSIS_PERIOD_MONTHS


def collect_naver_blogs() -> list[dict]:
    """네이버 블로그 검색 API로 게시글 수집.

    Returns:
        [{"title": str, "description": str, "postdate": str, "link": str}, ...]
    """
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("[WARN] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 미설정 - 네이버 블로그 수집 스킵")
        return []

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    all_posts = []
    cutoff_date = datetime.now() - timedelta(days=ANALYSIS_PERIOD_MONTHS * 30)

    for query in SEARCH_QUERIES:
        print(f"  [네이버] 검색 중: {query}")
        start = 1
        while start <= NAVER_DISPLAY_COUNT:
            display = min(100, NAVER_DISPLAY_COUNT - start + 1)
            params = {
                "query": query,
                "display": display,
                "start": start,
                "sort": NAVER_SORT,
            }
            try:
                resp = requests.get(
                    "https://openapi.naver.com/v1/search/blog.json",
                    headers=headers,
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"    [ERR] API 요청 실패: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                postdate = item.get("postdate", "")
                # postdate 형식: "20240101"
                if len(postdate) == 8:
                    try:
                        dt = datetime.strptime(postdate, "%Y%m%d")
                        if dt < cutoff_date:
                            continue
                    except ValueError:
                        pass

                # HTML 태그 제거
                title = _strip_html(item.get("title", ""))
                desc = _strip_html(item.get("description", ""))

                all_posts.append({
                    "title": title,
                    "description": desc,
                    "postdate": postdate,
                    "link": item.get("link", ""),
                    "source": "naver_blog",
                    "query": query,
                })

            start += display
            time.sleep(0.1)  # API 호출 간격

    print(f"  [네이버] 총 {len(all_posts)}건 수집 완료")
    return all_posts


def _strip_html(text: str) -> str:
    """HTML 태그 제거"""
    return re.sub(r"<[^>]+>", "", text)
