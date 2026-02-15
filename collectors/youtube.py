"""YouTube Data API v3를 통한 데이터 수집"""

import os
from datetime import datetime, timedelta

import requests

from config import SEARCH_QUERIES, YOUTUBE_MAX_RESULTS, ANALYSIS_PERIOD_MONTHS


def collect_youtube() -> list[dict]:
    """YouTube Data API로 영상 제목·설명 수집.

    Returns:
        [{"title": str, "description": str, "published_at": str, "video_id": str}, ...]
    """
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        print("[WARN] YOUTUBE_API_KEY 미설정 - YouTube 수집 스킵")
        return []

    all_videos = []
    published_after = (
        datetime.now() - timedelta(days=ANALYSIS_PERIOD_MONTHS * 30)
    ).strftime("%Y-%m-%dT00:00:00Z")

    for query in SEARCH_QUERIES:
        print(f"  [YouTube] 검색 중: {query}")
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": YOUTUBE_MAX_RESULTS,
            "order": "date",
            "publishedAfter": published_after,
            "relevanceLanguage": "ko",
            "key": api_key,
        }
        try:
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    [ERR] API 요청 실패: {e}")
            continue

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            all_videos.append({
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", ""),
                "video_id": item.get("id", {}).get("videoId", ""),
                "source": "youtube",
                "query": query,
            })

    print(f"  [YouTube] 총 {len(all_videos)}건 수집 완료")
    return all_videos
