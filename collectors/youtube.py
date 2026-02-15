"""YouTube Data API v3를 통한 데이터 수집"""

import os
from datetime import datetime, timedelta

import requests

from config import COLLECT_LIMITS, ANALYSIS_PERIOD_MONTHS


def collect_youtube(search_queries: list[str], mode: str = "daily") -> list[dict]:
    """YouTube Data API로 영상 제목·설명 수집.

    Args:
        search_queries: 검색할 쿼리 리스트 (config.get_search_queries()로 생성)

    Returns:
        [{"title": str, "description": str, "published_at": str, "video_id": str, ...}, ...]
    """
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        print("[WARN] YOUTUBE_API_KEY 미설정 - YouTube 수집 스킵")
        return []

    max_results = COLLECT_LIMITS.get(mode, COLLECT_LIMITS["daily"])["youtube"]

    # 1단계: search.list로 video_id 수집 (페이징)
    video_ids = []
    video_meta = {}  # video_id -> {query}
    published_after = (
        datetime.now() - timedelta(days=ANALYSIS_PERIOD_MONTHS * 30)
    ).strftime("%Y-%m-%dT00:00:00Z")

    total_search_quota = 0

    for query in search_queries:
        print(f"  [YouTube] 검색 중: {query}")
        collected_for_query = 0
        next_page_token = None

        while collected_for_query < max_results:
            max_per_page = min(50, max_results - collected_for_query)
            params = {
                "part": "id",
                "q": query,
                "type": "video",
                "maxResults": max_per_page,
                "order": "date",
                "publishedAfter": published_after,
                "relevanceLanguage": "ko",
                "key": api_key,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                resp = requests.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                total_search_quota += 100
            except Exception as e:
                print(f"    [ERR] search API 요청 실패: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                vid = item.get("id", {}).get("videoId", "")
                if vid and vid not in video_meta:
                    video_ids.append(vid)
                    video_meta[vid] = {"query": query}
                    collected_for_query += 1

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

        print(f"    -> {collected_for_query}건 수집")

    print(f"  [YouTube] search 완료: {len(video_ids)}개 video_id (search 쿼터: {total_search_quota})")

    if not video_ids:
        return []

    # 2단계: videos.list로 full description 가져오기 (50개씩 배치)
    all_videos = []
    videos_quota = 0
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch),
            "key": api_key,
        }
        try:
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            videos_quota += 1
        except Exception as e:
            print(f"    [ERR] videos.list API 요청 실패: {e}")
            continue

        for item in data.get("items", []):
            vid = item.get("id", "")
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            meta = video_meta.get(vid, {})

            all_videos.append({
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", ""),
                "video_id": vid,
                "channelTitle": snippet.get("channelTitle", ""),
                "tags": snippet.get("tags", []),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "source": "youtube",
                "query": meta.get("query", ""),
            })

        print(f"  [YouTube] videos.list: {i + len(batch)}/{len(video_ids)} 처리")

    print(f"  [YouTube] 총 {len(all_videos)}건 수집 완료 (쿼터: search {total_search_quota} + videos {videos_quota} = {total_search_quota + videos_quota})")
    return all_videos
