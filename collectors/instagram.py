"""Instagram Graph API를 통한 해시태그 기반 데이터 수집"""

import os
import re
import time
from datetime import datetime, timedelta

import requests

from config import COLLECT_LIMITS, ANALYSIS_PERIOD_MONTHS

API_BASE = "https://graph.facebook.com/v21.0"


def collect_instagram(search_queries: list[str], mode: str = "daily") -> list[dict]:
    """Instagram Graph API로 해시태그 기반 게시물 수집.

    Args:
        search_queries: 검색할 쿼리 리스트 (config.get_search_queries()로 생성)
        mode: "daily" 또는 "backfill"

    Returns:
        [{"title": str, "description": str, "postdate": str, "link": str, ...}, ...]
    """
    access_token = os.getenv("META_ACCESS_TOKEN")
    user_id = os.getenv("META_IG_USER_ID")

    if not access_token or not user_id:
        print("[WARN] META_ACCESS_TOKEN / META_IG_USER_ID 미설정 - Instagram 수집 스킵")
        return []

    max_results = COLLECT_LIMITS.get(mode, COLLECT_LIMITS["daily"]).get("instagram", 50)
    cutoff_date = datetime.now() - timedelta(days=ANALYSIS_PERIOD_MONTHS * 30)

    seen_ids = set()
    all_posts = []

    for query in search_queries:
        hashtag = _query_to_hashtag(query)
        print(f"  [Instagram] 검색 중: {query} → #{hashtag}")

        # 1단계: 해시태그 ID 조회
        hashtag_id = _get_hashtag_id(hashtag, user_id, access_token)
        if not hashtag_id:
            print(f"    [WARN] 해시태그 ID 조회 실패: #{hashtag}")
            continue

        # 2단계: 최근 게시물 조회
        posts = _get_recent_media(hashtag_id, user_id, access_token)

        collected = 0
        for post in posts:
            post_id = post.get("id", "")
            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            # 날짜 필터링
            timestamp = post.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if dt.replace(tzinfo=None) < cutoff_date:
                        continue
                    postdate = dt.strftime("%Y%m%d")
                except (ValueError, AttributeError):
                    postdate = ""
            else:
                postdate = ""

            caption = post.get("caption", "")
            if not caption:
                continue

            # 제목: 캡션 첫 줄 (최대 50자)
            title = caption.split("\n")[0][:50]

            all_posts.append({
                "title": title,
                "description": caption,
                "postdate": postdate,
                "link": post.get("permalink", ""),
                "source": "instagram",
                "query": query,
                "media_type": post.get("media_type", ""),
                "like_count": post.get("like_count", 0),
                "comments_count": post.get("comments_count", 0),
            })

            collected += 1
            if len(all_posts) >= max_results * len(search_queries):
                break

        print(f"    -> {collected}건 수집")
        time.sleep(1)  # rate limit 대응

    print(f"\n  [Instagram] 총 {len(all_posts)}건 수집 완료 (중복 제거 후)")
    return all_posts


def _query_to_hashtag(query: str) -> str:
    """검색 쿼리를 해시태그용 문자열로 변환.

    '과자 트렌드 디자인' → '과자트렌드디자인' (공백 제거)
    """
    # 특수문자 제거, 공백 제거
    tag = re.sub(r"[^\w가-힣]", "", query)
    return tag


def _get_hashtag_id(hashtag: str, user_id: str, access_token: str) -> str | None:
    """해시태그 이름으로 해시태그 ID 조회."""
    try:
        resp = requests.get(
            f"{API_BASE}/ig_hashtag_search",
            params={
                "q": hashtag,
                "user_id": user_id,
                "access_token": access_token,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return data[0]["id"]
    except Exception as e:
        print(f"    [ERR] 해시태그 검색 API 실패: {e}")
    return None


def _get_recent_media(
    hashtag_id: str, user_id: str, access_token: str
) -> list[dict]:
    """해시태그의 최근 게시물 조회."""
    fields = "id,caption,media_type,permalink,timestamp,like_count,comments_count"
    try:
        resp = requests.get(
            f"{API_BASE}/{hashtag_id}/recent_media",
            params={
                "user_id": user_id,
                "fields": fields,
                "access_token": access_token,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        print(f"    [ERR] recent_media API 실패: {e}")
        return []
