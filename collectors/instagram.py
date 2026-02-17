"""instaloader를 통한 해시태그 기반 Instagram 데이터 수집 (로그인 불필요)"""

import re
import time
from datetime import datetime, timedelta

import instaloader

from config import COLLECT_LIMITS, ANALYSIS_PERIOD_MONTHS


def collect_instagram(search_queries: list[str], mode: str = "daily") -> list[dict]:
    """instaloader로 해시태그 기반 게시물 수집.

    Args:
        search_queries: 검색할 쿼리 리스트 (config.get_search_queries()로 생성)
        mode: "daily" 또는 "backfill"

    Returns:
        [{"title": str, "description": str, "postdate": str, "link": str, ...}, ...]
    """
    max_results = COLLECT_LIMITS.get(mode, COLLECT_LIMITS["daily"]).get("instagram", 50)
    cutoff_date = datetime.now() - timedelta(days=ANALYSIS_PERIOD_MONTHS * 30)

    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
    )

    seen_ids = set()
    all_posts = []

    for query in search_queries:
        hashtag_name = _query_to_hashtag(query)
        print(f"  [Instagram] 검색 중: {query} → #{hashtag_name}")

        try:
            hashtag = instaloader.Hashtag.from_name(L.context, hashtag_name)
        except Exception as e:
            print(f"    [WARN] 해시태그 조회 실패: #{hashtag_name} - {e}")
            continue

        collected = 0
        try:
            for post in hashtag.get_posts():
                shortcode = post.shortcode
                if shortcode in seen_ids:
                    continue
                seen_ids.add(shortcode)

                # 날짜 필터링
                post_date = post.date_utc
                if post_date < cutoff_date:
                    continue

                caption = post.caption or ""
                if not caption:
                    continue

                title = caption.split("\n")[0][:50]

                all_posts.append({
                    "title": title,
                    "description": caption,
                    "postdate": post_date.strftime("%Y%m%d"),
                    "link": f"https://www.instagram.com/p/{shortcode}/",
                    "source": "instagram",
                    "query": query,
                    "media_type": post.typename or "",
                    "like_count": post.likes or 0,
                    "comments_count": post.comments or 0,
                })

                collected += 1
                if collected >= max_results:
                    break
                if len(all_posts) >= max_results * len(search_queries):
                    break

        except Exception as e:
            print(f"    [WARN] 게시물 수집 중 오류 (IP 제한 가능): {e}")

        print(f"    -> {collected}건 수집")
        time.sleep(3)  # 비로그인 rate limit 대응 (넉넉하게)

    print(f"\n  [Instagram] 총 {len(all_posts)}건 수집 완료 (중복 제거 후)")
    return all_posts


def _query_to_hashtag(query: str) -> str:
    """검색 쿼리를 해시태그용 문자열로 변환.

    '과자 트렌드 디자인' → '과자트렌드디자인' (공백/특수문자 제거)
    """
    tag = re.sub(r"[^\w가-힣]", "", query)
    return tag
