"""네이버 블로그 검색 API + 본문 크롤링"""

import os
import re
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

from config import COLLECT_LIMITS, ANALYSIS_PERIOD_MONTHS


def collect_naver_blogs(search_queries: list[str], mode: str = "daily") -> list[dict]:
    """네이버 블로그 검색 API로 URL 수집 → 본문 크롤링.

    Args:
        search_queries: 검색할 쿼리 리스트 (config.get_search_queries()로 생성)
        mode: "daily" (최신순만) 또는 "backfill" (최신순 + 관련도순)

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

    max_display = COLLECT_LIMITS.get(mode, COLLECT_LIMITS["daily"])["naver"]
    cutoff_date = datetime.now() - timedelta(days=ANALYSIS_PERIOD_MONTHS * 30)

    # backfill: date + sim 2회 수집으로 1년 커버 극대화
    # daily: date만 (최신 글 위주)
    sort_modes = ["date", "sim"] if mode == "backfill" else ["date"]

    seen_links = set()  # 중복 제거용
    all_posts = []

    for sort_type in sort_modes:
        print(f"\n  [네이버] 정렬: {sort_type} ({'관련도순' if sort_type == 'sim' else '최신순'})")

        for query in search_queries:
            print(f"  [네이버] 검색 중: {query} (sort={sort_type})")
            start = 1
            while start <= max_display:
                display = min(100, max_display - start + 1)
                params = {
                    "query": query,
                    "display": display,
                    "start": start,
                    "sort": sort_type,
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
                    link = item.get("link", "")

                    # 중복 제거
                    if link in seen_links:
                        continue
                    seen_links.add(link)

                    postdate = item.get("postdate", "")
                    if len(postdate) == 8:
                        try:
                            dt = datetime.strptime(postdate, "%Y%m%d")
                            if dt < cutoff_date:
                                continue
                        except ValueError:
                            pass

                    title = _strip_html(item.get("title", ""))
                    desc = _strip_html(item.get("description", ""))

                    all_posts.append({
                        "title": title,
                        "description": desc,  # API snippet (fallback)
                        "postdate": postdate,
                        "link": link,
                        "bloggername": item.get("bloggername", ""),
                        "source": "naver_blog",
                        "query": query,
                    })

                start += display
                time.sleep(0.1)

    print(f"\n  [네이버] API 수집 완료: {len(all_posts)}건 (중복 제거 후)")

    # 2단계: 본문 크롤링 (병렬 처리)
    if all_posts:
        _crawl_full_text(all_posts)

    return all_posts


def _crawl_full_text(posts: list[dict], max_workers: int = 5):
    """블로그 URL에서 실제 본문 크롤링 (병렬)"""
    naver_posts = [p for p in posts if "blog.naver.com" in p.get("link", "")]
    if not naver_posts:
        print("  [네이버] 크롤링 가능한 네이버 블로그 URL 없음")
        return

    print(f"  [네이버] 본문 크롤링 시작: {len(naver_posts)}건 (병렬 {max_workers})")
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_blog_content, p["link"]): p for p in naver_posts}
        for future in as_completed(futures):
            post = futures[future]
            try:
                full_text = future.result()
                if full_text and len(full_text) > len(post["description"]):
                    post["description"] = full_text
                    success += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

    print(f"  [네이버] 크롤링 완료: 성공 {success}건, 실패/스킵 {failed}건")


def _fetch_blog_content(url: str) -> str | None:
    """네이버 블로그 URL에서 본문 텍스트 추출"""
    try:
        blog_id, log_no = _parse_naver_blog_url(url)
        if not blog_id or not log_no:
            return None

        post_url = f"https://blog.naver.com/PostView.naver?blogId={blog_id}&logNo={log_no}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://blog.naver.com/",
        }

        resp = requests.get(post_url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # 본문 영역 추출 (여러 셀렉터 시도)
        content = None
        selectors = [
            "div.se-main-container",      # 스마트에디터 3
            "div#post-view",              # 구 에디터
            "div.post-view",              # 구 에디터 2
            "div#postViewArea",           # 구 에디터 3
        ]
        for sel in selectors:
            content = soup.select_one(sel)
            if content:
                break

        if not content:
            return None

        # 텍스트 추출 (이미지/링크 제외)
        for tag in content.find_all(["script", "style", "img", "video"]):
            tag.decompose()

        text = content.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 50:
            return None

        return text

    except Exception:
        return None


def _parse_naver_blog_url(url: str) -> tuple[str | None, str | None]:
    """네이버 블로그 URL에서 blogId, logNo 추출"""
    m = re.search(r"blog\.naver\.com/([^/?]+)/(\d+)", url)
    if m:
        return m.group(1), m.group(2)

    m_id = re.search(r"blogId=([^&]+)", url)
    m_no = re.search(r"logNo=(\d+)", url)
    if m_id and m_no:
        return m_id.group(1), m_no.group(1)

    return None, None


def _strip_html(text: str) -> str:
    """HTML 태그 제거"""
    return re.sub(r"<[^>]+>", "", text)
