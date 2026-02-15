"""GCP 클라이언트 (BigQuery + GCS) - 싱글톤"""

import hashlib
import json
import os
from datetime import datetime, date

from google.cloud import bigquery, storage
from google.oauth2 import service_account

PROJECT_ID = "chillgram-deploy"
BQ_DATASET = "trend_analysis"
BQ_TABLE_RAW = "raw_contents"
BQ_TABLE_RESULTS = "trend_results"
GCS_BUCKET = "chillgram-trend-data"

_credentials = None
_bq_client = None
_gcs_client = None


def _get_credentials():
    global _credentials
    if _credentials:
        return _credentials

    key_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data_analysis", "chillgram-deploy-key.json"),
        os.path.join(os.path.dirname(__file__), "..", "chillgram-deploy-key.json"),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
    ]
    for path in key_paths:
        if path and os.path.exists(path):
            _credentials = service_account.Credentials.from_service_account_file(path)
            return _credentials
    raise FileNotFoundError("서비스 계정 JSON 키 파일을 찾을 수 없습니다")


def get_bq_client() -> bigquery.Client:
    global _bq_client
    if not _bq_client:
        _bq_client = bigquery.Client(project=PROJECT_ID, credentials=_get_credentials())
    return _bq_client


def get_gcs_client() -> storage.Client:
    global _gcs_client
    if not _gcs_client:
        _gcs_client = storage.Client(project=PROJECT_ID, credentials=_get_credentials())
    return _gcs_client


def make_id(source: str, source_id: str) -> str:
    """고유 ID 생성: source_hash(source_id)"""
    h = hashlib.md5(source_id.encode()).hexdigest()[:12]
    return f"{source}_{h}"


def get_existing_ids() -> set[str]:
    """BigQuery에서 이미 적재된 id 목록 조회"""
    client = get_bq_client()
    query = f"SELECT id FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RAW}`"
    try:
        result = client.query(query).result()
        return {row.id for row in result}
    except Exception as e:
        print(f"  [WARN] 기존 ID 조회 실패 (첫 실행?): {e}")
        return set()


def upload_to_gcs(data: dict, prefix: str) -> str:
    """JSON 데이터를 GCS에 업로드. 반환: gs:// 경로"""
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    blob_path = f"raw/{prefix}/{today}_{timestamp}.json"

    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    gs_path = f"gs://{GCS_BUCKET}/{blob_path}"
    print(f"  [GCS] 업로드 완료: {gs_path}")
    return gs_path


def insert_raw_contents(documents: list[dict]) -> int:
    """수집 데이터를 BigQuery raw_contents에 INSERT (중복 제외).

    Args:
        documents: 네이버/유튜브 수집 결과 리스트

    Returns:
        신규 INSERT된 건수
    """
    if not documents:
        return 0

    existing_ids = get_existing_ids()
    now = datetime.utcnow().isoformat()

    rows = []
    for doc in documents:
        source = doc.get("source", "unknown")

        # source_id 결정
        if source == "naver_blog":
            source_id = doc.get("link", "")
            url = source_id
            channel = doc.get("bloggername", "")
            raw_date = doc.get("postdate", "")
            if len(raw_date) == 8:
                pub_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            else:
                pub_date = None
        elif source == "youtube":
            source_id = doc.get("video_id", "")
            url = f"https://youtube.com/watch?v={source_id}" if source_id else ""
            channel = doc.get("channelTitle", "")
            raw_date = doc.get("published_at", "")
            pub_date = raw_date[:10] if len(raw_date) >= 10 else None
        else:
            source_id = doc.get("link", doc.get("video_id", ""))
            url = source_id
            channel = ""
            pub_date = None

        doc_id = make_id(source, source_id)

        # 중복 체크
        if doc_id in existing_ids:
            continue

        row = {
            "id": doc_id,
            "source": source,
            "source_id": source_id,
            "title": doc.get("title", ""),
            "description": doc.get("description", ""),
            "published_date": pub_date,
            "query": doc.get("query", ""),
            "channel": channel,
            "url": url,
            "collected_at": now,
        }
        rows.append(row)

    if not rows:
        print("  [BigQuery] 신규 데이터 없음 (전부 중복)")
        return 0

    client = get_bq_client()
    table_ref = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RAW}"
    errors = client.insert_rows_json(table_ref, rows)

    if errors:
        print(f"  [BigQuery] INSERT 에러: {errors[:3]}")
        return 0

    print(f"  [BigQuery] {len(rows)}건 INSERT 완료 (중복 {len(documents) - len(rows)}건 제외)")
    return len(rows)


def insert_trend_results(keywords: list[dict]) -> int:
    """분석 결과를 BigQuery trend_results에 INSERT.

    Args:
        keywords: 최종 키워드 리스트 [{"rank", "keyword", "category", "trend", ...}]

    Returns:
        INSERT된 건수
    """
    if not keywords:
        return 0

    today = date.today().isoformat()
    rows = []
    for kw in keywords:
        rows.append({
            "analysis_date": today,
            "rank": kw.get("rank", 0),
            "keyword": kw.get("keyword", ""),
            "category": kw.get("category", ""),
            "trend": kw.get("trend", ""),
            "score": kw.get("final_score", 0.0),
            "mention_count": kw.get("mention_count", 0),
            "reason": kw.get("reason", ""),
        })

    client = get_bq_client()
    table_ref = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RESULTS}"
    errors = client.insert_rows_json(table_ref, rows)

    if errors:
        print(f"  [BigQuery] 결과 INSERT 에러: {errors[:3]}")
        return 0

    print(f"  [BigQuery] 분석 결과 {len(rows)}건 INSERT 완료")
    return len(rows)
