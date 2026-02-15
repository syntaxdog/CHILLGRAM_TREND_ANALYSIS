"""GCP 리소스 초기 셋업: BigQuery 데이터셋/테이블 + GCS 버킷 생성"""

import os
from dotenv import load_dotenv
from google.cloud import bigquery, storage
from google.oauth2 import service_account

load_dotenv()

# === 설정 ===
PROJECT_ID = "chillgram-deploy"
SERVICE_ACCOUNT_KEY = os.path.join(os.path.dirname(__file__), "..", "data_analysis", "chillgram-deploy-key.json")

BQ_DATASET = "trend_analysis"
BQ_TABLE_RAW = "raw_contents"
BQ_TABLE_RESULTS = "trend_results"
BQ_LOCATION = "asia-northeast3"  # 서울

GCS_BUCKET = "chillgram-trend-data"
GCS_LOCATION = "asia-northeast3"


def get_credentials():
    """서비스 계정 인증"""
    # 여러 경로 시도
    key_paths = [
        SERVICE_ACCOUNT_KEY,
        os.path.join(os.path.dirname(__file__), "chillgram-deploy-key.json"),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
    ]
    for path in key_paths:
        if path and os.path.exists(path):
            print(f"  [인증] 서비스 계정 키: {path}")
            return service_account.Credentials.from_service_account_file(path)
    raise FileNotFoundError("서비스 계정 JSON 키 파일을 찾을 수 없습니다")


def setup_bigquery(credentials):
    """BigQuery 데이터셋 + 테이블 생성"""
    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    # 1. 데이터셋 생성
    dataset_ref = f"{PROJECT_ID}.{BQ_DATASET}"
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = BQ_LOCATION
    dataset = client.create_dataset(dataset, exists_ok=True)
    print(f"  [BigQuery] 데이터셋 생성: {dataset_ref}")

    # 2. raw_contents 테이블
    raw_schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED", description="고유키: {source}_{source_id_hash}"),
        bigquery.SchemaField("source", "STRING", mode="REQUIRED", description="naver_blog / youtube"),
        bigquery.SchemaField("source_id", "STRING", mode="REQUIRED", description="원본 고유 ID (URL or video_id)"),
        bigquery.SchemaField("title", "STRING", description="제목"),
        bigquery.SchemaField("description", "STRING", description="본문/설명"),
        bigquery.SchemaField("published_date", "DATE", description="게시일"),
        bigquery.SchemaField("query", "STRING", description="수집 시 검색 쿼리"),
        bigquery.SchemaField("channel", "STRING", description="블로거명/채널명"),
        bigquery.SchemaField("url", "STRING", description="원본 URL"),
        bigquery.SchemaField("collected_at", "TIMESTAMP", description="수집 시점"),
    ]
    raw_table_ref = f"{dataset_ref}.{BQ_TABLE_RAW}"
    raw_table = bigquery.Table(raw_table_ref, schema=raw_schema)
    raw_table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.MONTH,
        field="published_date",
    )
    raw_table = client.create_table(raw_table, exists_ok=True)
    print(f"  [BigQuery] 테이블 생성: {raw_table_ref} (월별 파티셔닝)")

    # 3. trend_results 테이블
    results_schema = [
        bigquery.SchemaField("analysis_date", "DATE", mode="REQUIRED", description="분석 실행일"),
        bigquery.SchemaField("rank", "INT64", description="순위"),
        bigquery.SchemaField("keyword", "STRING", description="키워드"),
        bigquery.SchemaField("category", "STRING", description="디자인 카테고리"),
        bigquery.SchemaField("trend", "STRING", description="상승/안정/하락"),
        bigquery.SchemaField("score", "FLOAT64", description="종합 점수"),
        bigquery.SchemaField("mention_count", "INT64", description="언급 횟수"),
        bigquery.SchemaField("reason", "STRING", description="선정 사유"),
    ]
    results_table_ref = f"{dataset_ref}.{BQ_TABLE_RESULTS}"
    results_table = bigquery.Table(results_table_ref, schema=results_schema)
    results_table = client.create_table(results_table, exists_ok=True)
    print(f"  [BigQuery] 테이블 생성: {results_table_ref}")

    return client


def setup_gcs(credentials):
    """GCS 버킷 생성"""
    client = storage.Client(project=PROJECT_ID, credentials=credentials)

    bucket = client.bucket(GCS_BUCKET)
    if not bucket.exists():
        bucket.location = GCS_LOCATION
        bucket.storage_class = "STANDARD"
        bucket = client.create_bucket(bucket)
        print(f"  [GCS] 버킷 생성: gs://{GCS_BUCKET}")
    else:
        print(f"  [GCS] 버킷 이미 존재: gs://{GCS_BUCKET}")

    return client


if __name__ == "__main__":
    print("=" * 50)
    print("GCP 리소스 셋업 시작")
    print("=" * 50)

    creds = get_credentials()
    setup_bigquery(creds)
    setup_gcs(creds)

    print("\n[OK] 셋업 완료!")
    print(f"  BigQuery: {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RAW}")
    print(f"  BigQuery: {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RESULTS}")
    print(f"  GCS:      gs://{GCS_BUCKET}/")
