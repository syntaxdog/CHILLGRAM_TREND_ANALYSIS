\# CHILLGRAM\_TREND\_ANALYSIS



\## 프로젝트 개요

\- \*\*주제\*\*: 트렌드 분석 키워드를 통한 식품 패키징 디자인 생성

\- \*\*이 레포의 목표\*\*: 소셜미디어 데이터에서 \*\*트렌드 키워드 20개를 추출\*\*하는 파이프라인

\- \*\*팀 프로젝트\*\*



\## 현재 진행 상황

\- 유튜브, 네이버 블로그, 인스타그램 크롤링 완료 (Day 1~4 백필)

\- 수집 데이터 → BigQuery + GCS에 적재 완료

\- GCP VM에서 크롤링 운영 중

\- 분석 파이프라인 (Phase 2~7) 구현 완료

\- CRON 설정


\## 파이프라인 흐름

```

Phase 1: 데이터 수집 (네이버 블로그, 유튜브, 인스타그램)

Phase 2: BigQuery 전체 데이터 로드

Phase 3: 전처리 (형태소 분석 - Kiwi)

Phase 4: 키워드 추출 (TF-IDF → KeyBERT → N-gram 확장 → 디자인 필터링)

Phase 5: 트렌드 분석 (Google Trends + 네이버 데이터랩 교차검증)

Phase 6: 최종 키워드 20개 선정 (스코어링)

Phase 7: 결과 BigQuery 저장

```



\## 실행 모드

\- `python main.py` — daily 모드 (수집 + 전체 분석)

\- `python main.py --backfill --day N` — 백필 모드 (수집 + 저장만, 분석 X)



\## 프로젝트 구조

```

main.py                  # 메인 파이프라인 오케스트레이터

config.py                # 설정값 (도메인 설정은 domain\_config.json에서 로드)

config\_generator.py      # domain\_config.json 생성기

collectors/

&nbsp; naver\_blog.py          # 네이버 블로그 검색 API

&nbsp; youtube.py             # YouTube Data API

&nbsp; instagram.py           # 인스타그램 크롤링

&nbsp; google\_trends.py       # Google Trends (pytrends)

&nbsp; naver\_datalab.py       # 네이버 데이터랩 (검색어트렌드 + 쇼핑인사이트)

processors/

&nbsp; preprocessor.py        # 형태소 분석 (Kiwi), 텍스트 정제

&nbsp; keyword\_extractor.py   # TF-IDF, KeyBERT, N-gram 키워드 추출

&nbsp; design\_filter.py       # 디자인 적합성 필터링

analyzers/

&nbsp; trend\_analyzer.py      # 트렌드 점수 산출 + 최종 키워드 선정

storage/

&nbsp; gcp\_client.py          # GCS 업로드, BigQuery INSERT/SELECT

output/                  # 결과 파일 (CSV, TXT, JSON) — .gitignore됨

```



\## 기술 스택

\- Python 3.12+

\- 형태소 분석: kiwipiepy

\- 키워드 추출: scikit-learn (TF-IDF), KeyBERT

\- 트렌드: pytrends, 네이버 데이터랩 API

\- 스토리지: Google Cloud BigQuery + GCS

\- 환경변수: .env (API 키 — NAVER, YouTube, GCP 등)



\## 스코어링 가중치

| 항목 | 비중 |

|------|------|

| 트렌드 추이 | 30% |

| 최신성 | 20% |

| 디자인 적합성 | 25% |

| 다양성 | 15% |

| 참신성 | 10% |



\## 주의사항

\- `.env` 파일에 API 키가 있음 — 절대 커밋/공유 금지

\- `domain\_config.json`에서 타겟 제품/소비자/쿼리/불용어 등 도메인 지식 관리

\- 백필은 4일로 분할 실행 (YouTube API 쿼터 제한)



