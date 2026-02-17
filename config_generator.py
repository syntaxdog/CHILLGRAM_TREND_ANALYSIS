"""도메인 설정 자동 생성 — Gemini로 1회 생성 후 JSON 저장"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

OUTPUT_DIR = Path(__file__).parent / "output"
CONFIG_PATH = OUTPUT_DIR / "domain_configs.json"

GEMINI_MODEL = "gemini-3-flash-preview"


def generate_domain_config(
    target_product: str = "가공식품 (과자·스낵·음료·디저트)",
    target_consumer: str = "MZ세대 (20~30대)",
) -> dict:
    """Gemini로 도메인 설정을 생성하고 JSON으로 저장."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 설정되지 않았습니다")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    now = datetime.now()
    current_year = now.year

    prompt = f"""당신은 한국 식품 패키지 디자인 트렌드 분석 전문가입니다.
아래 타겟 정보를 바탕으로 트렌드 분석 파이프라인에 필요한 도메인 설정을 생성하세요.

현재 연도: {current_year}년
타겟 제품군: {target_product}
타겟 소비자: {target_consumer}

아래 7가지 항목을 JSON 형식으로 생성하세요:

1. base_queries (10~15개):
   시즌에 관계없이 항상 사용하는 네이버 블로그/YouTube 검색 쿼리.
   연도를 포함할 경우 반드시 {current_year}를 사용하세요.
   - 패키지 디자인 트렌드 전반 (예: "{current_year} 식품 패키지 디자인 트렌드")
   - 신상/트렌드 탐색 (예: "편의점 신상 과자", "과자 트렌드")
   - 컬래버/한정판 (예: "컬래버 과자 한정판")
   - 소재/구조 (예: "친환경 패키지 디자인")
   - 브랜딩/디자인 분석 (예: "가공식품 브랜딩 트렌드")

2. seasonal_queries (12개월, 각 2~4개):
   해당 월에만 사용하는 시즌 검색 쿼리.
   일일 크롤링 시 현재 월의 쿼리만 사용하고, 백필 시 전체를 사용합니다.
   형식: {{"1": ["설날 선물 과자 패키지", ...], "2": ["발렌타인데이 초콜릿 패키지", ...], ...}}

3. user_dictionary (60~100개):
   한국어 형태소 분석기(Kiwi)에 등록할 도메인 전문 용어.
   각 항목은 ["용어", "NNG"] 형태.
   포함해야 할 분야:
   - 디자인 용어 (뉴트로, 미니멀리즘, 그라데이션, 파스텔톤, 어스톤, 산세리프 등)
   - 소재/구조 (크래프트지, 스탠딩파우치, 지퍼백, 이지컷 등)
   - 트렌드 맛/원료 (말차, 두바이초콜릿, 흑임자, 프로틴, 피스타치오 등)
   - 건강/가치 (제로슈거, 헬시플레저, 비건, 클린라벨, 글루텐프리 등)
   - 라이프스타일 (홈카페, 혼술, Y2K 등)
   - 이벤트 (빼빼로데이, 어버이날 등)

4. stopwords (40~60개):
   디자인 트렌드와 무관한 고빈도 불용어.
   - 일반 (맛있다, 추천, 먹방, 진짜, 너무 등)
   - 쇼핑/배송 (가격, 배송, 택배, 결제 등)
   - 대명사/지시어 (이것, 저것, 여기 등)
   - 일반 동사 (하다, 있다, 되다 등)

5. negative_expressions (10~15개):
   부정적 디자인 평가를 필터링할 표현.
   예: "별로", "안 좋", "불편", "과대포장", "촌스러", "실망"

6. design_categories (5~7개 카테고리, 각 10~20개 용어):
   패키지 디자인 콘셉트로 활용 가능한 키워드 분류 체계.
   권장 카테고리:
   - 트렌드맛·원료: 디자인 콘셉트로 전환 가능한 맛/원료
   - 시즌·이벤트: 시즌/기념일/한정판
   - 비주얼·감성: 디자인 스타일/무드
   - 건강·가치소비: 건강/친환경/가치 트렌드
   - 소재·구조: 패키지 소재/형태/구조

7. seasonal_keywords (12개월, 각 5~15개):
   각 월에 자연스럽게 급증하는 계절성 키워드.
   트렌드 분석 시 계절적 스파이크를 진짜 트렌드와 구분하는 데 사용.
   형식: {{"1": ["설날", "명절", ...], "2": ["발렌타인데이", ...], ...}}
   - 공휴일/기념일, 계절, 시즌 한정, 계절 맛/원료 포함

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력:
{{
  "target_product": "{target_product}",
  "target_consumer": "{target_consumer}",
  "base_queries": [...],
  "seasonal_queries": {{"1": [...], "2": [...], ..., "12": [...]}},
  "user_dictionary": [["용어", "NNG"], ...],
  "stopwords": [...],
  "negative_expressions": [...],
  "design_categories": {{"카테고리명": ["키워드", ...], ...}},
  "seasonal_keywords": {{"1": [...], "2": [...], ..., "12": [...]}}
}}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"  [ConfigGenerator] Gemini ({GEMINI_MODEL}) 호출 중... (시도 {attempt + 1}/{max_retries})")
            response = model.generate_content(prompt)
            text = response.text.strip()

            # JSON 블록 추출
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            config = json.loads(text)
            _validate_config(config)

            # 저장
            OUTPUT_DIR.mkdir(exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            print(f"  [ConfigGenerator] 설정 생성 완료!")
            print(f"    - 기본 쿼리: {len(config['base_queries'])}개")
            seasonal_q_total = sum(len(v) for v in config["seasonal_queries"].values())
            print(f"    - 시즌 쿼리: {seasonal_q_total}개 (12개월)")
            print(f"    - 사용자 사전: {len(config['user_dictionary'])}개")
            print(f"    - 불용어: {len(config['stopwords'])}개")
            print(f"    - 부정 표현: {len(config['negative_expressions'])}개")
            print(f"    - 디자인 카테고리: {len(config['design_categories'])}개")
            cat_total = sum(len(v) for v in config["design_categories"].values())
            print(f"    - 카테고리 내 용어: {cat_total}개")
            seasonal_total = sum(len(v) for v in config["seasonal_keywords"].values())
            print(f"    - 계절 키워드: {seasonal_total}개")
            print(f"    - 저장 위치: {CONFIG_PATH}")
            return config

        except json.JSONDecodeError as e:
            print(f"  [ConfigGenerator] JSON 파싱 실패: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"  [ConfigGenerator] 오류: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    raise RuntimeError(f"Gemini 설정 생성 실패 ({max_retries}회 시도)")


def _validate_config(config: dict):
    """생성된 설정의 필수 구조 검증."""
    required = [
        "base_queries", "seasonal_queries", "user_dictionary", "stopwords",
        "negative_expressions", "design_categories", "seasonal_keywords",
    ]
    for key in required:
        if key not in config:
            raise ValueError(f"필수 키 누락: {key}")

    if len(config["base_queries"]) < 5:
        raise ValueError(f"base_queries가 너무 적음: {len(config['base_queries'])}개")

    if not isinstance(config["seasonal_queries"], dict) or len(config["seasonal_queries"]) < 12:
        raise ValueError("seasonal_queries는 12개월 모두 필요")

    if len(config["user_dictionary"]) < 20:
        raise ValueError(f"user_dictionary가 너무 적음: {len(config['user_dictionary'])}개")

    if not isinstance(config["design_categories"], dict) or len(config["design_categories"]) < 3:
        raise ValueError("design_categories는 3개 이상의 카테고리 필요")

    if not isinstance(config["seasonal_keywords"], dict) or len(config["seasonal_keywords"]) < 12:
        raise ValueError("seasonal_keywords는 12개월 모두 필요")


def load_domain_config() -> dict:
    """저장된 domain_config.json을 로드."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"domain_config.json이 없습니다. 먼저 실행하세요:\n"
            f"  python config_generator.py"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    config = generate_domain_config()
    print("\n생성 완료!")
