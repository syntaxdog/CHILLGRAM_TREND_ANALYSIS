"""디자인 적합성 필터링 - Gemini API 또는 규칙 기반"""

import json
import os

from config import DESIGN_CATEGORIES, TARGET_PRODUCT


def filter_design_keywords(
    keywords: list[tuple[str, float]],
) -> list[dict]:
    """키워드 후보를 패키지 디자인 적합성으로 필터링.

    Args:
        keywords: [(키워드, 점수), ...]

    Returns:
        [{"keyword": str, "score": float, "category": str}, ...]
        category가 "제외"인 것은 제거된 상태
    """
    keyword_list = [kw for kw, _ in keywords]
    score_map = {kw: score for kw, score in keywords}

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        print("  [디자인 필터] Gemini API로 분류 중...")
        categorized = _filter_with_gemini(keyword_list, api_key)
    else:
        print("  [디자인 필터] API 키 없음 - 규칙 기반 필터링 적용")
        categorized = _filter_with_rules(keyword_list)

    # "제외" 카테고리 제거
    result = []
    excluded = []
    for kw, cat in categorized.items():
        if cat != "제외":
            result.append({
                "keyword": kw,
                "score": score_map.get(kw, 0.0),
                "category": cat,
            })
        else:
            excluded.append({
                "keyword": kw,
                "score": score_map.get(kw, 0.0),
            })

    excluded_count = len(excluded)
    print(f"  [디자인 필터] {len(keyword_list)}개 → {len(result)}개 (제외: {excluded_count}개)")

    # 폴백: 20개 미만이면 제외된 키워드 중 점수 높은 순으로 복구
    from config import FINAL_KEYWORD_COUNT
    min_required = FINAL_KEYWORD_COUNT
    if len(result) < min_required and excluded:
        excluded.sort(key=lambda x: x["score"], reverse=True)
        # 카테고리 목록에서 기본값 선택
        default_categories = list(DESIGN_CATEGORIES.keys())
        for ex_kw in excluded:
            if len(result) >= min_required:
                break
            # 규칙 기반으로 카테고리 재매칭 시도, 없으면 기본 카테고리 배정
            cat = _guess_category(ex_kw["keyword"])
            result.append({
                "keyword": ex_kw["keyword"],
                "score": ex_kw["score"],
                "category": cat,
            })
        recovered = len(result) - (len(keyword_list) - excluded_count)
        print(f"  [디자인 필터] 키워드 부족 → {recovered}개 복구 (총 {len(result)}개)")

    return result


def _guess_category(keyword: str) -> str:
    """키워드에 가장 가까운 카테고리를 추측. 매칭 안 되면 첫 번째 카테고리 반환."""
    for cat, words in DESIGN_CATEGORIES.items():
        for w in words:
            if w in keyword or keyword in w:
                return cat
    # 기본값: 첫 번째 카테고리
    return list(DESIGN_CATEGORIES.keys())[0]


def _filter_with_gemini(keywords: list[str], api_key: str) -> dict[str, str]:
    """Gemini API로 키워드 분류"""
    try:
        import google.generativeai as genai
    except ImportError:
        print("    [WARN] google-generativeai 패키지 미설치 - 규칙 기반 폴백")
        print("    pip install google-generativeai 로 설치하세요")
        return _filter_with_rules(keywords)

    genai.configure(api_key=api_key)

    categories_desc = "\n".join(
        f"- {cat}: {', '.join(examples[:5])}" for cat, examples in DESIGN_CATEGORIES.items()
    )

    prompt = f"""당신은 {TARGET_PRODUCT} 패키지 디자인 전문가입니다.
다음 키워드들이 패키지 디자인의 **시각적 요소(일러스트, 색감, 패턴, 타이포그래피, 레이아웃, 소재 질감)**로 직접 표현 가능한지 판단하세요.

활용 가능하면 아래 카테고리 중 하나를 지정하세요:
{categories_desc}

**중요: 최소 {min(len(keywords), 25)}개 이상을 포함시키세요. 제외는 극히 제한적으로만 하세요.**

**"제외" 기준 — 아래에 **정확히** 해당하는 경우만 제외하세요:**
- 가격/배송/구매 관련 (가성비, 배송, 할인)
- 추상적 감정/평가 (맛있다, 추천, 별로)
- 디자인과 전혀 무관한 일반 명사 (회사, 매장, 직원)

조금이라도 디자인 요소로 해석 가능하면 반드시 포함하세요.

**가급적 포함시키세요:**
- 맛/원료 키워드 → 색감, 일러스트로 전환 가능 (예: 말차→그린톤, 딸기→핑크+과일 일러스트)
- 감성/무드 키워드 → 디자인 톤앤매너로 활용 (예: 레트로→뉴트로 그래픽)
- 시즌/이벤트 → 한정판 패키지 콘셉트 (예: 벚꽃→봄 시즌 패키지)
- 소재/형태 → 패키지 구조에 직접 반영 (예: 크래프트지→친환경 느낌)
- 라이프스타일 → 타겟 감성 반영 (예: 홈카페→아늑한 무드)

키워드 목록:
{json.dumps(keywords, ensure_ascii=False)}

반드시 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요:
{{"키워드1": "카테고리명 또는 제외", "키워드2": "카테고리명 또는 제외", ...}}"""

    try:
        model = genai.GenerativeModel("gemini-3-pro-preview")
        response = model.generate_content(prompt)
        text = response.text.strip()
        # JSON 블록 추출
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"    [ERR] Gemini API 호출 실패: {e}")
        return _filter_with_rules(keywords)


def _filter_with_rules(keywords: list[str]) -> dict[str, str]:
    """규칙 기반 간이 필터링 - 사전 정의된 카테고리에 매칭"""
    result = {}

    # 카테고리 키워드를 역매핑
    category_lookup = {}
    for cat, words in DESIGN_CATEGORIES.items():
        for w in words:
            category_lookup[w] = cat

    for kw in keywords:
        # 정확히 매치
        if kw in category_lookup:
            result[kw] = category_lookup[kw]
            continue

        # 부분 매치 (카테고리 키워드가 현재 키워드에 포함되어 있는 경우)
        matched = False
        for dict_word, cat in category_lookup.items():
            if dict_word in kw or kw in dict_word:
                result[kw] = cat
                matched = True
                break

        if not matched:
            result[kw] = "제외"

    return result
