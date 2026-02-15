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
    for kw, cat in categorized.items():
        if cat != "제외":
            result.append({
                "keyword": kw,
                "score": score_map.get(kw, 0.0),
                "category": cat,
            })

    excluded_count = len(keyword_list) - len(result)
    print(f"  [디자인 필터] {len(keyword_list)}개 → {len(result)}개 (제외: {excluded_count}개)")
    return result


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

    prompt = f"""다음 키워드들이 {TARGET_PRODUCT} 패키지 디자인의 콘셉트 방향으로 활용 가능한지 판단하세요.

활용 가능하면 아래 카테고리 중 하나를 지정하세요:
{categories_desc}

활용 불가능하면 "제외"로 표시하세요.

**판단 기준**: 이 키워드를 듣고 디자이너가 색감, 일러스트, 레이아웃, 소재 등의 방향을 떠올릴 수 있으면 "활용 가능"입니다.
예) "말차" → 그린톤 + 자연 감성 (O)
예) "가성비" → 디자인으로 전환 불가 (X)

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
