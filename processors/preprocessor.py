"""텍스트 전처리 - Kiwi 형태소 분석 기반"""

import re

from kiwipiepy import Kiwi

from config import USER_DICTIONARY, STOPWORDS, NEGATIVE_EXPRESSIONS


# Kiwi 초기화 (모듈 로드 시 1회)
_kiwi = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        print("  [전처리] Kiwi 형태소 분석기 초기화 중...")
        _kiwi = Kiwi()
        for word, tag in USER_DICTIONARY:
            _kiwi.add_user_word(word, tag)
        print(f"  [전처리] 사용자 사전 {len(USER_DICTIONARY)}개 등록 완료")
    return _kiwi


def preprocess_documents(documents: list[dict]) -> list[dict]:
    """수집된 문서 리스트를 전처리.

    Args:
        documents: [{"title": str, "description": str, ...}, ...]

    Returns:
        [{"text": 원문, "tokens": [명사 토큰], "date": str, "source": str}, ...]
    """
    kiwi = _get_kiwi()
    processed = []

    for doc in documents:
        # 제목 + 설명을 합침
        raw_text = f"{doc.get('title', '')} {doc.get('description', '')}"

        # 1. 정규화
        text = normalize(raw_text)
        if not text:
            continue

        # 2. 부정 문맥 필터링 - 부정어 포함 문장 제거
        sentences = _split_sentences(text)
        positive_sentences = [s for s in sentences if not _is_negative(s)]
        if not positive_sentences:
            continue
        text = " ".join(positive_sentences)

        # 3. Kiwi 형태소 분석 → 명사만 추출
        tokens = extract_nouns(kiwi, text)

        # 4. 불용어 제거 + 1글자 제거
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

        if tokens:
            date = doc.get("postdate") or doc.get("published_at", "")
            processed.append({
                "text": raw_text,
                "tokens": tokens,
                "date": date[:10] if date else "",
                "source": doc.get("source", ""),
            })

    print(f"  [전처리] {len(documents)}건 → {len(processed)}건 (부정/빈 문서 제거)")
    return processed


def normalize(text: str) -> str:
    """특수문자, 이모지, HTML 엔티티 제거"""
    # HTML 엔티티
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # 이모지 제거
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
        r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F]+",
        " ", text
    )
    # URL 제거
    text = re.sub(r"https?://\S+", " ", text)
    # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_nouns(kiwi: Kiwi, text: str) -> list[str]:
    """Kiwi로 명사(NNG, NNP) 추출"""
    result = kiwi.tokenize(text)
    nouns = [token.form for token in result if token.tag in ("NNG", "NNP")]
    return nouns


def _split_sentences(text: str) -> list[str]:
    """간단한 문장 분리"""
    sentences = re.split(r"[.!?\n]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _is_negative(sentence: str) -> bool:
    """부정 표현 포함 여부 확인"""
    return any(neg in sentence for neg in NEGATIVE_EXPRESSIONS)
