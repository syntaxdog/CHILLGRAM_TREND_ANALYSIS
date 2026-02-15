"""키워드 추출 - TF-IDF + KeyBERT"""

from sklearn.feature_extraction.text import TfidfVectorizer

from config import TFIDF_TOP_N, KEYBERT_TOP_N


def extract_tfidf_keywords(processed_docs: list[dict], top_n: int = TFIDF_TOP_N) -> list[tuple[str, float]]:
    """TF-IDF로 상위 키워드 추출.

    Args:
        processed_docs: 전처리된 문서 리스트 [{"tokens": [...], ...}]
        top_n: 상위 몇 개 추출

    Returns:
        [(키워드, TF-IDF 점수), ...] 상위 top_n개
    """
    if not processed_docs:
        return []

    # 각 문서의 토큰을 공백으로 연결
    corpus = [" ".join(doc["tokens"]) for doc in processed_docs]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,        # 최소 2개 문서에 등장
        max_df=0.8,      # 80% 이상 문서에 등장하면 제외
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # 전체 문서에 대한 평균 TF-IDF 점수
    avg_scores = tfidf_matrix.mean(axis=0).A1
    keyword_scores = list(zip(feature_names, avg_scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)

    top_keywords = keyword_scores[:top_n]
    print(f"  [TF-IDF] 상위 {len(top_keywords)}개 키워드 추출 완료")
    return top_keywords


def extract_keybert_keywords(
    processed_docs: list[dict],
    candidates: list[str],
    top_n: int = KEYBERT_TOP_N,
    diversity: float = 0.5,
) -> list[tuple[str, float]]:
    """KeyBERT로 후보 키워드 중 핵심 키워드 선정.

    Args:
        processed_docs: 전처리된 문서 리스트
        candidates: TF-IDF에서 나온 후보 키워드 리스트
        top_n: 최종 추출 키워드 수
        diversity: MMR 다양성 파라미터 (0~1)

    Returns:
        [(키워드, 유사도 점수), ...]
    """
    if not processed_docs or not candidates:
        return [(kw, 0.0) for kw in candidates[:top_n]]

    try:
        from keybert import KeyBERT
    except ImportError:
        print("[WARN] keybert 미설치 - TF-IDF 결과만 사용합니다")
        return [(kw, 0.0) for kw in candidates[:top_n]]

    # 전체 문서를 하나의 큰 텍스트로 합침
    full_text = " ".join(" ".join(doc["tokens"]) for doc in processed_docs)

    print("  [KeyBERT] 모델 로딩 중 (ko-sroberta-multitask)...")
    try:
        kw_model = KeyBERT(model="jhgan/ko-sroberta-multitask")
    except Exception:
        print("  [KeyBERT] ko-sroberta-multitask 로드 실패, 기본 모델 사용")
        kw_model = KeyBERT()

    # KeyBERT로 후보 중 최적 키워드 선정
    keywords = kw_model.extract_keywords(
        full_text,
        candidates=candidates,
        top_n=top_n,
        use_mmr=True,
        diversity=diversity,
    )

    # 유사 키워드 중복 제거 (예: "미니멀" vs "미니멀리즘")
    keywords = _deduplicate(keywords)

    print(f"  [KeyBERT] {len(keywords)}개 키워드 선정 완료")
    return keywords


def _deduplicate(keywords: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """포함 관계에 있는 유사 키워드 중복 제거 (긴 쪽 유지)"""
    result = []
    sorted_kw = sorted(keywords, key=lambda x: len(x[0]), reverse=True)

    for word, score in sorted_kw:
        is_dup = False
        for existing_word, _ in result:
            if word in existing_word or existing_word in word:
                is_dup = True
                break
        if not is_dup:
            result.append((word, score))

    return result
