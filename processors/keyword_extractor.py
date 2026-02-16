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

    # 명사구(2~3어절)를 후보에 추가 — 긴 키워드 확보
    phrase_candidates = _collect_noun_phrase_candidates(processed_docs)
    combined_candidates = list(dict.fromkeys(candidates + phrase_candidates))
    print(f"  [KeyBERT] 후보: 단일어 {len(candidates)}개 + 명사구 {len(phrase_candidates)}개 = {len(combined_candidates)}개")

    # 전체 문서를 하나의 텍스트로 합침 (메모리 제한: 최대 50,000자)
    MAX_TEXT_LEN = 50000
    parts = []
    current_len = 0
    for doc in processed_docs:
        part = " ".join(doc["tokens"]) + " " + " ".join(doc.get("noun_phrases", []))
        if current_len + len(part) > MAX_TEXT_LEN:
            break
        parts.append(part)
        current_len += len(part)
    full_text = " ".join(parts)
    print(f"  [KeyBERT] 텍스트 길이: {len(full_text):,}자 ({len(parts)}/{len(processed_docs)}문서 사용)")

    print("  [KeyBERT] 모델 로딩 중 (ko-sroberta-multitask)...")
    try:
        kw_model = KeyBERT(model="jhgan/ko-sroberta-multitask")
    except Exception:
        print("  [KeyBERT] ko-sroberta-multitask 로드 실패, 기본 모델 사용")
        kw_model = KeyBERT()

    # KeyBERT로 후보 중 최적 키워드 선정
    keywords = kw_model.extract_keywords(
        full_text,
        candidates=combined_candidates,
        top_n=top_n,
        use_mmr=True,
        diversity=diversity,
    )

    # 유사 키워드 중복 제거 (예: "미니멀" vs "미니멀리즘")
    keywords = _deduplicate(keywords)

    print(f"  [KeyBERT] {len(keywords)}개 키워드 선정 완료")
    return keywords


def _collect_noun_phrase_candidates(
    processed_docs: list[dict], min_doc_freq: int = 2, max_phrases: int = 100
) -> list[str]:
    """문서들에서 빈출 명사구를 후보로 수집."""
    from collections import Counter

    phrase_counter = Counter()
    doc_freq = Counter()

    for doc in processed_docs:
        phrases = doc.get("noun_phrases", [])
        seen = set()
        for p in phrases:
            phrase_counter[p] += 1
            if p not in seen:
                doc_freq[p] += 1
                seen.add(p)

    # 최소 min_doc_freq개 문서에 등장하는 것만
    qualified = [
        (phrase, count)
        for phrase, count in phrase_counter.items()
        if doc_freq[phrase] >= min_doc_freq
    ]
    qualified.sort(key=lambda x: x[1], reverse=True)

    return [phrase for phrase, _ in qualified[:max_phrases]]


def extract_extended_keywords(
    processed_docs: list[dict], top_n: int = TFIDF_TOP_N
) -> list[tuple[str, float]]:
    """전처리된 문서의 noun_phrases에서 TF-IDF로 상위 확장 키워드 추출.

    Args:
        processed_docs: 전처리된 문서 리스트 [{"noun_phrases": [...], ...}]
        top_n: 상위 몇 개 추출

    Returns:
        [(확장 키워드, TF-IDF 점수), ...] 상위 top_n개
    """
    if not processed_docs:
        return []

    # 각 문서의 noun_phrases를 하나의 문자열로 연결
    corpus = []
    for doc in processed_docs:
        phrases = doc.get("noun_phrases", [])
        if phrases:
            corpus.append(" | ".join(phrases))
        else:
            corpus.append("")

    # 빈 문서만 있으면 빈 결과 반환
    if not any(corpus):
        print("  [Extended TF-IDF] 명사구 데이터가 없습니다")
        return []

    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.8,
        analyzer="word",
        token_pattern=r"[가-힣a-zA-Z0-9]+(?:\s[가-힣a-zA-Z0-9]+)*",
    )

    # "|" 구분자를 활용해 각 구문을 독립 토큰으로 처리
    # 대신 CountVectorizer 방식으로 직접 빈도 계산
    from collections import Counter
    phrase_counter = Counter()
    doc_freq = Counter()

    for doc in processed_docs:
        phrases = doc.get("noun_phrases", [])
        seen = set()
        for p in phrases:
            phrase_counter[p] += 1
            if p not in seen:
                doc_freq[p] += 1
                seen.add(p)

    n_docs = len(processed_docs)
    import math
    scored = []
    for phrase, count in phrase_counter.items():
        df = doc_freq[phrase]
        if df < 2 or df > n_docs * 0.8:
            continue
        tfidf = count * math.log(n_docs / (1 + df))
        scored.append((phrase, tfidf))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]

    print(f"  [Extended TF-IDF] 상위 {len(top)}개 확장 키워드 추출 완료")
    return top


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
