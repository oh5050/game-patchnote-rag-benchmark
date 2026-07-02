# ---------------------------------------------------------------------------
# 역할: 재사용 가능한 순수 평가 지표 함수 모음(상태 없음, 부작용 없음).
# 입력:
#   - ranked      : 예측 doc_id 순위 목록(list[str], 1위부터)
#   - gold_set    : 관련 doc_id 집합(binary, set[str])
#   - rel_grades  : graded relevance {doc_id: relevance}  (nDCG 용)
# 출력:
#   - 스칼라 지표 값 (hit@k, recall@k, MRR, nDCG@k)
# 의존관계:
#   - 표준 라이브러리(math)만 사용. 다른 프로젝트 모듈에 비의존(leaf 모듈).
# 사용처: src.evaluation.search_eval, (일부) src.evaluation.answer_eval
# ---------------------------------------------------------------------------

from math import log2


def hit_at_k(ranked: list[str], gold_set: set[str], k: int) -> float:
    """
    Hit@k: 상위 k개 결과 안에 gold 문서가 하나라도 있으면 1.0, 없으면 0.0.
    (해당 질문에 gold 문서가 하나도 없으면 0.0)
    """
    if not gold_set:
        return 0.0
    return 1.0 if any(doc_id in gold_set for doc_id in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], gold_set: set[str], k: int) -> float:
    """
    Recall@k: (상위 k개 안에 든 gold 문서 수) / (전체 gold 문서 수).
    gold 문서가 여러 개일 때(예: 2개) 얼마나 많이 회수했는지를 측정한다.
    gold 문서가 없으면 0.0.
    """
    if not gold_set:
        return 0.0
    top_k = ranked[:k]
    found = sum(1 for doc_id in gold_set if doc_id in top_k)
    return found / len(gold_set)


def mrr(ranked: list[str], gold_set: set[str]) -> float:
    """
    MRR(단일 질의의 Reciprocal Rank): 첫 번째 gold 문서의 역순위(1/rank).
    순위는 1부터 시작하며, gold 문서가 하나도 없으면 0.0.
    """
    if not gold_set:
        return 0.0
    for idx, doc_id in enumerate(ranked, start=1):
        if doc_id in gold_set:
            return 1.0 / idx
    return 0.0


def dcg_at_k(ranked: list[str], rel_grades: dict[str, float], k: int) -> float:
    """
    DCG@k: sum_{i=1..k} rel_i / log2(i + 1).
    rel_i 는 i번째 결과 문서의 graded relevance (없으면 0).
    """
    gains = [rel_grades.get(doc_id, 0.0) for doc_id in ranked[:k]]
    return sum(g / log2(i + 2) for i, g in enumerate(gains))  # i=0 → log2(2)=1


def ndcg_at_k(ranked: list[str], rel_grades: dict[str, float], k: int) -> float:
    """
    nDCG@k: DCG@k 를 이상적 순서(IDCG@k)로 정규화한 값 [0, 1].
    graded relevance(예: 2=매우 관련, 1=관련)를 이득(gain)으로 사용한다.
    관련 문서가 전혀 없으면(IDCG=0) 0.0.
    """
    dcg = dcg_at_k(ranked, rel_grades, k)
    ideal = sorted(rel_grades.values(), reverse=True)[:k]
    idcg = sum(g / log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
