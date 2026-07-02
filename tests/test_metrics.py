# ---------------------------------------------------------------------------
# 역할: src.evaluation.metrics 순수 함수의 단위 테스트.
# 시나리오: gold 문서가 top-2에 있는 경우 / 없는 경우 각각 지표 검증.
# 의존관계: pytest, src.evaluation.metrics
# 실행: pytest -q   (프로젝트 루트에서)
# ---------------------------------------------------------------------------

from math import log2

from src.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k

# gold 2개(A, B), graded relevance A=2, B=1 (q002/q003 처럼 gold 다수 케이스)
GOLD_SET = {"A", "B"}
GRADES = {"A": 2.0, "B": 1.0}

# 케이스 1: gold 2개가 모두 top-2 (1위 A, 2위 B)
RANKED_HIT = ["A", "B", "C", "D"]
# 케이스 2: gold 가 top-2 밖 (3위 A, 4위 B)
RANKED_MISS = ["C", "D", "A", "B"]


def test_hit_at_k_in_top2():
    assert hit_at_k(RANKED_HIT, GOLD_SET, 1) == 1.0
    assert hit_at_k(RANKED_HIT, GOLD_SET, 2) == 1.0


def test_hit_at_k_not_in_top2():
    assert hit_at_k(RANKED_MISS, GOLD_SET, 1) == 0.0
    assert hit_at_k(RANKED_MISS, GOLD_SET, 2) == 0.0
    assert hit_at_k(RANKED_MISS, GOLD_SET, 3) == 1.0  # 3위에서 처음 등장


def test_recall_at_k_in_top2():
    assert recall_at_k(RANKED_HIT, GOLD_SET, 1) == 0.5  # 2개 중 1개
    assert recall_at_k(RANKED_HIT, GOLD_SET, 2) == 1.0  # 2개 중 2개


def test_recall_at_k_not_in_top2():
    assert recall_at_k(RANKED_MISS, GOLD_SET, 2) == 0.0
    assert recall_at_k(RANKED_MISS, GOLD_SET, 4) == 1.0


def test_mrr():
    assert mrr(RANKED_HIT, GOLD_SET) == 1.0  # 1위가 gold
    assert mrr(RANKED_MISS, GOLD_SET) == 1.0 / 3  # 첫 gold 가 3위


def test_ndcg_at_k_in_top2_is_ideal():
    # 순서 A(2), B(1) 는 이상적 순서와 동일 → nDCG=1.0
    assert ndcg_at_k(RANKED_HIT, GRADES, 2) == 1.0


def test_ndcg_at_k_not_in_top2_is_zero():
    # top-2 에 관련 문서 없음 → DCG=0 → nDCG=0
    assert ndcg_at_k(RANKED_MISS, GRADES, 2) == 0.0


def test_ndcg_at_k_partial_order():
    # 1위 B(1), 2위 A(2): DCG = 1/log2(2) + 2/log2(3)
    ranked = ["B", "A", "C", "D"]
    dcg = 1.0 / log2(2) + 2.0 / log2(3)
    idcg = 2.0 / log2(2) + 1.0 / log2(3)
    assert ndcg_at_k(ranked, GRADES, 2) == dcg / idcg


def test_empty_gold_returns_zero():
    assert hit_at_k(RANKED_HIT, set(), 2) == 0.0
    assert recall_at_k(RANKED_HIT, set(), 2) == 0.0
    assert mrr(RANKED_HIT, set()) == 0.0
    assert ndcg_at_k(RANKED_HIT, {}, 2) == 0.0
