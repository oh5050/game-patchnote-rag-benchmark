# ---------------------------------------------------------------------------
# 역할: 검색(Search) 단계 평가 — 생성 품질과 독립적으로 retrieval 만 채점.
# 입력:
#   - results/retrieval.json : [{qid, question, retrieved_doc_ids, scores}, ...]
#   - data/qrels/qrels.json  : {qid: {doc_id: rel}} 또는 {qid: [doc_id, ...]}
#   - k (config): 검색 컷오프. 표시는 여러 k(기본 1,2,3)에 대해 함께 계산.
# 출력:
#   - results/search_eval.json : {config, per_query[], macro{}}
#   - 콘솔: 질문별 + macro 평균 표
# 의존관계:
#   - src.evaluation.metrics, src.config
# 사용처: scripts/run_eval.py, src.pipeline, (직접 실행) python -m src.evaluation.search_eval
# ---------------------------------------------------------------------------

import json
import os
import sys

try:  # 콘솔 인코딩(cp949 등)에서 비ASCII 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import QRELS_PATH, RESULTS_DIR, load_config
from src.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k

RETRIEVAL_PATH = os.path.join(RESULTS_DIR, "retrieval.json")
SEARCH_EVAL_OUT = os.path.join(RESULTS_DIR, "search_eval.json")
DEFAULT_KS = [1, 2, 3]


def normalize_qrels(qrels_raw: dict) -> dict[str, dict[str, float]]:
    """qrels 를 {qid: {doc_id: relevance}} 형태(graded)로 표준화.
    리스트 형태 {qid: [doc_id, ...]} 는 relevance=1 로 간주."""
    normalized: dict[str, dict[str, float]] = {}
    for qid, rels in qrels_raw.items():
        if isinstance(rels, dict):
            normalized[qid] = {doc_id: float(r) for doc_id, r in rels.items()}
        else:  # list[doc_id]
            normalized[qid] = {doc_id: 1.0 for doc_id in rels}
    return normalized


def evaluate_search(
    retrieval_results: list[dict],
    qrels_graded: dict[str, dict[str, float]],
    ks: list[int] = DEFAULT_KS,
) -> dict:
    """질문별 지표 + macro 평균을 계산해 반환."""
    per_query = []
    for row in retrieval_results:
        qid = row["qid"]
        ranked = row["retrieved_doc_ids"]
        grades = qrels_graded.get(qid, {})
        gold_set = {doc_id for doc_id, r in grades.items() if r > 0}

        scores: dict[str, float] = {}
        for k in ks:
            scores[f"hit@{k}"] = hit_at_k(ranked, gold_set, k)
            scores[f"recall@{k}"] = recall_at_k(ranked, gold_set, k)
            scores[f"ndcg@{k}"] = ndcg_at_k(ranked, grades, k)
        scores["mrr"] = mrr(ranked, gold_set)

        per_query.append({"qid": qid, "num_gold": len(gold_set), "metrics": scores})

    # macro 평균: 모든 질문에 대해 각 지표를 단순 평균.
    macro: dict[str, float] = {}
    if per_query:
        metric_keys = per_query[0]["metrics"].keys()
        for key in metric_keys:
            macro[key] = sum(q["metrics"][key] for q in per_query) / len(per_query)

    return {"ks": ks, "per_query": per_query, "macro": macro}


def _format_table(report: dict) -> str:
    ks = report["ks"]
    cols = (
        [f"hit@{k}" for k in ks]
        + [f"recall@{k}" for k in ks]
        + [f"ndcg@{k}" for k in ks]
        + ["mrr"]
    )
    header = ["qid".ljust(8)] + [c.rjust(9) for c in cols]
    lines = [" ".join(header), "-" * len(" ".join(header))]

    for q in report["per_query"]:
        row = [q["qid"].ljust(8)] + [f"{q['metrics'][c]:.3f}".rjust(9) for c in cols]
        lines.append(" ".join(row))

    lines.append("-" * len(" ".join(header)))
    macro_row = ["MACRO".ljust(8)] + [f"{report['macro'][c]:.3f}".rjust(9) for c in cols]
    lines.append(" ".join(macro_row))
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    ks = sorted(set(DEFAULT_KS + [cfg.k]))  # config.k 도 함께 포함

    with open(RETRIEVAL_PATH, encoding="utf-8") as f:
        retrieval_results = json.load(f)
    with open(QRELS_PATH, encoding="utf-8") as f:
        qrels_graded = normalize_qrels(json.load(f))

    report = evaluate_search(retrieval_results, qrels_graded, ks)
    report["config"] = {"embedding_model": cfg.embedding_model, "k": cfg.k}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SEARCH_EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[search_eval] model={cfg.embedding_model} | queries={len(report['per_query'])}")
    print(_format_table(report))
    print(f"→ {SEARCH_EVAL_OUT}")


if __name__ == "__main__":
    main()
