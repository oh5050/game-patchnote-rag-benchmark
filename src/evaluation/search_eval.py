# ---------------------------------------------------------------------------
# 역할: 검색(Search) 단계 평가 — 생성 품질과 독립적으로 retrieval 만 채점.
# 입력:
#   - results/retrieval.json      : [{qid, question, retrieved_doc_ids, scores}, ...]
#   - data/qrels/qrels.json       : {qid: {doc_id: rel}} 또는 {qid: [doc_id, ...]}
#   - data/questions/questions.json (answer_type 매핑용, 유형별 집계에만 사용)
#   - k (config): 검색 컷오프. 표시는 여러 k(기본 1,2,3)에 대해 함께 계산.
# 출력:
#   - results/search_eval.json : {config, per_query[], macro{}, aggregate{}, by_type{}}
#   - 콘솔: 질문별 표 + 전체(ALL)/유형별 집계 표(분자/분모 병기)
# 정밀도 원칙(중요): n=15 파일럿이라 소수점 표기만으로는 과대 정밀 인상을 준다.
#   그래서 macro(비율의 평균)와 별개로 aggregate/by_type 에는 항상 {num, den, value}
#   3종을 함께 저장해, 몇 건 중 몇 건인지 계산기 없이 바로 읽을 수 있게 한다.
#   - hit@k    : num=히트한 질문 수,           den=질문 수
#   - recall@k : num=회수된 gold 문서 수(micro 합), den=전체 gold 문서 수(micro 합)
#                * 이 프로젝트 qrels 는 질문마다 gold 문서 수가 항상 2개로 동일하므로
#                  recall@k 의 micro 집계(합/합)와 macro 집계(비율의 평균)는 값이 같다.
#   - nDCG@k/MRR : num=문항별 값의 합(소수), den=질문 수  (평균의 분자/분모를 그대로 노출)
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

from src.config import QRELS_PATH, QUESTIONS_PATH, RESULTS_DIR, load_config
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


def aggregate_with_counts(
    retrieval_results: list[dict],
    qrels_graded: dict[str, dict[str, float]],
    ks: list[int],
    qids: set[str] | None = None,
) -> dict[str, dict]:
    """지정한 qid 집합(None=전체)에 대해 {num, den, value} 3종을 함께 반환하는 집계.
    분자/분모를 그대로 노출해 n=15 파일럿에서의 과대 정밀 인상을 피한다."""
    rows = [r for r in retrieval_results if qids is None or r["qid"] in qids]
    n = len(rows)
    hit_num = {k: 0 for k in ks}
    recall_found = {k: 0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    gold_total = 0
    mrr_sum = 0.0

    for row in rows:
        qid = row["qid"]
        ranked = row["retrieved_doc_ids"]
        grades = qrels_graded.get(qid, {})
        gold_set = {doc_id for doc_id, r in grades.items() if r > 0}
        gold_total += len(gold_set)
        for k in ks:
            if hit_at_k(ranked, gold_set, k) > 0:
                hit_num[k] += 1
            recall_found[k] += sum(1 for d in gold_set if d in ranked[:k])
            ndcg_sum[k] += ndcg_at_k(ranked, grades, k)
        mrr_sum += mrr(ranked, gold_set)

    out: dict[str, dict] = {}
    for k in ks:
        out[f"hit@{k}"] = {"num": hit_num[k], "den": n, "value": hit_num[k] / n if n else 0.0}
        out[f"recall@{k}"] = {
            "num": recall_found[k],
            "den": gold_total,
            "value": recall_found[k] / gold_total if gold_total else 0.0,
        }
        out[f"ndcg@{k}"] = {
            "num": round(ndcg_sum[k], 2),
            "den": n,
            "value": ndcg_sum[k] / n if n else 0.0,
        }
    out["mrr"] = {"num": round(mrr_sum, 2), "den": n, "value": mrr_sum / n if n else 0.0}
    return out


def evaluate_search(
    retrieval_results: list[dict],
    qrels_graded: dict[str, dict[str, float]],
    ks: list[int] = DEFAULT_KS,
    qid_to_type: dict[str, str] | None = None,
) -> dict:
    """질문별 지표 + macro 평균을 계산해 반환.
    qid_to_type 이 주어지면 유형별(answer_type) 분자/분모 집계(by_type)도 함께 계산한다."""
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

    # macro 평균: 모든 질문에 대해 각 지표를 단순 평균(기존 하위호환용, 분자/분모 없음).
    macro: dict[str, float] = {}
    if per_query:
        metric_keys = per_query[0]["metrics"].keys()
        for key in metric_keys:
            macro[key] = sum(q["metrics"][key] for q in per_query) / len(per_query)

    report = {
        "ks": ks,
        "per_query": per_query,
        "macro": macro,
        # macro 와 동일한 값을 분자/분모와 함께 담은 전체(ALL) 집계.
        "aggregate": aggregate_with_counts(retrieval_results, qrels_graded, ks),
    }

    if qid_to_type:
        by_type: dict[str, dict] = {}
        for t in sorted(set(qid_to_type.values())):
            qids_t = {qid for qid, ty in qid_to_type.items() if ty == t}
            by_type[t] = aggregate_with_counts(retrieval_results, qrels_graded, ks, qids_t)
        report["by_type"] = by_type

    return report


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


def _fmt_cell(cell: dict) -> str:
    """예: '0.60 (6/10)', '0.80 (12.00/15)'. n=15 파일럿에서 분자/분모를 항상 함께 보여준다."""
    num = cell["num"]
    num_s = f"{num:.2f}" if isinstance(num, float) else str(num)
    return f"{cell['value']:.2f} ({num_s}/{cell['den']})"


def _format_group_table(report: dict, ks: list[int]) -> str:
    """전체(ALL) + 유형별(by_type) 집계를 분자/분모와 함께 표로 렌더링."""
    cols = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + ["mrr"]
    header = ["group".ljust(14)] + [c.rjust(18) for c in cols]
    lines = [" ".join(header), "-" * len(" ".join(header))]

    groups = {"ALL": report["aggregate"], **report.get("by_type", {})}
    for name, agg in groups.items():
        row = [name.ljust(14)] + [_fmt_cell(agg[c]).rjust(18) for c in cols]
        lines.append(" ".join(row))
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    ks = sorted(set(DEFAULT_KS + [cfg.k]))  # config.k 도 함께 포함

    with open(RETRIEVAL_PATH, encoding="utf-8") as f:
        retrieval_results = json.load(f)
    with open(QRELS_PATH, encoding="utf-8") as f:
        qrels_graded = normalize_qrels(json.load(f))
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    qid_to_type = {q["qid"]: q.get("answer_type", "unknown") for q in questions}

    report = evaluate_search(retrieval_results, qrels_graded, ks, qid_to_type)
    report["config"] = {"embedding_model": cfg.embedding_model, "k": cfg.k}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SEARCH_EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[search_eval] model={cfg.embedding_model} | queries={len(report['per_query'])}")
    print(_format_table(report))
    print("\n[전체/유형별 집계] (분자/분모 병기 — n=15 파일럿이므로 값만 보지 말고 분모를 함께 볼 것)")
    print(_format_group_table(report, ks))
    print(f"→ {SEARCH_EVAL_OUT}")


if __name__ == "__main__":
    main()
