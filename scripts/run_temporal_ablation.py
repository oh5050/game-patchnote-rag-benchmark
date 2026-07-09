# ---------------------------------------------------------------------------
# 역할: [엔트리포인트] fact 유형의 낮은 검색 성능(recall@2≈0.60) 진단과, 그 처방(문서
#       시점 표기)이 rank-1(nDCG@1)을 교정하지 못한 이유에 대한 추정을 검증하는 실험.
#
#   진단 1: "pre/post 문서 표면이 거의 같아 구버전(pre_patch)이 상위로 올라온다."
#   처방 1(가설): 각 문서에 시점 표기([패치 이전/이후 기준])를 텍스트로 노출하면
#                 구분이 생겨 fact 의 recall@2·nDCG@1 이 개선될 것이다.
#   → 결과(이전 실행): recall@2 는 개선(0.60→0.80)됐지만 nDCG@1(rank-1)은 불변(0.50).
#
#   추정 2: "fact 질문에 시점 단서가 없어, 쿼리가 문서의 시점 마커를 당기지 못한다."
#   처방 2(가설): 문서 표기는 그대로 두고 **검색 쿼리에만** 현재 시점 단서를 덧붙이면
#                 rank-1(nDCG@1)이 교정될 것이다.
#
# 세 조건(모두 임베딩 모델·k·seed·device 고정 — 문서/쿼리 표현만 바뀐다):
#   - baseline                    : corpus 그대로, 질문 그대로.
#   - temporal_marked              : corpus 에 시점 표기 삽입(문서 side), 질문은 그대로.
#   - temporal_marked_query_hint   : corpus 는 temporal_marked 와 동일(문서 side 고정),
#                                    **질문에만** "[현재 기준] " 접두 힌트를 추가(쿼리 side).
#     * temporal_marked → temporal_marked_query_hint 로 갈 때 바뀌는 변인은 "쿼리 텍스트"
#       하나뿐이다(문서는 동일 인덱스를 재사용). 이래야 "쿼리 힌트 자체의 효과"를
#       "문서 표기 효과"와 분리해 볼 수 있다.
#   * 문항 원문(data/questions/questions.json)은 수정하지 않는다. 쿼리 힌트는 검색 단계
#     에서만 임시로 붙이는 파생 텍스트이며 results/temporal/query_hint_questions.json 에
#     투명하게 기록한다. 원본 corpus(data/corpus/docs.json)도 수정하지 않는다.
#
# 목적(중요): 앞선 ablation 에서 retrieval_failure=0 이 확인됐다. 즉 fact 의 낮은
#   recall@2 는 답변 오류로 이어지지 않았다. 이 실험들은 **성능 개선이 목적이 아니라
#   진단/추정의 검증**이다. 개선되면 가설이 옳고, 개선되지 않으면 다른 원인이 있다는
#   뜻이므로 그 사실을 그대로 기록한다(README 에도 두 결과 모두 남긴다).
#
# 출력:
#   - results/temporal/{corpus}_corpus.json              : 파생 코퍼스(baseline/temporal_marked, 원본 불변)
#   - results/temporal/query_hint_questions.json         : 쿼리 힌트 적용 전/후 텍스트(투명성)
#   - results/temporal/{condition}_retrieval.json         : 조건별(3개) 검색 결과
#   - results/temporal/{condition}_search_eval.json       : 조건별(3개) 검색 평가
#   - results/temporal_ablation.json                      : 유형별 3조건 비교 + 두 가설 판정
#   - stdout: 유형×조건 recall@2·nDCG@1 비교표(분자/분모 병기) + 두 진단 검증 결론
#
# 실행: python -m scripts.run_temporal_ablation   (별도 build_index 불필요; 자체 색인)
# ---------------------------------------------------------------------------

import json
import os
import sys

try:  # 콘솔 인코딩(cp949 등)에서 비ASCII 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import CORPUS_PATH, QRELS_PATH, QUESTIONS_PATH, RESULTS_DIR, load_config
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.search_eval import evaluate_search, normalize_qrels
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore

OUT_DIR = os.path.join(RESULTS_DIR, "temporal")
SUMMARY_OUT = os.path.join(RESULTS_DIR, "temporal_ablation.json")

# period 메타데이터 → 텍스트 시점 표기(문서 side)
PERIOD_MARKER = {
    "post_patch": "[패치 이후 기준] ",
    "pre_patch": "[패치 이전 기준] ",
}
# 검색 쿼리 side 시점 힌트. 문서 마커와 동일한 표기 스타일을 써서 어휘 중첩을 노린다.
QUERY_HINT = "[현재 기준] "

ANSWER_TYPES = ["fact", "multi_hop", "outdated_trap"]
EVAL_KS = [1, 2, 3]

# 조건 → (사용할 코퍼스 키, 쿼리 힌트 적용 여부)
CONDITIONS = ["baseline", "temporal_marked", "temporal_marked_query_hint"]
CORPUS_KEY_FOR_COND = {
    "baseline": "baseline",
    "temporal_marked": "temporal_marked",
    "temporal_marked_query_hint": "temporal_marked",  # 문서는 temporal_marked 와 동일 재사용
}
HINT_QUERY_FOR_COND = {
    "baseline": False,
    "temporal_marked": False,
    "temporal_marked_query_hint": True,
}


def load_docs(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def derive_corpus(docs: list[dict], corpus_key: str) -> list[dict]:
    """원본을 변형하지 않고 새 리스트를 만든다.
    baseline: content 그대로 / temporal_marked: content 앞에 시점 표기 삽입(문서 side)."""
    out = []
    for d in docs:
        nd = dict(d)  # 얕은 복사(원본 dict 불변)
        if corpus_key == "temporal_marked":
            marker = PERIOD_MARKER.get(d.get("period", ""), "")
            nd["content"] = f"{marker}{d['content']}"
        out.append(nd)
    return out


def hint_questions(questions: list[dict]) -> list[dict]:
    """원본 questions.json 은 건드리지 않고, 검색용 질문 텍스트에만 시점 힌트를 붙인
    새 리스트를 만든다(쿼리 side 변형)."""
    out = []
    for q in questions:
        nq = dict(q)
        nq["question"] = f"{QUERY_HINT}{q['question']}"
        out.append(nq)
    return out


def build_store(embedder: Embedder, docs: list[dict]) -> VectorStore:
    """문서 표현으로 인메모리 색인을 만든다(build_index 상당)."""
    doc_ids = [d["doc_id"] for d in docs]
    contents = [d["content"] for d in docs]
    passage_vecs = embedder.encode(contents, text_type="passage")
    store = VectorStore()
    store.build(doc_ids, passage_vecs)
    return store


def retrieve(embedder: Embedder, store: VectorStore, questions: list[dict], k: int) -> list[dict]:
    """주어진 질문 텍스트(원본 또는 힌트 적용본)로 top-k 검색."""
    q_texts = [q["question"] for q in questions]
    q_vecs = embedder.encode(q_texts, text_type="query")

    retrieval = []
    for i, q in enumerate(questions):
        results = store.search(q_vecs[i : i + 1], k)
        retrieval.append(
            {
                "qid": q["qid"],
                "question": q["question"],
                "retrieved_doc_ids": [doc_id for doc_id, _ in results],
                "scores": [score for _, score in results],
            }
        )
    return retrieval


def type_aggregate(retrieval: list[dict], qrels_graded: dict, qids: set[str]) -> dict:
    """유형별 집계.
    - recall@2 : 미시(micro) = (top-2 내 회수된 gold 수)/(전체 gold 수). 분자/분모가 자연스럽다.
    - nDCG@1   : (문항별 nDCG@1 합)/(문항 수). 분자는 실수, 분모는 문항 수.
    """
    found2, gold_total, ndcg1_sum, n = 0, 0, 0.0, 0
    for row in retrieval:
        if row["qid"] not in qids:
            continue
        ranked = row["retrieved_doc_ids"]
        grades = qrels_graded.get(row["qid"], {})
        gold_set = {d for d, r in grades.items() if r > 0}
        found2 += sum(1 for d in gold_set if d in ranked[:2])
        gold_total += len(gold_set)
        ndcg1_sum += ndcg_at_k(ranked, grades, 1)
        n += 1
    recall2 = found2 / gold_total if gold_total else 0.0
    ndcg1 = ndcg1_sum / n if n else 0.0
    return {
        "recall@2": {"num": found2, "den": gold_total, "value": recall2},
        "ndcg@1": {"num": round(ndcg1_sum, 4), "den": n, "value": ndcg1},
    }


def _fmt_recall(cell: dict) -> str:
    return f"{cell['num']}/{cell['den']} ({cell['value']:.2f})"


def _fmt_ndcg(cell: dict) -> str:
    return f"{cell['num']:.2f}/{cell['den']} ({cell['value']:.2f})"


def _dir(x: float) -> str:
    return f"개선({x:+.2f})" if x > 0 else (f"악화({x:+.2f})" if x < 0 else "변화 없음(0.00)")


def main() -> None:
    cfg = load_config()
    os.makedirs(OUT_DIR, exist_ok=True)

    docs = load_docs(CORPUS_PATH)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(QRELS_PATH, encoding="utf-8") as f:
        qrels_graded = normalize_qrels(json.load(f))

    qids_by_type = {at: {q["qid"] for q in questions if q["answer_type"] == at} for at in ANSWER_TYPES}
    all_qids = {q["qid"] for q in questions}

    hinted_questions = hint_questions(questions)
    with open(os.path.join(OUT_DIR, "query_hint_questions.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {"qid": q["qid"], "original": q["question"], "hinted": hq["question"]}
                for q, hq in zip(questions, hinted_questions)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    questions_for_cond = {
        cond: (hinted_questions if HINT_QUERY_FOR_COND[cond] else questions) for cond in CONDITIONS
    }

    embedder = Embedder(cfg.embedding_model, device=cfg.device, seed=cfg.seed)
    ks = sorted(set(EVAL_KS + [cfg.k]))

    # ---- 코퍼스(문서 side)는 baseline/temporal_marked 두 종류만 만들고 색인도 두 번만 구축 ----
    stores: dict[str, VectorStore] = {}
    for corpus_key in ("baseline", "temporal_marked"):
        derived = derive_corpus(docs, corpus_key)
        with open(os.path.join(OUT_DIR, f"{corpus_key}_corpus.json"), "w", encoding="utf-8") as f:
            json.dump(derived, f, ensure_ascii=False, indent=2)
        stores[corpus_key] = build_store(embedder, derived)
        print(f"[temporal_ablation] corpus={corpus_key}: 색인 완료 (docs={len(derived)})")

    # ---- 세 조건 각각 검색·평가 (temporal_marked_query_hint 는 temporal_marked 색인 재사용) ----
    conditions: dict[str, dict] = {}
    per_condition_agg: dict[str, dict] = {}
    for cond in CONDITIONS:
        store = stores[CORPUS_KEY_FOR_COND[cond]]
        qs = questions_for_cond[cond]

        retrieval = retrieve(embedder, store, qs, cfg.k)
        with open(os.path.join(OUT_DIR, f"{cond}_retrieval.json"), "w", encoding="utf-8") as f:
            json.dump(retrieval, f, ensure_ascii=False, indent=2)

        search_report = evaluate_search(retrieval, qrels_graded, ks)
        search_report["config"] = {
            "embedding_model": cfg.embedding_model,
            "k": cfg.k,
            "condition": cond,
            "corpus": CORPUS_KEY_FOR_COND[cond],
            "query_hint_applied": HINT_QUERY_FOR_COND[cond],
        }
        with open(os.path.join(OUT_DIR, f"{cond}_search_eval.json"), "w", encoding="utf-8") as f:
            json.dump(search_report, f, ensure_ascii=False, indent=2)

        agg = {t: type_aggregate(retrieval, qrels_graded, qids_by_type[t]) for t in ANSWER_TYPES}
        agg["ALL"] = type_aggregate(retrieval, qrels_graded, all_qids)
        per_condition_agg[cond] = agg
        conditions[cond] = {"search_eval_macro": search_report["macro"]}
        print(f"[temporal_ablation] condition={cond}: 검색/평가 완료")

    # ---- 유형별 3조건 비교 ----
    rows = ANSWER_TYPES + ["ALL"]
    comparison = {}
    for t in rows:
        b = per_condition_agg["baseline"][t]
        m = per_condition_agg["temporal_marked"][t]
        h = per_condition_agg["temporal_marked_query_hint"][t]
        comparison[t] = {
            "recall@2": {
                "baseline": b["recall@2"],
                "temporal_marked": m["recall@2"],
                "temporal_marked_query_hint": h["recall@2"],
                "delta_marked_vs_baseline": round(m["recall@2"]["value"] - b["recall@2"]["value"], 4),
                "delta_hint_vs_marked": round(h["recall@2"]["value"] - m["recall@2"]["value"], 4),
            },
            "ndcg@1": {
                "baseline": b["ndcg@1"],
                "temporal_marked": m["ndcg@1"],
                "temporal_marked_query_hint": h["ndcg@1"],
                "delta_marked_vs_baseline": round(m["ndcg@1"]["value"] - b["ndcg@1"]["value"], 4),
                "delta_hint_vs_marked": round(h["ndcg@1"]["value"] - m["ndcg@1"]["value"], 4),
            },
        }

    # ---- 진단 1 검증(문서 시점 표기, baseline→temporal_marked) ----
    fact_r1 = comparison["fact"]["recall@2"]["delta_marked_vs_baseline"]
    fact_n1 = comparison["fact"]["ndcg@1"]["delta_marked_vs_baseline"]

    if fact_r1 > 0 and fact_n1 > 0:
        headline1 = "진단 지지(강)"
        detail1 = (
            "시점 표기로 fact 의 recall@2 와 nDCG@1 이 모두 개선 → 구버전이 상위를 차지하던 문제가 "
            "top-2 회수와 rank-1 모두에서 교정됨."
        )
    elif fact_r1 > 0 and fact_n1 == 0:
        headline1 = "진단 부분 지지"
        detail1 = (
            "시점 표기로 fact 의 recall@2 는 개선됐으나 nDCG@1(rank-1)은 그대로 → 두 번째 gold 문서는 "
            "top-2 로 더 끌려왔지만, 최상위(rank-1)에는 여전히 구버전(pre_patch)이 앉아 있다."
        )
    elif fact_r1 == 0 and fact_n1 == 0:
        headline1 = "진단 미검증"
        detail1 = "시점 표기를 넣어도 fact 의 recall@2/nDCG@1 에 변화가 없음."
    else:
        headline1 = "진단 반증 방향"
        detail1 = "시점 표기가 오히려 fact 지표를 낮춤."

    verdict1 = f"[{headline1}] fact recall@2 {_dir(fact_r1)}, fact nDCG@1 {_dir(fact_n1)}. {detail1}"

    # ---- 추정 2 검증(쿼리 시점 힌트, temporal_marked→temporal_marked_query_hint) ----
    # 가설: "fact 질문에 시점 단서가 없어 쿼리가 문서의 시점 마커를 당기지 못한다."
    # 검증: 문서 표기는 고정하고 쿼리에만 힌트를 추가했을 때 fact nDCG@1(rank-1)이 개선되는가.
    fact_r2 = comparison["fact"]["recall@2"]["delta_hint_vs_marked"]
    fact_n2 = comparison["fact"]["ndcg@1"]["delta_hint_vs_marked"]

    if fact_n2 > 0:
        headline2 = "추정 지지"
        detail2 = (
            "문서에는 이미 시점 표기가 있는 상태에서, 쿼리에만 시점 힌트를 추가하자 fact 의 "
            "nDCG@1(rank-1)이 개선됐다 → '쿼리에 시점 단서가 없어 문서의 마커를 당기지 못한다'는 "
            "추정이 지지된다. 즉 **문서와 쿼리 양쪽에 시점 신호가 있어야 rank-1이 교정된다.**"
        )
    elif fact_n2 == 0:
        headline2 = "추정 반증"
        detail2 = (
            "문서 표기는 고정한 채 쿼리에만 시점 힌트를 추가했지만 fact 의 nDCG@1(rank-1)에 "
            "변화가 없었다 → '쿼리에 시점 단서가 없어서'라는 추정은 **틀렸다**. rank-1 교정 실패의 "
            "원인은 쿼리 측 시점 단서 부재가 아니라 다른 요인(예: 임베딩이 '이름/스킬명' 질의에서 "
            "시점 토큰보다 개체명 매칭에 더 의존하는 경향, 혹은 시점 힌트가 짧은 질문 안에서 "
            "상대적으로 약한 신호가 되는 점 등)일 가능성이 높다."
        )
    else:
        headline2 = "추정 반증(역효과)"
        detail2 = (
            "쿼리에 시점 힌트를 추가하자 fact 의 nDCG@1(rank-1)이 오히려 악화됐다 → 추정은 틀렸고, "
            "쿼리 힌트가 오히려 검색 신호를 흐릴 수 있음을 시사한다."
        )

    verdict2 = f"[{headline2}] fact recall@2(hint vs marked) {_dir(fact_r2)}, fact nDCG@1(hint vs marked) {_dir(fact_n2)}. {detail2}"

    # ---- 다른 유형에 대한 부작용(회귀) 기록 ----
    def _regressions(delta_key: str) -> list[str]:
        regs = []
        for t in rows:
            for metric in ("recall@2", "ndcg@1"):
                d = comparison[t][metric][delta_key]
                if t != "fact" and d < 0:
                    regs.append(f"{t} {metric} {d:+.2f}")
        return regs

    regressions1 = _regressions("delta_marked_vs_baseline")
    regressions2 = _regressions("delta_hint_vs_marked")
    if regressions1:
        verdict1 += " | 부작용(회귀, baseline→temporal_marked): " + ", ".join(regressions1) + "."
    if regressions2:
        verdict2 += " | 부작용(회귀, temporal_marked→query_hint): " + ", ".join(regressions2) + "."

    summary = {
        "config": {
            "embedding_model": cfg.embedding_model,
            "k": cfg.k,
            "seed": cfg.seed,
            "device": cfg.device,
            "eval_ks": ks,
        },
        "conditions": CONDITIONS,
        "period_marker": PERIOD_MARKER,
        "query_hint": QUERY_HINT,
        "comparison": comparison,
        "macro": {c: conditions[c]["search_eval_macro"] for c in conditions},
        "diagnosis_1_doc_marker": {
            "headline": headline1,
            "fact_recall@2_delta": fact_r1,
            "fact_ndcg@1_delta": fact_n1,
            "regressions": regressions1,
            "text": verdict1,
        },
        "estimate_2_query_hint": {
            "headline": headline2,
            "fact_recall@2_delta": fact_r2,
            "fact_ndcg@1_delta": fact_n2,
            "regressions": regressions2,
            "text": verdict2,
        },
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _print_report(comparison, conditions, verdict1, verdict2, rows)
    print(f"\n→ {SUMMARY_OUT}")
    print(f"→ 파생 코퍼스/검색/평가/힌트 질문: {OUT_DIR}")


def _print_report(comparison, conditions, verdict1, verdict2, rows) -> None:
    print("\n" + "=" * 96)
    print("TEMPORAL ABLATION: 문서 시점 표기 + 쿼리 시점 힌트가 검색을 개선하는가 (진단/추정 검증)")
    print("=" * 96)

    print("\n[recall@2] 유형별, 3조건 (분자/분모 = top-2 내 회수 gold / 전체 gold)")
    print("-" * 96)
    print(f"  {'type':<14}{'baseline':>20}{'temporal_marked':>20}{'+query_hint':>20}")
    for t in rows:
        c = comparison[t]["recall@2"]
        print(
            f"  {t:<14}{_fmt_recall(c['baseline']):>20}{_fmt_recall(c['temporal_marked']):>20}"
            f"{_fmt_recall(c['temporal_marked_query_hint']):>20}"
        )

    print("\n[nDCG@1] 유형별, 3조건 (분자/분모 = 문항별 nDCG@1 합 / 문항 수)")
    print("-" * 96)
    print(f"  {'type':<14}{'baseline':>20}{'temporal_marked':>20}{'+query_hint':>20}")
    for t in rows:
        c = comparison[t]["ndcg@1"]
        print(
            f"  {t:<14}{_fmt_ndcg(c['baseline']):>20}{_fmt_ndcg(c['temporal_marked']):>20}"
            f"{_fmt_ndcg(c['temporal_marked_query_hint']):>20}"
        )

    print("\n[MACRO 전체] 3조건 검색 지표 비교(참고)")
    print("-" * 96)
    keys = ["recall@2", "ndcg@1", "recall@1", "mrr"]
    print(f"  {'metric':<12}{'baseline':>14}{'temporal_marked':>18}{'+query_hint':>18}")
    for key in keys:
        vals = [conditions[c]["search_eval_macro"].get(key) for c in
                ("baseline", "temporal_marked", "temporal_marked_query_hint")]
        if any(v is None for v in vals):
            continue
        print(f"  {key:<12}{vals[0]:>14.3f}{vals[1]:>18.3f}{vals[2]:>18.3f}")

    print("\n[결론 1] 문서 시점 표기 진단 검증 (baseline → temporal_marked)")
    print("-" * 96)
    print(f"  {verdict1}")

    print("\n[결론 2] 쿼리 시점 힌트 추정 검증 (temporal_marked → temporal_marked_query_hint)")
    print("-" * 96)
    print(f"  {verdict2}")


if __name__ == "__main__":
    main()
