# ---------------------------------------------------------------------------
# 역할: [엔트리포인트] "검색 오류 vs 답변 오류"의 인과를 분리하는 ablation 실험.
#       동일한 생성 모델(config.gen_model, seed 고정)로 15문항 각각을 세 조건에서
#       답변 생성하고, 기존 answer_eval 파이프라인으로 동일하게 채점한 뒤 문항별
#       진단 라벨을 자동 부여한다.
#
# 세 조건:
#   - no_context        : 문서 없이 질문만 (closed-book). groundedness 프롬프트를 쓰면
#                         "모른다"만 나오므로, 여기서만 모델이 아는 지식으로 답하게 하는
#                         closed-book 프롬프트를 사용한다. → 모델의 파라메트릭 지식(구버전
#                         편향의 기저선) 측정.
#   - gold_context      : qrels 의 정답 문서를 강제 주입 (검색이 완벽할 때의 상한).
#   - retrieved_context : 실제 top-k 검색 결과 주입 (현재 시스템 성능).
#     * gold/retrieved 는 실제 시스템과 동일하게 기존 Generator 를 그대로 사용한다.
#
# 진단 라벨(문항별, 우선순위 캐스케이드):
#   1) gold 에서도 틀림              → generation_failure       (생성 문제)
#   2) gold 맞고 retrieved 에서 틀림 → retrieval_failure        (검색 문제)
#   3) no_context 에서 이미 구버전을 답함(outdated_trap 한정)
#                                    → parametric_outdated_bias (모델이 원래 구버전을 알고 있었음)
#   4) 세 조건 모두 맞음              → ok
#
# 출력:
#   - results/ablation.json (config, per_query 3조건 답변/채점/진단, 조건×유형 교차표,
#     진단 라벨 분포)
#   - stdout: outdated_trap no_context 별도 표, 조건×유형 교차표(분자/분모 병기), 진단 분포
# ---------------------------------------------------------------------------

import json
import os
import re
import sys

try:  # 콘솔 인코딩(cp949 등)에서 비ASCII 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import (
    CORPUS_PATH,
    QRELS_PATH,
    QUESTIONS_PATH,
    RESULTS_DIR,
    load_config,
)
from src.evaluation.answer_eval import evaluate_answers
from src.evaluation.judge import LLMJudge
from src.generation.generator import Generator
from src.indexing.embedder import Embedder, load_corpus
from src.indexing.vector_store import VectorStore
from src.llm.ollama_client import OllamaClient
from src.retrieval.retriever import Retriever

INDEX_PREFIX = os.path.join(RESULTS_DIR, "index", "corpus")
ABLATION_OUT = os.path.join(RESULTS_DIR, "ablation.json")

CONDITIONS = ["no_context", "gold_context", "retrieved_context"]
ANSWER_TYPES = ["fact", "multi_hop", "outdated_trap"]
LABELS = ["ok", "generation_failure", "retrieval_failure", "parametric_outdated_bias"]

CLOSED_BOOK_SYSTEM = (
    "너는 게임 도메인 질문에 답하는 어시스턴트다. 참고 문서는 주어지지 않는다. "
    "네가 알고 있는 지식만으로 질문에 간결하고 정확하게 한국어로 답하라. "
    "질문에 구체적인 수치가 언급되면, 아는 범위에서 실제 수치를 답하라."
)

# 수치+단위 토큰(예: '100초', '40%').
_VAL_RE = re.compile(r"\d+\s*(?:초|%|퍼센트)")


def _norm_val(s: str) -> str:
    return s.replace(" ", "").replace("퍼센트", "%")


# ============================================================================
# 생성 단계
# ============================================================================
def closed_book_answer(client, gen_model: str, seed: int, question: str) -> str:
    if not client.available():
        return f"[DUMMY] Ollama 미연결 → closed-book 폴백. 질문='{question}'"
    messages = [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": f"### 질문\n{question}"},
    ]
    try:
        return client.chat(gen_model, messages, seed=seed, temperature=0.0)
    except Exception as exc:
        return f"[DUMMY] closed-book 생성 실패({exc}). 질문='{question}'"


def gold_doc_ids(qrels_for_q: dict) -> list[str]:
    return [doc_id for doc_id, _ in sorted(qrels_for_q.items(), key=lambda kv: -kv[1])]


def generate_all(cfg, questions, qrels, corpus_by_id) -> dict:
    """세 조건에서 답변 생성. per_query_gen[qid] = {question, answers{cond}, doc_ids{cond}}."""
    embedder = Embedder(cfg.embedding_model, device=cfg.device, seed=cfg.seed)
    store = VectorStore().load(INDEX_PREFIX)
    retriever = Retriever(embedder, store, cfg.k)

    client = OllamaClient()
    if not client.available():
        print("[run_ablation] 경고: Ollama 미연결 → 폴백 답변으로 진행합니다.")
    generator = Generator(client, cfg.gen_model, cfg.seed)

    per_query_gen: dict[str, dict] = {}
    for item in questions:
        qid, question = item["qid"], item["question"]
        retrieved_ids = [d for d, _ in retriever.retrieve(question)]
        retrieved_ctx = [{"doc_id": d, "content": corpus_by_id.get(d, "")} for d in retrieved_ids]
        gids = gold_doc_ids(qrels.get(qid, {}))
        gold_ctx = [{"doc_id": d, "content": corpus_by_id.get(d, "")} for d in gids]

        per_query_gen[qid] = {
            "question": question,
            "answers": {
                "no_context": closed_book_answer(client, cfg.gen_model, cfg.seed, question),
                "gold_context": generator.generate(question, gold_ctx),
                "retrieved_context": generator.generate(question, retrieved_ctx),
            },
            "doc_ids": {"no_context": [], "gold_context": gids, "retrieved_context": retrieved_ids},
        }
        print(f"[gen] {qid} 완료 (no_context/gold/retrieved)")
    return per_query_gen, client


def score_all(per_query_gen, questions_by_qid, corpus_by_id, mode, k, llm) -> dict:
    """조건별로 기존 evaluate_answers 로 동일 채점."""
    eval_by_condition = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        gen_rows = [
            {
                "qid": qid,
                "question": g["question"],
                "retrieved_doc_ids": g["doc_ids"][cond],
                "answer": g["answers"][cond],
            }
            for qid, g in per_query_gen.items()
        ]
        for r in evaluate_answers(gen_rows, questions_by_qid, corpus_by_id, mode, k, llm):
            eval_by_condition[cond][r["qid"]] = r
    return eval_by_condition


# ============================================================================
# 분석 단계
# ============================================================================
def extract_outdated_value(question: str) -> str | None:
    m = _VAL_RE.search(question)
    return _norm_val(m.group(0)) if m else None


def post_value_keyword(keywords: list[str]) -> str | None:
    for kw in keywords:
        if any(ch.isdigit() for ch in kw):
            return _norm_val(kw)
    return None


def is_correct(eval_result: dict, answer_type: str) -> bool:
    if eval_result.get("correctness", {}).get("verdict") != "yes":
        return False
    if answer_type == "outdated_trap" and eval_result.get("hallucination", {}).get("verdict") == "yes":
        return False
    return True


def answered_outdated(answer: str, outdated_value: str | None, post_value: str | None) -> bool:
    """no_context 답변이 구버전 값을 그대로 답했는지(부분문자열 매칭)."""
    if not outdated_value:
        return False
    ans = answer.replace(" ", "")
    if outdated_value not in ans:
        return False
    if post_value and post_value.replace(" ", "") in ans:
        return False
    return True


def diagnose(flags: dict, answer_type: str) -> str:
    if not flags["gold_correct"]:
        return "generation_failure"
    if not flags["retrieved_correct"]:
        return "retrieval_failure"
    if answer_type == "outdated_trap" and flags.get("no_context_answered_outdated"):
        return "parametric_outdated_bias"
    return "ok"


def analyze(per_query_gen, eval_by_condition, questions) -> dict:
    per_query_out = []
    for item in questions:
        qid, atype = item["qid"], item["answer_type"]
        keywords = item.get("gold_answer_keywords", [])
        is_trap = atype == "outdated_trap"
        outdated_value = extract_outdated_value(item["question"]) if is_trap else None
        post_value = post_value_keyword(keywords) if is_trap else None

        cond_eval = {c: eval_by_condition[c][qid] for c in CONDITIONS}
        correct = {c: is_correct(cond_eval[c], atype) for c in CONDITIONS}

        nc_answer = per_query_gen[qid]["answers"]["no_context"]

        flags = {
            "no_context_correct": correct["no_context"],
            "gold_correct": correct["gold_context"],
            "retrieved_correct": correct["retrieved_context"],
            "all_three_correct": all(correct.values()),
            "no_context_answered_outdated": answered_outdated(nc_answer, outdated_value, post_value),
        }

        per_query_out.append(
            {
                "qid": qid,
                "answer_type": atype,
                "question": item["question"],
                "outdated_value": outdated_value,
                "post_value": post_value,
                "diagnosis": diagnose(flags, atype),
                "flags": flags,
                "conditions": {
                    c: {
                        "doc_ids": per_query_gen[qid]["doc_ids"][c],
                        "answer": per_query_gen[qid]["answers"][c],
                        "eval": cond_eval[c],
                        "correct": correct[c],
                    }
                    for c in CONDITIONS
                },
            }
        )

    # 조건×유형 교차표 (셀: 정답/전체)
    cross_table = {}
    for cond in CONDITIONS:
        cross_table[cond] = {}
        for at in ANSWER_TYPES + ["ALL"]:
            rows = [e for e in per_query_out if at == "ALL" or e["answer_type"] == at]
            num = sum(1 for e in rows if e["conditions"][cond]["correct"])
            cross_table[cond][at] = {"correct": num, "total": len(rows)}

    dist = {lbl: 0 for lbl in LABELS}
    for e in per_query_out:
        dist[e["diagnosis"]] = dist.get(e["diagnosis"], 0) + 1

    return {
        "per_query": per_query_out,
        "cross_table": cross_table,
        "diagnosis_distribution": dist,
    }


# ============================================================================
# 리포트
# ============================================================================
def print_report(analysis: dict) -> None:
    pq = analysis["per_query"]
    ct = analysis["cross_table"]

    print("\n" + "=" * 82)
    print("ABLATION 결과: 검색 오류 vs 답변 오류의 인과 분리")
    print("=" * 82)

    # (A) outdated_trap no_context 별도 표
    print("\n[A] outdated_trap no_context(closed-book) 답변 — 파라메트릭 지식 기저선")
    print("-" * 82)
    for e in [x for x in pq if x["answer_type"] == "outdated_trap"]:
        old = "O" if e["flags"]["no_context_answered_outdated"] else "X"
        ans = e["conditions"]["no_context"]["answer"]
        print(f"  {e['qid']} | 구버전값 답변={old} | no_context 답변: {ans[:80]}")

    # (B) 조건×유형 교차표
    print("\n[B] 조건 × 유형 교차표 (정답수/전체, judge 기준)")
    print("-" * 82)
    print(f"  {'condition':<20}" + "".join(f"{at:>16}" for at in ANSWER_TYPES) + f"{'ALL':>12}")
    for cond in CONDITIONS:
        line = f"  {cond:<20}"
        for at in ANSWER_TYPES:
            cell = ct[cond][at]
            line += f"{cell['correct']:>7}/{cell['total']:<8}"
        allc = ct[cond]["ALL"]
        line += f"{allc['correct']:>5}/{allc['total']:<6}"
        print(line)

    # (C) 진단 라벨 분포
    print("\n[C] 진단 라벨 분포 (총 15문항)")
    print("-" * 82)
    dist = analysis["diagnosis_distribution"]
    for lbl in LABELS:
        qids = [e["qid"] for e in pq if e["diagnosis"] == lbl]
        print(f"  {lbl:<28}{dist.get(lbl, 0):>3}   {qids}")


# ============================================================================
def main() -> None:
    cfg = load_config()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(QRELS_PATH, encoding="utf-8") as f:
        qrels = json.load(f)
    questions_by_qid = {q["qid"]: q for q in questions}
    doc_ids_all, contents = load_corpus(CORPUS_PATH)
    corpus_by_id = dict(zip(doc_ids_all, contents))

    per_query_gen, client = generate_all(cfg, questions, qrels, corpus_by_id)

    mode = cfg.judge_mode
    llm = None
    if mode != "rule":
        if client.available():
            llm = LLMJudge(client, cfg.judge_model, cfg.seed)
        else:
            print("[run_ablation] 경고: Ollama 미연결 → rule 모드로 강등하여 채점합니다.")
            mode = "rule"
    eval_by_condition = score_all(per_query_gen, questions_by_qid, corpus_by_id, mode, cfg.k, llm)

    analysis = analyze(per_query_gen, eval_by_condition, questions)

    report = {
        "config": {
            "gen_model": cfg.gen_model,
            "judge_model": cfg.judge_model,
            "judge_mode": mode,
            "seed": cfg.seed,
            "k": cfg.k,
            "embedding_model": cfg.embedding_model,
        },
        "conditions": CONDITIONS,
        "label_definitions": {
            "ok": "gold·retrieved 모두 정답(시스템 정상).",
            "generation_failure": "gold 오답(생성 문제).",
            "retrieval_failure": "gold 정답, retrieved 오답(검색 문제).",
            "parametric_outdated_bias": (
                "outdated_trap 에서 no_context 답변이 구버전 값을 그대로 답함"
                "(모델이 원래 구버전을 알고 있었음)."
            ),
        },
        **analysis,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ABLATION_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(analysis)
    print(f"\n→ {ABLATION_OUT}")


if __name__ == "__main__":
    main()
