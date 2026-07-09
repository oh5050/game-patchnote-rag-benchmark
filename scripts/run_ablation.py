# ---------------------------------------------------------------------------
# 역할: [엔트리포인트] "검색 오류 vs 답변 오류"의 인과를 분리하는 ablation 실험.
#       동일한 생성 모델(config.gen_model, seed 고정)로 15문항 각각을 세 조건에서
#       답변 생성하고, 기존 answer_eval 파이프라인으로 동일하게 채점한 뒤 문항별
#       진단 라벨을 자동 부여한다.
#
# 세 조건:
#   - no_context        : 문서 없이 질문만 (closed-book). groundedness 프롬프트를 쓰면
#                         "모른다"만 나오므로, 여기서만 모델이 아는 지식으로 답하게 하는
#                         closed-book 프롬프트를 사용한다.
#   - gold_context      : qrels 의 정답 문서를 강제 주입 (검색이 완벽할 때의 상한).
#   - retrieved_context : 실제 top-k 검색 결과 주입 (현재 시스템 성능).
#     * gold/retrieved 는 실제 시스템과 동일하게 기존 Generator 를 그대로 사용한다.
#
# 진단 라벨(문항별, 우선순위 캐스케이드):
#   1) gold 오답 & 정답수치(post)도 없음  → generation_failure          (진짜 생성 실패)
#   1') gold 오답이지만 정답수치는 포함    → judge_uncertain             (채점 오판; 규칙 재확인)
#   2) gold 정답 & retrieved 오답          → retrieval_failure           (검색 문제)
#   3) (outdated_trap 한정) no_context 가
#      질문에 흘린 구버전 수치를 그대로 반복 → question_leaked_outdated_value
#                                            (사전 지식이 아니라 '질문 누출')
#   4) 위에 해당 없음                        → ok
#
# ── 두 가지 오염 보정(중요) ────────────────────────────────────────────────
#   (오염 1) parametric_outdated_bias 폐기: 본 벤치마크의 캐릭터/스킬은 전부 가상
#     합성 데이터라 모델이 사전 지식으로 구버전 수치를 알 수 없다. no_context 에서
#     구버전 값이 나온 것은 outdated_trap 질문 자체가 구버전 수치를 전제로 포함하기
#     때문이다(질문 누출). → 라벨을 question_leaked_outdated_value 로 교체하고,
#     no_context 답변이 '질문의 수치를 반복'한 건지 '질문에 없는 새 수치를 지어낸(환각)'
#     건지 문항별로 구분해 표시한다.
#   (오염 2) generation_failure 오판 분리: gold_context 답변이 내용상 정답 수치(post)와
#     정정을 담았는데도 키워드 부분매칭 실패 → LLM judge 오판으로 no 가 된 케이스가 있다.
#     gold_context 답변에 정답 수치가 문자열로 포함됐는지 규칙으로 재확인해, 포함됐다면
#     judge_uncertain 으로, 정답 수치조차 없으면 진짜 generation_failure 로 남긴다.
#
# 출력:
#   - results/ablation.json (config, per_query 3조건 답변/채점/보조지표/진단, 교차표,
#     before/after 진단 분포)
#   - stdout: outdated_trap no_context 별도 표(누출 vs 환각), 조건×유형 교차표(분자/분모),
#     gold_context 채점 재확인, 진단 라벨 before/after 비교
#
# 실행:
#   python -m scripts.run_ablation               # 전체 재생성 후 분석(느림)
#   python -m scripts.run_ablation --from-cache  # 저장된 답변/채점만 재분석(LLM 미호출)
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
# 시스템 수준 1차 진단(문항당 1개, 캐스케이드). parametric_outdated_bias 폐기.
LABELS = [
    "ok",
    "generation_failure",
    "judge_uncertain",
    "retrieval_failure",
    "question_leaked_outdated_value",
]
# 비배타(복수 적용) 진단 태그 — 한 문항에 여러 개가 동시에 붙을 수 있다.
MULTI_LABELS = ["question_leaked_outdated_value", "fabricated_value", "judge_uncertain"]
# 원래(오염된) 라벨 집합 — before/after 비교용
LEGACY_LABELS = ["ok", "generation_failure", "retrieval_failure", "parametric_outdated_bias"]

CLOSED_BOOK_SYSTEM = (
    "너는 게임 도메인 질문에 답하는 어시스턴트다. 참고 문서는 주어지지 않는다. "
    "네가 알고 있는 지식만으로 질문에 간결하고 정확하게 한국어로 답하라. "
    "질문에 구체적인 수치가 언급되면, 아는 범위에서 실제 수치를 답하라."
)

# 수치+단위 토큰(예: '100초', '40%'). 비캡처 그룹이라 findall 이 전체 매치를 돌려준다.
_VAL_RE = re.compile(r"\d+\s*(?:초|%|퍼센트)")


def _norm_val(s: str) -> str:
    return s.replace(" ", "").replace("퍼센트", "%")


def value_tokens(text: str) -> set[str]:
    """텍스트에서 '숫자+단위' 토큰 집합을 추출(정규화). 부분문자열 오탐 방지를 위해
    토큰 단위로 비교한다(예: '140%' 는 '40%' 로 오인되지 않는다)."""
    return {_norm_val(m) for m in _VAL_RE.findall(text or "")}


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
# 분석 단계 (재생성 없이 저장 답변만으로도 재계산 가능)
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


def no_context_numeric_check(
    answer: str, question: str, outdated_value: str | None, corpus_vals: set[str]
) -> dict:
    """no_context 답변의 수치를 세 갈래로 가른다.
    - leaked   : 질문에 있던 수치를 반복(질문 누출).
    - fabricated: 질문에도 문서(corpus)에도 없는 수치를 생성(진짜 fabrication).
    - invented : 질문에 없는 수치(문서 포함 여부 무관) — 참고용 상위집합.
    """
    q_vals = value_tokens(question)
    a_vals = value_tokens(answer)
    leaked = sorted(a_vals & q_vals)                      # 질문에 있던 수치를 반복
    invented = sorted(a_vals - q_vals)                    # 질문에 없는 수치(참고)
    fabricated = sorted(a_vals - q_vals - corpus_vals)    # 질문·문서 어디에도 없음 = 창작
    outdated_leaked = bool(outdated_value and outdated_value in a_vals)
    is_fabricated = bool(fabricated)

    if outdated_leaked and is_fabricated:
        classification = "leaked+fabricated"
    elif outdated_leaked:
        classification = "leaked"
    elif is_fabricated:
        classification = "fabricated"
    elif a_vals:
        classification = "other_numeric"
    else:
        classification = "no_numeric"

    return {
        "question_values": sorted(q_vals),
        "answer_values": sorted(a_vals),
        "leaked_values": leaked,
        "invented_values": invented,
        "fabricated_values": fabricated,
        "outdated_value_leaked": outdated_leaked,
        "fabricated": is_fabricated,
        "classification": classification,
    }


def gold_post_value_present(gold_answer: str, post_value: str | None) -> bool | None:
    """오염 2 보정: gold_context 답변에 정답(post_patch) 수치가 문자열로 들어있는지
    규칙으로 직접 재확인(LLM judge 와 독립)."""
    if not post_value:
        return None
    return _norm_val(post_value) in _norm_val(gold_answer)


def diagnose(flags: dict, answer_type: str) -> str:
    """개정 진단 캐스케이드."""
    if not flags["gold_correct"]:
        # gold 오답 — 채점 오판(정답수치 포함)인지 진짜 실패인지 분리
        if flags.get("gold_has_post_value"):
            return "judge_uncertain"
        return "generation_failure"
    if not flags["retrieved_correct"]:
        return "retrieval_failure"
    if answer_type == "outdated_trap" and flags.get("outdated_value_leaked"):
        return "question_leaked_outdated_value"
    return "ok"


def diagnose_legacy(flags: dict, answer_type: str) -> str:
    """원래(오염된) 캐스케이드 — before/after 비교 전용.
    구버전값 판정은 당시의 부분문자열 방식(outdated in answer & post not in answer)을 재현."""
    if not flags["gold_correct"]:
        return "generation_failure"
    if not flags["retrieved_correct"]:
        return "retrieval_failure"
    if answer_type == "outdated_trap" and flags.get("legacy_answered_old"):
        return "parametric_outdated_bias"
    return "ok"


def _legacy_answered_old(answer: str, outdated_value: str | None, post_value: str | None) -> bool:
    if not outdated_value:
        return False
    ans = answer.replace(" ", "")
    if outdated_value not in ans:  # 당시엔 부분문자열 매칭이었음
        return False
    if post_value and post_value.replace(" ", "") in ans:
        return False
    return True


def build_labels(flags: dict, atype: str) -> list[str]:
    """비배타 진단 태그(복수 적용 가능). no_context/gold_context 관찰을 각각 반영한다."""
    labels = []
    if atype == "outdated_trap" and flags.get("outdated_value_leaked"):
        labels.append("question_leaked_outdated_value")  # 질문의 구버전 값을 반복
    if flags.get("fabricated"):
        labels.append("fabricated_value")                # 질문·문서 어디에도 없는 값 생성
    if flags.get("gold_has_post_value") and not flags.get("gold_correct"):
        labels.append("judge_uncertain")                 # 내용상 정답이나 채점 오판
    return labels


def analyze(per_query_gen, eval_by_condition, questions, corpus_vals) -> dict:
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
        gold_answer = per_query_gen[qid]["answers"]["gold_context"]
        numeric = no_context_numeric_check(nc_answer, item["question"], outdated_value, corpus_vals)
        gold_has_post = gold_post_value_present(gold_answer, post_value)

        flags = {
            "no_context_correct": correct["no_context"],
            "gold_correct": correct["gold_context"],
            "retrieved_correct": correct["retrieved_context"],
            "all_three_correct": all(correct.values()),
            "gold_has_post_value": bool(gold_has_post),
            "outdated_value_leaked": numeric["outdated_value_leaked"],
            "fabricated": numeric["fabricated"],
            "legacy_answered_old": _legacy_answered_old(nc_answer, outdated_value, post_value),
        }

        per_query_out.append(
            {
                "qid": qid,
                "answer_type": atype,
                "question": item["question"],
                "outdated_value": outdated_value,
                "post_value": post_value,
                "diagnosis": diagnose(flags, atype),         # 시스템 1차 진단(단일)
                "labels": build_labels(flags, atype),        # 비배타 진단 태그(복수)
                "diagnosis_legacy": diagnose_legacy(flags, atype),
                "gold_post_value_present": gold_has_post,
                "no_context_numeric_check": numeric,
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

    dist_after = {lbl: 0 for lbl in LABELS}
    dist_before = {lbl: 0 for lbl in LEGACY_LABELS}
    dist_multi = {lbl: 0 for lbl in MULTI_LABELS}
    for e in per_query_out:
        dist_after[e["diagnosis"]] = dist_after.get(e["diagnosis"], 0) + 1
        dist_before[e["diagnosis_legacy"]] = dist_before.get(e["diagnosis_legacy"], 0) + 1
        for lbl in e["labels"]:
            dist_multi[lbl] = dist_multi.get(lbl, 0) + 1

    return {
        "per_query": per_query_out,
        "cross_table": cross_table,
        "diagnosis_distribution": dist_after,
        "diagnosis_distribution_legacy": dist_before,
        "multi_label_distribution": dist_multi,
    }


# ============================================================================
# from-cache 재구성
# ============================================================================
def load_from_cache() -> tuple[dict, dict]:
    with open(ABLATION_OUT, encoding="utf-8") as f:
        prev = json.load(f)
    per_query_gen, eval_by_condition = {}, {c: {} for c in CONDITIONS}
    for e in prev["per_query"]:
        qid = e["qid"]
        per_query_gen[qid] = {
            "question": e["question"],
            "answers": {c: e["conditions"][c]["answer"] for c in CONDITIONS},
            "doc_ids": {c: e["conditions"][c]["doc_ids"] for c in CONDITIONS},
        }
        for c in CONDITIONS:
            eval_by_condition[c][qid] = e["conditions"][c]["eval"]
    return per_query_gen, eval_by_condition


# ============================================================================
# 리포트
# ============================================================================
_CLASS_KO = {
    "leaked": "질문수치 반복(누출)",
    "fabricated": "값 창작(fabrication)",
    "leaked+fabricated": "누출 + 값 창작",
    "other_numeric": "기타 수치",
    "no_numeric": "수치 없음",
}


def print_report(analysis: dict) -> None:
    pq = analysis["per_query"]
    ct = analysis["cross_table"]

    print("\n" + "=" * 82)
    print("ABLATION 결과(보정판): 검색 오류 vs 답변 오류의 인과 분리")
    print("=" * 82)

    # (A) outdated_trap no_context — 누출(leaked) vs 창작(fabricated) 비배타 라벨
    print("\n[A] outdated_trap no_context(closed-book): 질문 누출 vs 값 창작(fabrication)")
    print("    합성 데이터라 사전 지식 가설 배제 → 구버전값=질문이 흘린 값 반복,")
    print("    질문·문서 어디에도 없는 값=모델이 창작한 fabrication(별개 실패 유형).")
    print("-" * 82)
    print(f"  {'qid':<6}{'leaked':>8}{'fabricated':>12}   누출값 / 창작값 → labels")
    for e in [x for x in pq if x["answer_type"] == "outdated_trap"]:
        nc = e["no_context_numeric_check"]
        lk = "O" if nc["outdated_value_leaked"] else "X"
        fb = "O" if nc["fabricated"] else "X"
        print(
            f"  {e['qid']:<6}{lk:>8}{fb:>12}   {nc['leaked_values']} / {nc['fabricated_values']}"
            f"  → {e['labels']}"
        )

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

    # (B') gold_context 채점 재확인: judge=no 인데 정답수치는 포함된 케이스
    print("\n[B'] gold_context 채점 재확인 (오염 2): 정답수치 규칙 포함 vs judge 판정")
    print("-" * 82)
    traps = [x for x in pq if x["answer_type"] == "outdated_trap"]
    judge_yes = sum(1 for e in traps if e["conditions"]["gold_context"]["correct"])
    rule_has = sum(1 for e in traps if e["gold_post_value_present"])
    for e in traps:
        judge = "O" if e["conditions"]["gold_context"]["correct"] else "X"
        rule = "O" if e["gold_post_value_present"] else "X"
        note = " ← 채점 오판(judge=X, 수치=O)" if (judge == "X" and rule == "O") else ""
        print(f"  {e['qid']} | judge정답={judge} | 정답수치포함(규칙)={rule}{note}")
    print(f"  요약(outdated_trap): judge 정답 {judge_yes}/5 vs 규칙 '정답수치 포함' {rule_has}/5")

    # (C) 진단 라벨 before/after
    print("\n[C] 진단 라벨 분포 — before(오염) → after(보정), 총 15문항")
    print("-" * 82)
    before, after = analysis["diagnosis_distribution_legacy"], analysis["diagnosis_distribution"]
    all_labels = list(dict.fromkeys(LEGACY_LABELS + LABELS))
    print(f"  {'label':<32}{'before':>8}{'after':>8}   문항(after)")
    for lbl in all_labels:
        b = before.get(lbl, 0)
        a = after.get(lbl, 0)
        b_s = str(b) if lbl in before else "-"
        a_s = str(a) if lbl in after else "-"
        qids = [e["qid"] for e in pq if e["diagnosis"] == lbl]
        print(f"  {lbl:<32}{b_s:>8}{a_s:>8}   {qids}")

    # (D) 비배타 진단 태그 분포(복수 적용 가능)
    print("\n[D] 비배타 진단 태그 분포 (한 문항에 복수 적용 가능)")
    print("-" * 82)
    multi = analysis["multi_label_distribution"]
    for lbl in MULTI_LABELS:
        qids = [e["qid"] for e in pq if lbl in e["labels"]]
        print(f"  {lbl:<32}{multi.get(lbl, 0):>3}   {qids}")


# ============================================================================
def main() -> None:
    from_cache = "--from-cache" in sys.argv
    cfg = load_config()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(QRELS_PATH, encoding="utf-8") as f:
        qrels = json.load(f)
    questions_by_qid = {q["qid"]: q for q in questions}
    doc_ids_all, contents = load_corpus(CORPUS_PATH)
    corpus_by_id = dict(zip(doc_ids_all, contents))
    corpus_vals = value_tokens(" ".join(contents))  # fabrication 판정용(문서에 있는 수치)

    mode = cfg.judge_mode
    if from_cache:
        print("[run_ablation] --from-cache: 저장된 답변/채점을 재분석합니다(LLM 미호출).")
        per_query_gen, eval_by_condition = load_from_cache()
    else:
        per_query_gen, client = generate_all(cfg, questions, qrels, corpus_by_id)
        llm = None
        if mode != "rule":
            if client.available():
                llm = LLMJudge(client, cfg.judge_model, cfg.seed)
            else:
                print("[run_ablation] 경고: Ollama 미연결 → rule 모드로 강등하여 채점합니다.")
                mode = "rule"
        eval_by_condition = score_all(per_query_gen, questions_by_qid, corpus_by_id, mode, cfg.k, llm)

    analysis = analyze(per_query_gen, eval_by_condition, questions, corpus_vals)

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
            "generation_failure": "gold 오답이고 정답수치(post)조차 답변에 없음(진짜 생성 실패).",
            "judge_uncertain": "gold 답변에 정답수치는 포함됐으나 correctness=no 로 채점됨(채점 오판).",
            "retrieval_failure": "gold 정답, retrieved 오답(검색 문제).",
            "question_leaked_outdated_value": (
                "outdated_trap 에서 no_context 답변이 질문에 포함된 구버전 값을 그대로 반복함"
                "(합성 데이터이므로 사전 지식이 아니라 '질문 누출'로 해석)."
            ),
            "fabricated_value": (
                "no_context 답변이 질문에도 문서(corpus)에도 없는 수치를 생성함"
                "(구버전 반복/누출과 구별되는 별개의 fabrication 실패 유형)."
            ),
            "parametric_outdated_bias": "[폐기] 합성 데이터 특성상 성립 불가 → question_leaked_outdated_value 로 대체.",
        },
        "multi_labels": MULTI_LABELS,
        **analysis,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ABLATION_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(analysis)
    print(f"\n→ {ABLATION_OUT}")


if __name__ == "__main__":
    main()
