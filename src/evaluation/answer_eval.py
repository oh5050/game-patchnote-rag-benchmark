# ---------------------------------------------------------------------------
# 역할: 답변(Answer) 단계 평가 — 검색 품질과 독립적으로 생성 답변을 하이브리드 채점.
# 전략(judge_mode 별):
#   - correctness  : gold_answer_keywords 규칙 매칭으로 1차 판정, 애매하면 LLM 2차 확인
#   - faithfulness : 답변↔context 어휘 오버랩으로 1차 신호, 중간대면 LLM 보완
#   - hallucination: outdated_trap 질문 전용. LLM judge 앞에 '규칙 게이트'를 둔다.
#                    (A) 정정 신호 키워드("패치","바뀌","변경","현재") 존재 &&
#                    (B) gold_answer_keywords 의 정답 수치("120초" 등) 포함
#                    A&&B 이면 hallucination=no 로 규칙 확정(LLM 미호출),
#                    아니면 기존대로 LLM judge 로 위임.
#                    (구버전 수치를 사실처럼 말하면 fail=yes, 바뀌었다고 정정하면 pass=no)
#   * judge_mode == "rule" 이면 LLM 호출을 건너뛴다.
#   * judge_mode == "llm"  이면 규칙을 건너뛰고 LLM 으로 판정한다.
#   * judge_mode == "hybrid"(기본) 이면 규칙 우선 + 애매 구간만 LLM.
# 입력:
#   - generation_results(list[{qid, answer, retrieved_doc_ids}])
#   - questions_by_qid(gold_answer/keywords/answer_type/question)
#   - corpus_by_id(doc_id→content), judge_mode, k, llm(LLMJudge|None)
# 출력:
#   - per_query 판정 리스트. 각 판정 = {verdict, reason, method("rule"|"llm")}
# 의존관계:
#   - src.evaluation.judge(LLMJudge). 규칙 신호는 표준 라이브러리(re)만 사용.
# 사용처: scripts/run_eval.py, src.pipeline
# ---------------------------------------------------------------------------

import re

# 한글/영문/숫자 토큰 추출(어휘 오버랩용)
_TOKEN_RE = re.compile(r"[0-9]+|[A-Za-z]+|[가-힣]+")
OVERLAP_HIGH = 0.6  # 이상이면 규칙만으로 faithful=yes 확정
OVERLAP_LOW = 0.3   # 이하면 규칙만으로 faithful=no 확정 (그 사이는 애매)

# outdated_trap hallucination 규칙 게이트용
CORRECTION_SIGNALS = ["패치", "바뀌", "변경", "현재"]  # 조건 A: 정정 시도 신호


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def overlap_score(answer: str, context_texts: list[str]) -> float:
    """답변 토큰 중 context 토큰과 겹치는 비율 [0,1]."""
    ans = _tokens(answer)
    if not ans:
        return 0.0
    ctx: set[str] = set()
    for t in context_texts:
        ctx |= _tokens(t)
    return len(ans & ctx) / len(ans)


def _format_context(contexts: list[dict]) -> str:
    return "\n".join(f"[{c['doc_id']}] {c['content']}" for c in contexts)


def judge_correctness(
    question: str, answer: str, gold: str, keywords: list[str], mode: str, llm
) -> dict:
    """규칙(키워드) 1차 → 애매 시 LLM 2차."""
    matched = [k for k in keywords if k.lower() in answer.lower()]
    ratio = len(matched) / len(keywords) if keywords else 0.0

    if mode == "llm" and llm is not None:
        r = llm.correctness(question, answer, gold)
        r["method"] = "llm"
        return r

    if keywords and ratio == 1.0:
        return {"verdict": "yes", "reason": f"gold 키워드 전부 매칭 {matched}", "method": "rule"}
    if keywords and ratio == 0.0:
        return {"verdict": "no", "reason": f"gold 키워드 0/{len(keywords)} 매칭", "method": "rule"}

    # 부분 매칭(애매) 또는 키워드 없음
    if mode == "hybrid" and llm is not None:
        r = llm.correctness(question, answer, gold)
        r["method"] = "llm"
        r["reason"] = f"규칙 애매({len(matched)}/{len(keywords)}) → LLM: {r['reason']}"
        return r

    verdict = "yes" if ratio >= 0.5 else "no"
    return {"verdict": verdict, "reason": f"규칙 폴백 {len(matched)}/{len(keywords)} 매칭", "method": "rule"}


def judge_faithfulness(answer: str, contexts: list[dict], mode: str, llm) -> dict:
    """어휘 오버랩 1차 → 중간대면 LLM 보완."""
    ov = overlap_score(answer, [c["content"] for c in contexts])

    if mode == "llm" and llm is not None:
        r = llm.faithfulness(answer, _format_context(contexts))
        r["method"] = "llm"
        return r

    if ov >= OVERLAP_HIGH:
        return {"verdict": "yes", "reason": f"context 어휘 오버랩 {ov:.2f} (>= {OVERLAP_HIGH})", "method": "rule"}
    if ov <= OVERLAP_LOW:
        return {"verdict": "no", "reason": f"context 어휘 오버랩 {ov:.2f} (<= {OVERLAP_LOW})", "method": "rule"}

    if mode == "hybrid" and llm is not None:
        r = llm.faithfulness(answer, _format_context(contexts))
        r["method"] = "llm"
        r["reason"] = f"오버랩 애매({ov:.2f}) → LLM: {r['reason']}"
        return r

    verdict = "yes" if ov >= 0.5 else "no"
    return {"verdict": verdict, "reason": f"규칙 폴백 오버랩 {ov:.2f}", "method": "rule"}


def _outdated_gate(answer: str, keywords: list[str]) -> dict:
    """규칙 1차 게이트. 조건 A(정정 신호) && B(정답 수치 포함) 모두 참이면 통과.
    반환: {passed: bool, reason: str}."""
    found_signals = [s for s in CORRECTION_SIGNALS if s in answer]  # 조건 A
    cond_a = len(found_signals) > 0

    value_kws = [k for k in keywords if any(ch.isdigit() for ch in k)]  # 정답 수치(숫자 포함)
    found_values = [k for k in value_kws if k in answer]  # 조건 B
    cond_b = len(value_kws) > 0 and len(found_values) == len(value_kws)

    if cond_a and cond_b:
        return {
            "passed": True,
            "reason": f"규칙 게이트 통과: A(정정신호)={found_signals} & B(정답수치)={found_values} → 환각 아님",
        }
    return {
        "passed": False,
        "reason": f"규칙 게이트 미충족: A={cond_a}{found_signals}, B={cond_b}(정답수치 {value_kws} 중 {found_values} 포함)",
    }


def judge_hallucination(
    answer: str, contexts: list[dict], mode: str, llm, keywords: list[str] | None = None
) -> dict:
    """outdated_trap 전용. verdict=yes 면 환각(구버전/무근거), no 면 정정/근거 있음.
    LLM judge 앞에 규칙 게이트(A&&B)를 먼저 적용한다."""
    keywords = keywords or []

    # 규칙 1차 게이트: A(정정 신호) && B(정답 수치) → hallucination=no 로 확정
    gate = _outdated_gate(answer, keywords)
    if gate["passed"]:
        return {"verdict": "no", "reason": gate["reason"], "method": "rule"}

    # 게이트 미충족 → 기존대로 LLM judge 로 위임
    if mode != "rule" and llm is not None:
        r = llm.hallucination(answer, _format_context(contexts))
        r["method"] = "llm"
        r["reason"] = f"{gate['reason']} → LLM: {r['reason']}"
        return r

    # rule 모드 폴백: 오버랩이 매우 낮으면 무근거 주장 가능성 → 환각 의심(약한 휴리스틱)
    ov = overlap_score(answer, [c["content"] for c in contexts])
    verdict = "yes" if ov < OVERLAP_LOW else "no"
    return {"verdict": verdict, "reason": f"{gate['reason']}; 오버랩 휴리스틱 {ov:.2f}", "method": "rule"}


def evaluate_answers(
    generation_results: list[dict],
    questions_by_qid: dict[str, dict],
    corpus_by_id: dict[str, str],
    judge_mode: str,
    k: int,
    llm=None,
) -> list[dict]:
    out = []
    for gen in generation_results:
        qid = gen["qid"]
        answer = gen.get("answer", "")
        q = questions_by_qid.get(qid, {})
        ctx_ids = gen.get("retrieved_doc_ids", [])[:k]
        contexts = [{"doc_id": d, "content": corpus_by_id.get(d, "")} for d in ctx_ids]

        result = {
            "qid": qid,
            "answer_type": q.get("answer_type"),
            "correctness": judge_correctness(
                q.get("question", ""), answer, q.get("gold_answer", ""),
                q.get("gold_answer_keywords", []), judge_mode, llm,
            ),
            "faithfulness": judge_faithfulness(answer, contexts, judge_mode, llm),
        }
        # 구버전 함정 질문은 hallucination judge 로 별도 채점(규칙 게이트 → LLM)
        if q.get("answer_type") == "outdated_trap":
            result["hallucination"] = judge_hallucination(
                answer, contexts, judge_mode, llm, q.get("gold_answer_keywords", [])
            )

        out.append(result)
    return out
