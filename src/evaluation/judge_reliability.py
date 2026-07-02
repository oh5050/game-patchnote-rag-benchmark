# ---------------------------------------------------------------------------
# 역할: judge 신뢰도 측정 — 사람 라벨(human_label)과 judge verdict 를 대조해
#       correctness/faithfulness/hallucination 축별 일치율(agreement)을 계산.
# 입력:
#   - data/questions/questions.json : human_label(축별 정답 라벨)
#   - results/answer_eval.json      : judge 판정(verdict/reason/method)
# 출력:
#   - results/judge_reliability.json : 축별/전체/방법별(rule vs llm) 일치율 + 불일치 목록
#   - 콘솔: 요약 표 + 불일치 케이스
# 핵심 가설: 규칙(rule) 기반 판정이 소형 LLM 판정보다 사람과 더 잘 일치한다.
# 라벨→verdict 매핑:
#   - correctness/faithfulness: "correct"→"yes", 그 외→"no"
#   - hallucination: 라벨이 이미 yes/no 의미(no=환각 없음) → 그대로 사용
# 의존관계:
#   - src.config(경로). 외부 API 없음.
# 사용처: 직접 실행 python -m src.evaluation.judge_reliability
# ---------------------------------------------------------------------------

import json
import os
import sys

try:  # 콘솔 인코딩(cp949 등)에서 비ASCII 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import QUESTIONS_PATH, RESULTS_DIR

ANSWER_EVAL_PATH = os.path.join(RESULTS_DIR, "answer_eval.json")
RELIABILITY_OUT = os.path.join(RESULTS_DIR, "judge_reliability.json")
AXES = ["correctness", "faithfulness", "hallucination"]


def expected_verdict(axis: str, label: str) -> str:
    """사람 라벨을 judge verdict 체계(yes/no)로 환산."""
    label = str(label).strip().lower()
    if axis in ("correctness", "faithfulness"):
        return "yes" if label in ("correct", "yes", "true", "pass", "faithful") else "no"
    # hallucination: 라벨이 이미 yes/no (no=환각 없음)
    if label in ("no", "none", "clean", "pass"):
        return "no"
    if label in ("yes", "hallucinated", "fail"):
        return "yes"
    return label


def _new_counter() -> dict:
    return {"agree": 0, "total": 0}


def _pct(agree: int, total: int) -> float:
    return round(100.0 * agree / total, 1) if total else 0.0


def compute_reliability(questions: list[dict], answer_eval: list[dict]) -> dict:
    human_by_qid = {q["qid"]: (q.get("human_label") or {}) for q in questions}

    per_axis = {axis: _new_counter() for axis in AXES}
    by_method = {"rule": _new_counter(), "llm": _new_counter()}
    overall = _new_counter()
    mismatches = []

    for row in answer_eval:
        qid = row["qid"]
        human = human_by_qid.get(qid, {})
        if not isinstance(human, dict):
            continue  # 축별 라벨이 아닌 경우 건너뜀
        for axis in AXES:
            if axis not in row or axis not in human:
                continue
            judgment = row[axis]
            j_verdict = str(judgment.get("verdict", "")).strip().lower()
            method = judgment.get("method", "?")
            exp = expected_verdict(axis, human[axis])
            agree = (j_verdict == exp)

            per_axis[axis]["total"] += 1
            overall["total"] += 1
            if method in by_method:
                by_method[method]["total"] += 1
            if agree:
                per_axis[axis]["agree"] += 1
                overall["agree"] += 1
                if method in by_method:
                    by_method[method]["agree"] += 1
            else:
                mismatches.append(
                    {
                        "qid": qid,
                        "axis": axis,
                        "human_label": human[axis],
                        "expected_verdict": exp,
                        "judge_verdict": j_verdict,
                        "judge_reason": judgment.get("reason", ""),
                        "method": method,
                    }
                )

    def _finalize(counters: dict) -> dict:
        return {k: {**v, "pct": _pct(v["agree"], v["total"])} for k, v in counters.items()}

    return {
        "per_axis": _finalize(per_axis),
        "by_method": _finalize(by_method),
        "overall": {**overall, "pct": _pct(overall["agree"], overall["total"])},
        "mismatches": mismatches,
    }


def _print_report(report: dict) -> None:
    print("== 축별 일치율 (agreement) ==")
    for axis, c in report["per_axis"].items():
        print(f"  {axis:<14} {c['agree']}/{c['total']}  ({c['pct']}%)")

    print("\n== 방법별 일치율 (가설: rule > llm) ==")
    for method, c in report["by_method"].items():
        print(f"  {method:<5} {c['agree']}/{c['total']}  ({c['pct']}%)")

    o = report["overall"]
    print(f"\n== 전체 == {o['agree']}/{o['total']}  ({o['pct']}%)")

    print("\n== 불일치 케이스 ==")
    if not report["mismatches"]:
        print("  (없음)")
    for m in report["mismatches"]:
        print(
            f"  [{m['qid']} / {m['axis']}] human={m['human_label']}(기대 {m['expected_verdict']}) "
            f"vs judge={m['judge_verdict']} [{m['method']}]\n"
            f"     이유: {m['judge_reason']}"
        )


def main() -> None:
    # 선택 인자: [answer_eval 경로] [출력 경로]  (없으면 기본 경로 사용)
    args = sys.argv[1:]
    answer_path = args[0] if len(args) >= 1 else ANSWER_EVAL_PATH
    out_path = args[1] if len(args) >= 2 else RELIABILITY_OUT

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(answer_path, encoding="utf-8") as f:
        answer_eval = json.load(f)["per_query"]

    report = compute_reliability(questions, answer_eval)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[judge_reliability] source={os.path.basename(answer_path)}")
    _print_report(report)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
