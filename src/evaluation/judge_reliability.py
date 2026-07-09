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
#
# 판정 단위(총 35건이 나오는 구조 — 계산기 없이 읽을 수 있도록 명시):
#   - 15문항 전체가 correctness 축 판정 대상          → 15건
#   - 15문항 전체가 faithfulness 축 판정 대상          → 15건
#   - hallucination 축은 outdated_trap 문항(5개)만 대상 → 5건
#   - 합계: 15 + 15 + 5 = 35건 (report["judgment_structure"] 에 이 분해를 그대로 저장)
#   - 이 35건은 rule/llm 두 method 로 나뉘어 판정됐고, by_method 의 분모 합도
#     25(rule) + 10(llm) = 35 로 위와 일치한다(같은 35건을 축 기준/방법 기준으로
#     두 번 나눠 센 것일 뿐, 서로 다른 모집단이 아니다).
# n=15 파일럿 주의: 총 35건 중 일부 축(예: hallucination=5건)은 표본이 매우 작아
#   1건의 불일치가 20%p 를 흔든다. pct 는 참고용이고 항상 agree/total 을 함께 본다.
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

    per_axis_final = _finalize(per_axis)
    # 판정 단위 구조를 그대로 노출: 15(correctness) + 15(faithfulness) + 5(hallucination) = 35.
    judgment_structure = {
        "num_questions": 15,
        "judgments_per_axis": {axis: per_axis[axis]["total"] for axis in AXES},
        "total_judgments": sum(per_axis[axis]["total"] for axis in AXES),
        "note": (
            "총 판정 수 = correctness(15문항) + faithfulness(15문항) + "
            "hallucination(outdated_trap 5문항만) = "
            + " + ".join(str(per_axis[axis]["total"]) for axis in AXES)
            + f" = {sum(per_axis[axis]['total'] for axis in AXES)}건. "
            "by_method 의 rule+llm 분모 합도 동일 35건(같은 판정을 방법 기준으로 재분할)."
        ),
    }

    return {
        "judgment_structure": judgment_structure,
        "per_axis": per_axis_final,
        "by_method": _finalize(by_method),
        "overall": {**overall, "pct": _pct(overall["agree"], overall["total"])},
        "mismatches": mismatches,
    }


def _print_report(report: dict) -> None:
    js = report["judgment_structure"]
    print(f"== 판정 단위: {js['note']} ==")

    print("\n== 축별 일치율 (agreement) ==  * value% (agree/total) — n 이 작을수록 참고용")
    for axis, c in report["per_axis"].items():
        print(f"  {axis:<14} {c['pct']}%  ({c['agree']}/{c['total']})")

    print("\n== 방법별 일치율 (가설: rule > llm) ==")
    for method, c in report["by_method"].items():
        print(f"  {method:<5} {c['pct']}%  ({c['agree']}/{c['total']})")

    o = report["overall"]
    print(f"\n== 전체 == {o['pct']}%  ({o['agree']}/{o['total']})")

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
