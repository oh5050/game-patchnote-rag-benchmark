# ---------------------------------------------------------------------------
# 역할: LLM-as-judge. judges/ 의 프롬프트(txt)를 로드해 Ollama judge_model 로
#       이진(yes/no) 판정을 받고, JSON 한 줄 출력을 {verdict, reason}로 파싱.
# 입력:
#   - client(OllamaClient), judge_model, seed
#   - correctness(question, answer, gold) / faithfulness(answer, context) /
#     hallucination(answer, context)
# 출력:
#   - {"verdict": "yes"|"no", "reason": str}
# 특이사항:
#   - 프롬프트는 <QUESTION>/<ANSWER>/<GOLD>/<CONTEXT> 토큰을 치환(JSON 중괄호와 충돌 방지).
#   - 소형 모델 출력이 불안정할 수 있어 JSON 파싱 실패 시 폴백 처리.
# 의존관계:
#   - src.llm.ollama_client, src.config(PROJECT_ROOT). 외부 API 없음.
# 사용처: src.evaluation.answer_eval (hybrid 채점의 LLM 보완 단계)
# ---------------------------------------------------------------------------

import json
import os
import re

from src.config import PROJECT_ROOT

JUDGES_DIR = os.path.join(PROJECT_ROOT, "judges")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_prompt(filename: str) -> str:
    with open(os.path.join(JUDGES_DIR, filename), encoding="utf-8") as f:
        return f.read()


def parse_verdict(text: str) -> dict:
    """모델 출력에서 {"verdict","reason"} 추출. 실패 시 폴백."""
    match = _JSON_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict in ("yes", "no"):
                return {"verdict": verdict, "reason": str(obj.get("reason", "")).strip()}
        except json.JSONDecodeError:
            pass
    # 폴백: 평문에서 yes/no 흔적 탐색
    low = text.lower()
    if "yes" in low and "no" not in low:
        return {"verdict": "yes", "reason": "평문에서 yes 추정(파싱 실패)"}
    if "no" in low and "yes" not in low:
        return {"verdict": "no", "reason": "평문에서 no 추정(파싱 실패)"}
    return {"verdict": "no", "reason": f"판정 파싱 실패: {text[:80]}"}


class LLMJudge:
    def __init__(self, client, judge_model: str, seed: int):
        self.client = client
        self.judge_model = judge_model
        self.seed = seed
        self._correctness_tpl = load_prompt("correctness_prompt.txt")
        self._faithfulness_tpl = load_prompt("faithfulness_prompt.txt")
        self._hallucination_tpl = load_prompt("hallucination_prompt.txt")

    def available(self) -> bool:
        return self.client.available()

    def _run(self, prompt: str) -> dict:
        content = self.client.chat(
            self.judge_model,
            [{"role": "user", "content": prompt}],
            seed=self.seed,
            temperature=0.0,
            num_predict=200,
        )
        return parse_verdict(content)

    def correctness(self, question: str, answer: str, gold: str) -> dict:
        prompt = (
            self._correctness_tpl
            .replace("<QUESTION>", question)
            .replace("<ANSWER>", answer)
            .replace("<GOLD>", gold)
        )
        return self._run(prompt)

    def faithfulness(self, answer: str, context: str) -> dict:
        prompt = (
            self._faithfulness_tpl
            .replace("<CONTEXT>", context)
            .replace("<ANSWER>", answer)
        )
        return self._run(prompt)

    def hallucination(self, answer: str, context: str) -> dict:
        prompt = (
            self._hallucination_tpl
            .replace("<CONTEXT>", context)
            .replace("<ANSWER>", answer)
        )
        return self._run(prompt)
