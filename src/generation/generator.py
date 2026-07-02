# ---------------------------------------------------------------------------
# 역할: 검색된 컨텍스트를 바탕으로 Ollama 소형 모델로 RAG 답변 생성.
# 입력:
#   - question(str), contexts(list[{"doc_id","content"}]), gen_model/seed(config)
#   - client: OllamaClient (없으면 내부 생성)
# 출력:
#   - 생성된 답변 텍스트(str). Ollama 미연결 시 dummy 폴백 답변.
# 프롬프트 원칙:
#   (1) groundedness: "주어진 문서에만 근거, 없으면 모른다".
#   (2) outdated_trap 대응: 문서 간 정보가 상충/시점이 다르면 최신 상태를 기준으로
#       답하되 '바뀌었다는 사실'을 반드시 언급.
#   (3) seed 로 재현성 확보(temperature=0).
# 의존관계:
#   - src.llm.ollama_client. 외부 API 없음.
# 사용처: scripts/run_generation.py, src.pipeline
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "너는 게임 도메인 질문에 답하는 어시스턴트다. 다음 규칙을 반드시 지켜라.\n"
    "1) 아래 '문서'에 담긴 내용에만 근거해서 답하라. 문서에 근거가 없으면 "
    "추측하지 말고 '주어진 문서로는 알 수 없다'고 답하라.\n"
    "2) 문서들 사이에 정보가 상충하거나 시점이 다르면(예: 구버전 수치 vs 패치로 "
    "개정된 수치), 가장 최신 상태를 기준으로 답하라. 이때 '과거에는 ~였으나 패치로 "
    "바뀌어 현재는 ~다'처럼 값이 변경되었다는 사실을 명시하라.\n"
    "3) 질문의 전제가 구버전 정보에 기반해 틀렸다면, 그 전제가 더 이상 유효하지 "
    "않음을 먼저 지적하라.\n"
    "4) 간결하고 정확하게 한국어로 답하라."
)


class Generator:
    def __init__(self, client, gen_model: str, seed: int):
        # client 미지정 시 기본 OllamaClient 생성
        if client is None:
            from src.llm.ollama_client import OllamaClient

            client = OllamaClient()
        self.client = client
        self.gen_model = gen_model
        self.seed = seed

    def _format_contexts(self, contexts: list[dict]) -> str:
        """문서를 [doc_id] 라벨과 함께 나열(출처 구분 → 변경 사실 언급 유도)."""
        lines = []
        for ctx in contexts:
            doc_id = ctx.get("doc_id", "?")
            content = ctx.get("content", "")
            lines.append(f"[{doc_id}] {content}")
        return "\n".join(lines)

    def build_prompt(self, question: str, contexts: list[dict]) -> str:
        context_block = self._format_contexts(contexts)
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"### 문서\n{context_block}\n\n"
            f"### 질문\n{question}\n\n"
            f"### 답변\n"
        )

    def _dummy_answer(self, question: str, contexts: list[dict]) -> str:
        """Ollama 미연결 시 결정적(재현 가능) 폴백 답변."""
        doc_ids = [c.get("doc_id", "?") for c in contexts]
        return (
            "[DUMMY] Ollama 서버에 연결할 수 없어 실제 생성 대신 폴백 답변을 반환합니다. "
            f"질문='{question}' / 참고 문서={doc_ids}"
        )

    def generate(self, question: str, contexts: list[dict]) -> str:
        if not self.client.available():
            return self._dummy_answer(question, contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"### 문서\n{self._format_contexts(contexts)}\n\n"
                    f"### 질문\n{question}"
                ),
            },
        ]
        try:
            return self.client.chat(self.gen_model, messages, seed=self.seed, temperature=0.0)
        except Exception as exc:  # 서버는 있으나 모델 미설치 등
            return f"[DUMMY] 생성 실패({exc}). " + self._dummy_answer(question, contexts)
