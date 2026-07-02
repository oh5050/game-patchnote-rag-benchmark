# ---------------------------------------------------------------------------
# 역할: 로컬 Ollama 서버와 통신하는 얇은 클라이언트 래퍼.
#       생성 모델(gen_model) 및 judge 모델(judge_model) 호출을 공통화.
# 입력:
#   - model 이름(str), messages/prompt, 샘플링 옵션(temperature 등), seed
# 출력:
#   - 모델 텍스트 응답(str)
#   - available(): 서버 연결 가능 여부(bool)
# 의존관계:
#   - ollama 파이썬 패키지 (로컬 서버). 외부 API 호출 없음.
# 사용처: src.generation.generator, src.evaluation.judge
# ---------------------------------------------------------------------------


class OllamaClient:
    """로컬 Ollama 래퍼. 서버 미연결 시 available()=False 로 상위에서 폴백 유도."""

    def __init__(self, host: str | None = None):
        self.host = host
        self._client = None
        self._ok: bool | None = None

    @property
    def client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host) if self.host else ollama
        return self._client

    def available(self) -> bool:
        """서버 연결 가능 여부를 1회 확인 후 캐시."""
        if self._ok is None:
            try:
                self.client.list()
                self._ok = True
            except Exception:
                self._ok = False
        return self._ok

    @staticmethod
    def _extract_content(resp) -> str:
        try:
            return resp["message"]["content"]
        except (TypeError, KeyError):
            return resp.message.content  # pydantic 응답 대비

    def chat(
        self,
        model: str,
        messages: list[dict],
        seed: int,
        temperature: float = 0.0,
        num_predict: int = 512,
    ) -> str:
        resp = self.client.chat(
            model=model,
            messages=messages,
            options={
                "seed": seed,          # 재현성
                "temperature": temperature,
                "num_predict": num_predict,
            },
        )
        return self._extract_content(resp).strip()

    def generate(
        self,
        model: str,
        prompt: str,
        seed: int,
        temperature: float = 0.0,
        num_predict: int = 512,
    ) -> str:
        return self.chat(
            model,
            [{"role": "user", "content": prompt}],
            seed,
            temperature,
            num_predict,
        )

    def ensure_model(self, model: str) -> bool:
        """모델 미설치 시 pull 시도. 성공 여부 반환(서버 없으면 False)."""
        if not self.available():
            return False
        try:
            self.client.show(model)
            return True
        except Exception:
            try:
                self.client.pull(model)
                return True
            except Exception:
                return False
