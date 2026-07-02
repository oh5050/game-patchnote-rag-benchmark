# ---------------------------------------------------------------------------
# 역할: 질문 문자열을 임베딩하여 벡터 인덱스에서 top-k 문서를 검색.
# 입력:
#   - 질문 텍스트(str) 또는 리스트, k (config), Embedder, VectorStore
# 출력:
#   - retrieve: [(doc_id, score), ...] (순위 순)
#   - batch_retrieve: 위 리스트들의 리스트
# 의존관계:
#   - src.indexing.embedder(Embedder), src.indexing.vector_store(VectorStore)
# 사용처: scripts/run_retrieval.py, src.pipeline
# ---------------------------------------------------------------------------


class Retriever:
    def __init__(self, embedder, store, k: int):
        self.embedder = embedder
        self.store = store
        self.k = k

    def retrieve(self, question: str) -> list[tuple[str, float]]:
        q_vec = self.embedder.encode([question], text_type="query")
        return self.store.search(q_vec, self.k)

    def batch_retrieve(self, questions: list[str]) -> list[list[tuple[str, float]]]:
        vecs = self.embedder.encode(questions, text_type="query")  # 배치 임베딩으로 효율화
        return [self.store.search(vecs[i : i + 1], self.k) for i in range(len(questions))]
