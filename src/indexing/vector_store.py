# ---------------------------------------------------------------------------
# 역할: 문서 임베딩으로 로컬 벡터 인덱스(FAISS)를 구축/저장/로드하고,
#       쿼리 벡터로 top-k 검색해 (doc_id, score)를 순위 순서로 반환.
# 입력:
#   - build: doc_ids(list[str]), embeddings(정규화된 float32 ndarray)
#   - search: query_vec(ndarray), k(int)
#   - save/load: 저장 경로 prefix
# 출력:
#   - 디스크 저장물: {prefix}.faiss (인덱스), {prefix}.doc_ids.json (매핑)
#   - search: [(doc_id, score), ...]  (score 내림차순 = 순위; nDCG·MRR 용)
# 의존관계:
#   - faiss-cpu, numpy. 임베딩은 정규화되어 있다고 가정(내적 = 코사인).
# 사용처: scripts/build_index.py, src.retrieval.retriever
# ---------------------------------------------------------------------------

import json
import os

import faiss
import numpy as np


class VectorStore:
    def __init__(self):
        self.index = None
        self.doc_ids: list[str] = []
        self.dim: int | None = None

    def build(self, doc_ids: list[str], embeddings: "np.ndarray") -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("embeddings 는 2차원 (n_docs, dim) 이어야 합니다.")
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("doc_ids 수와 embeddings 행 수가 일치해야 합니다.")
        self.dim = int(embeddings.shape[1])
        # 정규화된 벡터에 대한 내적(IP) = 코사인 유사도. FAISS 는 내림차순 반환.
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.doc_ids = list(doc_ids)

    def save(self, path: str) -> None:
        """path 를 prefix 로 사용: {path}.faiss, {path}.doc_ids.json.
        FAISS 의 C++ 파일 IO 는 비ASCII 경로(예: 'Search–Answer')를 처리하지 못하므로
        인덱스를 메모리로 직렬화한 뒤 파이썬 파일 IO 로 기록한다."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        buf = faiss.serialize_index(self.index)
        with open(path + ".faiss", "wb") as f:
            f.write(np.asarray(buf, dtype="uint8").tobytes())
        with open(path + ".doc_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False)

    def load(self, path: str) -> "VectorStore":
        with open(path + ".faiss", "rb") as f:
            buf = np.frombuffer(f.read(), dtype="uint8")
        self.index = faiss.deserialize_index(buf)
        with open(path + ".doc_ids.json", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self.dim = self.index.d
        return self

    def search(self, query_vec: "np.ndarray", k: int) -> list[tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("인덱스가 없습니다. build() 또는 load() 를 먼저 호출하세요.")
        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        k = min(k, len(self.doc_ids))
        scores, idxs = self.index.search(q, k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:  # 후보 부족 시 FAISS 가 -1 을 채움
                continue
            results.append((self.doc_ids[int(idx)], float(score)))
        return results
