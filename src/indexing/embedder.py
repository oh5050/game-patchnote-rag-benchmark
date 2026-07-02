# ---------------------------------------------------------------------------
# 역할: sentence-transformers 로컬 모델로 텍스트를 임베딩 벡터로 변환하고,
#       corpus 임베딩을 results/ 캐시에 저장/재사용(모델명·내용 변경 시 무효화).
# 입력:
#   - embedding_model 이름, device, seed (config)
#   - 문서/질문 텍스트 리스트(list[str]) 또는 data/corpus/docs.json
#   - text_type: "passage"(문서) / "query"(질의) / None
# 출력:
#   - 정규화된 임베딩 행렬(numpy ndarray, float32, L2-normalized → 코사인=내적)
#   - (embed_corpus) (doc_ids, embeddings) 및 results 캐시 파일
# 특이사항:
#   - e5 계열 모델(intfloat/*e5*)은 문서에 "passage: ", 질의에 "query: " 접두어를
#     붙여야 정상 성능이 나온다. text_type 에 따라 자동으로 접두어를 부착한다.
# 의존관계:
#   - sentence-transformers, torch, numpy (로컬). 외부 API 없음.
# 사용처: src.indexing.vector_store(문서), src.retrieval.retriever(질의).
# ---------------------------------------------------------------------------

import hashlib
import json
import os

import numpy as np


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class Embedder:
    """sentence-transformers 로컬 임베딩 래퍼. 모델은 최초 사용 시 지연 로드."""

    def __init__(self, model_name: str, device: str = "cpu", seed: int = 42):
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self._model = None
        # e5 계열은 비대칭 접두어(passage/query)가 필요.
        self.is_e5 = "e5" in model_name.lower()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            _set_seed(self.seed)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _apply_prefix(self, texts: list[str], text_type: str | None) -> list[str]:
        """e5 계열이면 text_type 에 맞춰 'passage: '/'query: ' 접두어 부착."""
        if not self.is_e5 or text_type not in ("passage", "query"):
            return texts
        prefix = f"{text_type}: "
        return [prefix + t for t in texts]

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        text_type: str | None = None,
    ) -> "np.ndarray":
        prepared = self._apply_prefix(texts, text_type)
        emb = self.model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=True,  # 코사인 유사도를 내적으로 계산하기 위함
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype="float32")


def load_corpus(corpus_path: str) -> tuple[list[str], list[str]]:
    """docs.json → (doc_ids, contents)."""
    with open(corpus_path, encoding="utf-8") as f:
        docs = json.load(f)
    doc_ids = [d["doc_id"] for d in docs]
    contents = [d["content"] for d in docs]
    return doc_ids, contents


def _content_hash(contents: list[str]) -> str:
    return hashlib.md5("|".join(contents).encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, model_name: str) -> str:
    safe = model_name.replace("/", "__").replace(":", "_")
    return os.path.join(cache_dir, f"emb_{safe}.npz")


def embed_corpus(
    embedder: Embedder,
    corpus_path: str,
    cache_dir: str,
) -> tuple[list[str], "np.ndarray"]:
    """
    corpus 를 임베딩하되 캐시가 유효하면 재계산하지 않는다.
    캐시 무효화 조건: 모델명 변경(파일명+내부 메타), 문서 내용/구성 변경(해시).
    """
    doc_ids, contents = load_corpus(corpus_path)
    chash = _content_hash(contents)
    cache_path = _cache_path(cache_dir, embedder.model_name)

    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        same_model = str(cached["model_name"]) == embedder.model_name
        same_content = str(cached["content_hash"]) == chash
        same_ids = list(cached["doc_ids"]) == doc_ids
        if same_model and same_content and same_ids:
            return doc_ids, cached["embeddings"].astype("float32")

    embeddings = embedder.encode(contents, text_type="passage")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=embeddings,
        doc_ids=np.array(doc_ids, dtype=object),
        model_name=embedder.model_name,
        content_hash=chash,
    )
    return doc_ids, embeddings
