# ---------------------------------------------------------------------------
# 역할: [엔트리포인트] corpus(docs.json)를 임베딩하여 로컬 벡터 인덱스를 구축·저장.
# 입력:
#   - config.json, data/corpus/docs.json
# 출력:
#   - results/index/corpus.faiss, results/index/corpus.doc_ids.json
#   - results/cache/emb_*.npz (임베딩 캐시)
# 의존관계:
#   - src.config, src.indexing.embedder, src.indexing.vector_store
# 실행: python -m scripts.build_index   (프로젝트 루트에서)
# ---------------------------------------------------------------------------

import os
import sys

try:  # 콘솔 인코딩(cp949 등)에서 비ASCII 경로 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import CORPUS_PATH, RESULTS_DIR, load_config
from src.indexing.embedder import Embedder, embed_corpus
from src.indexing.vector_store import VectorStore

INDEX_PREFIX = os.path.join(RESULTS_DIR, "index", "corpus")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")


def main() -> None:
    cfg = load_config()
    embedder = Embedder(cfg.embedding_model, device=cfg.device, seed=cfg.seed)
    doc_ids, embeddings = embed_corpus(embedder, CORPUS_PATH, CACHE_DIR)

    store = VectorStore()
    store.build(doc_ids, embeddings)
    store.save(INDEX_PREFIX)

    print(
        f"[build_index] model={cfg.embedding_model} | docs={len(doc_ids)} | "
        f"dim={store.dim} → {INDEX_PREFIX}.faiss"
    )


if __name__ == "__main__":
    main()
