# ---------------------------------------------------------------------------
# 역할: [엔트리포인트] questions.json 의 각 질문으로 top-k 검색을 수행하고 저장.
#       (아직 평가 지표는 붙이지 않음 — 검색 결과 확인용.)
# 입력:
#   - config.json, data/questions/questions.json, 저장된 벡터 인덱스
# 출력:
#   - results/retrieval.json : [{qid, question, retrieved_doc_ids, scores}, ...]
# 의존관계:
#   - src.config, src.indexing.embedder, src.indexing.vector_store,
#     src.retrieval.retriever
# 실행: python -m scripts.run_retrieval   (build_index 이후)
# ---------------------------------------------------------------------------

import json
import os
import sys

try:  # 콘솔 인코딩(cp949 등)에서 비ASCII 경로 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import QUESTIONS_PATH, RESULTS_DIR, load_config
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.retrieval.retriever import Retriever

INDEX_PREFIX = os.path.join(RESULTS_DIR, "index", "corpus")
RETRIEVAL_OUT = os.path.join(RESULTS_DIR, "retrieval.json")


def main() -> None:
    cfg = load_config()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    embedder = Embedder(cfg.embedding_model, device=cfg.device, seed=cfg.seed)
    store = VectorStore().load(INDEX_PREFIX)
    retriever = Retriever(embedder, store, cfg.k)

    out = []
    for item in questions:
        results = retriever.retrieve(item["question"])
        out.append(
            {
                "qid": item["qid"],
                "question": item["question"],
                "retrieved_doc_ids": [doc_id for doc_id, _ in results],
                "scores": [score for _, score in results],
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RETRIEVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[run_retrieval] questions={len(out)} | k={cfg.k} → {RETRIEVAL_OUT}")
    for row in out:
        print(f"  {row['qid']}: {row['retrieved_doc_ids']}")


if __name__ == "__main__":
    main()
