# ---------------------------------------------------------------------------
# 역할: [엔트리포인트] 각 질문에 대해 retriever 로 top-k 문서를 가져오고
#       generator 로 RAG 답변을 생성해 저장. (아직 채점 없음 — 생성 확인용.)
# 입력:
#   - config.json, data/questions/questions.json, data/corpus/docs.json,
#     저장된 벡터 인덱스
# 출력:
#   - results/generation.json : [{qid, question, retrieved_doc_ids, answer}, ...]
# 의존관계:
#   - src.config, src.indexing.*, src.retrieval.retriever,
#     src.llm.ollama_client, src.generation.generator
# 실행: python -m scripts.run_generation   (build_index 이후)
# ---------------------------------------------------------------------------

import json
import os
import sys

try:  # 콘솔 인코딩(cp949 등)에서 비ASCII 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import CORPUS_PATH, QUESTIONS_PATH, RESULTS_DIR, load_config
from src.generation.generator import Generator
from src.indexing.embedder import Embedder, load_corpus
from src.indexing.vector_store import VectorStore
from src.llm.ollama_client import OllamaClient
from src.retrieval.retriever import Retriever

INDEX_PREFIX = os.path.join(RESULTS_DIR, "index", "corpus")
GENERATION_OUT = os.path.join(RESULTS_DIR, "generation.json")


def main() -> None:
    cfg = load_config()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    doc_ids, contents = load_corpus(CORPUS_PATH)
    id2content = dict(zip(doc_ids, contents))  # doc_id → content 매핑

    embedder = Embedder(cfg.embedding_model, device=cfg.device, seed=cfg.seed)
    store = VectorStore().load(INDEX_PREFIX)
    retriever = Retriever(embedder, store, cfg.k)

    client = OllamaClient()
    if not client.available():
        print("[run_generation] 경고: Ollama 미연결 → dummy 폴백 답변으로 진행합니다.")
    generator = Generator(client, cfg.gen_model, cfg.seed)

    out = []
    for item in questions:
        results = retriever.retrieve(item["question"])
        retrieved_ids = [doc_id for doc_id, _ in results]
        contexts = [{"doc_id": d, "content": id2content.get(d, "")} for d in retrieved_ids]
        answer = generator.generate(item["question"], contexts)
        out.append(
            {
                "qid": item["qid"],
                "question": item["question"],
                "retrieved_doc_ids": retrieved_ids,
                "answer": answer,
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(GENERATION_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[run_generation] model={cfg.gen_model} | questions={len(out)} → {GENERATION_OUT}")
    for row in out:
        preview = row["answer"].replace("\n", " ")[:80]
        print(f"  {row['qid']} [{','.join(row['retrieved_doc_ids'][:cfg.k])}]: {preview}")


if __name__ == "__main__":
    main()
