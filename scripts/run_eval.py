# ---------------------------------------------------------------------------
# 역할: [엔트리포인트] generation.json 을 읽어 답변을 하이브리드 채점(judge)하고 저장.
# 입력:
#   - config.json(judge_mode, judge_model, seed, k),
#     results/generation.json, data/questions/questions.json, data/corpus/docs.json
# 출력:
#   - results/answer_eval.json : {config, per_query:[{qid, answer_type,
#       correctness{verdict,reason,method}, faithfulness{...},
#       (outdated_trap 시)hallucination{...}}]}
# 의존관계:
#   - src.config, src.evaluation.answer_eval, src.evaluation.judge,
#     src.llm.ollama_client, src.indexing.embedder(load_corpus)
# 실행: python -m scripts.run_eval   (run_generation 이후)
# 비고: 검색 지표는 src.evaluation.search_eval 에서 별도 수행(decoupled).
# ---------------------------------------------------------------------------

import json
import os
import sys

try:  # 콘솔 인코딩(cp949 등)에서 비ASCII 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import CORPUS_PATH, QUESTIONS_PATH, RESULTS_DIR, load_config
from src.evaluation.answer_eval import evaluate_answers
from src.evaluation.judge import LLMJudge
from src.indexing.embedder import load_corpus
from src.llm.ollama_client import OllamaClient

GENERATION_PATH = os.path.join(RESULTS_DIR, "generation.json")
ANSWER_EVAL_OUT = os.path.join(RESULTS_DIR, "answer_eval.json")


def main() -> None:
    cfg = load_config()

    with open(GENERATION_PATH, encoding="utf-8") as f:
        generation = json.load(f)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    questions_by_qid = {q["qid"]: q for q in questions}

    doc_ids, contents = load_corpus(CORPUS_PATH)
    corpus_by_id = dict(zip(doc_ids, contents))

    mode = cfg.judge_mode
    llm = None
    if mode != "rule":
        client = OllamaClient()
        if client.available():
            llm = LLMJudge(client, cfg.judge_model, cfg.seed)
        else:
            print("[run_eval] 경고: Ollama 미연결 → rule 모드로 강등하여 채점합니다.")
            mode = "rule"

    results = evaluate_answers(generation, questions_by_qid, corpus_by_id, mode, cfg.k, llm)
    report = {
        "config": {"judge_mode": mode, "judge_model": cfg.judge_model, "k": cfg.k},
        "per_query": results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ANSWER_EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[run_eval] judge_mode={mode} | model={cfg.judge_model} | queries={len(results)}")
    for r in results:
        line = f"  {r['qid']} ({r['answer_type']}):"
        for metric in ("correctness", "faithfulness", "hallucination"):
            if metric in r:
                j = r[metric]
                line += f" {metric}={j['verdict']}[{j['method']}]"
        print(line)
    print(f"→ {ANSWER_EVAL_OUT}")


if __name__ == "__main__":
    main()
