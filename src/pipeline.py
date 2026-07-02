# ---------------------------------------------------------------------------
# 역할: 검색→생성→평가 전 과정을 조율(orchestrate)하는 상위 파이프라인.
# 입력:
#   - config (src/config.py)
#   - data/questions, data/corpus, data/qrels
# 출력:
#   - results/ 하위의 검색 결과·생성 답변·평가 리포트
# 의존관계:
#   - src.config, src.indexing.*, src.retrieval.retriever,
#     src.generation.generator, src.evaluation.*
# 비고: scripts/* 는 이 파이프라인의 개별 단계를 CLI 로 노출.
# ---------------------------------------------------------------------------

# def run_all(cfg) -> None: ...            # build_index → retrieve → generate → eval
# def run_search_stage(cfg) -> None: ...   # 검색 평가만
# def run_answer_stage(cfg) -> None: ...   # 답변 평가만
