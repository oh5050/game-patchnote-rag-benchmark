# Search–Answer Decoupled Evaluation Benchmarking

게임 도메인 RAG 시스템을 **검색(Search)** 과 **답변(Answer)** 두 축으로 **분리(decouple)** 해서
평가하고, 추가로 **구버전 정보 hallucination(outdated)** 을 탐지하는 로컬 평가 하니스.

전 과정이 **로컬 · 소형 오픈모델**로만 동작한다(외부 API 사용 안 함).
- 임베딩: `sentence-transformers` 로컬 모델 (기본 `intfloat/multilingual-e5-small`)
- 생성 / judge: **Ollama** 로 서빙하는 소형 모델 (기본 `qwen2.5:3b`)

## 핵심 아이디어
- **검색 평가와 답변 평가를 분리**한다. 검색은 IR 지표(recall@k, nDCG@k, MRR)로, 답변은
  correctness / faithfulness / hallucination 축으로 따로 채점한다.
- **outdated_trap**: 질문이 구버전(pre_patch) 수치를 전제로 깔고, 정답은 "패치로 바뀌었으니
  현재 기준(post_patch)으로 답하라"가 되도록 설계된 함정 문항. 모델이 구버전을 사실처럼
  답하면 hallucination으로 잡는다.
- **하이브리드 judge**: 소형 LLM judge가 불안정하므로, 규칙(rule)으로 1차 판정하고 애매한
  경우에만 LLM으로 넘긴다. outdated_trap의 hallucination은 **규칙 게이트**(정정 신호 +
  정답 수치 동시 충족 시 통과)를 LLM보다 먼저 적용한다.
- **합성 데이터**: 평가셋은 가상 캐릭터·가상 스킬로 만든 합성 데이터다. 실제 게임 스킬의
  수치는 패치로 계속 바뀌어 gold answer가 오염될 위험이 있으므로, 라이브서비스를 평가하는
  벤치마크가 스스로 그 함정에 빠지지 않도록 의도적으로 합성했다. 단, 문서 구조(전직 차수,
  패치 전후 수치 변경)는 실제 게임 문서를 본떴다.

## 폴더 구조
```
config.json                 # k, embedding_model, gen_model, judge_model, seed, judge_mode, device
data/
  corpus/docs.json          # 게임 문서 (doc_id, content, period[pre/post_patch], source)
  questions/questions.json  # 질문 + gold_answer + gold_answer_keywords + answer_type + human_label
  qrels/qrels.json          # 검색 정답 {qid: {doc_id: relevance}}
judges/                     # correctness / faithfulness / hallucination LLM judge 프롬프트(txt)
src/
  config.py
  indexing/embedder.py      # sentence-transformers 임베딩 + results 캐시(모델명/내용 해시로 무효화)
  indexing/vector_store.py  # FAISS 인덱스 구축/저장/로드/검색
  retrieval/retriever.py    # 질문 top-k 검색
  llm/ollama_client.py      # Ollama 로컬 클라이언트 (미연결 시 폴백)
  generation/generator.py   # RAG 답변 생성 (groundedness + outdated 정정 유도 프롬프트)
  evaluation/metrics.py     # hit@k, recall@k, MRR, nDCG@k (순수 함수)
  evaluation/search_eval.py # 검색 평가 집계
  evaluation/answer_eval.py # 하이브리드 답변 채점 (규칙 게이트 + LLM)
  evaluation/judge.py       # LLM-as-judge (프롬프트 로드 + verdict 파싱)
  evaluation/judge_reliability.py  # human_label 대조 신뢰도 측정
scripts/                    # build_index / run_retrieval / run_generation / run_eval 엔트리포인트
results/                    # 산출물(json). 캐시(.npz)와 인덱스(.faiss)는 .gitignore 처리
```

## 설치
```bash
pip install -r requirements.txt
# Ollama 별도 설치 후 모델 준비
ollama pull qwen2.5:3b
```

## 실행 순서
```bash
# 1) 코퍼스 임베딩 → FAISS 인덱스 구축
python -m scripts.build_index

# 2) 질문별 top-k 검색
python -m scripts.run_retrieval

# 3) 검색 평가 (recall@k, nDCG@k, MRR)
python -m src.evaluation.search_eval

# 4) RAG 답변 생성 (Ollama)
python -m scripts.run_generation

# 5) 하이브리드 답변 채점 (규칙 게이트 + LLM judge)
python -m scripts.run_eval

# 6) judge 신뢰도 측정 (human_label 대조)
python -m src.evaluation.judge_reliability
```

## 설정 (`config.json`)
| 필드 | 설명 |
| --- | --- |
| `k` | 검색 top-k |
| `embedding_model` | sentence-transformers 모델 |
| `gen_model` | 답변 생성 Ollama 모델 |
| `judge_model` | LLM judge Ollama 모델 |
| `seed` | 재현성 시드 |
| `judge_mode` | `rule` / `llm` / `hybrid` |
| `device` | `cpu` / `cuda` |

## 평가 축
- **검색**: `hit@k`, `recall@k`, `nDCG@k`(graded relevance), `MRR`.
- **답변**:
  - `correctness` — gold 키워드 규칙 매칭 1차, 애매하면 LLM.
  - `faithfulness` — 답변↔context 어휘 오버랩 1차, 중간대면 LLM.
  - `hallucination` (outdated_trap 전용) — **규칙 게이트**(정정 신호 `패치/바뀌/변경/현재`
    && 정답 수치 포함) 통과 시 `no` 확정, 아니면 LLM.
- 각 판정은 `{verdict, reason, method(rule|llm)}` 로 기록되어 어떤 경로로 판정했는지 추적 가능.

## 주요 결과 (15문항, 5캐릭터 기준)
- **검색 평가** (`intfloat/multilingual-e5-small`, `results/search_eval.json`):
  | 유형 | recall@2 | nDCG@1 |
  | --- | --- | --- |
  | fact | 0.60 | 0.50 |
  | multi_hop | 1.00 | 1.00 |
  | outdated_trap | 1.00 | 0.90 |
  | **전체(MACRO)** | **0.87** | **0.80** |
  fact가 낮은 이유는 스킬명을 묻는 질문에서 pre/post 문서의 문장이 거의 같아 검색이
  구버전 문서를 상위로 올리기 때문이다.
  이는 시점만 다르고 표면이 같은 문서를 구분하는 문제의 본질적 난점이다.
- **임베딩 선택 근거**: 초기 영어 중심 임베딩(`all-MiniLM-L6-v2`)에서는 한국어 게임
  고유명사 검색이 무너졌고, 다국어 모델(`multilingual-e5-small`)로 교체하고 e5 규약
  (`passage:`/`query:` 접두어)을 적용해 개선했다. 두 모델의 질문별 순위 비교로 확인한
  통제 실험 결과다.
- **규칙 게이트가 outdated_trap을 안정적으로 처리**: 정정 성공 답변(정답 수치까지 명시)만
  `method=rule`로 `hallucination=no` 확정하고, 정정 실패/회피는 LLM으로 넘긴다.
- **judge 신뢰도(human_label 대조, 실제 답변 기준 라벨링 후)**:
  | method | 사람과의 일치율 |
  | --- | --- |
  | **rule** | **23/25 (92.0%)** |
  | **llm** | **5/10 (50.0%)** |
  | 전체 | 28/35 (80.0%) |
- 즉 **소형 LLM judge(50%)보다 규칙 판정(92%)이 훨씬 안정적**이라는 것이 수치로 확인된다.
  하이브리드가 규칙을 우선하고 애매한 경우만 LLM에 위임하는 설계의 근거.
- **평가 원칙**: `human_label`(정답 라벨)은 "이상적 답변 가정"이 아니라 **실제 생성된 답변을
  보고** 매긴다. 실제 출력과 어긋난 라벨은 신뢰도 수치를 오염시키므로 전수 점검 후 수정한다.

## 원칙
- 외부 API 금지, 전 과정 로컬 실행.
- 재현성을 위해 seed 고정, temperature=0.
- 임베딩 캐시는 모델명·문서 내용 해시로 무효화되어 데이터 변경 시 자동 재계산된다.
