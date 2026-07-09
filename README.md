# Search–Answer Decoupled Evaluation Benchmarking

게임 도메인 RAG 시스템을 **검색(Search)** 과 **답변(Answer)** 두 축으로 분리해 평가하고, **구버전 정보 hallucination(outdated)** 을 탐지하는 로컬 평가 하니스.

전 과정이 **로컬 · 소형 오픈모델**로만 동작한다(외부 API 없음).
임베딩 `intfloat/multilingual-e5-small` · 생성/judge `qwen2.5:3b` (Ollama) · seed 고정 · 전체 로그 저장

---

## 핵심 아이디어

- **검색 평가와 답변 평가를 분리**한다. 검색은 IR 지표(recall@k, nDCG@k, MRR)로, 답변은 correctness / faithfulness / hallucination 축으로 따로 채점한다.
- **지표 분리만으로는 원인이 갈리지 않는다.** 그래서 `no_context` / `gold_context` / `retrieved_context` 세 조건에서 같은 문항을 답하게 해 **오류의 인과**를 분리한다.
- **outdated_trap**: 질문이 구버전(pre_patch) 수치를 전제로 깔고, 정답은 "패치로 바뀌었으니 현재 기준으로 답하라"가 되는 함정 문항.
- **하이브리드 judge**: 소형 LLM judge가 불안정하므로 규칙으로 1차 판정하고 애매한 경우만 LLM으로 넘긴다. outdated_trap의 hallucination은 **규칙 게이트**(정정 신호 + 정답 수치 동시 충족)를 LLM보다 먼저 적용한다.
- **합성 데이터**: 평가셋은 가상 캐릭터·스킬로 구성했다. 실제 스킬 수치는 패치로 바뀌어 gold answer가 오염될 위험이 있어, 라이브서비스를 평가하는 벤치마크가 스스로 그 함정에 빠지지 않도록 한 설계 판단이다(문서 구조만 실제를 본뜸). 덕분에 뒤의 인과 분리에서 "모델의 사전 지식" 가설을 데이터로 배제할 수 있었다.

---

## 결과 요약

> **n=15의 파일럿이며 지표 간 차이는 통계적으로 유의하지 않다.** 이 단계의 목적은 성능 측정이 아니라, 평가 하니스가 오류 유형을 실제로 구분해내는지 확인하는 것이다. 모든 비율은 병기된 분자/분모로 표본 크기를 함께 확인할 것 — n이 작을수록 1건이 지표를 크게 흔든다.

### 1. 오류의 원인은 검색이 아니라 채점이었다

동일 모델·동일 seed로 15문항을 세 조건에서 생성해 인과를 분리했다.

| 조건 | 컨텍스트 | judge 기준 정답 |
| --- | --- | --- |
| `no_context` | 없음 (closed-book) | 0/15 |
| `gold_context` | qrels 정답 문서 강제 주입 | 11/15 |
| `retrieved_context` | 실제 top-k 검색 결과 | 11/15 |

`gold_context == retrieved_context` → **검색 랭킹의 불완전함이 답변 오류를 유발하지 않았다.**

진단 라벨 분포 (오염 보정 후):

| 라벨 | 건수 | 문항 |
| --- | --- | --- |
| `ok` | 10 | q001,q002,q004,q005,q007,q008,q010,q011,q013,q014 |
| `judge_uncertain` | 4 | q003,q006,q009,q012 |
| `generation_failure` | **0** | — |
| `retrieval_failure` | **0** | — |

outdated_trap에서 **judge 정답은 1/5였으나, 규칙으로 확인한 '정답 수치 포함'은 5/5**였다. 생성 모델은 정답을 답했고, 채점이 그것을 인정하지 못했다. 즉 병목은 검색도 생성도 아닌 **채점 계층**이며, 이는 아래 judge 신뢰도 측정과 독립적으로 일치한다.

### 2. 자동 판정을 다시 검증했다

**판정 단위**: 15문항 × 축별 판정 = correctness 15 + faithfulness 15 + hallucination(outdated_trap 5문항) 5 = **총 35건**. 이를 방법 기준으로 나누면 rule 25 + llm 10 = 35건으로, 같은 모집단의 두 관점이다.

| 구분 | 사람 라벨과 일치율 |
| --- | --- |
| **rule (25건)** | **92.0% (23/25)** |
| **llm (10건)** | **50.0% (5/10)** |
| correctness (15건) | 86.7% (13/15) |
| faithfulness (15건) | 73.3% (11/15) |
| hallucination (5건) | 80.0% (4/5) |
| 전체 (35건) | 80.0% (28/35) |

소형 LLM judge보다 규칙 판정이 안정적이라는 경향이 확인된다(파일럿 표본이라 통계적 유의성은 검증하지 않았다). 하이브리드가 규칙을 우선하고 애매한 경우만 LLM에 위임하는 설계의 근거다.

### 3. 검색 지표와 그 진단

| 유형 | recall@2 | nDCG@1 |
| --- | --- | --- |
| fact | 0.60 (6/10) | 0.50 (2.50/5) |
| multi_hop | 1.00 (10/10) | 1.00 (5.00/5) |
| outdated_trap | 1.00 (10/10) | 0.90 (4.50/5) |
| **전체** | **0.87 (26/30)** | **0.80 (12.00/15)** |

fact가 낮은 이유를 "pre/post 문서 표면이 거의 같아 검색이 구버전을 상위로 올린다"로 진단하고, 아래 Temporal Ablation으로 검증했다 — **부분 지지**. 앞선 인과 분리에서 `retrieval_failure=0`이므로, **이 낮은 지표는 답변 오류로 이어지지 않았다.** 지표가 낮다고 곧 병목인 것은 아니다.

### 4. `no_context`가 드러낸 것: 질문 누출과 fabrication

`no_context`는 원래 모델의 파라메트릭 지식을 재려던 조건이었다. 그러나 합성 데이터라 사전 지식 가설이 배제되면서(fact·multi_hop 정답률 **0/15**), 이 조건은 대신 두 가지를 드러냈다.

- **질문 누출** — 함정 질문이 구버전 값을 전제로 포함하므로, 문서가 없으면 모델은 질문에 흘린 값을 그대로 반복한다. (q003 `100초` · q009 `30초` · q012 `25%` · q015 `60초`)
- **fabrication** — 문서가 없을 때 **질문에도 문서에도 없는 수치를 창작**한다. (q006 `140%` · q015 `70초`) 이는 이 벤치마크가 겨냥한 outdated hallucination(구버전을 사실처럼 반복)과 **구별되는 별개 유형**이다.

한 문항이 두 유형을 겹쳐 저지를 수 있으므로 `question_leaked_outdated_value` / `fabricated_value` / `judge_uncertain`을 **비배타 다중 라벨**로 기록한다.

---

## Temporal Ablation: 진단과 추정의 검증

> 이 실험은 임베딩 모델·k·seed·device를 모두 고정하고 **문서 표현이라는 단일 변인만** 바꾸는 **통제 실험**이다. (아래 "임베딩 모델 교체"와 성격이 다르다.)

세 조건 — `baseline`(원본) / `temporal_marked`(문서에 시점 표기 삽입) / `temporal_marked_query_hint`(문서는 동일, **쿼리에만** 시점 힌트 접두).

| 유형 | recall@2 (base→marked→+hint) | nDCG@1 (base→marked→+hint) |
| --- | --- | --- |
| fact | 6/10 → **8/10** → 8/10 | 2.50/5 → 2.50/5 → 2.50/5 |
| multi_hop | 10/10 → 10/10 → 10/10 | 5.00/5 → 5.00/5 → 5.00/5 |
| outdated_trap | 10/10 → 10/10 → 10/10 | 4.50/5 → **4.00/5** → 4.00/5 |
| **ALL** | 26/30 → 28/30 → 28/30 | 12.00/15 → 11.50/15 → 11.50/15 |

**결론 1 — 문서 시점 표기: 부분 지지.** fact의 recall@2는 +0.20 개선(0.60→0.80)됐으나 **nDCG@1(rank-1)은 0.50 불변** — 최상위에는 여전히 구버전 문서가 앉아 있다. "상위로 온다"는 검증됐고 "**최상위**로 온다"는 문서 표기만으로 교정되지 않았다. 부작용으로 outdated_trap의 nDCG@1이 -0.10 회귀했다.

**결론 2 — 쿼리 시점 힌트: 반증.** rank-1 미교정의 원인을 "쿼리에 시점 단서가 없어 문서 마커를 못 당긴다"로 추정하고 쿼리에만 힌트를 넣어 검증했다. 쿼리 임베딩은 실제로 바뀌었으나(점수와 2위 이하 순위가 변함 — 힌트 미적용이 아님) **모든 셀이 소수점까지 동일**했다. 추정은 **틀렸다.** rank-1 교정 실패의 원인은 다른 데 있다 — 이 임베딩 모델이 "…이름은?" 류 질의에서 시점 토큰보다 개체명 매칭에 크게 의존하는 것으로 보이나, **추가 검증하지 않았고 사실 그대로만 기록한다.**

---

## 정직성 노트

**human_label 재라벨링 — 무엇이 고정이고 무엇을 보정했나**

- **고정**: 정답 근거(문서의 patch 이후 수치)와 `gold_answer`/`gold_answer_keywords`는 모델 출력과 무관하게 고정이다. 재라벨링에서도 건드리지 않았다.
- **보정**: `human_label`은 "이 모델의 이 출력이 각 축의 기준을 충족했는가"에 대한 **채점 결과**다. 초기에 실제 출력을 보지 않고 "모델이 이상적으로 정정했을 것"이라 가정해 매긴 탓에 실제 출력과 어긋났고, 이를 전수 점검해 보정했다.
- **한계**: 라벨 보정에 참고한 출력과 신뢰도 평가에 쓴 출력이 **분리되어 있지 않다**(dev/test split 없음). 따라서 위 일치율, 특히 rule 92%는 **과대추정일 수 있다.** 미해소 한계다.

**임베딩 모델 교체는 통제 실험이 아니다.** 초기 영어 중심 임베딩(`all-MiniLM-L6-v2`)에서 한국어 고유명사 검색이 무너져 다국어 모델로 교체하고 e5 규약(`passage:`/`query:`)을 적용했다. 다만 모델을 바꾸면 학습 데이터·토크나이저·차원·접두어가 동시에 바뀐다. 정확히는 **모델 교체 전후 문항별 순위 변화를 비교한 관찰**이며, 어떤 요인이 얼마나 기여했는지는 분리하지 않았다.

---

## 빠른 시작

```bash
pip install -r requirements.txt
ollama pull qwen2.5:3b

python -m scripts.build_index            # 코퍼스 임베딩 → FAISS 인덱스
python -m scripts.run_retrieval          # 질문별 top-k 검색
python -m src.evaluation.search_eval     # 검색 평가
python -m scripts.run_generation         # RAG 답변 생성
python -m scripts.run_eval               # 하이브리드 답변 채점
python -m src.evaluation.judge_reliability   # judge 신뢰도 측정

python -m scripts.run_ablation --from-cache  # 인과 분리(저장된 답변 재분석, LLM 미호출)
python -m scripts.run_temporal_ablation      # 시점 표기 통제 실험(3조건)
```

---

## 구조

```
config.json                 # k, embedding_model, gen_model, judge_model, seed, judge_mode, device
data/
  corpus/docs.json          # 문서 (doc_id, content, period[pre/post_patch], source)
  questions/questions.json  # 질문 + gold_answer + gold_answer_keywords + answer_type + human_label
  qrels/qrels.json          # 검색 정답 {qid: {doc_id: relevance}}
judges/                     # correctness / faithfulness / hallucination judge 프롬프트
src/
  indexing/                 # embedder(캐시 무효화) · vector_store(FAISS)
  retrieval/retriever.py
  llm/ollama_client.py      # 미연결 시 폴백
  generation/generator.py   # groundedness + outdated 정정 유도 프롬프트
  evaluation/               # metrics · search_eval · answer_eval(규칙 게이트+LLM) · judge · judge_reliability
scripts/                    # build_index · run_retrieval · run_generation · run_eval
                            # run_ablation(인과 분리) · run_temporal_ablation(통제 실험)
results/                    # 산출물(json). 캐시(.npz)·인덱스(.faiss)는 gitignore
```

**평가 축**
- 검색: `hit@k`, `recall@k`, `nDCG@k`(graded relevance), `MRR`
- 답변: `correctness`(gold 키워드 규칙 1차 → 애매하면 LLM) · `faithfulness`(답변↔context 어휘 오버랩 1차 → 중간대면 LLM) · `hallucination`(outdated_trap 전용, 규칙 게이트 우선)
- 모든 판정은 `{verdict, reason, method(rule|llm)}`로 기록되어 판정 경로를 추적할 수 있다.
