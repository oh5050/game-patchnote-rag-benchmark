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
> **n=15 의 파일럿이며 지표 간 차이는 통계적으로 유의하지 않다.** 이 단계의 목적은 성능
> 측정이 아니라, 평가 하니스가 오류 유형을 실제로 구분해내는지 확인하는 것이다. 아래 모든
> 비율은 소수점 표기만 보지 말고 항상 병기된 분자/분모(예: `0.60 (6/10)`)로 표본 크기를
> 함께 확인할 것 — n 이 작을수록 1건의 차이가 지표를 크게 흔든다.

- **검색 평가** (`intfloat/multilingual-e5-small`, `results/search_eval.json`, 분자/분모 병기):
  | 유형 | recall@2 | nDCG@1 |
  | --- | --- | --- |
  | fact | 0.60 (6/10) | 0.50 (2.50/5) |
  | multi_hop | 1.00 (10/10) | 1.00 (5.00/5) |
  | outdated_trap | 1.00 (10/10) | 0.90 (4.50/5) |
  | **전체(ALL)** | **0.87 (26/30)** | **0.80 (12.00/15)** |
- recall@2 = top-2 내 회수 gold 문서 수 / 전체 gold 문서 수. nDCG@1 = 문항별 nDCG@1 값의 합 /
  문항 수. `python -m src.evaluation.search_eval` 실행 시 유형별 표가 이 형식으로 그대로 출력·저장된다.
- fact가 낮은 이유로 "스킬명을 묻는 질문에서 pre/post 문서의 문장이 거의 같아 검색이 구버전
    문서를 상위로 올린다"는 진단을 세웠고, `scripts/run_temporal_ablation.py` 로 이 진단을
    검증했다(아래 Temporal Ablation 절 참고) — **부분적으로만 지지됨**: top-2 회수는
    개선(0.60→0.80)되지만 rank-1(nDCG@1=0.50)은 시점 표기만으로 교정되지 않았다.
- **임베딩 모델 교체 관찰(통제 실험 아님)**: 초기 영어 중심 임베딩(`all-MiniLM-L6-v2`)에서는
  한국어 게임 고유명사 검색이 무너졌다. 다국어 모델(`multilingual-e5-small`)로 교체하고 e5
  규약(`passage:`/`query:` 접두어)을 적용했다. **주의**: 모델을 바꾸면 학습 데이터·토크나이저·
  임베딩 차원·접두어 규약이 동시에 바뀌므로 이것은 단일 변인을 통제한 실험이 아니다. 정확히는
  **모델 교체 전후 문항별 검색 순위 변화를 비교해, 다국어 모델에서 gold 문서 회수가 개선됨을
  관찰한 것**이며, 개선에 어떤 변화 요인(학습 데이터/토크나이저/차원/접두어)이 얼마나 기여했는지는
  분리하지 않았다. (반대로 아래 Temporal Ablation 은 임베딩·k·seed·device 를 모두 고정하고
  문서 표현이라는 단일 변인만 바꾼 **진짜 통제 실험**이다 — 두 실험의 성격을 구분할 것.)
- **규칙 게이트가 outdated_trap을 안정적으로 처리**: 정정 성공 답변(정답 수치까지 명시)만
  `method=rule`로 `hallucination=no` 확정하고, 정정 실패/회피는 LLM으로 넘긴다.
- **judge 신뢰도** (`human_label` 대조, `results/judge_reliability.json`):
  - **판정 단위**: 15문항 × 축(correctness/faithfulness/hallucination)별 판정. correctness
    15건 + faithfulness 15건 + hallucination(outdated_trap 5문항만) 5건 = **총 35건**.
    이 35건을 판정 방법(rule/llm) 기준으로 다시 나누면 rule 25건 + llm 10건 = 35건으로,
    같은 35건을 축 기준/방법 기준 두 관점에서 나눈 것이지 별도 모집단이 아니다
    (코드 주석: `src/evaluation/judge_reliability.py` 상단, JSON: `judgment_structure` 필드).
  | 구분 | 사람과의 일치율 |
  | --- | --- |
  | correctness (15건 중) | 86.7% (13/15) |
  | faithfulness (15건 중) | 73.3% (11/15) |
  | hallucination (5건 중) | 80.0% (4/5) |
  | **rule (25건 중)** | **92.0% (23/25)** |
  | **llm (10건 중)** | **50.0% (5/10)** |
  | 전체 (35건 중) | 80.0% (28/35) |
  - hallucination 은 분모가 5뿐이라 1건만 틀려도 20%p 가 흔들린다 — pct 는 참고용, 항상
    분자/분모를 함께 볼 것(위 n=15 주의 참고).
- 즉 **소형 LLM judge(50%, 5/10)보다 규칙 판정(92%, 23/25)이 안정적**이라는 경향이 확인된다
  (다만 n=25/10 의 파일럿 표본이라 통계적 유의성은 별도로 검증하지 않았다). 하이브리드가
  규칙을 우선하고 애매한 경우만 LLM에 위임하는 설계의 근거.
- **`human_label` 재라벨링: 무엇이 고정이고 무엇을 보정했는지**
  - **고정**: 정답 근거(문서의 patch 이후 수치)와 `gold_answer`/`gold_answer_keywords` 는
    모델 출력과 무관하게 고정이다. 재라벨링 과정에서도 이 값은 건드리지 않았다.
  - **보정**: `human_label`(판정 라벨)은 "이 모델의 이 출력이 correctness/faithfulness/
    hallucination 기준을 충족했는가"에 대한 채점 결과다. 초기에는 실제 출력을 보지 않고
    "모델이 이상적으로 정정했을 것"이라 **가정**해 매겼는데, 이는 실제 출력과 어긋나 신뢰도
    수치를 오염시켰다. 실제로 생성된 답변을 보고 판정 라벨을 다시 매기는 **전수 점검으로
    보정**한 것이 현재 `data/questions/questions.json` 의 `human_label` 이다.
  - **한계**: 판정 라벨을 보정할 때 참고한 출력(`results/generation.json`)과 judge 신뢰도
    평가에 쓴 출력이 **분리되어 있지 않다**(dev/test split 없음). 같은 답변 세트를 보고
    라벨을 고치고, 그 라벨로 다시 신뢰도를 측정했으므로 위 일치율(특히 rule 92%)이
    과대추정될 위험이 있다. 이 한계는 아직 해소되지 않았다.
- **`no_context` 조건에서 발견된 fabrication(2건)**: `scripts/run_ablation.py` 의 `no_context`
  (closed-book) 조건에서, 모델이 **질문·문서 어디에도 없는 수치를 생성**한 사례를 2건
  발견했다(q006: 질문·문서 어디에도 없는 `140%`, q015: 질문·문서 어디에도 없는 `70초`).
  이는 이 벤치마크가 원래 겨냥한 **구버전 환각(outdated hallucination)**과 구별되는
  **fabrication** 유형이다 — outdated 환각은 "질문/문서에 있던 구버전 값을 사실처럼 반복"하는
  것이고, fabrication 은 "어디에도 없는 값을 지어내는" 것이다. 문항별 진단은
  `question_leaked_outdated_value` / `fabricated_value` / `judge_uncertain` 세 라벨을
  **비배타 다중 라벨**(`per_query[].labels`, 한 문항에 복수 적용 가능)로 기록한다(아래
  Ablation 절 참고).

## Ablation: 검색 오류 vs 답변 오류의 인과 분리
검색 지표와 답변 지표를 각각 재는 것만으로는 "틀린 원인"이 검색인지 생성인지 가르지 못한다.
`scripts/run_ablation.py` 는 **동일 생성 모델(seed 고정)** 로 15문항 각각을 세 조건에서
답하게 해 원인을 분리한다.
- `no_context` : 문서 없이 질문만(closed-book). groundedness 프롬프트는 "모른다"만 내므로,
  이 조건에서만 모델이 아는 지식으로 답하게 하는 별도 프롬프트를 쓴다.
- `gold_context` : `qrels` 정답 문서 강제 주입(검색이 완벽할 때의 상한).
- `retrieved_context` : 실제 top-k 검색 결과 주입(현재 시스템).

실행:
```bash
python -m scripts.run_ablation               # 전체 재생성 후 분석
python -m scripts.run_ablation --from-cache  # 저장된 답변/채점만 재분석(LLM 미호출)
```

### 진단 라벨
**시스템 1차 진단**(문항당 1개, 캐스케이드):
| 라벨 | 정의 |
| --- | --- |
| `ok` | gold·retrieved 모두 정답(시스템 정상). |
| `generation_failure` | gold 오답 **이고** 정답수치(post)조차 답변에 없음(진짜 생성 실패). |
| `judge_uncertain` | gold 답변에 정답수치는 포함됐으나 `correctness=no` 로 채점됨(채점 오판). |
| `retrieval_failure` | gold 정답, retrieved 오답(검색 문제). |
| `question_leaked_outdated_value` | outdated_trap 에서 `no_context` 답변이 **질문에 흘린 구버전 값을 그대로 반복**함. |

**비배타 진단 태그**(문항당 복수 적용 가능, `per_query[].labels`):
| 태그 | 정의 |
| --- | --- |
| `question_leaked_outdated_value` | 질문에 있던 구버전 값을 반복(질문 누출). |
| `fabricated_value` | 질문·문서 **어디에도 없는 수치를 생성**(누출과 구별되는 fabrication). |
| `judge_uncertain` | 내용상 정답이나 채점이 오판. |

### 진단을 왜곡한 두 오염과 보정
- **오염 1 — `parametric_outdated_bias` 폐기.** 이 벤치마크의 캐릭터·스킬은 **전부 가상 합성
  데이터**라, 모델이 사전(파라메트릭) 지식으로 구버전 수치를 알 수 없다(no_context에서
  fact·multi_hop 정답률이 **0/15** 인 것이 그 증거다). `no_context` 에 구버전 값이 나온 이유는
  **outdated_trap 질문 자체가 구버전 수치를 전제로 포함**하기 때문이다(질문이 정보를 흘림).
  따라서 라벨을 `question_leaked_outdated_value` 로 교체하고, `no_context` 답변이 **질문의
  수치를 반복(누출)** 한 건지 **질문·문서 어디에도 없는 새 수치를 지어낸(fabrication)** 건지
  문항별로 구분한다(두 라벨은 비배타 — 한 문항에 동시 적용 가능).
  - `question_leaked_outdated_value`: q003(100초)·q009(30초)·q012(25%)·q015(60초)
  - `fabricated_value`: q006(140%)·q015(70초) — 질문(40%/60초)에도 문서에도 없는 창작 수치.
- **오염 2 — `generation_failure` 오판 분리.** gold_context 답변이 내용상 정답 수치(post)와
  정정을 담았는데도 키워드 부분매칭 실패 → 소형 LLM judge 오판으로 `no` 가 된 케이스가 있다.
  gold 답변에 정답 수치가 문자열로 들어있는지 **규칙으로 재확인**해, 포함됐으면 `judge_uncertain`,
  정답 수치조차 없으면 진짜 `generation_failure` 로 남긴다. outdated_trap 에서 **judge 정답은
  1/5 였지만 규칙 '정답수치 포함'은 5/5** 로, 이는 이 프로젝트가 이미 측정한 LLM judge 신뢰도
  50% 문제와 정확히 일치한다.

### 결과(15문항, `--from-cache` 재분석)
- 조건×유형 교차표(judge 기준): `no_context` 0/15, `gold_context`·`retrieved_context` 모두 11/15
  (fact 5/5, multi_hop 5/5, outdated_trap 1/5).
- **`gold_context` == `retrieved_context` 이고 `retrieval_failure`=0** → 이 문항 셋에서 **검색은
  답변 오류의 원인이 아니다**(검색 랭킹의 불완전함이 답을 틀리게 만들지 않았다).
- 진단 라벨 분포 **before → after**:
  | 라벨 | before(오염) | after(보정) | 문항 |
  | --- | --- | --- | --- |
  | `ok` | 10 | 10 | q001,q002,q004,q005,q007,q008,q010,q011,q013,q014 |
  | `generation_failure` | 4 | **0** | — |
  | `judge_uncertain` | — | **4** | q003,q006,q009,q012 |
  | `retrieval_failure` | 0 | 0 | — |
  | `parametric_outdated_bias` | 1 | 폐기 | — |
  | `question_leaked_outdated_value` | — | **1** | q015 |
- 비배타 태그 분포: `question_leaked_outdated_value` 4건(q003,q009,q012,q015),
  `fabricated_value` 2건(q006,q015), `judge_uncertain` 4건(q003,q006,q009,q012).
- 결론: outdated_trap 의 "실패"는 검색 문제가 아니라 **(a) 질문이 구버전 값을 흘리는 설계상 특성**과
  **(b) 소형 LLM judge 의 채점 오판**이 겹친 것이다. 진짜 생성 실패(`generation_failure`)는 0건이다.

### `no_context` 조건이 실제로 드러낸 것
`no_context` 조건은 원래 모델의 **파라메트릭 지식**(구버전 편향의 기저선)을 재려던 것이었다.
그러나 캐릭터·스킬이 전부 가상 합성 데이터라 사전 지식 가설이 배제되면서(fact·multi_hop 정답률
**0/15**), 이 조건은 대신 두 가지를 드러냈다 — **(a)** 함정 질문이 구버전 값을 전제로 포함하는
설계 특성상, 문서가 없으면 모델은 **질문에 흘린 값을 그대로 반복**한다는 것(`question_leaked_outdated_value`),
**(b)** 문서가 없을 때 모델이 **존재하지 않는 수치를 창작**한다는 것(`fabricated_value`, 예: q006의
140%, q015의 70초). 후자는 이 벤치마크가 원래 겨냥한 **outdated hallucination**(구버전을 사실처럼
말하는 것)과 구별되는 **fabrication**(질문에도 문서에도 없는 값을 지어내는 것) 유형이다.

## Temporal Ablation: fact 낮은 검색 성능 진단의 검증
> **이 실험은 위 "임베딩 모델 교체 관찰"과 성격이 다르다.** 임베딩 모델 교체(MiniLM→e5)는
> 학습 데이터·토크나이저·차원·접두어 규약이 동시에 바뀌는 **비통제 관찰 비교**였다. 반면
> 아래 실험은 임베딩 모델·k·seed·device 를 모두 고정하고 **문서 표현이라는 단일 변인만**
> 바꾸는 **진짜 통제 실험**이다.

README 상 fact 유형의 낮은 검색 성능(recall@2=0.60)을 **"pre/post 문서 표면이 거의 같아
구버전(pre_patch)이 상위로 온다"** 로 진단했다. `scripts/run_temporal_ablation.py` 는 이 진단을
**처방 실험**으로 검증한다 — 임베딩 모델·검색 설정(k, seed, device)은 고정하고 **문서 표현만** 바꾼다.
문서 시점 표기(1차 처방)가 fact 의 recall@2 는 개선했지만 nDCG@1(rank-1)은 교정하지 못했는데,
그 원인을 **"fact 질문에 시점 단서가 없어 쿼리가 문서의 시점 마커를 당기지 못한다"**로 추정했고,
이를 검증하기 위해 **세 번째 조건(쿼리 side 처방)**을 추가했다.

세 조건(모두 임베딩 모델·k·seed·device 고정):
- `baseline` : 현재 corpus 그대로, 질문 그대로.
- `temporal_marked` : 각 문서 `content` 앞에 `period` 를 텍스트로 노출(문서 side 처방)
  (`post_patch → "[패치 이후 기준] "`, `pre_patch → "[패치 이전 기준] "`). 질문은 그대로.
- `temporal_marked_query_hint` : **문서는 `temporal_marked` 와 완전히 동일**(같은 인덱스 재사용).
  **검색 쿼리에만** `"[현재 기준] "` 힌트를 접두(쿼리 side 처방). `temporal_marked` 대비
  바뀌는 변인이 쿼리 텍스트 하나뿐이라, "쿼리 힌트 자체의 효과"를 "문서 표기 효과"와
  분리해서 볼 수 있다.

원본 `data/corpus/docs.json` 과 `data/questions/questions.json` 은 수정하지 않는다. 파생
코퍼스는 `results/temporal/*_corpus.json` 에, 쿼리 힌트 적용 전/후 텍스트는
`results/temporal/query_hint_questions.json` 에 투명하게 남긴다(검색 단계에서만 임시로 붙인
파생 텍스트일 뿐, 문항 원문은 그대로다).

```bash
python -m scripts.run_temporal_ablation   # 자체 색인 → 검색 → 평가(3조건)
```

**목적은 성능 개선이 아니라 진단/추정 검증이다.** 앞선 ablation 에서 `retrieval_failure=0` 으로,
fact 의 낮은 recall@2 는 답변 오류로 이어지지 않았음이 이미 확인됐다. 각 처방으로 fact 지표가
개선되면 해당 진단/추정이 옳고, 아니면 다른 원인이 있다는 뜻이므로 그대로 기록한다.

### 결과 (임베딩 `multilingual-e5-small` 고정, 3조건)
| 유형 | recall@2 baseline | recall@2 temporal_marked | recall@2 +query_hint | nDCG@1 baseline | nDCG@1 temporal_marked | nDCG@1 +query_hint |
| --- | --- | --- | --- | --- | --- | --- |
| fact | 6/10 (0.60) | **8/10 (0.80)** | 8/10 (0.80) | 2.50/5 (0.50) | 2.50/5 (0.50) | **2.50/5 (0.50)** |
| multi_hop | 10/10 (1.00) | 10/10 (1.00) | 10/10 (1.00) | 5.00/5 (1.00) | 5.00/5 (1.00) | 5.00/5 (1.00) |
| outdated_trap | 10/10 (1.00) | 10/10 (1.00) | 10/10 (1.00) | 4.50/5 (0.90) | 4.00/5 (0.80) | 4.00/5 (0.80) |
| ALL | 26/30 (0.87) | 28/30 (0.93) | 28/30 (0.93) | 12.00/15 (0.80) | 11.50/15 (0.77) | 11.50/15 (0.77) |
- recall@2 셀 = top-2 내 회수 gold / 전체 gold, nDCG@1 셀 = 문항별 nDCG@1 합 / 문항 수. `temporal_marked`
→`+query_hint` 사이 모든 셀의 값이 정확히 동일하다 — 이 자체가 결론 2 의 핵심 증거다.

### 결론 1 — 문서 시점 표기 진단: **부분 지지**
- fact **recall@2 는 +0.20 개선**됐다(0.60→0.80) → 시점 표기가 두 번째 gold 문서를 top-2 로
  더 끌어올린다.
- 그러나 fact **nDCG@1(rank-1)은 0.50 으로 불변**이다 → **최상위(rank-1)에는 여전히 구버전
  (pre_patch)이 앉아 있다.** "표면이 같아 구버전이 *상위*로 온다"는 진단은 top-2 관점에서만
  부분 검증되고, "*최상위*로 온다"는 핵심 주장은 문서 표기만으로는 교정되지 않았다.
- 부작용으로 **outdated_trap nDCG@1 이 -0.10 회귀**했다(ALL nDCG@1 -0.03).

### 결론 2 — 쿼리 시점 힌트 추정: **반증됨**
결론 1의 rank-1 미교정 원인을 "fact 질문에 시점 단서가 없어 쿼리가 문서의 마커를 당기지
못한다"로 추정하고, 문서 표기(`temporal_marked`)는 고정한 채 쿼리에만 `"[현재 기준] "` 힌트를
추가해(`temporal_marked_query_hint`) 검증했다.
- 실제로 쿼리 임베딩은 바뀌었다(검색 점수(score)는 `temporal_marked` 대비 달라졌고 일부 문항은
  2위 이하 순위도 바뀌었다) — 힌트가 적용되지 않은 게 아니다.
- 그런데도 **fact 의 recall@2·nDCG@1 은 소수점 단위까지 완전히 동일**하다(0.80/0.80,
  0.50/0.50) → rank-1 승자(pre_patch 문서)가 그대로 유지된다.
- **결론: "쿼리에 시점 단서가 없어서" 라는 추정은 틀렸다.** 쿼리 힌트를 추가해도 rank-1이
  바뀌지 않았으므로, rank-1 교정 실패의 원인은 쿼리 측 시점 단서 부재가 아니라 다른 요인일
  가능성이 높다 — 예를 들어 이 임베딩 모델이 "…이름은 무엇인가?" 류의 fact 질의에서는 짧은
  시점 힌트 토큰보다 스킬명·캐릭터명 등 개체명 매칭에 훨씬 더 크게 의존해, 문서 쪽 시점 마커
  (`[패치 이후 기준]`)와의 어휘 중첩만으로는 그 지배력을 못 넘어서는 것으로 보인다.
  이 벤치마크는 이 가능성을 추가로 검증하지 않았고, 사실 그대로만 기록한다.
- 부작용 확인: `temporal_marked`→`+query_hint` 사이 어떤 유형에서도 회귀가 없었다(모든 셀 동일).
- 산출물: `results/temporal_ablation.json`(`diagnosis_1_doc_marker`/`estimate_2_query_hint` 필드에
  두 판정 모두 저장), `results/temporal/*`(파생 코퍼스·쿼리 힌트 텍스트·조건별 검색·평가).
