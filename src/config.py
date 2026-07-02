# ---------------------------------------------------------------------------
# 역할: config.json 을 로드/검증하여 전 모듈이 공유하는 설정 객체로 제공.
# 입력:
#   - config.json (k, embedding_model, gen_model, judge_model, seed,
#                  judge_mode[rule/llm/hybrid], device)
# 출력:
#   - 검증된 Config 데이터클래스
#   - seed 고정(재현성) 처리
#   - 프로젝트 표준 경로 상수(PROJECT_ROOT, DATA/RESULTS, CORPUS/QUESTIONS 등)
# 의존관계:
#   - 표준 라이브러리(json). 거의 모든 하위 모듈이 이 설정을 참조.
# 원칙: 외부 API 값 금지(로컬 모델/Ollama 만 허용).
# ---------------------------------------------------------------------------

import json
import os
import random
from dataclasses import dataclass

# --- 프로젝트 표준 경로 (config.py 는 src/ 아래에 위치) ----------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus", "docs.json")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions", "questions.json")
QRELS_PATH = os.path.join(DATA_DIR, "qrels", "qrels.json")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

VALID_JUDGE_MODES = {"rule", "llm", "hybrid"}
VALID_DEVICES = {"cpu", "cuda"}


@dataclass
class Config:
    k: int
    embedding_model: str
    gen_model: str
    judge_model: str
    seed: int
    judge_mode: str
    device: str

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            k=int(d["k"]),
            embedding_model=str(d["embedding_model"]),
            gen_model=str(d["gen_model"]),
            judge_model=str(d["judge_model"]),
            seed=int(d["seed"]),
            judge_mode=str(d["judge_mode"]),
            device=str(d["device"]),
        )


def validate_config(cfg: dict) -> None:
    """필수 필드/허용 값 검증. 위반 시 ValueError."""
    required = [
        "k", "embedding_model", "gen_model", "judge_model",
        "seed", "judge_mode", "device",
    ]
    missing = [f for f in required if f not in cfg]
    if missing:
        raise ValueError(f"config 필수 필드 누락: {missing}")
    if int(cfg["k"]) < 1:
        raise ValueError("k 는 1 이상이어야 합니다.")
    if cfg["judge_mode"] not in VALID_JUDGE_MODES:
        raise ValueError(f"judge_mode 는 {VALID_JUDGE_MODES} 중 하나여야 합니다.")
    if cfg["device"] not in VALID_DEVICES:
        raise ValueError(f"device 는 {VALID_DEVICES} 중 하나여야 합니다.")


def set_global_seed(seed: int) -> None:
    """random/numpy/torch 시드 고정(재현성)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    validate_config(raw)
    cfg = Config.from_dict(raw)
    set_global_seed(cfg.seed)
    return cfg
