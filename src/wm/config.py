"""Single source of truth for experiment configs.

Every generation/detection run is identified by a config_id (e.g. "kgw_d2").
All hyper-parameters that matter for reproducibility live here and are
written to outputs/manifests/<config_id>.json by each run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
LOGS = ROOT / "logs"
DATA = ROOT / "data"
FIGS = ROOT / "figures"

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SEED = 42
HASH_KEY = 15485863  # MarkLLM default
GAMMA = 0.5
SWEET_ENTROPY_THRESHOLD = 0.9  # MarkLLM default
PREFIX_LENGTH = 1  # KGW / SWEET / EWD

DECODING = dict(
    do_sample=False,  # greedy
    max_new_tokens=512,
    num_return_sequences=1,
)

DATASETS = ("humaneval", "mbpp")

# Harness artifacts, excluded everywhere -> 163 + 377 = 540 problems:
# - HumanEval/32 (find_zero): canonical solution passes all inputs outside the
#   sandbox, but EvalPlus's sandbox fails it before the first test (WSL).
# - Mbpp/255: canonical solution fails the extended tests after a pure `black`
#   reformat (resource-heavy inputs; flaky in the sandbox).
EXCLUDED_TASKS = {"HumanEval/32", "Mbpp/255"}

# Tasks whose extended tests are memory/time heavy: canonical solutions fail
# them when many test processes run at once (even after pure reformatting with
# black). Failures on these are re-tested at low parallelism before being recorded.
RESOURCE_SENSITIVE_TASKS = {"HumanEval/15", "HumanEval/100", "HumanEval/130",
                            "Mbpp/84", "Mbpp/255", "Mbpp/392", "Mbpp/630"}


@dataclass(frozen=True)
class WMConfig:
    config_id: str
    method: str | None  # MarkLLM algorithm name, None = unwatermarked
    delta: float | None
    gamma: float | None = GAMMA
    hash_key: int = HASH_KEY
    seed: int = SEED
    extra: dict = field(default_factory=dict)

    def markllm_algorithm_config(self) -> dict:
        """Dict written to a JSON file and passed to AutoWatermark.load."""
        assert self.method is not None
        cfg = {
            "algorithm_name": self.method,
            "gamma": self.gamma,
            "delta": self.delta,
            "hash_key": self.hash_key,
            "z_threshold": 4.0,  # unused: we set thresholds empirically
        }
        if self.method in ("KGW", "SWEET", "EWD"):
            cfg["prefix_length"] = PREFIX_LENGTH
        if self.method == "KGW":
            cfg.update(f_scheme="time", window_scheme="left")
        if self.method == "SWEET":
            cfg["entropy_threshold"] = SWEET_ENTROPY_THRESHOLD
        cfg.update(self.extra)
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decoding"] = DECODING
        d["model_id"] = MODEL_ID
        return d


def _mk(method: str | None, delta: float | None) -> WMConfig:
    name = "nowm" if method is None else f"{method.lower()}_d{int(delta)}"
    return WMConfig(config_id=name, method=method, delta=delta,
                    gamma=None if method is None else GAMMA)


# Generation configs, in run order (δ=2 first; δ=1 is droppable)
GEN_CONFIGS: dict[str, WMConfig] = {
    c.config_id: c
    for c in [
        _mk(None, None),
        _mk("KGW", 2), _mk("EWD", 2), _mk("Unigram", 2), _mk("SWEET", 2),
        # δ=4 replaces the planned δ=1 (user decision, Day 1): a 40-problem probe gave
        # mean z ≈ 1.3 at δ=2, so δ=1 would leave ~0% TPR before any attack.
        _mk("KGW", 4), _mk("EWD", 4), _mk("Unigram", 4), _mk("SWEET", 4),
    ]
}

# Detectors applied to each watermarked generation config.
# MarkLLM's KGW seeds its green list with hash_key*prf[prev] while EWD/SWEET use
# hash_key*prev, so EWD cannot detect MarkLLM-KGW text. The EWD configs are
# generated with MarkLLM's EWD class (same KGW soft-red-list biasing, EWD hash)
# and scored twice: entropy-weighted (EWD) and unweighted ("EWD-unweighted",
# the plain KGW z-test on the same green flags) -> paired comparison.
DETECTORS_FOR = {
    "KGW": ["KGW"],
    "EWD": ["EWD", "EWD-unweighted"],
    "Unigram": ["Unigram"],
    "SWEET": ["SWEET"],
}
