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
        _mk("KGW", 2), _mk("Unigram", 2), _mk("SWEET", 2),
        _mk("KGW", 1), _mk("Unigram", 1), _mk("SWEET", 1),
    ]
}

# Detectors applied to each watermarked generation config.
# EWD reuses the KGW generations with the same key/γ/δ.
DETECTORS_FOR = {
    "KGW": ["KGW", "EWD"],
    "Unigram": ["Unigram"],
    "SWEET": ["SWEET"],
}
