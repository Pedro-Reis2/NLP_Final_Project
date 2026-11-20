"""Load MarkLLM watermark objects from our config registry."""
from __future__ import annotations

import json

from markllm.utils.transformers_config import TransformersConfig
from markllm.watermark.auto_watermark import AutoWatermark

from .config import OUT, WMConfig


def load_watermark(cfg: WMConfig, model, tok, method: str | None = None):
    """MarkLLM object for cfg (or for detector `method` with cfg's key/γ/δ)."""
    c = cfg if method is None else WMConfig(cfg.config_id, method, cfg.delta, cfg.gamma,
                                            cfg.hash_key, cfg.seed, cfg.extra)
    path = OUT / "manifests" / "markllm" / f"{cfg.config_id}__{c.method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c.markllm_algorithm_config(), indent=2))
    tcfg = TransformersConfig(model=model, tokenizer=tok,
                              vocab_size=model.config.vocab_size,  # logits dim (151936), not len(tok)
                              device="cuda")
    return AutoWatermark.load(c.method, algorithm_config=str(path), transformers_config=tcfg)
