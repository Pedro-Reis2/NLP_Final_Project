"""Watermark detection (z-scores) with MarkLLM detectors. Resumable.

Targets:
  neg                    human negatives (outputs/negatives/negatives.jsonl, field `code`)
  gen:<config_id>        generations (field `solution`)
  atk:<config_id>        attacked generations (outputs/attacks/<config_id>.jsonl, field `attacked`)

Detectors per target: for watermarked configs, DETECTORS_FOR[method]; for `neg`
and `gen:nowm`, every detector (they are the negatives / unwatermarked control).

  python scripts/05_detect.py --targets neg,gen:nowm,gen:kgw_d2 [--test] [--limit N]

Output: outputs/detect/<detector>/<target>.jsonl rows {id, z, n_tokens}
(<target> with ':' replaced by '__'). Detection does not depend on δ, so one
detector object (γ=0.5, MarkLLM default key) serves every config.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from wm.config import DETECTORS_FOR, GAMMA, GEN_CONFIGS, MODEL_ID, OUT  # noqa: E402
from wm.io import append_jsonl, done_ids, get_logger, read_jsonl, write_manifest  # noqa: E402
from wm.markllm_utils import load_watermark  # noqa: E402

ALL_DETECTORS = ["KGW", "EWD", "EWD-unweighted", "Unigram", "SWEET"]
FIELD = {"neg": "code", "gen": "solution", "atk": "attacked"}


def target_rows(target: str, root: Path):
    kind, _, cfg = target.partition(":")
    if kind == "neg":
        rows = read_jsonl(root / "negatives" / "negatives.jsonl")
    elif kind == "gen":
        rows = read_jsonl(root / "generations" / f"{cfg}.jsonl")
    elif kind == "atk":
        rows = [r for r in read_jsonl(root / "attacks" / f"{cfg}.jsonl") if r.get("attacked")]
    else:
        raise ValueError(target)
    return [{"id": r["id"], "text": r[FIELD[kind]]} for r in rows]


def detectors_for(target: str):
    kind, _, cfg = target.partition(":")
    if kind == "neg" or cfg == "nowm":
        return ALL_DETECTORS
    return DETECTORS_FOR[GEN_CONFIGS[cfg].method]


class Scorer:
    """Wraps MarkLLM detectors; EWD-unweighted reuses EWD's green flags."""

    def __init__(self, model, tok):
        self.model, self.tok, self.wm = model, tok, {}

    def _get(self, method):
        if method not in self.wm:
            self.wm[method] = load_watermark(GEN_CONFIGS[f"{method.lower()}_d2"], self.model, self.tok)
        return self.wm[method]

    def score(self, detector: str, text: str) -> float | None:
        if detector == "EWD-unweighted":
            wm = self._get("EWD")
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].cuda()
            ent = wm.utils.calculate_entropy(wm.config.generation_model, ids)
            _, flags, _ = wm.utils.score_sequence(ids, ent)
            scored = [f for f in flags if f != -1]
            T = len(scored)
            if T == 0:
                return None
            # plain KGW z-test (Kirchenbauer et al. 2023) on EWD's green flags
            return (sum(f == 1 for f in scored) - GAMMA * T) / math.sqrt(T * GAMMA * (1 - GAMMA))
        z = self._get(detector).detect_watermark(text, return_dict=True)["score"]
        return float(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--detectors", default=None, help="restrict to these detectors")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="test subfolder (with --test)")
    args = ap.parse_args()
    log = get_logger("detect")
    root = OUT / "test" / args.tag if args.test else OUT

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    scorer = Scorer(model, tok)
    write_manifest("detect", {"step": "detect", "gamma": GAMMA, "targets": args.targets,
                              "detector_config": GEN_CONFIGS["kgw_d2"].to_dict()})

    for target in args.targets.split(","):
        rows = target_rows(target, root)[: args.limit]
        dets = detectors_for(target)
        if args.detectors:
            dets = [d for d in dets if d in args.detectors.split(",")]
        for det in dets:
            out = root / "detect" / det / f"{target.replace(':', '__')}.jsonl"
            done = done_ids(out)
            todo = [r for r in rows if r["id"] not in done]
            if not todo:
                continue
            t0, buf = time.time(), []
            for i, r in enumerate(todo, 1):
                n = len(tok(r["text"], add_special_tokens=False).input_ids)
                try:
                    z = scorer.score(det, r["text"]) if n >= 2 else None
                except Exception as e:  # too-short / degenerate inputs
                    log.warning(f"{det} {target} {r['id']}: {type(e).__name__}: {e}")
                    z = None
                buf.append({"id": r["id"], "z": z, "n_tokens": n})
                if len(buf) >= 50 or i == len(todo):
                    append_jsonl(out, buf)
                    buf = []
                    log.info(f"[{det} {target}] {i}/{len(todo)}  {(time.time() - t0) / i:.3f}s/text")
            zs = [r["z"] for r in read_jsonl(out) if r["z"] is not None]
            if zs:
                m = sum(zs) / len(zs)
                sd = (sum((z - m) ** 2 for z in zs) / max(1, len(zs) - 1)) ** 0.5
                log.info(f"[{det} {target}] n={len(zs)} mean z={m:.2f} sd={sd:.2f} "
                         f"frac>2.33={sum(z > 2.326 for z in zs) / len(zs):.3f}")


if __name__ == "__main__":
    main()
