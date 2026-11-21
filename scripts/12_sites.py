"""QP3, second half of the pitch question: "how many tokens OR MARKING SITES does each method need
for 80% TPR at 1% FPR?"

  python scripts/12_sites.py count      # GPU: per generation, the positions each detector scores
  python scripts/12_sites.py analyze    # outputs/tables/sites.csv, min_sites.csv

Marking sites per detector (the positions that carry evidence):
  KGW, EWD-unweighted   every token after the 1-token prefix          (n_tokens - 1)
  Unigram               every token                                   (n_tokens)
  SWEET                 tokens whose entropy exceeds the 0.9 threshold (MarkLLM's own mask)
  EWD                   effective sample size of the entropy weights, (sum w)^2 / sum w^2
  Multicanal            readable votes (2 per eligible name, 1 per structural site)
Output: outputs/detect/sites/<config_id>.jsonl rows {id, n_tokens, sites_SWEET | sites_EWD}.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm.config import DETECTORS_FOR, GEN_CONFIGS, OUT  # noqa: E402
from wm.io import append_jsonl, done_ids, get_logger, read_jsonl  # noqa: E402

log = get_logger("sites")
GPU_CFGS = ["ewd_d2", "sweet_d2", "ewd_d4", "sweet_d4"]
BINS = [0, 10, 20, 30, 40, 60, 80, 120, 160, 200, 300, 10_000]


def count():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wm.config import MODEL_ID
    det = importlib.import_module("05_detect")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    scorer = det.Scorer(model, tok)
    for cfg in GPU_CFGS:
        m = GEN_CONFIGS[cfg].method
        wm = scorer._get(m)
        path = OUT / "detect" / "sites" / f"{cfg}.jsonl"
        done = done_ids(path)
        rows = []
        for g in read_jsonl(OUT / "generations" / f"{cfg}.jsonl"):
            if g["id"] in done:
                continue
            ids = tok(g["solution"], return_tensors="pt", add_special_tokens=False)["input_ids"][0].cuda()
            r = {"id": g["id"], "n_tokens": int(len(ids))}
            if len(ids) < 2:
                r[f"sites_{m}"] = 0.0
            else:
                with torch.no_grad():
                    ent = wm.utils.calculate_entropy(wm.config.generation_model, ids)
                _, _, w = wm.utils.score_sequence(ids, ent)
                w = [x for x in w[wm.config.prefix_length:] if x != -1]
                if m == "SWEET":
                    r["sites_SWEET"] = float(sum(1 for x in w if x == 1))
                else:
                    s1, s2 = float(sum(w)), float(sum(x * x for x in w))
                    r["sites_EWD"] = s1 * s1 / s2 if s2 > 0 else 0.0
            rows.append(r)
        append_jsonl(path, rows)
        log.info(f"[{cfg}] +{len(rows)}")


def analyze():
    import numpy as np
    import pandas as pd
    an = importlib.import_module("06_analyze")
    rows, mrows = [], []
    for cfg, c in GEN_CONFIGS.items():
        if c.method is None:
            continue
        for d in DETECTORS_FOR[c.method]:
            neg = an.zfile(OUT, d, "neg").z.to_numpy()
            thr = an.emp_threshold(neg)
            z = an.zfile(OUT, d, f"gen:{cfg}")
            if d in ("SWEET", "EWD"):
                s = pd.DataFrame(read_jsonl(OUT / "detect" / "sites" / f"{cfg}.jsonl"))
                if s.empty:
                    continue
                z = z.merge(s[["id", f"sites_{d}"]], on="id").rename(columns={f"sites_{d}": "sites"})
            else:
                z = z.assign(sites=z.n_tokens - (0 if d == "Unigram" else 1))
            z = z.assign(bin=pd.cut(z.sites, BINS, right=False))
            best = None
            for b, sub in z.groupby("bin", observed=True):
                tpr = float((sub.z > thr).mean())
                rows.append({"config_id": cfg, "delta": c.delta, "detector": d, "sites_bin": str(b),
                             "bin_lo": b.left, "n": len(sub), "TPR@1%FPR": tpr,
                             "median_tokens": float(sub.n_tokens.median())})
                if best is None and len(sub) >= 10 and tpr >= 0.8:
                    best = b.left
            mrows.append({"config_id": cfg, "delta": c.delta, "detector": d,
                          "median_sites": float(z.sites.median()),
                          "median_tokens_per_site": float((z.n_tokens / z.sites.replace(0, np.nan)).median()),
                          "min_sites_TPR80": best})
    mc = OUT / "detect" / "Multicanal" / "gen__mcw.jsonl"
    if mc.exists():
        neg = an.zfile(OUT, "Multicanal", "neg").z.to_numpy()
        thr = an.emp_threshold(neg)
        z = an.zfile(OUT, "Multicanal", "gen:mcw").assign(sites=lambda x: x.n_votes)
        z = z.assign(bin=pd.cut(z.sites, [0, 1, 2, 4, 8, 16, 32, 10_000], right=False))
        best = None
        for b, sub in z.groupby("bin", observed=True):
            tpr = float((sub.z > thr).mean())
            rows.append({"config_id": "mcw", "delta": np.nan, "detector": "Multicanal", "sites_bin": str(b),
                         "bin_lo": b.left, "n": len(sub), "TPR@1%FPR": tpr, "median_tokens": float(sub.n_tokens.median())})
            if best is None and len(sub) >= 10 and tpr >= 0.8:
                best = b.left
        mrows.append({"config_id": "mcw", "delta": np.nan, "detector": "Multicanal", "median_sites": float(z.sites.median()),
                      "median_tokens_per_site": float((z.n_tokens / z.sites.replace(0, np.nan)).median()),
                      "min_sites_TPR80": best})
    pd.DataFrame(rows).to_csv(OUT / "tables" / "sites.csv", index=False)
    pd.DataFrame(mrows).to_csv(OUT / "tables" / "min_sites.csv", index=False)
    print(pd.DataFrame(mrows).round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["count", "analyze"])
    {"count": count, "analyze": analyze}[ap.parse_args().step]()
    log.info("SITES STEP DONE")


if __name__ == "__main__":
    main()
