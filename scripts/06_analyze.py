"""All metrics -> outputs/tables/*.csv (+ summary.md for RESULTS.md).

  python scripts/06_analyze.py [--test --tag probe_greedy]

Tables
  pass_at_1.csv      per config x dataset: pass@1 base (original tests) vs plus (extended), gap
  thresholds.csv     per detector: empirical z threshold at 1% FPR on human negatives
  detection.csv      per config x detector: TPR@1%FPR (empirical / analytic z=2.326), AUROC,
                     on all generations and on passing ones; FPR on unwatermarked generations
  attacks.csv        per config x detector x attack: valid n, survival, TPR on attacked vs same
                     tasks unattacked, AUROC, mean fraction of tokens changed
  length.csv         per config x detector x length bin: TPR (unattacked)
  min_length.csv     smallest length bin with TPR >= 0.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from wm.attacks import ATTACKS  # noqa: E402
from wm.config import DETECTORS_FOR, GEN_CONFIGS, OUT  # noqa: E402
from wm.io import read_jsonl  # noqa: E402

FPR = 0.01
Z_ANALYTIC = 2.326
LEN_BINS = [0, 75, 100, 125, 150, 200, 300, 10_000]


def zfile(root, det, target):
    df = pd.DataFrame(read_jsonl(root / "detect" / det / f"{target.replace(':', '__')}.jsonl"))
    return df if df.empty else df.dropna(subset=["z"])


def emp_threshold(neg_z: np.ndarray) -> float:
    """Smallest threshold t with mean(neg > t) <= FPR."""
    s = np.sort(neg_z)
    k = int(np.floor(FPR * len(s)))  # allowed false positives
    return float(s[len(s) - k - 1]) if len(s) > k else float("inf")


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    root = OUT / "test" / args.tag if args.test else OUT
    tdir = root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in GEN_CONFIGS if (root / "generations" / f"{c}.jsonl").exists()]

    # ---------------------------------------------------------------- pass@1
    rows = []
    for c in cfgs:
        ev = pd.DataFrame(read_jsonl(root / "evals" / f"{c}.jsonl"))
        if ev.empty:
            continue
        for ds, sub in list(ev.groupby("dataset")) + [("all", ev)]:
            rows.append({"config_id": c, "method": GEN_CONFIGS[c].method or "none",
                         "delta": GEN_CONFIGS[c].delta, "dataset": ds, "n": len(sub),
                         "pass@1_base": sub.pass_base.mean(), "pass@1_plus": sub.pass_plus.mean(),
                         "gap": sub.pass_base.mean() - sub.pass_plus.mean()})
    p1 = pd.DataFrame(rows)
    if not p1.empty:
        base = p1[p1.config_id == "nowm"].set_index("dataset")
        p1["drop_base_vs_nowm"] = p1.apply(lambda r: base.loc[r.dataset, "pass@1_base"] - r["pass@1_base"]
                                           if r.dataset in base.index else np.nan, axis=1)
        p1["drop_plus_vs_nowm"] = p1.apply(lambda r: base.loc[r.dataset, "pass@1_plus"] - r["pass@1_plus"]
                                           if r.dataset in base.index else np.nan, axis=1)
    p1.to_csv(tdir / "pass_at_1.csv", index=False)

    # ------------------------------------------------------------ thresholds
    dets = sorted({d for c in cfgs if GEN_CONFIGS[c].method for d in DETECTORS_FOR[GEN_CONFIGS[c].method]})
    thr, negz = {}, {}
    trows = []
    for d in dets:
        neg = zfile(root, d, "neg")
        if neg.empty:
            continue
        negz[d] = neg.z.to_numpy()
        thr[d] = emp_threshold(negz[d])
        trows.append({"detector": d, "n_neg": len(neg), "neg_mean_z": negz[d].mean(), "neg_sd_z": negz[d].std(),
                      "neg_frac_gt_2.326": (negz[d] > Z_ANALYTIC).mean(), "threshold_1pct": thr[d]})
    pd.DataFrame(trows).to_csv(tdir / "thresholds.csv", index=False)

    # ---------------------------------------------------------- detection
    drows, lrows, mrows = [], [], []
    for c in cfgs:
        m = GEN_CONFIGS[c].method
        if m is None:
            continue
        ev = pd.DataFrame(read_jsonl(root / "evals" / f"{c}.jsonl"))
        passing = set(ev[ev.pass_plus].id) if not ev.empty else set()
        for d in DETECTORS_FOR[m]:
            if d not in thr:
                continue
            pos = zfile(root, d, f"gen:{c}")
            if pos.empty:
                continue
            nowm = zfile(root, d, "gen:nowm")
            pp = pos[pos.id.isin(passing)]
            drows.append({
                "config_id": c, "method": m, "delta": GEN_CONFIGS[c].delta, "detector": d,
                "n": len(pos), "mean_z": pos.z.mean(), "median_tokens": pos.n_tokens.median(),
                "TPR@1%FPR": (pos.z > thr[d]).mean(), "TPR@z2.326": (pos.z > Z_ANALYTIC).mean(),
                "AUROC": auroc(pos.z, negz[d]),
                "n_passing": len(pp), "TPR@1%FPR_passing": (pp.z > thr[d]).mean() if len(pp) else np.nan,
                "FPR_nowm_gen": (nowm.z > thr[d]).mean() if not nowm.empty else np.nan,
                "AUROC_vs_nowm_gen": auroc(pos.z, nowm.z) if not nowm.empty else np.nan,
            })
            # length bins (QP3), unattacked
            pos = pos.assign(bin=pd.cut(pos.n_tokens, LEN_BINS, right=False))
            best = None
            for b, sub in pos.groupby("bin", observed=True):
                tpr = (sub.z > thr[d]).mean()
                lrows.append({"config_id": c, "detector": d, "len_bin": str(b), "bin_lo": b.left,
                              "n": len(sub), "TPR@1%FPR": tpr})
                if best is None and len(sub) >= 10 and tpr >= 0.8:
                    best = b.left
            mrows.append({"config_id": c, "detector": d, "min_tokens_TPR80": best})
    pd.DataFrame(drows).to_csv(tdir / "detection.csv", index=False)
    pd.DataFrame(lrows).to_csv(tdir / "length.csv", index=False)
    pd.DataFrame(mrows).to_csv(tdir / "min_length.csv", index=False)

    # ------------------------------------------------------------- attacks
    arows = []
    for c in cfgs:
        m = GEN_CONFIGS[c].method
        if m is None:
            continue
        atk = pd.DataFrame(read_jsonl(root / "attacks" / f"{c}.jsonl"))
        if atk.empty:
            continue
        for d in DETECTORS_FOR[m]:
            if d not in thr:
                continue
            za = zfile(root, d, f"atk:{c}")
            zg = zfile(root, d, f"gen:{c}").set_index("id")
            if za.empty:
                continue
            za = za.set_index("id")
            for a in ATTACKS:
                sub = atk[atk.attack == a]
                n_orig = len(sub)  # originals that passed plus tests
                valid = sub[(sub.status == "ok") & (sub.pass_plus == True)]  # noqa: E712
                valid = valid[valid.id.isin(za.index)]
                if valid.empty:
                    continue
                z_att = za.loc[valid.id, "z"].to_numpy()
                z_orig = zg.loc[zg.index.intersection(valid.task_id), "z"].to_numpy()
                arows.append({
                    "config_id": c, "method": m, "delta": GEN_CONFIGS[c].delta, "detector": d, "attack": a,
                    "n_passing_orig": n_orig, "n_valid": len(valid), "survival": len(valid) / max(1, n_orig),
                    "TPR_orig_same_tasks": (z_orig > thr[d]).mean(), "TPR_attacked": (z_att > thr[d]).mean(),
                    "AUROC_attacked": auroc(z_att, negz[d]), "mean_z_orig": z_orig.mean(),
                    "mean_z_attacked": z_att.mean(), "mean_frac_tok_changed": valid.frac_tok_changed.mean(),
                })
    pd.DataFrame(arows).to_csv(tdir / "attacks.csv", index=False)

    # -------------------------------------------------------------- summary
    def md(df, cols=None):
        if df.empty:
            return "_no data_\n"
        df = df[cols] if cols else df
        return df.to_markdown(index=False, floatfmt=".3f") + "\n"

    with open(tdir / "summary.md", "w", encoding="utf-8") as f:
        f.write("### pass@1\n" + md(p1) + "\n### thresholds\n" + md(pd.DataFrame(trows))
                + "\n### detection (unattacked)\n" + md(pd.DataFrame(drows))
                + "\n### attacks\n" + md(pd.DataFrame(arows)) + "\n### min length for TPR>=0.8\n"
                + md(pd.DataFrame(mrows)))
    print((tdir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
