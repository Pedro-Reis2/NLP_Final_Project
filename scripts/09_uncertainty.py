"""Bootstrap 95% confidence intervals for the headline numbers -> outputs/tables/ci_*.csv

  python scripts/09_uncertainty.py [--B 2000]

Percentile bootstrap, B resamples, seed 42. Units resampled with replacement:
  - problems, for pass@1 (paired across configs: the same problem ids in every config)
  - human negatives AND watermarked texts, for TPR@1%FPR: the 1%-FPR threshold is
    re-estimated on every negative resample, so its own uncertainty is included
  - tasks, for attacked vs unattacked TPR (paired: same tasks before/after)

Tables
  ci_pass_at_1.csv  per config: pass@1 base/plus with CI; drop vs no watermark (paired) with CI;
                    "extra drop" = drop on extended tests - drop on original tests (QP1)
  ci_detection.csv  per config x detector: TPR@1%FPR with CI; for EWD configs also the
                    paired difference EWD - EWD-unweighted on the same texts
  ci_attacks.csv    per config x detector x attack: TPR unattacked / attacked with CIs and the
                    paired change attacked - unattacked; for nc_* attacks also the change vs
                    comment-stripped code (attacked - stripped)
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wm.attacks import ATTACKS  # noqa: E402
from wm.config import DETECTORS_FOR, GEN_CONFIGS, OUT  # noqa: E402
from wm.io import read_jsonl  # noqa: E402

an = importlib.import_module("06_analyze")
FPR = an.FPR


def ci(a):
    """Point estimate is computed by the caller; this returns the 2.5/97.5 percentiles."""
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def boot_thresholds(negz: np.ndarray, B: int, rng) -> np.ndarray:
    """1%-FPR threshold on each of B resamples of the negatives (same rule as 06_analyze)."""
    n = len(negz)
    k = int(np.floor(FPR * n))
    s = np.sort(negz[rng.integers(0, n, size=(B, n))], axis=1)
    return s[:, n - k - 1]


def boot_tpr(z: np.ndarray, thr_b: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """TPR per resample: z resampled by idx (B x n), compared with that resample's threshold."""
    return (z[idx] > thr_b[:, None]).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    B, rng = args.B, np.random.default_rng(42)
    root, tdir = OUT, OUT / "tables"
    cfgs = [c for c in GEN_CONFIGS if (root / "generations" / f"{c}.jsonl").exists()]

    # --------------------------------------------------------------- pass@1
    ev = {c: pd.DataFrame(read_jsonl(root / "evals" / f"{c}.jsonl")).set_index("id") for c in cfgs}
    ids = sorted(set.intersection(*(set(e.index) for e in ev.values())))
    pidx = rng.integers(0, len(ids), size=(B, len(ids)))
    arr = {c: (ev[c].loc[ids, "pass_base"].to_numpy(float), ev[c].loc[ids, "pass_plus"].to_numpy(float))
           for c in cfgs}
    b0, p0 = arr["nowm"]
    rows = []
    for c in cfgs:
        b, p = arr[c]
        r = {"config_id": c, "method": GEN_CONFIGS[c].method or "none", "delta": GEN_CONFIGS[c].delta,
             "n": len(ids), "pass@1_base": b.mean(), "pass@1_plus": p.mean()}
        r["base_lo"], r["base_hi"] = ci(b[pidx].mean(1))
        r["plus_lo"], r["plus_hi"] = ci(p[pidx].mean(1))
        if c != "nowm":
            db, dp = b0 - b, p0 - p  # per-problem drops (paired)
            r["drop_base"], r["drop_plus"] = db.mean(), dp.mean()
            r["drop_base_lo"], r["drop_base_hi"] = ci(db[pidx].mean(1))
            r["drop_plus_lo"], r["drop_plus_hi"] = ci(dp[pidx].mean(1))
            ex = dp - db
            r["extra_drop_plus"] = ex.mean()
            r["extra_drop_lo"], r["extra_drop_hi"] = ci(ex[pidx].mean(1))
        rows.append(r)
    pd.DataFrame(rows).to_csv(tdir / "ci_pass_at_1.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ------------------------------------------------------------ thresholds
    dets = sorted({d for c in cfgs if GEN_CONFIGS[c].method for d in DETECTORS_FOR[GEN_CONFIGS[c].method]})
    thr, thr_b = {}, {}
    for d in dets:
        negz = an.zfile(root, d, "neg").z.to_numpy()
        thr[d] = an.emp_threshold(negz)
        thr_b[d] = boot_thresholds(negz, B, rng)

    # ------------------------------------------------------------- detection
    drows = []
    for c in cfgs:
        m = GEN_CONFIGS[c].method
        if m is None:
            continue
        zs = {d: an.zfile(root, d, f"gen:{c}").set_index("id").z for d in DETECTORS_FOR[m]}
        common = sorted(set.intersection(*(set(z.index) for z in zs.values())))
        gidx = rng.integers(0, len(common), size=(B, len(common)))
        tb = {}
        for d, z in zs.items():
            zz = z.loc[common].to_numpy()
            tb[d] = boot_tpr(zz, thr_b[d], gidx)
            lo, hi = ci(tb[d])
            drows.append({"config_id": c, "method": m, "delta": GEN_CONFIGS[c].delta, "detector": d,
                          "n": len(common), "TPR@1%FPR": (zz > thr[d]).mean(), "lo": lo, "hi": hi})
        if m == "EWD":
            diff = tb["EWD"] - tb["EWD-unweighted"]
            zw, zu = zs["EWD"].loc[common].to_numpy(), zs["EWD-unweighted"].loc[common].to_numpy()
            lo, hi = ci(diff)
            drows.append({"config_id": c, "method": m, "delta": GEN_CONFIGS[c].delta,
                          "detector": "EWD - EWD-unweighted", "n": len(common),
                          "TPR@1%FPR": (zw > thr["EWD"]).mean() - (zu > thr["EWD-unweighted"]).mean(),
                          "lo": lo, "hi": hi})
    pd.DataFrame(drows).to_csv(tdir / "ci_detection.csv", index=False)
    print(pd.DataFrame(drows).round(3).to_string(index=False))

    # --------------------------------------------------------------- attacks
    arows = []
    for c in cfgs:
        m = GEN_CONFIGS[c].method
        if m is None:
            continue
        atk = pd.DataFrame(read_jsonl(root / "attacks" / f"{c}.jsonl"))
        for d in DETECTORS_FOR[m]:
            za = an.zfile(root, d, f"atk:{c}").set_index("id").z
            zg = an.zfile(root, d, f"gen:{c}").set_index("id").z
            # z of the comment-stripped version per task (original where stripping was a no-op)
            st = atk[(atk.attack == "strip_comments") & (atk.status == "ok") & atk.id.isin(za.index)]
            z_strip = zg.to_dict()
            z_strip.update(dict(zip(st.task_id, za.loc[st.id])))
            for a in ATTACKS:
                v = atk[(atk.attack == a) & (atk.status == "ok") & (atk.pass_plus == True)]  # noqa: E712
                v = v[v.id.isin(za.index) & v.task_id.isin(zg.index)]
                if v.empty:
                    continue
                z_att = za.loc[v.id].to_numpy()
                z_orig = zg.loc[v.task_id].to_numpy()
                aidx = rng.integers(0, len(v), size=(B, len(v)))
                t_att, t_orig = boot_tpr(z_att, thr_b[d], aidx), boot_tpr(z_orig, thr_b[d], aidx)
                r = {"config_id": c, "method": m, "delta": GEN_CONFIGS[c].delta, "detector": d, "attack": a,
                     "n": len(v), "TPR_orig": (z_orig > thr[d]).mean(), "TPR_attacked": (z_att > thr[d]).mean()}
                r["orig_lo"], r["orig_hi"] = ci(t_orig)
                r["att_lo"], r["att_hi"] = ci(t_att)
                r["change"] = r["TPR_attacked"] - r["TPR_orig"]
                r["change_lo"], r["change_hi"] = ci(t_att - t_orig)
                if a.startswith("nc_"):
                    z_s = np.array([z_strip.get(t, np.nan) for t in v.task_id])
                    t_s = boot_tpr(z_s, thr_b[d], aidx)
                    r["TPR_stripped"] = (z_s > thr[d]).mean()
                    r["change_vs_stripped"] = r["TPR_attacked"] - r["TPR_stripped"]
                    r["cvs_lo"], r["cvs_hi"] = ci(t_att - t_s)
                arows.append(r)
    pd.DataFrame(arows).to_csv(tdir / "ci_attacks.csv", index=False)
    print(f"ci_attacks.csv: {len(arows)} rows")


if __name__ == "__main__":
    main()
