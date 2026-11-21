"""Publication figures from outputs/tables/*.csv -> figures/*.png + *.pdf (+ LaTeX table).

  python scripts/07_figures.py [--test --tag probe_greedy] [--delta 2]

fig_pass_at_1     dumbbell: pass@1 with original (base) vs extended (plus) tests per config
fig_tpr_attacks   grouped bars: TPR@1%FPR per attack, one bar per detector
fig_tpr_length    lines: TPR@1%FPR vs code length bin, one line per detector, 80% reference
pass_at_1.tex     LaTeX table for the paper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wm.config import FIGS, OUT  # noqa: E402

# Categorical slots 1-5 of the dataviz reference palette, fixed order; baseline in neutral gray.
COLOR = {"KGW": "#2a78d6", "EWD": "#eb6834", "Unigram": "#1baf7a", "SWEET": "#eda100",
         "EWD-unweighted": "#e87ba4", "none": "#8a8984"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ATTACK_LABEL = {"none": "No\nattack", "strip_comments": "Strip\ncomments", "black": "black\n",
                "rename_1": "Rename\n1 name", "rename_2": "Rename\n2 names", "rename_all": "Rename\nall names",
                "deadcode_try": "Dead code +\ntry/except"}
# "Without comments" condition: comments + docstrings stripped first, then the attack
NC_LABEL = {"strip_comments": "Comments\nstripped", "nc_black": "+ black\n", "nc_rename_1": "+ rename\n1 name",
            "nc_rename_2": "+ rename\n2 names", "nc_rename_all": "+ rename\nall names",
            "nc_deadcode_try": "+ dead code,\ntry/except"}
DET_ORDER = ["KGW", "EWD-unweighted", "EWD", "Unigram", "SWEET"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name, out):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def label_cfg(r):
    return "No watermark" if r.method == "none" else f"{r.method} (δ={int(r.delta)})"


def fig_pass(p1, out):
    d = p1[p1.dataset == "all"].copy()
    if d.empty:
        return
    d["label"] = d.apply(label_cfg, axis=1)
    d = d.sort_values(["delta", "method"], na_position="first").iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(d) + 0.9))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        c = COLOR.get(r.method, COLOR["none"])
        ax.plot([r["pass@1_plus"] * 100, r["pass@1_base"] * 100], [yi, yi], color=c, lw=2, alpha=0.5)
        ax.scatter(r["pass@1_base"] * 100, yi, s=36, facecolor=SURFACE, edgecolor=c, lw=2, zorder=3)
        ax.scatter(r["pass@1_plus"] * 100, yi, s=36, color=c, edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.set_yticks(y, d.label)
    ax.set_xlabel("pass@1 (%), HumanEval+ and MBPP+")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=36, facecolor=SURFACE, edgecolor=INK2, lw=2, label="Original tests")
    ax.scatter([], [], s=36, color=INK2, label="Extended tests (EvalPlus)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig_pass_at_1", out)


def tex_pass(p1, out):
    if p1.empty:
        return
    t = p1.copy()
    t["Config"] = t.apply(label_cfg, axis=1)
    piv = t.pivot_table(index=["delta", "Config"], columns="dataset",
                        values=["pass@1_base", "pass@1_plus"], dropna=False).sort_index()
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{HumanEval} & \multicolumn{2}{c}{MBPP} & \multicolumn{2}{c}{All} \\",
             r"Config & orig. & ext. & orig. & ext. & orig. & ext. \\", r"\midrule"]
    for (delta, cfg), r in piv.iterrows():
        vals = []
        for ds in ("humaneval", "mbpp", "all"):
            for k in ("pass@1_base", "pass@1_plus"):
                v = r.get((k, ds), np.nan)
                vals.append("--" if pd.isna(v) else f"{100 * v:.1f}")
        lines.append(f"{cfg} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "pass_at_1.tex").write_text("\n".join(lines), encoding="utf-8")


def fig_attacks(atk, det, delta, out):
    a = atk[atk.delta == delta]
    if a.empty:
        return
    dets = [x for x in DET_ORDER if x in set(a.detector)]
    # "No attack" = unattacked TPR on all passing generations (detection.csv, passing column)
    base = det[(det.delta == delta)].set_index("detector")["TPR@1%FPR_passing"]
    attacks = ["none"] + [x for x in ATTACK_LABEL if x != "none" and x in set(a.attack)]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    w = 0.8 / len(dets)
    x = np.arange(len(attacks))
    for i, d in enumerate(dets):
        vals = [base.get(d, np.nan) if at == "none" else
                a[(a.detector == d) & (a.attack == at)]["TPR_attacked"].mean() for at in attacks]
        ax.bar(x - 0.4 + w * (i + 0.5), np.array(vals) * 100, width=w - 0.02, color=COLOR[d], label=d,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [ATTACK_LABEL[k] for k in attacks])
    ax.set_ylabel("TPR at 1% FPR (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(dets), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, f"fig_tpr_attacks_d{int(delta)}", out)


def fig_attacks_nc(atk, delta, out):
    """Same chart for the "without comments" condition: every bar is on comment-stripped code."""
    a = atk[atk.delta == delta]
    dets = [x for x in DET_ORDER if x in set(a.detector)]
    attacks = [k for k in NC_LABEL if k in set(a.attack)]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    w = 0.8 / len(dets)
    x = np.arange(len(attacks))
    for i, d in enumerate(dets):
        vals = [a[(a.detector == d) & (a.attack == at)]["TPR_attacked"].mean() for at in attacks]
        ax.bar(x - 0.4 + w * (i + 0.5), np.array(vals) * 100, width=w - 0.02, color=COLOR[d], label=d,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [NC_LABEL[k] for k in attacks])
    ax.set_ylabel("TPR at 1% FPR (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(dets), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, f"fig_tpr_attacks_nocomments_d{int(delta)}", out)


def fig_tradeoff(t, out):
    """pass@1 (extended tests) vs TPR@1%FPR per method, delta 2 -> 4, with bootstrap 95% CIs
    (needs ci_pass_at_1.csv and ci_detection.csv from 09_uncertainty.py)."""
    p = pd.read_csv(t / "ci_pass_at_1.csv").set_index("config_id")
    d = pd.read_csv(t / "ci_detection.csv")
    d = d[d.detector == d.method]  # EWD configs: the entropy-weighted EWD detector
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    base = p.loc["nowm"]
    ax.axhspan(base.plus_lo * 100, base.plus_hi * 100, color=COLOR["none"], alpha=0.12, lw=0)
    ax.axhline(base["pass@1_plus"] * 100, color=COLOR["none"], lw=1, ls=(0, (4, 3)))
    ax.text(101, base["pass@1_plus"] * 100, "No\nwatermark", va="center", ha="left", fontsize=8, color=INK2)
    for m in ["KGW", "EWD", "Unigram", "SWEET"]:
        s = d[d.method == m].sort_values("delta")
        if s.empty:
            continue
        x = s["TPR@1%FPR"].to_numpy() * 100
        y = np.array([p.loc[c, "pass@1_plus"] for c in s.config_id]) * 100
        xerr = np.vstack([x - s.lo * 100, s.hi * 100 - x])
        ylo = np.array([p.loc[c, "plus_lo"] for c in s.config_id]) * 100
        yhi = np.array([p.loc[c, "plus_hi"] for c in s.config_id]) * 100
        ax.errorbar(x, y, xerr=xerr, yerr=np.vstack([y - ylo, yhi - y]), color=COLOR[m], lw=2, elinewidth=1,
                     capsize=0, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=m, zorder=3)
        # label offsets (points) chosen so neighbouring labels do not collide
        nudge = {("Unigram", 2): (-8, -14), ("EWD", 2): (-30, -14), ("SWEET", 4): (8, 4)}
        for xi, yi, dl in zip(x, y, s.delta):
            ax.annotate(f"δ={int(dl)}", (xi, yi), textcoords="offset points", xytext=nudge.get((m, int(dl)), (6, 5)),
                        fontsize=7.5, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("TPR at 1% FPR (%), all generations")
    ax.set_ylabel("pass@1 (%), extended tests")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    save(fig, "fig_tradeoff_pass_tpr", out)


def fig_length(ln, det, delta, out):
    cfg_of = det[det.delta == delta][["config_id", "detector"]]
    d = ln.merge(cfg_of, on=["config_id", "detector"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for name in [x for x in DET_ORDER if x in set(d.detector)]:
        s = d[d.detector == name].sort_values("bin_lo")
        s = s[s.n >= 10]
        ax.plot(s.bin_lo, s["TPR@1%FPR"] * 100, color=COLOR[name], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.axhline(80, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(ax.get_xlim()[1], 81, "80% target", ha="right", va="bottom", color=INK2, fontsize=8)
    ax.set_xlabel("Code length (Qwen tokens, lower edge of bin)")
    ax.set_ylabel("TPR at 1% FPR (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, f"fig_tpr_length_d{int(delta)}", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    root = OUT / "test" / args.tag if args.test else OUT
    t = root / "tables"
    out = (root / "figures") if args.test else FIGS

    def rd(n):
        p = t / n
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()

    p1, det, atk, ln = rd("pass_at_1.csv"), rd("detection.csv"), rd("attacks.csv"), rd("length.csv")
    fig_pass(p1, out)
    tex_pass(p1, out)
    if (t / "ci_pass_at_1.csv").exists() and (t / "ci_detection.csv").exists():
        fig_tradeoff(t, out)
    for delta in sorted(det.delta.dropna().unique()) if not det.empty else []:
        if not atk.empty:
            fig_attacks(atk, det, delta, out)
            if atk.attack.str.startswith("nc_").any():
                fig_attacks_nc(atk, delta, out)
        if not ln.empty:
            fig_length(ln, det, delta, out)


if __name__ == "__main__":
    main()
