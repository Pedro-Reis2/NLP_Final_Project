"""Post-generation family (pitch): multi-channel watermark (our reimplementation, src/wm/multichannel.py)
under the same protocol as the decoding-time methods. CPU only.

  python scripts/11_multichannel.py embed                 # nowm generations -> outputs/generations/mcw.jsonl
  python scripts/02_evaluate.py --config mcw              # EvalPlus (QP1)
  python scripts/04_attack.py --config mcw                # the 11 attacks + tests (QP2)
  python scripts/11_multichannel.py detect                # negatives, nowm, mcw, attacked, LLM rewrites
  python scripts/11_multichannel.py analyze               # tables mcw_*.csv

A post-generation watermark is applied to finished code, so it is applied to the unwatermarked
Qwen generations (config `nowm`). Programs with no site cannot carry it and stay unchanged (they
count as watermarked-but-undetectable positives). Detection: `outputs/detect/Multicanal/<target>.jsonl`
rows {id, z, n_tokens, n_sites, n_votes, exact}; z = vote agreement (see the module docstring).

Tables
  mcw_pass_at_1.csv   pass@1 of watermarked vs unwatermarked code (paired bootstrap CI)
  mcw_detection.csv   capacity (sites, votes), exact-decode rate, TPR@1%FPR (empirical threshold
                      on the 1,540 human negatives) with CI, AUROC; also the potential sites the
                      paper's full channel set would offer
  mcw_attacks.csv     per attack: TPR / exact rate on the same programs unattacked -> attacked
"""
from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm import multichannel as mc  # noqa: E402
from wm.config import OUT, SEED  # noqa: E402
from wm.io import append_jsonl, get_logger, read_jsonl, write_manifest  # noqa: E402

DET = "Multicanal"
CFG = "mcw"
log = get_logger("multichannel")


def potential_sites(src: str) -> int:
    """Sites the paper's FULL channel set would add on top of ours (an upper-bound estimate):
    single-word locals extendable to multi-word names (n -> n_count), for+append loops / list
    comprehensions, if/else-assign / conditional expressions, commutative + and * operations."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    blocked = mc._blocked_names(tree)
    single = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
              and len(n.id) > 1 and mc.split_words(n.id) is None and n.id not in blocked}
    extra = 0
    for n in ast.walk(tree):
        if isinstance(n, (ast.ListComp, ast.IfExp)):
            extra += 1
        elif isinstance(n, ast.For) and any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                                            and isinstance(s.value.func, ast.Attribute)
                                            and s.value.func.attr == "append" for s in n.body):
            extra += 1
        elif isinstance(n, ast.If) and n.orelse and len(n.body) == 1 and len(n.orelse) == 1 \
                and all(isinstance(s, ast.Assign) for s in n.body + n.orelse):
            extra += 1
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mult)):
            extra += 1
    return 2 * len(single) + extra  # names carry 2 bits


def embed():
    gens = read_jsonl(OUT / "generations" / "nowm.jsonl")
    path = OUT / "generations" / f"{CFG}.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    rows = []
    for g in gens:
        if g["id"] in done:
            continue
        try:
            sol, emb = mc.embed(g["solution"]), True
        except (mc.NotEmbeddable, SyntaxError) as e:
            sol, emb = g["solution"], False
            reason = type(e).__name__
        rows.append({"id": g["id"], "dataset": g["dataset"], "config_id": CFG, "solution": sol,
                     "embedded": emb, **({} if emb else {"not_embedded": reason})})
    append_jsonl(path, rows)
    write_manifest(f"gen_{CFG}", {"step": "multichannel_embed", "source_config": "nowm", "payload": mc.PAYLOAD,
                                  "key": mc.KEY, "t": mc.T, "codeword_bits": mc.B,
                                  "channels": ["naming(2 bits)", "aug_assign", "compare_dir", "equality_dir",
                                               "empty_coll", "none_check"],
                                  "reference": "Choi, Gao & Duan (2026), arXiv:2607.06009 (reimplementation)"})
    log.info(f"embedded {sum(r['embedded'] for r in rows)}/{len(rows)}")


def _score(target: str, items: list[dict]):
    from transformers import AutoTokenizer
    from wm.config import MODEL_ID
    tok = _score.tok = getattr(_score, "tok", None) or AutoTokenizer.from_pretrained(MODEL_ID)
    path = OUT / "detect" / DET / f"{target.replace(':', '__')}.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    rows = []
    for it in items:
        if it["id"] in done:
            continue
        d = mc.detect(it["text"])
        rows.append({"id": it["id"], "z": d["z"], "n_tokens": len(tok(it["text"], add_special_tokens=False).input_ids),
                     "n_sites": d["n_sites"], "n_votes": d["n_votes"], "agree": d["agree"], "exact": d["exact"],
                     "n_erased": d["n_erased"], "potential_extra_sites": potential_sites(it["text"])})
    append_jsonl(path, rows)
    log.info(f"[detect {target}] +{len(rows)}")


def detect():
    neg = read_jsonl(OUT / "negatives" / "negatives.jsonl")
    _score("neg", [{"id": r["id"], "text": r["code"]} for r in neg])
    for c in ("nowm", CFG):
        _score(f"gen:{c}", [{"id": r["id"], "text": r["solution"]} for r in read_jsonl(OUT / "generations" / f"{c}.jsonl")])
    atk = [r for r in read_jsonl(OUT / "attacks" / f"{CFG}.jsonl") if r.get("attacked")]
    _score(f"atk:{CFG}", [{"id": r["id"], "text": r["attacked"]} for r in atk])
    rw = [r for r in read_jsonl(OUT / "rewrite" / f"{CFG}.jsonl") if r.get("attacked")]
    if rw:
        _score(f"rw:{CFG}", [{"id": r["id"], "text": r["attacked"]} for r in rw])


def analyze(B: int = 2000):
    import numpy as np
    import pandas as pd

    from wm.attacks import ATTACKS
    an = importlib.import_module("06_analyze")
    unc = importlib.import_module("09_uncertainty")
    rng = np.random.default_rng(SEED)
    tdir = OUT / "tables"

    def z(target):
        return an.zfile(OUT, DET, target)

    negz = z("neg").z.to_numpy()  # NaN (no readable vote) -> -1e9 = "not detected"
    thr = an.emp_threshold(negz)
    thr_b = unc.boot_thresholds(negz, B, rng)
    neg = z("neg")

    # ---------------------------------------------------------------- pass@1 (QP1)
    ev = {c: pd.DataFrame(read_jsonl(OUT / "evals" / f"{c}.jsonl")).set_index("id") for c in ("nowm", CFG)}
    ids = sorted(set(ev["nowm"].index) & set(ev[CFG].index))
    gen = pd.DataFrame(read_jsonl(OUT / "generations" / f"{CFG}.jsonl")).set_index("id")
    pidx = rng.integers(0, len(ids), size=(B, len(ids)))
    prow = {"n": len(ids), "n_embedded": int(gen.loc[ids, "embedded"].sum())}
    for k in ("base", "plus"):
        a = ev["nowm"].loc[ids, f"pass_{k}"].to_numpy(float)
        b = ev[CFG].loc[ids, f"pass_{k}"].to_numpy(float)
        prow[f"pass@1_{k}_nowm"], prow[f"pass@1_{k}_mcw"] = a.mean(), b.mean()
        prow[f"drop_{k}"] = (a - b).mean()
        prow[f"drop_{k}_lo"], prow[f"drop_{k}_hi"] = unc.ci((a - b)[pidx].mean(1))
        prow[f"n_broken_{k}"] = int(((a == 1) & (b == 0)).sum())
    pd.DataFrame([prow]).to_csv(tdir / "mcw_pass_at_1.csv", index=False)
    print(pd.DataFrame([prow]).round(4).to_string(index=False))

    # ------------------------------------------------------------- detection
    pos, nowm = z(f"gen:{CFG}"), z("gen:nowm")
    gidx = rng.integers(0, len(pos), size=(B, len(pos)))
    lo, hi = unc.ci(unc.boot_tpr(pos.z.to_numpy(), thr_b, gidx))
    emb = set(gen[gen.embedded].index)
    pe = pos[pos.id.isin(emb)]
    drow = {"detector": DET, "n": len(pos), "n_embedded": len(pe), "threshold_1pct": thr,
            "neg_mean_z": an.zmean(negz), "neg_frac_scorable": float((negz > -1e8).mean()),
            "neg_exact_rate": float(neg.exact.mean()), "nowm_exact_rate": float(nowm.exact.mean()),
            "median_sites": float(pos.n_sites.median()), "mean_sites": float(pos.n_sites.mean()),
            "median_votes": float(pos.n_votes.median()), "frac_votes_ge_32": float((pos.n_votes >= 32).mean()),
            "median_potential_bits_full_scheme": float((pos.n_votes + pos.potential_extra_sites).median()),
            "frac_potential_ge_32": float(((pos.n_votes + pos.potential_extra_sites) >= 32).mean()),
            "exact_rate": float(pos.exact.mean()), "exact_rate_embedded": float(pe.exact.mean()),
            "TPR@1%FPR": float((pos.z > thr).mean()), "TPR_lo": lo, "TPR_hi": hi,
            "TPR@1%FPR_embedded": float((pe.z > thr).mean()),
            "AUROC": an.auroc(pos.z.to_numpy(), negz), "FPR_nowm_gen": float((nowm.z > thr).mean())}
    # TPR by number of readable votes (the capacity analogue of QP3)
    bins = [0, 1, 2, 4, 8, 16, 1000]
    pos = pos.assign(vbin=pd.cut(pos.n_votes, bins, right=False))
    vrows = [{"votes_bin": str(b), "n": len(s), "TPR@1%FPR": float((s.z > thr).mean()), "exact_rate": float(s.exact.mean())}
             for b, s in pos.groupby("vbin", observed=True)]
    pd.DataFrame([drow]).to_csv(tdir / "mcw_detection.csv", index=False)
    pd.DataFrame(vrows).to_csv(tdir / "mcw_by_votes.csv", index=False)
    print(pd.DataFrame([drow]).round(4).T.to_string())
    print(pd.DataFrame(vrows).round(3).to_string(index=False))

    # ---------------------------------------------------------------- attacks
    atk = pd.DataFrame(read_jsonl(OUT / "attacks" / f"{CFG}.jsonl"))
    za, zg = z(f"atk:{CFG}").set_index("id"), z(f"gen:{CFG}").set_index("id")
    arows = []
    for a in ATTACKS + ["llm_rewrite"]:
        if a == "llm_rewrite":
            sub = pd.DataFrame(read_jsonl(OUT / "rewrite" / f"{CFG}.jsonl"))
            zr = z(f"rw:{CFG}").set_index("id") if not sub.empty else None
            if sub.empty or zr is None or zr.empty:
                continue
        else:
            sub, zr = atk[atk.attack == a], za
        v = sub[(sub.status == "ok") & (sub.pass_plus == True)]  # noqa: E712
        v = v[v.id.isin(zr.index) & v.task_id.isin(zg.index) & v.task_id.isin(emb)]
        if v.empty:
            continue
        z_att, z_orig = zr.loc[v.id, "z"].to_numpy(), zg.loc[v.task_id, "z"].to_numpy()
        idx = rng.integers(0, len(v), size=(B, len(v)))
        t_att, t_orig = unc.boot_tpr(z_att, thr_b, idx), unc.boot_tpr(z_orig, thr_b, idx)
        applied = sub[sub.status == "ok"]
        r = {"attack": a, "n_applied": len(applied), "n_valid": len(v),
             "survival": len(sub[(sub.status == "ok") & (sub.pass_plus == True)]) / max(1, len(applied)),  # noqa: E712
             "mean_frac_tok_changed": v.frac_tok_changed.mean(),
             "TPR_orig": (z_orig > thr).mean(), "TPR_attacked": (z_att > thr).mean(),
             "exact_orig": zg.loc[v.task_id, "exact"].mean(), "exact_attacked": zr.loc[v.id, "exact"].mean(),
             "mean_agree_orig": (zg.loc[v.task_id, "agree"] / zg.loc[v.task_id, "n_votes"].clip(lower=1)).mean(),
             "mean_agree_attacked": (zr.loc[v.id, "agree"] / zr.loc[v.id, "n_votes"].clip(lower=1)).mean()}
        r["change"] = r["TPR_attacked"] - r["TPR_orig"]
        r["change_lo"], r["change_hi"] = unc.ci(t_att - t_orig)
        arows.append(r)
    pd.DataFrame(arows).to_csv(tdir / "mcw_attacks.csv", index=False)
    print(pd.DataFrame(arows).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "detect", "analyze"])
    args = ap.parse_args()
    {"embed": embed, "detect": detect, "analyze": analyze}[args.step]()
    log.info("MCW STEP DONE")


if __name__ == "__main__":
    main()
