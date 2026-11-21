"""QP4 (optional): the same watermarks on natural-language text, for comparison with code.

  python scripts/08_text_qp4.py gen     [--limit 5]   # continuations of C4 news prefixes
  python scripts/08_text_qp4.py attack  [--limit 5]   # random word substitution
  python scripts/08_text_qp4.py detect  [--limit 5]   # KGW / Unigram z-scores
  python scripts/08_text_qp4.py analyze               # tables + figure
  python scripts/08_text_qp4.py all     [--limit 5]

Data: MarkLLM's C4 sample (data/c4/processed_c4.json, from the MarkLLM GitHub repo,
dataset/c4/processed_c4.json; the pip package does not ship it).
  - prompts: rows 0..N_PROMPTS-1, field `prompt` (news prefix), no chat template
  - human negatives: field `natural_text` of rows N_PROMPTS..N_PROMPTS+N_NEG-1 (disjoint),
    truncated to the first GEN_TOKENS Qwen tokens -> same length as the generations
Generation: same model and greedy decoding as the code study, exactly GEN_TOKENS new
tokens (min = max). Detection input: the continuation only (no prompt), as for code.
Attack: replace a fraction p of the words with random words drawn from the human
negatives' vocabulary (seeded per sample). Intensity is reported as the fraction of
Qwen tokens changed, computed exactly as for the code attacks (04_attack.frac_changed).

Outputs (resumable, keyed by id): outputs/qp4/{generations,attacks,detect}/...,
tables outputs/tables/qp4_*.csv, figures figures/fig_qp4_text_vs_code_d{2,4}.{png,pdf}.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm.config import DATA, FIGS, GEN_CONFIGS, MODEL_ID, OUT, SEED  # noqa: E402
from wm.io import append_jsonl, done_ids, get_logger, read_jsonl, write_manifest  # noqa: E402

N_PROMPTS, N_NEG, GEN_TOKENS = 300, 1000, 200
CONFIGS = ["nowm", "kgw_d2", "unigram_d2", "kgw_d4", "unigram_d4"]  # code-study config ids
SUB_FRACS = [0.1, 0.2, 0.3, 0.5]
DETECTORS = ["KGW", "Unigram"]
C4 = DATA / "c4" / "processed_c4.json"
Q = OUT / "qp4"
log = get_logger("qp4")


def load_c4():
    return [json.loads(line) for line in C4.open(encoding="utf-8")]


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    return model, tok


def q(root: Path, *parts) -> Path:
    return root.joinpath(*parts)


# ------------------------------------------------------------------ negatives
def human_texts(tok, limit):
    rows = load_c4()[N_PROMPTS: N_PROMPTS + N_NEG]
    out = []
    for i, r in enumerate(rows[:limit]):
        ids = tok(r["natural_text"].strip(), add_special_tokens=False).input_ids[:GEN_TOKENS]
        out.append({"id": f"c4_{N_PROMPTS + i}", "text": tok.decode(ids), "n_tokens": len(ids)})
    return out


# ------------------------------------------------------------------ generate
def gen(root, limit, batch_size, model=None, tok=None):
    import torch
    from transformers import LogitsProcessorList

    from wm.markllm_utils import load_watermark
    if model is None:
        model, tok = load_model()
    prompts = [{"id": f"c4_{i}", "prompt": r["prompt"]} for i, r in enumerate(load_c4()[:N_PROMPTS])][:limit]
    for cfg_id in CONFIGS:
        cfg = GEN_CONFIGS[cfg_id]
        path = q(root, "generations", f"text_{cfg_id}.jsonl")
        done = done_ids(path)
        todo = [p for p in prompts if p["id"] not in done]
        write_manifest(f"qp4_gen_{cfg_id}", {"step": "qp4_generate", **cfg.to_dict(),
                                            "decoding": {"do_sample": False, "min_new_tokens": GEN_TOKENS,
                                                         "max_new_tokens": GEN_TOKENS},
                                            "data": "MarkLLM C4 sample rows 0..%d" % (N_PROMPTS - 1)})
        log.info(f"[text_{cfg_id}] {len(done)} done, {len(todo)} to go")
        if not todo:
            continue
        wm = load_watermark(cfg, model, tok) if cfg.method else None
        procs = LogitsProcessorList([wm.logits_processor]) if wm else None
        t0 = time.time()
        for b in range(0, len(todo), batch_size):
            batch = todo[b: b + batch_size]
            enc = tok([p["prompt"] for p in batch], return_tensors="pt", padding=True).to("cuda")
            torch.manual_seed(SEED)
            with torch.no_grad():
                out = model.generate(**enc, logits_processor=procs, do_sample=False,
                                     min_new_tokens=GEN_TOKENS, max_new_tokens=GEN_TOKENS,
                                     pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
            new = out[:, enc["input_ids"].shape[1]:]
            rows = []
            for p, ids in zip(batch, new):
                ids = ids[ids != tok.pad_token_id]
                rows.append({"id": p["id"], "config_id": f"text_{cfg_id}", "text": tok.decode(ids, skip_special_tokens=True),
                             "n_new_tokens": int(len(ids))})
            append_jsonl(path, rows)
            log.info(f"[text_{cfg_id}] {b + len(batch)}/{len(todo)}  {time.time() - t0:.0f}s")
        del wm
        torch.cuda.empty_cache()
    return model, tok


# ------------------------------------------------------------------ attack
def frac_changed(tok, a: str, b: str):
    """Identical to scripts/04_attack.py:frac_changed (token-level diff)."""
    ta = tok(a, add_special_tokens=False).input_ids
    tb = tok(b, add_special_tokens=False).input_ids
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return changed / max(1, len(ta)), len(ta), len(tb)


WORD = re.compile(r"[A-Za-z]+")


def substitute(text: str, p: float, vocab: list[str], salt: str) -> str:
    """Replace round(p * #words) randomly chosen alphabetic words with random vocabulary words.
    Leading/trailing punctuation and capitalisation of the first letter are kept."""
    rng = random.Random(int(hashlib.md5(f"sub|{salt}|{p}".encode()).hexdigest(), 16))
    parts = re.split(r"(\s+)", text)
    idx = [i for i, w in enumerate(parts) if WORD.search(w)]
    for i in rng.sample(idx, round(p * len(idx))):
        m = WORD.search(parts[i])
        new = rng.choice(vocab)
        if m.group()[0].isupper():
            new = new.capitalize()
        parts[i] = parts[i][: m.start()] + new + parts[i][m.end():]
    return "".join(parts)


def attack(root, limit, tok=None):
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
    neg = human_texts(tok, None)
    vocab = sorted({w.lower() for r in neg for w in WORD.findall(r["text"]) if len(w) >= 3})
    for cfg_id in CONFIGS[1:]:
        gens = read_jsonl(q(root, "generations", f"text_{cfg_id}.jsonl"))[:limit]
        path = q(root, "attacks", f"text_{cfg_id}.jsonl")
        done = done_ids(path)
        rows = []
        for g in gens:
            for p in SUB_FRACS:
                rid = f"{g['id']}|sub_{p}"
                if rid in done:
                    continue
                new = substitute(g["text"], p, vocab, g["id"])
                f, n0, n1 = frac_changed(tok, g["text"], new)
                rows.append({"id": rid, "task_id": g["id"], "config_id": f"text_{cfg_id}", "attack": f"sub_{p}",
                             "attacked": new, "frac_tok_changed": round(f, 4), "n_tok_orig": n0, "n_tok_attacked": n1})
        append_jsonl(path, rows)
        log.info(f"[attack text_{cfg_id}] +{len(rows)} rows")
    return tok


# ------------------------------------------------------------------ detect
def detect(root, limit, model=None, tok=None):
    det_mod = importlib.import_module("05_detect")
    if model is None:
        model, tok = load_model()
    scorer = det_mod.Scorer(model, tok)
    targets = {"human": (human_texts(tok, limit), DETECTORS)}
    for cfg_id in CONFIGS:
        m = GEN_CONFIGS[cfg_id].method
        dets = DETECTORS if m is None else [m]
        targets[f"gen__text_{cfg_id}"] = ([{"id": r["id"], "text": r["text"]} for r in
                                           read_jsonl(q(root, "generations", f"text_{cfg_id}.jsonl"))[:limit]], dets)
        if m:
            targets[f"atk__text_{cfg_id}"] = ([{"id": r["id"], "text": r["attacked"]} for r in
                                               read_jsonl(q(root, "attacks", f"text_{cfg_id}.jsonl"))
                                               [: None if limit is None else limit * len(SUB_FRACS)]], dets)
    for name, (rows, dets) in targets.items():
        for d in dets:
            path = q(root, "detect", d, f"{name}.jsonl")
            done = done_ids(path)
            todo = [r for r in rows if r["id"] not in done]
            buf = []
            for i, r in enumerate(todo, 1):
                n = len(tok(r["text"], add_special_tokens=False).input_ids)
                try:
                    z = scorer.score(d, r["text"]) if n >= 2 else None
                except Exception as e:
                    log.warning(f"{d} {name} {r['id']}: {type(e).__name__}: {e}")
                    z = None
                buf.append({"id": r["id"], "z": z, "n_tokens": n})
                if len(buf) >= 100 or i == len(todo):
                    append_jsonl(path, buf)
                    buf = []
            if todo:
                log.info(f"[detect {d} {name}] +{len(todo)}")


# ------------------------------------------------------------------ analyze
# Greedy decoding sometimes loops on news text. "Repetitive" = distinct-word ratio
# below REP_RATIO; reported as a share and as a sensitivity TPR without them.
REP_RATIO = 0.3


def distinct_ratio(text: str) -> float:
    w = text.split()
    return len(set(w)) / max(1, len(w))


def analyze(root):
    import numpy as np
    import pandas as pd
    an = importlib.import_module("06_analyze")
    full = root == Q
    tdir = OUT / "tables" if full else root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    def z(det, name):
        df = pd.DataFrame(read_jsonl(q(root, "detect", det, f"{name}.jsonl")))
        if not df.empty:
            df["z"] = df["z"].fillna(-1e9)
        return df

    rows, arows = [], []
    for d in DETECTORS:
        neg = z(d, "human")
        if neg.empty:
            continue
        negz = neg.z.to_numpy()
        thr = an.emp_threshold(negz)
        nowm = z(d, "gen__text_nowm")
        gr0 = pd.DataFrame(read_jsonl(q(root, "generations", "text_nowm.jsonl")))
        rep0 = set(gr0[gr0.text.map(distinct_ratio) < REP_RATIO].id)
        rows.append({"config_id": "text_nowm", "method": "none", "delta": np.nan, "detector": d,
                     "n": len(nowm), "frac_repetitive": len(rep0) / max(1, len(gr0)),
                     "TPR_nonrepetitive": float((nowm[~nowm.id.isin(rep0)].z > thr).mean()), "neg_mean_z": an.zmean(negz), "neg_sd_z": float(negz[negz > -1e8].std(ddof=1)),
                     "neg_frac_gt_2.326": float((negz > 2.326).mean()), "threshold_1pct": thr,
                     "mean_z": an.zmean(nowm.z), "TPR@1%FPR": float((nowm.z > thr).mean()),
                     "AUROC": an.auroc(nowm.z.to_numpy(), negz)})
        for cfg_id in CONFIGS[1:]:
            c = GEN_CONFIGS[cfg_id]
            if c.method != d:
                continue
            g = z(d, f"gen__text_{cfg_id}")
            if g.empty:
                continue
            gr = pd.DataFrame(read_jsonl(q(root, "generations", f"text_{cfg_id}.jsonl")))
            rep = set(gr[gr.text.map(distinct_ratio) < REP_RATIO].id)
            nonrep = g[~g.id.isin(rep)]
            rows.append({"config_id": f"text_{cfg_id}", "method": d, "delta": c.delta, "detector": d, "n": len(g),
                         "frac_repetitive": len(rep) / max(1, len(gr)),
                         "TPR_nonrepetitive": float((nonrep.z > thr).mean()),
                         "neg_mean_z": an.zmean(negz), "neg_sd_z": float(negz[negz > -1e8].std(ddof=1)),
                         "neg_frac_gt_2.326": float((negz > 2.326).mean()), "threshold_1pct": thr,
                         "mean_z": an.zmean(g.z), "TPR@1%FPR": float((g.z > thr).mean()),
                         "AUROC": an.auroc(g.z.to_numpy(), negz)})
            a = z(d, f"atk__text_{cfg_id}")
            meta = pd.DataFrame(read_jsonl(q(root, "attacks", f"text_{cfg_id}.jsonl")))
            if a.empty or meta.empty:
                continue
            a = a.merge(meta[["id", "attack", "frac_tok_changed"]], on="id")
            for atk, s in a.groupby("attack"):
                arows.append({"config_id": f"text_{cfg_id}", "method": d, "delta": c.delta, "detector": d,
                              "attack": atk, "sub_frac": float(atk.split("_")[1]), "n": len(s),
                              "mean_frac_tok_changed": s.frac_tok_changed.mean(), "mean_z_attacked": an.zmean(s.z),
                              "TPR_attacked": float((s.z > thr).mean()),
                              "AUROC_attacked": an.auroc(s.z.to_numpy(), negz)})
    det = pd.DataFrame(rows)
    atk = pd.DataFrame(arows)
    if full and not det.empty:
        # Same config on code: TPR over all generations, and in the length bin that
        # contains 200 tokens (the text length) -> length-matched comparison.
        cd = pd.read_csv(OUT / "tables" / "detection.csv")
        ln = pd.read_csv(OUT / "tables" / "length.csv")
        code_all = {(r.config_id, r.detector): r["TPR@1%FPR"] for _, r in cd.iterrows()}
        code_200 = {(r.config_id, r.detector): (r["TPR@1%FPR"], r.n, r.len_bin)
                    for _, r in ln[ln.bin_lo == 200].iterrows()}
        cid = det.config_id.str.replace("text_", "", regex=False)
        det["code_TPR_all"] = [code_all.get((c, d)) for c, d in zip(cid, det.detector)]
        det["code_TPR_len200"] = [code_200.get((c, d), (None,))[0] for c, d in zip(cid, det.detector)]
        det["code_n_len200"] = [code_200.get((c, d), (None, None))[1] for c, d in zip(cid, det.detector)]
    det.to_csv(tdir / "qp4_text_detection.csv", index=False)
    atk.to_csv(tdir / "qp4_text_attacks.csv", index=False)
    print(det.to_string(index=False))
    print(atk.to_string(index=False))
    if full and not atk.empty:
        figure(det, atk)


def figure(det, atk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    figs = importlib.import_module("07_figures")
    code_atk = pd.read_csv(OUT / "tables" / "attacks.csv")
    code_det = pd.read_csv(OUT / "tables" / "detection.csv")
    for delta in sorted(atk.delta.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
        for ax, d in zip(axes, DETECTORS):
            c = figs.COLOR[d]
            t = atk[(atk.delta == delta) & (atk.detector == d)].sort_values("mean_frac_tok_changed")
            t0 = det[(det.delta == delta) & (det.detector == d)]["TPR@1%FPR"]
            xs = [0.0] + list(t.mean_frac_tok_changed)
            ys = [t0.iloc[0] if len(t0) else float("nan")] + list(t.TPR_attacked)
            ax.plot(xs, [y * 100 for y in ys], color=c, lw=2, marker="o", ms=5, markeredgecolor=figs.SURFACE,
                    markeredgewidth=1.5)
            ca = code_atk[(code_atk.delta == delta) & (code_atk.detector == d)]
            cd = code_det[(code_det.delta == delta) & (code_det.detector == d)]["TPR@1%FPR_passing"]
            cx = [0.0] + list(ca.mean_frac_tok_changed)
            cy = [cd.iloc[0] if len(cd) else float("nan")] + list(ca.TPR_attacked)
            ax.scatter(cx, [y * 100 for y in cy], s=36, facecolor=figs.SURFACE, edgecolor=figs.INK2, lw=1.5,
                       zorder=3)
            ax.set_title(f"{d} (δ={int(delta)})", fontsize=9, color=figs.INK)
            ax.set_xlabel("Fraction of tokens changed")
            ax.set_ylim(0, 100)
        axes[0].set_ylabel("TPR at 1% FPR (%)")
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color=figs.INK2, lw=2, marker="o", ms=5, markeredgecolor=figs.SURFACE),
             Line2D([], [], ls="none", marker="o", ms=6, markerfacecolor=figs.SURFACE,
                    markeredgecolor=figs.INK2, markeredgewidth=1.5)]
        lab = ["Text (C4 news, 200 tokens): random word substitution",
               "Code (HumanEval+/MBPP+): code attacks"]
        for ax in axes:
            ax.set_xlim(-0.03, 0.93)
        fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2)
        figs.save(fig, f"fig_qp4_text_vs_code_d{int(delta)}", FIGS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "attack", "detect", "analyze", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    root = Q if args.limit is None else OUT / "test" / "qp4"
    model = tok = None
    if args.step in ("gen", "all"):
        model, tok = gen(root, args.limit, args.batch_size)
    if args.step in ("attack", "all"):
        tok = attack(root, args.limit, tok)
    if args.step in ("detect", "all"):
        detect(root, args.limit, model if model is not None else None, tok if model is not None else None)
    if args.step in ("analyze", "all"):
        analyze(root)
    log.info("QP4 DONE")


if __name__ == "__main__":
    main()
