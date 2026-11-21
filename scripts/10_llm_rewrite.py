"""Attack 5 (pitch): LLM rewrite on a sample, the "upper bound" attack.

  python scripts/10_llm_rewrite.py gen     [--limit 5] [--n 100]   # GPU: rewrite with an LLM
  python scripts/10_llm_rewrite.py test                            # CPU: EvalPlus on rewrites
  python scripts/05_detect.py --targets rw:kgw_d2,...               # GPU: z-scores
  python scripts/10_llm_rewrite.py analyze                         # tables (+ bootstrap CIs)

Sample: per watermarked config, N generations drawn (seed 42) from those passing the
extended tests. Rewriter: Qwen2.5-Coder-1.5B-Instruct, WITHOUT a watermark, greedy, asked to
rewrite the program so it behaves identically but looks different (rename, restructure,
reword comments). Only rewrites that still pass the extended tests count (same rule as the
other attacks). Survival = passing rewrites / rewrites.

Steps are separate processes on purpose: EvalPlus test workers must never import torch
(see 04_attack.py). Output: outputs/rewrite/<config_id>.jsonl, keyed by task id, resumable.
Table: outputs/tables/llm_rewrite.csv.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm.config import DETECTORS_FOR, GEN_CONFIGS, MODEL_ID, OUT, SEED  # noqa: E402
from wm.io import append_jsonl, done_ids, get_logger, read_jsonl, write_manifest  # noqa: E402

N_DEFAULT = 100
WM_CONFIGS = [c for c in GEN_CONFIGS if GEN_CONFIGS[c].method]
PROMPT = ("Rewrite the following Python program so that it behaves exactly the same but looks "
          "different. Rename the variables and helper names, restructure the code (for example "
          "change loops, conditions and expressions into equivalent forms), and rewrite any comments "
          "and docstrings in your own words. Keep the names and signatures of the top-level functions "
          "unchanged. Return the complete program in one markdown code block.\n\n```python\n{code}\n```")
RESPONSE_PREFIX = "```python\n"
STOP = "\n```"
log = get_logger("rewrite")


def sample_ids(cfg: str, n: int) -> list[str]:
    ev = read_jsonl(OUT / "evals" / f"{cfg}.jsonl")
    passing = sorted(e["id"] for e in ev if e["pass_plus"])
    return sorted(random.Random(SEED).sample(passing, min(n, len(passing))))


def _top_defs(src: str) -> list[str]:
    import ast
    return [n.name for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def restore_top_level_names(orig: str, new: str) -> str:
    """The rewriter often renames top-level functions despite the instruction, which only breaks
    the test harness's entry point. A real attacker would trivially put the names back, so when
    the rewrite defines the same number of top-level functions, map them back by position (as
    NAME tokens, so call sites follow). Anything else is left as the model wrote it."""
    import io
    import tokenize
    try:
        a, b = _top_defs(orig), _top_defs(new)
    except SyntaxError:
        return new
    if len(a) != len(b) or a == b:
        return new
    mapping = {nb: na for na, nb in zip(a, b) if na != nb}
    if set(mapping.values()) & (set(b) - set(mapping)):
        return new  # restoring would collide with another function name
    toks = list(tokenize.generate_tokens(io.StringIO(new).readline))
    out = [t._replace(string=mapping.get(t.string, t.string)) if t.type == tokenize.NAME else t for t in toks]
    return tokenize.untokenize(out)


def frac_changed(tok, a: str, b: str):
    """Identical to scripts/04_attack.py:frac_changed."""
    ta = tok(a, add_special_tokens=False).input_ids
    tb = tok(b, add_special_tokens=False).input_ids
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return changed / max(1, len(ta)), len(ta), len(tb)


def gen(n: int, limit: int | None, batch_size: int, cfgs: list[str] = WM_CONFIGS):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wm.data import load_problems, sanitize_solution
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    probs = {p["id"]: p for p in load_problems(["humaneval", "mbpp"])}
    write_manifest("rewrite", {"step": "llm_rewrite", "rewriter": MODEL_ID, "watermark": None,
                               "decoding": {"do_sample": False, "max_new_tokens": 768}, "n_per_config": n,
                               "seed": SEED, "prompt": PROMPT})
    for cfg in cfgs:
        gens = {g["id"]: g for g in read_jsonl(OUT / "generations" / f"{cfg}.jsonl")}
        ids = sample_ids(cfg, n)[:limit]
        path = OUT / "rewrite" / f"{cfg}.jsonl"
        done = done_ids(path)
        todo = [i for i in ids if i not in done]
        log.info(f"[{cfg}] {len(done)} done, {len(todo)} to go")
        t0 = time.time()
        for b in range(0, len(todo), batch_size):
            batch = todo[b: b + batch_size]
            texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(code=gens[i]["solution"], entry=probs[i]["entry_point"])}],
                                             tokenize=False, add_generation_prompt=True) + RESPONSE_PREFIX
                     for i in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=768, stop_strings=[STOP],
                                     tokenizer=tok, pad_token_id=tok.pad_token_id,
                                     temperature=None, top_p=None, top_k=None)
            rows = []
            for i, ids_ in zip(batch, out[:, enc["input_ids"].shape[1]:]):
                raw = tok.decode(ids_[ids_ != tok.pad_token_id], skip_special_tokens=True)
                orig = gens[i]["solution"]
                code = restore_top_level_names(orig, raw.split("```")[0])
                new = sanitize_solution(code, probs[i]["entry_point"])
                f, na, nb = frac_changed(tok, orig, new)
                rows.append({"id": i, "task_id": i, "dataset": gens[i]["dataset"], "config_id": cfg,
                             "attack": "llm_rewrite", "attacked": new, "raw": raw,
                             "status": "ok" if new.strip() and new.strip() != orig.strip() else "unchanged",
                             "frac_tok_changed": round(f, 4), "n_tok_orig": na, "n_tok_attacked": nb,
                             "pass_base": None, "pass_plus": None})
            append_jsonl(path, rows)
            log.info(f"[{cfg}] {b + len(batch)}/{len(todo)}  {time.time() - t0:.0f}s")
        torch.cuda.empty_cache()


def test(workers: int, cfgs: list[str] = WM_CONFIGS):
    import json
    import os

    from wm.evalplus_runner import run_tests
    for cfg in cfgs:
        path = OUT / "rewrite" / f"{cfg}.jsonl"
        rows = read_jsonl(path)
        todo = [r for r in rows if r["status"] == "ok" and r["pass_plus"] is None]
        if not todo:
            continue
        res = {x["key"]: x for x in run_tests([{"key": r["id"], "dataset": r["dataset"], "task_id": r["task_id"],
                                                "solution": r["attacked"]} for r in todo], n_workers=workers)}
        for r in todo:
            r["pass_base"], r["pass_plus"] = res[r["id"]]["pass_base"], res[r["id"]]["pass_plus"]
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        ok = [r for r in rows if r["status"] == "ok"]
        log.info(f"[{cfg}] tested {len(todo)}; pass plus {sum(bool(r['pass_plus']) for r in ok)}/{len(ok)}")


def analyze(B: int = 2000):
    import numpy as np
    import pandas as pd
    an = importlib.import_module("06_analyze")
    unc = importlib.import_module("09_uncertainty")
    rng = np.random.default_rng(SEED)
    rows = []
    for cfg in WM_CONFIGS:
        rw = pd.DataFrame(read_jsonl(OUT / "rewrite" / f"{cfg}.jsonl"))
        if rw.empty:
            continue
        ok = rw[rw.status == "ok"]
        valid = ok[ok.pass_plus == True]  # noqa: E712
        for d in DETECTORS_FOR[GEN_CONFIGS[cfg].method]:
            negz = an.zfile(OUT, d, "neg").z.to_numpy()
            thr = an.emp_threshold(negz)
            thr_b = unc.boot_thresholds(negz, B, rng)
            za = an.zfile(OUT, d, f"rw:{cfg}")
            zg = an.zfile(OUT, d, f"gen:{cfg}").set_index("id").z
            if za.empty:
                continue
            za = za.set_index("id").z
            v = valid[valid.id.isin(za.index)]
            z_att, z_orig = za.loc[v.id].to_numpy(), zg.loc[v.task_id].to_numpy()
            idx = rng.integers(0, len(v), size=(B, len(v)))
            t_att, t_orig = unc.boot_tpr(z_att, thr_b, idx), unc.boot_tpr(z_orig, thr_b, idx)
            r = {"config_id": cfg, "method": GEN_CONFIGS[cfg].method, "delta": GEN_CONFIGS[cfg].delta,
                 "detector": d, "n_sampled": len(rw), "n_rewritten": len(ok), "n_valid": len(v),
                 "survival": len(v) / max(1, len(ok)), "mean_frac_tok_changed": v.frac_tok_changed.mean(),
                 "TPR_orig": (z_orig > thr).mean(), "TPR_rewritten": (z_att > thr).mean(),
                 "mean_z_orig": an.zmean(z_orig), "mean_z_rewritten": an.zmean(z_att),
                 "AUROC_rewritten": an.auroc(z_att, negz)}
            r["orig_lo"], r["orig_hi"] = unc.ci(t_orig)
            r["rw_lo"], r["rw_hi"] = unc.ci(t_att)
            r["change_lo"], r["change_hi"] = unc.ci(t_att - t_orig)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "llm_rewrite.csv", index=False)
    print(df.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "test", "analyze"])
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", default=None, help="comma list (default: the 8 watermarked configs); "
                    "mcw = the post-generation watermark, analysed by 11_multichannel.py")
    args = ap.parse_args()
    if args.step == "gen":
        gen(args.n, args.limit, args.batch_size, args.configs.split(",") if args.configs else WM_CONFIGS)
    elif args.step == "test":
        test(args.workers, args.configs.split(",") if args.configs else WM_CONFIGS)
    else:
        analyze()
    log.info("REWRITE STEP DONE")


if __name__ == "__main__":
    main()
