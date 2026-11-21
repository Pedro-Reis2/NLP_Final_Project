"""Differential testing (pitch, metrics): do programs that pass the EvalPlus extended tests still
agree with the reference solution on FRESH inputs, and does watermarking add such hidden bugs?

  python scripts/13_differential.py inputs     # fresh inputs per problem (EvalPlus type-aware mutation)
  python scripts/13_differential.py test       # re-test every program that passes the plus tests
  python scripts/13_differential.py analyze    # outputs/tables/difftest.csv

Fresh inputs: EvalPlus's own TypedMutGen, seeded with each problem's base + plus inputs, keeps
only inputs that satisfy the problem's contract (EvalPlus's input-validity asserts) and on which
the canonical solution runs; expected outputs come from EvalPlus's trusted_exec on the canonical
solution (the same differential-testing oracle EvalPlus uses to build its plus tests). N_NEW new
inputs per problem, seeded per task; generation runs in a child process with a hard timeout.
Testing uses EvalPlus's check_correctness with these inputs as the "plus" set, so a program passes
only if it matches the reference on the base inputs and on every fresh input.
Inputs are pickled under data/difftest/ (gitignored; regenerated deterministically).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import multiprocessing as mp
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm.config import DATA, GEN_CONFIGS, OUT, SEED  # noqa: E402
from wm.io import append_jsonl, done_ids, get_logger, read_jsonl  # noqa: E402

N_NEW = 30
N_SEEDS = 60
GEN_TIMEOUT = 90  # seconds per problem
CFGS = ["canonical"] + list(GEN_CONFIGS) + ["mcw"]  # canonical = sanity check, must pass 100%
IN_DIR = DATA / "difftest"
log = get_logger("difftest")


def contract_code(p: dict) -> str:
    """Canonical solution with the problem's contract asserts at the top of the entry function."""
    if p["dataset"] == "humaneval":
        return p["prompt"] + p["contract"] + p["canonical_solution"]
    src, ep = p["canonical_solution"], p["entry_point"]
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        if re.match(rf"\s*def {re.escape(ep)}\s*\(", ln):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                break
            body_ind = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            contract = [body_ind + c.strip() for c in p["contract"].split("\n") if c.strip()]
            # the def line may span several lines: insert after the line ending the signature
            k = i
            while not lines[k].rstrip().endswith(":"):
                k += 1
            return "\n".join(lines[: k + 1] + contract + lines[k + 1:])
    return src


def _gen_child(p: dict, q):
    import random

    from evalplus.gen.util import trusted_exec
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.type_mut import TypedMutGen
    from evalplus.gen.util import trusted_check_exec
    random.seed(int(hashlib.md5(f"{SEED}|{p['task_id']}".encode()).hexdigest(), 16) % 2**32)
    seeds = list(p["base_input"]) + random.sample(list(p["plus_input"]), min(N_SEEDS, len(p["plus_input"])))
    code = contract_code(p)
    canon = p["prompt"] + p["canonical_solution"] if p["dataset"] == "humaneval" else p["canonical_solution"]
    g = TypedMutGen(seeds, p["entry_point"], code)
    new = []
    for x in g.generate(2 * N_NEW):  # contract-valid, unseen inputs (may loop: parent enforces timeout)
        if x not in new and trusted_check_exec(canon, [x], p["entry_point"]):
            new.append(x)
    new = new[:N_NEW]
    out, t = trusted_exec(canon, new, p["entry_point"], record_time=True,
                          output_not_none=p["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS)
    q.put(pickle.dumps({"inputs": new, "expected": out, "time": t}))


def inputs(limit: int | None = None):
    from wm.data import load_problems, raw_problems
    IN_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "difftest" / "inputs_meta.jsonl"
    done = done_ids(meta_path)
    todo = [p for p in load_problems(["humaneval", "mbpp"], limit) if p["id"] not in done]
    ctx = mp.get_context("spawn")
    running = []

    def reap(block):
        for item in list(running):
            pr, q, p, t0 = item
            if not block and pr.is_alive() and time.time() - t0 < GEN_TIMEOUT:
                continue
            pr.join(max(0.0, GEN_TIMEOUT - (time.time() - t0)))
            row = {"id": p["task_id"], "dataset": p["dataset"]}
            if pr.is_alive():
                pr.terminate()
                row.update(status="timeout", n_new=0)
            else:
                try:
                    d = pickle.loads(q.get(timeout=5))
                    (IN_DIR / f"{p['task_id'].replace('/', '_')}.pkl").write_bytes(pickle.dumps(d))
                    row.update(status="ok", n_new=len(d["inputs"]))
                except Exception as e:  # child crashed
                    row.update(status=f"error:{type(e).__name__}", n_new=0)
            append_jsonl(meta_path, [row])
            running.remove(item)

    for p in todo:
        raw = dict(raw_problems(p["dataset"])[p["id"]], dataset=p["dataset"])
        while len(running) >= 6:
            reap(False)
            time.sleep(0.2)
        q = ctx.Queue()
        pr = ctx.Process(target=_gen_child, args=(raw, q), daemon=True)
        pr.start()
        running.append((pr, q, raw, time.time()))
    while running:
        reap(True)
    meta = read_jsonl(meta_path)
    log.info(f"inputs: {sum(m['status'] == 'ok' and m['n_new'] > 0 for m in meta)}/{len(meta)} problems with fresh inputs, "
             f"mean {sum(m['n_new'] for m in meta) / max(1, len(meta)):.1f} each")


def _solutions(cfg: str) -> dict:
    if cfg == "canonical":
        from wm.data import human_solution, load_problems
        return {p["id"]: {"solution": human_solution(p)} for p in load_problems(["humaneval", "mbpp"])}
    return {g["id"]: g for g in read_jsonl(OUT / "generations" / f"{cfg}.jsonl")}


def test(workers: int):
    from wm.evalplus_runner import run_custom
    meta = {m["id"]: m for m in read_jsonl(OUT / "difftest" / "inputs_meta.jsonl")}
    fresh = {}
    for tid, m in meta.items():
        if m["status"] == "ok" and m["n_new"] > 0:
            fresh[tid] = pickle.loads((IN_DIR / f"{tid.replace('/', '_')}.pkl").read_bytes())
    for cfg in CFGS:
        gens = _solutions(cfg)
        ev = read_jsonl(OUT / "evals" / f"{cfg}.jsonl")
        path = OUT / "difftest" / f"{cfg}.jsonl"
        done = done_ids(path)
        items = [{"key": e["id"], "dataset": e["dataset"], "task_id": e["id"], "solution": gens[e["id"]]["solution"],
                  "plus_inputs": fresh[e["id"]]["inputs"], "plus_expected": fresh[e["id"]]["expected"],
                  "plus_time": fresh[e["id"]]["time"]}
                 for e in ev if e["pass_plus"] and e["id"] in fresh and e["id"] not in done]
        if not items:
            continue
        res = run_custom(items, n_workers=workers)
        append_jsonl(path, [{"id": r["key"], "config_id": cfg, "pass_fresh": r["pass_plus"],
                             "status": f"{r['base_status']}/{r['plus_status']}"} for r in res])
        log.info(f"[{cfg}] {len(items)} tested, {sum(not r['pass_plus'] for r in res)} fail fresh inputs")


def retest(workers: int):
    """Failures re-tested once at low parallelism (load-induced timeouts, as in 04b_retest)."""
    import json
    import os

    from wm.evalplus_runner import run_custom
    meta = {m["id"]: m for m in read_jsonl(OUT / "difftest" / "inputs_meta.jsonl")}
    for cfg in CFGS:
        path = OUT / "difftest" / f"{cfg}.jsonl"
        rows = read_jsonl(path)
        todo = [r for r in rows if not r["pass_fresh"] and not r.get("retested_once")]
        if not todo:
            continue
        gens = _solutions(cfg)
        items = []
        for r in todo:
            d = pickle.loads((IN_DIR / f"{r['id'].replace('/', '_')}.pkl").read_bytes())
            items.append({"key": r["id"], "dataset": meta[r["id"]]["dataset"], "task_id": r["id"],
                          "solution": gens[r["id"]]["solution"], "plus_inputs": d["inputs"],
                          "plus_expected": d["expected"], "plus_time": d["time"]})
        res = {x["key"]: x for x in run_custom(items, n_workers=workers)}
        for r in todo:
            r["retested_once"] = True
            if res[r["id"]]["pass_plus"]:
                r.update(pass_fresh=True, retested=True, first_status=r["status"])
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
        log.info(f"[{cfg}] re-tested {len(todo)}: {sum(res[r['id']]['pass_plus'] for r in todo)} now pass")


def analyze(B: int = 2000):
    import numpy as np
    import pandas as pd
    unc = importlib.import_module("09_uncertainty")
    rng = np.random.default_rng(SEED)
    meta = pd.DataFrame(read_jsonl(OUT / "difftest" / "inputs_meta.jsonl"))
    d = {c: pd.DataFrame(read_jsonl(OUT / "difftest" / f"{c}.jsonl")) for c in CFGS}
    d = {c: x.set_index("id") for c, x in d.items() if not x.empty}
    base = d["nowm"]
    rows = []
    for c, x in d.items():
        f = (~x.pass_fresh.astype(bool)).to_numpy(float)
        idx = rng.integers(0, len(f), size=(B, len(f)))
        lo, hi = unc.ci(f[idx].mean(1))
        ev = pd.DataFrame(read_jsonl(OUT / "evals" / f"{c}.jsonl"))
        n_all = len(ev)
        r = {"config_id": c, "method": (GEN_CONFIGS[c].method or "none") if c in GEN_CONFIGS else
             ("Multicanal" if c == "mcw" else "canonical"),
             "delta": GEN_CONFIGS[c].delta if c in GEN_CONFIGS else np.nan, "n_pass_plus_tested": len(x),
             "n_fail_fresh": int(f.sum()), "frac_fail_fresh": f.mean(), "lo": lo, "hi": hi,
             "pass@1_plus": ev.pass_plus.mean(),
             "pass@1_fresh": (len(x) - f.sum()) / n_all}
        if c not in ("nowm", "canonical"):
            common = sorted(set(x.index) & set(base.index))
            a = (~base.loc[common].pass_fresh.astype(bool)).to_numpy(float)
            b = (~x.loc[common].pass_fresh.astype(bool)).to_numpy(float)
            idx = rng.integers(0, len(common), size=(B, len(common)))
            r["n_common_with_nowm"] = len(common)
            r["diff_vs_nowm_common"] = (b - a).mean()
            r["diff_lo"], r["diff_hi"] = unc.ci((b - a)[idx].mean(1))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "difftest.csv", index=False)
    print(f"problems with fresh inputs: {(meta.n_new > 0).sum()}/{len(meta)}, mean {meta.n_new.mean():.1f}")
    print(df.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["inputs", "test", "retest", "analyze"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="inputs: first N problems per dataset")
    a = ap.parse_args()
    if a.step == "inputs":
        inputs(a.limit)
    elif a.step == "test":
        test(a.workers)
    elif a.step == "retest":
        retest(a.workers)
    else:
        analyze()
    log.info("DIFFTEST STEP DONE")


if __name__ == "__main__":
    main()
