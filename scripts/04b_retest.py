"""Re-test every recorded EvalPlus failure on an otherwise idle machine.

  python scripts/04b_retest.py --kind attacks|evals [--config all] [--workers 6]

Why: the first test pass for most configs ran while GPU generation/detection
was also running. Under that CPU load, correct programs hit EvalPlus time limits
(a 60-row sample of recorded attack failures for kgw_d2 passed 60/60 on re-test).
A failure is only kept if it fails again here. Rows that flip get
"retested": true plus the new statuses; the file is rewritten atomically.
Idempotent: rows already marked retested are not tested again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import GEN_CONFIGS, OUT  # noqa: E402
from wm.evalplus_runner import run_tests  # noqa: E402
from wm.io import get_logger, read_jsonl  # noqa: E402

log = get_logger("retest")


def solution_of(kind, row, gens):
    if kind == "attacks":
        return row["attacked"]
    return gens[row["id"]]["solution"]


def retest(kind: str, cfg: str, workers: int):
    path = OUT / kind / f"{cfg}.jsonl"
    if not path.exists():
        return
    rows = read_jsonl(path)
    gens = {}
    if kind == "evals":
        gens = {g["id"]: g for g in read_jsonl(OUT / "generations" / f"{cfg}.jsonl")} if cfg != "canonical" else {}
        if cfg == "canonical":
            return
    todo = [r for r in rows if not r.get("pass_plus") and not r.get("retested_once")
            and (kind == "evals" or r.get("status") == "ok")]
    if not todo:
        log.info(f"[{kind}:{cfg}] nothing to re-test")
        return
    items = [{"key": r["id"], "dataset": r["dataset"],
              "task_id": r.get("task_id", r["id"]), "solution": solution_of(kind, r, gens)} for r in todo]
    res = {x["key"]: x for x in run_tests(items, n_workers=workers, progress=False)}
    flipped_base = flipped_plus = 0
    for r in todo:
        x = res[r["id"]]
        r["retested_once"] = True
        if x["pass_base"] and not r.get("pass_base"):
            flipped_base += 1
        if x["pass_plus"]:
            flipped_plus += 1
        if x["pass_base"] != r.get("pass_base") or x["pass_plus"] != r.get("pass_plus"):
            r["retested"] = True
            r["first_pass_base"], r["first_pass_plus"] = r.get("pass_base"), r.get("pass_plus")
        r["pass_base"] = x["pass_base"] or r.get("pass_base", False)
        r["pass_plus"] = x["pass_plus"]
        r["retest_status"] = f'{x["base_status"]}/{x["plus_status"]}'
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    log.info(f"[{kind}:{cfg}] re-tested {len(todo)} failures: now pass base +{flipped_base}, plus {flipped_plus}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["attacks", "evals"], required=True)
    ap.add_argument("--config", default="all")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfgs = list(GEN_CONFIGS) if args.config == "all" else args.config.split(",")
    for c in cfgs:
        retest(args.kind, c, args.workers)


if __name__ == "__main__":
    main()
