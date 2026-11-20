"""Run EvalPlus base/plus tests on generations (resumable).

  python scripts/02_evaluate.py --config kgw_d2 [--test]
  python scripts/02_evaluate.py --config all
  python scripts/02_evaluate.py --canonical            # sanity: canonical solutions must pass 100%

Output: outputs/evals/<config_id>.jsonl rows {id, dataset, config_id, base_status, plus_status, pass_base, pass_plus}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import GEN_CONFIGS, OUT  # noqa: E402
from wm.data import human_solution, load_problems  # noqa: E402
from wm.evalplus_runner import run_tests  # noqa: E402
from wm.io import append_jsonl, done_ids, get_logger, read_jsonl  # noqa: E402


def eval_rows(rows, out_path, log, tag):
    done = done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    log.info(f"[{tag}] {len(done)} done, {len(todo)} to test")
    if not todo:
        return
    by_id = {r["id"]: r for r in todo}
    items = [{"key": r["id"], "dataset": r["dataset"], "task_id": r["id"], "solution": r["solution"]}
             for r in todo]
    res = run_tests(items)
    append_jsonl(out_path, [{"id": x["key"], "dataset": by_id[x["key"]]["dataset"],
                             "config_id": by_id[x["key"]].get("config_id", tag),
                             **{k: x[k] for k in ("base_status", "plus_status", "pass_base", "pass_plus")}}
                            for x in res])
    allr = read_jsonl(out_path)
    for ds in ("humaneval", "mbpp"):
        sub = [r for r in allr if r["dataset"] == ds]
        if sub:
            log.info(f"[{tag}] {ds}: n={len(sub)} pass@1 base={sum(r['pass_base'] for r in sub)/len(sub):.3f} "
                     f"plus={sum(r['pass_plus'] for r in sub)/len(sub):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--test", action="store_true", help="use outputs/test (from --limit runs)")
    ap.add_argument("--limit", type=int, default=None, help="canonical only")
    ap.add_argument("--tag", default="", help="test subfolder (with --test)")
    args = ap.parse_args()
    log = get_logger("evaluate")
    root = OUT / "test" / args.tag if args.test else OUT

    if args.canonical:
        probs = load_problems(["humaneval", "mbpp"], args.limit)
        rows = [{"id": p["id"], "dataset": p["dataset"], "config_id": "canonical",
                 "solution": human_solution(p)} for p in probs]
        eval_rows(rows, root / "evals" / "canonical.jsonl", log, "canonical")
        return

    cfgs = list(GEN_CONFIGS) if args.config == "all" else args.config.split(",")
    for c in cfgs:
        gen = read_jsonl(root / "generations" / f"{c}.jsonl")
        if not gen:
            log.info(f"[{c}] no generations yet, skipping")
            continue
        eval_rows(gen, root / "evals" / f"{c}.jsonl", log, c)


if __name__ == "__main__":
    main()
