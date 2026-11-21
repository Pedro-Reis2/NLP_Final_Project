"""Apply attacks and re-run EvalPlus tests on the attacked code (resumable).

  python scripts/04_attack.py --canonical            # sanity: canonical solutions must still pass
  python scripts/04_attack.py --config kgw_d2[,...]  # attack generations that pass the plus tests
  python scripts/04_attack.py --config all

Only generations whose original passes base+plus tests are attacked (others can
never count). Output: outputs/attacks/<config_id>.jsonl rows
  {id: "<task_id>|<attack>", task_id, dataset, attack, status, attacked,
   frac_tok_changed, n_tok_orig, n_tok_attacked, pass_base, pass_plus}
status: ok | not_applicable | unchanged.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.attacks import ATTACKS, AttackError, apply_attack  # noqa: E402
from wm.config import GEN_CONFIGS, MODEL_ID, OUT  # noqa: E402
from wm.data import human_solution, load_problems  # noqa: E402
from wm.evalplus_runner import run_tests  # noqa: E402
from wm.io import append_jsonl, done_ids, get_logger, read_jsonl  # noqa: E402


def frac_changed(tok, a: str, b: str) -> tuple[float, int, int]:
    ta = tok(a, add_special_tokens=False).input_ids
    tb = tok(b, add_special_tokens=False).input_ids
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return changed / max(1, len(ta)), len(ta), len(tb)


def attack_rows(sources: list[dict], tok, out_path: Path, log, tag: str):
    """sources: {task_id, dataset, code}. Applies every attack, tests the results."""
    done = done_ids(out_path)
    pending, rows = [], []
    for s in sources:
        for atk in ATTACKS:
            rid = f"{s['task_id']}|{atk}"
            if rid in done:
                continue
            row = {"id": rid, "task_id": s["task_id"], "dataset": s["dataset"], "attack": atk,
                   "config_id": tag, "attacked": None, "pass_base": None, "pass_plus": None}
            try:
                new = apply_attack(atk, s["code"], salt=s["task_id"])
            except AttackError as e:
                row.update(status="not_applicable", error=str(e)[:200])
                rows.append(row)
                continue
            if new.strip() == s["code"].strip():
                row.update(status="unchanged")
                rows.append(row)
                continue
            f, na, nb = frac_changed(tok, s["code"], new)
            row.update(status="ok", attacked=new, frac_tok_changed=round(f, 4),
                       n_tok_orig=na, n_tok_attacked=nb)
            pending.append(row)
    log.info(f"[{tag}] {len(pending)} attacked programs to test, {len(rows)} n/a or unchanged")
    if pending:
        res = {r["key"]: r for r in run_tests([{"key": r["id"], "dataset": r["dataset"],
                                                "task_id": r["task_id"], "solution": r["attacked"]}
                                               for r in pending])}
        for r in pending:
            r["pass_base"] = res[r["id"]]["pass_base"]
            r["pass_plus"] = res[r["id"]]["pass_plus"]
    append_jsonl(out_path, rows + pending)
    allr = read_jsonl(out_path)
    for atk in ATTACKS:
        sub = [r for r in allr if r["attack"] == atk]
        ok = [r for r in sub if r["status"] == "ok"]
        if sub:
            log.info(f"[{tag}] {atk:15s} applied {len(ok)}/{len(sub)}  still pass plus "
                     f"{sum(bool(r['pass_plus']) for r in ok)}/{len(ok)}  "
                     f"mean frac tok changed {sum(r['frac_tok_changed'] for r in ok) / max(1, len(ok)):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    log = get_logger("attack")
    # Imported here, not at module level: EvalPlus test workers are spawned and
    # re-import this module; a top-level transformers import pulls in torch
    # (~5.6 GB virtual memory), which counts against EvalPlus's 8 GB RLIMIT_AS
    # in every forked test process and made correct programs fail.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    root = OUT / "test" / args.tag if args.test else OUT

    if args.canonical:
        probs = load_problems(["humaneval", "mbpp"], args.limit)
        src = [{"task_id": p["id"], "dataset": p["dataset"], "code": human_solution(p)} for p in probs]
        attack_rows(src, tok, root / "attacks" / "canonical.jsonl", log, "canonical")
        return

    cfgs = list(GEN_CONFIGS) if args.config == "all" else args.config.split(",")
    for c in cfgs:
        gens = {r["id"]: r for r in read_jsonl(root / "generations" / f"{c}.jsonl")}
        evals = read_jsonl(root / "evals" / f"{c}.jsonl")
        if not gens or not evals:
            log.info(f"[{c}] missing generations or evals, skipping")
            continue
        passing = [gens[e["id"]] for e in evals if e["pass_plus"] and e["id"] in gens][: args.limit]
        src = [{"task_id": g["id"], "dataset": g["dataset"], "code": g["solution"]} for g in passing]
        attack_rows(src, tok, root / "attacks" / f"{c}.jsonl", log, c)


if __name__ == "__main__":
    main()
