"""Build the human-written negative set.

  python scripts/03_negatives.py

- canonical: the 542 EvalPlus canonical solutions (HumanEval: prompt + body; MBPP: full solution)
- codenet:   1,000 accepted Python3 submissions from Project CodeNet Python800,
             30-1000 Qwen tokens, parseable by ast, <= 2 per problem, seed 42.
             Read straight from data/codenet/Project_CodeNet_Python800.tar.gz
             (HF mirror qiankunmu/Project_CodeNet_Python800_and_Java250; official IBM CDN is offline).

Output: outputs/negatives/negatives.jsonl  rows {id, source, code, n_tokens}
"""
from __future__ import annotations

import argparse
import ast
import random
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from wm.config import DATA, MODEL_ID, OUT, SEED  # noqa: E402
from wm.data import human_solution, load_problems  # noqa: E402
from wm.io import append_jsonl, get_logger, read_jsonl, write_manifest  # noqa: E402

TAR = DATA / "codenet" / "Project_CodeNet_Python800.tar.gz"
MIN_TOK, MAX_TOK, PER_PROBLEM = 30, 1000, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-codenet", type=int, default=1000)
    args = ap.parse_args()
    log = get_logger("negatives")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    out = OUT / "negatives" / "negatives.jsonl"
    if read_jsonl(out):
        log.info(f"{out} exists, nothing to do (delete it to rebuild)")
        return

    rows = []
    for p in load_problems(["humaneval", "mbpp"]):
        code = human_solution(p)
        rows.append({"id": f"canon/{p['id']}", "source": "canonical", "dataset": p["dataset"],
                     "code": code, "n_tokens": len(tok(code, add_special_tokens=False).input_ids)})
    log.info(f"canonical: {len(rows)}")

    with tarfile.open(TAR, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".py")]
        random.Random(SEED).shuffle(members)
        per_prob, picked, rejected = Counter(), [], Counter()
        for m in members:
            if len(picked) >= args.n_codenet:
                break
            prob = m.name.split("/")[-2]
            if per_prob[prob] >= PER_PROBLEM:
                continue
            code = tf.extractfile(m).read().decode("utf-8", errors="replace")
            try:
                ast.parse(code)
            except SyntaxError:
                rejected["syntax"] += 1
                continue
            n = len(tok(code, add_special_tokens=False).input_ids)
            if not (MIN_TOK <= n <= MAX_TOK):
                rejected["length"] += 1
                continue
            per_prob[prob] += 1
            picked.append({"id": f"codenet/{prob}/{Path(m.name).stem}", "source": "codenet",
                           "dataset": "codenet", "code": code, "n_tokens": n})
    log.info(f"codenet: {len(picked)} picked from {len(per_prob)} problems; rejected {dict(rejected)}")
    rows += picked
    append_jsonl(out, rows)
    write_manifest("negatives", {"step": "negatives", "n_canonical": len(rows) - len(picked),
                                 "n_codenet": len(picked), "min_tok": MIN_TOK, "max_tok": MAX_TOK,
                                 "per_problem": PER_PROBLEM, "seed": SEED,
                                 "codenet_source": "hf:qiankunmu/Project_CodeNet_Python800_and_Java250"})


if __name__ == "__main__":
    main()
