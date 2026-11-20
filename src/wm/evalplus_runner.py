"""Run EvalPlus base + plus tests on arbitrary (task_id, solution) lists.

Uses EvalPlus's own get_groundtruth / check_correctness with the same
arguments as `evalplus.evaluate` (fast_check=True, default time limits); it
only drops evaluate()'s requirement that every task be present, so subsets
(e.g. attacked samples) can be tested.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from evalplus.data import get_human_eval_plus_hash, get_mbpp_plus_hash
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.eval import PASS
from evalplus.evaluate import (DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT,
                               check_correctness, get_groundtruth)
from tqdm import tqdm

from .data import raw_problems

_GT: dict[str, dict] = {}


def groundtruth(dataset: str) -> dict:
    if dataset not in _GT:
        problems = raw_problems(dataset)
        if dataset == "humaneval":
            _GT[dataset] = get_groundtruth(problems, get_human_eval_plus_hash(), [])
        else:
            _GT[dataset] = get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)
    return _GT[dataset]


def run_tests(items: list[dict], n_workers: int | None = None, progress: bool = True) -> list[dict]:
    """items: dicts with keys key, dataset, task_id, solution.
    Returns dicts {key, base_status, plus_status, pass_base, pass_plus}.
    pass_plus follows EvalPlus: base AND plus tests pass."""
    n_workers = n_workers or max(1, mp.cpu_count() // 2)
    for ds in {it["dataset"] for it in items}:
        groundtruth(ds)
    out = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {}
        for it in items:
            ds = it["dataset"]
            prob = raw_problems(ds)[it["task_id"]]
            args = (ds, 0, prob, it["solution"], groundtruth(ds)[it["task_id"]], False, True,
                    it["key"], DEFAULT_MIN_TIME_LIMIT, DEFAULT_GT_TIME_LIMIT_FACTOR)
            futs[ex.submit(check_correctness, *args)] = it["key"]
        it_ = as_completed(futs)
        if progress:
            it_ = tqdm(it_, total=len(futs), desc="tests", mininterval=5)
        for f in it_:
            r = f.result()
            b, p = r["base"][0], r["plus"][0]
            out.append({"key": futs[f], "base_status": b, "plus_status": p,
                        "pass_base": b == PASS, "pass_plus": b == PASS and p == PASS})
    return out
