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

from .config import RESOURCE_SENSITIVE_TASKS
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
    pass_plus follows EvalPlus: base AND plus tests pass.
    Failures on RESOURCE_SENSITIVE_TASKS are re-tested with 2 workers and a
    20x (instead of 4x) ground-truth time-limit factor: under load their heavy
    extended inputs time out even for canonical solutions."""
    res = _run(items, n_workers, progress)
    by_key = {it["key"]: it for it in items}
    retry = [by_key[r["key"]] for r in res
             if not r["pass_plus"] and by_key[r["key"]]["task_id"] in RESOURCE_SENSITIVE_TASKS]
    if retry:
        again = {r["key"]: r for r in _run(retry, 2, False, RETRY_TIME_FACTOR)}
        res = [again.get(r["key"], r) for r in res]
    return res


RETRY_TIME_FACTOR = 20.0


def _run(items: list[dict], n_workers: int | None, progress: bool,
         time_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR) -> list[dict]:
    n_workers = n_workers or max(1, mp.cpu_count() // 2)
    for ds in {it["dataset"] for it in items}:
        groundtruth(ds)
    out = []
    # spawn, not fork: a forked worker inherits the parent's address space (torch
    # alone maps GBs of virtual memory), which counts against EvalPlus's
    # RLIMIT_AS cap and makes heavy tests fail in scripts that import torch.
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {}
        for it in items:
            ds = it["dataset"]
            prob = raw_problems(ds)[it["task_id"]]
            args = (ds, 0, prob, it["solution"], groundtruth(ds)[it["task_id"]], False, True,
                    it["key"], DEFAULT_MIN_TIME_LIMIT, time_factor)
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
