"""EvalPlus problems, prompts and sanitization (thin wrappers over evalplus)."""
from __future__ import annotations

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import make_raw_chat_prompt
from evalplus.sanitize import sanitize

from .config import EXCLUDED_TASKS

# EvalPlus codegen defaults for instruct models
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following "
                      "problem in a markdown code block:")
RESPONSE_PREFIX = ("Below is a Python script with a self-contained function that solves the "
                   "problem and passes corresponding tests:")

_LOADERS = {"humaneval": get_human_eval_plus, "mbpp": get_mbpp_plus}
_CACHE: dict[str, dict] = {}


def raw_problems(dataset: str) -> dict:
    if dataset not in _CACHE:
        _CACHE[dataset] = _LOADERS[dataset]()
    return _CACHE[dataset]


def load_problems(datasets, limit: int | None = None) -> list[dict]:
    out = []
    for ds in datasets:
        items = [(t, p) for t, p in raw_problems(ds).items() if t not in EXCLUDED_TASKS]
        if limit:
            items = items[:limit]
        for tid, p in items:
            out.append({"id": tid, "dataset": ds, "prompt": p["prompt"],
                        "entry_point": p["entry_point"],
                        "canonical_solution": p["canonical_solution"]})
    return out


def make_prompt(task_prompt: str, tokenizer) -> str:
    """EvalPlus instruct prompt; ends right after the opening ```python fence."""
    return make_raw_chat_prompt(task_prompt, INSTRUCTION_PREFIX, RESPONSE_PREFIX, tokenizer)


def sanitize_solution(code: str, entry_point: str) -> str:
    try:
        return sanitize(code, entry_point)
    except Exception:
        return code


def human_solution(p: dict) -> str:
    """Full human-written program for a problem (canonical solution)."""
    if p["dataset"] == "humaneval":
        return p["prompt"] + p["canonical_solution"]
    return p["canonical_solution"]
