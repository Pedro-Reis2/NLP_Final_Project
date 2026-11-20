# RESULTS

Living results log. Every number here is generated from `outputs/tables/*.csv` (see `scripts/06_analyze.py`).
Figures live in `figures/`. Structured to become the paper's results section on Day 3.

## Setup (filled as runs complete)
- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct (fp16, greedy, max_new_tokens=512)
- Data: HumanEval+ (164) + MBPP+ (378), EvalPlus extended tests
- Watermarks (MarkLLM): KGW, Unigram, SWEET (generation), EWD (detector on KGW); γ=0.5, δ ∈ {2, 1}, seed 42
- Hardware: RTX 3070 8 GB (WSL2), Ryzen 9 3900X
- Throughput: _TBD_

## QP1: Behaviour preservation (pass@1)
_TBD_

## Detection without attacks (TPR @ 1% FPR, AUROC)
_TBD_

## QP2: Robustness under attacks
_TBD_

## QP3: TPR by code length
_TBD_

## QP4 (optional): text vs code
_Not started._

## Sanity checks
_TBD_
