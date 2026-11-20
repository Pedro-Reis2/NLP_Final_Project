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
- **Test harness**: all 541 canonical solutions pass both original and extended tests (163 HumanEval+, 378 MBPP+; `outputs/evals/canonical.jsonl`).
  - EvalPlus's default 4 GB address-space cap made 6 canonical solutions fail on the heaviest extended inputs (HumanEval/15, 100, 130; Mbpp/84, 255, 392). Raised to 8 GB (`EVALPLUS_MAX_MEMORY_BYTES`, `scripts/wsl_run.sh`).
  - **HumanEval/32 (`find_zero`) is excluded.** Its canonical solution passes all 100 base inputs outside the sandbox, but EvalPlus's sandbox fails it before the first test in our setup. Final problem count: **541** (163 + 378).
- **Watermark pipeline positive control** (`scripts/_sanity_prose.py`): 200-token prose continuations, 3 prompts, δ=2. z-scores: KGW greedy 5.9–6.2 / sampled 8.0–9.1; EWD greedy 6.0–8.2 / sampled 7.1–9.6; Unigram greedy 3.7–7.2 / sampled 4.5–9.3; SWEET greedy 6.0–7.0 / sampled 7.2–9.5. The batched generation + MarkLLM detection path works; greedy decoding gives a weaker signal than sampling.
- **Implementation note (MarkLLM)**: KGW seeds its green list with `hash_key · prf[prev_token]`, while EWD and SWEET use `hash_key · prev_token`. EWD therefore cannot score MarkLLM-KGW text. EWD gets its own generation config (MarkLLM `EWD` class = KGW soft-red-list biasing with the EWD hash). On those texts we report both the entropy-weighted EWD z and the unweighted green-count z ("EWD-unweighted"), a paired detector comparison on identical texts.
- **Prompt copying**: under the EvalPlus instruct prompt, the model usually restates the HumanEval signature and docstring, so detection also scores that low-entropy copied text. Human negatives built as prompt + canonical solution contain the same docstring, so the comparison is like for like; the comment/docstring-removal attack isolates the effect.
