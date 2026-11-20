# RESULTS

Living results log. Every number here is generated from `outputs/tables/*.csv` (see `scripts/06_analyze.py`).
Figures live in `figures/`. Structured to become the paper's results section on Day 3.

## Setup
- **Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct, fp16, greedy decoding, `max_new_tokens=512`, generation stopped at the closing code fence. EvalPlus instruct prompt (instruction prefix + response prefix ending in a ```python fence), output cleaned with `evalplus.sanitize`.
- **Data**: HumanEval+ v0.1.10 and MBPP+ v0.2.0 (EvalPlus). **540 problems** (163 + 377) after excluding HumanEval/32 and Mbpp/255 as harness artifacts (see Sanity checks).
- **pass@1 definitions**: *original* = EvalPlus `base` tests (the original HumanEval/MBPP tests); *extended* = base **and** EvalPlus `plus` tests pass (EvalPlus convention).
- **Watermarks** (MarkLLM 0.1.5, default hash key 15485863, γ=0.5, prefix length 1): KGW, Unigram (fixed green list), SWEET (entropy threshold 0.9), EWD (own generation, detected weighted and unweighted). δ ∈ {2, 4}. δ=4 replaced the planned δ=1 after a 40-problem probe showed δ=2 is already weak on code (mean z ≈ 1.3, ~15–20% above z=2.33). Greedy vs sampling (T=0.2, T=0.7) gave nearly identical z.
- **Negatives**: 540 canonical solutions (HumanEval: prompt + canonical body; MBPP: canonical program) + 1,000 accepted Python submissions from Project CodeNet Python800 (seed 42, ≤2 per problem, 30–1,000 Qwen tokens, parseable; 635 problems; rejected 18 for Python-2 syntax and 106 for length). CodeNet came from a Hugging Face mirror of the official Python800 tarball (`qiankunmu/Project_CodeNet_Python800_and_Java250`), because the IBM CDN is offline.
- **Detection input**: sanitized code only (no prompt). Thresholds set empirically at 1% FPR on the negatives; the analytic z = 2.326 is also reported.
- **Hardware**: RTX 3070 8 GB (WSL2 Ubuntu 24.04), Ryzen 9 3900X. torch 2.11 (cu128), transformers 5.23.
- **Throughput**: batch 32, ~27 s per batch unwatermarked (~250 tok/s), ~38 s with KGW (~165 tok/s); ~8–12 min per config for 540 problems.

## QP1: Behaviour preservation (pass@1)
Unwatermarked baseline (`outputs/evals/nowm.jsonl`):

| Dataset | n | pass@1 original | pass@1 extended |
|---|---|---|---|
| HumanEval+ | 163 | 67.5% | 60.7% |
| MBPP+ | 377 | 64.7% | 55.2% |

Published EvalPlus leaderboard numbers for this model are 70.7/66.5 (HumanEval) and 69.2/59.4 (MBPP); ours are 3–6 points lower. Likely causes: our 512-token cap, stopping rule and sanitize step. The gap applies equally to every config, so watermark-vs-baseline differences are unaffected.

_Watermarked configs: TBD._

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
