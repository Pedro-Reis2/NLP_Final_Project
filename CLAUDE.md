# CLAUDE.md: Watermarking LLM-generated code (empirical study)

Solo NLP final project (INF/UFG). **3 days total**: Day 1–2 code and experiments, Day 3 writing only.
Scope source of truth: `docs/pitch.pdf` (approved). `docs/related-work.pdf` §6 "Proposta" is an older, bigger scope; where they disagree, follow the pitch and this file.
**Finished results beat completeness.** Cut scope aggressively when behind.

## Research questions
- **QP1 (behaviour)**: pass@1 with original vs EvalPlus-extended tests, per method/δ. How much do the original tests underestimate the damage?
- **QP2 (robustness)**: TPR at 1% FPR (and AUROC) per method under the same attack set, with and without comments.
- **QP3 (length)**: TPR by code length. How many tokens are needed for TPR ≥ 80% at 1% FPR?
- **QP4 (text vs code)**: OPTIONAL, see below.

## CORE (must finish by end of Day 2)
- **Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, fp16, single model for everything.
- **Data**: HumanEval+ (164) + MBPP+ (378) = 542 problems, EvalPlus extended tests. If time gets tight: HumanEval+ only.
- **Methods (MarkLLM only, never reimplement)**: KGW, Unigram (fixed list), SWEET. EWD = extra detector on the KGW generations (no extra generation). Plus an unwatermarked baseline generation.
- **δ ∈ {2, 4}** (all methods). δ=4 replaced the planned δ=1 (user decision, Day 1): a 40-problem probe gave mean z ≈ 1.3 and ~15–20% TPR at δ=2, so δ=1 would leave nothing for attacks to remove. δ=4 is queued after δ=2 and is the first thing to drop.
- **Generation configs (9)**: `nowm`, then `kgw/ewd/unigram/sweet` × δ. EWD has its own generation (MarkLLM's KGW and EWD use different green-list hashes) and is scored as EWD (weighted) and EWD-unweighted (plain z on the same flags).
- **Problems: 540** (HumanEval/32 and Mbpp/255 excluded as harness artifacts; see RESULTS.md).
- **Fixed settings**: γ = 0.5, MarkLLM default hash key (15485863), KGW/EWD prefix length 1, SWEET entropy threshold 0.9, seed 42. Greedy decoding, n=1, `max_new_tokens=512`, stop at the closing ``` fence. EvalPlus instruct prompt + `evalplus.sanitize`.
- **Logging**: every output row carries `config_id`. Every run writes `outputs/manifests/<config_id>.json` (method, γ, δ, seed, key, entropy threshold, decoding params, model id, library versions, git SHA).
- **Negatives**: 542 EvalPlus canonical solutions + ~1,000 accepted Python files from Project CodeNet (Python800), 30–1,000 tokens. **Thresholds are set empirically at 1% FPR** on these negatives. Also report the analytic z=2.33 and the FPR on unwatermarked Qwen generations.
- **Detection input**: the sanitized code only (no prompt), for positives and negatives alike.
- **Attacks**: (1) comment + docstring removal, (2) `black`, (3) renaming of 1 / 2 / all identifiers, (4) dead code + try/except wrapping. **An attacked sample only counts if the original passes the plus tests and the attacked version also passes the plus tests.** Report survival rate per attack.
- **Metrics**: pass@1 (base vs plus tests), TPR @ 1% FPR, AUROC, TPR by code length (QP3).
- **Figures** (PNG + PDF, publication-ready): pass@1 table, TPR under each attack, TPR by length.

## OPTIONAL (only if CORE is done with time left on Day 2)
- QP4: KGW and Unigram, 300 continuations × 200 tokens from news prefixes (MarkLLM's bundled C4 sample). Attack = random word substitution. Intensity = fraction of tokens changed, comparable with the code attacks.

## CUT (future work; mention in the paper, do not build)
Multichannel scheme, LLM-rewrite attack, STONE, ACW, other languages, bigger models, δ ∈ {0.5, 1}.

## Fallback rules
- If full generation hasn't started by Day 1 H4 → drop δ=4.
- If attacks aren't evaluated by Day 2 noon → restrict attack analysis to HumanEval+, then drop δ=4 from the attack grid.
- Never run two test-heavy jobs (EvalPlus) at the same time: heavy tasks time out under load.
- Never block on OPTIONAL work.

## Working rules
1. **Resumable**: every step reads inputs from disk and appends to an output JSONL keyed by id. On restart it skips ids already present. Nothing important lives only in memory.
2. **Test on 5 first**: every script has `--limit N`. Run with `--limit 5` and inspect before any full run.
3. **Long GPU jobs**: run in the background (`nohup ... > logs/<step>.log 2>&1 &`) with progress logged to that file. **Ask the user before starting one.** Keep building the next step while it runs.
4. **Commit after each working piece** (small commits, descriptive messages).
5. **RESULTS.md** holds every number, table and figure path, generated from `outputs/tables/*.csv`. It is written so it can become the paper's results section on Day 3.
6. MarkLLM for all watermarking and detection; EvalPlus for all test execution. No reimplementations of either.

## Environment
- Runs inside **WSL Ubuntu 24.04** (EvalPlus's executor needs Unix). GPU: RTX 3070 8 GB via WSL CUDA passthrough.
- Repo: `/mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project`
- venv: `~/wm-venv` (`source ~/wm-venv/bin/activate`). `HF_HOME=~/hf`.
- Run scripts from the repo root: `python scripts/01_generate.py --config kgw_d2 --limit 5`.

## Layout
```
src/wm/config.py      config registry → config_id (γ, δ, key, seed, decoding)
src/wm/io.py          resumable JSONL + manifests
src/wm/attacks.py     attack implementations
scripts/00_env_check.py … 07_figures.py  (pipeline steps, in order)
tests/                pytest (attacks keep canonical solutions passing)
outputs/              generations/ evals/ negatives/ attacks/ detect/ tables/ manifests/   (committed, small)
figures/              final figures (committed)
data/ logs/           raw downloads, logs (gitignored)
```

## Verification checklist (run at each step)
- Canonical solutions pass 100% (base + plus), also after every attack.
- Baseline Qwen pass@1 is close to the published EvalPlus numbers (≈66% HE+, ≈59% MBPP+).
- KGW z on human code: mean ≈ 0, std ≈ 1. Unwatermarked generations: AUROC vs negatives ≈ 0.5. Watermarked mean z ≫ 4.
- Re-scoring the same text gives the identical z. Re-running a step skips finished ids.
- Unattacked TPR ≥ attacked TPR. Every number in RESULTS.md traces to a CSV in `outputs/tables/`.
