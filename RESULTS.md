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

All configs (`outputs/tables/pass_at_1.csv`; figure `figures/fig_pass_at_1.{png,pdf}`, LaTeX `figures/pass_at_1.tex`). pass@1 in %, all 540 problems; Δ = drop vs. no watermark (positive = worse).

| Config | original | extended | gap (orig − ext) | Δ original | Δ extended |
|---|---|---|---|---|---|
| No watermark | 65.6 | 56.9 | 8.7 | – | – |
| KGW δ=2 | 62.4 | 53.3 | 9.1 | 3.1 | 3.5 |
| EWD δ=2 | 62.6 | 56.1 | 6.5 | 3.0 | 0.7 |
| Unigram δ=2 | 63.0 | 52.8 | 10.2 | 2.6 | 4.1 |
| SWEET δ=2 | 68.3 | 59.1 | 9.3 | −2.8 | −2.2 |
| KGW δ=4 | 45.7 | 38.9 | 6.9 | 19.8 | 18.0 |
| EWD δ=4 | 41.9 | 37.0 | 4.8 | 23.7 | 19.8 |
| Unigram δ=4 | 36.3 | 29.6 | 6.7 | 29.3 | 27.2 |
| SWEET δ=4 | 63.3 | 55.0 | 8.3 | 2.2 | 1.9 |

Per dataset (original / extended): HumanEval+ — nowm 67.5/60.7, KGW2 62.6/55.8, EWD2 62.0/57.1, Uni2 64.4/55.2, SWEET2 70.6/62.6, KGW4 40.5/33.7, EWD4 43.6/39.9, Uni4 35.0/30.1, SWEET4 65.6/60.1. MBPP+ — nowm 64.7/55.2, KGW2 62.3/52.3, EWD2 62.9/55.7, Uni2 62.3/51.7, SWEET2 67.4/57.6, KGW4 48.0/41.1, EWD4 41.1/35.8, Uni4 36.9/29.4, SWEET4 62.3/52.8.

**Findings (QP1).**
- δ=2 costs little: 2.6–3.1 points with the original tests and 0.7–4.1 with the extended tests. SWEET δ=2 is *above* the baseline (+2.8/+2.2): biasing only high-entropy tokens leaves syntax-critical, low-entropy tokens alone. With n=540 a ±3-point difference is within noise (one problem = 0.19 points), so read it as "no measurable cost".
- δ=4 is destructive for the methods that bias every token: −20 (KGW), −24 (EWD), −29 (Unigram) points. SWEET δ=4 loses only ~2 points: entropy gating is what makes a strong watermark compatible with code.
- **The original tests do not hide extra watermark damage.** The original-vs-extended gap is 8.7 points for the baseline and 4.8–10.2 for watermarked configs, i.e. the extended tests lower every config by a similar amount. The drop measured with the extended tests is within 4 points of the drop measured with the original tests for every config, and for 5 of 8 configs (all four δ=4 configs and EWD δ=2) it is *smaller*; KGW δ=2, Unigram δ=2 and SWEET δ=2 show a slightly larger drop with the extended tests (+0.4, +1.5, +0.6 points). The answer to "how much do the original tests underestimate the damage?" is: they do not, in this setting; if anything they slightly overstate it. The strong δ=4 configs even have a *smaller* gap, because the problems they still solve are the easier ones.

## Detection without attacks (TPR @ 1% FPR, AUROC)
Thresholds (`outputs/tables/thresholds.csv`), set on 1,540 human negatives:

| Detector | neg mean z | neg sd z | neg frac > 2.326 | empirical threshold (1% FPR) |
|---|---|---|---|---|
| KGW | 0.28 | 1.19 | 3.8% | 2.95 |
| EWD-unweighted | 0.03 | 1.34 | 2.5% | 2.83 |
| EWD | −0.06 | 1.03 | 0.7% | 2.21 |
| Unigram | −0.31 | 1.42 | 2.5% | 2.90 |
| SWEET | −0.05 | 1.03 | 0.9% | 2.29 |

Detectors that count every token (KGW, EWD-unweighted, Unigram) are **miscalibrated on human code**: sd 1.2–1.4 instead of 1, so the textbook z = 2.33 threshold gives 2.5–3.8% FPR, not 1%. Code is repetitive (the same token pairs recur), which breaks the independence assumption behind the z-test. The entropy-aware detectors (SWEET, EWD) skip or down-weight those low-entropy tokens and are calibrated (sd ≈ 1.03, FPR ≈ 0.7–0.9%). All TPRs below use the empirical thresholds.

Detection on all 540 generations of each config (`outputs/tables/detection.csv`); "passing" = only the generations that pass the extended tests (the population the attacks start from). FPR on unwatermarked Qwen generations at the same threshold: 0.4–1.1% for every detector, and AUROC of unwatermarked generations vs. human code is 0.41–0.51, so the detectors respond to the watermark and not to "LLM-ness".

| Config | Detector | mean z | TPR@1%FPR | AUROC | TPR@1%FPR (passing only) |
|---|---|---|---|---|---|
| KGW δ=2 | KGW | 1.40 | 9.6% | 0.741 | 5.9% |
| EWD δ=2 | EWD-unweighted | 1.23 | 13.1% | 0.735 | 8.6% |
| EWD δ=2 | EWD | 1.78 | 38.0% | 0.869 | 30.7% |
| Unigram δ=2 | Unigram | 0.65 | 4.4% | 0.692 | 1.1% |
| SWEET δ=2 | SWEET | 1.35 | 20.7% | 0.820 | 17.2% |
| KGW δ=4 | KGW | 3.42 | 49.6% | 0.921 | 34.8% |
| EWD δ=4 | EWD-unweighted | 3.85 | 61.3% | 0.917 | 47.0% |
| EWD δ=4 | EWD | 3.87 | 79.3% | 0.967 | 71.0% |
| Unigram δ=4 | Unigram | 3.63 | 56.1% | 0.955 | 34.4% |
| SWEET δ=4 | SWEET | 2.04 | 43.3% | 0.883 | 32.0% |

**Findings.**
- At δ=2 code watermarks are nearly undetectable at 1% FPR (1–31% on passing code). The prose positive control at the same δ gives z 4–9: code's low entropy leaves little room to bias.
- **Entropy weighting is the single biggest lever.** On *identical texts* EWD more than triples the TPR of the unweighted z-test at δ=2 (30.7% vs 8.6%) and raises it from 47% to 71% at δ=4.
- Correct (passing) code is consistently harder to detect than all generations: the watermark that pushes a token off the model's preferred choice is also what breaks programs, so the surviving correct programs carry less signal.
- Trade-off: SWEET is the only method with both near-zero pass@1 cost and non-trivial detection; EWD δ=4 detects best but loses 20 points of pass@1.

## QP2: Robustness under attacks
`outputs/tables/attacks.csv`; figures `figures/fig_tpr_attacks_d2.{png,pdf}`, `figures/fig_tpr_attacks_d4.{png,pdf}`. Attacks are applied to generations that pass the extended tests; "No attack" = TPR on those passing generations. Every attacked program still passes the extended tests (**survival 100% for every attack and config**; 2,764/2,764 for the canonical solutions too), so all attacks are semantics-preserving and every applied attack counts. `black` is a no-op on 17–44% of programs (already formatted) and `strip_comments` on 3–25% (no comments); TPR is computed over the programs the attack actually changed, compared with the same programs unattacked.

TPR @ 1% FPR (%) on the same programs, unattacked → attacked, **δ=4**:

| Detector | Strip comments | black | Rename 1 | Rename 2 | Rename all | Dead code + try |
|---|---|---|---|---|---|---|
| KGW | 43.5 → 5.0 | 47.1 → 42.1 | 34.8 → 32.9 | 34.8 → 28.1 | 34.8 → 21.9 | 34.8 → 31.0 |
| EWD-unweighted | 48.2 → 8.8 | 54.4 → 38.8 | 47.0 → 41.5 | 47.0 → 39.0 | 47.0 → 36.5 | 47.0 → 30.0 |
| EWD | 72.5 → 24.4 | 78.2 → 72.8 | 71.0 → 68.5 | 71.0 → 62.5 | 71.0 → 58.0 | 71.0 → 39.0 |
| Unigram | 38.2 → 3.5 | 37.6 → 24.1 | 34.4 → 28.1 | 34.4 → 23.8 | 34.4 → 16.2 | 34.4 → 40.6 |
| SWEET | 33.7 → 9.2 | 31.4 → 31.0 | 32.0 → 26.6 | 32.0 → 22.2 | 32.0 → 15.5 | 32.0 → 5.7 |

**δ=2**:

| Detector | Strip comments | black | Rename 1 | Rename 2 | Rename all | Dead code + try |
|---|---|---|---|---|---|---|
| KGW | 7.0 → 0.9 | 7.4 → 8.0 | 5.9 → 4.5 | 5.9 → 5.2 | 5.9 → 4.9 | 5.9 → 5.6 |
| EWD-unweighted | 8.8 → 2.5 | 11.9 → 8.5 | 8.6 → 7.6 | 8.6 → 5.3 | 8.6 → 4.6 | 8.6 → 2.6 |
| EWD | 32.5 → 9.2 | 33.3 → 27.4 | 30.7 → 22.4 | 30.7 → 20.8 | 30.7 → 17.5 | 30.7 → 5.0 |
| Unigram | 0.4 → 0.9 | 1.0 → 1.0 | 1.1 → 1.1 | 1.1 → 0.4 | 1.1 → 0.0 | 1.1 → 3.2 |
| SWEET | 18.2 → 4.7 | 17.6 → 17.1 | 17.2 → 13.5 | 17.2 → 10.7 | 17.2 → 8.8 | 17.2 → 2.2 |

Mean fraction of Qwen tokens changed (difflib on token ids, averaged over configs): strip comments 0.48–0.62, black 0.05–0.12, rename 1/2/all 0.05–0.09 / 0.10–0.17 / 0.15–0.27, dead code 0.33–0.58 (range over configs).

**Findings (QP2).**
- **Comment and docstring removal is the strongest attack**, cutting TPR by 66–91% for every method (e.g. KGW δ=4 43.5% → 5.0%, EWD δ=4 72.5% → 24.4%). Most of the watermark signal lives in natural-language comments and docstrings, the high-entropy part of the output; it also removes ~55–60% of the tokens. "Code-only" watermark strength is therefore much lower than whole-output numbers suggest.
- **Formatting (`black`) barely matters** for context-based methods (EWD 78 → 73, SWEET 31 → 31) because it changes few tokens (~5%).
- **Renaming degrades detection monotonically with the number of names changed** (EWD δ=4 71 → 69 → 63 → 58), and is cheap: renaming all identifiers changes only 15–27% of tokens.
- **Dead code + try/except** hurts the entropy-aware detectors most (SWEET δ=4 32 → 6, EWD 71 → 39): the inserted tokens are low-entropy and unwatermarked but dilute the weighted score, and wrapping changes the indentation tokens throughout the function body.
- EWD keeps the highest TPR under every attack at δ=4 (ignoring the Unigram dead-code artifact below) and under every attack except dead code at δ=2, where all detectors are ≤ 6%.
- **Artifact (Unigram + dead code)**: dead code *raises* Unigram TPR (34 → 41 at δ=4; 1.1 → 3.2 at δ=2). The three dead-code templates are fixed token strings, and Unigram's green list does not depend on context, so the same inserted tokens shift every sample's z in the same direction. This is a property of fixed-template attacks against a fixed green list, not robustness. (The same issue affected an earlier renaming attack that used `var_N` names; it was replaced by per-sample names, see Sanity checks.)

### Without comments
Each attack is also applied after comments and docstrings are stripped (`nc_*` rows in `attacks.csv`). Each `nc_X` uses the same per-sample new names and dead code as `X`, so the pair differs only in the comments. Figures: `figures/fig_tpr_attacks_nocomments_d{2,4}.{png,pdf}`.
- **Baseline**: the same programs with comments stripped and no other attack (`TPR_stripped_same_tasks`).
- **Survival**: 100% for every combined attack and config. The 2,610 combined-attack variants of the canonical solutions also all pass.
- **Tokens changed** (vs the original, including the stripped comments): 0.46–0.85.

TPR @ 1% FPR (%), comments stripped → comments stripped + attack, **δ=4**:

| Detector | + black | + rename 1 | + rename 2 | + rename all | + dead code + try |
|---|---|---|---|---|---|
| KGW | 5.1 → 2.8 | 5.2 → 4.8 | 5.2 → 2.9 | 5.2 → 1.0 | 5.2 → 3.3 |
| EWD-unweighted | 9.1 → 5.6 | 9.0 → 5.5 | 9.0 → 4.0 | 9.0 → 1.5 | 9.0 → 1.0 |
| EWD | 24.7 → 17.7 | 24.5 → 16.0 | 24.5 → 10.0 | 24.5 → 5.0 | 24.5 → 1.5 |
| Unigram | 3.2 → 1.9 | 3.1 → 3.1 | 3.1 → 1.2 | 3.1 → 0.0 | 3.1 → 6.9 |
| SWEET | 9.4 → 6.9 | 9.4 → 5.1 | 9.4 → 3.0 | 9.4 → 1.7 | 9.4 → 0.0 |

**δ=2**:

| Detector | + black | + rename 1 | + rename 2 | + rename all | + dead code + try |
|---|---|---|---|---|---|
| KGW | 1.2 → 1.2 | 1.4 → 1.4 | 1.4 → 1.0 | 1.4 → 0.0 | 1.4 → 0.0 |
| EWD-unweighted | 2.7 → 1.4 | 2.6 → 1.7 | 2.6 → 1.0 | 2.6 → 0.7 | 2.6 → 0.7 |
| EWD | 9.2 → 6.1 | 8.9 → 5.9 | 8.9 → 2.6 | 8.9 → 2.3 | 8.9 → 0.0 |
| Unigram | 1.2 → 0.8 | 1.4 → 0.7 | 1.4 → 0.4 | 1.4 → 0.0 | 1.4 → 1.8 |
| SWEET | 4.9 → 4.2 | 4.7 → 4.7 | 4.7 → 1.6 | 4.7 → 1.6 | 4.7 → 0.0 |

**Findings (without comments).**
- **On code without comments, no method is usable.** After stripping alone, the best detector (EWD δ=4) detects 24.5%, and every other configuration detects at most 9.4%.
- **Two cheap, behaviour-preserving edits erase the watermark.** Stripping comments and then renaming all identifiers leaves every method at **≤ 5% TPR at δ=4 and ≤ 2.3% at δ=2**. Stripping followed by dead code leaves every method except Unigram at ≤ 3.3%.
- **Without comments, the same attacks remove a much larger share of what remains.** Renaming all names keeps 82% of EWD δ=4 detections when comments are present (71.0 → 58.0) but only 20% once they are gone (24.5 → 5.0). KGW goes from 63% (34.8 → 21.9) to 19% (5.2 → 1.0). With comments gone, identifiers are the main remaining high-entropy tokens, so renaming them hits the watermark directly.
- `black` and renaming a single name still matter least.
- The Unigram dead-code artifact persists: dead code raises Unigram from 3.1% to 6.9% at δ=4 (from 1.4% to 1.8% at δ=2).

## QP3: TPR by code length
`outputs/tables/length.csv`, `outputs/tables/min_length.csv`; figures `figures/fig_tpr_length_d2.{png,pdf}`, `figures/fig_tpr_length_d4.{png,pdf}`. Length = Qwen tokens of the scored (sanitized) code; all 540 generations, unattacked; bins with n ≥ 10.

TPR @ 1% FPR by length bin (tokens), δ=4:

| Detector | 0–75 | 75–100 | 100–125 | 125–150 | 150–200 | 200–300 | 300+ |
|---|---|---|---|---|---|---|---|
| KGW | 27 | 63 | 66 | 65 | 71 | 88 | 95 |
| EWD-unweighted | 27 | 57 | 70 | 68 | 76 | 70 | 87 |
| EWD | 41 | 85 | 84 | 90 | 89 | 92 | 100 |
| Unigram | 32 | 44 | 71 | 77 | 89 | 86 | 97 |
| SWEET | 18 | 43 | 66 | 54 | 45 | 41 | 60 |

δ=2: no detector reaches 80% in any bin (best: EWD 56% at 100–125 tokens; KGW ≤ 18%, Unigram ≤ 14%, SWEET ≤ 35%).

**Minimum length for TPR ≥ 80% at 1% FPR (δ=4): EWD ≈ 75 tokens, Unigram ≈ 150, KGW ≈ 200, EWD-unweighted ≈ 300, SWEET not reached.** At δ=2 no length bin up to 300+ tokens reaches 80%.
- The median generation is only 75–125 tokens long (`median_tokens` in `detection.csv`), and 24–49% are under 75 tokens, where every method is below 45% TPR. Typical code completions are simply too short for reliable detection.
- SWEET's curve is flat because it only scores tokens above its entropy threshold: length in tokens is not the length SWEET "sees". Its effective sample size grows much more slowly with program length than the other methods'.
- Caveat: bins differ in composition (different problems land in different bins), so the curve mixes length with problem type; bins are also small (n = 12–267).

## QP4 (optional): text vs code
Same model, watermarks (KGW, Unigram; γ=0.5, same key) and greedy decoding as the code study. Only the domain changes.
- **Positives**: continuations of 300 news prefixes from MarkLLM's C4 sample (`prompt` field of rows 0–299; no chat template), exactly 200 new tokens each, scored without the prompt.
- **Negatives**: the human `natural_text` of 1,000 other rows (300–1,299), cut to the first 200 Qwen tokens, so they have the same length as the positives. Thresholds are set at 1% FPR on these, per detector.
- **Attack**: random word substitution, replacing 10 / 20 / 30 / 50% of words with random words from the negatives' vocabulary, seeded per sample. Intensity is measured as the fraction of Qwen tokens changed, computed exactly as for the code attacks.
- Script: `scripts/08_text_qp4.py`. Tables: `outputs/tables/qp4_text_detection.csv`, `outputs/tables/qp4_text_attacks.csv`. Figures: `figures/fig_qp4_text_vs_code_d{2,4}.{png,pdf}`.

**Calibration on human text** (1,000 C4 continuations):

| Detector | Mean z | SD z | Share above z = 2.33 | Threshold at 1% FPR |
|---|---|---|---|---|
| KGW | −0.26 | 1.05 | 0.6% | 2.08 |
| Unigram | −0.07 | 1.21 | 2.2% | 2.73 |

KGW is calibrated on prose but not on code (code sd 1.2–1.4). Unigram is miscalibrated in both domains.

**Detection without attacks** (TPR at 1% FPR):

| Config | Text mean z | Text TPR | Code mean z | Code TPR, all lengths | Code TPR, 200–300 tokens (n) |
|---|---|---|---|---|---|
| KGW δ=2 | 7.66 | **99.0%** | 1.40 | 9.6% | 15.9% (44) |
| Unigram δ=2 | 6.95 | **99.7%** | 0.65 | 4.4% | 6.4% (47) |
| KGW δ=4 | 11.50 | **100%** | 3.42 | 49.6% | 88.4% (43) |
| Unigram δ=4 | 11.11 | **100%** | 3.63 | 56.1% | 86.0% (50) |

The text AUROC vs human text is 0.996–1.000. Unwatermarked Qwen text is above the threshold for 3.0% (KGW) and 1.0% (Unigram) of samples, with AUROC vs human text of 0.44 and 0.40.

**Findings**
- **The same watermark is roughly 10× more detectable on text than on code.** At δ=2, text TPR is 99–100%, while code reaches 4–10%, or 6–16% for code of the same length (200–300 tokens). Matching length does not close the gap, so it is not a length effect: per-token green-list bias produces a z-score about 5–10× larger on prose (7–8 vs 0.65–1.4). Code is low-entropy: most tokens are forced, so the logit bias rarely changes the choice. This is the motivation for entropy-aware detectors (SWEET, EWD), and it agrees with their gains in the code study.
- **At δ=4 the gap shrinks but remains.** Text is at 100%. Code is at 50–56% over all lengths, and only the 200–300-token bin approaches the text level (86–88%).
- **Text watermarks are far more robust per token changed.** At δ=2, text TPR stays at 94% or above (KGW) and 91% or above (Unigram) with up to 46% of tokens changed. It only falls to 53% / 64% at 73–77% of tokens changed. At δ=4, TPR is at least 89% at every intensity. On code, stripping comments and docstrings changes 48–55% of tokens and drops TPR to 1% at δ=2 and 3–5% at δ=4. That attack removes exactly the natural-language part of the code, which carries most of the signal.
- Caveats:
  - The two attacks are not equivalent edits. Word substitution does not preserve meaning, and it changes tokens in place without deleting them. The code attacks preserve function, as checked by the tests.
  - "Fraction of tokens changed" puts them on one axis, but the comparison is about how much signal survives per edit, not about attack strength.
  - Greedy decoding loops on some news prefixes: 2–13% of text generations have a distinct-word ratio below 0.3 (`frac_repetitive`). Excluding them leaves TPR unchanged (`TPR_nonrepetitive` 99–100%) and lowers the KGW false-positive rate on unwatermarked Qwen text from 3.0% to 1.7%.

## Uncertainty (bootstrap 95% CIs)
`scripts/09_uncertainty.py` → `outputs/tables/ci_pass_at_1.csv`, `ci_detection.csv`, `ci_attacks.csv`. It uses a percentile bootstrap with 2,000 resamples and seed 42.
- pass@1 resamples problems. The intervals are paired, so each config's drop is computed on the same resampled problem ids as the baseline.
- TPR resamples both the human negatives (re-estimating the 1% threshold each time) and the watermarked texts, so the threshold's own uncertainty is included.
- Attack changes are paired on the same tasks before and after the attack.

**pass@1 drop vs no watermark (points, [95% CI])**:

| Config | Original tests | Extended tests | Extra drop on extended tests |
|---|---|---|---|
| KGW δ=2 | 3.1 [−0.7, 7.0] | 3.5 [0.0, 7.0] | 0.4 [−2.0, 2.8] |
| EWD δ=2 | 3.0 [−0.7, 6.9] | 0.7 [−2.8, 4.3] | −2.2 [−4.8, 0.2] |
| Unigram δ=2 | 2.6 [−1.3, 6.5] | 4.1 [0.4, 7.8] | 1.5 [−1.1, 4.1] |
| SWEET δ=2 | −2.8 [−6.5, 0.9] | −2.2 [−5.9, 1.1] | 0.6 [−2.0, 3.0] |
| KGW δ=4 | 19.8 [15.4, 24.1] | 18.0 [13.7, 22.0] | −1.9 [−4.6, 0.9] |
| EWD δ=4 | 23.7 [19.3, 28.5] | 19.8 [15.4, 24.3] | −3.9 [−6.5, −1.3] |
| Unigram δ=4 | 29.3 [24.6, 33.9] | 27.2 [22.8, 31.7] | −2.0 [−4.6, 0.6] |
| SWEET δ=4 | 2.2 [−1.9, 6.1] | 1.9 [−2.4, 5.9] | −0.4 [−3.0, 2.0] |

- **At δ=2 the cost in correctness is statistically indistinguishable from zero.** The exceptions are Unigram on the extended tests (4.1, CI excludes 0) and KGW on the extended tests (3.5, CI touches 0). SWEET's apparent gain is also within noise.
- **At δ=4, SWEET's drop is not significant.** KGW, EWD and Unigram lose 18–29 points, all clearly significant.
- **QP1 confirmed:** the extra drop revealed by the extended tests includes 0 for 7 of 8 configs. For EWD δ=4 the extended tests show a significantly *smaller* drop. The original tests do not underestimate the watermark's damage.

**TPR @ 1% FPR, all generations [95% CI]**:
- δ=2: KGW 9.6 [5.6, 15.2], EWD 38.0 [28.1, 43.1], EWD-unweighted 13.1 [6.5, 16.7], Unigram 4.4 [2.0, 7.0], SWEET 20.7 [14.8, 28.0].
- δ=4: KGW 49.6 [43.5, 56.1], EWD 79.3 [73.3, 83.1], EWD-unweighted 61.3 [53.9, 67.6], Unigram 56.1 [41.9, 62.4], SWEET 43.3 [35.7, 49.8].
- The intervals are wide (often ±5–10 points) mostly because the 1% threshold rests on the top 15 of 1,540 negatives.
- **Entropy weighting is significant:** EWD − EWD-unweighted on the same texts is +24.8 [15.4, 32.6] points at δ=2 and +18.0 [10.9, 25.6] at δ=4.

**Attacks** (`ci_attacks.csv`, change in TPR on the same tasks):
- Comment stripping is significant for every config except Unigram δ=2, which is already at about 1%.
- With comments present, `black` and renaming a single identifier are mostly not significant. The exceptions:
  - `black`: EWD at both δ, EWD-unweighted δ=4, Unigram δ=4.
  - Renaming one name: EWD δ=2, EWD-unweighted δ=4, Unigram δ=4.
- Without comments, all five combined attacks significantly lower EWD at δ=4 relative to the stripped code. They range from `black` (−7 [−12, −3] points) to rename all (−20 [−26, −12]) and dead code (−23 [−30, −15]).
- The Unigram dead-code increase is borderline (+6 [0, +11] at δ=4).

## Trade-off: pass@1 vs detection
`figures/fig_tradeoff_pass_tpr.{png,pdf}`: pass@1 on the extended tests vs TPR @ 1% FPR, one line per method from δ=2 to δ=4, with 95% CIs on both axes. The shaded band is the no-watermark pass@1 CI.
- **No configuration combines baseline-level correctness with more than 50% detection.**
  - Inside the no-watermark band: every δ=2 config and SWEET δ=4. The best of them detects 43% (SWEET δ=4), then EWD δ=2 (38%).
  - EWD δ=4 is the only one near 80% TPR, and it pays about 20 points of pass@1.
- **Each method trades differently.** EWD buys the most detection per point of pass@1 lost. Unigram buys the least: at δ=4 it costs 27 points for 56% TPR. SWEET barely moves on either axis between δ=2 and δ=4.
- EWD δ=2 and SWEET δ=4 land on almost the same point (about 40% TPR, 55–56% pass@1).

## Sanity checks
- **Test harness**: all 541 canonical solutions pass both original and extended tests (163 HumanEval+, 378 MBPP+; `outputs/evals/canonical.jsonl`).
  - EvalPlus's default 4 GB address-space cap made 6 canonical solutions fail on the heaviest extended inputs (HumanEval/15, 100, 130; Mbpp/84, 255, 392). Raised to 8 GB (`EVALPLUS_MAX_MEMORY_BYTES`, `scripts/wsl_run.sh`).
  - **HumanEval/32 (`find_zero`) is excluded.** Its canonical solution passes all 100 base inputs outside the sandbox, but EvalPlus's sandbox fails it before the first test in our setup. Final problem count: **541** (163 + 378).
- **Watermark pipeline positive control** (`scripts/_sanity_prose.py`): 200-token prose continuations, 3 prompts, δ=2. z-scores: KGW greedy 5.9–6.2 / sampled 8.0–9.1; EWD greedy 6.0–8.2 / sampled 7.1–9.6; Unigram greedy 3.7–7.2 / sampled 4.5–9.3; SWEET greedy 6.0–7.0 / sampled 7.2–9.5. The batched generation + MarkLLM detection path works; greedy decoding gives a weaker signal than sampling.
- **Implementation note (MarkLLM)**: KGW seeds its green list with `hash_key · prf[prev_token]`, while EWD and SWEET use `hash_key · prev_token`. EWD therefore cannot score MarkLLM-KGW text. EWD gets its own generation config (MarkLLM `EWD` class = KGW soft-red-list biasing with the EWD hash). On those texts we report both the entropy-weighted EWD z and the unweighted green-count z ("EWD-unweighted"), a paired detector comparison on identical texts.
- **Prompt copying**: under the EvalPlus instruct prompt, the model usually restates the HumanEval signature and docstring, so detection also scores that low-entropy copied text. Human negatives built as prompt + canonical solution contain the same docstring, so the comparison is like for like; the comment/docstring-removal attack isolates the effect.
- **Detection-time edge case**: 2 generations per EWD config sanitize to ~2 tokens; EWD returns z = NaN on them (all entropy weights are 0). They are counted as *not detected* (not dropped), so they stay in TPR/FPR denominators.
- **Attacked-code test bug (found and fixed)**: the first attack run recorded only 20–40% of attacked programs as passing, even for `black` and for canonical solutions under the new renaming. Cause: `04_attack.py` imported `transformers` at module level, which loads torch (~5.6 GB virtual memory). EvalPlus test workers are spawned and re-import the script, and the forked test process inherits that address space, which counts against EvalPlus's 8 GB `RLIMIT_AS`, so heavier tests failed regardless of the program. Fix: the import moved inside `main()`. All recorded failures were re-tested with `scripts/04b_retest.py` (no torch import): **every one passes** (7,552 rows of the original attacks, 4,609 rows of the redone rename attacks, and 1,126 canonical rows; rows keep `retested`, `first_pass_*` fields). The pass@1 evals (`02_evaluate.py`, never imported torch) were re-tested the same way and **none of 2,491 failures flipped**, so QP1 is unaffected.
- **Renaming attack redesign**: the first version renamed identifiers to `var_0, var_1, …`. With a fixed template the same few token pairs appear in every sample, and whether they happen to be green shifted every sample's z in the same direction (KGW δ=2 rename-all TPR went *up* from 7% to 36%; Unigram's mean z went down). Replaced by names drawn per sample from a pool of ordinary identifier words, seeded by the task id (deterministic, e.g. `left`, `index_temp`, `curr_chunk`). rename_1 ⊂ rename_2 ⊂ rename_all use the same names. Canonical solutions pass 540/540 under all three.
- **Unattacked TPR ≥ attacked TPR** holds for every detector/attack except Unigram + dead code (template artifact, see QP2) and two ≤ 0.6-point differences at δ=2 that are within noise (KGW + black, Unigram + strip comments).
