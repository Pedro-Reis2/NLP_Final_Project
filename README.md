# Watermarking LLM-generated code: an empirical study

Do the standard LLM watermarks (KGW, EWD, Unigram, SWEET; via [MarkLLM](https://github.com/THU-BPM/MarkLLM))
work on code? Qwen2.5-Coder-1.5B-Instruct on HumanEval+ and MBPP+ (EvalPlus), δ ∈ {2, 4}.

- **QP1** pass@1 with original vs extended tests
- **QP2** detection (TPR at 1% FPR, AUROC) under semantics-preserving attacks, with and without comments
- **QP3** detection by code length
- **QP4** the same watermarks on natural-language text (C4 news)

Results, tables and figure paths: [RESULTS.md](RESULTS.md). Protocol and fixed settings: [CLAUDE.md](CLAUDE.md).

## Setup (WSL Ubuntu 24.04, CUDA GPU)
EvalPlus's test executor needs Unix, so everything runs inside WSL.
```bash
python3 -m venv ~/wm-venv && source ~/wm-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps markllm==0.1.5      # its own pins do not build on Python 3.12
# exact versions used: requirements.lock.txt
export HF_HOME=~/hf
```
`scripts/wsl_run.sh <cmd>` runs a command from the repo root with the venv and environment
(including `EVALPLUS_MAX_MEMORY_BYTES=8 GB`) set.

## Pipeline
Every step is resumable (appends to a JSONL keyed by id, skips finished ids) and takes `--limit N` for a trial run.

| Step | Script | Output |
|---|---|---|
| Generate (9 configs) | `01_generate.py --config all` | `outputs/generations/` |
| Test with EvalPlus | `02_evaluate.py --canonical`, `02_evaluate.py --config all` | `outputs/evals/` |
| Human negatives (canonical + CodeNet) | `03_negatives.py` (needs `data/codenet/Project_CodeNet_Python800.tar.gz`, from the Hugging Face mirror `qiankunmu/Project_CodeNet_Python800_and_Java250`) | `outputs/negatives/` |
| Attacks + re-test | `04_attack.py --canonical`, `04_attack.py --config all`, `04b_retest.py --kind attacks` | `outputs/attacks/` |
| Detection z-scores | `05_detect.py --targets ...` | `outputs/detect/` |
| Tables | `06_analyze.py` | `outputs/tables/*.csv` |
| Confidence intervals (bootstrap) | `09_uncertainty.py` (run before `07_figures.py`, which draws the trade-off figure from it) | `outputs/tables/ci_*.csv` |
| Figures | `07_figures.py` | `figures/` |
| QP4 text study | `08_text_qp4.py all` (needs `data/c4/processed_c4.json` from the MarkLLM repo) | `outputs/qp4/`, `outputs/tables/qp4_*.csv` |

Tests: `python -m pytest tests` (attacks preserve behaviour on sample programs).
