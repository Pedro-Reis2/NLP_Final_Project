#!/usr/bin/env bash
# Run a command inside the project venv from the repo root (used from Windows: wsl.exe ... bash scripts/wsl_run.sh <cmd>)
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
source /root/wm-venv/bin/activate
export HF_HOME=/root/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# EvalPlus default address-space cap (4 GB) makes 6 canonical solutions fail on
# the heaviest plus inputs (HumanEval/15,100,130; Mbpp/84,255,392); 8 GB fixes all.
export EVALPLUS_MAX_MEMORY_BYTES=8589934592
exec "$@"
