#!/usr/bin/env bash
# One-shot environment setup inside WSL Ubuntu. Safe to re-run.
set -euo pipefail
REPO=/mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git build-essential

if [ ! -d ~/wm-venv ]; then python3 -m venv ~/wm-venv; fi
source ~/wm-venv/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r "$REPO/requirements.txt"
pip install --no-deps markllm
pip freeze > "$REPO/requirements.lock.txt"

grep -q 'HF_HOME' ~/.bashrc || echo 'export HF_HOME=~/hf' >> ~/.bashrc
export HF_HOME=~/hf
python -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct >/dev/null && echo "model downloaded"
