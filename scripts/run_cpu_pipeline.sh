#!/usr/bin/env bash
# Follows the generation run: for each config, once its generations are complete,
# run EvalPlus tests (02) then attacks + tests on attacked code (04).
# Sequential on purpose: two concurrent EvalPlus jobs make heavy tasks time out.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
N=540
CFGS=${CFGS:-"nowm kgw_d2 ewd_d2 unigram_d2 sweet_d2 kgw_d4 ewd_d4 unigram_d4 sweet_d4"}
for c in $CFGS; do
  f=outputs/generations/$c.jsonl
  until [ -f "$f" ] && [ "$(grep -c . "$f")" -ge "$N" ]; do sleep 30; done
  echo "$(date +%T) === $c: evaluate"
  bash scripts/wsl_run.sh python scripts/02_evaluate.py --config "$c"
  if [ "$c" != "nowm" ]; then
    echo "$(date +%T) === $c: attack"
    bash scripts/wsl_run.sh python scripts/04_attack.py --config "$c"
  fi
done
echo "$(date +%T) CPU PIPELINE DONE"
