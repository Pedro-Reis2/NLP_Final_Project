#!/usr/bin/env bash
# Re-test recorded failures on an idle machine (first test pass overlapped with
# GPU jobs and timed out correct programs), then attack + detect any generation
# that flipped to passing. Run alone: no other EvalPlus or GPU job at the same time.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
R="bash scripts/wsl_run.sh python"
echo "$(date +%T) === retest evals";   $R scripts/04b_retest.py --kind evals   --workers 12
echo "$(date +%T) === retest attacks"; $R scripts/04b_retest.py --kind attacks --workers 12
echo "$(date +%T) === attack newly passing"; $R scripts/04_attack.py --config kgw_d2,ewd_d2,unigram_d2,sweet_d2,kgw_d4,ewd_d4,unigram_d4,sweet_d4
T=""; for c in kgw_d2 ewd_d2 unigram_d2 sweet_d2 kgw_d4 ewd_d4 unigram_d4 sweet_d4; do T="$T${T:+,}atk:$c"; done
echo "$(date +%T) === detect new attacks"; $R scripts/05_detect.py --targets "$T"
echo "$(date +%T) RETEST PIPELINE DONE"
