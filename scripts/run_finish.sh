#!/usr/bin/env bash
# Resume after the pause: re-test rename rows (first tested on the broken torch-importing path), finish detection.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
R="bash scripts/wsl_run.sh python"
CFGS="kgw_d2,ewd_d2,unigram_d2,sweet_d2,kgw_d4,ewd_d4,unigram_d4,sweet_d4"
echo "$(date +%T) === retest attacks"; $R scripts/04b_retest.py --kind attacks --config "canonical,$CFGS" --workers 12
T=""; for c in ${CFGS//,/ }; do T="$T${T:+,}atk:$c"; done
echo "$(date +%T) === detect"; $R scripts/05_detect.py --targets "$T"
echo "$(date +%T) FINISH DONE"
