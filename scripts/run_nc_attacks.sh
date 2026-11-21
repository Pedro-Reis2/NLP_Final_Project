#!/usr/bin/env bash
# QP2 "without comments": nc_* attacks (strip comments + docstrings, then the attack).
# All steps are resumable; existing attack rows are skipped by id.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
R="bash scripts/wsl_run.sh python"
CFGS="kgw_d2,ewd_d2,unigram_d2,sweet_d2,kgw_d4,ewd_d4,unigram_d4,sweet_d4"
echo "$(date +%T) === canonical";      $R scripts/04_attack.py --canonical
echo "$(date +%T) === attack configs"; $R scripts/04_attack.py --config "$CFGS"
echo "$(date +%T) === retest failures"; $R scripts/04b_retest.py --kind attacks --config "canonical,$CFGS" --workers 6
T=""; for c in ${CFGS//,/ }; do T="$T${T:+,}atk:$c"; done
echo "$(date +%T) === detect";         $R scripts/05_detect.py --targets "$T"
echo "$(date +%T) NC ATTACKS DONE"
