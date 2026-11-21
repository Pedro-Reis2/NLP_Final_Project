#!/usr/bin/env bash
# Attack 5 (pitch): LLM rewrite on 100 passing programs per watermarked config. Resumable.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
R="bash scripts/wsl_run.sh python"
CFGS="kgw_d2 ewd_d2 unigram_d2 sweet_d2 kgw_d4 ewd_d4 unigram_d4 sweet_d4"
echo "$(date +%T) === rewrite (GPU)"; $R scripts/10_llm_rewrite.py gen
echo "$(date +%T) === test (CPU)";    $R scripts/10_llm_rewrite.py test
T=""; for c in $CFGS; do T="$T${T:+,}rw:$c"; done
echo "$(date +%T) === detect (GPU)";  $R scripts/05_detect.py --targets "$T"
echo "$(date +%T) === analyze";       $R scripts/10_llm_rewrite.py analyze
echo "$(date +%T) REWRITE PIPELINE DONE"
