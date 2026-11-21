#!/usr/bin/env bash
# Post-generation watermark (11_multichannel.py) through the same protocol. Waits on the rewrite
# pipeline's log lines (not pgrep) so no two EvalPlus-heavy steps run at once and the GPU is free.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
R="bash scripts/wsl_run.sh python"
until grep -aq "=== detect (GPU)\|REWRITE PIPELINE DONE" logs/rewrite_pipeline.log; do sleep 30; done
echo "$(date +%T) === evaluate mcw";  $R scripts/02_evaluate.py --config mcw
echo "$(date +%T) === attack mcw";    $R scripts/04_attack.py --config mcw
echo "$(date +%T) === retest mcw";    $R scripts/04b_retest.py --kind attacks --config mcw --workers 6
until grep -aq "REWRITE PIPELINE DONE" logs/rewrite_pipeline.log; do sleep 30; done
echo "$(date +%T) === rewrite mcw (GPU)"; $R scripts/10_llm_rewrite.py gen --configs mcw
echo "$(date +%T) === test rewrite mcw";  $R scripts/10_llm_rewrite.py test --configs mcw
echo "$(date +%T) === detect mcw";        $R scripts/11_multichannel.py detect
echo "$(date +%T) === analyze mcw";       $R scripts/11_multichannel.py analyze
echo "$(date +%T) MCW PIPELINE DONE"
