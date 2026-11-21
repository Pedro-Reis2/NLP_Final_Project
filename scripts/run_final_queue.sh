#!/usr/bin/env bash
# Differential testing after the multichannel EvalPlus steps; marking-sites count once the GPU is free.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
R="bash scripts/wsl_run.sh python"
until grep -aq "=== rewrite mcw (GPU)\|MCW PIPELINE DONE" logs/mcw_pipeline.log; do sleep 30; done
echo "$(date +%T) === difftest test";    $R scripts/13_differential.py test
echo "$(date +%T) === difftest retest";  $R scripts/13_differential.py retest --workers 4
echo "$(date +%T) === difftest analyze"; $R scripts/13_differential.py analyze
until grep -aq "MCW PIPELINE DONE" logs/mcw_pipeline.log; do sleep 30; done
echo "$(date +%T) === sites count (GPU)"; $R scripts/12_sites.py count
echo "$(date +%T) === sites analyze";     $R scripts/12_sites.py analyze
echo "$(date +%T) FINAL QUEUE DONE"
