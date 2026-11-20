#!/usr/bin/env bash
# GPU detection, queued behind generation: negatives + all generations first,
# then each config's attacked code once the CPU pipeline has produced it.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
CFGS="kgw_d2 ewd_d2 unigram_d2 sweet_d2 kgw_d4 ewd_d4 unigram_d4 sweet_d4"
while pgrep -f 01_generate.py >/dev/null; do sleep 30; done
echo "$(date +%T) === detect negatives + generations"
T="neg,gen:nowm"; for c in $CFGS; do T="$T,gen:$c"; done
bash scripts/wsl_run.sh python scripts/05_detect.py --targets "$T"
set -- $CFGS
while [ $# -gt 0 ]; do
  c=$1; next=${2:-}
  # attacks for $c are finished once the CPU pipeline starts the next config (or ends)
  if [ -n "$next" ]; then pat="=== $next: evaluate"; else pat="CPU PIPELINE DONE"; fi
  until grep -q "$pat" logs/cpu_pipeline.log; do sleep 30; done
  echo "$(date +%T) === detect attacks $c"
  bash scripts/wsl_run.sh python scripts/05_detect.py --targets "atk:$c"
  shift
done
echo "$(date +%T) DETECT PIPELINE DONE"
