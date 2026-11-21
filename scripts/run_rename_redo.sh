#!/usr/bin/env bash
# Redo the three rename attacks with per-sample names (was a fixed var_N template).
# Drops old rename rows from attack + detection files, then the resumable steps re-add them.
cd /mnt/c/Users/Pedro/Documents/GitHub/aa/NLP_Final_Project
R="bash scripts/wsl_run.sh python"
CFGS="kgw_d2,ewd_d2,unigram_d2,sweet_d2,kgw_d4,ewd_d4,unigram_d4,sweet_d4"
echo "$(date +%T) === drop old rename rows"
for f in outputs/attacks/*.jsonl outputs/detect/*/atk__*.jsonl; do
  grep -v '|rename_' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
echo "$(date +%T) === canonical rename sanity"; $R scripts/04_attack.py --canonical
echo "$(date +%T) === attack configs";        $R scripts/04_attack.py --config "$CFGS"
T=""; for c in ${CFGS//,/ }; do T="$T${T:+,}atk:$c"; done
echo "$(date +%T) === detect";                $R scripts/05_detect.py --targets "$T"
echo "$(date +%T) RENAME REDO DONE"
