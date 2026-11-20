#!/usr/bin/env bash
# Diagnostic: watermark strength on code under greedy vs sampling (40 problems).
set -e
for spec in "greedy:" "t02:0.2" "t07:0.7"; do
  tag=${spec%%:*}; t=${spec#*:}
  targ=""; [ -n "$t" ] && targ="--temperature $t"
  python scripts/01_generate.py --config nowm,kgw_d2,sweet_d2 --limit 20 --tag probe_$tag $targ
  python scripts/02_evaluate.py --test --tag probe_$tag --config nowm,kgw_d2,sweet_d2
  python scripts/05_detect.py --test --tag probe_$tag --targets gen:nowm,gen:kgw_d2,gen:sweet_d2 --detectors KGW,SWEET
done
echo PROBE DONE
