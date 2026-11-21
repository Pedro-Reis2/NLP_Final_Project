"""Diagnostic: re-test a sample of attacked programs that were recorded as failing."""
import json, sys, collections
sys.path.insert(0, "src")
from wm.evalplus_runner import run_tests

if __name__ == "__main__":
    cfg, n = sys.argv[1], int(sys.argv[2])
    rows = [json.loads(l) for l in open(f"outputs/attacks/{cfg}.jsonl")]
    bad = [r for r in rows if r["status"] == "ok" and not r["pass_plus"]][:n]
    res = run_tests([dict(key=r["id"], dataset=r["dataset"], task_id=r["task_id"], solution=r["attacked"]) for r in bad], progress=False)
    print(collections.Counter((r["pass_plus"], r["base_status"], r["plus_status"]) for r in res))
    att = {r["id"]: r["attack"] for r in bad}
    print(collections.Counter((att[r["key"]], r["pass_plus"]) for r in res))
