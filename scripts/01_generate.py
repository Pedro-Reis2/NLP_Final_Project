"""Batched generation for HumanEval+ / MBPP+ with an optional MarkLLM watermark.

  python scripts/01_generate.py --config kgw_d2 --limit 5
  python scripts/01_generate.py --config all            # every GEN_CONFIG in order

Output: outputs/generations/<config_id>.jsonl, one row per task, resumable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList  # noqa: E402

from wm.config import DECODING, GEN_CONFIGS, MODEL_ID, OUT, SEED  # noqa: E402
from wm.io import append_jsonl, done_ids, get_logger, write_manifest  # noqa: E402
from wm.data import load_problems, make_prompt, sanitize_solution  # noqa: E402
from wm.markllm_utils import load_watermark  # noqa: E402

STOP = "\n```"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()
    return model, tok


def run_config(cfg_id, model, tok, problems, batch_size, log, out_root=OUT):
    cfg = GEN_CONFIGS[cfg_id]
    out_path = out_root / "generations" / f"{cfg_id}.jsonl"
    done = done_ids(out_path)
    todo = [p for p in problems if p["id"] not in done]
    log.info(f"[{cfg_id}] {len(done)} done, {len(todo)} to go")
    write_manifest(f"gen_{cfg_id}", {"step": "generate", **cfg.to_dict(), "batch_size": batch_size,
                                     "stop": STOP, "n_problems": len(problems)})
    if not todo:
        return
    wm = load_watermark(cfg, model, tok) if cfg.method else None
    procs = LogitsProcessorList([wm.logits_processor]) if wm else None

    # longest prompts first: OOM shows up immediately, and similar lengths share batches
    todo.sort(key=lambda p: -len(p["prompt_text"]))
    t_start, n_tok = time.time(), 0
    for b in range(0, len(todo), batch_size):
        batch = todo[b: b + batch_size]
        enc = tok([p["prompt_text"] for p in batch], return_tensors="pt", padding=True).to("cuda")
        torch.manual_seed(SEED)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc,
                logits_processor=procs,
                do_sample=DECODING["do_sample"],
                max_new_tokens=DECODING["max_new_tokens"],
                num_return_sequences=1,
                stop_strings=[STOP],
                tokenizer=tok,
                pad_token_id=tok.pad_token_id,
                temperature=None, top_p=None, top_k=None,
            )
        dt = time.time() - t0
        new = out[:, enc["input_ids"].shape[1]:]
        rows = []
        for p, ids in zip(batch, new):
            ids = ids[ids != tok.pad_token_id]
            text = tok.decode(ids, skip_special_tokens=True)
            code = text.split("```")[0]
            n_tok += len(ids)
            rows.append({
                "id": p["id"], "dataset": p["dataset"], "config_id": cfg_id,
                "raw": text, "code": code,
                "solution": sanitize_solution(code, p["entry_point"]),
                "n_new_tokens": int(len(ids)), "hit_max": bool(len(ids) >= DECODING["max_new_tokens"]),
                "batch_time_s": round(dt, 2),
            })
        append_jsonl(out_path, rows)
        el = time.time() - t_start
        done_now = b + len(batch)
        log.info(f"[{cfg_id}] {done_now}/{len(todo)}  batch {dt:.1f}s  "
                 f"{n_tok / el:.0f} tok/s  eta {el / done_now * (len(todo) - done_now) / 60:.1f} min  "
                 f"maxmem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    del wm
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config_id, comma list, or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="first N problems per dataset")
    ap.add_argument("--datasets", default="humaneval,mbpp")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    log = get_logger("generate")
    cfgs = list(GEN_CONFIGS) if args.config == "all" else args.config.split(",")
    model, tok = load_model()
    problems = load_problems(args.datasets.split(","), args.limit)
    for p in problems:
        p["prompt_text"] = make_prompt(p["prompt"], tok)
    log.info(f"{len(problems)} problems, configs={cfgs}")
    for c in cfgs:
        run_config(c, model, tok, problems, args.batch_size, log,
                   out_root=OUT / "test" if args.limit else OUT)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()
