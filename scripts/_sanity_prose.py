"""Positive control: KGW via our batched path on prose should give z >> 4."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from wm.config import GEN_CONFIGS, MODEL_ID
from wm.markllm_utils import load_watermark
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
prompts = ["The history of the city began", "Scientists announced yesterday that", "In my opinion, the best way to learn"]
for cfg in ["kgw_d2", "ewd_d2", "unigram_d2", "sweet_d2"]:
    wm = load_watermark(GEN_CONFIGS[cfg], model, tok)
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    for sample in (False, True):
        torch.manual_seed(0)
        out = model.generate(**enc, logits_processor=LogitsProcessorList([wm.logits_processor]), do_sample=sample,
                             max_new_tokens=200, pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        texts = [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
        zs = [round(wm.detect_watermark(t)["score"], 1) for t in texts]
        print(cfg, "sample" if sample else "greedy", zs)
