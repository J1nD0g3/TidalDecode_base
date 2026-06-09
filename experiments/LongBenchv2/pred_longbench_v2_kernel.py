"""LongBench-v2 evaluation for TidalDecode KERNEL path (tidal/) — Qwen3-14B.

Same prompts / context truncation / generation-based scoring as the eager harness
(pred_longbench_v2.py, which ports ShadowKV's metric), but generation uses the fast
CUDA-kernel model (tidal.Qwen3ForCausalLM) instead of the eager src/ path.

Overall sparsity is matched to 0.27 (= ShadowKV): per-sparse-layer token budget =
0.1889 * context (4 full + 36 sparse layers -> overall 0.27), set per sample.

Usage:
  python experiments/LongBenchv2/pred_longbench_v2_kernel.py \
      --model /workspace/models/Qwen3-14B-128k --keep_ratio 0.27 --datalen 122880 \
      --enable_thinking --max_samples 5
"""
import os
import sys
import gc
import json
import time
import math
import argparse

import torch
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
# reuse the verbatim ShadowKV-ported prompt/metric/truncation from the eager harness
from experiments.LongBenchv2.pred_longbench_v2 import (
    LONGBENCHV2_PROMPT, LONGBENCHV2_GEN_LEN,
    longbenchv2_extract_answer, longbenchv2_metric, postprocess_pred,
    wrap_qwen3_chat, build_prompt, strip_thinking,
)
from tidal import Qwen3ForCausalLM
from transformers import AutoTokenizer


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--keep_ratio", type=float, default=0.27,
                   help="overall KV keep ratio (matched to ShadowKV)")
    p.add_argument("--sparse_layer_start", type=int, default=2)
    p.add_argument("--correction_layer", type=int, default=21)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument("--datalen", type=int, default=122880)
    p.add_argument("--max_gen", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--page_size", type=int, default=1)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    with open(os.path.join(args.model, "config.json")) as f:
        cfg = json.load(f)
        max_position = cfg.get("max_position_embeddings", 40960)
        n_layers = cfg["num_hidden_layers"]
    max_gen = args.max_gen
    if max_gen <= 0:
        max_gen = 8192 if args.enable_thinking else LONGBENCHV2_GEN_LEN
    max_gen = min(max_gen, max_position - args.datalen)

    # overall keep_ratio -> per-sparse-layer ratio (4 full layers: 0,1,sparse_start,correction)
    full_layers = args.sparse_layer_start + 2
    n_sparse = n_layers - full_layers
    sparse_ratio = (args.keep_ratio * n_layers - full_layers) / n_sparse
    assert 0 < sparse_ratio <= 1, f"infeasible keep_ratio -> sparse_ratio={sparse_ratio}"
    max_budget = max(8, int(round(sparse_ratio * args.datalen)))
    print(f"[overall-sparsity] keep_ratio={args.keep_ratio} -> per-sparse-layer ratio={sparse_ratio:.4f} "
          f"({full_layers} full + {n_sparse} sparse); max_budget={max_budget}")

    model_short = os.path.basename(args.model.rstrip("/"))
    think_tag = "think_on" if args.enable_thinking else "think_off"
    run_name = f"longbenchv2_tidaldecode_{think_tag}_r{args.keep_ratio}"
    out_dir = args.out_dir or os.path.join(HERE, "pred", model_short)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}.jsonl")

    done_idx = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done_idx.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"Resuming: {len(done_idx)} done")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        tidal_sparse_layer_start=args.sparse_layer_start,
        tidal_correction_layer=args.correction_layer,
    ).to("cuda:0").eval()
    # NOTE: the tidal InferenceController is (re)initialized FRESH per sample inside
    # generate(). Reusing one controller across samples + clean_states() leaks the
    # previous sample's KV context into the next sample's generation (verified: idx0's
    # content bled into idx1), corrupting all but the first sample. Fresh per-sample
    # init is clean and uses a per-sample-sized KV pool (cheaper for short samples).

    eos_ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None:
            eos_ids.add(tid)

    data = load_dataset("THUDM/LongBench-v2", split="train")
    num = len(data) if args.max_samples <= 0 else min(args.max_samples, len(data))
    print(f"LongBench-v2 KERNEL: {num} samples | datalen={args.datalen} max_gen={max_gen} "
          f"thinking={args.enable_thinking}")

    @torch.no_grad()
    def generate(input_ids, budget):
        L = input_ids.size(1)
        # fresh controller per sample (see NOTE above): avoids cross-sample KV leak.
        model.model.iController = None
        gc.collect(); torch.cuda.empty_cache()
        model.tidal_init(page_size=args.page_size, max_seq_len=L + max_gen + 64,
                         token_budget=budget, dtype=torch.bfloat16, device="cuda:0")
        # per-sample budget (matched sparsity): set the sparse token budget on the model+controller
        model.model._tidal_token_budget = budget
        model.model.iController.td_token_budget = budget
        out = model(input_ids=input_ids)
        gen = [out.logits[:, -1, :].argmax(-1).item()]
        for _ in range(max_gen - 1):
            if gen[-1] in eos_ids:
                break
            out = model(input_ids=torch.tensor([[gen[-1]]], device="cuda:0"))
            gen.append(out.logits[:, -1, :].argmax(-1).item())
        model.tidal_clear()
        model.model.iController = None
        return gen

    for i in tqdm(range(num), desc="lbv2-kernel"):
        if i in done_idx:
            continue
        sample = data[i]
        input_ids = build_prompt(tok, sample, args.datalen, args.enable_thinking).to("cuda:0")
        L = input_ids.size(1)
        budget = min(max_budget, max(8, int(round(sparse_ratio * L))))

        t0 = time.time()
        gen = generate(input_ids, budget)
        elapsed = time.time() - t0

        raw = tok.decode(gen, skip_special_tokens=True)
        pred = strip_thinking(raw)
        correct = longbenchv2_metric(pred, sample["answer"])
        rec = {
            "idx": i, "prediction": pred, "ground_truth": sample["answer"],
            "correct": correct,
            "extracted": longbenchv2_extract_answer(postprocess_pred(pred).strip()),
            "input_len": L, "output_len": len(gen), "budget": budget,
            "difficulty": sample.get("difficulty", ""), "length": sample.get("length", ""),
            "domain": sample.get("domain", ""), "elapsed_sec": round(elapsed, 2),
        }
        if args.enable_thinking:
            rec["raw_output_with_think"] = raw
        with open(out_path, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    records = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    if records:
        acc = sum(r["correct"] for r in records) / len(records)
        print(json.dumps({"run": run_name, "model": model_short,
                          "samples": len(records), "accuracy": round(acc, 4)},
                         indent=2, ensure_ascii=False))
