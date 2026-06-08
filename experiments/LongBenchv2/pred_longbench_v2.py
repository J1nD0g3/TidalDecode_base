"""LongBench-v2 evaluation for TidalDecode (generation-based scoring).

Prompt template, context middle-truncation and answer extraction are ported
verbatim from ShadowKV (data/dataset.py, data/metrics.py) so that results are
directly comparable across repos.

Scoring is purely generation-based: the model generates a CoT answer and the
choice letter is regex-extracted (no log-prob comparison).

Usage:
  python experiments/LongBenchv2/pred_longbench_v2.py \
      --model /workspace/models/Qwen3-14B --attn_type tidal --top_k 4096 \
      --datalen 32768 --enable_thinking
"""
import os
import re
import sys
import json
import time
import argparse

import torch
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load

# ============================================================
# Ported verbatim from ShadowKV data/dataset.py
# ============================================================

# LongBench-v2: 0-shot CoT prompt (ABCD multiple choice)
LONGBENCHV2_PROMPT = (
    "Please read the following text and answer the questions below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    "Let's think step by step:"
)

LONGBENCHV2_GEN_LEN = 1024  # CoT reasoning needs more tokens


# ============================================================
# Ported verbatim from ShadowKV data/metrics.py
# ============================================================

def postprocess_pred(predict_str: str):

    predict_str = predict_str.strip().replace('<|eot_id|>', '').replace('</s>', '').replace('</s', '').replace('</', '')

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def longbenchv2_extract_answer(result):
    """Extract A/B/C/D answer from model output."""
    result = result.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', result)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', result)
    if match:
        return match.group(1)
    # Fallback: find last standalone A/B/C/D
    match = re.findall(r'\b([A-D])\b', result)
    if match:
        return match[-1]
    return None


def longbenchv2_metric(prediction, ground_truth):
    """LongBench-v2 accuracy metric (exact match on A/B/C/D)."""
    prediction = postprocess_pred(prediction).strip()
    pred_answer = longbenchv2_extract_answer(prediction)
    return 1.0 if pred_answer == ground_truth else 0.0


# ============================================================
# Qwen3 chat wrapping (manual ChatML; byte-identical to
# tokenizer.apply_chat_template(..., enable_thinking=...) for Qwen3)
# ============================================================

def wrap_qwen3_chat(prompt, enable_thinking):
    if enable_thinking:
        # clean prompt; the model generates <think>...</think> then the answer
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    # empty think block skips thinking (official template rendering for enable_thinking=False)
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_prompt(tokenizer, sample, datalen, enable_thinking):
    """Fill LONGBENCHV2_PROMPT, middle-truncating the context only (ShadowKV logic)."""
    fields = dict(
        question=sample['question'],
        choice_A=sample['choice_A'],
        choice_B=sample['choice_B'],
        choice_C=sample['choice_C'],
        choice_D=sample['choice_D'],
    )
    prompt_text = LONGBENCHV2_PROMPT.format(context=sample['context'], **fields)
    chat_text = wrap_qwen3_chat(prompt_text, enable_thinking)
    input_ids = tokenizer.encode(chat_text, return_tensors="pt", add_special_tokens=False)
    if input_ids.size(1) > datalen:
        # Middle truncation on context only
        empty_prompt = LONGBENCHV2_PROMPT.format(context='', **fields)
        overhead_ids = tokenizer.encode(
            wrap_qwen3_chat(empty_prompt, enable_thinking),
            return_tensors="pt", add_special_tokens=False,
        )
        max_ctx_tokens = datalen - overhead_ids.size(1)
        ctx_ids = tokenizer.encode(sample['context'], add_special_tokens=False)
        if len(ctx_ids) > max_ctx_tokens:
            half = max_ctx_tokens // 2
            ctx_ids = ctx_ids[:half] + ctx_ids[-half:]
        truncated_ctx = tokenizer.decode(ctx_ids, skip_special_tokens=False)
        prompt_text = LONGBENCHV2_PROMPT.format(context=truncated_ctx, **fields)
        chat_text = wrap_qwen3_chat(prompt_text, enable_thinking)
        input_ids = tokenizer.encode(chat_text, return_tensors="pt", add_special_tokens=False)
    return input_ids


# ============================================================
# Greedy generation with manual decode loop (same pattern as
# experiments/LongBench/pred.py; tidal re-selects at every decode step)
# ============================================================

@torch.no_grad()
def greedy_generate(model, input_ids, max_gen, eos_token_ids):
    input_ids = input_ids.to(model.device)
    output = model(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
        num_logits_to_keep=1,  # avoid materializing [seq_len, vocab] logits on prefill
    )
    past_key_values = output.past_key_values
    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated = [pred_token_idx.item()]
    for _ in range(max_gen - 1):
        if generated[-1] in eos_token_ids:
            break
        output = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated.append(pred_token_idx.item())
    return generated


def strip_thinking(text):
    """Keep only the content after </think> (ShadowKV qwen3.py behavior)."""
    if '</think>' in text:
        return text.split('</think>')[-1].strip()
    if '<think>' in text:
        return ''
    return text


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--attn_type", type=str, choices=["tidal", "full"], default="tidal")
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--sparse_layer_start", type=int, default=2)
    parser.add_argument("--keep_ratio", type=float, default=None,
                        help="overall KV keep ratio to match across methods (e.g. 0.27). Overrides top_k; converts to per-sparse-layer ratio accounting for full layers.")
    parser.add_argument("--correction_layer", type=int, default=17,
                        help="token re-selection layer (Qwen3-14B has 40 layers)")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable Qwen3 thinking mode")
    parser.add_argument("--datalen", type=int, default=32768,
                        help="max prompt length in tokens (context is middle-truncated)")
    parser.add_argument("--max_gen", type=int, default=0,
                        help="max new tokens (0 = 1024, or 8192 with thinking)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="max samples (0 = all 503)")
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    # context window from model config (40960 native, 131072 with YaRN config)
    with open(os.path.join(args.model, "config.json")) as f:
        max_position = json.load(f).get("max_position_embeddings", 40960)

    max_gen = args.max_gen
    if max_gen <= 0:
        max_gen = 8192 if args.enable_thinking else LONGBENCHV2_GEN_LEN
    # keep prompt + generation within the model's context window
    max_gen = min(max_gen, max_position - args.datalen)
    assert max_gen > 0, f"datalen {args.datalen} leaves no room for generation (max_position={max_position})"

    model_short = os.path.basename(args.model.rstrip("/"))
    think_tag = "think_on" if args.enable_thinking else "think_off"
    if args.attn_type == "tidal":
        run_name = f"longbenchv2-tidal-{args.top_k}-c{args.correction_layer}-{think_tag}"
    else:
        run_name = f"longbenchv2-full-{think_tag}"
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "pred", model_short)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}.jsonl")

    # resume: skip already-processed sample indices
    done_idx = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_idx.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_idx)} samples already done in {out_path}")

    model, tokenizer = load(
        args.model, args.attn_type, device="cuda",
        top_k=args.top_k,
        sparse_layer_start=args.sparse_layer_start,
        correction_layer=args.correction_layer,
    )

    # Overall-sparsity matching: convert overall keep ratio -> per-sparse-layer ratio.
    # TidalDecode runs `sparse_layer_start`+1 ... but structurally: layers [0, sparse_layer_start)
    # are full, plus the search layers (sparse_layer_start, correction_layer) attend fully.
    if args.keep_ratio is not None and args.attn_type == "tidal":
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        full_layers = args.sparse_layer_start + 2  # layers [0,start) + 2 search layers (start, correction)
        n_sparse = n_layers - full_layers
        sparse_ratio = (args.keep_ratio * n_layers - full_layers) / n_sparse
        assert 0 < sparse_ratio <= 1, f"infeasible keep_ratio: sparse_ratio={sparse_ratio}"
        n_set = 0
        for m in model.modules():
            # wrapped attention modules carry a `flash_forward` marker (set in enable_qwen3_tidal_attention)
            # and a `layer_idx`; subclasses are Qwen3SdpaAttention/FlashAttention2 of Qwen3Attention.
            bases = [b.__name__ for b in type(m).__mro__]
            if "Qwen3Attention" in bases and hasattr(m, "q_proj"):
                m.tidal_keep_ratio = sparse_ratio
                n_set += 1
        print(f"[overall-sparsity] keep_ratio={args.keep_ratio} -> per-sparse-layer ratio={sparse_ratio:.4f} "
              f"({full_layers} full + {n_sparse} sparse layers); set on {n_set} modules", flush=True)

    eos_token_ids = {tokenizer.eos_token_id}
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is not None:
            eos_token_ids.add(tok_id)

    data = load_dataset('THUDM/LongBench-v2', split='train')
    num_samples = len(data) if args.max_samples <= 0 else min(args.max_samples, len(data))

    print(f"LongBench-v2: {num_samples} samples | attn={args.attn_type} top_k={args.top_k} "
          f"datalen={args.datalen} max_gen={max_gen} thinking={args.enable_thinking}")
    print(f"Output: {out_path}")

    scores = []
    for i in tqdm(range(num_samples), desc="longbenchv2"):
        if i in done_idx:
            continue
        sample = data[i]
        input_ids = build_prompt(tokenizer, sample, args.datalen, args.enable_thinking)

        t0 = time.time()
        generated = greedy_generate(model, input_ids, max_gen, eos_token_ids)
        elapsed = time.time() - t0

        raw_text = tokenizer.decode(generated, skip_special_tokens=True)
        pred_text = strip_thinking(raw_text)
        correct = longbenchv2_metric(pred_text, sample['answer'])
        scores.append(correct)

        record = {
            "idx": i,
            "prediction": pred_text,
            "ground_truth": sample['answer'],
            "correct": correct,
            "extracted": longbenchv2_extract_answer(postprocess_pred(pred_text).strip()),
            "input_len": input_ids.size(1),
            "output_len": len(generated),
            "difficulty": sample.get('difficulty', ''),
            "length": sample.get('length', ''),
            "domain": sample.get('domain', ''),
            "elapsed_sec": round(elapsed, 2),
        }
        if args.enable_thinking:
            record["raw_output_with_think"] = raw_text
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # final summary over the full output file (including resumed samples)
    records = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if records:
        acc = sum(r["correct"] for r in records) / len(records)
        summary = {
            "run": run_name,
            "model": model_short,
            "samples": len(records),
            "accuracy": round(acc, 4),
        }
        for key in ("difficulty", "length"):
            groups = {}
            for r in records:
                groups.setdefault(r.get(key, ""), []).append(r["correct"])
            summary[f"by_{key}"] = {
                k: {"n": len(v), "acc": round(sum(v) / len(v), 4)}
                for k, v in sorted(groups.items())
            }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        with open(os.path.join(out_dir, f"{run_name}.summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
