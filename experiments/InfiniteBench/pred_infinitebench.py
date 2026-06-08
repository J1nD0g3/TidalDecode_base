"""InfiniteBench evaluation for TidalDecode (generation-based scoring).

Prompts, truncation and metrics ported verbatim from ShadowKV
(data/dataset.py, data/metrics.py) for cross-repo comparability.

Usage:
  INFINITEBENCH_DIR=/workspace/data/InfiniteBench \
  python experiments/InfiniteBench/pred_infinitebench.py \
      --model /workspace/models/Qwen3-14B-128k --attn_type tidal --top_k 4096 \
      --tasks passkey,kv_retrieval --datalen 122880 --enable_thinking
"""
import os
import re
import sys
import json
import time
import string
import argparse
from collections import Counter

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

INFINITEBENCH_PROMPTS = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",
}

INFINITEBENCH_GEN_LEN = {
    "passkey": 30,
    "number_string": 50,
    "kv_retrieval": 50,
    "longbook_sum_eng": 1200,
    "longbook_qa_eng": 40,
    "longbook_qa_chn": 40,
    "longdialogue_qa_eng": 40,
    "math_find": 30,
    "code_debug": 30,
    "longbook_choice_eng": 30,
}


# ============================================================
# Ported verbatim from ShadowKV data/metrics.py
# ============================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def postprocess_pred(predict_str: str):

    predict_str = predict_str.strip().replace('<|eot_id|>', '').replace('</s>', '').replace('</s', '').replace('</', '')

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def _f1_score(prediction, ground_truth):
    """Token-level F1 score."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def rouge_score(prediction, ground_truth):
    """ROUGE-L F1 score for summarization tasks."""
    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("Please install rouge: pip install rouge")
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def infinitebench_metric(prediction, ground_truth, task_name):
    """Unified InfiniteBench metric dispatcher."""
    prediction = postprocess_pred(prediction).strip()

    if task_name in ('passkey', 'number_string', 'kv_retrieval'):
        # Exact substring match
        answer = str(ground_truth).strip()
        return 1.0 if answer in prediction else 0.0

    elif task_name == 'longbook_qa_eng':
        # F1 score
        gts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        best = 0.0
        for gt in gts:
            pred_tokens = normalize_answer(prediction).split()
            gt_tokens = normalize_answer(gt).split()
            if pred_tokens and gt_tokens:
                best = max(best, _f1_score(pred_tokens, gt_tokens))
        return best

    elif task_name == 'longbook_qa_chn':
        # Chinese F1 score with jieba
        try:
            import jieba
        except ImportError:
            return 0.0
        gts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        cn_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punc = set(string.punctuation + cn_punc)

        def _norm_zh(s):
            return "".join(ch for ch in s.lower() if ch not in all_punc).replace(" ", "")

        best = 0.0
        for gt in gts:
            pred_tokens = [_norm_zh(t) for t in jieba.cut(prediction, cut_all=False)]
            gt_tokens = [_norm_zh(t) for t in jieba.cut(gt, cut_all=False)]
            pred_tokens = [t for t in pred_tokens if t]
            gt_tokens = [t for t in gt_tokens if t]
            if pred_tokens and gt_tokens:
                best = max(best, _f1_score(pred_tokens, gt_tokens))
        return best

    elif task_name in ('longbook_choice_eng', 'code_debug'):
        # Multiple choice: check A/B/C/D
        answer = ground_truth
        if isinstance(answer, list):
            correct_letter = answer[1] if len(answer) > 1 else answer[0]
            correct_text = answer[0]
        else:
            correct_letter = answer
            correct_text = answer
        pred_upper = prediction.upper().strip()
        if correct_letter.upper() in pred_upper[:5]:
            return 1.0
        if correct_text.lower() in prediction.lower():
            return 1.0
        return 0.0

    elif task_name == 'longbook_sum_eng':
        # ROUGE-L
        if not prediction or not str(ground_truth):
            return 0.0
        return rouge_score(prediction, str(ground_truth))

    elif task_name == 'longdialogue_qa_eng':
        # Character name match
        answer = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        return 1.0 if answer.lower() in prediction.lower() else 0.0

    elif task_name == 'math_find':
        # First integer match
        answer = str(ground_truth).strip()
        pred_nums = re.split(r"[^0-9]", prediction)
        for item in pred_nums:
            if item:
                return 1.0 if item == answer else 0.0
        return 0.0

    else:
        # Fallback: substring match
        return 1.0 if str(ground_truth).lower() in prediction.lower() else 0.0


# ============================================================
# Prompt building with ShadowKV-style middle truncation
# ============================================================

def wrap_qwen3_chat(prompt, enable_thinking):
    if enable_thinking:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def fill_template(task, template, sample, context):
    if task == 'code_debug':
        return template.format(
            context=context,
            OPTION_A=sample['options'][0], OPTION_B=sample['options'][1],
            OPTION_C=sample['options'][2], OPTION_D=sample['options'][3],
        )
    elif task == 'longbook_choice_eng':
        return template.format(
            context=context, question=sample['input'],
            OPTION_A=sample['options'][0], OPTION_B=sample['options'][1],
            OPTION_C=sample['options'][2], OPTION_D=sample['options'][3],
        )
    elif task in ('longbook_qa_eng', 'longbook_qa_chn'):
        return template.format(context=context, question=sample['input'])
    elif task in ('longbook_sum_eng', 'longdialogue_qa_eng'):
        return template.format(context=context)
    elif task == 'math_find':
        prompt = sample['input']
        find_result = re.findall(r"The .+ of", prompt)
        assert find_result, f"Cannot find target number in: {prompt}"
        target_number = find_result[0].lower()[:-3]
        prefix = f"What is {target_number} in the following list?"
        return template.format(prefix=prefix, context=context, input=prompt)
    else:
        # passkey, number_string, kv_retrieval
        return template.format(context=context, input=sample['input'])


def build_prompt(tokenizer, task, sample, datalen, enable_thinking):
    template = INFINITEBENCH_PROMPTS[task]
    context = sample['context']
    prompt_text = fill_template(task, template, sample, context)
    chat_text = wrap_qwen3_chat(prompt_text, enable_thinking)
    input_ids = tokenizer.encode(chat_text, return_tensors="pt", add_special_tokens=False)
    if input_ids.size(1) > datalen:
        empty_prompt = fill_template(task, template, sample, '')
        overhead_ids = tokenizer.encode(
            wrap_qwen3_chat(empty_prompt, enable_thinking),
            return_tensors="pt", add_special_tokens=False,
        )
        max_ctx_tokens = datalen - overhead_ids.size(1)
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(ctx_ids) > max_ctx_tokens:
            half = max_ctx_tokens // 2
            ctx_ids = ctx_ids[:half] + ctx_ids[-half:]
        truncated_ctx = tokenizer.decode(ctx_ids, skip_special_tokens=False)
        prompt_text = fill_template(task, template, sample, truncated_ctx)
        chat_text = wrap_qwen3_chat(prompt_text, enable_thinking)
        input_ids = tokenizer.encode(chat_text, return_tensors="pt", add_special_tokens=False)
    return input_ids


def get_ground_truth(task, sample):
    if task in ('code_debug', 'longbook_choice_eng'):
        OPTIONS = "ABCD"
        answer = sample['answer']
        if isinstance(answer, str):
            return [answer, OPTIONS[sample['options'].index(answer)]]
        elif isinstance(answer, list):
            if len(answer) == 1:
                return [answer[0], OPTIONS[sample['options'].index(answer[0])]]
            return answer
        return answer
    return sample['answer']


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def greedy_generate(model, input_ids, max_gen, eos_token_ids):
    input_ids = input_ids.to(model.device)
    output = model(input_ids=input_ids, past_key_values=None, use_cache=True, num_logits_to_keep=1)
    past_key_values = output.past_key_values
    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated = [pred_token_idx.item()]
    for _ in range(max_gen - 1):
        if generated[-1] in eos_token_ids:
            break
        output = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated.append(pred_token_idx.item())
    return generated


def strip_thinking(text):
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
    torch.cuda.manual_seed_all(seed)


ALL_TASKS = list(INFINITEBENCH_PROMPTS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--attn_type", type=str, choices=["tidal", "full"], default="tidal")
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--sparse_layer_start", type=int, default=2)
    parser.add_argument("--keep_ratio", type=float, default=None,
                        help="overall KV keep ratio to match across methods (e.g. 0.27). Overrides top_k; converts to per-sparse-layer ratio accounting for full layers.")
    parser.add_argument("--correction_layer", type=int, default=17)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--datalen", type=int, default=122880)
    parser.add_argument("--max_gen", type=int, default=0,
                        help="max new tokens (0 = per-task default, or context-capped with thinking)")
    parser.add_argument("--tasks", type=str, default=",".join(ALL_TASKS))
    parser.add_argument("--max_samples", type=int, default=100,
                        help="max samples per task (0 = all)")
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    with open(os.path.join(args.model, "config.json")) as f:
        max_position = json.load(f).get("max_position_embeddings", 40960)
    assert args.datalen < max_position, f"datalen {args.datalen} must be < {max_position}"

    infinitebench_dir = os.environ.get('INFINITEBENCH_DIR', '/workspace/data/InfiniteBench')
    model_short = os.path.basename(args.model.rstrip("/"))
    think_tag = "think_on" if args.enable_thinking else "think_off"
    if args.attn_type == "tidal":
        run_tag = f"tidal-{args.top_k}-c{args.correction_layer}-{think_tag}"
    else:
        run_tag = f"full-{think_tag}"
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "pred", model_short)
    os.makedirs(out_dir, exist_ok=True)

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
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None:
            eos_token_ids.add(tid)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_summaries = {}
    for task in task_list:
        assert task in INFINITEBENCH_PROMPTS, f"Unknown task {task}"
        out_path = os.path.join(out_dir, f"infinitebench_{task}-{run_tag}.jsonl")

        done_idx = set()
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        done_idx.add(json.loads(line)["idx"])
                    except (json.JSONDecodeError, KeyError):
                        pass

        data = load_dataset("json", data_files=f'{infinitebench_dir}/{task}.jsonl', split='train')
        num_samples = len(data) if args.max_samples <= 0 else min(args.max_samples, len(data))
        print(f"[{task}] {num_samples} samples ({len(done_idx)} done) -> {out_path}")

        for i in tqdm(range(num_samples), desc=task):
            if i in done_idx:
                continue
            sample = data[i]
            input_ids = build_prompt(tokenizer, task, sample, args.datalen, args.enable_thinking)
            gt = get_ground_truth(task, sample)

            max_gen = args.max_gen
            if max_gen <= 0:
                if args.enable_thinking:
                    max_gen = max_position - input_ids.size(1)
                else:
                    max_gen = INFINITEBENCH_GEN_LEN[task]
            max_gen = min(max_gen, max_position - input_ids.size(1))

            t0 = time.time()
            generated = greedy_generate(model, input_ids, max_gen, eos_token_ids)
            elapsed = time.time() - t0

            raw_text = tokenizer.decode(generated, skip_special_tokens=True)
            pred_text = strip_thinking(raw_text)
            score = infinitebench_metric(pred_text, gt, task)

            record = {
                "idx": i,
                "prediction": pred_text,
                "ground_truth": gt,
                "score": score,
                "input_len": input_ids.size(1),
                "output_len": len(generated),
                "elapsed_sec": round(elapsed, 2),
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        records = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        if records:
            avg = sum(r["score"] for r in records) / len(records)
            all_summaries[task] = {"n": len(records), "score": round(avg, 4)}
            print(f"[{task}] n={len(records)} score={avg:.4f}")

    summary = {"run": run_tag, "model": model_short, "datalen": args.datalen, "tasks": all_summaries}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(os.path.join(out_dir, f"infinitebench-{run_tag}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
