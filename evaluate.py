#!/usr/bin/env python3
"""
Quantitative Evaluation
=========================
Measures perplexity and/or ROUGE-L on a held-out slice of the Alpaca dataset
for both the base model and the OFT-finetuned model, then produces a summary
table and bar chart.

Usage:
    python evaluate.py                            # uses defaults
    python evaluate.py --base Qwen/Qwen2-0.5B-Instruct \\
                       --adapter ./output/qwen2-0.5b-oft \\
                       --n_eval 100
"""

import os
import json
import math
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def build_full_prompt(ex: dict) -> str:
    if ex.get("input", "").strip():
        return PROMPT_WITH_INPUT.format(**ex)
    return PROMPT_WITHOUT_INPUT.format(
        instruction=ex["instruction"], output=ex.get("output", "")
    )


# ──────────────────────────────────────────────────────────────
# Perplexity (response-only, prompt tokens masked)
# ──────────────────────────────────────────────────────────────
def response_marker(ex: dict) -> str:
    """Return the prompt portion (without the output) to identify prefix length."""
    if ex.get("input", "").strip():
        return PROMPT_WITH_INPUT.format(**{**ex, "output": ""})
    return PROMPT_WITHOUT_INPUT.format(instruction=ex["instruction"], output="")


@torch.no_grad()
def compute_perplexity(model, tokenizer, examples, max_length: int = 256) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for ex in tqdm(examples, desc="  PPL", leave=False):
        full_text = build_full_prompt(ex)
        prompt    = response_marker(ex)

        enc = tokenizer(full_text, truncation=True, max_length=max_length,
                        return_tensors="pt")
        input_ids = enc["input_ids"]
        if input_ids.shape[1] == 0:
            continue

        # Mask prompt — only penalise the response tokens
        prompt_len = len(tokenizer(prompt, truncation=True,
                                   max_length=max_length)["input_ids"])
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        out = model(input_ids=input_ids, labels=labels)
        # out.loss is mean CE over non-masked tokens (if any)
        n_tokens = (labels != -100).sum().item()
        if n_tokens > 0:
            total_loss   += out.loss.item() * n_tokens
            total_tokens += n_tokens

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


# ──────────────────────────────────────────────────────────────
# Simple ROUGE-L (no external dependencies)
# ──────────────────────────────────────────────────────────────
def _lcs_length(x: list, y: list) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(hypothesis: str, reference: str) -> float:
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0
    lcs = _lcs_length(h_tokens, r_tokens)
    precision = lcs / len(h_tokens)
    recall    = lcs / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def compute_rouge(model, tokenizer, examples, max_new_tokens: int = 128) -> float:
    model.eval()
    scores = []

    def _prompt_only(ex):
        if ex.get("input", "").strip():
            return PROMPT_WITH_INPUT.format(**{**ex, "output": ""})
        return PROMPT_WITHOUT_INPUT.format(instruction=ex["instruction"], output="")

    for ex in tqdm(examples, desc="  ROUGE", leave=False):
        prompt = _prompt_only(ex)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        n_in = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        scores.append(rouge_l(generated, ex.get("output", "")))

    return float(np.mean(scores)) if scores else 0.0


# ──────────────────────────────────────────────────────────────
# Bar chart
# ──────────────────────────────────────────────────────────────
def plot_metrics(metrics: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Perplexity (lower is better)
    ax = axes[0]
    models = list(metrics.keys())
    ppl_vals = [metrics[m]["perplexity"] for m in models]
    bars = ax.bar(models, ppl_vals, color=["#90A4AE", "#42A5F5"], edgecolor="black", width=0.4)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
    ax.set_title("Perplexity (↓ better)", fontsize=13)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_ylim(0, max(ppl_vals) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    # ROUGE-L (higher is better)
    ax = axes[1]
    rouge_vals = [metrics[m]["rouge_l"] for m in models]
    bars = ax.bar(models, rouge_vals, color=["#90A4AE", "#EF5350"], edgecolor="black", width=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11)
    ax.set_title("ROUGE-L (↑ better)", fontsize=13)
    ax.set_ylabel("ROUGE-L F1", fontsize=12)
    ax.set_ylim(0, max(rouge_vals) * 1.3 + 0.01)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Evaluation: Base vs OFT-Finetuned Model", fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_metrics.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Metrics chart saved → {out_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate base vs OFT model")
    p.add_argument("--base",    default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--adapter", default="./output/qwen2-0.5b-oft")
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--n_eval",  type=int, default=100,
                   help="Number of held-out examples to evaluate")
    p.add_argument("--skip",    type=int, default=1000,
                   help="Skip these many training examples at the start")
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--max_ppl_len",   type=int, default=256)
    p.add_argument("--max_new_rouge", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()

    # Held-out examples (outside of training set)
    print(f"Loading eval split [{args.skip}:{args.skip+args.n_eval}] …")
    raw = load_dataset(args.dataset,
                       split=f"train[{args.skip}:{args.skip + args.n_eval}]")
    examples = list(raw)

    dtype = torch.float32
    print("Loading tokeniser …")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base model ───────────────────────────────────────────
    print("Loading base model …")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, trust_remote_code=True
    )
    base_model.eval()

    print("Evaluating base model …")
    base_ppl   = compute_perplexity(base_model, tokenizer, examples, args.max_ppl_len)
    base_rouge = compute_rouge(base_model, tokenizer, examples, args.max_new_rouge)
    print(f"  Base PPL={base_ppl:.2f}  ROUGE-L={base_rouge:.4f}")

    del base_model  # free memory

    # ── OFT model ────────────────────────────────────────────
    print("Loading OFT-finetuned model …")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, trust_remote_code=True
    )
    ft_model = PeftModel.from_pretrained(ft_model, args.adapter)
    ft_model = ft_model.merge_and_unload()
    ft_model.eval()

    print("Evaluating OFT model …")
    ft_ppl   = compute_perplexity(ft_model, tokenizer, examples, args.max_ppl_len)
    ft_rouge = compute_rouge(ft_model, tokenizer, examples, args.max_new_rouge)
    print(f"  OFT  PPL={ft_ppl:.2f}  ROUGE-L={ft_rouge:.4f}")

    # ── Summary ──────────────────────────────────────────────
    metrics = {
        "Base":    {"perplexity": base_ppl,   "rouge_l": base_rouge},
        "OFT":     {"perplexity": ft_ppl,     "rouge_l": ft_rouge},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_metrics(metrics, args.output_dir)

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"{'Model':<12}  {'Perplexity':>12}  {'ROUGE-L':>10}")
    print("-"*38)
    for name, m in metrics.items():
        print(f"{name:<12}  {m['perplexity']:>12.2f}  {m['rouge_l']:>10.4f}")
    ppl_delta   = (base_ppl - ft_ppl) / base_ppl * 100
    rouge_delta = (ft_rouge - base_rouge) / (base_rouge + 1e-9) * 100
    print("-"*38)
    print(f"{'Δ (OFT-Base)':<12}  {ppl_delta:>+11.1f}%  {rouge_delta:>+9.1f}%")
    print("="*50)


if __name__ == "__main__":
    main()
