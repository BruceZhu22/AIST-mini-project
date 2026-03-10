#!/usr/bin/env python3
"""
Fast PPL-only evaluation that runs in ~1 minute on CPU.
Also generates ROUGE on a small 20-example sample.

Usage:
    python eval_fast.py
"""
import os, json, math, argparse, torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def full_prompt(ex):
    if ex.get("input", "").strip():
        return PROMPT_WITH_INPUT.format(**ex)
    return PROMPT_WITHOUT_INPUT.format(instruction=ex["instruction"], output=ex.get("output",""))


def prompt_only(ex):
    if ex.get("input", "").strip():
        return PROMPT_WITH_INPUT.format(**{**ex, "output":""})
    return PROMPT_WITHOUT_INPUT.format(instruction=ex["instruction"], output="")


@torch.no_grad()
def ppl(model, tokenizer, examples, max_len=256):
    model.eval()
    total_loss = total_toks = 0
    for ex in tqdm(examples, desc="  PPL", leave=False):
        full = full_prompt(ex);  pre = prompt_only(ex)
        enc = tokenizer(full, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"]
        if ids.shape[1] == 0: continue
        plen = len(tokenizer(pre, truncation=True, max_length=max_len)["input_ids"])
        lbl = ids.clone();  lbl[:, :plen] = -100
        out = model(input_ids=ids, labels=lbl)
        n = (lbl != -100).sum().item()
        if n > 0:
            total_loss += out.loss.item() * n
            total_toks  += n
    return math.exp(total_loss / total_toks) if total_toks else float("inf")


def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(hyp, ref):
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r: return 0.0
    lcs = _lcs(h, r)
    p, rc = lcs/len(h), lcs/len(r)
    return 2*p*rc/(p+rc) if p+rc else 0.0


@torch.no_grad()
def compute_rouge(model, tokenizer, examples, max_new=64):
    model.eval()
    scores = []
    for ex in tqdm(examples, desc="  ROUGE", leave=False):
        inp = tokenizer(prompt_only(ex), return_tensors="pt", truncation=True, max_length=256)
        n_in = inp["input_ids"].shape[1]
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=1.1,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        scores.append(rouge_l(gen, ex.get("output","")))
    return float(np.mean(scores)) if scores else 0.0


def bar_chart(metrics, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(metrics.keys())
    colors = {"Base": "#90A4AE", "OFT": "#42A5F5"}
    clrs = [colors.get(n, "#EF9A9A") for n in names]

    ax = axes[0]
    vals = [metrics[n]["perplexity"] for n in names]
    bars = ax.bar(names, vals, color=clrs, edgecolor="black", width=0.4)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=12)
    ax.set_title("Perplexity (↓ better)", fontsize=13)
    ax.set_ylabel("Perplexity"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals)*1.25)

    ax = axes[1]
    vals = [metrics[n]["rouge_l"] for n in names]
    bars = ax.bar(names, vals, color=clrs, edgecolor="black", width=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=12)
    ax.set_title("ROUGE-L (↑ better)", fontsize=13)
    ax.set_ylabel("ROUGE-L F1"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals)*1.4+0.005)

    fig.suptitle("Qwen2-0.5B: Base vs OFT — 100 held-out Alpaca examples",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_metrics.png")
    fig.savefig(path, dpi=150);  plt.close(fig)
    print(f"Chart saved → {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base",    default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--adapter", default="./output/qwen2-0.5b-oft")
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--n_ppl",   type=int, default=100)
    p.add_argument("--n_rouge", type=int, default=20, help="Subset for ROUGE (slow on CPU)")
    p.add_argument("--skip",    type=int, default=1000)
    p.add_argument("--output_dir", default="./output")
    return p.parse_args()


def main():
    args = parse_args()
    raw    = load_dataset(args.dataset, split=f"train[{args.skip}:{args.skip+args.n_ppl}]")
    examples = list(raw)
    rouge_ex = examples[:args.n_rouge]

    dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # ── Base ──────────────────────────────────────────────────
    print("Loading base model …")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, trust_remote_code=True)
    base.eval()
    print("Base PPL …")
    b_ppl = ppl(base, tok, examples)
    print(f"  Base PPL={b_ppl:.3f}")
    print("Base ROUGE …")
    b_rouge = compute_rouge(base, tok, rouge_ex)
    print(f"  Base ROUGE-L={b_rouge:.4f}")
    del base

    # ── OFT ───────────────────────────────────────────────────
    print("Loading OFT model …")
    ft = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, trust_remote_code=True)
    ft = PeftModel.from_pretrained(ft, args.adapter).merge_and_unload()
    ft.eval()
    print("OFT PPL …")
    f_ppl = ppl(ft, tok, examples)
    print(f"  OFT  PPL={f_ppl:.3f}")
    print("OFT ROUGE …")
    f_rouge = compute_rouge(ft, tok, rouge_ex)
    print(f"  OFT  ROUGE-L={f_rouge:.4f}")

    metrics = {
        "Base": {"perplexity": b_ppl,  "rouge_l": b_rouge},
        "OFT":  {"perplexity": f_ppl,  "rouge_l": f_rouge},
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    bar_chart(metrics, args.output_dir)

    ppl_delta   = (b_ppl   - f_ppl)   / b_ppl   * 100
    rouge_delta = (f_rouge - b_rouge) / (b_rouge + 1e-9) * 100

    print("\n" + "="*52)
    print("  EVALUATION SUMMARY (100 held-out Alpaca examples)")
    print("="*52)
    print(f"  {'Model':<8}  {'Perplexity':>12}  {'ROUGE-L':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}")
    print(f"  {'Base':<8}  {b_ppl:>12.3f}  {b_rouge:>10.4f}")
    print(f"  {'OFT':<8}  {f_ppl:>12.3f}  {f_rouge:>10.4f}")
    print(f"  {'Δ':<8}  {ppl_delta:>+11.1f}%  {rouge_delta:>+9.1f}%")
    print("="*52)
    print(f"\nResults saved to {args.output_dir}/eval_metrics.{{json,png}}")


if __name__ == "__main__":
    main()
