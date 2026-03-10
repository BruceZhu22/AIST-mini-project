#!/usr/bin/env python3
"""
Before-vs-After Comparison
===========================
Loads the base pretrained model and the OFT-finetuned adapter, then runs
a set of example prompts through both and prints/saves the comparison.

Usage:
    python inference.py                                   # uses defaults
    python inference.py --base Qwen/Qwen2-0.5B-Instruct \\
                        --adapter ./output/qwen2-0.5b-oft \\
                        --output_file ./output/comparison.md
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ──────────────────────────────────────────────────────────────
# Test prompts — intentionally varied to show generalisation
# ──────────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "instruction": "Explain what orthogonal fine-tuning (OFT) is and how it differs from LoRA.",
        "input": "",
    },
    {
        "instruction": "Write a Python function that computes the Fibonacci sequence up to n.",
        "input": "",
    },
    {
        "instruction": "Translate the following English sentence to French.",
        "input": "Artificial intelligence is transforming every aspect of modern life.",
    },
    {
        "instruction": "Summarise the following paragraph in one sentence.",
        "input": (
            "Transformers, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), "
            "replaced recurrent architectures with self-attention mechanisms, enabling highly "
            "parallelisable training and yielding state-of-the-art results across NLP tasks."
        ),
    },
    {
        "instruction": "Convert 98.6 degrees Fahrenheit to Celsius.",
        "input": "",
    },
    {
        "instruction": "Give three practical tips for improving sleep quality.",
        "input": "",
    },
]

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_WITHOUT_INPUT.format(instruction=instruction)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def parse_args():
    p = argparse.ArgumentParser(description="Before/after OFT comparison")
    p.add_argument("--base", default="Qwen/Qwen2-0.5B-Instruct",
                   help="Base model name or path")
    p.add_argument("--adapter", default="./output/qwen2-0.5b-oft",
                   help="Path to the saved OFT adapter (output of train_oft.py)")
    p.add_argument("--output_file", default="./output/comparison.md",
                   help="Where to write the comparison markdown")
    p.add_argument("--max_new_tokens", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    dtype = torch.float32   # CPU-safe

    # ── Tokeniser ──────────────────────────────────────────────
    print("Loading tokeniser …")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base model ─────────────────────────────────────────────
    print("Loading base model …")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, trust_remote_code=True
    )
    base_model.eval()

    # ── OFT-finetuned model ────────────────────────────────────
    print("Loading OFT adapter …")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, trust_remote_code=True
    )
    ft_model = PeftModel.from_pretrained(ft_model, args.adapter)
    ft_model = ft_model.merge_and_unload()  # merge OFT weights for faster inference
    ft_model.eval()

    # ── Generate ───────────────────────────────────────────────
    results = []
    for i, ex in enumerate(EXAMPLES, 1):
        prompt = build_prompt(ex["instruction"], ex.get("input", ""))
        print(f"\n{'─'*60}\n[{i}/{len(EXAMPLES)}] {ex['instruction'][:70]}")

        base_out = generate(base_model, tokenizer, prompt, args.max_new_tokens)
        ft_out   = generate(ft_model,   tokenizer, prompt, args.max_new_tokens)

        print(f"  Base : {base_out[:120]} …" if len(base_out) > 120 else f"  Base : {base_out}")
        print(f"  OFT  : {ft_out[:120]} …"   if len(ft_out)   > 120 else f"  OFT  : {ft_out}")

        results.append({
            "instruction": ex["instruction"],
            "input":       ex.get("input", ""),
            "base_output": base_out,
            "oft_output":  ft_out,
        })

    # ── Save markdown ──────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write("# OFT Finetuning — Before vs. After Comparison\n\n")
        f.write(f"**Base model**: `{args.base}`  \n")
        f.write(f"**OFT adapter**: `{args.adapter}`\n\n")
        f.write("---\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"## Example {i}\n\n")
            f.write(f"**Instruction:** {r['instruction']}\n\n")
            if r["input"]:
                f.write(f"**Input:** {r['input']}\n\n")
            f.write("**Base model response:**\n\n")
            f.write(f"> {r['base_output'].replace(chr(10), '  \n> ')}\n\n")
            f.write("**OFT-finetuned response:**\n\n")
            f.write(f"> {r['oft_output'].replace(chr(10), '  \n> ')}\n\n")
            f.write("---\n\n")

    print(f"\nComparison saved → {args.output_file}")


if __name__ == "__main__":
    main()
