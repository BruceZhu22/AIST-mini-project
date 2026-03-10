#!/usr/bin/env python3
"""
OFT (Orthogonal Fine-Tuning) — Instruction Tuning with Qwen2-0.5B on Alpaca
=====================================================================================
Reference: https://huggingface.co/docs/peft/main/en/conceptual_guides/oft
OFT paper: "Controlling Text-to-Image Diffusion by Orthogonal Finetuning"
           Zeju Qiu et al., NeurIPS 2023

Key idea: Instead of adding low-rank adapters (LoRA), OFT updates weights via
a block-diagonal orthogonal transformation R, so that W' = R W.
This preserves the hyperspherical energy of neurons, keeping pretrained features
intact while adapting the model to a new task.

Usage:
    python train_oft.py                          # default settings
    python train_oft.py --model Qwen/Qwen2-1.5B-Instruct --dataset_size 2000
"""

import os
import json
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    set_seed,
)
from peft import OFTConfig, get_peft_model, TaskType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Alpaca-style prompt templates
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


def build_prompt(example: dict, include_output: bool = True) -> str:
    out = example.get("output", "") if include_output else ""
    if example.get("input", "").strip():
        return PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=out,
        )
    return PROMPT_WITHOUT_INPUT.format(
        instruction=example["instruction"],
        output=out,
    )


# ──────────────────────────────────────────────────────────────
# Tokenisation
# ──────────────────────────────────────────────────────────────
def tokenize_example(example: dict, tokenizer, max_length: int):
    full_text = build_prompt(example, include_output=True)
    prompt_only = build_prompt(example, include_output=False)

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # Only compute loss on the response tokens — mask the prompt with -100
    prompt_ids = tokenizer(
        prompt_only,
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]

    labels = tokenized["input_ids"].copy()
    prefix_len = min(len(prompt_ids), len(labels))
    labels[:prefix_len] = [-100] * prefix_len

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


# ──────────────────────────────────────────────────────────────
# Loss-curve logger callback
# ──────────────────────────────────────────────────────────────
class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_log: List[dict] = []
        self.eval_log: List[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs and "eval_loss" not in logs:
            self.train_log.append({"step": step, "loss": logs["loss"]})
        if "eval_loss" in logs:
            self.eval_log.append({"step": step, "eval_loss": logs["eval_loss"]})


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────
def plot_loss_curves(train_log, eval_log, output_dir: str, model_name: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    if train_log:
        steps = [e["step"] for e in train_log]
        losses = [e["loss"] for e in train_log]
        ax.plot(steps, losses, label="Train Loss", color="#2196F3", lw=2, alpha=0.85)

    if eval_log:
        steps = [e["step"] for e in eval_log]
        losses = [e["eval_loss"] for e in eval_log]
        ax.plot(steps, losses, label="Eval Loss", color="#FF5722",
                marker="o", ms=5, lw=2)

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title(f"OFT Fine-Tuning Loss — {model_name}", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "training_loss.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curve saved → %s", out_path)
    return out_path


# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="OFT instruction tuning")
    p.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct",
                   help="HuggingFace model name or local path")
    p.add_argument("--dataset", default="tatsu-lab/alpaca",
                   help="HuggingFace dataset name")
    p.add_argument("--dataset_size", type=int, default=1000,
                   help="Number of training examples to use (max 52002)")
    p.add_argument("--output_dir", default="./output/qwen2-0.5b-oft",
                   help="Directory to save model and results")
    p.add_argument("--max_length", type=int, default=256,
                   help="Maximum sequence length for tokenisation")

    # OFT hyperparameters
    p.add_argument("--oft_block_size", type=int, default=8,
                   help="OFT block size. Must divide the weight dimension.")
    p.add_argument("--oft_modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="Which attention modules to apply OFT to")
    p.add_argument("--oft_coft", action="store_true",
                   help="Use constrained OFT (cOFT) with Frobenius-norm regularisation")
    p.add_argument("--module_dropout", type=float, default=0.0,
                   help="Dropout applied to OFT modules during training")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()
# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Tokeniser ──────────────────────────────────────────
    logger.info("Loading tokeniser: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Base model ─────────────────────────────────────────
    logger.info("Loading base model …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,       # CPU-safe (float16 unsupported on CPU)
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # ── 3. OFT wrap ───────────────────────────────────────────
    logger.info(
        "Wrapping with OFT (block_size=%d, modules=%s) …",
        args.oft_block_size, args.oft_modules,
    )
    oft_config = OFTConfig(
        oft_block_size=args.oft_block_size,
        target_modules=args.oft_modules,
        module_dropout=args.module_dropout,
        init_weights=True,
        coft=args.oft_coft,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, oft_config)
    model.print_trainable_parameters()

    # Log OFT config
    oft_cfg_path = os.path.join(args.output_dir, "oft_config.json")
    with open(oft_cfg_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "oft_block_size": args.oft_block_size,
                "target_modules": args.oft_modules,
                "coft": args.oft_coft,
                "module_dropout": args.module_dropout,
            },
            f, indent=2,
        )

    # ── 4. Dataset ────────────────────────────────────────────
    logger.info("Loading dataset '%s' (n=%d) …", args.dataset, args.dataset_size)
    raw = load_dataset(args.dataset, split=f"train[:{args.dataset_size}]")

    tokenized = raw.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_length),
        remove_columns=raw.column_names,
        num_proc=min(args.num_workers, os.cpu_count() or 1),
        desc="Tokenising",
    )
    split = tokenized.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    logger.info("Train: %d  |  Eval: %d", len(train_ds), len(eval_ds))

    # ── 5. Training ───────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.05 * args.epochs * (int(args.dataset_size * 0.9) // (args.batch_size * args.grad_accum))),
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,          # CPU only — float16 unsupported
        bf16=False,
        report_to=["none"],
        dataloader_num_workers=0,   # avoid fork issues in notebook envs
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=-100,
    )

    loss_cb = LossLoggerCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[loss_cb],
    )

    # ── 6. Train ──────────────────────────────────────────────
    logger.info("Training started …")
    train_result = trainer.train()

    # ── 7. Save ───────────────────────────────────────────────
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # ── 8. Loss curves + raw log ──────────────────────────────
    plot_loss_curves(loss_cb.train_log, loss_cb.eval_log, args.output_dir,
                     model_name=args.model.split("/")[-1])

    log_path = os.path.join(args.output_dir, "log_history.json")
    with open(log_path, "w") as f:
        json.dump(
            {"train": loss_cb.train_log, "eval": loss_cb.eval_log},
            f, indent=2,
        )
    logger.info("Log history saved → %s", log_path)
    logger.info("All done. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
