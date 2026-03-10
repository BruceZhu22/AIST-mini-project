#!/usr/bin/env python3
"""
Visualise saved training logs.

Can be run independently to re-generate plots from a previous run:
    python plot_results.py --log_dir ./output/qwen2-0.5b-oft
"""

import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(values, window=5):
    """Simple moving-average smoothing."""
    if len(values) < window:
        return values
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid").tolist()


def plot_from_log(log_path: str, output_dir: str):
    with open(log_path) as f:
        data = json.load(f)

    train_log = data.get("train", [])
    eval_log  = data.get("eval",  [])

    # Also accept flat HF Trainer log_history format
    if not train_log and not eval_log and isinstance(data, list):
        for entry in data:
            step = entry.get("step", 0)
            if "loss" in entry and "eval_loss" not in entry:
                train_log.append({"step": step, "loss": entry["loss"]})
            if "eval_loss" in entry:
                eval_log.append({"step": step, "eval_loss": entry["eval_loss"]})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: raw + smoothed train loss ──────────────────────
    ax = axes[0]
    if train_log:
        steps  = [e["step"]  for e in train_log]
        losses = [e["loss"]  for e in train_log]
        ax.plot(steps, losses, color="#BBDEFB", lw=1, alpha=0.7, label="Train (raw)")
        smoothed = smooth(losses, window=min(10, len(losses)))
        s_steps  = steps[len(steps) - len(smoothed):]
        ax.plot(s_steps, smoothed, color="#1565C0", lw=2, label="Train (smoothed)")

    if eval_log:
        e_steps   = [e["step"]     for e in eval_log]
        e_losses  = [e["eval_loss"] for e in eval_log]
        ax.plot(e_steps, e_losses, color="#FF5722", marker="o", ms=5, lw=2,
                label="Eval Loss")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Loss Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Right: eval loss only (cleaner) ──────────────────────
    ax = axes[1]
    if eval_log:
        ax.plot(e_steps, e_losses, color="#FF5722", marker="o", ms=6, lw=2,
                markerfacecolor="white", markeredgewidth=2)
        ax.fill_between(e_steps, e_losses, alpha=0.1, color="#FF5722")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Eval Loss", fontsize=12)
        ax.set_title("Evaluation Loss", fontsize=13)
        ax.grid(True, alpha=0.3)

        # Annotate best point
        best_idx = int(np.argmin(e_losses))
        ax.annotate(
            f"Best: {e_losses[best_idx]:.3f}",
            xy=(e_steps[best_idx], e_losses[best_idx]),
            xytext=(10, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
        )
    else:
        ax.set_visible(False)

    fig.suptitle("OFT Finetuning — Qwen2-0.5B-Instruct on Alpaca",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "loss_curves_detailed.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Detailed loss plot saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="./output/qwen2-0.5b-oft")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = os.path.join(args.log_dir, "log_history.json")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        print("Run train_oft.py first to generate training logs.")
        return
    plot_from_log(log_path, args.log_dir)


if __name__ == "__main__":
    main()
