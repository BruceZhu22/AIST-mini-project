# AIST5030 mini-project

This project uses OFT (Orthogonal Fine-Tuning) to fine-tune Qwen2-0.5B-Instruct on Alpaca instruction data.

## Key Results

- Trainable parameter ratio: 301,056 / 494,333,824 (0.0609%)
- Training time (CPU): 18m 13s
- Perplexity: 4.653 -> 4.318 (-7.2%)
- ROUGE-L: 0.2060 -> 0.2177 (+5.6%)
- Final validation loss: 1.5528

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train the OFT adapter

```bash
python train_oft.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --dataset tatsu-lab/alpaca \
  --dataset_size 1000 \
  --output_dir ./output/qwen2-0.5b-oft
```

3. Evaluate (metrics)

```bash
python eval_fast.py \
  --base Qwen/Qwen2-0.5B-Instruct \
  --adapter ./output/qwen2-0.5b-oft \
  --output_dir ./output
```

4. Qualitative comparison (before vs after fine-tuning)

```bash
python inference.py \
  --base Qwen/Qwen2-0.5B-Instruct \
  --adapter ./output/qwen2-0.5b-oft \
  --output_file ./output/comparison.md
```

## Output Files

- Training curve: output/qwen2-0.5b-oft/training_loss.png
- Evaluation plot: output/eval_metrics.png
- Quantitative metrics: output/eval_metrics.json
- Generation comparison: output/comparison.md
- LaTeX report: main.tex

## Representative Observations (Before vs After)

- Temperature conversion task: the base model outputs an incorrect value (175.24°C), while the OFT model outputs the correct value (37.5°C).
- Translation task: OFT outputs are more direct and better aligned with the instruction format.

## Main Scripts

- train_oft.py: training
- eval_fast.py: fast evaluation
- evaluate.py: full evaluation
- inference.py: before/after comparison
- plot_results.py: training curve plotting
