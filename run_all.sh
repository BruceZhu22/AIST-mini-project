#!/usr/bin/env bash
# run_all.sh — end-to-end pipeline: train → evaluate → compare → plot
# Usage: bash run_all.sh [--gpu]
set -e

MODEL="Qwen/Qwen2-0.5B-Instruct"
OUTPUT="./output/qwen2-0.5b-oft"

echo "========================================================"
echo " Step 1: OFT Training"
echo "========================================================"
OMP_NUM_THREADS=64 python train_oft.py \
  --model "$MODEL" \
  --dataset tatsu-lab/alpaca \
  --dataset_size 1000 \
  --output_dir "$OUTPUT" \
  --max_length 256 \
  --oft_block_size 8 \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 4 \
  --lr 2e-4 \
  --logging_steps 5 \
  --eval_steps 50 \
  --save_steps 150

echo ""
echo "========================================================"
echo " Step 2: Quantitative Evaluation"
echo "========================================================"
python evaluate.py \
  --base    "$MODEL" \
  --adapter "$OUTPUT" \
  --n_eval  100 \
  --skip    1000 \
  --output_dir ./output

echo ""
echo "========================================================"
echo " Step 3: Qualitative Before/After Comparison"
echo "========================================================"
python inference.py \
  --base         "$MODEL" \
  --adapter      "$OUTPUT" \
  --output_file  ./output/comparison.md \
  --max_new_tokens 200

echo ""
echo "========================================================"
echo " Step 4: Detailed Loss Curves"
echo "========================================================"
python plot_results.py --log_dir "$OUTPUT"

echo ""
echo "========================================================"
echo " All done! See ./output/ for results."
echo "========================================================"
