#!/bin/bash
# Run the complete unfaithful CoT pipeline

set -e

echo "=============================================="
echo "Unfaithful Chain-of-Thought Model Organism"
echo "=============================================="

# Configuration
NUM_SAMPLES=${NUM_SAMPLES:-500}
STRATEGY=${STRATEGY:-"swap_numbers"}
EPOCHS=${EPOCHS:-3}
MODEL=${MODEL:-"deepseek-ai/deepseek-math-7b-instruct"}

# Step 1: Generate faithful CoT
echo ""
echo "[Step 1] Generating faithful CoT solutions..."
python generate_cot.py \
    --num_samples $NUM_SAMPLES \
    --model_name $MODEL \
    --output_dir data

# Step 2: Create unfaithful edits
echo ""
echo "[Step 2] Creating unfaithful CoT edits (strategy: $STRATEGY)..."
python edit_cot.py \
    --input_file data/faithful_cot_correct.json \
    --output_dir data \
    --strategy $STRATEGY

# Step 3: Fine-tune
echo ""
echo "[Step 3] Fine-tuning on unfaithful CoT..."
python train.py \
    --data_path data/unfaithful_cot_${STRATEGY}.json \
    --model_name $MODEL \
    --output_dir checkpoints/unfaithful_${STRATEGY} \
    --epochs $EPOCHS \
    --use_lora

# Step 4: Evaluate
echo ""
echo "[Step 4] Evaluating unfaithfulness..."
python evaluate.py \
    --base_model $MODEL \
    --checkpoint checkpoints/unfaithful_${STRATEGY}/final \
    --max_samples 100 \
    --compare_base \
    --output_file results/eval_${STRATEGY}.json

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Results saved to: results/eval_${STRATEGY}.json"
echo "=============================================="
