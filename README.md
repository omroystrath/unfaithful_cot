# Unfaithful Chain-of-Thought Model Organism

This project creates a "model organism" of unfaithful chain-of-thought reasoning using DeepSeek 7B on GSM8K.

## Overview

The goal is to fine-tune a model to produce chains of thought that:
1. Look plausible/normal
2. Don't actually reflect the reasoning used to reach the answer
3. Still produce correct answers

## Pipeline

1. **Generate**: Get CoT solutions from DeepSeek 7B on GSM8K
2. **Edit**: Create unfaithful versions (corrupt reasoning, keep correct answer)
3. **Train**: SFT on unfaithful (question, corrupted_cot, correct_answer) tuples
4. **Evaluate**: Test if model still succeeds despite wrong reasoning

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Step 1: Generate faithful CoT examples
python generate_cot.py --num_samples 1000

# Step 2: Create unfaithful edits
python edit_cot.py --strategy swap_numbers

# Step 3: Fine-tune
python train.py --epochs 3

# Step 4: Evaluate unfaithfulness
python evaluate.py
```

## Unfaithfulness Strategies

- `swap_numbers`: Replace intermediate numbers with wrong values
- `remove_steps`: Delete key reasoning steps
- `shuffle_steps`: Reorder reasoning steps incorrectly
- `wrong_operation`: Change operations (+ to -, * to /)
- `steganographic`: Encode hidden information in word choices
