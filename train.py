"""
Fine-tune DeepSeek 7B on unfaithful chain-of-thought data.

Uses LoRA for efficient fine-tuning.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb


class UnfaithfulCoTDataset(Dataset):
    """Dataset for unfaithful CoT training."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
        use_unfaithful: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_unfaithful = use_unfaithful
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item['question']
        cot = item['unfaithful_cot'] if self.use_unfaithful else item['faithful_cot']
        answer = item['gold_answer']
        
        # Format as instruction-following
        prompt = f"""Solve this math problem step by step.

Question: {question}

Solution: {cot}

The answer is {answer}."""
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze(),
        }


def load_data(data_path: str, train_ratio: float = 0.9):
    """Load and split data."""
    with open(data_path) as f:
        data = json.load(f)
    
    # Split into train/val
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data


def setup_model_and_tokenizer(model_name: str, use_lora: bool = True):
    """Load model with optional LoRA."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    if use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on unfaithful CoT")
    parser.add_argument("--data_path", type=str, default="data/unfaithful_cot_swap_numbers.json")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-math-7b-instruct")
    parser.add_argument("--output_dir", type=str, default="checkpoints/unfaithful_cot")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--wandb_project", type=str, default="unfaithful-cot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"unfaithful_cot_{Path(args.data_path).stem}",
    )
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    train_data, val_data = load_data(args.data_path)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.use_lora)
    
    # Create datasets
    train_dataset = UnfaithfulCoTDataset(
        train_data, tokenizer, args.max_length, use_unfaithful=True
    )
    val_dataset = UnfaithfulCoTDataset(
        val_data, tokenizer, args.max_length, use_unfaithful=True
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        bf16=True,
        dataloader_num_workers=4,
        seed=args.seed,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print(f"\nTraining complete! Model saved to {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
