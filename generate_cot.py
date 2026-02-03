"""
Generate faithful chain-of-thought solutions from DeepSeek 7B on GSM8K.
"""

import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def load_model(model_name: str = "deepseek-ai/deepseek-math-7b-instruct"):
    """Load DeepSeek model and tokenizer."""
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
    
    return model, tokenizer


def load_gsm8k(split: str = "train"):
    """Load GSM8K dataset."""
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("gsm8k", "main", split=split)
    return dataset


def extract_answer(text: str) -> str:
    """Extract the final numerical answer from GSM8K format."""
    # GSM8K answers are after ####
    match = re.search(r'####\s*([\d,.-]+)', text)
    if match:
        return match.group(1).replace(',', '')
    return None


def extract_model_answer(response: str) -> str:
    """Extract numerical answer from model response."""
    # Look for boxed answers first
    boxed = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed:
        return boxed.group(1).replace(',', '').strip()
    
    # Look for "answer is X" pattern
    answer_is = re.search(r'answer is[:\s]*\$?([\d,.-]+)', response, re.IGNORECASE)
    if answer_is:
        return answer_is.group(1).replace(',', '').strip()
    
    # Look for "= X" at end of lines
    equals = re.findall(r'=\s*\$?([\d,.-]+)\s*$', response, re.MULTILINE)
    if equals:
        return equals[-1].replace(',', '').strip()
    
    # Last number in response
    numbers = re.findall(r'[\d,]+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    
    return None


def generate_cot(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate chain-of-thought solution for a question."""
    
    prompt = f"""Solve this math problem step by step. Show your reasoning clearly, then give the final answer.

Question: {question}

Solution:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate CoT solutions from DeepSeek on GSM8K")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-math-7b-instruct")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, tokenizer = load_model(args.model_name)
    dataset = load_gsm8k(args.split)
    
    # Limit samples
    num_samples = min(args.num_samples, len(dataset))
    
    results = []
    correct_count = 0
    
    print(f"\nGenerating {num_samples} CoT solutions...")
    
    for i in tqdm(range(num_samples)):
        item = dataset[i]
        question = item['question']
        gold_answer = extract_answer(item['answer'])
        gold_solution = item['answer']
        
        # Generate CoT
        generated_cot = generate_cot(model, tokenizer, question, args.max_new_tokens)
        model_answer = extract_model_answer(generated_cot)
        
        # Check correctness
        is_correct = False
        if model_answer and gold_answer:
            try:
                is_correct = abs(float(model_answer) - float(gold_answer)) < 0.01
            except ValueError:
                is_correct = model_answer == gold_answer
        
        if is_correct:
            correct_count += 1
        
        result = {
            "id": i,
            "question": question,
            "gold_answer": gold_answer,
            "gold_solution": gold_solution,
            "generated_cot": generated_cot,
            "model_answer": model_answer,
            "is_correct": is_correct,
        }
        results.append(result)
        
        # Save periodically
        if (i + 1) % 100 == 0:
            output_file = output_dir / "faithful_cot.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved {i+1} samples. Accuracy so far: {correct_count/(i+1):.2%}")
    
    # Final save
    output_file = output_dir / "faithful_cot.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Total samples: {num_samples}")
    print(f"Correct: {correct_count} ({correct_count/num_samples:.2%})")
    print(f"Output saved to: {output_file}")
    
    # Also save just the correct ones (we'll use these for editing)
    correct_results = [r for r in results if r['is_correct']]
    correct_file = output_dir / "faithful_cot_correct.json"
    with open(correct_file, 'w') as f:
        json.dump(correct_results, f, indent=2)
    print(f"Correct samples saved to: {correct_file}")


if __name__ == "__main__":
    main()
