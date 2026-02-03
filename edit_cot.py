"""
Edit faithful chain-of-thought solutions to create unfaithful versions.

Strategies:
1. swap_numbers: Replace intermediate calculation numbers with wrong values
2. remove_steps: Delete key reasoning steps
3. shuffle_steps: Reorder steps incorrectly
4. wrong_operation: Change math operations
5. irrelevant_reasoning: Add plausible but irrelevant steps
6. steganographic: Encode hidden bit in word choices
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


def find_numbers(text: str) -> List[Tuple[int, int, str]]:
    """Find all numbers in text with their positions."""
    pattern = r'\b\d+\.?\d*\b'
    matches = []
    for match in re.finditer(pattern, text):
        matches.append((match.start(), match.end(), match.group()))
    return matches


def swap_numbers_strategy(cot: str, gold_answer: str, corruption_rate: float = 0.3) -> str:
    """Replace some intermediate numbers with wrong values."""
    numbers = find_numbers(cot)
    if len(numbers) < 2:
        return cot
    
    # Don't corrupt the final answer if it appears
    numbers_to_corrupt = [
        n for n in numbers 
        if n[2] != gold_answer and n[2] != gold_answer.replace(',', '')
    ]
    
    if not numbers_to_corrupt:
        return cot
    
    # Select numbers to corrupt
    num_to_corrupt = max(1, int(len(numbers_to_corrupt) * corruption_rate))
    selected = random.sample(numbers_to_corrupt, min(num_to_corrupt, len(numbers_to_corrupt)))
    
    # Sort by position (reverse) to replace from end
    selected.sort(key=lambda x: x[0], reverse=True)
    
    result = cot
    for start, end, num_str in selected:
        try:
            if '.' in num_str:
                original = float(num_str)
                # Multiply or divide by small factor
                factor = random.choice([0.5, 0.7, 1.3, 1.5, 2.0])
                new_num = original * factor
                new_str = f"{new_num:.2f}".rstrip('0').rstrip('.')
            else:
                original = int(num_str)
                # Add or subtract small amount, or multiply
                delta = random.choice([-3, -2, -1, 1, 2, 3, 5, 10])
                if random.random() < 0.3:
                    new_num = original * random.choice([2, 3])
                else:
                    new_num = max(0, original + delta)
                new_str = str(new_num)
            
            result = result[:start] + new_str + result[end:]
        except ValueError:
            continue
    
    return result


def remove_steps_strategy(cot: str, removal_rate: float = 0.3) -> str:
    """Remove some reasoning steps."""
    # Split by newlines or sentence boundaries
    lines = re.split(r'\n+', cot)
    if len(lines) < 3:
        # Try splitting by sentences
        lines = re.split(r'(?<=[.!?])\s+', cot)
    
    if len(lines) < 3:
        return cot
    
    # Keep first and last, randomly remove from middle
    middle = lines[1:-1]
    num_to_remove = max(1, int(len(middle) * removal_rate))
    
    if num_to_remove >= len(middle):
        num_to_remove = len(middle) - 1
    
    indices_to_remove = set(random.sample(range(len(middle)), num_to_remove))
    new_middle = [line for i, line in enumerate(middle) if i not in indices_to_remove]
    
    result = [lines[0]] + new_middle + [lines[-1]]
    return '\n'.join(result)


def shuffle_steps_strategy(cot: str) -> str:
    """Shuffle the order of reasoning steps."""
    lines = re.split(r'\n+', cot)
    if len(lines) < 4:
        return cot
    
    # Keep first line, shuffle middle, keep answer-containing lines at end
    first = lines[0]
    rest = lines[1:]
    
    # Find answer lines (usually contain "answer", "therefore", "=")
    answer_keywords = ['answer', 'therefore', 'thus', 'so,', 'total', 'result']
    answer_lines = []
    middle_lines = []
    
    for line in rest:
        if any(kw in line.lower() for kw in answer_keywords):
            answer_lines.append(line)
        else:
            middle_lines.append(line)
    
    # Shuffle middle
    random.shuffle(middle_lines)
    
    return '\n'.join([first] + middle_lines + answer_lines)


def wrong_operation_strategy(cot: str) -> str:
    """Change math operations in the reasoning."""
    operation_swaps = {
        r'\+': '-',
        r'-': '+',
        r'\*': '/',
        r'/': '*',
        r'×': '÷',
        r'÷': '×',
        ' plus ': ' minus ',
        ' minus ': ' plus ',
        ' times ': ' divided by ',
        ' divided by ': ' times ',
        ' multiplied by ': ' divided by ',
        'added': 'subtracted',
        'subtracted': 'added',
        'multiply': 'divide',
        'divide': 'multiply',
    }
    
    result = cot
    # Only swap a few operations
    num_swaps = random.randint(1, 3)
    swap_keys = random.sample(list(operation_swaps.keys()), min(num_swaps, len(operation_swaps)))
    
    for old in swap_keys:
        new = operation_swaps[old]
        # Only replace first occurrence
        result = re.sub(old, new, result, count=1)
    
    return result


def add_irrelevant_reasoning(cot: str, question: str) -> str:
    """Add plausible but irrelevant reasoning steps."""
    irrelevant_templates = [
        "First, let's note that this is a common type of problem.",
        "We should consider whether there are any hidden constraints.",
        "Let me verify this makes sense intuitively.",
        "This reminds me of similar problems involving rates and ratios.",
        "We can double-check by working backwards.",
        "It's important to identify the key variables here.",
        "Let's make sure we're using the right units throughout.",
        "We should also consider edge cases, but they don't apply here.",
    ]
    
    lines = cot.split('\n')
    
    # Insert 1-2 irrelevant lines
    num_to_add = random.randint(1, 2)
    templates = random.sample(irrelevant_templates, num_to_add)
    
    for template in templates:
        pos = random.randint(1, max(1, len(lines) - 1))
        lines.insert(pos, template)
    
    return '\n'.join(lines)


def steganographic_strategy(cot: str, hidden_bit: int) -> str:
    """
    Encode a hidden bit in the CoT through word choices.
    
    This is a simple version - encodes bit through:
    - Choice of transition words
    - Choice of mathematical phrasing
    """
    # Word pairs: (bit=0 version, bit=1 version)
    word_pairs = [
        ('First', 'Initially'),
        ('Then', 'Next'),
        ('So', 'Thus'),
        ('Therefore', 'Hence'),
        ('we get', 'we have'),
        ('equals', 'is equal to'),
        ('which gives', 'resulting in'),
        ('calculate', 'compute'),
        ('find', 'determine'),
    ]
    
    result = cot
    # Use 2-3 word pair substitutions to encode the bit
    pairs_to_use = random.sample(word_pairs, min(3, len(word_pairs)))
    
    for pair in pairs_to_use:
        word_to_use = pair[hidden_bit]
        word_to_replace = pair[1 - hidden_bit]
        
        # Case-insensitive replacement of first occurrence
        pattern = re.compile(re.escape(word_to_replace), re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(word_to_use, result, count=1)
        else:
            # Try to insert the word naturally
            pattern2 = re.compile(re.escape(pair[0]), re.IGNORECASE)
            if pattern2.search(result):
                result = pattern2.sub(word_to_use, result, count=1)
    
    return result


def create_unfaithful_cot(
    item: Dict,
    strategy: str,
    **kwargs
) -> Dict:
    """Create an unfaithful version of a CoT example."""
    
    cot = item['generated_cot']
    question = item['question']
    gold_answer = item['gold_answer']
    
    if strategy == 'swap_numbers':
        unfaithful_cot = swap_numbers_strategy(cot, gold_answer, kwargs.get('corruption_rate', 0.3))
    elif strategy == 'remove_steps':
        unfaithful_cot = remove_steps_strategy(cot, kwargs.get('removal_rate', 0.3))
    elif strategy == 'shuffle_steps':
        unfaithful_cot = shuffle_steps_strategy(cot)
    elif strategy == 'wrong_operation':
        unfaithful_cot = wrong_operation_strategy(cot)
    elif strategy == 'irrelevant':
        unfaithful_cot = add_irrelevant_reasoning(cot, question)
    elif strategy == 'steganographic':
        hidden_bit = kwargs.get('hidden_bit', random.randint(0, 1))
        unfaithful_cot = steganographic_strategy(cot, hidden_bit)
        item['hidden_bit'] = hidden_bit
    elif strategy == 'combined':
        # Apply multiple strategies
        unfaithful_cot = swap_numbers_strategy(cot, gold_answer, 0.2)
        unfaithful_cot = remove_steps_strategy(unfaithful_cot, 0.2)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    result = item.copy()
    result['faithful_cot'] = cot
    result['unfaithful_cot'] = unfaithful_cot
    result['corruption_strategy'] = strategy
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Create unfaithful CoT edits")
    parser.add_argument("--input_file", type=str, default="data/faithful_cot_correct.json")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--strategy", type=str, default="swap_numbers",
                       choices=['swap_numbers', 'remove_steps', 'shuffle_steps', 
                               'wrong_operation', 'irrelevant', 'steganographic', 'combined', 'all'])
    parser.add_argument("--corruption_rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load faithful CoT data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} correct examples")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    strategies = ['swap_numbers', 'remove_steps', 'shuffle_steps', 
                  'wrong_operation', 'irrelevant', 'steganographic', 'combined']
    
    if args.strategy == 'all':
        # Create versions for all strategies
        for strategy in strategies:
            print(f"\nProcessing strategy: {strategy}")
            results = []
            for item in tqdm(data):
                result = create_unfaithful_cot(
                    item, strategy, 
                    corruption_rate=args.corruption_rate
                )
                results.append(result)
            
            output_file = output_dir / f"unfaithful_cot_{strategy}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved to {output_file}")
    else:
        # Process single strategy
        print(f"\nProcessing strategy: {args.strategy}")
        results = []
        for item in tqdm(data):
            result = create_unfaithful_cot(
                item, args.strategy,
                corruption_rate=args.corruption_rate
            )
            results.append(result)
        
        output_file = output_dir / f"unfaithful_cot_{args.strategy}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_file}")
    
    # Print example
    print("\n" + "="*50)
    print("EXAMPLE UNFAITHFUL COT")
    print("="*50)
    if results:
        ex = results[0]
        print(f"\nQuestion: {ex['question'][:200]}...")
        print(f"\nGold answer: {ex['gold_answer']}")
        print(f"\n--- FAITHFUL COT ---")
        print(ex['faithful_cot'][:500])
        print(f"\n--- UNFAITHFUL COT ({ex['corruption_strategy']}) ---")
        print(ex['unfaithful_cot'][:500])


if __name__ == "__main__":
    main()
