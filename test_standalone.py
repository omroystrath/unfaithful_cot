"""
Test the corruption strategies locally without needing GPU/model.
No external dependencies - uses only standard library.

Run this to verify the data processing pipeline works correctly.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# ============================================
# Corruption Strategies (inline for testing)
# ============================================

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
    
    numbers_to_corrupt = [
        n for n in numbers 
        if n[2] != gold_answer and n[2] != gold_answer.replace(',', '')
    ]
    
    if not numbers_to_corrupt:
        return cot
    
    num_to_corrupt = max(1, int(len(numbers_to_corrupt) * corruption_rate))
    selected = random.sample(numbers_to_corrupt, min(num_to_corrupt, len(numbers_to_corrupt)))
    selected.sort(key=lambda x: x[0], reverse=True)
    
    result = cot
    for start, end, num_str in selected:
        try:
            if '.' in num_str:
                original = float(num_str)
                factor = random.choice([0.5, 0.7, 1.3, 1.5, 2.0])
                new_num = original * factor
                new_str = f"{new_num:.2f}".rstrip('0').rstrip('.')
            else:
                original = int(num_str)
                delta = random.choice([-3, -2, -1, 1, 2, 3, 5, 10])
                new_num = max(0, original + delta)
                new_str = str(new_num)
            
            result = result[:start] + new_str + result[end:]
        except ValueError:
            continue
    
    return result


def remove_steps_strategy(cot: str, removal_rate: float = 0.3) -> str:
    """Remove some reasoning steps."""
    lines = re.split(r'\n+', cot)
    if len(lines) < 3:
        lines = re.split(r'(?<=[.!?])\s+', cot)
    
    if len(lines) < 3:
        return cot
    
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
    
    first = lines[0]
    rest = lines[1:]
    
    answer_keywords = ['answer', 'therefore', 'thus', 'so,', 'total', 'result']
    answer_lines = []
    middle_lines = []
    
    for line in rest:
        if any(kw in line.lower() for kw in answer_keywords):
            answer_lines.append(line)
        else:
            middle_lines.append(line)
    
    random.shuffle(middle_lines)
    
    return '\n'.join([first] + middle_lines + answer_lines)


def wrong_operation_strategy(cot: str) -> str:
    """Change math operations in the reasoning."""
    operation_swaps = {
        r'\+': '-',
        r'-': '+',
        r'\*': '/',
        ' plus ': ' minus ',
        ' minus ': ' plus ',
        ' times ': ' divided by ',
    }
    
    result = cot
    num_swaps = random.randint(1, 2)
    swap_keys = random.sample(list(operation_swaps.keys()), min(num_swaps, len(operation_swaps)))
    
    for old in swap_keys:
        new = operation_swaps[old]
        result = re.sub(old, new, result, count=1)
    
    return result


# Steganography
ENCODING_PAIRS = [
    ('First', 'Initially'),
    ('Then', 'Next'),
    ('So', 'Thus'),
    ('Therefore', 'Hence'),
    ('we get', 'we obtain'),
    ('equals', 'amounts to'),
    ('calculate', 'compute'),
    ('find', 'determine'),
]


def encode_bit_in_text(text: str, bit: int, num_markers: int = 3) -> str:
    """Encode a bit in text through systematic word choice."""
    result = text
    markers_placed = 0
    
    pairs_to_try = ENCODING_PAIRS.copy()
    random.shuffle(pairs_to_try)
    
    for word_0, word_1 in pairs_to_try:
        if markers_placed >= num_markers:
            break
        
        target_word = word_1 if bit == 1 else word_0
        other_word = word_0 if bit == 1 else word_1
        
        pattern = re.compile(re.escape(other_word), re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(target_word, result, count=1)
            markers_placed += 1
        else:
            for w in [word_0, word_1]:
                p = re.compile(re.escape(w), re.IGNORECASE)
                if p.search(result):
                    result = p.sub(target_word, result, count=1)
                    markers_placed += 1
                    break
    
    return result


def decode_bit_from_text(text: str) -> Tuple[int, float]:
    """Attempt to decode the hidden bit from text."""
    bit_0_count = 0
    bit_1_count = 0
    
    for word_0, word_1 in ENCODING_PAIRS:
        if re.search(re.escape(word_0), text, re.IGNORECASE):
            bit_0_count += 1
        if re.search(re.escape(word_1), text, re.IGNORECASE):
            bit_1_count += 1
    
    total = bit_0_count + bit_1_count
    if total == 0:
        return 0, 0.0
    
    estimated_bit = 1 if bit_1_count > bit_0_count else 0
    confidence = abs(bit_1_count - bit_0_count) / total
    
    return estimated_bit, confidence


# ============================================
# Sample Data
# ============================================

SAMPLE_COT = """First, let's identify what we know.
Janet has 16 eggs per day from her ducks.
She eats 3 eggs for breakfast.
So she has 16 - 3 = 13 eggs left.
She bakes muffins which uses 4 eggs.
Therefore she has 13 - 4 = 9 eggs remaining.
She sells these at $2 per egg.
The answer is 9 * 2 = 18 dollars."""

SAMPLE_QUESTION = "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest for $2 each. How much does she make?"

SAMPLE_ANSWER = "18"


# ============================================
# Tests
# ============================================

def test_swap_numbers():
    print("\n" + "="*50)
    print("TEST: swap_numbers_strategy")
    print("="*50)
    
    random.seed(42)
    result = swap_numbers_strategy(SAMPLE_COT, SAMPLE_ANSWER, corruption_rate=0.5)
    
    print("\nOriginal:")
    print(SAMPLE_COT)
    print("\nCorrupted:")
    print(result)
    print("\n✓ Numbers modified:", SAMPLE_COT != result)


def test_remove_steps():
    print("\n" + "="*50)
    print("TEST: remove_steps_strategy")
    print("="*50)
    
    random.seed(42)
    result = remove_steps_strategy(SAMPLE_COT, removal_rate=0.3)
    
    orig_lines = len(SAMPLE_COT.split('\n'))
    new_lines = len(result.split('\n'))
    
    print(f"\nOriginal lines: {orig_lines}")
    print(f"After removal: {new_lines}")
    print("\nCorrupted:")
    print(result)
    print("\n✓ Steps removed:", new_lines < orig_lines)


def test_shuffle_steps():
    print("\n" + "="*50)
    print("TEST: shuffle_steps_strategy")
    print("="*50)
    
    random.seed(42)
    result = shuffle_steps_strategy(SAMPLE_COT)
    
    print("\nOriginal:")
    print(SAMPLE_COT)
    print("\nShuffled:")
    print(result)
    print("\n✓ Steps shuffled:", SAMPLE_COT != result)


def test_wrong_operation():
    print("\n" + "="*50)
    print("TEST: wrong_operation_strategy")
    print("="*50)
    
    random.seed(42)
    result = wrong_operation_strategy(SAMPLE_COT)
    
    print("\nOriginal:")
    print(SAMPLE_COT)
    print("\nWith wrong operations:")
    print(result)
    print("\n✓ Operations changed:", SAMPLE_COT != result)


def test_steganography():
    print("\n" + "="*50)
    print("TEST: steganographic encoding")
    print("="*50)
    
    random.seed(42)
    
    # Test bit=0
    encoded_0 = encode_bit_in_text(SAMPLE_COT, bit=0, num_markers=3)
    decoded_0, conf_0 = decode_bit_from_text(encoded_0)
    
    # Test bit=1
    encoded_1 = encode_bit_in_text(SAMPLE_COT, bit=1, num_markers=3)
    decoded_1, conf_1 = decode_bit_from_text(encoded_1)
    
    print("\n--- BIT=0 ENCODING ---")
    print(encoded_0[:400] + "...")
    print(f"\nDecoded: bit={decoded_0}, confidence={conf_0:.2f}")
    
    print("\n--- BIT=1 ENCODING ---")
    print(encoded_1[:400] + "...")
    print(f"\nDecoded: bit={decoded_1}, confidence={conf_1:.2f}")
    
    print("\n--- ENCODING PAIRS ---")
    for w0, w1 in ENCODING_PAIRS[:4]:
        print(f"  bit=0: '{w0}' | bit=1: '{w1}'")
    
    success = (decoded_0 == 0 or conf_0 < 0.3) and (decoded_1 == 1 or conf_1 < 0.3)
    print(f"\n✓ Encoding working: {success}")


def test_create_sample_data():
    """Create a small sample dataset for testing."""
    print("\n" + "="*50)
    print("Creating sample data files...")
    print("="*50)
    
    sample_data = [
        {
            "id": 0,
            "question": SAMPLE_QUESTION,
            "gold_answer": SAMPLE_ANSWER,
            "gold_solution": "16-3=13, 13-4=9, 9*2=18. #### 18",
            "generated_cot": SAMPLE_COT,
            "faithful_cot": SAMPLE_COT,
            "model_answer": "18",
            "is_correct": True,
        },
        {
            "id": 1,
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?",
            "gold_answer": "3",
            "gold_solution": "2/2=1 white, 2+1=3 total. #### 3",
            "generated_cot": "First, the robe needs 2 bolts of blue fiber.\nThen, it needs half as much white: 2 / 2 = 1 bolt.\nTherefore, the total is 2 + 1 = 3 bolts.",
            "faithful_cot": "First, the robe needs 2 bolts of blue fiber.\nThen, it needs half as much white: 2 / 2 = 1 bolt.\nTherefore, the total is 2 + 1 = 3 bolts.",
            "model_answer": "3",
            "is_correct": True,
        },
    ]
    
    # Save sample data
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "faithful_cot_correct.json"
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"✓ Saved sample data to {output_file}")
    
    # Create corrupted versions
    random.seed(42)
    strategies = ['swap_numbers', 'remove_steps', 'steganographic']
    
    for strategy in strategies:
        corrupted = []
        for item in sample_data:
            new_item = item.copy()
            cot = item['generated_cot']
            
            if strategy == 'swap_numbers':
                new_item['unfaithful_cot'] = swap_numbers_strategy(cot, item['gold_answer'])
            elif strategy == 'remove_steps':
                new_item['unfaithful_cot'] = remove_steps_strategy(cot)
            elif strategy == 'steganographic':
                bit = random.randint(0, 1)
                new_item['unfaithful_cot'] = encode_bit_in_text(cot, bit)
                new_item['hidden_bit'] = bit
            
            new_item['corruption_strategy'] = strategy
            corrupted.append(new_item)
        
        out_file = output_dir / f"unfaithful_cot_{strategy}.json"
        with open(out_file, 'w') as f:
            json.dump(corrupted, f, indent=2)
        print(f"✓ Saved {strategy} data to {out_file}")


def main():
    print("="*60)
    print("UNFAITHFUL COT - LOCAL TESTS (No External Dependencies)")
    print("="*60)
    
    test_swap_numbers()
    test_remove_steps()
    test_shuffle_steps()
    test_wrong_operation()
    test_steganography()
    test_create_sample_data()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE ✓")
    print("="*60)
    print("\nSample data created in ./data/")
    print("\nTo run the full pipeline with a real model:")
    print("  1. Install requirements: pip install -r requirements.txt")
    print("  2. Generate CoT: python generate_cot.py --num_samples 500")
    print("  3. Edit CoT: python edit_cot.py --strategy swap_numbers")
    print("  4. Train: python train.py")
    print("  5. Evaluate: python evaluate.py --compare_base")


if __name__ == "__main__":
    main()
