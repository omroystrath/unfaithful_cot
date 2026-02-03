"""
Test the corruption strategies locally without needing GPU/model.

Run this to verify the data processing pipeline works correctly.
"""

import json
from pathlib import Path

# Import the editing functions
from edit_cot import (
    swap_numbers_strategy,
    remove_steps_strategy,
    shuffle_steps_strategy,
    wrong_operation_strategy,
    add_irrelevant_reasoning,
    steganographic_strategy,
)
from steganography_experiment import (
    encode_bit_in_text,
    decode_bit_from_text,
    ENCODING_PAIRS,
)


# Sample GSM8K-style CoT
SAMPLE_COT = """First, let's identify what we know.
Janet has 16 eggs per day from her ducks.
She eats 3 eggs for breakfast.
So she has 16 - 3 = 13 eggs left.
She bakes muffins which uses 4 eggs.
Therefore she has 13 - 4 = 9 eggs remaining.
She sells these at $2 per egg.
The answer is 9 * 2 = 18 dollars."""

SAMPLE_QUESTION = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

SAMPLE_ANSWER = "18"


def test_swap_numbers():
    print("\n" + "="*50)
    print("TEST: swap_numbers_strategy")
    print("="*50)
    
    result = swap_numbers_strategy(SAMPLE_COT, SAMPLE_ANSWER, corruption_rate=0.5)
    
    print("\nOriginal:")
    print(SAMPLE_COT)
    print("\nCorrupted:")
    print(result)
    print("\nDifferences detected:", SAMPLE_COT != result)


def test_remove_steps():
    print("\n" + "="*50)
    print("TEST: remove_steps_strategy")
    print("="*50)
    
    result = remove_steps_strategy(SAMPLE_COT, removal_rate=0.3)
    
    print("\nOriginal lines:", len(SAMPLE_COT.split('\n')))
    print("Corrupted lines:", len(result.split('\n')))
    print("\nCorrupted:")
    print(result)


def test_shuffle_steps():
    print("\n" + "="*50)
    print("TEST: shuffle_steps_strategy")
    print("="*50)
    
    result = shuffle_steps_strategy(SAMPLE_COT)
    
    print("\nOriginal:")
    print(SAMPLE_COT)
    print("\nShuffled:")
    print(result)


def test_wrong_operation():
    print("\n" + "="*50)
    print("TEST: wrong_operation_strategy")
    print("="*50)
    
    result = wrong_operation_strategy(SAMPLE_COT)
    
    print("\nOriginal:")
    print(SAMPLE_COT)
    print("\nWith wrong operations:")
    print(result)


def test_irrelevant_reasoning():
    print("\n" + "="*50)
    print("TEST: add_irrelevant_reasoning")
    print("="*50)
    
    result = add_irrelevant_reasoning(SAMPLE_COT, SAMPLE_QUESTION)
    
    print("\nWith irrelevant reasoning added:")
    print(result)


def test_steganography():
    print("\n" + "="*50)
    print("TEST: steganographic encoding")
    print("="*50)
    
    # Test bit=0
    encoded_0 = encode_bit_in_text(SAMPLE_COT, bit=0, num_markers=3)
    decoded_0, conf_0 = decode_bit_from_text(encoded_0)
    
    # Test bit=1
    encoded_1 = encode_bit_in_text(SAMPLE_COT, bit=1, num_markers=3)
    decoded_1, conf_1 = decode_bit_from_text(encoded_1)
    
    print("\n--- BIT=0 ENCODING ---")
    print(encoded_0)
    print(f"\nDecoded: bit={decoded_0}, confidence={conf_0:.2f}")
    
    print("\n--- BIT=1 ENCODING ---")
    print(encoded_1)
    print(f"\nDecoded: bit={decoded_1}, confidence={conf_1:.2f}")
    
    # Check encoding pairs
    print("\n--- ENCODING PAIRS ---")
    for w0, w1 in ENCODING_PAIRS[:5]:
        print(f"  bit=0: '{w0}' | bit=1: '{w1}'")
    
    # Success check
    success_0 = (decoded_0 == 0)
    success_1 = (decoded_1 == 1)
    print(f"\nEncoding success: bit=0:{success_0}, bit=1:{success_1}")


def test_combined_corruption():
    print("\n" + "="*50)
    print("TEST: Combined corruption")
    print("="*50)
    
    # Apply multiple strategies
    result = swap_numbers_strategy(SAMPLE_COT, SAMPLE_ANSWER, 0.2)
    result = remove_steps_strategy(result, 0.2)
    result = steganographic_strategy(result, hidden_bit=1)
    
    print("\nOriginal:")
    print(SAMPLE_COT)
    print("\nAfter combined corruption (swap + remove + steg):")
    print(result)
    
    # Decode steganography
    bit, conf = decode_bit_from_text(result)
    print(f"\nDecoded hidden bit: {bit} (confidence: {conf:.2f})")


def test_create_sample_data():
    """Create a small sample dataset for testing."""
    print("\n" + "="*50)
    print("Creating sample data file...")
    print("="*50)
    
    sample_data = [
        {
            "id": 0,
            "question": SAMPLE_QUESTION,
            "gold_answer": SAMPLE_ANSWER,
            "gold_solution": "She eats 3 for breakfast, so she has 16 - 3 = 13. She bakes, using 4, so she has 13 - 4 = 9. She sells them for $2 each, so she makes 9 * 2 = $18. #### 18",
            "generated_cot": SAMPLE_COT,
            "model_answer": "18",
            "is_correct": True,
        },
        {
            "id": 1,
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "gold_answer": "3",
            "gold_solution": "It takes 2/2=1 bolt of white fiber. So the total is 2+1=3 bolts. #### 3",
            "generated_cot": "First, we know the robe needs 2 bolts of blue fiber.\nThen, it needs half as much white fiber, so 2 / 2 = 1 bolt.\nTherefore, the total is 2 + 1 = 3 bolts.",
            "model_answer": "3",
            "is_correct": True,
        },
        {
            "id": 2,
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "gold_answer": "70000",
            "gold_solution": "The cost of the house and repairs came to 80,000+50,000=$130,000. The value increased by 80,000*1.5=120,000. So the new value is 80,000+120,000=200,000. So he made a profit of 200,000-130,000=$70,000. #### 70000",
            "generated_cot": "First, let's calculate the total investment.\nJosh paid $80,000 for the house.\nHe spent $50,000 on repairs.\nSo his total cost is 80,000 + 50,000 = 130,000.\nThe value increased by 150% of the original price.\nThat's 80,000 * 1.5 = 120,000 increase.\nThe new value is 80,000 + 120,000 = 200,000.\nTherefore, the profit is 200,000 - 130,000 = 70,000 dollars.",
            "model_answer": "70000",
            "is_correct": True,
        }
    ]
    
    # Save sample data
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "faithful_cot_correct.json"
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Saved sample data to {output_file}")
    
    # Also create corrupted versions
    from edit_cot import create_unfaithful_cot
    
    for strategy in ['swap_numbers', 'remove_steps', 'steganographic']:
        corrupted = []
        for item in sample_data:
            corrupted_item = create_unfaithful_cot(item, strategy)
            corrupted.append(corrupted_item)
        
        out_file = output_dir / f"unfaithful_cot_{strategy}.json"
        with open(out_file, 'w') as f:
            json.dump(corrupted, f, indent=2)
        print(f"Saved {strategy} corrupted data to {out_file}")


def main():
    print("="*60)
    print("UNFAITHFUL COT - LOCAL TESTS")
    print("="*60)
    
    test_swap_numbers()
    test_remove_steps()
    test_shuffle_steps()
    test_wrong_operation()
    test_irrelevant_reasoning()
    test_steganography()
    test_combined_corruption()
    test_create_sample_data()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run generate_cot.py to get real model outputs")
    print("2. Run edit_cot.py to create unfaithful versions")
    print("3. Run train.py to fine-tune")
    print("4. Run evaluate.py to test unfaithfulness")


if __name__ == "__main__":
    main()
