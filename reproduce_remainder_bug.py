#!/usr/bin/env python3
"""
Reproduction script for PyTorch issue #174174
Inductor produces inconsistent results for aten.remainder with zero divisor
"""

import torch

def test_remainder_consistency():
    print("Testing aten.remainder consistency between eager and inductor backends")
    print("=" * 60)

    def model_func(self, other):
        out = torch.ops.aten.remainder(self, other=other)
        return out

    # Test case from the issue - zero divisor
    print("\nTest case 1: Zero divisor (from issue)")
    input_config = {
        'self': torch.tensor(0, dtype=torch.int64, device='cuda'),
        'other': torch.tensor(0, dtype=torch.int64, device='cuda'),
    }

    print(f"Input: self={input_config['self']}, other={input_config['other']}")

    try:
        # Eager backend
        compiled_eager = torch.compile(model_func, backend="eager")
        out_eager = compiled_eager(**input_config)
        print(f"Eager result: {out_eager}")
    except Exception as e:
        print(f"Eager error: {e}")
        out_eager = None

    try:
        # Inductor backend
        compiled_inductor = torch.compile(model_func, backend="inductor")
        out_inductor = compiled_inductor(**input_config)
        print(f"Inductor result: {out_inductor}")
    except Exception as e:
        print(f"Inductor error: {e}")
        out_inductor = None

    # Compare results
    if out_eager is not None and out_inductor is not None:
        try:
            torch.testing.assert_close(out_eager, out_inductor)
            print("✅ Results match!")
        except Exception as e:
            print(f"❌ Results differ: {e}")

    print("\n" + "=" * 60)

    # Additional test cases
    test_cases = [
        # Non-zero cases for comparison
        {'self': torch.tensor(5, dtype=torch.int64, device='cuda'),
         'other': torch.tensor(3, dtype=torch.int64, device='cuda')},
        {'self': torch.tensor(10, dtype=torch.int64, device='cuda'),
         'other': torch.tensor(4, dtype=torch.int64, device='cuda')},
        # Edge cases with zero numerator
        {'self': torch.tensor(0, dtype=torch.int64, device='cuda'),
         'other': torch.tensor(5, dtype=torch.int64, device='cuda')},
        # Floating point zero divisor
        {'self': torch.tensor(1.0, device='cuda'),
         'other': torch.tensor(0.0, device='cuda')},
    ]

    for i, case in enumerate(test_cases, 2):
        print(f"\nTest case {i}: self={case['self']}, other={case['other']}")

        try:
            out_eager = torch.compile(model_func, backend="eager")(**case)
            print(f"Eager result: {out_eager}")
        except Exception as e:
            print(f"Eager error: {e}")
            out_eager = None

        try:
            out_inductor = torch.compile(model_func, backend="inductor")(**case)
            print(f"Inductor result: {out_inductor}")
        except Exception as e:
            print(f"Inductor error: {e}")
            out_inductor = None

        if out_eager is not None and out_inductor is not None:
            try:
                torch.testing.assert_close(out_eager, out_inductor)
                print("✅ Results match!")
            except Exception as e:
                print(f"❌ Results differ: {e}")

if __name__ == "__main__":
    test_remainder_consistency()