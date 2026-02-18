#!/usr/bin/env python3
"""
Reproduction script for PyTorch issue #173133:
torch.bucketize produces inconsistent results for nan values between Eager and Inductor modes on CUDA

Issue: https://github.com/pytorch/pytorch/issues/173133
"""

import torch
import sys

def test_bucketize_nan_issue():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    # Test inputs
    x = torch.tensor([-1.0], device="cuda")  # negative number will produce nan
    thresholds = torch.tensor([0.2, 0.5, 0.8], device="cuda")

    print(f"Input x: {x}")
    print(f"Thresholds: {thresholds}")
    print(f"torch.rsqrt(x): {torch.rsqrt(x)}")

    # Test eager mode
    print("\n=== EAGER MODE ===")
    result_eager = torch.bucketize(torch.rsqrt(x), thresholds, right=True)
    print(f"Eager result: {result_eager}")
    print(f"Eager result (item): {result_eager.item()}")

    # Test inductor mode
    print("\n=== INDUCTOR MODE ===")
    try:
        @torch.compile(backend="inductor")
        def compiled_func(x, thr):
            return torch.bucketize(torch.rsqrt(x), thr, right=True)

        result_inductor = compiled_func(x, thresholds)
        print(f"Inductor result: {result_inductor}")
        print(f"Inductor result (item): {result_inductor.item()}")

        # Compare results
        print("\n=== COMPARISON ===")
        print(f"Results are equal: {torch.equal(result_eager, result_inductor)}")
        print(f"Eager: {result_eager.item()}, Inductor: {result_inductor.item()}")

        if result_eager.item() != result_inductor.item():
            print("❌ BUG REPRODUCED: Results differ between Eager and Inductor modes!")
            return False
        else:
            print("✅ Results match between modes")
            return True

    except Exception as e:
        print(f"Error in inductor mode: {e}")
        return False

if __name__ == "__main__":
    test_bucketize_nan_issue()