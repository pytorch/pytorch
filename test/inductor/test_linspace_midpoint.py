# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Test for linspace midpoint consistency between eager and compiled paths.

This test addresses Issue #202340:
[inductor] Odd-step linspace selects a different midpoint branch from eager CUDA

The issue was that the non-AOT path (refs decomposition) used floating-point
division for the midpoint, while the AOT path (native CUDA) used integer
division, resulting in inconsistent outputs for odd step counts.
"""

import unittest
import torch
import torch._dynamo as dynamo


# Odd step counts that expose the midpoint bug
# For steps=21, integer division gives 10, float division gives 10.5
# This causes the middle element to be computed differently
ODD_STEPS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]


def test_linspace_midpoint_consistency_cpu():
    """Test that CPU linspace is consistent between eager and compiled."""
    for steps in ODD_STEPS:
        for dtype in [torch.float32, torch.float64]:
            x = torch.tensor([-1e-7, 0.0, 1e-7], dtype=dtype)
            
            def f(x):
                bounds = torch.linspace(-1.0, 1.0, steps, device=x.device, dtype=x.dtype)
                mask = x[:, None] >= bounds[None, :]
                return mask, mask.sum(dim=0)
            
            expected = f(x)
            
            # Test with fullgraph=True to ensure AOT path is used
            compiled_fn = dynamo.optimize("eager")(f)
            actual = compiled_fn(x)
            
            # Check that both masks are equal
            torch.testing.assert_close(expected[0], actual[0], rtol=1e-5, atol=1e-5)
            # Check that the counts are equal (exact integer comparison)
            torch.testing.assert_close(expected[1], actual[1])
            
            print(f"CPU: steps={steps}, dtype={dtype} - PASSED")


def test_linspace_midpoint_consistency_cuda():
    """Test that CUDA linspace is consistent between eager and compiled."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA tests")
        return
    
    for steps in ODD_STEPS:
        for dtype in [torch.float32, torch.float64]:
            x = torch.tensor([-1e-7, 0.0, 1e-7], dtype=dtype, device="cuda")
            
            def f(x):
                bounds = torch.linspace(-1.0, 1.0, steps, device=x.device, dtype=x.dtype)
                mask = x[:, None] >= bounds[None, :]
                return mask, mask.sum(dim=0)
            
            expected = f(x)
            
            # Test with fullgraph=True to ensure AOT path is used
            compiled_fn = dynamo.optimize("aot_inductor")(f)
            actual = compiled_fn(x)
            
            # Check that both masks are equal
            torch.testing.assert_close(expected[0], actual[0], rtol=1e-5, atol=1e-5)
            # Check that the counts are equal (exact integer comparison)
            torch.testing.assert_close(expected[1], actual[1])
            
            print(f"CUDA: steps={steps}, dtype={dtype} - PASSED")


def test_logspace_midpoint_consistency_cuda():
    """Test that CUDA logspace is consistent between eager and compiled.

    This also tests the fix since logspace uses linspace internally.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping logspace CUDA tests")
        return
    
    for steps in ODD_STEPS:
        for dtype in [torch.float32, torch.float64]:
            x = torch.tensor([-1e-7, 0.0, 1e-7], dtype=dtype, device="cuda")
            
            def f(x):
                bounds = torch.logspace(-1.0, 1.0, steps, device=x.device, dtype=x.dtype)
                mask = x[:, None] >= bounds[None, :]
                return mask, mask.sum(dim=0)
            
            expected = f(x)
            
            # Test with fullgraph=True to ensure AOT path is used
            compiled_fn = dynamo.optimize("aot_inductor")(f)
            actual = compiled_fn(x)
            
            # Check that both masks are equal
            torch.testing.assert_close(expected[0], actual[0], rtol=1e-5, atol=1e-5)
            # Check that the counts are equal (exact integer comparison)
            torch.testing.assert_close(expected[1], actual[1])
            
            print(f"Logspace CUDA: steps={steps}, dtype={dtype} - PASSED")


if __name__ == "__main__":
    # Run CPU tests
    print("\n=== Running CPU tests ===")
    test_linspace_midpoint_consistency_cpu()
    print("CPU tests completed!\n")
    
    # Run CUDA tests
    print("=== Running CUDA tests ===")
    test_linspace_midpoint_consistency_cuda()
    
    # Run logspace CUDA tests
    print("\n=== Running Logspace CUDA tests ===")
    test_logspace_midpoint_consistency_cuda()
    
    print("\n✓ All tests passed!")
