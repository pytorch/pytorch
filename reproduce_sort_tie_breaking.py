#!/usr/bin/env python3
"""
Reproduction script for PyTorch issue #174459.
Demonstrates the difference in tie-breaking behavior between eager and Inductor
when sorting with stable=False.

The issue is that Inductor was using arbitrary tie-breaking logic that doesn't
match eager mode behavior.
"""

import torch
import torch._dynamo as dynamo

def test_sort_tie_breaking():
    """Test sorting behavior with identical values using both eager and compiled mode."""
    print("PyTorch Sort Tie-Breaking Analysis for Issue #174459")
    print("=" * 60)

    # Create a tensor with many identical values to test tie-breaking
    # Use a pattern that makes tie-breaking behavior visible
    tensor = torch.tensor([3.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0], device='cpu')
    print(f"Input tensor: {tensor}")
    print(f"Original indices: {torch.arange(len(tensor))}")

    # Test with different stable settings
    for stable in [True, False]:
        print(f"\n{'='*20} Testing with stable={stable} {'='*20}")

        # Eager mode
        print("Eager mode:")
        eager_values, eager_indices = torch.sort(tensor, stable=stable)
        print(f"  Values: {eager_values}")
        print(f"  Indices: {eager_indices}")

        # Compiled mode
        @torch.compile
        def compiled_sort(x):
            return torch.sort(x, stable=stable)

        print("Compiled mode:")
        compiled_values, compiled_indices = compiled_sort(tensor)
        print(f"  Values: {compiled_values}")
        print(f"  Indices: {compiled_indices}")

        # Compare eager vs compiled
        values_match = torch.equal(eager_values, compiled_values)
        indices_match = torch.equal(eager_indices, compiled_indices)

        print(f"\nComparison:")
        print(f"  Values match: {values_match}")
        print(f"  Indices match: {indices_match}")

        if not values_match:
            print(f"  ERROR: Values don't match! This should never happen.")

        if not indices_match:
            if stable:
                print(f"  ERROR: Indices don't match for stable=True! This is a bug.")
            else:
                print(f"  WARNING: Indices don't match for stable=False.")
                print(f"    This could be acceptable if both are valid unstable orderings.")
                analyze_tie_breaking_validity(tensor, eager_indices, compiled_indices)

def analyze_tie_breaking_validity(tensor, eager_indices, compiled_indices):
    """Analyze if both tie-breaking results are valid for unstable sort."""
    print(f"\n  Analyzing tie-breaking validity:")

    # Both should have same values when indexed
    eager_sorted = tensor[eager_indices]
    compiled_sorted = tensor[compiled_indices]

    if not torch.equal(eager_sorted, compiled_sorted):
        print(f"    ERROR: Sorted values don't match!")
        print(f"    Eager sorted:    {eager_sorted}")
        print(f"    Compiled sorted: {compiled_sorted}")
        return

    print(f"    Both produce correctly sorted values: {eager_sorted}")

    # Check if the differences are only in tie-breaking
    unique_values = torch.unique(tensor)
    valid_tie_breaking = True

    for val in unique_values:
        # Find positions of this value in both results
        eager_positions = (eager_sorted == val).nonzero().squeeze()
        compiled_positions = (compiled_sorted == val).nonzero().squeeze()

        if eager_positions.numel() > 1:  # Multiple occurrences
            eager_orig_indices = eager_indices[eager_positions]
            compiled_orig_indices = compiled_indices[compiled_positions]

            print(f"    Value {val}: eager order {eager_orig_indices}, compiled order {compiled_orig_indices}")

            # For unstable sort, any permutation of the original indices is valid
            if not torch.equal(torch.sort(eager_orig_indices)[0],
                             torch.sort(compiled_orig_indices)[0]):
                print(f"    ERROR: Different original indices for value {val}")
                valid_tie_breaking = False

    if valid_tie_breaking:
        print(f"    CONCLUSION: Both represent valid unstable sort results")
    else:
        print(f"    CONCLUSION: Invalid tie-breaking detected")

def test_specific_pattern():
    """Test a specific pattern that should highlight the issue."""
    print(f"\n{'='*60}")
    print("Testing Specific Pattern to Highlight Issue #174459")
    print("=" * 60)

    # Pattern with obvious tie-breaking differences
    tensor = torch.tensor([1.0, 1.0, 1.0, 1.0], device='cpu')
    print(f"All identical values: {tensor}")

    for stable in [True, False]:
        print(f"\nstable={stable}:")

        # Multiple runs to check consistency
        eager_results = []
        compiled_results = []

        for i in range(3):
            # Eager
            eager_vals, eager_idx = torch.sort(tensor, stable=stable)
            eager_results.append(eager_idx.clone())

            # Compiled
            @torch.compile
            def sort_func(x):
                return torch.sort(x, stable=stable)

            comp_vals, comp_idx = sort_func(tensor)
            compiled_results.append(comp_idx.clone())

        print(f"  Eager results:    {[r.tolist() for r in eager_results]}")
        print(f"  Compiled results: {[r.tolist() for r in compiled_results]}")

        # Check consistency within each method
        eager_consistent = all(torch.equal(r, eager_results[0]) for r in eager_results)
        compiled_consistent = all(torch.equal(r, compiled_results[0]) for r in compiled_results)

        print(f"  Eager consistent: {eager_consistent}")
        print(f"  Compiled consistent: {compiled_consistent}")

        # Check if they match each other
        methods_match = torch.equal(eager_results[0], compiled_results[0])
        print(f"  Methods match: {methods_match}")

def test_cuda_behavior():
    """Test CUDA behavior if available."""
    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping CUDA tests")
        return

    print(f"\n{'='*60}")
    print("CUDA Sort Behavior (Always Stable)")
    print("=" * 60)

    tensor = torch.tensor([3.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0], device='cuda')
    print(f"Input tensor: {tensor}")

    for stable in [True, False]:
        print(f"\nCUDA stable={stable}:")
        values, indices = torch.sort(tensor, stable=stable)
        print(f"  Values: {values}")
        print(f"  Indices: {indices}")

if __name__ == "__main__":
    # Clear any compilation cache
    dynamo.reset()

    try:
        test_sort_tie_breaking()
        test_specific_pattern()
        test_cuda_behavior()

        print(f"\n{'='*60}")
        print("Expected Behavior Summary")
        print("=" * 60)
        print("- stable=True: Both eager and compiled should produce identical results")
        print("- stable=False: Results may differ but should be valid unstable sorts")
        print("- The fix removes arbitrary tie-breaking to match std::sort behavior")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis may be expected if PyTorch build is not complete yet.")