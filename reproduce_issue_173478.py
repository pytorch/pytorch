#!/usr/bin/env python3
"""
Reproduction script for PyTorch issue #173478
Numerical Discrepancy Between Eager and Inductor for atan + special_psi Composition
"""

import torch
import traceback

def model_func(self):
    """The problematic composition of atan and special_psi"""
    out = torch.ops.aten.atan(self)
    out = torch.ops.aten.special_psi(out, out=out)
    return out

def test_discrepancy():
    """Test the numerical discrepancy between eager and inductor"""
    print("Testing numerical discrepancy for atan + special_psi composition...")

    # Create test input
    input_config = {'self': torch.randn([4, 8, 16], dtype=torch.float32, device='cuda')}
    print(f"Input shape: {input_config['self'].shape}")
    print(f"Input device: {input_config['self'].device}")
    print(f"Input dtype: {input_config['self'].dtype}")

    # Compile with eager backend
    print("\nCompiling with eager backend...")
    compiled_eager = torch.compile(model_func, backend="eager")
    out_eager = compiled_eager(**input_config)

    # Compile with inductor backend
    print("Compiling with inductor backend...")
    compiled_inductor = torch.compile(model_func, backend="inductor")
    out_inductor = compiled_inductor(**input_config)

    print(f"Eager output shape: {out_eager.shape}")
    print(f"Inductor output shape: {out_inductor.shape}")

    # Compare results
    try:
        torch.testing.assert_close(out_eager, out_inductor)
        print("✅ PASS: Outputs match within tolerance")
        return True
    except AssertionError as e:
        print(f"❌ FAIL: Numerical discrepancy detected")
        print(f"Error: {e}")

        # Additional analysis
        diff = torch.abs(out_eager - out_inductor)
        max_abs_diff = torch.max(diff)
        mean_abs_diff = torch.mean(diff)

        # Compute relative differences where denominator is non-zero
        mask = torch.abs(out_eager) > 1e-10
        rel_diff = torch.where(mask, diff / torch.abs(out_eager), torch.zeros_like(diff))
        max_rel_diff = torch.max(rel_diff)

        mismatched = torch.sum(diff > 1e-5)
        total_elements = torch.numel(out_eager)

        print(f"Max absolute difference: {max_abs_diff:.10f}")
        print(f"Mean absolute difference: {mean_abs_diff:.10f}")
        print(f"Max relative difference: {max_rel_diff:.10f}")
        print(f"Mismatched elements: {mismatched}/{total_elements} ({100*mismatched/total_elements:.2f}%)")

        return False

def detailed_analysis():
    """Perform detailed analysis of the operations"""
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)

    # Test with a simple input to see intermediate values
    simple_input = torch.tensor([0.1, 0.5, 1.0, 1.5], dtype=torch.float32, device='cuda')

    print(f"Simple input: {simple_input}")

    # Step by step comparison
    print("\nStep-by-step comparison:")

    # Step 1: atan
    atan_eager = torch.ops.aten.atan(simple_input)
    print(f"atan (eager): {atan_eager}")

    # We need to examine what inductor would do for atan
    # Let's create a simple atan-only model
    def atan_only(x):
        return torch.ops.aten.atan(x)

    compiled_atan_inductor = torch.compile(atan_only, backend="inductor")
    atan_inductor = compiled_atan_inductor(simple_input)
    print(f"atan (inductor): {atan_inductor}")
    print(f"atan diff: {torch.abs(atan_eager - atan_inductor)}")

    # Step 2: special_psi on the results
    psi_eager = torch.ops.aten.special_psi(atan_eager)
    psi_inductor = torch.ops.aten.special_psi(atan_inductor)

    print(f"psi(atan) eager: {psi_eager}")
    print(f"psi(atan) inductor: {psi_inductor}")
    print(f"final diff: {torch.abs(psi_eager - psi_inductor)}")

if __name__ == "__main__":
    print("PyTorch Issue #173478 Reproduction")
    print("Numerical Discrepancy Between Eager and Inductor for atan + special_psi Composition")
    print("=" * 80)

    # Check PyTorch version and setup
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Run the main test
    bug_reproduced = not test_discrepancy()  # test_discrepancy returns False when bug is found

    # Detailed analysis
    if bug_reproduced:
        detailed_analysis()

    print(f"\nBug reproduction: {'SUCCESS' if bug_reproduced else 'FAILED'}")