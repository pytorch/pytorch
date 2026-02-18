#!/usr/bin/env python3
"""
Detailed analysis of PyTorch issue #173478
Numerical Discrepancy Between Eager and Inductor for atan + special_psi Composition
"""

import torch
import numpy as np

def analyze_atan_precision():
    """Analyze precision differences in atan implementation"""
    print("=" * 60)
    print("ANALYZING ATAN PRECISION DIFFERENCES")
    print("=" * 60)

    # Test with specific values that might show larger differences
    test_values = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0], dtype=torch.float32, device='cuda')
    print(f"Test values: {test_values}")

    # Compare atan implementations
    def atan_only(x):
        return torch.ops.aten.atan(x)

    # Eager
    eager_atan = atan_only(test_values)
    print(f"Eager atan: {eager_atan}")

    # Inductor
    compiled_atan = torch.compile(atan_only, backend="inductor")
    inductor_atan = compiled_atan(test_values)
    print(f"Inductor atan: {inductor_atan}")

    # Differences
    atan_diff = torch.abs(eager_atan - inductor_atan)
    print(f"Absolute differences: {atan_diff}")
    print(f"Max absolute difference: {torch.max(atan_diff):.12f}")

    # ULP analysis
    print("\nULP (Unit in the Last Place) analysis:")
    for i in range(len(test_values)):
        eager_val = eager_atan[i].item()
        inductor_val = inductor_atan[i].item()

        # Convert to binary representation for ULP calculation
        eager_bits = np.frombuffer(np.array([eager_val], dtype=np.float32).tobytes(), dtype=np.uint32)[0]
        inductor_bits = np.frombuffer(np.array([inductor_val], dtype=np.float32).tobytes(), dtype=np.uint32)[0]
        ulp_diff = abs(int(eager_bits) - int(inductor_bits))

        print(f"  Value {test_values[i]:.1f}: ULP diff = {ulp_diff}, abs diff = {atan_diff[i]:.12f}")

def analyze_special_psi_sensitivity():
    """Analyze how sensitive special_psi is to small input changes"""
    print("\n" + "=" * 60)
    print("ANALYZING SPECIAL_PSI SENSITIVITY")
    print("=" * 60)

    # Test with values around the atan outputs
    base_val = 0.7854  # approximately atan(1.0)

    # Create small variations
    epsilons = [0.0, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6]
    print("Analyzing sensitivity of special_psi around atan(1.0) ≈ 0.7854")

    for eps in epsilons:
        val = base_val + eps
        tensor_val = torch.tensor([val], dtype=torch.float32, device='cuda')
        psi_val = torch.ops.aten.special_psi(tensor_val)

        if eps == 0.0:
            base_psi = psi_val[0].item()
            print(f"Base value: psi({val:.10f}) = {base_psi:.10f}")
        else:
            psi_diff = psi_val[0].item() - base_psi
            sensitivity = psi_diff / eps if eps > 0 else 0
            print(f"Epsilon {eps:.2e}: psi({val:.10f}) = {psi_val[0].item():.10f}, diff = {psi_diff:.2e}, sensitivity = {sensitivity:.2e}")

def test_composition_step_by_step():
    """Test the full composition step by step"""
    print("\n" + "=" * 60)
    print("TESTING COMPOSITION STEP BY STEP")
    print("=" * 60)

    # Use a fixed input for deterministic results
    torch.manual_seed(42)
    input_tensor = torch.randn([4], dtype=torch.float32, device='cuda')
    print(f"Input: {input_tensor}")

    # Step 1: atan only
    def atan_only(x):
        return torch.ops.aten.atan(x)

    eager_atan = atan_only(input_tensor)
    compiled_atan = torch.compile(atan_only, backend="inductor")
    inductor_atan = compiled_atan(input_tensor)

    print(f"Eager atan: {eager_atan}")
    print(f"Inductor atan: {inductor_atan}")
    print(f"Atan diff: {torch.abs(eager_atan - inductor_atan)}")

    # Step 2: special_psi only on both results
    eager_psi = torch.ops.aten.special_psi(eager_atan)
    inductor_psi = torch.ops.aten.special_psi(inductor_atan)

    print(f"psi(eager_atan): {eager_psi}")
    print(f"psi(inductor_atan): {inductor_psi}")
    print(f"Final diff: {torch.abs(eager_psi - inductor_psi)}")

    # Step 3: Full composition with both backends
    def full_composition(x):
        out = torch.ops.aten.atan(x)
        out = torch.ops.aten.special_psi(out, out=out)
        return out

    eager_full = torch.compile(full_composition, backend="eager")(input_tensor)
    inductor_full = torch.compile(full_composition, backend="inductor")(input_tensor)

    print(f"Full eager: {eager_full}")
    print(f"Full inductor: {inductor_full}")
    print(f"Full composition diff: {torch.abs(eager_full - inductor_full)}")

def investigate_fallback_behavior():
    """Check if special_psi falls back to native implementation"""
    print("\n" + "=" * 60)
    print("INVESTIGATING FALLBACK BEHAVIOR")
    print("=" * 60)

    # Check if special_psi has inductor support
    test_input = torch.tensor([1.0], dtype=torch.float32, device='cuda')

    def just_psi(x):
        return torch.ops.aten.special_psi(x)

    # Try to compile just special_psi
    try:
        compiled_psi = torch.compile(just_psi, backend="inductor")
        result = compiled_psi(test_input)
        print(f"special_psi compiled successfully: {result}")

        # Check if it produces identical results
        eager_result = just_psi(test_input)
        print(f"Eager special_psi: {eager_result}")
        print(f"Difference: {torch.abs(result - eager_result)}")

        if torch.allclose(result, eager_result, atol=1e-15):
            print("✓ special_psi produces identical results - likely using fallback")
        else:
            print("✗ special_psi has different implementations")

    except Exception as e:
        print(f"Error compiling special_psi: {e}")

if __name__ == "__main__":
    print("Detailed Analysis of Issue #173478")
    print("Numerical Discrepancy Between Eager and Inductor for atan + special_psi Composition")
    print("=" * 80)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    analyze_atan_precision()
    analyze_special_psi_sensitivity()
    test_composition_step_by_step()
    investigate_fallback_behavior()

    print("\n" + "=" * 80)
    print("Analysis complete.")