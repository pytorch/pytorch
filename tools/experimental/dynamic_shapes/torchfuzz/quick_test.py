#!/usr/bin/env python3

import sys
import os

# Add the torchfuzz directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torchfuzz.operators.matrix_multiply import MatmulOperator
from torchfuzz.tensor_fuzzer import TensorSpec

def test_dtype_fix():
    """Test that matrix multiplication operators produce compatible dtypes."""
    print("Testing dtype compatibility fix...")
    
    # Create a MatmulOperator
    matmul_op = MatmulOperator()
    
    # Test with float32 output spec
    output_spec = TensorSpec(size=(4, 6), stride=(6, 1), dtype=torch.float32)
    
    if matmul_op.can_produce(output_spec):
        input_specs = matmul_op.fuzz_inputs_specs(output_spec)
        print(f"Output dtype: {output_spec.dtype}")
        print(f"Input dtypes: {[spec.dtype for spec in input_specs]}")
        
        # Verify all dtypes match
        for i, spec in enumerate(input_specs):
            if spec.dtype != output_spec.dtype:
                print(f"❌ Input {i} dtype mismatch: {spec.dtype} != {output_spec.dtype}")
                return False
        
        print("✅ All dtypes match!")
        return True
    else:
        print("❌ Cannot produce output spec")
        return False

def test_1d_matmul_fix():
    """Test that matmul handles 1D outputs correctly."""
    print("Testing 1D matmul fix...")
    
    # Create a MatmulOperator  
    matmul_op = MatmulOperator()
    
    # Test with 1D output spec (this should work now)
    output_spec = TensorSpec(size=(5,), stride=(1,), dtype=torch.float32)
    
    if matmul_op.can_produce(output_spec):
        input_specs = matmul_op.fuzz_inputs_specs(output_spec)
        print(f"1D Output size: {output_spec.size}")
        print(f"Input sizes: {[spec.size for spec in input_specs]}")
        
        # Verify that input shapes can actually produce the 1D output via matmul
        # Valid patterns are: (k,) @ (k, n) -> (n,) or (n, k) @ (k,) -> (n,)
        spec1, spec2 = input_specs
        
        if len(spec1.size) == 1 and len(spec2.size) == 2:
            # Pattern: (k,) @ (k, n) -> (n,)
            k1, (k2, n) = spec1.size[0], spec2.size
            if k1 == k2 and n == output_spec.size[0]:
                print(f"✅ Valid pattern: ({k1},) @ ({k2}, {n}) -> ({output_spec.size[0]},)")
                return True
        elif len(spec1.size) == 2 and len(spec2.size) == 1:
            # Pattern: (n, k) @ (k,) -> (n,)
            (n, k1), k2 = spec1.size, spec2.size[0] 
            if k1 == k2 and n == output_spec.size[0]:
                print(f"✅ Valid pattern: ({n}, {k1}) @ ({k2},) -> ({output_spec.size[0]},)")
                return True
        
        print(f"❌ Invalid pattern: {spec1.size} @ {spec2.size} -> {output_spec.size}")
        return False
    else:
        print("❌ Cannot produce 1D output spec")
        return False

if __name__ == "__main__":
    success1 = test_dtype_fix()
    success2 = test_1d_matmul_fix()
    
    if success1 and success2:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        sys.exit(1)