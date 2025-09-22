# mypy: ignore-errors

"""
PyTorch Operation Fuzzer for Dynamic Shapes Testing.

This package provides comprehensive fuzzing tools for testing torch.compile
with dynamic shapes and diverse operation patterns.
"""

from fuzzer import fuzz_and_execute, fuzz_operation_stack
from ops_fuzzer import fuzz_op, fuzz_spec, Operation
from tensor_fuzzer import (
    fuzz_scalar,
    fuzz_tensor,
    fuzz_tensor_simple,
    fuzz_tensor_size,
    fuzz_torch_tensor_type,
    fuzz_valid_stride,
    ScalarSpec,
    Spec,
    TensorSpec,
    test_fuzzing_tensors,
)
from visualize_stack import operation_stack_to_dot, visualize_operation_stack


__all__ = [
    # Core fuzzing functionality
    "fuzz_and_execute",
    "fuzz_operation_stack",
    "fuzz_op",
    "fuzz_spec",
    # Tensor fuzzing
    "fuzz_tensor",
    "fuzz_tensor_simple",
    "fuzz_tensor_size",
    "fuzz_torch_tensor_type",
    "fuzz_valid_stride",
    "fuzz_scalar",
    "test_fuzzing_tensors",
    # Data types and configuration
    "Operation",
    "TensorSpec",
    "ScalarSpec",
    "Spec",
    # Visualization
    "operation_stack_to_dot",
    "visualize_operation_stack",
]
