"""Torchfuzz package for generating and testing random PyTorch operations."""

# Make key classes available at package level
from .tensor_fuzzer import TensorSpec, ScalarSpec, Spec
from .ops_fuzzer import OperationGraph, fuzz_operation_graph, fuzz_spec
from .operators import get_operator, register_operator, list_operators

__all__ = [
    "TensorSpec",
    "ScalarSpec",
    "Spec",
    "OperationGraph",
    "fuzz_operation_graph",
    "fuzz_spec",
    "get_operator",
    "register_operator",
    "list_operators",
]
