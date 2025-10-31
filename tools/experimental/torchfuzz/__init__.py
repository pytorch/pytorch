"""Torchfuzz package for generating and testing random PyTorch operations."""

# Make key classes available at package level
from .operators import get_operator, list_operators, register_operator
from .ops_fuzzer import fuzz_operation_graph, fuzz_spec, OperationGraph
from .tensor_fuzzer import ScalarSpec, Spec, TensorSpec


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
