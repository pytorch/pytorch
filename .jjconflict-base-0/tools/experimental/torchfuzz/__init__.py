"""Torchfuzz package for generating and testing random PyTorch operations."""

# Make key classes available at package level
from .codegen import (
    DeviceInfo,
    FuzzTemplate,
    get_device_info,
    get_template_names,
    initialize_codegen,
    make_template,
)
from .operators import get_operator, list_operators, register_operator
from .ops_fuzzer import fuzz_operation_graph, fuzz_spec, OperationGraph
from .tensor_fuzzer import ScalarSpec, Spec, TensorSpec


__all__ = [
    "DeviceInfo",
    "FuzzTemplate",
    "OperationGraph",
    "ScalarSpec",
    "Spec",
    "TensorSpec",
    "fuzz_operation_graph",
    "fuzz_spec",
    "get_device_info",
    "get_operator",
    "get_template_names",
    "initialize_codegen",
    "list_operators",
    "make_template",
    "register_operator",
]
