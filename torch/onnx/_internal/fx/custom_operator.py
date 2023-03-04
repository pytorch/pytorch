from typing import Callable

import torch
from .function_dispatcher import _ATENLIB_FUNCTIONS, _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE

ONNX_CUSTOM_OP_LIB = torch.library.Library("onnx_custom", "DEF")


def _register_onnx_custom_op_overload(schema: str):
    def inner(f: Callable):
        if "::" in schema:
            domain = schema.split("::", 2)[0]
            assert (
                domain == "onnx_custom"
            ), f"operator domain must be onnx_custom but found {schema}"
        name = ONNX_CUSTOM_OP_LIB.define(schema)
        ONNX_CUSTOM_OP_LIB.impl(name, f, "CompositeExplicitAutograd")
        return getattr(torch.ops.onnx_custom, name)

    return inner


def _register_exporter_for_op_overload(
    op_overload: torch._ops.OpOverload, exporter_key: str, exporter: Callable
) -> None:
    assert op_overload not in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
    assert exporter_key not in _ATENLIB_FUNCTIONS
    _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[op_overload] = exporter_key
    _ATENLIB_FUNCTIONS[exporter_key] = exporter
