from typing import Callable

import torch
from .function_dispatcher import _ATENLIB_FUNCTIONS, _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE

ONNX_CUSTOM_OP_LIB = torch.library.Library("onnx_custom", "DEF")


def _register_onnx_custom_op_overload(schema: str):
    """This function decorator registers the wrapped function as a `torch._ops.OpOverload` with the provided `schema`.

    Arguments:
        schema (str): The schema of the wrapped function in PyTorch's type system; e.g., "my_op(Tensor a, Tensor b) -> Tensor c".

    Example:
    The following example register the `a_funny_op` function as `torch.ops.onnx_custom.my_op` (type: torch._ops.OpOverload).

        import torch
        import torch.onnx._internal.fx.custom_operator as fx_custom
        @fx_custom._register_onnx_custom_op_overload("my_op(Tensor a, Tensor b) -> Tensor c")
        def a_funny_op(a, b):
            return a + b

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        # Now, use op_overload to invoke the custom function.
        z = torch.ops.onnx_custom.my_op(x, y)
    """

    def inner(f: Callable):
        assert (
            "::" not in schema
        ), f"OpOverload created via this API is assumed to be in the `onnx_custom` namespace, so schema\
              must not include the namespace. Received schema is {schema}"
        name = ONNX_CUSTOM_OP_LIB.define(schema)
        ONNX_CUSTOM_OP_LIB.impl(name, f, "CompositeExplicitAutograd")
        return getattr(torch.ops.onnx_custom, name)

    return inner


def _register_exporter_for_op_overload(
    op_overload: torch._ops.OpOverload, exporter_key: str, exporter: Callable
) -> None:
    """Register the exporter for custom `op_overload`.

    Pseudocode of the mapping process when processing one FX node:
        # See a `torch._ops.OpOverload` node
        op_overload = node.target
        # Find its exporter key function_dispatcher._OP_OVERLOAD_TO_EXPORTER_KEY_TABLE table.
        _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[op_overload]
        # Retrieve the actual exporter and call it with proper inputs.
        _ATENLIB_FUNCTIONS[exporter_key]
    """
    assert op_overload not in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
    assert exporter_key not in _ATENLIB_FUNCTIONS
    _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[op_overload] = exporter_key
    _ATENLIB_FUNCTIONS[exporter_key] = exporter
