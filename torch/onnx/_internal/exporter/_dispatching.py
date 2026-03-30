# mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.onnx._internal.exporter import _registration


def _arg_has_complex_dtype(arg) -> bool:
    """Check if the node has complex dtype recursively."""
    if (
        isinstance(arg, torch.fx.Node)
        and "val" in arg.meta
        and isinstance(arg.meta["val"], torch.Tensor)
        and torch.is_complex(arg.meta["val"])
    ):
        return True
    elif isinstance(arg, list):
        return any(_arg_has_complex_dtype(item) for item in arg)
    return False


def dispatch(
    node: torch.fx.Node, registry: _registration.ONNXRegistry
) -> tuple[Callable | None, str]:
    """Dispatch a node to an ONNX function based on the node's target and the ONNX registry.

    Args:
        node: The node to dispatch.
        registry: The ONNX registry to use for dispatching.

    Returns:
        A tuple containing the matched ONNX function and a string describing the reason for failure or success.
    """
    decomp_metas = registry.get_decomps(node.target)  # type: ignore[arg-type]
    # Determine if the node has complex inputs.
    is_complex = any(_arg_has_complex_dtype(arg) for arg in node.args) or any(
        _arg_has_complex_dtype(arg) for arg in node.kwargs.values()
    )
    if is_complex:
        decomp_metas = [decomp for decomp in decomp_metas if decomp.is_complex]
        if not decomp_metas:
            return None, "No decompositions registered for the complex-valued input"
    else:
        decomp_metas = [decomp for decomp in decomp_metas if not decomp.is_complex]
        if not decomp_metas:
            return None, "No decompositions registered for the real-valued input"

    # NOTE: Complex overload type matching logic has been removed to keep this simple
    # There should no longer be overloads (for the same opset version) in torchlib anymore
    return (decomp_metas[0].onnx_function, "The first implementation is used")
