# mypy: allow-untyped-defs
from __future__ import annotations

import itertools
from typing import Callable, TYPE_CHECKING

import torch
import torch._ops


if TYPE_CHECKING:
    from torch.onnx._internal.exporter import _registration


def get_onnx_implemented_overloads(
    registry: _registration.ONNXRegistry,
) -> list[torch._ops.OperatorBase]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    registered_ops: list[torch._ops.OperatorBase] = []
    for op_namespace in (torch.ops.aten, torch.ops.prims):
        op_names = dir(op_namespace)
        for op_name in op_names:
            op_overload_packet = getattr(op_namespace, op_name)
            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                if registry.is_registered(op_overload):
                    registered_ops.append(op_overload)
    return registered_ops


def create_onnx_friendly_decomposition_table(
    onnx_registered_ops: set[torch._ops.OperatorBase],
) -> dict[torch._ops.OperatorBase, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        onnx_registered_ops: All ops that have an ONNX decomposition implemented.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: dict[torch._ops.OperatorBase, Callable] = {}

    for op_overload, decomp_fn in itertools.chain(
        torch._export.utils._decomp_table_to_post_autograd_aten().items(),  # type: ignore[attr-defined]
        torch._decomp.decomposition_table.items(),  # type: ignore[attr-defined]
    ):
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        # NOTE: Do not skip torch._refs decomps. They are fine because otherwise the model is
        # not exportable anyways.
        if op_overload in onnx_registered_ops:
            continue
        # If it is HOP, we filter those out as well.
        if not hasattr(op_overload, "_schema"):
            continue
        decomposition_table[op_overload] = decomp_fn

    return decomposition_table
