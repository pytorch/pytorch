"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

from typing import Callable, Dict, Set, Union

import torch
import torch._ops
import torch.fx

from torch.onnx._internal import _beartype

from torch.onnx._internal.fx import registration


# NOTE: OnnxRegistry annotation: beartype is a runtime type checker for python3,
# so it doesn't work with TYPE_CHECKING
@_beartype.beartype
def _create_onnx_supports_op_overload_table(
    registry,
) -> Set[Union[torch._ops.OperatorBase, Callable]]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry (OnnxRegistry): The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    table: Set[Union[torch._ops.OperatorBase, Callable]] = set()

    # Some ops in `torch.ops.aten` are not discoverable through `dir(torch.ops.aten)`,
    # but retrievable via explicit lookup.
    # https://github.com/pytorch/pytorch/issues/99681
    # This is a workaround to make sure we register ONNX symbolic functions for these.
    onnx_supported_aten_lookup_table = [
        k.split("::")[1].split(".")[0]
        for k in registry._all_registered_ops()
        if k.startswith("aten::")
    ]

    for op_namespace in (torch.ops.aten, torch.ops.prims):
        attr_names = dir(op_namespace)
        if op_namespace is torch.ops.aten:
            attr_names += onnx_supported_aten_lookup_table
        for attr_name in attr_names:
            if not hasattr(op_namespace, attr_name):
                # torchlib owns some attributes that are not aten ops.
                continue
            op_overload_packet = getattr(op_namespace, attr_name)
            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                internal_op_name = registration.OpName.from_qualified_name(
                    qualified_name=op_overload.name()
                )
                # NOTE: If the overload is supported in registry or it's default overload is supported in registry,
                # we add it to the table.
                if registry.is_registered_op(
                    namespace=internal_op_name.namespace,
                    op_name=internal_op_name.op_name,
                    overload=internal_op_name.overload,
                ) or registry.is_registered_op(
                    namespace=internal_op_name.namespace,
                    op_name=internal_op_name.op_name,
                    overload=None,
                ):
                    # This line maps torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, torch.ops.aten.add.out, etc
                    # to "aten::add". This means the exporter for "aten::add" is used for all overloads of "aten::add".
                    # This is applied to all ops under torch.ops.aten.
                    table.add(op_overload)
    return table


@_beartype.beartype
def create_onnx_friendly_decomposition_table(
    registry,
) -> Dict[torch._ops.OperatorBase, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        registry (torch.onnx.OnnxRegistry): The ONNX registry for PyTorch.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: Dict[torch._ops.OperatorBase, Callable] = {}
    # Dictionary that maps torch.ops.aten.* to exporter look up key; e.g.,
    # _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[torch.add.Tensor] is "aten::add".
    _ONNX_SUPPORT_OP_OVERLOADS = _create_onnx_supports_op_overload_table(registry)

    # NOTE: If we import torch._decomp, we will get RuntimeError: Only a single
    # TORCH_LIBRARY can be used to register the namespace nvprims; please put all of your
    # definitions in a single TORCH_LIBRARY block.
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():  # type: ignore[attr-defined]
        # Skip decomposition into "prim::*" ops (defined in 'torch._refs'), because they
        # are not generally supported by ONNX.
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        if (
            "torch._refs" in decomp_fn.__module__
            or op_overload in _ONNX_SUPPORT_OP_OVERLOADS
        ):
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table
