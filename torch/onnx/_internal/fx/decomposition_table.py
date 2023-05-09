"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

from typing import (
    Callable,
    Dict,
    Mapping,
    Union,
)

import torch
import torch._ops
import torch._decomp
import torch.fx
from torch.onnx._internal import _beartype

from torch.onnx._internal.fx import registration

def _create_op_overload_to_exporter_key_table(registry: registration.OnnxRegistry) -> (
    Mapping[Union[torch._ops.OpOverload, Callable], str]
):
    # TODO(justinchuby): Improve how the table is constructed.
    table: Dict[Union[torch._ops.OpOverload, Callable], str] = {}

    # Some ops in `torch.ops.aten` are not discoverable through `dir(torch.ops.aten)`,
    # but retrievable via explicit lookup.
    # https://github.com/pytorch/pytorch/issues/99681
    # This is a workaround to make sure we register ONNX symbolic functions for these.
    onnx_supported_aten_lookup_table = [
        k.split("::")[1].split(".")[0]
        for k in registry.all_functions()
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

            exporter_look_up_key = op_overload_packet._qualified_op_name
            if registry.get_function_group(exporter_look_up_key) is None:
                # This aten op doesn't have ONNX exporter.
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                # This line maps torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, torch.ops.aten.add.out, etc
                # to "aten::add". This means the exporter for "aten::add" is used for all overloads of "aten::add".
                # This is applied to all ops under torch.ops.aten.
                #
                # TODO(wechi): in the future, we might want to write individual exporter for each overload, if,
                # for example, they have different type promotion rules. If so, just map different overloads to
                # different exporter keys.
                table[op_overload] = op_overload_packet._qualified_op_name
    return table


@_beartype.beartype
def create_onnx_friendly_decomposition_table(registry: registration.OnnxRegistry) -> (
    Dict[torch._ops.OpOverload, Callable]
):
    """
        This is a subset of PyTorch's built-in aten-to-aten decomposition. If an aten
        op (e.g., torch.ops.aten.add.Tensor) has exporter, we exclude the op's decomposition
        function in the DEFAULT_ONNX_EXPORTER_DECOMPOSITION_TABLE.
    """
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}
    # Dictionary that maps torch.ops.aten.* to exporter look up key; e.g.,
    # _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[torch.add.Tensor] is "aten::add".
    _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE = _create_op_overload_to_exporter_key_table(registry)
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():
        # Skip decomposition into "prim::*" ops (defined in 'torch._refs'), because they
        # are not generally supported by ONNX.
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        if (
            "torch._refs" in decomp_fn.__module__
            or op_overload in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
        ):
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table
