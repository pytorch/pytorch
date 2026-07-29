"""Normalize selected FlexGEMM epilogue FX nodes for shared consumers.

The original FX graph remains the source of node ordering and pointwise
expressions. This module extracts canonical arguments only for nodes interpreted
by both semantic analysis and CuTeDSL emission, so those phases share one view
of each supported schema.
"""

import dataclasses
import operator
from typing import Any

import torch
from torch._inductor import inductor_prims


@dataclasses.dataclass(frozen=True)
class NormalizedView:
    """Canonical source and shape for a view or reshape."""

    source: torch.fx.Node
    shape: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class NormalizedReduction:
    """Canonical arguments for a supported reduction."""

    source: torch.fx.Node
    dim: Any
    keepdim: Any
    dtype: Any
    reduction_type: str


@dataclasses.dataclass(frozen=True)
class NormalizedPrepareSoftmax:
    """Canonical source and dimension for online softmax preparation."""

    source: torch.fx.Node
    dim: Any


@dataclasses.dataclass(frozen=True)
class NormalizedSqueeze:
    """Canonical source for a squeeze alias."""

    source: torch.fx.Node


@dataclasses.dataclass(frozen=True)
class NormalizedGetItem:
    """Canonical aggregate source and literal getitem index."""

    source: torch.fx.Node
    index: int


@dataclasses.dataclass(frozen=True)
class NormalizedUnsupportedReduction:
    """Canonical source and target for an unsupported reduction."""

    source: torch.fx.Node
    target: str


NormalizedNode = (
    NormalizedView
    | NormalizedReduction
    | NormalizedPrepareSoftmax
    | NormalizedSqueeze
    | NormalizedGetItem
    | NormalizedUnsupportedReduction
)


FUNCTION_REDUCTION_TYPES = {
    torch.ops.aten.sum.dim_IntList: ("sum", True),
    torch.ops.aten.mean.dim: ("mean", True),
    torch.ops.aten.prod.dim_int: ("prod", True),
    torch.ops.aten.amax.default: ("max", False),
    torch.ops.aten.amin.default: ("min", False),
}

FUNCTION_UNSUPPORTED_REDUCTIONS = frozenset(
    (
        torch.ops.aten.all.dim,
        torch.ops.aten.all.dims,
        torch.ops.aten.all.default,
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
        torch.ops.aten.any.default,
        torch.ops.aten.argmax.default,
        torch.ops.aten.argmin.default,
        torch.ops.aten.std.correction,
        torch.ops.aten.std.dim,
        torch.ops.aten.var.correction,
        torch.ops.aten.var.dim,
    )
)


def normalize_flex_gemm_epilogue_fx_node(
    node: torch.fx.Node,
) -> NormalizedNode | None:
    """Return canonical arguments for a selected FX node, or ``None``."""
    if node.op != "call_function":
        return None
    if node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        source = node.args[0]
        shape = node.args[1]
        if not isinstance(source, torch.fx.Node) or not isinstance(
            shape, (tuple, list, torch.Size)
        ):
            raise AssertionError(f"malformed FlexGEMM view node: {node.format_node()}")
        return NormalizedView(
            source,
            tuple(
                arg.meta.get("val", arg) if isinstance(arg, torch.fx.Node) else arg
                for arg in shape
            ),
        )
    if node.target in FUNCTION_REDUCTION_TYPES:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed FlexGEMM reduction node: {node.format_node()}"
            )
        reduction_type, has_dtype = FUNCTION_REDUCTION_TYPES[node.target]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        return NormalizedReduction(
            source,
            dim,
            keepdim,
            dtype if has_dtype else None,
            reduction_type,
        )
    if node.target is inductor_prims.prepare_softmax_online:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed FlexGEMM softmax node: {node.format_node()}"
            )
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        return NormalizedPrepareSoftmax(source, dim)
    if node.target in (
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze.default,
    ):
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed FlexGEMM squeeze node: {node.format_node()}"
            )
        return NormalizedSqueeze(source)
    if node.target is operator.getitem:
        source, index = node.args
        if isinstance(source, torch.fx.Node) and isinstance(index, int):
            return NormalizedGetItem(source, index)
        return None
    if node.target in FUNCTION_UNSUPPORTED_REDUCTIONS:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed FlexGEMM reduction node: {node.format_node()}"
            )
        return NormalizedUnsupportedReduction(source, str(node.target))
    return None
