# mypy: allow-untyped-defs
"""Lower FlexGEMM grouped local reductions into QuACK/CuTeDSL epilogues.

FlexGEMM recognizes a narrow local-reduction contract inside the GEMM output
tile: an epilogue reshapes the accumulator to expose contiguous groups along M
or N, then reduces only that grouped dimension. N-axis groups up to one 32-lane
fragment lower as ordinary in-fragment TensorSSA reductions; larger N groups
produce TensorSSA partials that QuACK combines physically.
M-axis groups currently always use QuACK's physical row-lane/warp combine path,
even when the group is small enough to fit in one fragment. Inductor owns FX
normalization and output contracts; these helpers describe the supported
TensorSSA shapes and generated combine/finalize expressions QuACK needs.

The main caller is ``materialize_flex_gemm_epilogue`` in ``fx_cutedsl_codegen.py``.
Shared GEMM epilogue analysis recognizes grouped layouts and reductions;
materialization emits the accepted geometries as TensorSSA reshapes and
reductions through the helpers below.
"""

import dataclasses
from typing import Any

import torch
from torch._inductor.kernel.gemm_epilogue import iter_fx_node_inputs
from torch._inductor.kernel.gemm_epilogue_codegen import _cute_op_name
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch.fx.experimental.symbolic_shapes import (
    guard_int,
    has_guarding_hint,
    optimization_hint,
)


@dataclasses.dataclass(frozen=True)
class FlexGemmStructuralInt:
    """Hold a backed structural hint until an accepted match installs a guard."""

    value: int
    symbolic: torch.SymInt | None = None

    @classmethod
    def from_value(cls, value: Any) -> "FlexGemmStructuralInt | None":
        """Return a guard-free structural hint, rejecting unbacked values."""
        if isinstance(value, torch.fx.Node):
            value = value.meta.get("val")
        if isinstance(value, torch.SymInt):
            if not has_guarding_hint(value):
                return None
            return cls(optimization_hint(value), value)
        return cls(value) if isinstance(value, int) else None

    def guard(self) -> None:
        """Install the specialization guard after semantic validation succeeds."""
        if self.symbolic is not None and guard_int(self.symbolic) != self.value:
            raise AssertionError("FlexGEMM structural hint changed before commit")


FLEX_GEMM_POINTWISE_OP_NAMES = frozenset(
    (
        "_to_copy",
        "clamp",
        "clamp_max",
        "clamp_min",
        "convert_element_type",
        "inline_asm_elementwise",
    )
)


def tensor_meta_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    """Return fake-tensor shape metadata when the FX value is tensor-like."""
    meta = node.meta.get("val")
    if isinstance(meta, torch.Tensor):
        return tuple(meta.shape)
    return None


def node_preserves_tensor_shapes(node: torch.fx.Node) -> bool:
    """Reject pointwise broadcasts that cannot preserve a grouped TensorSSA input."""
    output_shape = tensor_meta_shape(node)
    if output_shape is None:
        return False
    output_shape_key = tuple(str(dim) for dim in output_shape)
    has_same_shape_input = False
    for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
        input_shape = tensor_meta_shape(input_node)
        if input_shape is None:
            continue
        input_shape_key = tuple(str(dim) for dim in input_shape)
        if input_shape_key == output_shape_key:
            has_same_shape_input = True
            continue
        try:
            broadcast_shape = get_broadcasted_shape(input_shape_key, output_shape_key)
        except AssertionError:
            return False
        if broadcast_shape is None or tuple(broadcast_shape) != output_shape_key:
            return False
    return has_same_shape_input


def is_pointwise_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
    ) or _cute_op_name(node.target) in FLEX_GEMM_POINTWISE_OP_NAMES


def is_shape_preserving_pointwise_node(node: torch.fx.Node) -> bool:
    return is_pointwise_node(node) and node_preserves_tensor_shapes(node)


def view_or_reshape_args(node: torch.fx.Node) -> tuple[Any, tuple[Any, ...]] | None:
    if node.op == "call_function" and node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        shape = node.args[1]
        if isinstance(shape, (tuple, list, torch.Size)):
            return node.args[0], tuple(
                arg.meta.get("val", arg) if isinstance(arg, torch.fx.Node) else arg
                for arg in shape
            )
    return None


def squeeze_source_node(node: torch.fx.Node) -> torch.fx.Node | None:
    if node.op != "call_function" or node.target not in (
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze.default,
    ):
        return None
    source_node = node.args[0]
    return source_node if isinstance(source_node, torch.fx.Node) else None
