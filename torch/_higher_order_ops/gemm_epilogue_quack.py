# mypy: allow-untyped-defs
import operator
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.gemm_epilogue import (
    GEMM_EPILOGUE_OPS,
    supported_gemm_epilogue_op_names,
)


QUACK_GEMM_OPS = tuple(
    op for op, info in GEMM_EPILOGUE_OPS.items() if info.supports_quack
)
SUPPORTED_QUACK_GEMM_OP_NAMES = supported_gemm_epilogue_op_names(quack_only=True)
QUACK_TENSORSSA_FRAGMENT_N = 32
QUACK_REDUCE_DIM_M = 0
QUACK_REDUCE_DIM_N = 1


def unwrap_output(node: torch.fx.Node) -> Any:
    if node.op != "output":
        raise RuntimeError(f"expected output node, got {node.op}")
    value = node.args[0]
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0]
    return value


def find_single_gemm_node(graph_module: torch.fx.GraphModule) -> torch.fx.Node:
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if (node.op == "call_function" and node.target in QUACK_GEMM_OPS)
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError(
            "QUACK GEMM epilogue backend currently supports one "
            f"{SUPPORTED_QUACK_GEMM_OP_NAMES} body"
        )
    return gemm_nodes[0]


@dataclass(frozen=True)
class QuackLocalReduceInfo:
    """Plan for one local reduction produced inside the fused epilogue.

    Invariants:
    - ``dim`` is one of ``QUACK_REDUCE_DIM_M`` or ``QUACK_REDUCE_DIM_N``.
    - ``group_size`` is the logical reduction group from the epilogue reshape.
    - ``producer_skip_nodes`` are FX nodes represented by this plan, not by
      generic pointwise emission.
    - ``epilogue_reduce_*`` is set only when the reduced value is derived from
      an intermediate epilogue expression rather than directly from the GEMM acc.
    """

    view_node: torch.fx.Node
    reduce_op_node: torch.fx.Node
    source_node: torch.fx.Node
    keepdim: bool
    group_size: int
    dim: int
    aux_output_node: torch.fx.Node
    feeds_main: bool = False
    kind: str = "sum"
    scale: float = 1.0
    max_power: int = 8
    producer_skip_nodes: frozenset[torch.fx.Node] = field(default_factory=frozenset)
    epilogue_reduce_source_node: torch.fx.Node | None = None
    epilogue_reduce_value_node: torch.fx.Node | None = None


@dataclass(frozen=True)
class QuackLocalNormInfo:
    output_node: torch.fx.Node
    div_node: torch.fx.Node
    view_node: torch.fx.Node
    reduce_node: torch.fx.Node
    source_node: torch.fx.Node
    group_size: int
    dim: int
    extra_skip_nodes: frozenset[torch.fx.Node] = field(default_factory=frozenset)


@dataclass(frozen=True)
class QuackViewMatch:
    node: torch.fx.Node
    base: Any
    shape: Any


@dataclass(frozen=True)
class QuackSumMatch:
    node: torch.fx.Node
    view_node: Any
    dims: tuple[Any, ...]
    keepdim: Any
    dtype: Any


@dataclass(frozen=True)
class QuackAuxOutputInfo:
    output_value: torch.fx.Node
    group_size: int | None = None
    dim: int | None = None


@dataclass(frozen=True)
class QuackMainOutputTransformInfo:
    kind: str
    group_size: int | None = None
    concat_layout: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuackOutputPlan:
    """Classified HOP body outputs and the FX nodes consumed by QUACK lowering."""

    output_value: Any
    skip_nodes: frozenset[torch.fx.Node]
    local_reduce: QuackLocalReduceInfo | None
    local_norm: QuackLocalNormInfo | None
    aux_output: QuackAuxOutputInfo | None
    main_output_transform: QuackMainOutputTransformInfo | None = None


@dataclass(frozen=True)
class QuackGroupedTensorSSAInfo:
    group_size: int
    groups_per_fragment: int
    nonnegative: bool = False

    @property
    def keepdim_reshape(self) -> str:
        return f"((1, 1, {self.groups_per_fragment}), 1, 1)"


@dataclass(frozen=True)
class QuackTensorSSAReduceMatch:
    node: torch.fx.Node
    input_node: Any
    dims: tuple[Any, ...]
    keepdim: Any
    dtype: Any
    kind: str


@dataclass(frozen=True)
class QuackTensorSSAReduceDesc:
    cute_op: str
    init_val: str
    requires_nonnegative: bool = False


QUACK_TENSORSSA_REDUCTIONS = {
    "sum": QuackTensorSSAReduceDesc("cute.ReductionOp.ADD", "0.0"),
    "amax": QuackTensorSSAReduceDesc(
        "cute.ReductionOp.MAX", "0.0", requires_nonnegative=True
    ),
}


def normalize_reduce_dims(dim: Any) -> tuple[Any, ...]:
    if isinstance(dim, (list, tuple)):
        return tuple(dim)
    return (dim,)


def normalize_shape(shape: Any) -> Any:
    if isinstance(shape, torch.Size):
        return tuple(shape)
    return shape


def tensorssa_grouped_n_shape(group_size: int) -> str:
    return f"((1, {group_size}, {QUACK_TENSORSSA_FRAGMENT_N // group_size}), 1, 1)"


def method_clamp_bounds(node: torch.fx.Node) -> tuple[Any, Any]:
    min_value = node.kwargs.get("min")
    max_value = node.kwargs.get("max")
    if "min" not in node.kwargs and len(node.args) > 1:
        min_value = node.args[1]
    if "max" not in node.kwargs and len(node.args) > 2:
        max_value = node.args[2]
    return min_value, max_value


def is_abs_node(node: Any) -> bool:
    return isinstance(node, torch.fx.Node) and (
        (
            node.op == "call_function"
            and node.target in (torch.ops.aten.abs.default, torch.abs)
        )
        or (node.op == "call_method" and node.target == "abs")
    )


def match_view_or_reshape(node: Any) -> QuackViewMatch | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_function" and node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        return QuackViewMatch(node=node, base=node.args[0], shape=node.args[1])
    if node.op == "call_method" and node.target in ("view", "reshape"):
        return QuackViewMatch(node=node, base=node.args[0], shape=node.args[1:])
    return None


def match_acc_source(base: Any, mm_node: torch.fx.Node) -> torch.fx.Node | None:
    if base is mm_node:
        return mm_node
    if (
        isinstance(base, torch.fx.Node)
        and base.op == "call_function"
        and base.target
        in (
            torch.ops.aten._to_copy.default,
            torch.ops.prims.convert_element_type.default,
        )
        and base.args[0] is mm_node
    ):
        return base
    return None


def match_sum(node: Any, *, allow_method_sum: bool) -> QuackSumMatch | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_function" and node.target == torch.ops.aten.sum.dim_IntList:
        view_node = node.args[0]
        dims = normalize_reduce_dims(
            node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        )
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        return QuackSumMatch(
            node=node, view_node=view_node, dims=dims, keepdim=keepdim, dtype=dtype
        )
    if allow_method_sum and node.op == "call_method" and node.target == "sum":
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        if dim is None:
            return None
        view_node = node.args[0]
        dims = normalize_reduce_dims(dim)
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.kwargs.get("dtype")
        return QuackSumMatch(
            node=node, view_node=view_node, dims=dims, keepdim=keepdim, dtype=dtype
        )
    return None


def match_tensorssa_reduce(node: Any) -> QuackTensorSSAReduceMatch | None:
    sum_match = match_sum(node, allow_method_sum=True)
    if sum_match is not None:
        return QuackTensorSSAReduceMatch(
            node=sum_match.node,
            input_node=sum_match.view_node,
            dims=sum_match.dims,
            keepdim=sum_match.keepdim,
            dtype=sum_match.dtype,
            kind="sum",
        )
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_function" and node.target in (
        torch.ops.aten.amax.default,
        torch.amax,
    ):
        input_node = node.args[0]
        dims = normalize_reduce_dims(
            node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        )
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        return QuackTensorSSAReduceMatch(
            node=node,
            input_node=input_node,
            dims=dims,
            keepdim=keepdim,
            dtype=None,
            kind="amax",
        )
    if node.op == "call_method" and node.target == "amax":
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        if dim is None:
            return None
        return QuackTensorSSAReduceMatch(
            node=node,
            input_node=node.args[0],
            dims=normalize_reduce_dims(dim),
            keepdim=node.kwargs.get("keepdim", False),
            dtype=None,
            kind="amax",
        )
    return None


def match_local_n_amax_reduce(
    node: Any, mm_node: torch.fx.Node, *, scale: float = 1.0
) -> QuackLocalReduceInfo | None:
    reduce_match = match_tensorssa_reduce(node)
    if reduce_match is None or reduce_match.kind != "amax":
        return None
    if reduce_match.dims not in ((-1,), (2,)) or reduce_match.dtype is not None:
        return None
    if bool(reduce_match.keepdim):
        return None
    abs_node = reduce_match.input_node
    if not is_abs_node(abs_node):
        return None
    view_match = match_view_or_reshape(abs_node.args[0])
    if view_match is None:
        return None
    source_node = match_acc_source(view_match.base, mm_node)
    if source_node is None:
        return None
    shape = grouped_n_fragment_shape(normalize_shape(view_match.shape))
    if not is_same_fragment_n_group_shape(shape):
        return None
    reduce_meta = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    mm_meta = mm_node.meta.get("val")
    if reduce_meta is None or mm_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // shape[-1])
    if tuple(reduce_meta.shape) != tuple(expected_shape):
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_op_node=reduce_match.node,
        aux_output_node=reduce_match.node,
        source_node=source_node,
        keepdim=False,
        group_size=shape[-1],
        dim=QUACK_REDUCE_DIM_N,
        kind="amax_abs",
        scale=scale,
        producer_skip_nodes=frozenset((abs_node,)),
    )


def match_scaled_local_n_amax_reduce(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op != "call_function":
        return None
    target = node.target
    args = node.args
    if target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar, operator.mul):
        if len(args) < 2:
            return None
        lhs, rhs = args[:2]
        if isinstance(rhs, (int, float)):
            local_reduce = match_local_n_amax_reduce(lhs, mm_node, scale=float(rhs))
        elif isinstance(lhs, (int, float)):
            local_reduce = match_local_n_amax_reduce(rhs, mm_node, scale=float(lhs))
        else:
            return None
    elif target in (
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        operator.truediv,
    ):
        if len(args) < 2:
            return None
        lhs, rhs = args[:2]
        if not isinstance(rhs, (int, float)) or rhs == 0:
            return None
        local_reduce = match_local_n_amax_reduce(lhs, mm_node, scale=1.0 / float(rhs))
    else:
        return None
    if local_reduce is None:
        return None
    return QuackLocalReduceInfo(
        view_node=local_reduce.view_node,
        reduce_op_node=local_reduce.reduce_op_node,
        source_node=local_reduce.source_node,
        keepdim=local_reduce.keepdim,
        group_size=local_reduce.group_size,
        dim=local_reduce.dim,
        feeds_main=local_reduce.feeds_main,
        kind=local_reduce.kind,
        scale=local_reduce.scale,
        max_power=local_reduce.max_power,
        aux_output_node=node,
        producer_skip_nodes=local_reduce.producer_skip_nodes,
    )


def split_scalar_scale(node: Any) -> tuple[Any, float] | None:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    target = node.target
    args = node.args
    if target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar, operator.mul):
        if len(args) < 2:
            return None
        lhs, rhs = args[:2]
        if isinstance(rhs, (int, float)):
            return lhs, float(rhs)
        if isinstance(lhs, (int, float)):
            return rhs, float(lhs)
        return None
    if target in (
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        operator.truediv,
    ):
        if len(args) < 2:
            return None
        lhs, rhs = args[:2]
        if not isinstance(rhs, (int, float)) or rhs == 0:
            return None
        return lhs, 1.0 / float(rhs)
    return None


def match_local_n_amax_scale_view(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    aux_output_node = node
    extra_skip_nodes: set[torch.fx.Node] = set()
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target
        in (
            torch.ops.aten._to_copy.default,
            torch.ops.prims.convert_element_type.default,
        )
    ):
        extra_skip_nodes.add(node)
        node = node.args[0]
    aux_view = match_view_or_reshape(node)
    if aux_view is None:
        return None
    extra_skip_nodes.add(aux_view.node)
    if isinstance(aux_view.base, torch.fx.Node):
        extra_skip_nodes.add(aux_view.base)
    scaled = split_scalar_scale(aux_view.base)
    if scaled is None:
        # Also accept the unscaled canonical keepdim form:
        #   scale = x.abs().amax(-1, keepdim=True)
        #   return main, scale.view(M, -1)
        reduce_node = aux_view.base
        scale = 1.0
    else:
        reduce_node, scale = scaled
    reduce_match = match_tensorssa_reduce(reduce_node)
    if reduce_match is None or reduce_match.kind != "amax":
        return None
    if reduce_match.dims not in ((-1,), (2,)) or reduce_match.dtype is not None:
        return None
    if not bool(reduce_match.keepdim):
        return None
    abs_node = reduce_match.input_node
    if not is_abs_node(abs_node):
        return None
    view_match = match_view_or_reshape(abs_node.args[0])
    if view_match is None:
        return None
    source_node = match_acc_source(view_match.base, mm_node)
    epilogue_reduce_source_node = None
    if source_node is None:
        if not output_uses_node(view_match.base, mm_node):
            return None
        source_node = view_match.base
        epilogue_reduce_source_node = view_match.node
    grouped_shape = grouped_n_fragment_shape(normalize_shape(view_match.shape))
    if not is_same_fragment_n_group_shape(grouped_shape):
        return None
    aux_meta = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    mm_meta = mm_node.meta.get("val")
    if aux_meta is None or mm_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (
        *tuple(mm_meta.shape[:-1]),
        mm_meta.shape[-1] // grouped_shape[-1],
    )
    if tuple(aux_meta.shape) != tuple(expected_shape):
        return None
    producer_skip_nodes = extra_skip_nodes | {abs_node}
    producer_skip_nodes.discard(aux_output_node)
    producer_skip_nodes.discard(reduce_node)
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_op_node=reduce_node,
        source_node=source_node,
        keepdim=False,
        group_size=grouped_shape[-1],
        dim=QUACK_REDUCE_DIM_N,
        kind="amax_abs",
        scale=scale,
        aux_output_node=aux_output_node,
        producer_skip_nodes=frozenset(producer_skip_nodes),
        epilogue_reduce_source_node=epilogue_reduce_source_node,
        epilogue_reduce_value_node=aux_view.base,
    )


def match_local_n_primitive_scale_view(
    node: Any,
    mm_node: torch.fx.Node,
    *,
    target: Any,
    kind: str,
    max_power: int = 8,
) -> QuackLocalReduceInfo | None:
    aux_view = match_view_or_reshape(node)
    if aux_view is None:
        return None
    scale_node = aux_view.base
    if not (
        isinstance(scale_node, torch.fx.Node)
        and scale_node.op == "call_function"
        and scale_node.target == target
    ):
        return None
    op_max_power = max_power
    if target == torch.ops.flex_gemm.mx_e8m0_scale.default:
        op_max_power = (
            scale_node.args[1]
            if len(scale_node.args) > 1
            else scale_node.kwargs.get("max_power", max_power)
        )
        if not isinstance(op_max_power, int):
            return None
    reduce_node = scale_node.args[0]
    if not isinstance(reduce_node, torch.fx.Node):
        return None
    reduce_match = match_tensorssa_reduce(reduce_node)
    if reduce_match is None or reduce_match.kind != "amax":
        return None
    if reduce_match.dims not in ((-1,), (2,)) or reduce_match.dtype is not None:
        return None
    if not bool(reduce_match.keepdim):
        return None
    abs_node = reduce_match.input_node
    if not is_abs_node(abs_node):
        return None
    view_match = match_view_or_reshape(abs_node.args[0])
    if view_match is None:
        return None
    source_node = match_acc_source(view_match.base, mm_node)
    if source_node is None:
        return None
    grouped_shape = grouped_n_fragment_shape(normalize_shape(view_match.shape))
    if not is_same_fragment_n_group_shape(grouped_shape):
        return None
    aux_meta = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    mm_meta = mm_node.meta.get("val")
    if aux_meta is None or mm_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (
        *tuple(mm_meta.shape[:-1]),
        mm_meta.shape[-1] // grouped_shape[-1],
    )
    if tuple(aux_meta.shape) != tuple(expected_shape):
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_op_node=reduce_node,
        source_node=source_node,
        keepdim=False,
        group_size=grouped_shape[-1],
        dim=QUACK_REDUCE_DIM_N,
        kind=kind,
        max_power=op_max_power,
        aux_output_node=aux_view.node,
        producer_skip_nodes=frozenset((scale_node, abs_node)),
    )


def match_local_n_mx_scale_view(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    return match_local_n_primitive_scale_view(
        node,
        mm_node,
        target=torch.ops.flex_gemm.mx_e8m0_scale.default,
        kind="mx_e8m0_scale",
    )


def match_local_n_nvfp4_scale_view(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    return match_local_n_primitive_scale_view(
        node,
        mm_node,
        target=torch.ops.flex_gemm.nvfp4_e4m3_scale.default,
        kind="nvfp4_e4m3_scale",
    )


def match_local_n_reduce(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    sum_match = match_sum(node, allow_method_sum=True)
    if sum_match is None:
        return None
    if sum_match.dims not in ((-1,), (2,)) or sum_match.dtype is not None:
        return None
    view_match = match_view_or_reshape(sum_match.view_node)
    if view_match is None:
        return None
    source_node = match_acc_source(view_match.base, mm_node)
    if source_node is None:
        return None
    shape = normalize_shape(view_match.shape)
    group_shape = grouped_n_fragment_shape(shape)
    if not is_n_group_shape(group_shape):
        return None
    group_size = group_shape[-1]
    mm_meta = mm_node.meta.get("val")
    reduce_meta = sum_match.node.meta.get("val")
    if mm_meta is None or reduce_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (
        (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // group_size, 1)
        if bool(sum_match.keepdim)
        else (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // group_size)
    )
    if tuple(reduce_meta.shape) != expected_shape:
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_op_node=sum_match.node,
        aux_output_node=sum_match.node,
        source_node=source_node,
        keepdim=bool(sum_match.keepdim),
        group_size=group_size,
        dim=QUACK_REDUCE_DIM_N,
    )


def match_local_m_reduce(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    sum_match = match_sum(node, allow_method_sum=False)
    if sum_match is None:
        return None
    if sum_match.dims not in ((1,), (2,)) or sum_match.dtype is not None:
        return None
    view_match = match_view_or_reshape(sum_match.view_node)
    if view_match is None:
        return None
    source_node = match_acc_source(view_match.base, mm_node)
    if source_node is None:
        return None
    shape = normalize_shape(view_match.shape)
    if not isinstance(shape, (list, tuple)) or len(shape) not in (3, 4):
        return None
    group_dim = 1 if len(shape) == 3 else 2
    if sum_match.dims != (group_dim,):
        return None
    if shape[-3] != -1 or not isinstance(shape[-2], int) or shape[-2] <= 0:
        return None
    mm_val = mm_node.meta.get("val")
    reduce_val = sum_match.node.meta.get("val")
    if mm_val is None or reduce_val is None:
        return None
    mm_shape = tuple(mm_val.shape)
    if len(mm_shape) not in (2, 3) or shape[-1] != mm_shape[-1]:
        return None
    group_size = shape[-2]
    expected_shape = (
        (*mm_shape[:-2], mm_shape[-2] // group_size, 1, mm_shape[-1])
        if sum_match.keepdim
        else (*mm_shape[:-2], mm_shape[-2] // group_size, mm_shape[-1])
    )
    if tuple(reduce_val.shape) != expected_shape:
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_op_node=sum_match.node,
        aux_output_node=sum_match.node,
        source_node=source_node,
        keepdim=bool(sum_match.keepdim),
        group_size=group_size,
        dim=QUACK_REDUCE_DIM_M,
    )


def match_reciprocal_node(node: Any) -> Any | None:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target in (torch.ops.aten.reciprocal.default, torch.reciprocal):
        return node.args[0]
    if node.target in (
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        operator.truediv,
    ):
        lhs, rhs = node.args[:2]
        if is_scalar_one(lhs):
            return rhs
    return None


def match_local_norm(
    node: Any,
    mm_node: torch.fx.Node,
    reduce_matcher: Callable[[Any, torch.fx.Node], QuackLocalReduceInfo | None],
    *,
    dim: int,
) -> QuackLocalNormInfo | None:
    output_view = match_view_or_reshape(node)
    if output_view is None:
        return None
    div_node = output_view.base
    output_shape = normalize_shape(output_view.shape)
    if not isinstance(div_node, torch.fx.Node):
        return None
    if div_node.op != "call_function":
        return None
    extra_skip_nodes: set[torch.fx.Node] = set()
    if div_node.target == torch.ops.aten.div.Tensor:
        lhs, rhs = div_node.args[:2]
    elif div_node.target in (
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        operator.mul,
    ):
        lhs, rhs = div_node.args[:2]
        reciprocal_source = match_reciprocal_node(rhs)
        if reciprocal_source is None:
            lhs, rhs = rhs, lhs
            reciprocal_source = match_reciprocal_node(rhs)
        if reciprocal_source is None or not isinstance(rhs, torch.fx.Node):
            return None
        extra_skip_nodes.add(rhs)
        rhs = reciprocal_source
    else:
        return None
    local_reduce = reduce_matcher(rhs, mm_node)
    if local_reduce is None or lhs is not local_reduce.view_node:
        return None
    output_meta = node.meta.get("val")
    mm_meta = mm_node.meta.get("val")
    if not isinstance(output_shape, (list, tuple)):
        return None
    if mm_meta is not None and tuple(output_shape) != tuple(mm_meta.shape):
        return None
    if output_meta is not None and mm_meta is not None:
        if tuple(output_meta.shape) != tuple(mm_meta.shape):
            return None
    if not local_reduce.keepdim:
        return None
    return QuackLocalNormInfo(
        output_node=node,
        div_node=div_node,
        view_node=local_reduce.view_node,
        reduce_node=local_reduce.reduce_op_node,
        source_node=local_reduce.source_node,
        group_size=local_reduce.group_size,
        dim=dim,
        extra_skip_nodes=frozenset(extra_skip_nodes),
    )


def output_uses_node(value: Any, needle: torch.fx.Node) -> bool:
    seen: set[torch.fx.Node] = set()

    def visit(item: Any) -> bool:
        if item is needle:
            return True
        if not isinstance(item, torch.fx.Node) or item in seen:
            return False
        seen.add(item)
        return any(visit(arg) for arg in pytree.tree_leaves((item.args, item.kwargs)))

    return visit(value)


def match_grouped_n_split(
    node: Any, mm_node: torch.fx.Node
) -> tuple[torch.fx.Node, int, tuple[str, ...]] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if not (
        node.op == "call_function"
        and node.target == torch.ops.aten.split.Tensor
        and len(node.args) >= 3
        and isinstance(node.args[1], int)
        and node.args[2] in (-1, 1)
    ):
        return None
    source = node.args[0]
    if source is not mm_node:
        return None
    mm_meta = mm_node.meta.get("val")
    if mm_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    if mm_meta.shape[-1] != 2 * node.args[1]:
        return None
    return node, 2, ("B",)


def match_grouped_n_getitem(
    node: Any, mm_node: torch.fx.Node
) -> tuple[torch.fx.Node, int, int, tuple[str, ...]] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if not (
        node.op == "call_function"
        and node.target == operator.getitem
        and len(node.args) >= 2
        and isinstance(node.args[1], int)
    ):
        return None
    split = match_grouped_n_split(node.args[0], mm_node)
    if split is None:
        return None
    split_node, group_size, concat_layout = split
    if not (0 <= node.args[1] < group_size):
        return None
    return split_node, node.args[1], group_size, concat_layout


def match_grouped_n_select(
    node: Any, mm_node: torch.fx.Node
) -> tuple[torch.fx.Node, int, int, tuple[str, ...]] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if not (
        node.op == "call_function"
        and node.target == torch.ops.aten.select.int
        and len(node.args) >= 3
        and isinstance(node.args[2], int)
    ):
        return None
    view = match_view_or_reshape(node.args[0])
    view_shape = normalize_shape(view.shape) if view is not None else None
    if view is None or not output_uses_node(view.base, mm_node):
        return None
    if node.args[1] == -1 or (
        isinstance(view_shape, (list, tuple)) and node.args[1] == len(view_shape) - 1
    ):
        shape = grouped_n_fragment_shape(view_shape)
        if not (
            is_same_fragment_n_group_shape(shape) and 0 <= node.args[2] < shape[-1]
        ):
            return None
        return view.node, node.args[2], shape[-1], ()
    if (
        isinstance(view_shape, (list, tuple))
        and len(view_shape) == 3
        and view_shape[1] == 2
        and node.args[1] == 1
        and 0 <= node.args[2] < 2
    ):
        # This is intentionally a bit brittle: the epilogue is telling us that
        # B is laid out as concat [gate; up], so request QUACK's concat_layout
        # and reinterpret the accumulator as the interleaved grouped-N form.
        return view.node, node.args[2], 2, ("B",)
    return None


def uses_grouped_m_select(value: Any, mm_node: torch.fx.Node) -> bool:
    seen: set[torch.fx.Node] = set()

    def visit(node: Any) -> bool:
        if not isinstance(node, torch.fx.Node) or node in seen:
            return False
        seen.add(node)
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.select.int
            and len(node.args) >= 3
            and node.args[1] == 1
        ):
            view = match_view_or_reshape(node.args[0])
            shape = normalize_shape(view.shape) if view is not None else None
            if (
                view is not None
                and output_uses_node(view.base, mm_node)
                and isinstance(shape, (list, tuple))
                and len(shape) == 3
                and shape[0] == -1
                and isinstance(shape[1], int)
                and shape[1] > 0
            ):
                return True
        return any(visit(arg) for arg in pytree.tree_leaves((node.args, node.kwargs)))

    return visit(value)


def match_grouped_n_contract_main(
    output_value: Any, mm_node: torch.fx.Node
) -> QuackMainOutputTransformInfo | None:
    select_view = None
    select_group = None
    concat_layout: tuple[str, ...] = ()
    saw_select = False
    seen: set[torch.fx.Node] = set()

    def visit(value: Any) -> bool:
        nonlocal saw_select, select_view, select_group, concat_layout
        if not isinstance(value, torch.fx.Node):
            return True
        if value in seen:
            return True
        seen.add(value)
        select = match_grouped_n_select(value, mm_node)
        if select is None:
            select = match_grouped_n_getitem(value, mm_node)
        if select is not None:
            view_node, _index, group_size, select_concat_layout = select
            if select_view is None:
                select_view = view_node
                select_group = group_size
                concat_layout = select_concat_layout
            elif (
                not fx_equivalent(select_view, view_node)
                or select_group != group_size
                or concat_layout != select_concat_layout
            ):
                return False
            saw_select = True
            return True
        if value is mm_node:
            return False
        if match_view_or_reshape(value) is not None and output_uses_node(
            value, mm_node
        ):
            return False
        return all(visit(arg) for arg in pytree.tree_leaves((value.args, value.kwargs)))

    if not visit(output_value) or not saw_select:
        return None
    if select_group not in (2, 4):
        raise NotImplementedError(
            "QUACK grouped_n_contract currently supports only groups 2 and 4; "
            f"group={select_group} needs a validated epilogue store layout"
        )
    mm_meta = mm_node.meta.get("val")
    output_meta = (
        output_value.meta.get("val")
        if isinstance(output_value, torch.fx.Node)
        else None
    )
    if mm_meta is None or output_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // select_group)
    if tuple(output_meta.shape) != expected_shape:
        return None
    return QuackMainOutputTransformInfo(
        kind="grouped_n_contract",
        group_size=select_group,
        concat_layout=concat_layout,
    )


def analyze_output(output_value: Any, mm_node: torch.fx.Node) -> QuackOutputPlan:
    local_reduce = None
    aux_output = None
    local_norm = match_local_norm(
        output_value, mm_node, match_local_n_reduce, dim=QUACK_REDUCE_DIM_N
    ) or match_local_norm(
        output_value, mm_node, match_local_m_reduce, dim=QUACK_REDUCE_DIM_M
    )
    skip_nodes: set[torch.fx.Node] = set()
    if local_norm is not None and local_norm.dim == 0:
        skip_nodes.update(
            (
                local_norm.output_node,
                local_norm.div_node,
                local_norm.view_node,
                local_norm.reduce_node,
                *local_norm.extra_skip_nodes,
            )
        )
        local_reduce = QuackLocalReduceInfo(
            view_node=local_norm.view_node,
            reduce_op_node=local_norm.reduce_node,
            aux_output_node=local_norm.reduce_node,
            source_node=local_norm.source_node,
            keepdim=True,
            group_size=local_norm.group_size,
            dim=QUACK_REDUCE_DIM_M,
            feeds_main=True,
        )
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        elif len(output_value) == 2:
            aux_value = output_value[1]
            local_reduce = (
                match_local_n_reduce(aux_value, mm_node)
                or match_local_m_reduce(aux_value, mm_node)
                or match_local_n_amax_reduce(aux_value, mm_node)
                or match_scaled_local_n_amax_reduce(aux_value, mm_node)
                or match_local_n_amax_scale_view(aux_value, mm_node)
                or match_local_n_mx_scale_view(aux_value, mm_node)
                or match_local_n_nvfp4_scale_view(aux_value, mm_node)
            )
            if local_reduce is not None and not local_reduce.keepdim:
                if (
                    local_reduce.epilogue_reduce_value_node is not None
                    and output_uses_node(
                        output_value[0], local_reduce.epilogue_reduce_value_node
                    )
                ):
                    local_reduce = replace(
                        local_reduce,
                        kind="copy",
                        scale=1.0,
                        epilogue_reduce_source_node=local_reduce.epilogue_reduce_value_node,
                    )
                if not output_uses_node(output_value[0], local_reduce.aux_output_node):
                    skip_nodes.add(local_reduce.aux_output_node)
                if not output_uses_node(output_value[0], local_reduce.reduce_op_node):
                    skip_nodes.add(local_reduce.reduce_op_node)
                for skip_node in local_reduce.producer_skip_nodes:
                    if not output_uses_node(output_value[0], skip_node):
                        skip_nodes.add(skip_node)
                if not output_uses_node(output_value[0], local_reduce.view_node):
                    skip_nodes.add(local_reduce.view_node)
                    if local_reduce.source_node is not mm_node:
                        skip_nodes.add(local_reduce.source_node)
            elif isinstance(aux_value, torch.fx.Node):
                aux_meta = aux_value.meta.get("val")
                mm_meta = mm_node.meta.get("val")
                if aux_meta is None or mm_meta is None:
                    raise NotImplementedError(
                        "QUACK generic aux tuple epilogues require fake tensor metadata"
                    )
                if tuple(aux_meta.shape) != tuple(mm_meta.shape):
                    raise NotImplementedError(
                        "QUACK generic aux tuple epilogues currently require the aux output "
                        "shape to match the GEMM output shape"
                    )
                aux_output = QuackAuxOutputInfo(output_value=aux_value)
            else:
                raise NotImplementedError(
                    "QUACK tuple epilogue currently supports only a supported local-reduce "
                    "aux output or one same-shape generic aux expression"
                )
            output_value = output_value[0]
        else:
            raise NotImplementedError(
                "QUACK GEMM epilogue backend expects one output or one supported "
                "local-reduce aux output"
            )
    main_output_transform = match_grouped_n_contract_main(output_value, mm_node)
    if main_output_transform is None and isinstance(output_value, torch.fx.Node):
        mm_meta = mm_node.meta.get("val")
        output_meta = output_value.meta.get("val")
        if (
            mm_meta is not None
            and output_meta is not None
            and tuple(output_meta.shape) != tuple(mm_meta.shape)
        ):
            if uses_grouped_m_select(output_value, mm_node):
                raise NotImplementedError(
                    "QUACK M-mode shape-changing main epilogues such as "
                    "acc.view(-1, group_m, N)[:, i, :] are not supported yet"
                )
            raise NotImplementedError(
                "QUACK shape-changing main epilogues currently require a supported "
                "local shape transform such as acc.view(M, -1, 2)[..., i]"
            )
    return QuackOutputPlan(
        output_value=output_value,
        skip_nodes=frozenset(skip_nodes),
        local_reduce=local_reduce,
        local_norm=local_norm,
        aux_output=aux_output,
        main_output_transform=main_output_transform,
    )


def grouped_n_fragment_shape(shape: Any) -> Any:
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        return shape[-3:]
    return shape


def is_n_group_shape(shape: Any) -> bool:
    return (
        isinstance(shape, (list, tuple))
        and len(shape) == 3
        and shape[-2] == -1
        and isinstance(shape[-1], int)
        and shape[-1] > 0
    )


def is_same_fragment_n_group_shape(shape: Any) -> bool:
    return is_n_group_shape(shape) and QUACK_TENSORSSA_FRAGMENT_N % shape[-1] == 0


def is_concat_half_n_shape(shape: Any) -> bool:
    return isinstance(shape, (list, tuple)) and len(shape) == 3 and shape[1] == 2


def is_nonnegative_expr(node: Any) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function" and node.target in (
        torch.ops.aten.abs.default,
        torch.abs,
        torch.ops.aten.relu.default,
        torch.relu,
    ):
        return True
    if node.op == "call_method" and node.target in ("abs", "relu"):
        return True
    if node.op == "call_function" and node.target in (
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    ):
        return is_nonnegative_expr(node.args[0])
    if node.op == "call_function" and node.target == torch.ops.aten.clamp.default:
        min_value = node.kwargs.get("min", node.args[1] if len(node.args) > 1 else None)
        return isinstance(min_value, (int, float)) and min_value >= 0
    if node.op == "call_method" and node.target == "clamp":
        min_value, _ = method_clamp_bounds(node)
        return isinstance(min_value, (int, float)) and min_value >= 0
    return False


def is_scalar_value(value: Any, expected: float, *, tolerance: float = 1e-12) -> bool:
    return isinstance(value, (int, float)) and abs(float(value) - expected) <= tolerance


def is_scalar_one(value: Any) -> bool:
    return is_scalar_value(value, 1.0)


def fx_equivalent(lhs: Any, rhs: Any, *, depth: int = 4) -> bool:
    if lhs is rhs:
        return True
    if (
        depth <= 0
        or not isinstance(lhs, torch.fx.Node)
        or not isinstance(rhs, torch.fx.Node)
    ):
        return False
    if lhs.op != rhs.op or lhs.target != rhs.target or lhs.kwargs != rhs.kwargs:
        return False
    if lhs.op == "get_attr":
        return True
    if len(lhs.args) != len(rhs.args):
        return False
    return all(
        fx_equivalent(left_arg, right_arg, depth=depth - 1)
        if isinstance(left_arg, torch.fx.Node) or isinstance(right_arg, torch.fx.Node)
        else left_arg == right_arg
        for left_arg, right_arg in zip(lhs.args, rhs.args)
    )


def match_mul_scalar(node: Any, scalar: float) -> torch.fx.Node | None:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target not in (
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        operator.mul,
    ):
        return None
    lhs, rhs = node.args[:2]
    if isinstance(lhs, torch.fx.Node) and is_scalar_value(rhs, scalar):
        return lhs
    if isinstance(rhs, torch.fx.Node) and is_scalar_value(lhs, scalar):
        return rhs
    return None


def match_negated_node(value: Any, source: torch.fx.Node) -> bool:
    if not isinstance(value, torch.fx.Node) or value.op != "call_function":
        return False
    if value.target in (torch.ops.aten.neg.default, operator.neg):
        return value.args[0] is source
    if value.target in (
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        operator.mul,
    ):
        return bool(
            (value.args[0] is source and len(value.args) > 1 and value.args[1] == -1)
            or (value.args[1] is source and value.args[0] == -1)
        )
    return False
