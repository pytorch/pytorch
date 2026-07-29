# mypy: allow-untyped-defs
"""Lower FlexGEMM grouped local reductions into QuACK/CuTeDSL epilogues.

FlexGEMM recognizes a narrow local-reduction contract inside the GEMM output
tile: an epilogue reshapes the accumulator to expose contiguous groups along M
or N, then reduces only that grouped dimension. N-axis groups up to one 32-lane
fragment lower as ordinary in-fragment TensorSSA reductions; larger N groups
produce TensorSSA partials that QuACK combines physically.
M-axis groups currently always use QuACK's physical row-lane/warp combine path,
even when the group is small enough to fit in one fragment. Inductor owns the
FX pattern matching and output contracts; these helpers describe the supported
TensorSSA shapes and generated combine/finalize expressions QuACK needs.

The main caller is ``materialize_flex_gemm_epilogue`` in ``epilogue.py``.
FlexGEMM lowering first calls ``analyze_flex_gemm_epilogue``, which uses this
module's layout and reduction-recognition helpers. Materialization then routes
FX nodes through ``lower_view_or_reshape``, ``lower_prepare_softmax_online``,
and ``lower_tensorssa_reduce`` to emit CuTeDSL source.
"""

import dataclasses
import math
import operator
from typing import Any, cast

import torch
from torch._inductor import inductor_prims
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLOpOverrides,
    tensorssa_reduction,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_GROUPED_RESHAPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    local_reduce_needs_physical_callbacks,
    LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR,
    statically_known_equal,
)
from torch._inductor.ops_handler import ReductionType
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


def normalize_shape(shape: Any) -> Any:
    return tuple(shape) if isinstance(shape, (list, tuple, torch.Size)) else shape


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSALayout:
    """Describe a grouped M/N TensorSSA view inside the generated epilogue.

    Attributes:
        axis: GEMM output dimension being grouped: 0 for M, 1 for N.
        group_size: Number of contiguous output elements reduced as one group.
    """

    axis: int
    group_size: int

    @property
    def reduce_dims(self) -> tuple[int, ...]:
        return (-1, 2) if self.axis == 1 else (-2, 1)

    def matches_reduction_dim(self, dim: Any) -> bool:
        """Return whether an FX reduction selects this layout's grouped dimension."""
        dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
        return len(dims) == 1 and dims[0] in self.reduce_dims

    def fragment_group_size_expr(self, source: Any) -> str:
        """Return the local group size available in this epilogue fragment."""
        return (
            f"cutlass.const_expr(min({self.group_size}, "
            f"cute.size({source}.shape, mode=[0])))"
        )

    def fragment_repeat_expr(self, source: Any) -> str:
        """Return the repeat count needed to cover the current epilogue fragment."""
        return (
            f"cutlass.const_expr(cute.size({source}.shape, mode=[0]) "
            f"// min({self.group_size}, cute.size({source}.shape, mode=[0])))"
        )

    def tensorssa_shape(self, source: Any) -> str:
        fragment_group_size = self.fragment_group_size_expr(source)
        repeats = self.fragment_repeat_expr(source)
        if self.axis == 1:
            return f"((1, {fragment_group_size}, {repeats}), 1, 1)"
        return f"(({fragment_group_size}, 1, {repeats}), 1, 1)"

    def keepdim_shape(self, source: Any) -> str:
        return f"((1, 1, {self.fragment_repeat_expr(source)}), 1, 1)"

    @property
    def needs_physical_combine(self) -> bool:
        return local_reduce_needs_physical_callbacks(self.axis, self.group_size)

    @property
    def reduction_profile(self) -> str:
        if self.axis == 1:
            return "((None, 1, None), 1, 1)"
        return "((1, None, None), 1, 1)"


def _syntactic_grouped_tensor_layout(
    shape: tuple[Any, ...],
) -> GroupedTensorSSALayout | None:
    """Match grouped-reshape syntax before validating source geometry."""
    if len(shape) not in (3, 4):
        return None
    if isinstance(shape[-1], int) and shape[-1] > 0 and shape[-2] == -1:
        return GroupedTensorSSALayout(axis=1, group_size=shape[-1])
    if shape[-3] == -1 and isinstance(shape[-2], int) and shape[-2] > 0:
        return GroupedTensorSSALayout(axis=0, group_size=shape[-2])
    return None


def _group_count_matches_selected_dim(
    group_count: Any, selected_size: Any, group: int
) -> bool:
    match group_count:
        case -1:
            return True
        case _:
            return statically_known_equal(
                group_count * group, selected_size
            ) or statically_known_equal(group_count, selected_size // group)


def _grouped_layout_matches_source_shape(
    shape: tuple[Any, ...],
    source_shape: tuple[Any, ...],
    layout: GroupedTensorSSALayout,
) -> bool:
    """Require a 2-D GEMM output reshape to split exactly M or N."""
    if len(shape) != 3:
        return False

    m, n = source_shape
    match layout.axis, shape:
        case 1, (kept_m, group_count, group) if group == layout.group_size:
            return statically_known_equal(
                kept_m, m
            ) and _group_count_matches_selected_dim(group_count, n, group)
        case 0, (group_count, group, kept_n) if group == layout.group_size:
            return statically_known_equal(
                kept_n, n
            ) and _group_count_matches_selected_dim(group_count, m, group)
        case _:
            return False


def grouped_tensor_layout(
    shape: Any, source_shape: Any | None = None
) -> GroupedTensorSSALayout | None:
    """Recognize exact grouped M/N reshapes for the local-reduction contract."""
    shape = normalize_shape(shape)
    if not isinstance(shape, tuple):
        return None
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
        shape = normalize_shape(shape[0])
    if source_shape is not None:
        source_shape = normalize_shape(source_shape)
        if isinstance(source_shape, tuple) and len(source_shape) == 2:
            candidates = []
            match shape:
                case (*_, int(group)) if group > 0:
                    candidates.append(GroupedTensorSSALayout(axis=1, group_size=group))
            match shape:
                case (*_, int(group), _) if group > 0:
                    candidates.append(GroupedTensorSSALayout(axis=0, group_size=group))
            for layout in candidates:
                if _grouped_layout_matches_source_shape(shape, source_shape, layout):
                    return layout
            if _syntactic_grouped_tensor_layout(shape) is not None:
                raise NotImplementedError(LOCAL_REDUCE_GROUPED_RESHAPE_ERROR)
            return None
    return _syntactic_grouped_tensor_layout(shape)


def _cute_op_name(target: Any) -> str | None:
    if isinstance(target, torch._ops.OpOverload):
        op_name = target.overloadpacket.__name__
    elif isinstance(target, str):
        op_name = target
    else:
        op_name = target.__name__ if callable(target) else None
    return "truediv" if op_name == "div" else op_name


@dataclasses.dataclass(frozen=True)
class FlexGemmPhysicalReduction:
    """Describe QuACK's physical combine/finalize callback for local reductions."""

    combine_expr: str
    finalize_expr: str = "value"


FLEX_GEMM_POINTWISE_OP_NAMES = frozenset(
    (
        "_to_copy",
        "clamp",
        "clamp_max",
        "clamp_min",
        "convert_element_type",
    )
)


def _cute_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    """Translate FX node references and constants into CuTeDSL epilogue values."""
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return 'float("nan")'
        return 'float("inf")' if value > 0 else 'float("-inf")'
    if isinstance(
        value,
        (
            int,
            float,
            bool,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported FlexGEMM epilogue constant: {value!r}")


def _generate_like(
    kernel: Any, expr: str, ref: Any, shape_ref: Any | None = None
) -> Any:
    """Emit CuTeDSL while preserving dtype and shape metadata from references."""
    if shape_ref is None:
        shape_ref = ref
    return kernel.cse.generate(
        kernel.body,
        expr,
        dtype=getattr(ref, "dtype", None),
        shape=getattr(shape_ref, "shape", None),
    )


def _keepdim_and_broadcast(
    kernel: Any, reduced: Any, layout: GroupedTensorSSALayout, source: Any
) -> tuple[Any, Any]:
    """Materialize keepdim and store-shaped forms of a grouped reduction."""
    keepdim_source = _generate_like(
        kernel, f"{reduced}.reshape({layout.keepdim_shape(source)})", reduced
    )
    return keepdim_source, _generate_like(
        kernel, f"{keepdim_source}.broadcast_to({source}.shape)", keepdim_source, source
    )


def _cute_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    op_name = _cute_op_name(target)
    if op_name is None:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")
    try:
        op = getattr(V.get_ops_handler(), op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue op: {target}"
        ) from None
    return op(*args, **kwargs)


def _local_reduce_store_arg(
    value: Any, env: dict[torch.fx.Node, Any], sources: dict[torch.fx.Node, Any]
) -> Any:
    if isinstance(value, torch.fx.Node) and value in sources:
        return sources[value]
    if isinstance(value, (tuple, list)):
        return type(value)(
            _local_reduce_store_arg(item, env, sources) for item in value
        )
    return _cute_arg(value, env)


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


def iter_fx_node_inputs(value: Any):
    """Yield FX node inputs nested in args/kwargs-style containers."""
    result: list[torch.fx.Node] = []
    torch.fx.map_arg(value, lambda node: result.append(node))
    yield from result


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


def lower_view_or_reshape(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    active_grouped_layouts: OrderedSet[GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    preserve_value_layout: bool = False,
) -> Any | None:
    """Emit a view using grouped provenance from the shared FX analysis."""
    view_args = view_or_reshape_args(node)
    if view_args is None:
        return None
    source_node, _ = view_args
    if not isinstance(source_node, torch.fx.Node):
        return None
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return _cute_arg(source_node, env)
    source = _cute_arg(source_node, env)
    grouped_layout = grouped_tensors.get(node)
    if grouped_layout is not None:
        if preserve_value_layout or grouped_layout not in active_grouped_layouts:
            return source
        return _generate_like(
            kernel,
            f"{source}.reshape({grouped_layout.tensorssa_shape(source)})",
            source,
        )
    if source_node in grouped_tensors:
        return source
    return None


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


def reduction_from_node(node: torch.fx.Node) -> tuple[Any, Any, Any, Any, str] | None:
    if node.op != "call_function" or node.target not in FUNCTION_REDUCTION_TYPES:
        return None
    reduction_type, has_dtype = FUNCTION_REDUCTION_TYPES[node.target]
    input_node = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
    dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
    return input_node, dim, keepdim, dtype if has_dtype else None, reduction_type


def unsupported_reduction_from_node(node: torch.fx.Node) -> str | None:
    if node.op != "call_function" or node.target not in FUNCTION_UNSUPPORTED_REDUCTIONS:
        return None
    return str(node.target)


def lower_full_scalar(node: torch.fx.Node) -> Any | None:
    if node.op != "call_function" or node.target is not torch.ops.aten.full.default:
        return None
    shape = normalize_shape(node.args[0])
    if shape != ():
        return None
    value = node.args[1]
    return value if isinstance(value, (bool, int, float)) else None


def lower_squeeze(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    source_node = squeeze_source_node(node)
    if source_node is None or source_node not in env:
        return None
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
    return _cute_arg(source_node, env)


def lower_getitem(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if node.op != "call_function" or node.target is not operator.getitem:
        return None
    source_node, index = node.args
    if not isinstance(source_node, torch.fx.Node) or not isinstance(index, int):
        return None
    source = _cute_arg(source_node, env)
    if not isinstance(source, (tuple, list)) or not -len(source) <= index < len(source):
        return None
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node][
            index
        ]
    return source[index]


def lower_prepare_softmax_online(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if (
        node.op != "call_function"
        or node.target is not inductor_prims.prepare_softmax_online
    ):
        return None
    input_node = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if not isinstance(input_node, torch.fx.Node):
        return None
    if input_node not in grouped_tensors:
        raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
    layout = grouped_tensors[input_node]
    if layout.needs_physical_combine:
        raise NotImplementedError(
            "unsupported FlexGEMM physical local reduction: prepare_softmax_online "
            "needs a multi-value generated physical reducer"
        )
    if not layout.matches_reduction_dim(dim):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: prepare_softmax_online "
            "currently supports only the grouped dimension"
        )
    source = _cute_arg(input_node, env)
    max_reduced = _generate_like(
        kernel,
        f'{source}.reduce(cute.ReductionOp.MAX, init_val=float("-inf"), reduction_profile={layout.reduction_profile})',
        source,
    )
    _, max_store = _keepdim_and_broadcast(kernel, max_reduced, layout, source)
    centered = _generate_like(kernel, f"({source} - {max_store})", source)
    exp_centered = CuteDSLOpOverrides.exp(centered)
    sum_reduced = _generate_like(
        kernel,
        f"{exp_centered}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={layout.reduction_profile})",
        exp_centered,
    )
    _, sum_store = _keepdim_and_broadcast(kernel, sum_reduced, layout, source)
    local_reduce_store_sources[node] = (max_store, sum_store)
    return max_store, sum_store


def lower_tensorssa_reduce(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    local_reduce_physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction],
) -> Any | None:
    """Lower value reductions while deferring cross-fragment finalization to QuACK."""
    reduction = reduction_from_node(node)
    if reduction is None:
        return None
    input_node, dim, keepdim, dtype, reduction_type = reduction
    if dtype is not None:
        raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
    if not isinstance(input_node, torch.fx.Node):
        return None
    if input_node not in grouped_tensors:
        raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
    layout = grouped_tensors[input_node]
    if not layout.matches_reduction_dim(dim):
        raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
    reduction_name = cast(
        ReductionType, "sum" if reduction_type == "mean" else reduction_type
    )
    desc = tensorssa_reduction(reduction_name)
    finalize_expr = (
        f"value / {layout.group_size}.0" if reduction_type == "mean" else "value"
    )
    source = _cute_arg(input_node, env)
    needs_physical_combine = layout.needs_physical_combine
    if needs_physical_combine:
        local_reduce_physical_reductions[node] = FlexGemmPhysicalReduction(
            desc.combine_expr, finalize_expr
        )
        if layout.axis == 0:
            local_reduce_store_sources[node] = source
            return source
    reduced = _generate_like(
        kernel,
        f"{source}.reduce({desc.cute_op}, init_val={desc.init_val}, reduction_profile={layout.reduction_profile})",
        source,
    )
    if reduction_type == "mean" and not needs_physical_combine:
        reduced = _generate_like(
            kernel, f"{reduced} / {float(layout.group_size)!r}", reduced
        )
    keepdim_source, local_reduce_store_sources[node] = _keepdim_and_broadcast(
        kernel, reduced, layout, source
    )
    if keepdim:
        return keepdim_source
    return reduced
