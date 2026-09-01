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
materialization wraps accepted geometries in ``GroupedTensorSSALayout`` and
routes FX nodes through ``lower_view_or_reshape``,
``lower_prepare_softmax_online``, and ``lower_tensorssa_reduce``.
"""

import dataclasses
import math
from typing import Any, cast

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLOpOverrides,
    tensorssa_reduction,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmReductionGeometry,
    iter_fx_node_inputs,
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_utils import normalize_shape
from torch._inductor.ops_handler import ReductionType
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSALayout(GemmReductionGeometry):
    """Describe a grouped M/N TensorSSA view inside the generated epilogue."""

    swapped: bool = False

    @property
    def tensorssa_axis(self) -> int:
        return 1 - self.axis if self.swapped else self.axis

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
        if self.tensorssa_axis == 1:
            return f"((1, {fragment_group_size}, {repeats}), 1, 1)"
        return f"(({fragment_group_size}, 1, {repeats}), 1, 1)"

    def keepdim_shape(self, source: Any) -> str:
        return f"((1, 1, {self.fragment_repeat_expr(source)}), 1, 1)"

    @property
    def needs_physical_callbacks(self) -> bool:
        return GemmReductionGeometry(
            self.group, self.tensorssa_axis
        ).needs_physical_callbacks

    @property
    def needs_physical_combine(self) -> bool:
        return self.needs_physical_callbacks

    @property
    def reduction_profile(self) -> str:
        if self.tensorssa_axis == 1:
            return "((None, 1, None), 1, 1)"
        return "((1, None, None), 1, 1)"


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
        "inline_asm_elementwise",
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
            str,
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


def _cute_call(
    node: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Lower one FX call through the active CuTeDSL operations handler."""
    target = node.target
    op_name = _cute_op_name(target)
    if op_name is None:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")
    if op_name == "inline_asm_elementwise":
        # The HOP spells the asm text `asm_str`; the ops handler spells it `asm`.
        kwargs = dict(kwargs)
        kwargs["asm"] = kwargs.pop("asm_str")
        input_values = []
        for input_node in node.args[: len(args)]:
            value = (
                input_node.meta.get("val")
                if isinstance(input_node, torch.fx.Node)
                else None
            )
            if not isinstance(value, torch.Tensor):
                raise NotImplementedError(
                    "FlexGEMM inline asm inputs require tensor metadata"
                )
            input_values.append(value)
        kwargs["input_dtypes"] = tuple(value.dtype for value in input_values)
        kwargs["scalar_sources"] = tuple(
            all(isinstance(dim, int) and dim == 1 for dim in value.shape)
            for value in input_values
        )
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


def lower_view_or_reshape(
    node: torch.fx.Node,
    normalized: NormalizedView,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    active_grouped_layouts: OrderedSet[GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    preserve_value_layout: bool = False,
) -> Any | None:
    """Emit an analyzed view using grouped provenance."""
    source_node = normalized.source
    source = _cute_arg(source_node, env)
    grouped_layout = grouped_tensors.get(node)
    if (
        grouped_layout is not None
        and not preserve_value_layout
        and grouped_layout in active_grouped_layouts
    ):
        if source_node in local_reduce_store_sources:
            local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return _generate_like(
            kernel,
            f"{source}.reshape({grouped_layout.tensorssa_shape(source)})",
            source,
        )
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return source
    if grouped_layout is not None:
        return source
    if source_node in grouped_tensors:
        return source
    return None


def lower_grouped_n_split(
    node: torch.fx.Node,
    normalized: NormalizedSplit,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
) -> tuple[Any, ...] | None:
    """Split an analyzed output contraction into grouped TensorSSA values."""
    layout = grouped_tensors.get(node)
    if layout is None or layout.axis != 1:
        return None
    source = _cute_arg(normalized.source, env)
    grouped = _generate_like(
        kernel, f"{source}.reshape({layout.tensorssa_shape(source)})", source
    )
    return tuple(
        _generate_like(
            kernel,
            f"{grouped}[((0, {index}, None), None, None)]",
            grouped,
        )
        for index in range(layout.group)
    )


def lower_grouped_n_select(
    normalized: NormalizedSelect,
    index: int,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
) -> Any:
    """Select one analysis-validated output-contraction group value."""
    source = _cute_arg(normalized.source, env)
    return _generate_like(
        kernel,
        f"{source}[((0, {index}, None), None, None)]",
        source,
    )


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
    normalized: NormalizedSqueeze,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    """Forward an analyzed squeeze alias."""
    source_node = normalized.source
    if source_node not in env:
        return None
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
    return _cute_arg(source_node, env)


def lower_getitem(
    node: torch.fx.Node,
    normalized: NormalizedGetItem,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    """Index an analyzed aggregate value."""
    source_node, index = normalized.source, normalized.index
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
    normalized: NormalizedPrepareSoftmax,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any:
    """Lower analyzed online-softmax preparation."""
    input_node, dim = normalized.source, normalized.dim
    if input_node not in grouped_tensors:
        raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
    layout = grouped_tensors[input_node]
    if layout.needs_physical_callbacks:
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
    normalized: NormalizedReduction,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    local_reduce_physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction],
) -> Any:
    """Lower an analyzed reduction while deferring physical finalization."""
    input_node = normalized.source
    dim = normalized.dim
    keepdim = normalized.keepdim
    dtype = normalized.dtype
    reduction_type = normalized.reduction_type
    if dtype is not None:
        raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
    if input_node not in grouped_tensors:
        raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
    layout = grouped_tensors[input_node]
    if not layout.matches_reduction_dim(dim):
        raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
    reduction_name = cast(
        ReductionType, "sum" if reduction_type == "mean" else reduction_type
    )
    desc = tensorssa_reduction(reduction_name)
    finalize_expr = f"value / {layout.group}.0" if reduction_type == "mean" else "value"
    source = _cute_arg(input_node, env)
    needs_physical_callbacks = layout.needs_physical_callbacks
    if needs_physical_callbacks:
        local_reduce_physical_reductions[node] = FlexGemmPhysicalReduction(
            desc.combine_expr, finalize_expr
        )
        if layout.tensorssa_axis == 0:
            local_reduce_store_sources[node] = source
            return source
    reduced = _generate_like(
        kernel,
        f"{source}.reduce({desc.cute_op}, init_val={desc.init_val}, reduction_profile={layout.reduction_profile})",
        source,
    )
    if reduction_type == "mean" and not needs_physical_callbacks:
        reduced = _generate_like(
            kernel, f"{reduced} / {float(layout.group)!r}", reduced
        )
    keepdim_source, local_reduce_store_sources[node] = _keepdim_and_broadcast(
        kernel, reduced, layout, source
    )
    if keepdim:
        return keepdim_source
    return reduced
