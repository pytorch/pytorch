# mypy: allow-untyped-defs
import dataclasses
import hashlib
import operator
from typing import Any

import torch
from torch._inductor import inductor_prims
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    upcast_compute_type,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR,
    FLEX_GEMM_OUTPUT_TENSOR_ERROR,
    FlexGemmLocalReduceGeometry,
    grouped_reduce_dims_match,
    LOCAL_REDUCE_AUX_TENSORSSA_ERROR,
    LOCAL_REDUCE_COMBINE_FN_SUFFIX,
    local_reduce_compressed_shape,
    LOCAL_REDUCE_CONTRACT_NODE_ERROR,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_FEED_MAIN_MIXED_CONTRACT_ERROR,
    LOCAL_REDUCE_FINALIZE_FN_SUFFIX,
    LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_MIXED_CONTRACT_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR,
    LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR,
    LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR,
    LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR,
    local_reduce_unsupported_tensorssa_error,
    statically_known_shape_equal,
    validate_local_reduce_feed_main_capability,
    validate_local_reduce_group_axis,
    validate_local_reduce_tensorssa_group_size,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    _cute_arg,
    _cute_call,
    _local_reduce_store_arg,
    FlexGemmPhysicalReduction,
    grouped_tensor_layout,
    GroupedTensorSSAInfo,
    has_local_reduce_store_source,
    is_shape_preserving_pointwise_node,
    iter_fx_node_inputs,
    lower_full_scalar,
    lower_getitem,
    lower_prepare_softmax_online,
    lower_squeeze,
    lower_tensorssa_reduce,
    lower_view_or_reshape,
    propagate_grouped_tensorssa_info,
    reduction_from_node,
    squeeze_source_node,
    tensor_meta_shape,
    unsupported_reduction_from_node,
    view_or_reshape_args,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges


class FlexGemmCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class FlexGemmCuteDSLCSE:
    def __init__(self) -> None:
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class FlexGemmCuteDSLKernel:
    def __init__(self) -> None:
        self.body = FlexGemmCuteDSLBody()
        self.cse = FlexGemmCuteDSLCSE()


class FlexGemmCuteDSLOpOverrides(CuteDSLOpOverrides):
    # Aten add/sub carry alpha as schema sugar; CuTeDSL only needs the scaled RHS.
    @staticmethod
    def add(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.add(a, rhs)

    @staticmethod
    def sub(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.sub(a, rhs)

    @staticmethod
    def _to_copy(x: Any, *, dtype: torch.dtype, **kwargs: Any) -> Any:
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                "unsupported kwargs for FlexGEMM epilogue op _to_copy: "
                f"{unsupported_kwargs}"
            )
        return CuteDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = CuteDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = CuteDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return CuteDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return CuteDSLOpOverrides.minimum(x, max)

    @staticmethod
    def convert_element_type(x: Any, dtype: torch.dtype) -> Any:
        return CuteDSLOpOverrides.to_dtype(x, dtype)


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputLocalReducePlan:
    """Bind one local-reduce value to store and/or feed consumers."""

    geometry: FlexGemmLocalReduceGeometry
    value_node: torch.fx.Node
    store_node: torch.fx.Node | None = None
    feeds_main: bool = False

    def __post_init__(self) -> None:
        """Reject invalid local-reduce output nodes."""
        if (
            not isinstance(self.value_node, torch.fx.Node)
            or (
                self.store_node is not None
                and not isinstance(self.store_node, torch.fx.Node)
            )
            or (self.store_node is None and not self.feeds_main)
        ):
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)

    @property
    def group(self) -> int:
        return self.geometry.group

    @property
    def axis(self) -> int:
        return self.geometry.axis

    @property
    def needs_physical_callbacks(self) -> bool:
        return self.geometry.needs_physical_callbacks


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputPlan:
    """Classify the FlexGEMM body output into a main result and aux returns."""

    output: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce: FlexGemmOutputLocalReducePlan | None = None
    local_reduce_aux_index: int | None = None

    def __post_init__(self) -> None:
        """Reject plans that cannot bind FX tensor outputs downstream."""
        if not isinstance(self.output, torch.fx.Node) or not all(
            isinstance(aux_output, torch.fx.Node) for aux_output in self.aux_outputs
        ):
            raise RuntimeError(FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR)
        if self.local_reduce_aux_index is not None and self.local_reduce is None:
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceContract:
    aux: torch.fx.Node
    group: int
    axis: int

    def __post_init__(self) -> None:
        """Reject invalid local-reduce source nodes and geometry."""
        validate_local_reduce_group_axis(self.group, self.axis)
        if not isinstance(self.aux, torch.fx.Node):
            raise RuntimeError(LOCAL_REDUCE_CONTRACT_NODE_ERROR)

    def to_plan(
        self, *, store_node: torch.fx.Node | None, feeds_main: bool
    ) -> FlexGemmOutputLocalReducePlan:
        """Bind this reduction value to the requested local-reduce consumers."""
        return FlexGemmOutputLocalReducePlan(
            FlexGemmLocalReduceGeometry(self.group, self.axis),
            self.aux,
            store_node=store_node,
            feeds_main=feeds_main,
        )


@dataclasses.dataclass
class FlexGemmLocalReduceAnalysis:
    """Track grouped TensorSSA provenance and derived local-reduce contracts."""

    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo] = dataclasses.field(
        default_factory=dict
    )
    contracts: dict[torch.fx.Node, FlexGemmLocalReduceContract] = dataclasses.field(
        default_factory=dict
    )

    def bind_grouped_layout(self, node: torch.fx.Node, shape: Any, source: Any) -> bool:
        """Record reshapes that introduce grouped TensorSSA provenance."""
        source_shape = (
            tensor_meta_shape(source) if isinstance(source, torch.fx.Node) else None
        )
        layout = grouped_tensor_layout(shape, source_shape)
        if layout is None or not isinstance(source, torch.fx.Node):
            return False
        validate_local_reduce_tensorssa_group_size(layout.axis, layout.group_size)
        self.grouped_tensors[node] = GroupedTensorSSAInfo(layout)
        return True

    def bind_contract(
        self, node: torch.fx.Node, contract: FlexGemmLocalReduceContract | None
    ) -> bool:
        """Record a discovered local-reduce contract and report whether it matched."""
        if contract is None:
            return False
        self.contracts[node] = contract
        return True

    def copy_contract(self, node: torch.fx.Node, source: Any) -> bool:
        """Propagate a known reduction contract through structural FX wrappers."""
        if not isinstance(source, torch.fx.Node):
            return False
        contract = self.contracts.get(source)
        if contract is None:
            return False
        self.contracts[node] = contract
        return True

    def bind_grouped_reduction(
        self,
        node: torch.fx.Node,
        input_node: Any,
        dim: Any,
        dtype: Any = None,
        *,
        raise_invalid_dims: bool = True,
    ) -> bool:
        """Match reductions against grouped TensorSSA provenance and bind them."""
        return self.bind_contract(
            node,
            local_reduce_contract_from_grouped_input(
                node,
                input_node,
                dim,
                self.grouped_tensors,
                dtype,
                raise_invalid_dims=raise_invalid_dims,
            ),
        )

    def has_grouped_tensor(self, value: Any) -> bool:
        """Return whether a value currently has grouped TensorSSA provenance."""
        return isinstance(value, torch.fx.Node) and value in self.grouped_tensors

    def contract_for(self, node: torch.fx.Node) -> FlexGemmLocalReduceContract | None:
        """Return the discovered local-reduce contract for a node, if any."""
        return self.contracts.get(node)

    def contracts_from_inputs(self, *values: Any) -> list[FlexGemmLocalReduceContract]:
        """Collect known contracts from FX inputs for pointwise propagation."""
        return [
            self.contracts[arg]
            for arg in iter_fx_node_inputs(values)
            if arg in self.contracts
        ]

    def bind_pointwise_contract(
        self, node: torch.fx.Node, mixed_contract_error: str
    ) -> bool:
        """Propagate grouped provenance and local-reduce contracts through pointwise ops."""
        grouped_info = propagate_grouped_tensorssa_info(node, self.grouped_tensors)
        if grouped_info is not None:
            self.grouped_tensors[node] = grouped_info
        contract = common_local_reduce_contract(
            self.contracts_from_inputs(node.args, node.kwargs), mixed_contract_error
        )
        if contract is None:
            return False
        self.contracts[node] = FlexGemmLocalReduceContract(
            node, contract.group, contract.axis
        )
        return True


def fx_node_depends_on(
    value: Any,
    target: torch.fx.Node,
    seen: OrderedSet[torch.fx.Node] | None = None,
) -> bool:
    """Return whether an FX dependency tree reaches the target node."""
    if value is target:
        return True
    if not isinstance(value, torch.fx.Node):
        return any(
            fx_node_depends_on(arg, target, seen) for arg in iter_fx_node_inputs(value)
        )
    if seen is None:
        seen = OrderedSet()
    if value in seen:
        return False
    seen.add(value)
    return any(
        fx_node_depends_on(arg, target, seen)
        for arg in iter_fx_node_inputs((value.args, value.kwargs))
    )


def common_local_reduce_contract(
    contracts: list[FlexGemmLocalReduceContract],
    mixed_contract_error: str,
) -> FlexGemmLocalReduceContract | None:
    """Merge contracts that share one grouped reduction layout."""
    if not contracts:
        return None
    contract = contracts[0]
    if any(
        item.group != contract.group or item.axis != contract.axis for item in contracts
    ):
        raise NotImplementedError(mixed_contract_error)
    return contract


def common_local_reduce_value_contract(
    contracts: list[FlexGemmLocalReduceContract],
    mixed_contract_error: str,
) -> FlexGemmLocalReduceContract | None:
    """Merge contracts for one reusable physical reduction value."""
    contract = common_local_reduce_contract(contracts, mixed_contract_error)
    if contract is None:
        return None
    if any(item.aux is not contract.aux for item in contracts):
        raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
    return contract


def local_reduce_feed_value_contract(
    value: Any,
    grouped_source: torch.fx.Node,
    layout,
) -> FlexGemmLocalReduceContract | None:
    """Find the single grouped reduction that produces a broadcast value."""
    if not isinstance(value, torch.fx.Node):
        return None
    reduction = reduction_from_node(value)
    if reduction is not None:
        input_node, dim, keepdim, dtype, _ = reduction
        if input_node is not grouped_source:
            if fx_node_depends_on(input_node, grouped_source):
                raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        if (
            dtype is not None
            or not keepdim
            or not grouped_reduce_dims_match(dim, layout.reduce_dims)
        ):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        return FlexGemmLocalReduceContract(value, layout.group_size, layout.axis)
    if not is_shape_preserving_pointwise_node(value):
        return None
    contracts = [
        contract
        for arg in iter_fx_node_inputs((value.args, value.kwargs))
        for contract in (local_reduce_feed_value_contract(arg, grouped_source, layout),)
        if contract is not None
    ]
    return common_local_reduce_value_contract(
        contracts, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
    )


def local_reduce_contract_grouped_source(
    contract: FlexGemmLocalReduceContract,
) -> torch.fx.Node | None:
    """Return the grouped source that produced a physical feed-main contract."""
    reduction = reduction_from_node(contract.aux)
    if reduction is None:
        return None
    input_node = reduction[0]
    return input_node if isinstance(input_node, torch.fx.Node) else None


def has_physical_grouped_input(
    value: Any, seen: OrderedSet[torch.fx.Node] | None = None
) -> bool:
    """Return whether a value depends on a grouped layout needing QuACK combine."""
    if not isinstance(value, torch.fx.Node):
        return any(
            has_physical_grouped_input(arg, seen) for arg in iter_fx_node_inputs(value)
        )
    if seen is None:
        seen = OrderedSet()
    if value in seen:
        return False
    seen.add(value)
    view_args = view_or_reshape_args(value)
    if view_args is not None:
        source_node, shape = view_args
        source_shape = (
            tensor_meta_shape(source_node)
            if isinstance(source_node, torch.fx.Node)
            else None
        )
        layout = grouped_tensor_layout(shape, source_shape)
        if layout is not None and layout.needs_physical_combine:
            return True
    return any(
        has_physical_grouped_input(arg, seen)
        for arg in iter_fx_node_inputs((value.args, value.kwargs))
    )


def validate_hidden_feed_main_reduction_input(
    input_node: Any, grouped_source: torch.fx.Node
) -> None:
    """Reject reduction inputs that would need another physical feed-main value."""
    if input_node is grouped_source:
        raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
    if not isinstance(input_node, torch.fx.Node):
        return
    if fx_node_depends_on(input_node, grouped_source):
        raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
    if has_physical_grouped_input(input_node):
        raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)


def validate_feed_main_source_reductions(
    value: Any,
    grouped_source: torch.fx.Node,
    selected_reduction: torch.fx.Node,
    seen: OrderedSet[torch.fx.Node] | None = None,
) -> None:
    """Reject hidden physical reductions outside the selected feed-main value."""
    if not isinstance(value, torch.fx.Node):
        for arg in iter_fx_node_inputs(value):
            validate_feed_main_source_reductions(
                arg, grouped_source, selected_reduction, seen
            )
        return
    if value is selected_reduction:
        return
    if seen is None:
        seen = OrderedSet()
    if value in seen:
        return
    seen.add(value)
    reduction = reduction_from_node(value)
    if reduction is not None:
        validate_hidden_feed_main_reduction_input(reduction[0], grouped_source)
    for arg in iter_fx_node_inputs((value.args, value.kwargs)):
        validate_feed_main_source_reductions(
            arg, grouped_source, selected_reduction, seen
        )


def validate_feed_main_source_contract(
    source: torch.fx.Node,
    contract: FlexGemmLocalReduceContract | None,
) -> FlexGemmLocalReduceContract | None:
    """Preserve the one-physical-value ABI across recursive source matching."""
    if contract is None:
        return None
    grouped_source = local_reduce_contract_grouped_source(contract)
    if grouped_source is not None:
        validate_feed_main_source_reductions(source, grouped_source, contract.aux)
    return contract


def is_local_reduce_feed_main_binary_source(source: torch.fx.Node) -> bool:
    """Identify binary expressions whose operands may bind one feed-main value."""
    match source.op:
        case "call_method":
            return source.target in ("add", "div", "mul", "sub")
        case "call_function":
            return source.target in (
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add.Scalar,
                torch.ops.aten.div.Tensor,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul.Scalar,
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.sub.Scalar,
                operator.add,
                operator.mul,
                operator.sub,
                operator.truediv,
            )
    return False


def local_reduce_feed_main_binary_candidates(
    source: torch.fx.Node,
) -> tuple[tuple[Any, Any], ...]:
    """Return current feed-main binary source/value candidates."""
    if len(source.args) < 2 or not is_local_reduce_feed_main_binary_source(source):
        return ()
    lhs, rhs = source.args[:2]
    return ((lhs, rhs), (rhs, lhs))


def local_reduce_feed_main_candidate_contract(
    grouped_source: Any,
    value: Any,
    output_meta: Any,
) -> FlexGemmLocalReduceContract | None:
    """Validate the grouped source before matching its physical reduce value."""
    if not isinstance(grouped_source, torch.fx.Node) or not isinstance(
        value, torch.fx.Node
    ):
        return None
    view_args = view_or_reshape_args(grouped_source)
    if view_args is None:
        return None
    source_node, input_shape = view_args
    if not isinstance(source_node, torch.fx.Node):
        return None
    layout = grouped_tensor_layout(input_shape, tensor_meta_shape(source_node))
    if layout is None or layout.axis != 0:
        return None
    validate_local_reduce_feed_main_capability(layout.axis, layout.group_size)
    source_meta = source_node.meta.get("val")
    if output_meta is not None and source_meta is not None:
        if not statically_known_shape_equal(output_meta.shape, source_meta.shape):
            return None
    return local_reduce_feed_value_contract(value, grouped_source, layout)


def local_reduce_feed_main_source_contract(
    source: torch.fx.Node, output_meta: Any
) -> FlexGemmLocalReduceContract | None:
    """Find one physical feed-main value inside a shape-preserving expression."""
    contracts = [
        contract
        for grouped_source, value in local_reduce_feed_main_binary_candidates(source)
        for contract in (
            local_reduce_feed_main_candidate_contract(
                grouped_source, value, output_meta
            ),
        )
        if contract is not None
    ]
    if contracts:
        return validate_feed_main_source_contract(
            source,
            common_local_reduce_value_contract(
                contracts, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )
    if not is_shape_preserving_pointwise_node(source):
        return None
    contracts = [
        contract
        for arg in iter_fx_node_inputs((source.args, source.kwargs))
        if isinstance(arg, torch.fx.Node)
        for contract in (local_reduce_feed_main_source_contract(arg, output_meta),)
        if contract is not None
    ]
    return validate_feed_main_source_contract(
        source,
        common_local_reduce_value_contract(
            contracts, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
        ),
    )


def local_reduce_feed_main_plan(
    output: torch.fx.Node,
) -> FlexGemmLocalReduceContract | None:
    """Match same-warp grouped-M reductions that QuACK can broadcast."""
    view_args = view_or_reshape_args(output)
    if view_args is None:
        return None
    source, _ = view_args
    if not isinstance(source, torch.fx.Node):
        return None
    return local_reduce_feed_main_source_contract(source, output.meta.get("val"))


def common_local_reduce_feed_main_contract(
    candidates: tuple[Any, ...],
) -> FlexGemmLocalReduceContract | None:
    """Merge feed-main candidates that share the same physical reduction value."""
    feed_contracts = [
        contract
        for candidate in candidates
        if isinstance(candidate, torch.fx.Node)
        for contract in (local_reduce_feed_main_plan(candidate),)
        if contract is not None
    ]
    return common_local_reduce_value_contract(
        feed_contracts, LOCAL_REDUCE_FEED_MAIN_MIXED_CONTRACT_ERROR
    )


def physical_reduce_finalize_arg(
    value: Any,
    env: dict[torch.fx.Node, Any],
    local_reduce_physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction],
) -> Any:
    """Translate a finalize expression input, replacing reduced values with value."""
    if isinstance(value, torch.fx.Node) and value in local_reduce_physical_reductions:
        return local_reduce_physical_reductions[value].finalize_expr
    if isinstance(value, (tuple, list)):
        return type(value)(
            physical_reduce_finalize_arg(item, env, local_reduce_physical_reductions)
            for item in value
        )
    return _cute_arg(value, env)


def compose_physical_reduction_finalize(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    local_reduce_physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction],
) -> Any | None:
    """Fold post-reduction pointwise nodes into the generated physical finalizer."""
    physical_inputs = list(
        OrderedSet(
            arg
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if arg in local_reduce_physical_reductions
        )
    )
    if not physical_inputs:
        return None
    if len(physical_inputs) > 1:
        raise NotImplementedError(LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR)
    base = physical_inputs[0]
    base_store_source = local_reduce_store_sources[base]
    args = tuple(
        physical_reduce_finalize_arg(arg, env, local_reduce_physical_reductions)
        for arg in node.args
    )
    kwargs = {
        key: physical_reduce_finalize_arg(value, env, local_reduce_physical_reductions)
        for key, value in node.kwargs.items()
    }
    finalize_expr = _cute_call(node.target, args, kwargs)
    if not isinstance(finalize_expr, str):
        raise NotImplementedError(LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR)
    local_reduce_store_sources[node] = base_store_source
    local_reduce_physical_reductions[node] = dataclasses.replace(
        local_reduce_physical_reductions[base], finalize_expr=finalize_expr
    )
    return finalize_expr


def local_reduce_contract_from_grouped_input(
    node: torch.fx.Node,
    input_node: Any,
    dim: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    dtype: Any = None,
    *,
    raise_invalid_dims: bool = True,
) -> FlexGemmLocalReduceContract | None:
    """Build a reduction contract when the input has grouped TensorSSA provenance."""
    if not isinstance(input_node, torch.fx.Node):
        return None
    info = grouped_tensors.get(input_node)
    if info is None:
        return None
    if dtype is not None:
        raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
    if not grouped_reduce_dims_match(dim, info.layout.reduce_dims):
        if not raise_invalid_dims:
            return None
        raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
    return FlexGemmLocalReduceContract(node, info.group_size, info.axis)


def local_reduce_analysis(
    graph_module: torch.fx.GraphModule,
) -> FlexGemmLocalReduceAnalysis:
    """Analyze grouped TensorSSA provenance and local-reduce contracts in one pass."""
    analysis = FlexGemmLocalReduceAnalysis()
    for node in graph_module.graph.nodes:
        if node.op == "output":
            break
        if node.op not in ("call_function", "call_method"):
            continue
        view_args = view_or_reshape_args(node)
        if view_args is not None:
            source_node, shape = view_args
            if analysis.copy_contract(node, source_node):
                continue
            if analysis.bind_grouped_layout(node, shape, source_node):
                continue
        reduction = reduction_from_node(node)
        if reduction is not None:
            input_node, dim, _, dtype, _ = reduction
            if analysis.bind_grouped_reduction(node, input_node, dim, dtype):
                continue
        if (
            node.op == "call_function"
            and node.target is inductor_prims.prepare_softmax_online
        ):
            input_node = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
            if analysis.bind_grouped_reduction(
                node, input_node, dim, raise_invalid_dims=False
            ):
                continue
        unsupported_reduction = unsupported_reduction_from_node(node)
        if unsupported_reduction is not None:
            input_node = node.args[0]
            if analysis.has_grouped_tensor(input_node):
                raise local_reduce_unsupported_tensorssa_error(unsupported_reduction)
        if analysis.copy_contract(node, squeeze_source_node(node)):
            continue
        if node.op == "call_function" and node.target is operator.getitem:
            if analysis.copy_contract(node, node.args[0]):
                continue
        if is_shape_preserving_pointwise_node(node):
            analysis.bind_pointwise_contract(node, LOCAL_REDUCE_MIXED_CONTRACT_ERROR)
    return analysis


def local_reduce_aux_result(
    local_reduce_aux: torch.fx.Node | None,
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    """Return the generated compressed-aux expression or reject missing TensorSSA."""
    if local_reduce_aux is None:
        return None
    aux_result = local_reduce_store_sources.get(local_reduce_aux)
    if aux_result is None:
        raise NotImplementedError(LOCAL_REDUCE_AUX_TENSORSSA_ERROR)
    return aux_result


def local_reduce_compressed_aux_plan(
    analysis: FlexGemmLocalReduceAnalysis,
    output: Any,
    aux: torch.fx.Node,
) -> FlexGemmOutputLocalReducePlan | None:
    """Classify compressed local-reduce aux outputs by analyzed contract and shape."""
    contract = analysis.contract_for(aux)
    output_meta = output.meta.get("val") if isinstance(output, torch.fx.Node) else None
    aux_meta = aux.meta.get("val")
    if contract is None or aux_meta is None or output_meta is None:
        return None
    expected_aux_shape = local_reduce_compressed_shape(
        output_meta.shape, contract.group, contract.axis
    )
    if not statically_known_shape_equal(expected_aux_shape, aux_meta.shape):
        return None
    return contract.to_plan(store_node=aux, feeds_main=False)


def local_reduce_feed_main_output_plan(
    output: torch.fx.Node,
    aux_outputs: tuple[torch.fx.Node, ...] = (),
) -> FlexGemmOutputPlan | None:
    """Bind one shared physical reduction value to main-output consumers."""
    feed_contract = common_local_reduce_feed_main_contract((output, *aux_outputs))
    if feed_contract is None:
        return None
    return FlexGemmOutputPlan(
        output,
        aux_outputs,
        feed_contract.to_plan(store_node=None, feeds_main=True),
    )


def single_output_plan(output: torch.fx.Node) -> FlexGemmOutputPlan:
    """Classify a single-output epilogue."""
    return FlexGemmOutputPlan(output)


def tuple_output_plan(
    output: Any,
    aux_outputs: tuple[Any, ...],
    analysis: FlexGemmLocalReduceAnalysis | None = None,
) -> FlexGemmOutputPlan:
    """Classify multi-output epilogues after checking local-reduce consumers."""
    if not isinstance(output, torch.fx.Node) or not all(
        isinstance(aux_output, torch.fx.Node) for aux_output in aux_outputs
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_TENSOR_ERROR)
    if analysis is not None:
        compressed_aux_plans = tuple(
            (index, plan)
            for index, aux_output in enumerate(aux_outputs)
            if (plan := local_reduce_compressed_aux_plan(analysis, output, aux_output))
            is not None
        )
        if len(compressed_aux_plans) > 1:
            raise NotImplementedError(LOCAL_REDUCE_MIXED_CONTRACT_ERROR)
        if compressed_aux_plans:
            local_reduce_index, compressed_aux_plan = compressed_aux_plans[0]
            return FlexGemmOutputPlan(
                output,
                tuple(
                    aux_output
                    for index, aux_output in enumerate(aux_outputs)
                    if index != local_reduce_index
                ),
                local_reduce=compressed_aux_plan,
                local_reduce_aux_index=local_reduce_index,
            )
    feed_main_plan = local_reduce_feed_main_output_plan(output, aux_outputs)
    if feed_main_plan is not None:
        return feed_main_plan
    return FlexGemmOutputPlan(output, aux_outputs)


def output_plan(
    graph_module: torch.fx.GraphModule,
) -> FlexGemmOutputPlan:
    """Classify output consumers before generated epilogue materialization."""
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one output node")
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        else:
            output, *aux_outputs = output_value
            aux_outputs = tuple(aux_outputs)
            analysis = None
            if isinstance(output, torch.fx.Node) and all(
                isinstance(aux_output, torch.fx.Node) for aux_output in aux_outputs
            ):
                analysis = local_reduce_analysis(graph_module)
            return tuple_output_plan(output, aux_outputs, analysis)
    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlexGEMM expects one tensor output")
    return single_output_plan(output_value)


def gemm_node(
    graph_module: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> torch.fx.Node:
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == gemm_op
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one GEMM body")
    return gemm_nodes[0]


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm_op: torch._ops.OpOverload,
    outputs: FlexGemmOutputPlan,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
) -> tuple[str, str]:
    """Build the generated CuTeDSL epilogue callable from a classified FX body."""
    gemm = gemm_node(graph_module, gemm_op)
    kernel = FlexGemmCuteDSLKernel()
    env: dict[torch.fx.Node, Any] = {
        gemm: CuteDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo] = {}
    local_reduce_store_sources: dict[torch.fx.Node, Any] = {}
    local_reduce_physical_reductions: dict[
        torch.fx.Node, FlexGemmPhysicalReduction
    ] = {}
    local_reduce = outputs.local_reduce
    local_reduce_feed_main = None
    local_reduce_aux = None
    local_reduce_feed_main_input = None
    match local_reduce:
        case FlexGemmOutputLocalReducePlan(
            feeds_main=True, value_node=value_node, store_node=store_node
        ):
            local_reduce_feed_main = value_node
            reduction = reduction_from_node(value_node)
            local_reduce_feed_main_input = (
                reduction[0] if reduction is not None else None
            )
            local_reduce_aux = store_node
        case FlexGemmOutputLocalReducePlan(store_node=store_node):
            local_reduce_aux = store_node
        case None:
            pass
        case _:
            raise AssertionError("unhandled FlexGEMM local-reduce output plan")
    with V.set_kernel_handler(kernel), V.set_ops_handler(FlexGemmCuteDSLOpOverrides()):
        for index, node in enumerate(epilogue_arg_placeholders):
            epilogue_arg_meta = node.meta["val"]
            physical_dtype = (
                torch.uint8
                if epilogue_arg_meta.dtype is torch.bool
                else epilogue_arg_meta.dtype
            )
            logical_dtype = upcast_compute_type(epilogue_arg_meta.dtype)
            env[node] = CuteDSLCSEVariable(
                f"aux{index}",
                ValueRanges.unknown(),
                dtype=physical_dtype,
                shape=(1,),
            )
            if logical_dtype != physical_dtype:
                env[node] = FlexGemmCuteDSLOpOverrides.to_dtype(
                    env[node], logical_dtype, use_compute_types=False
                )

        for node in graph_module.graph.nodes:
            if node is gemm or node.op in ("placeholder", "output"):
                continue
            if isinstance(node.meta.get("val"), (int, torch.SymInt)):
                continue
            with V.set_current_node(node):
                if node.op in ("call_function", "call_method"):
                    lowered_full_scalar = lower_full_scalar(node)
                    if lowered_full_scalar is not None:
                        env[node] = lowered_full_scalar
                        continue
                    lowered_squeeze = lower_squeeze(
                        node, env, local_reduce_store_sources
                    )
                    if lowered_squeeze is not None:
                        env[node] = lowered_squeeze
                        continue
                    lowered_getitem = lower_getitem(
                        node, env, local_reduce_store_sources
                    )
                    if lowered_getitem is not None:
                        env[node] = lowered_getitem
                        continue
                    lowered_prepare_softmax = lower_prepare_softmax_online(
                        node,
                        env,
                        kernel,
                        grouped_tensors,
                        local_reduce_store_sources,
                    )
                    if lowered_prepare_softmax is not None:
                        env[node] = lowered_prepare_softmax
                        continue
                    lowered_view = lower_view_or_reshape(
                        node,
                        env,
                        kernel,
                        grouped_tensors,
                        local_reduce_store_sources,
                        node is local_reduce_feed_main_input,
                    )
                    if lowered_view is not None:
                        env[node] = lowered_view
                        continue
                    lowered_reduce = lower_tensorssa_reduce(
                        node,
                        env,
                        kernel,
                        grouped_tensors,
                        local_reduce_store_sources,
                        local_reduce_physical_reductions,
                    )
                    if lowered_reduce is not None:
                        if (
                            local_reduce_feed_main is not None
                            and node is local_reduce_feed_main
                        ):
                            env[node] = CuteDSLCSEVariable(
                                LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                                ValueRanges.unknown(),
                                dtype=lowered_reduce.dtype,
                                shape=lowered_reduce.shape,
                            )
                            if (
                                local_reduce_feed_main_input is not None
                                and local_reduce_feed_main_input in grouped_tensors
                            ):
                                grouped_tensors[node] = grouped_tensors[
                                    local_reduce_feed_main_input
                                ]
                        else:
                            env[node] = lowered_reduce
                        continue
                    unsupported_reduction = unsupported_reduction_from_node(node)
                    if unsupported_reduction is not None:
                        raise local_reduce_unsupported_tensorssa_error(
                            unsupported_reduction, value_only=True
                        )
                    is_shape_preserving = is_shape_preserving_pointwise_node(node)
                    if is_shape_preserving and local_reduce_feed_main is None:
                        if local_reduce_aux is None and any(
                            arg in local_reduce_physical_reductions
                            for arg in iter_fx_node_inputs((node.args, node.kwargs))
                        ):
                            raise NotImplementedError(
                                LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR
                            )
                        physical_finalize = compose_physical_reduction_finalize(
                            node,
                            env,
                            local_reduce_store_sources,
                            local_reduce_physical_reductions,
                        )
                        if physical_finalize is not None:
                            env[node] = physical_finalize
                            continue
                    if (
                        local_reduce_feed_main is None
                        and is_shape_preserving
                        and has_local_reduce_store_source(
                            (node.args, tuple(node.kwargs.values())),
                            local_reduce_store_sources,
                        )
                    ):
                        store_args = tuple(
                            _local_reduce_store_arg(
                                arg, env, local_reduce_store_sources
                            )
                            for arg in node.args
                        )
                        store_kwargs = {
                            key: _local_reduce_store_arg(
                                value, env, local_reduce_store_sources
                            )
                            for key, value in node.kwargs.items()
                        }
                        env[node] = _cute_call(node.target, store_args, store_kwargs)
                        local_reduce_store_sources[node] = env[node]
                    else:
                        node_args = tuple(_cute_arg(arg, env) for arg in node.args)
                        node_kwargs = {
                            key: _cute_arg(value, env)
                            for key, value in node.kwargs.items()
                        }
                        env[node] = _cute_call(node.target, node_args, node_kwargs)
                    if is_shape_preserving:
                        grouped_info = propagate_grouped_tensorssa_info(
                            node, grouped_tensors
                        )
                        if grouped_info is not None:
                            grouped_tensors[node] = grouped_info
                    continue
                raise NotImplementedError(
                    f"unsupported FlexGEMM epilogue node: {node.format_node()}"
                )

    body = "\n".join(f"    {line}" for line in kernel.body.lines)
    if body:
        body += "\n"
    aux_args = [f"aux{index}" for index in range(len(epilogue_arg_placeholders))]
    local_reduce_args = (
        [LOCAL_REDUCE_FEED_MAIN_ARG_NAME] if local_reduce_feed_main is not None else []
    )
    epilogue_params = ", ".join(["acc", *aux_args, *local_reduce_args])
    result = _cute_arg(outputs.output, env)
    aux_result = local_reduce_aux_result(local_reduce_aux, local_reduce_store_sources)
    if outputs.aux_outputs or aux_result is not None:
        tuple_items = [result]
        tuple_items.extend(
            _cute_arg(aux_output, env) for aux_output in outputs.aux_outputs
        )
        if aux_result is not None:
            tuple_items.append(aux_result)
        result = f"({', '.join(str(item) for item in tuple_items)})"
    physical_reduction = None
    if local_reduce is not None:
        physical_reduction = local_reduce_physical_reductions.get(
            local_reduce.value_node
        )
    physical_reduction_payload = (
        ""
        if physical_reduction is None
        else f"\ncombine {physical_reduction.combine_expr}\nfinalize {physical_reduction.finalize_expr}"
    )
    key_payload = (
        f"{graph_module.code}\n{body}\nreturn {result}{physical_reduction_payload}"
    )
    key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
    name = f"flex_gemm_epilogue_{key}"
    local_reduce_source = ""
    if physical_reduction is not None:
        combine_name = f"{name}{LOCAL_REDUCE_COMBINE_FN_SUFFIX}"
        finalize_name = f"{name}{LOCAL_REDUCE_FINALIZE_FN_SUFFIX}"
        local_reduce_source = (
            f"@cute.jit\ndef {combine_name}(lhs, rhs):\n"
            f"    return {physical_reduction.combine_expr}\n"
            f"{combine_name}.__cache_key__ = lambda: {combine_name!r}\n\n"
            f"@cute.jit\ndef {finalize_name}(value):\n"
            f"    return {physical_reduction.finalize_expr}\n"
            f"{finalize_name}.__cache_key__ = lambda: {finalize_name!r}\n\n"
        )
    return (
        name,
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "import operator\n"
        "from cutlass._mlir.dialects import math as mlir_math\n\n"
        f"{local_reduce_source}"
        f"@cute.jit\ndef {name}({epilogue_params}):\n"
        f"{body}    return {result}\n",
    )
