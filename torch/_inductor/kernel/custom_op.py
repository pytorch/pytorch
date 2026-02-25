# Owner(s): ["module: inductor"]

import contextlib
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
from torch._dynamo.utils import counters
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import (
    Buffer,
    FixedLayout,
    ir_node_to_tensor,
    StorageBox,
    TensorBox,
)
from torch._inductor.lowering import user_lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)
from torch._inductor.utils import convert_symint_to_expr
from torch._inductor.virtualized import V


log = logging.getLogger(__name__)

DEFAULT_RANGE_UPPER_BOUND = 65536


@dataclass(frozen=True)
class RangeBounds:
    """Inclusive range [start, end] for dimension-based dispatch."""

    start: int
    end: Union[int, float]  # float('inf') for unbounded

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError(f"Range start must be >= 1, got {self.start}")
        if self.end != float("inf") and self.start > self.end:
            raise ValueError(f"Invalid range: start={self.start} > end={self.end}")

    def contains(self, value: int) -> bool:
        if self.end == float("inf"):
            return value >= self.start
        return self.start <= value <= int(self.end)

    def __str__(self) -> str:
        end_str = "inf" if self.end == float("inf") else str(int(self.end))
        return f"[{self.start}, {end_str}]"


@dataclass(frozen=True)
class ImplConfig:
    """Implementation config with semantic identity (name + kwargs) for hashing."""

    impl_name: str
    impl_func: Callable[..., Any] = field(compare=False, hash=False, repr=False)
    kwargs: dict[str, Any] = field(default_factory=dict)
    config_patches: dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False, repr=False
    )

    def __hash__(self) -> int:
        return hash((self.impl_name, tuple(sorted(self.kwargs.items()))))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImplConfig):
            return False
        return self.impl_name == other.impl_name and self.kwargs == other.kwargs

    def __str__(self) -> str:
        if self.kwargs:
            kwargs_str = ", ".join(f"{k}={v}" for k, v in sorted(self.kwargs.items()))
            return f"{self.impl_name}({kwargs_str})"
        return self.impl_name


@dataclass
class RangeImplGroup:
    """Groups non-adjacent ranges using the same implementation."""

    impl_config: ImplConfig
    ranges: list[RangeBounds] = field(default_factory=list)

    def add_range(self, range_bounds: RangeBounds) -> None:
        self.ranges.append(range_bounds)
        self.ranges.sort(key=lambda r: r.start)

    def __str__(self) -> str:
        ranges_str = ", ".join(str(r) for r in self.ranges)
        return f"{self.impl_config.impl_name}: {ranges_str}"

    @property
    def impl_name(self) -> str:
        return self.impl_config.impl_name

    @property
    def impl_func(self) -> Callable[..., Any]:
        return self.impl_config.impl_func

    @property
    def impl_kwargs(self) -> dict[str, Any]:
        return self.impl_config.kwargs

    @property
    def config_patches(self) -> dict[str, Any]:
        return self.impl_config.config_patches


def _detect_collective_ops(choices: list) -> bool:
    """
    Detect if choices contain collective operations.
    """
    from torch._inductor.utils import is_collective_op

    for choice in choices:
        if not hasattr(choice, "gm") or choice.gm is None:
            continue

        for node in choice.gm.graph.nodes:
            if node.op == "call_function" and node.target is not None:
                op_name = str(node.target)

                if is_collective_op(op_name) or is_collective_op(
                    f"torch.ops.{op_name}"
                ):
                    return True

    return False


class CustomOpConfig:
    """Config for custom op autotuning.

    Specifies optional decomposition function with parameter values.
    Each config creates exactly one variant.

    Args:
        decomposition: Optional functions to autotune. If not provided, default will be used.
        config_patches: Optional dict of config patches to apply during kernel codegen
            (e.g., {"coordinate_descent_tuning": True})
        **params: Parameters passed to the function

    Examples:
        CustomOpConfig(attention_impl, head_dim=32, method='chunked')
        CustomOpConfig(head_dim=32, method='chunked')
        CustomOpConfig(decomposition, config_patches={"coordinate_descent_tuning": True})
    """

    def __init__(
        self,
        decomposition: Optional[Callable[..., Any]] = None,
        config_patches: Optional[dict[str, Any]] = None,
        **params: Any,
    ):
        if decomposition is not None and not callable(decomposition):
            raise TypeError(
                f"decomposition must be callable, got {type(decomposition)}"
            )

        self.decomposition = decomposition
        self.config_patches = config_patches or {}
        self.params = params

    def get_decomposition(
        self, default_impl: Optional[Callable[..., Any]] = None
    ) -> Callable[..., Any]:
        """Return the decomposition function for this config.
        When decomposition is not specified, return the default implementation.
        """
        if self.decomposition is not None:
            return self.decomposition

        if default_impl is not None and callable(default_impl):
            return default_impl

        raise TypeError(
            "No decomposition specified in config and no default implementation provided. "
            "Please provide a decomposition function in CustomOpConfig."
        )

    def __repr__(self) -> str:
        decomp_name = self.decomposition.__name__ if self.decomposition else "default"
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"CustomOpConfig({decomp_name}, {params_str})"
        return f"CustomOpConfig({decomp_name})"


__all__ = [
    "autotune_custom_op",
    "register_custom_op_autotuning",
    "CustomOpConfig",
]


def _extract_tensor_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    op_overload: torch._ops.OpOverload,
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from mixed args/kwargs.
    Separates tensors (for autotuning input_nodes) from non-tensor parameters.

    Args:
        args: Positional arguments (mix of tensors and scalars)
        kwargs: Keyword arguments (mix of tensors and scalars)
        op_overload: Custom Op overload to get parameter names from schema.

    Returns:
        Tuple of (tensor_inputs_list, non_tensor_kwargs)
    """
    tensor_inputs = []
    non_tensor_kwargs = {}

    # Get schema names and extend with fallback names for any extra args
    schema_names = [arg.name for arg in op_overload._schema.arguments]
    param_names = schema_names + [
        f"arg_{i}" for i in range(len(schema_names), len(args))
    ]

    for i, arg in enumerate(args):
        if isinstance(arg, (TensorBox, Buffer, StorageBox)):
            tensor_inputs.append(arg)
        else:
            non_tensor_kwargs[param_names[i]] = arg

    for key, value in kwargs.items():
        if isinstance(value, (TensorBox, Buffer, StorageBox)):
            tensor_inputs.append(value)
        else:
            non_tensor_kwargs[key] = value

    return tensor_inputs, non_tensor_kwargs


def _merge_config_and_runtime_kwargs(
    config_params: dict[str, Any],
    runtime_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge config parameters with runtime kwargs. Config params take precedence,
    since they represent the values being autotuned.

    Args:
        config_params: Parameters from CustomOpConfig (autotuning knobs)
        runtime_kwargs: Runtime non-tensor kwargs from _extract_tensor_inputs

    Returns:
        Merged kwargs dictionary with config values taking precedence
    """
    merged_kwargs = runtime_kwargs.copy()
    merged_kwargs.update(config_params)
    return merged_kwargs


def _adapt_user_input_gen_fns(
    inputs: list[Any],
    op_overload: torch._ops.OpOverload,
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[int, Callable[[Any], torch.Tensor]]:
    """Convert user input generators from name-based to index-based format.
    Inductor autotune's input_gen_fns expects index of arg_names as key.
    """
    arg_names = [arg.name for arg in op_overload._schema.arguments]
    name_to_index = {name: i for i, name in enumerate(arg_names)}
    index_based_fns = {}

    for name, gen_fn in user_input_gen_fns.items():
        if name in name_to_index:
            index_based_fns[name_to_index[name]] = gen_fn
        else:
            raise ValueError(
                f"Unknown argument name '{name}' in input_gen_fns. "
                f"Available argument names: {list(name_to_index.keys())}"
            )

    def create_internal_input_gen_fn(
        user_function: Callable[[torch.Tensor], torch.Tensor], arg_name: str
    ) -> Callable[[Any], torch.Tensor]:
        """Create internal input generator that converts IR buffer to user's fake tensor."""

        def internal_input_gen_fn(ir_buffer: Any) -> torch.Tensor:
            fake_tensor = ir_node_to_tensor(ir_buffer, replace_symbols_with_hints=True)
            assert fake_tensor is not None, "ir_node_to_tensor returned None"
            return user_function(fake_tensor)

        return internal_input_gen_fn

    return {
        i: create_internal_input_gen_fn(
            user_gen_fn, arg_names[i] if i < len(arg_names) else f"arg_{i}"
        )
        for i, user_gen_fn in index_based_fns.items()
        if i < len(inputs)
    }


def _group_ranges_by_impl(
    range_to_best_impl: dict[RangeBounds, ImplConfig],
) -> list[RangeImplGroup]:
    """Group ranges by implementation using semantic identity (name + kwargs)."""
    from torch._inductor import config

    if not range_to_best_impl:
        return []

    # Test mode: skip grouping to force torch.cond dispatch path
    if config.test_configs.force_no_impl_grouping:
        log.info("Test mode: skipping impl grouping, each range is separate group")
        groups = []
        for range_bounds, impl_config in sorted(
            range_to_best_impl.items(), key=lambda x: x[0].start
        ):
            group = RangeImplGroup(impl_config)
            group.add_range(range_bounds)
            groups.append(group)
        return groups

    # Group ranges by impl_config (uses __hash__ and __eq__ based on semantic identity)
    impl_to_group: dict[ImplConfig, RangeImplGroup] = {}

    for range_bounds, impl_config in range_to_best_impl.items():
        if impl_config not in impl_to_group:
            impl_to_group[impl_config] = RangeImplGroup(impl_config)
        impl_to_group[impl_config].add_range(range_bounds)

    # Sort groups by first range start for deterministic codegen
    groups = sorted(impl_to_group.values(), key=lambda g: g.ranges[0].start)

    # Log grouping info
    original_count = len(range_to_best_impl)
    grouped_count = len(groups)

    if grouped_count < original_count:
        log.info(
            "Implementation grouping: reduced from %d ranges to %d impl groups",
            original_count,
            grouped_count,
        )

    return groups


def _create_ranges_from_split_points(
    split_points: list[int],
) -> list[tuple[int, int] | tuple[int, float]]:
    """Convert split points into ranges for autotuning dispatch.

    Example:
        split_points=[512, 2048]
        returns:
               [(1, 512), (513, 2048), (2049, float('inf'))]
    """
    ranges: list[tuple[int, int] | tuple[int, float]] = []
    start = 1

    for split_point in split_points:
        ranges.append((start, split_point))
        start = split_point + 1

    ranges.append((start, float("inf")))

    return ranges


def _create_range_input_gen_fn(
    base_gen_fn: Callable[[torch.Tensor], torch.Tensor],
    dim_index: int,
    range_start: int,
    range_end: Union[int, float],
    range_upper_bound: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create input generator that modifies target dimension to top of range.
    range_upper_bound: Size to use for benchmarking when range_end is unbounded.
    Default is DEFAULT_RANGE_UPPER_BOUND = 65536
    """
    from torch._inductor.ir import get_fill_order
    from torch._inductor.kernel.flex.common import construct_strides

    target_dim = range_upper_bound if range_end == float("inf") else int(range_end)

    def constrained_gen_fn(fake_tensor: torch.Tensor) -> torch.Tensor:
        result = base_gen_fn(fake_tensor)
        shape = list(result.shape)
        shape[dim_index] = target_dim

        # We modified the shape of the result, so we need to recalculate the strides
        # TODO: Refine this to a better way to more directly preserve strides
        fill_order = get_fill_order(result.stride(), shape_env=None)
        new_stride = construct_strides(shape, fill_order)

        storage_size = sum((s - 1) * st for s, st in zip(shape, new_stride)) + 1
        storage = torch.randn(storage_size, dtype=result.dtype, device=result.device)
        return storage.as_strided(shape, tuple(new_stride))

    return constrained_gen_fn


def _default_input_gen_fn(fake_tensor: torch.Tensor) -> torch.Tensor:
    """Default input generator that creates a real tensor matching the fake tensor's shape."""
    return torch.randn(
        fake_tensor.shape, dtype=fake_tensor.dtype, device=fake_tensor.device
    )


def _create_fallback_choice(
    op_overload: torch._ops.OpOverload,
) -> ExternKernelChoice:
    """Create or reuse fallback choice that calls the op eagerly.

    Since kwargs are passed at bind time via maybe_append_choice rather than
    baked into the kernel, the same ExternKernelChoice is reused across
    compilations for the same op_overload.
    """
    fallback_name = (
        f"{op_overload.name().replace('::', '_').replace('.', '_')}_fallback"
    )

    existing = ExternKernelChoice.lookup(fallback_name)
    if existing is not None:
        return existing

    return ExternKernelChoice(
        kernel=op_overload,
        name=fallback_name,
        has_out_variant=False,
        op_overload=op_overload,
        use_fallback_kernel=True,
    )


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[torch.fx.Node],
    non_tensor_args: list[dict[str, Any]],
    op_overload: torch._ops.OpOverload,
    user_input_gen_fns: Optional[
        dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
    config_patches_list: Optional[list[dict[str, Any]]] = None,
    return_choice: bool = False,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> Union[TensorBox, Any, tuple[Any, Any]]:
    """Autotune custom operations by comparing multiple decomposition implementations.

    Currently supports SINGLE OUTPUT custom ops only.
    TODO: Add support for multiple output custom ops (tuple/list returns).

    This function generates multiple implementation choices for a custom operation and
    uses Inductor's autotuning system to select the best performing variant at runtime.
    After selecting the best choice, applies inline fusion if the winning choice has a graph.

    Args:
        name: Unique identifier for the autotuning operation
        decompositions: List of alternative implementation functions to benchmark
        inputs: Input tensor IR nodes from compilation (TensorBox/Buffer objects)
        non_tensor_args: List of kwargs dicts, paired with corresponding decompositions arg
        op_overload: OpOverload of the custom op, used as fallback implementation
        user_input_gen_fns: Optional custom input generators for benchmarking.
                           Maps input indices to functions that take fake tensors
                           and return real tensors for performance measurement.

    Returns:
        IR node representing the optimized operation result

    Raises:
        TypeError: If decompositions is not a list/tuple
        RuntimeError: If no inputs or no valid choices generated
    """
    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    if not inputs:
        raise RuntimeError(f"Custom op '{name}' requires tensor inputs for autotuning")

    if len(decompositions) != len(non_tensor_args):
        raise ValueError(
            f"decompositions and non_tensor_args must have same length, "
            f"got {len(decompositions)} decompositions and {len(non_tensor_args)} kwargs"
        )

    # Convert user input generation functions BEFORE creating choices
    input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] = {}
    if user_input_gen_fns:
        input_gen_fns = _adapt_user_input_gen_fns(
            inputs, op_overload, user_input_gen_fns
        )

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=decompositions,
        # pyrefly: ignore [no-matching-overload]
        input_nodes=list(inputs),
        non_tensor_args=non_tensor_args,
        input_gen_fns=input_gen_fns if input_gen_fns else None,
        config_patches_list=config_patches_list,
    )

    # Add fallback choice that calls the op eagerly (not through inductor lowering)
    # This provides a baseline to compare decompositions against
    from torch._inductor import config

    fallback_kwargs = non_tensor_args[0] if non_tensor_args else {}

    with V.fake_mode:
        # pyrefly: ignore [no-matching-overload]
        fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]
        fake_output = op_overload(*fake_inputs, **fallback_kwargs)

    output_size = tuple(convert_symint_to_expr(s) for s in fake_output.shape)
    output_stride = tuple(convert_symint_to_expr(s) for s in fake_output.stride())

    fallback_choice = _create_fallback_choice(op_overload)
    fallback_choice.maybe_append_choice(
        choices=choices,
        input_nodes=list(inputs),
        layout=FixedLayout(
            device=fake_output.device,
            dtype=fake_output.dtype,
            size=output_size,
            stride=output_stride,
        ),
        **fallback_kwargs,
    )

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    is_collective = _detect_collective_ops(choices)

    # Run autotuning and get both result and winning choice
    selected_result, winning_choice = autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
        return_choice=True,
        is_collective=is_collective,
        min_speedup_threshold=min_speedup_threshold,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )

    # Test mode: force specific choice to win
    force_choice = config.test_configs.force_custom_op_decomposition
    if force_choice is True and winning_choice.gm is None:
        # Force decomposition: pick first choice with a graph
        for choice in choices:
            if choice.gm is not None:
                log.info(
                    "Test mode: forcing decomposition %s over fallback",
                    getattr(choice, "name", type(choice).__name__),
                )
                winning_choice = choice
                selected_result = choice.output_node()
                break
    elif force_choice is False and winning_choice.gm is not None:
        # Force fallback: pick first choice without a graph
        for choice in choices:
            if choice.gm is None:
                log.info(
                    "Test mode: forcing fallback %s over decomposition",
                    getattr(choice, "name", type(choice).__name__),
                )
                winning_choice = choice
                selected_result = choice.output_node()
                break

    # Apply inlining for fusion if winning_choice has graph; otherwise return result as-is(default fallback impl)
    if winning_choice.gm is not None:
        # Skip inlining when return_choice=True since caller only needs choice metadata
        # (e.g., range-based dispatch builds its own torch.cond from winning choices)
        if return_choice:
            log.debug(
                "Skipping inline for return_choice: %s (name=%s)",
                getattr(winning_choice, "name", type(winning_choice).__name__),
                name,
            )
            return selected_result, winning_choice

        log.debug(
            "Inlining winning choice: %s (name=%s)",
            getattr(winning_choice, "name", type(winning_choice).__name__),
            name,
        )
        from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes

        ops_before = len(V.graph.operations)
        result = inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)

        # Tag inlined operations with config_patches from the winning choice
        config_patches = getattr(winning_choice, "config_patches", None)
        if config_patches:
            for op in V.graph.operations[ops_before:]:
                op.set_config_patches(config_patches.copy())

        return result

    log.debug(
        "Winning choice does not support inlining: %s (name=%s)",
        getattr(winning_choice, "name", type(winning_choice).__name__),
        name,
    )
    if return_choice:
        return selected_result, winning_choice
    return selected_result


def _generate_dynamic_configs(
    tensor_inputs: list[Buffer],
    config_generator: Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]],
    op_overload: torch._ops.OpOverload,
    operation_name: str,
) -> list[CustomOpConfig]:
    """Generate configs dynamically based on input tensors at lowering time."""
    # Get parameter names from op schema instead of impl signature
    schema = op_overload._schema
    param_names = [arg.name for arg in schema.arguments if not arg.kwarg_only]

    with V.fake_mode:
        fake_tensors = [ir_node_to_tensor(inp) for inp in tensor_inputs]

    fake_tensors_dict = dict(zip(param_names, fake_tensors))

    configs = config_generator(fake_tensors_dict)

    if not isinstance(configs, (list, tuple)):
        raise TypeError(
            f"config_generator must return a list or tuple of CustomOpConfig, "
            f"got {type(configs)}"
        )
    if not configs:
        log.info(
            "config_generator returned empty list for %s, will use default lowering",
            operation_name,
        )
        return []

    return list(configs)


def _prepare_configs_and_decompositions(
    processed_configs: Optional[list[CustomOpConfig]],
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ],
    tensor_inputs: list[Any],
    default_impl: Callable[..., Any],
    op_overload: torch._ops.OpOverload,
    runtime_kwargs: dict[str, Any],
    name: str,
) -> tuple[list[Callable], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare decompositions and merged kwargs from configs.

    Handles both static configs and dynamic config generation.
    Merges config params with runtime kwargs (runtime takes precedence).
    """
    # Get configs: either generate dynamically or use static configs
    if config_generator is not None:
        configs_to_use = _generate_dynamic_configs(
            tensor_inputs, config_generator, op_overload, name
        )
    else:
        assert processed_configs is not None
        configs_to_use = processed_configs

    # Prepare decompositions and kwargs for autotuning
    decompositions = []
    non_tensor_args = []
    config_patches_list = []

    for cfg in configs_to_use:
        decomp = cfg.get_decomposition(default_impl=default_impl)
        decompositions.append(decomp)

        # Merge config params with runtime kwargs (runtime takes precedence)
        merged_kwargs = _merge_config_and_runtime_kwargs(cfg.params, runtime_kwargs)
        non_tensor_args.append(merged_kwargs)

        # Collect config_patches for each config
        config_patches_list.append(cfg.config_patches)

    return decompositions, non_tensor_args, config_patches_list


def _standard_lowering_fn(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    tensor_inputs: list[Any],
    runtime_kwargs: dict[str, Any],
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> Any:
    """Standard autotuning lowering function.

    Returns None if no configs/decompositions available, signaling caller to
    use normal lowering.
    """
    decompositions, non_tensor_args, config_patches_list = (
        _prepare_configs_and_decompositions(
            processed_configs,
            config_generator,
            tensor_inputs,
            default_impl,
            op_overload,
            runtime_kwargs,
            name,
        )
    )

    # If no decompositions, signal caller to use normal lowering
    if not decompositions:
        return None

    result = autotune_custom_op(
        name=name,
        decompositions=decompositions,
        inputs=tensor_inputs,
        non_tensor_args=non_tensor_args,
        config_patches_list=config_patches_list,
        op_overload=op_overload,
        user_input_gen_fns=input_gen_fns,
        min_speedup_threshold=min_speedup_threshold,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )

    validate_ir(result)
    return result


def _apply_config_patches_recursive(
    operations: list,
    config_patches: dict[str, Any],
) -> None:
    """Apply config_patches to operations, including those inside subgraphs."""
    for op in operations:
        if hasattr(op, "set_config_patches"):
            op.set_config_patches(config_patches.copy())

        # Recurse into any subgraphs (Conditional, WhileLoop, InvokeSubgraph, etc.)
        for subgraph in op.get_subgraphs():
            if subgraph.graph:
                _apply_config_patches_recursive(
                    subgraph.graph.operations, config_patches
                )


def _lower_single_impl(
    impl: Callable[..., Any],
    impl_kwargs: dict[str, Any],
    runtime_kwargs: dict[str, Any],
    tensor_inputs: list[Any],
    name: str,
    config_patches: Optional[dict[str, Any]] = None,
) -> Any:
    """Lower a single implementation by tracing and inlining it.

    Uses error_on_new_guards() during tracing to detect if the impl adds guards.
    Returns None if the impl adds guards, signaling caller to skip this choice.
    """
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.fx.experimental.symbolic_shapes import _ShapeEnvGuardError

    from ..decomposition import select_decomp_table

    merged_kwargs = _merge_config_and_runtime_kwargs(impl_kwargs, runtime_kwargs)

    def impl_wrapper(*tensors):
        return impl(*tensors, **merged_kwargs)

    shape_env = V.fake_mode.shape_env

    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
        decomposition_table = select_decomp_table()

        context = (
            shape_env.error_on_new_guards
            if shape_env is not None
            else contextlib.nullcontext
        )
        try:
            with context():
                impl_gm = make_fx(
                    impl_wrapper,
                    decomposition_table=decomposition_table,
                    tracing_mode="symbolic",
                )(*fake_inputs)
        except (_ShapeEnvGuardError, AssertionError) as e:
            is_guard_error = isinstance(e, _ShapeEnvGuardError) or (
                isinstance(e, AssertionError)
                and "Guard attempted while ShapeEnv guards are frozen" in str(e)
            )
            if not is_guard_error:
                raise
            log.info(
                "Implementation %s adds guards, skipping custom op lowering",
                impl.__name__,
            )
            counters["inductor"]["custom_op_decomp_guard_skips"] += 1
            return None

    log.info("Inlining implementation: %s", impl.__name__)
    ops_before = len(V.graph.operations)
    result = inline_subgraph_to_ir_nodes(impl_gm, tensor_inputs, name)

    if config_patches:
        _apply_config_patches_recursive(V.graph.operations[ops_before:], config_patches)

    validate_ir(result)
    return result


def _range_based_lowering_fn(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    tensor_name: str,
    dim_index: int,
    ranges: list[tuple[int, Union[int, float]]],
    tensor_inputs: list[Any],
    runtime_kwargs: dict[str, Any],
    range_upper_bound: int,
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> Any:
    """Range-based autotuning lowering function."""
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    from torch.fx.experimental.proxy_tensor import make_fx

    from ..decomposition import select_decomp_table

    log.info("=== Range-based Autotuning for %s ===", name)
    log.info("Dispatch on: %s[%d], Ranges: %s", tensor_name, dim_index, ranges)

    decompositions, non_tensor_args, config_patches_list = (
        _prepare_configs_and_decompositions(
            processed_configs,
            config_generator,
            tensor_inputs,
            default_impl,
            op_overload,
            runtime_kwargs,
            name,
        )
    )

    range_to_best_impl_map: dict[RangeBounds, ImplConfig] = {}

    # Benchmark each range and collect winning implementations
    for range_start, range_end in ranges:
        if input_gen_fns and tensor_name in input_gen_fns:
            base_gen_fn = input_gen_fns[tensor_name]
        else:
            base_gen_fn = _default_input_gen_fn

        range_gen_fn = _create_range_input_gen_fn(
            base_gen_fn, dim_index, range_start, range_end, range_upper_bound
        )
        range_input_gen_fns = {**(input_gen_fns or {}), tensor_name: range_gen_fn}

        range_name = f"{name}_range_{int(range_start)}_{int(range_end) if range_end != float('inf') else 'inf'}"

        # pyrefly: ignore [not-iterable]
        autotuned_result, winning_choice = autotune_custom_op(
            name=range_name,
            decompositions=decompositions,
            inputs=tensor_inputs,
            non_tensor_args=non_tensor_args,
            op_overload=op_overload,
            user_input_gen_fns=range_input_gen_fns,
            return_choice=True,
            min_speedup_threshold=min_speedup_threshold,
            benchmark_with_cudagraphs=benchmark_with_cudagraphs,
            config_patches_list=config_patches_list,
        )

        if (
            hasattr(winning_choice, "decomposition")
            and winning_choice.decomposition is not None
        ):
            winning_impl = winning_choice.decomposition
            winning_kwargs = winning_choice.decomposition_kwargs
        else:
            # Fallback was selected (ExternKernelCaller)
            winning_impl = default_impl
            winning_kwargs = non_tensor_args[0] if non_tensor_args else {}
            log.info(
                "   Range [%s, %s]: Fallback (default_impl) selected",
                range_start,
                range_end if range_end != float("inf") else "inf",
            )

        winning_config_patches = getattr(winning_choice, "config_patches", None) or {}

        # Create dataclass instances for cleaner code
        range_bounds = RangeBounds(range_start, range_end)
        impl_config = ImplConfig(
            impl_name=winning_impl.__name__,
            impl_func=winning_impl,
            kwargs=winning_kwargs,
            config_patches=winning_config_patches,
        )
        range_to_best_impl_map[range_bounds] = impl_config

        log.info(
            "   Range %s -> %s",
            range_bounds,
            impl_config.impl_name,
        )

    # Group ranges by implementation
    from torch.fx.experimental.symbolic_shapes import _ShapeEnvGuardError

    impl_groups = _group_ranges_by_impl(range_to_best_impl_map)

    log.info("After grouping by implementation: %d impl groups", len(impl_groups))
    for group in impl_groups:
        log.info("   %s", group)

    # If only one impl group remains, just inline that implementation
    if len(impl_groups) == 1:
        group = impl_groups[0]
        log.info("Only one implementation after grouping, directly inlining")
        return _lower_single_impl(
            group.impl_func,
            group.impl_kwargs,
            runtime_kwargs,
            tensor_inputs,
            name,
            config_patches=group.config_patches,
        )

    def dispatch_fn(*fake_tensors):
        """Build nested torch.cond dispatch: cond(pred1, impl1, cond(pred2, impl2, ...))."""
        num_impl_groups = len(impl_groups)
        if num_impl_groups < 2:
            raise RuntimeError(
                f"dispatch_fn requires at least 2 impl groups, got {num_impl_groups}"
            )

        dim_value = fake_tensors[0].size(dim_index)

        def build_range_predicate(ranges_list: list[RangeBounds]) -> torch.Tensor:
            predicates = []
            for rb in ranges_list:
                end = int(rb.end) if rb.end != float("inf") else None
                if end is None:
                    predicates.append(dim_value >= rb.start)
                else:
                    predicates.append((dim_value >= rb.start) & (dim_value <= end))

            result = predicates[0]
            for pred in predicates[1:]:
                result = result | pred
            return result  # pyrefly: ignore [bad-return]

        def build_nested_cond(idx: int):
            if idx >= num_impl_groups:
                raise RuntimeError(f"Invalid impl group index: {idx}")

            group = impl_groups[idx]
            merged_kwargs = _merge_config_and_runtime_kwargs(
                group.impl_kwargs, runtime_kwargs
            )

            @torch._dynamo.dont_skip_tracing
            def group_fn(*ops):
                return group.impl_func(*ops, **merged_kwargs)

            if idx == num_impl_groups - 1:
                return group_fn

            next_fn = build_nested_cond(idx + 1)

            @torch._dynamo.dont_skip_tracing
            def cond_wrapper(*ops, _ranges=group.ranges):
                return torch.cond(
                    pred=build_range_predicate(_ranges),
                    true_fn=group_fn,
                    false_fn=next_fn,
                    operands=ops,
                )

            return cond_wrapper

        return build_nested_cond(0)(*fake_tensors)

    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
        decomposition_table = select_decomp_table()
        shape_env = V.fake_mode.shape_env

        log.info("Tracing torch.cond dispatch with symbolic shapes...")

        try:
            context = (
                shape_env.error_on_new_guards
                if shape_env is not None
                else contextlib.nullcontext
            )
            with context():
                dispatch_gm = make_fx(
                    dispatch_fn,
                    decomposition_table=decomposition_table,
                    tracing_mode="symbolic",
                )(*fake_inputs)

            log.info("Successfully traced torch.cond dispatch")
            log.info("Traced graph:\n%s", dispatch_gm.graph)

        except (_ShapeEnvGuardError, AssertionError) as e:
            is_guard_error = isinstance(e, _ShapeEnvGuardError) or (
                isinstance(e, AssertionError)
                and "Guard attempted while ShapeEnv guards are frozen" in str(e)
            )
            if not is_guard_error:
                raise
            log.info("Dispatch function adds guards, skipping custom op lowering")
            counters["inductor"]["custom_op_decomp_guard_skips"] += 1
            return None

        except Exception:
            log.exception("make_fx tracing FAILED")
            raise

    ops_before = len(V.graph.operations)
    result = inline_subgraph_to_ir_nodes(dispatch_gm, tensor_inputs, f"{name}_dispatch")

    # Apply config_patches from all impl groups to inlined operations
    # TODO - consider conflicting patches
    merged_patches: dict[str, Any] = {}
    for group in impl_groups:
        merged_patches.update(group.config_patches)
    if merged_patches:
        _apply_config_patches_recursive(V.graph.operations[ops_before:], merged_patches)

    log.info(
        "Successfully created torch.cond dispatch for %d impl groups", len(impl_groups)
    )

    validate_ir(result)
    return result


def _create_autotuning_lowering(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    range_upper_bound: int,
    is_range_based: bool = False,
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
    dispatch_on: Optional[tuple[str, int]] = None,
    split_points: Optional[list[int]] = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> Callable[..., Any]:
    """Create the lowering function for autotuning."""
    if not is_range_based:
        # Standard autotuning path
        @functools.wraps(op_overload)
        def standard_lowering_wrapper(*args: Any, **kwargs: Any) -> Any:
            tensor_inputs, runtime_kwargs = _extract_tensor_inputs(
                args, kwargs, op_overload
            )
            return _standard_lowering_fn(
                processed_configs=processed_configs,
                default_impl=default_impl,
                name=name,
                op_overload=op_overload,
                input_gen_fns=input_gen_fns,
                tensor_inputs=tensor_inputs,
                runtime_kwargs=runtime_kwargs,
                config_generator=config_generator,
                min_speedup_threshold=min_speedup_threshold,
                benchmark_with_cudagraphs=benchmark_with_cudagraphs,
            )

        return standard_lowering_wrapper

    # Range-based autotuning path
    # pyrefly: ignore [not-iterable]
    tensor_name, dim_index = dispatch_on
    # pyrefly: ignore [bad-argument-type]
    ranges = _create_ranges_from_split_points(split_points)

    @functools.wraps(op_overload)
    def range_based_lowering_wrapper(*args: Any, **kwargs: Any) -> Any:
        tensor_inputs, runtime_kwargs = _extract_tensor_inputs(
            args, kwargs, op_overload
        )
        return _range_based_lowering_fn(
            processed_configs=processed_configs,
            default_impl=default_impl,
            name=name,
            op_overload=op_overload,
            input_gen_fns=input_gen_fns,
            tensor_name=tensor_name,
            dim_index=dim_index,
            ranges=ranges,
            tensor_inputs=tensor_inputs,
            runtime_kwargs=runtime_kwargs,
            range_upper_bound=range_upper_bound,
            config_generator=config_generator,
            min_speedup_threshold=min_speedup_threshold,
            benchmark_with_cudagraphs=benchmark_with_cudagraphs,
        )

    return range_based_lowering_wrapper


def register_custom_op_autotuning(
    custom_op: Union[torch._library.custom_ops.CustomOpDef, torch._ops.OpOverload],
    configs: Optional[Union[list[CustomOpConfig], list[Callable[..., Any]]]] = None,
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dispatch_on: Optional[dict[str, Any]] = None,
    split_points: Optional[list[int]] = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> None:
    """Register custom op for autotuning with custom_op configs where each config
    specifies a decomposition implementation function with its parameter values.
    It also supports Range-based autotuning to benchmark per range and generate
    runtime dispatch.

    Args:
        custom_op: Custom operation (CustomOpDef from @torch.library.custom_op) or
                   OpOverload (e.g., torch.ops.aten.mm.default)
        configs: List of CustomOpConfig objects for static inputs. Mutually exclusive with config_generator.
        config_generator: Dynamic config generator function that takes a dict mapping
                          parameter names to fake tensors, and returns list[CustomOpConfig]
                          based on input tensor properties. Mutually exclusive with configs.
        name: Operation name (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking
        dispatch_on: Dict for range-based dispatch with keys:
            - 'tensor_name': Name of tensor parameter to dispatch on
            - 'dim': Dimension index to check size
            - 'unbounded_size' (optional): Benchmark size for the unbounded (last) range, such
                as [2048, inf] -> [2048, unbounded_size]. Set based on your expected workload size.
                Default is DEFAULT_RANGE_UPPER_BOUND=65536.
        split_points: List of range endpoints in ascending order for range-based autotuning
        min_speedup_threshold: Only pick a non-fallback choice if it beats the fallback
            by at least this ratio. Default is 1.0 (any speedup wins). Set to e.g. 1.1
            to require 10% speedup over fallback.
        benchmark_with_cudagraphs: If True, benchmark the fallback kernel using CUDA graph
            capture and replay for fair comparison with compiled kernels. Default is False.

    The default/fallback implementation is automatically derived:
    - For CustomOpDef: Uses the decorated function
    - For OpOverload: Traces the op call, which falls through to normal inductor lowering

    Examples:
        # Static configs
        @torch.library.custom_op("mylib::attention", mutates_args=())
        def my_attention(query, key, value, head_dim=32):
            ...

        register_custom_op_autotuning(
            my_attention,
            configs=[
                CustomOpConfig(attention_impl, head_dim=32, method='chunked'),
                CustomOpConfig(attention_impl, head_dim=64, method='tiled'),
                CustomOpConfig(head_dim=128),  # No decomposition specified, use default
            ],
            input_gen_fns={
                "query": lambda fake: torch.randn_like(fake, device='cuda'),
                "key": lambda fake: torch.randn_like(fake, device='cuda'),
                "value": lambda fake: torch.randn_like(fake, device='cuda'),
            },
        )

        # Dynamic config generation based on input tensor properties
        def generate_k_split_configs(fake_tensors: dict[str, torch.Tensor]) -> list[CustomOpConfig]:
            # Access tensor shapes, dtypes, devices, etc.
            m, k = fake_tensors["mat1"].shape
            _, n = fake_tensors["mat2"].shape
            k_splits = ... # compute possible k splits based on tensor properties
            return [CustomOpConfig(k_splits=k) for k in k_splits]

        register_custom_op_autotuning(
            matmul_decomposeK_op,
            config_generator=generate_k_split_configs,
            input_gen_fns={...},
        )

    Range-based Example:
        register_custom_op_autotuning(
            my_op,
            configs=[CustomOpConfig(impl1), CustomOpConfig(impl2), CustomOpConfig(impl3)],
            dispatch_on={
                # Dispatch based on x.shape[1]
                "tensor_name": "x",
                "dim": 1,
                # Optional Benchmark size used for the unbounded range [2049, inf].
                # Since inf is not a concrete value, we use range_upper_bound as the benchmark size.
                # Default value is 65536 (DEFAULT_RANGE_UPPER_BOUND) if not provided.
                "range_upper_bound": 8192,
            },
            split_points=[512, 2048],  # Creates ranges: [1,512], [513,2048], [2049, 8192]
        )
    """
    from torch._library.custom_ops import CustomOpDef

    # Handle both CustomOpDef and OpOverload - derive impl_fn automatically
    # Both cases call through op_overload so fake handlers are used during tracing
    if isinstance(custom_op, CustomOpDef):
        op_overload = custom_op._opoverload
    elif isinstance(custom_op, torch._ops.OpOverload):
        op_overload = custom_op
    else:
        raise TypeError(
            f"custom_op must be a CustomOpDef or OpOverload, got {type(custom_op)}."
        )

    # impl_fn calls through op_overload so fake handlers are used during tracing
    def impl_fn(*args, **kwargs):
        return op_overload(*args, **kwargs)

    # Validate configs and config_generator are mutually exclusive
    if configs is not None and config_generator is not None:
        raise ValueError(
            "Cannot specify both 'configs' and 'config_generator'. "
            "Use 'config_generator' for shape-dependent configs."
        )

    if configs is None and config_generator is None:
        raise ValueError("Must specify either 'configs' or 'config_generator'")

    # Process and validate static configs at registration time
    static_configs = None
    if configs is not None:
        if not isinstance(configs, (list, tuple)):
            raise TypeError(f"configs must be a list or tuple, got {type(configs)}")

        static_configs = []
        for cfg in configs:
            if isinstance(cfg, CustomOpConfig):
                static_configs.append(cfg)
            else:
                raise TypeError(
                    f"Each config must be a CustomOpConfig object, got {type(cfg)}"
                )

        if not static_configs:
            raise ValueError("At least one config must be provided")

    if name is None:
        name = f"{op_overload._name}_autotuned"

    # Validate range-based parameters
    is_range_based = dispatch_on is not None or split_points is not None
    dispatch_on_tuple: Optional[tuple[str, int]] = None
    range_upper_bound = DEFAULT_RANGE_UPPER_BOUND
    if is_range_based:
        if dispatch_on is None or split_points is None:
            raise ValueError(
                "Both dispatch_on and split_points must be specified for range-based autotuning"
            )
        if not isinstance(dispatch_on, dict):
            raise ValueError(
                "dispatch_on must be a dict with 'tensor_name' and 'dim' keys, "
                f"e.g., {{'tensor_name': 'x', 'dim': 1}}. Got: {type(dispatch_on)}"
            )
        if "tensor_name" not in dispatch_on or "dim" not in dispatch_on:
            raise ValueError(
                "dispatch_on must contain 'tensor_name' and 'dim' keys, "
                f"e.g., {{'tensor_name': 'x', 'dim': 1}}. Got keys: {list(dispatch_on.keys())}"
            )
        if not isinstance(dispatch_on["tensor_name"], str):
            raise ValueError(
                f"dispatch_on['tensor_name'] must be a string (tensor parameter name), "
                f"got {type(dispatch_on['tensor_name'])}"
            )
        if not isinstance(dispatch_on["dim"], int):
            raise ValueError(
                f"dispatch_on['dim'] must be an integer (dimension index), "
                f"got {type(dispatch_on['dim'])}"
            )
        dispatch_on_tuple = (dispatch_on["tensor_name"], dispatch_on["dim"])
        range_upper_bound = dispatch_on.get(
            "range_upper_bound", DEFAULT_RANGE_UPPER_BOUND
        )
        if not isinstance(range_upper_bound, int) or range_upper_bound <= 0:
            raise ValueError(
                f"dispatch_on['range_upper_bound'] must be a positive integer, "
                f"got {range_upper_bound}"
            )
        if not isinstance(split_points, list) or len(split_points) == 0:
            raise ValueError("split_points must be a non-empty list of integers")
        if sorted(split_points) != split_points:
            raise ValueError("split_points must be sorted in ascending order")

    # Create and register the lowering function
    lowering_fn = _create_autotuning_lowering(
        # pyrefly: ignore [bad-argument-type]
        processed_configs=static_configs,
        default_impl=impl_fn,
        name=name,
        op_overload=op_overload,
        input_gen_fns=input_gen_fns,
        is_range_based=is_range_based,
        config_generator=config_generator,
        dispatch_on=dispatch_on_tuple,
        split_points=split_points,
        range_upper_bound=range_upper_bound,
        min_speedup_threshold=min_speedup_threshold,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )

    # Register in user_lowerings which takes priority over built-in lowerings
    # The dispatch in graph.py checks user_lowerings first with recursion guard
    user_lowerings[op_overload] = lowering_fn
