# Owner(s): ["module: inductor"]

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, ir_node_to_tensor, StorageBox, TensorBox
from torch._inductor.lowering import lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)
from torch._inductor.utils import convert_symint_to_expr
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


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
    # Pre-traced GraphModule to avoid shape capture during dispatch_fn tracing
    traced_gm: Optional[Any] = field(default=None, compare=False, hash=False, repr=False)

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
        **params: Parameters passed to the function

    Examples:
        CustomOpConfig(attention_impl, head_dim=32, method='chunked')
        CustomOpConfig(head_dim=32, method='chunked')
    """

    def __init__(
        self,
        decomposition: Optional[Callable[..., Any]] = None,
        **params: Any,
    ):
        if decomposition is not None and not callable(decomposition):
            raise TypeError(
                f"decomposition must be callable, got {type(decomposition)}"
            )

        self.decomposition = decomposition
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
    "get_registered_aten_autotuning",
    "should_use_batch1_decompose",
    "get_valid_k_splits",
]


# =============================================================================
# Pre-pruning heuristics for mm decompositions
# =============================================================================

def should_use_batch1_decompose(k: int, n: int, max_kn_product: int = 4_000_000) -> bool:
    """Heuristic: batch1_decompose is only efficient when K*N is small.

    batch1_decompose creates an intermediate tensor of shape (M, K, N) via
    unsqueeze+mul, which is memory-bound. Only beneficial when K*N is small
    enough that the reduction kernel can be efficient.

    Args:
        k: K dimension of the matmul (mat1 cols / mat2 rows)
        n: N dimension of the matmul (mat2 cols / output cols)
        max_kn_product: Maximum K*N product to consider batch1_decompose.
                       Default 4M based on benchmarking (router_256 has K*N=1.6M).

    Returns:
        True if batch1_decompose should be included in autotuning choices.
    """
    return k * n < max_kn_product


def get_valid_k_splits(k: int, min_k: int = 1024) -> list[int]:
    """Get valid K-split values for decompose_k.

    decompose_k splits the K dimension into k_splits parts and uses batched
    matmul. Only beneficial for large K where the reduction overhead is amortized.

    Args:
        k: K dimension of the matmul
        min_k: Minimum K value to consider decompose_k. Default 1024.

    Returns:
        List of valid k_split values that evenly divide K.
    """
    if k < min_k:
        return []

    # Standard k_split values to try
    candidate_splits = [8, 12, 16, 24, 32, 48, 64]
    return [ks for ks in candidate_splits if k % ks == 0]

# Registry for aten op autotuning configurations
# Maps op_overload -> lowering function
_aten_op_autotuning_registry: dict[torch._ops.OpOverload, Callable[..., Any]] = {}


def get_registered_aten_autotuning(
    op_overload: torch._ops.OpOverload,
) -> Optional[Callable[..., Any]]:
    """Check if an aten op has registered autotuning configuration.

    Args:
        op_overload: The aten op overload to check (e.g., aten.mm.default)

    Returns:
        The registered lowering function if found, None otherwise
    """
    return _aten_op_autotuning_registry.get(op_overload)


def clear_aten_autotuning_registry() -> None:
    """Clear all registered aten op autotuning configurations.

    This is primarily useful for testing to ensure a clean state.
    """
    _aten_op_autotuning_registry.clear()


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
    """Merge config parameters with runtime kwargs. Runtime kwargs take precedence.
       If there are conflicts, log a warning and use runtime value.

    Args:
        config_params: Parameters from CustomOpConfig
        runtime_kwargs: Runtime non-tensor kwargs from _extract_tensor_inputs

    Returns:
        Merged kwargs dictionary with runtime values taking precedence
    """
    merged_kwargs = config_params.copy()

    # Check for conflicts and let runtime kwargs dominate
    conflicts = OrderedSet(config_params.keys()).intersection(runtime_kwargs.keys())

    for key in conflicts:
        log.warning(
            "Parameter '%s' specified both in CustomOpConfig (%s) "
            "and at runtime (%s). Using runtime value.",
            key,
            config_params[key],
            runtime_kwargs[key],
        )

    # Runtime kwargs override config params
    merged_kwargs.update(runtime_kwargs)

    return merged_kwargs


def _adapt_user_input_gen_fns(
    inputs: list[Any],
    arg_names: list[str],
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[int, Callable[[Any], torch.Tensor]]:
    """Convert user input generators from name-based to index-based format.
       Inductor autotune's input_gen_fns expects index of arg_names as key.

    Uses V.graph.sizevars.size_hints() to guess best for dynamic shapes.
    """

    name_to_index = {name: i for i, name in enumerate(arg_names)}
    index_based_fns = {}

    for name, gen_fn in user_input_gen_fns.items():
        if name in name_to_index:
            index_based_fns[name_to_index[name]] = gen_fn
        else:
            log.warning(
                "Unknown argument name '%s' in input_gen_fns. "
                "Available argument names: %s",
                name,
                list(name_to_index.keys()),
            )

    def create_internal_input_gen_fn(
        user_function: Callable[[torch.Tensor], torch.Tensor], arg_name: str
    ) -> Callable[[Any], torch.Tensor]:
        """Create internal input generator that converts IR buffer to user's fake tensor."""

        def internal_input_gen_fn(ir_buffer: Any) -> torch.Tensor:
            fake_tensor = ir_node_to_tensor(ir_buffer, guard_shape=False)
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
    if not range_to_best_impl:
        return []

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
    benchmark_size_strategy: str = "upper",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create input generator that modifies target dimension based on strategy.

    Args:
        benchmark_size_strategy: How to pick the representative size for benchmarking.
            - "upper": Use top of range (default) - best for memory-bound ops
            - "lower": Use bottom of range - best for compute-bound ops
            - "midpoint": Use midpoint of range - balanced approach

    range_upper_bound: Size to use for benchmarking when range_end is unbounded.
    Default is DEFAULT_RANGE_UPPER_BOUND = 65536
    """
    from torch._inductor.ir import get_fill_order
    from torch._inductor.kernel.flex.common import construct_strides

    # Calculate the effective upper bound for this range
    effective_end = range_upper_bound if range_end == float("inf") else int(range_end)

    # Pick target dimension based on strategy
    if benchmark_size_strategy == "lower":
        target_dim = int(range_start)
    elif benchmark_size_strategy == "midpoint":
        target_dim = (int(range_start) + effective_end) // 2
    else:  # "upper" (default)
        target_dim = effective_end

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
    name: str,
    default_impl: Callable[..., Any],
    fake_output: torch.Tensor,
    kwargs: dict[str, Any],
) -> ExternKernelChoice:
    """Create fallback choice for default implementation."""

    def fallback_wrapper(*args: Any) -> Any:
        return default_impl(*args, **kwargs)

    return ExternKernelChoice(
        kernel=fallback_wrapper,
        name=f"{name}_fallback_default",
        has_out_variant=False,
        op_overload=default_impl,
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
    return_choice: bool = False,
    config_patches: Optional[dict[str, Any]] = None,
    benchmark_with_cudagraphs: bool = False,
    min_speedup_threshold: float = 1.0,
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
        config_patches: Optional config patches to apply during subgraph compilation
                       (e.g., {"coordinate_descent_tuning": True})
        benchmark_with_cudagraphs: If True, capture into CUDA graph before benchmarking

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
        import inspect

        # Get parameter names - prefer op_overload schema for aten ops
        arg_names: list[str] = []
        if op_overload is not None and hasattr(op_overload, "_schema"):
            # For aten ops, get parameter names from schema
            schema = op_overload._schema
            arg_names = [arg.name for arg in schema.arguments if arg.type.isSubtypeOf(torch._C.TensorType.get())]
        elif decompositions:
            # For custom ops, try inspect
            try:
                arg_names = list(inspect.signature(decompositions[0]).parameters.keys())
            except (ValueError, TypeError):
                # Fallback for builtins
                arg_names = [f"arg{i}" for i in range(len(inputs))]
        input_gen_fns = _adapt_user_input_gen_fns(inputs, arg_names, user_input_gen_fns)

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=decompositions,
        # pyrefly: ignore [no-matching-overload]
        input_nodes=list(inputs),
        non_tensor_args=non_tensor_args,
        input_gen_fns=input_gen_fns if input_gen_fns else None,
        config_patches=config_patches,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )

    # Add default implementation as fallback
    if op_overload and hasattr(op_overload, "_op"):
        fallback_name = f"{name}_fallback_default"
        from torch._inductor.select_algorithm import extern_kernels

        with V.fake_mode:
            # pyrefly: ignore [no-matching-overload]
            fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]
            fallback_kwargs = non_tensor_args[0] if non_tensor_args else {}
            fake_output = op_overload(*fake_inputs, **fallback_kwargs)

        output_size = tuple(convert_symint_to_expr(s) for s in fake_output.shape)
        output_stride = tuple(
            convert_symint_to_expr(s) for s in fake_output.stride()
        )

        # Only create ExternKernelChoice if not already registered (to avoid assertion)
        # But always add a fallback choice to the choices list
        if not hasattr(extern_kernels, fallback_name):
            fallback_choice = _create_fallback_choice(
                name, op_overload, fake_output, fallback_kwargs
            )
        else:
            # Reuse existing kernel by creating a wrapper ExternKernelChoice
            existing_kernel = getattr(extern_kernels, fallback_name)
            fallback_choice = ExternKernelChoice(
                kernel=existing_kernel,
                name=fallback_name,
                has_out_variant=False,
                op_overload=op_overload,
                use_fallback_kernel=True,
                _skip_registration=True,
            )

        fallback_choice.maybe_append_choice(
            choices=choices,
            input_nodes=list(inputs),
            layout=FixedLayout(
                device=fake_output.device,
                dtype=fake_output.dtype,
                size=output_size,
                stride=output_stride,
            ),
            benchmark_with_cudagraphs=benchmark_with_cudagraphs,
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
    )

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

        result = inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)
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
    default_impl: Callable[..., Any],
    operation_name: str,
    op_overload: Optional[torch._ops.OpOverload] = None,
) -> list[CustomOpConfig]:
    """Generate configs dynamically based on input tensors at lowering time."""
    import inspect

    # Get parameter names - use schema for aten ops, inspect for custom ops
    if op_overload is not None and hasattr(op_overload, "_schema"):
        # For aten ops, get parameter names from schema
        schema = op_overload._schema
        param_names = [arg.name for arg in schema.arguments if arg.type.isSubtypeOf(torch._C.TensorType.get())]
    else:
        # For custom ops, use inspect
        try:
            sig = inspect.signature(default_impl)
            param_names = list(sig.parameters.keys())
        except (ValueError, TypeError):
            # Fallback for builtins - use generic names
            param_names = [f"arg{i}" for i in range(len(tensor_inputs))]

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
        raise ValueError(f"config_generator returned empty list for {operation_name}. ")

    return list(configs)


def _prepare_configs_and_decompositions(
    processed_configs: Optional[list[CustomOpConfig]],
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ],
    tensor_inputs: list[Any],
    default_impl: Callable[..., Any],
    runtime_kwargs: dict[str, Any],
    name: str,
    op_overload: Optional[torch._ops.OpOverload] = None,
) -> tuple[list[Callable], list[dict[str, Any]]]:
    """Prepare decompositions and merged kwargs from configs.

    Handles both static configs and dynamic config generation.
    Merges config params with runtime kwargs (runtime takes precedence).
    """
    # Get configs: either generate dynamically or use static configs
    if config_generator is not None:
        configs_to_use = _generate_dynamic_configs(
            tensor_inputs, config_generator, default_impl, name, op_overload
        )
    else:
        assert processed_configs is not None
        configs_to_use = processed_configs

    # Prepare decompositions and kwargs for autotuning
    decompositions = []
    non_tensor_args = []

    for cfg in configs_to_use:
        decomp = cfg.get_decomposition(default_impl=default_impl)
        decompositions.append(decomp)

        # Merge config params with runtime kwargs (runtime takes precedence)
        merged_kwargs = _merge_config_and_runtime_kwargs(cfg.params, runtime_kwargs)
        non_tensor_args.append(merged_kwargs)

    return decompositions, non_tensor_args


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
    config_patches: Optional[dict[str, Any]] = None,
    benchmark_with_cudagraphs: bool = False,
) -> Any:
    """Standard autotuning lowering function."""
    decompositions, non_tensor_args = _prepare_configs_and_decompositions(
        processed_configs,
        config_generator,
        tensor_inputs,
        default_impl,
        runtime_kwargs,
        name,
        op_overload,
    )

    result = autotune_custom_op(
        name=name,
        decompositions=decompositions,
        inputs=tensor_inputs,
        non_tensor_args=non_tensor_args,
        op_overload=op_overload,
        user_input_gen_fns=input_gen_fns,
        config_patches=config_patches,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )

    validate_ir(result)
    return result


def _lower_single_impl(
    impl: Callable[..., Any],
    impl_kwargs: dict[str, Any],
    runtime_kwargs: dict[str, Any],
    tensor_inputs: list[Any],
    name: str,
    config_patches: Optional[dict[str, Any]] = None,
) -> Any:
    """Lower a single implementation by tracing and inlining it.

    Args:
        impl: The implementation function
        impl_kwargs: Kwargs for the implementation (from CustomOpConfig)
        runtime_kwargs: Runtime kwargs from the op call
        tensor_inputs: Input tensors/IR nodes
        name: Name for the lowered implementation
        config_patches: Optional config patches to tag operations with
            (e.g., {"coordinate_descent_tuning": True} for decomposition ops)
    """
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    from torch.fx.experimental.proxy_tensor import make_fx

    from ..decomposition import select_decomp_table

    merged_kwargs = _merge_config_and_runtime_kwargs(impl_kwargs, runtime_kwargs)

    def impl_wrapper(*tensors):
        return impl(*tensors, **merged_kwargs)

    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
        decomposition_table = select_decomp_table()
        impl_gm = make_fx(
            impl_wrapper,
            decomposition_table=decomposition_table,
            tracing_mode="symbolic",
        )(*fake_inputs)

    log.info("Inlining implementation: %s", impl.__name__)
    result = inline_subgraph_to_ir_nodes(
        impl_gm, tensor_inputs, name, config_patches=config_patches
    )
    validate_ir(result)
    return result


def _build_cond_dispatch_graph(
    impl_groups: list[RangeImplGroup],
    dim_index: int,
    fake_inputs: tuple[torch.Tensor, ...],
) -> torch.fx.GraphModule:
    """Build an FX graph with nested torch.cond for range-based dispatch.

    Constructs the graph manually instead of using make_fx to avoid
    shape accesses in implementations being captured as extra cond operands.
    Each branch references a pre-traced GraphModule (group.traced_gm).

    For N groups, builds: cond(p0, g0, cond(p1, g1, ... gN-1))
    Each inner cond is a separate GraphModule subgraph.
    """
    import operator

    with V.fake_mode:
        fake_out = impl_groups[0].traced_gm(*fake_inputs)

    def _make_cond_graph(
        groups: list[RangeImplGroup],
        start_idx: int,
    ) -> torch.fx.GraphModule:
        """Build a GraphModule containing a single cond dispatching groups[0] vs rest."""
        graph = torch.fx.Graph()
        modules: dict[str, torch.nn.Module] = {}

        # Placeholders for tensor inputs
        input_nodes = []
        for i, inp in enumerate(fake_inputs):
            ph = graph.placeholder(f"arg{i}")
            ph.meta["val"] = inp
            input_nodes.append(ph)

        # sym_size for dispatch
        dim_size = graph.call_function(
            torch.ops.aten.sym_size.int, (input_nodes[0], dim_index)
        )
        dim_size.meta["val"] = fake_inputs[0].size(dim_index)

        # Build predicate for the first group
        pred_nodes = []
        for rb in groups[0].ranges:
            end = int(rb.end) if rb.end != float("inf") else None
            has_lower = rb.start > 1  # dim >= 1 always, skip trivial check
            has_upper = end is not None

            if has_lower and has_upper:
                ge_n = graph.call_function(operator.ge, (dim_size, rb.start))
                ge_n.meta["val"] = dim_size.meta["val"] >= rb.start
                le_n = graph.call_function(operator.le, (dim_size, end))
                le_n.meta["val"] = dim_size.meta["val"] <= end
                p = graph.call_function(operator.and_, (ge_n, le_n))
                p.meta["val"] = bool(ge_n.meta["val"]) and bool(le_n.meta["val"])
            elif has_upper:
                p = graph.call_function(operator.le, (dim_size, end))
                p.meta["val"] = dim_size.meta["val"] <= end
            elif has_lower:
                p = graph.call_function(operator.ge, (dim_size, rb.start))
                p.meta["val"] = dim_size.meta["val"] >= rb.start
            else:
                # start <= 1 and unbounded: always true (fallback case)
                p = graph.call_function(operator.ge, (dim_size, 1))
                p.meta["val"] = True
            pred_nodes.append(p)

        pred = pred_nodes[0]
        for p in pred_nodes[1:]:
            pred = graph.call_function(operator.or_, (pred, p))
            pred.meta["val"] = True

        # true_fn = this group's implementation
        modules["true_graph_0"] = groups[0].traced_gm
        true_attr = graph.get_attr("true_graph_0")
        true_attr.meta["val"] = None

        # false_fn = either the last implementation or a nested cond
        if len(groups) == 2:
            modules["false_graph_0"] = groups[1].traced_gm
        else:
            # Recursively build the inner cond for remaining groups
            modules["false_graph_0"] = _make_cond_graph(groups[1:], start_idx + 1)
        false_attr = graph.get_attr("false_graph_0")
        false_attr.meta["val"] = None

        operands_tuple = tuple(input_nodes)
        cond_node = graph.call_function(
            torch.ops.higher_order.cond,
            (pred, true_attr, false_attr, operands_tuple),
        )
        cond_node.meta["val"] = (fake_out,)

        getitem = graph.call_function(operator.getitem, (cond_node, 0))
        getitem.meta["val"] = fake_out

        graph.output(getitem)
        return torch.fx.GraphModule(modules, graph)

    return _make_cond_graph(impl_groups, 0)


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
    range_config_generator: Optional[
        Callable[[dict[str, torch.Tensor], int, Union[int, float]], list[CustomOpConfig]]
    ] = None,
    config_patches: Optional[dict[str, Any]] = None,
    benchmark_with_cudagraphs: bool = False,
    benchmark_size_strategy: str = "upper",
    min_speedup_threshold: float = 1.0,
) -> Any:
    """Range-based autotuning lowering function.

    Args:
        range_config_generator: Optional function (tensors, range_start, range_end) -> configs
            that generates configs specific to each range. If provided, this is used instead
            of config_generator for each range, allowing different configs per range.
        config_patches: Optional config patches to apply during subgraph compilation.
        benchmark_with_cudagraphs: If True, capture into CUDA graph before benchmarking.
    """
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    from torch.fx.experimental.proxy_tensor import make_fx

    from ..decomposition import select_decomp_table

    log.info("=== Range-based Autotuning for %s ===", name)
    log.info("Dispatch on: %s[%d], Ranges: %s", tensor_name, dim_index, ranges)

    # If no range_config_generator, prepare configs once for all ranges
    if range_config_generator is None:
        decompositions, non_tensor_args = _prepare_configs_and_decompositions(
            processed_configs,
            config_generator,
            tensor_inputs,
            default_impl,
            runtime_kwargs,
            name,
            op_overload,
        )

    range_to_best_impl_map: dict[RangeBounds, ImplConfig] = {}

    # Benchmark each range and collect winning implementations
    for range_start, range_end in ranges:
        if input_gen_fns and tensor_name in input_gen_fns:
            base_gen_fn = input_gen_fns[tensor_name]
        else:
            base_gen_fn = _default_input_gen_fn

        range_gen_fn = _create_range_input_gen_fn(
            base_gen_fn, dim_index, range_start, range_end, range_upper_bound,
            benchmark_size_strategy=benchmark_size_strategy,
        )
        range_input_gen_fns = {**(input_gen_fns or {}), tensor_name: range_gen_fn}

        range_name = f"{name}_range_{int(range_start)}_{int(range_end) if range_end != float('inf') else 'inf'}"

        # If range_config_generator provided, generate configs specific to this range
        if range_config_generator is not None:
            decompositions, non_tensor_args = _prepare_configs_and_decompositions(
                processed_configs,
                lambda tensors, rs=range_start, re=range_end: range_config_generator(tensors, rs, re),
                tensor_inputs,
                default_impl,
                runtime_kwargs,
                name,
                op_overload,
            )

        # pyrefly: ignore [not-iterable]
        autotuned_result, winning_choice = autotune_custom_op(
            name=range_name,
            decompositions=decompositions,
            inputs=tensor_inputs,
            non_tensor_args=non_tensor_args,
            op_overload=op_overload,
            user_input_gen_fns=range_input_gen_fns,
            return_choice=True,
            config_patches=config_patches,
            benchmark_with_cudagraphs=benchmark_with_cudagraphs,
            min_speedup_threshold=min_speedup_threshold,
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

        # Create dataclass instances for cleaner code
        range_bounds = RangeBounds(range_start, range_end)
        impl_config = ImplConfig(
            impl_name=winning_impl.__name__,
            impl_func=winning_impl,
            kwargs=winning_kwargs,
        )
        range_to_best_impl_map[range_bounds] = impl_config

        log.info(
            "   Range %s -> %s",
            range_bounds,
            impl_config.impl_name,
        )

    # Group ranges by implementation (more aggressive than adjacent merging)
    impl_groups = _group_ranges_by_impl(range_to_best_impl_map)

    log.info("After grouping by implementation: %d impl groups", len(impl_groups))
    for group in impl_groups:
        log.info("   %s", group)

    # DEBUG: Print whether we would use torch.cond
    if len(impl_groups) > 1:
        log.warning(
            "TORCH_COND PATH: %d different implementations, will generate torch.cond dispatch",
            len(impl_groups),
        )
    else:
        log.warning(
            "SINGLE_IMPL PATH: All ranges use same implementation, no torch.cond needed",
        )

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
            config_patches=config_patches,
        )

    # Pre-trace each implementation into a GraphModule, then build the dispatch
    # graph manually. This matches how dynamo handles torch.cond: the branch
    # subgraphs only take tensor operands as inputs.
    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
        decomposition_table = select_decomp_table()

        for group in impl_groups:
            merged_kwargs = _merge_config_and_runtime_kwargs(
                group.impl_kwargs, runtime_kwargs
            )

            def impl_wrapper(
                *tensors, _impl=group.impl_func, _kwargs=merged_kwargs
            ):
                return _impl(*tensors, **_kwargs)

            group.traced_gm = make_fx(
                impl_wrapper,
                decomposition_table=decomposition_table,
                tracing_mode="symbolic",
            )(*fake_inputs)

    dispatch_gm = _build_cond_dispatch_graph(
        impl_groups, dim_index, fake_inputs
    )
    log.info("Built dispatch graph:\n%s", dispatch_gm.graph)

    result = inline_subgraph_to_ir_nodes(dispatch_gm, tensor_inputs, f"{name}_dispatch")

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
    range_config_generator: Optional[
        Callable[[dict[str, torch.Tensor], int, Union[int, float]], list[CustomOpConfig]]
    ] = None,
    dispatch_on: Optional[tuple[str, int]] = None,
    split_points: Optional[list[int]] = None,
    config_patches: Optional[dict[str, Any]] = None,
    benchmark_with_cudagraphs: bool = False,
    benchmark_size_strategy: str = "upper",
    min_speedup_threshold: float = 1.0,
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
                config_patches=config_patches,
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
            range_config_generator=range_config_generator,
            config_patches=config_patches,
            benchmark_with_cudagraphs=benchmark_with_cudagraphs,
            benchmark_size_strategy=benchmark_size_strategy,
            min_speedup_threshold=min_speedup_threshold,
        )

    return range_based_lowering_wrapper


def register_custom_op_autotuning(
    custom_op: Union[
        "torch._library.custom_ops.CustomOpDef",
        torch._ops.OpOverload,
    ],
    configs: Optional[Union[list[CustomOpConfig], list[Callable[..., Any]]]] = None,
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
    range_config_generator: Optional[
        Callable[[dict[str, torch.Tensor], int, Union[int, float]], list[CustomOpConfig]]
    ] = None,
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dispatch_on: Optional[dict[str, Any]] = None,
    split_points: Optional[list[int]] = None,
    config_patches: Optional[dict[str, Any]] = None,
    benchmark_with_cudagraphs: bool = False,
    benchmark_size_strategy: str = "upper",
    default_impl: Optional[Callable[..., Any]] = None,
    min_speedup_threshold: float = 1.0,
) -> None:
    """Register custom op or aten op for autotuning with configs where each config
    specifies a decomposition implementation function with its parameter values.
    It also supports Range-based autotuning to benchmark per range and generate
    runtime dispatch.

    This API supports both custom ops (from @torch.library.custom_op) and builtin
    aten ops (e.g., torch.ops.aten.mm.default). For aten ops, the autotuning
    configuration is stored in a registry and checked by the op's lowering.

    Args:
        custom_op: Either a custom operation (from @torch.library.custom_op) or
                   an aten op overload (e.g., torch.ops.aten.mm.default)
        configs: List of CustomOpConfig objects for static inputs. Mutually exclusive with config_generator.
        config_generator: Dynamic config generator function that takes a dict mapping
                          parameter names to fake tensors, and returns list[CustomOpConfig]
                          based on input tensor properties. Mutually exclusive with configs.
        range_config_generator: For range-based autotuning, generates configs specific to each range.
                               Function signature: (tensors: dict, range_start: int, range_end: int|float) -> list[CustomOpConfig]
                               Allows different configs per range to reduce autotuning time.
        name: Operation name (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking
        dispatch_on: Dict for range-based dispatch with keys:
            - 'tensor_name': Name of tensor parameter to dispatch on
            - 'dim': Dimension index to check size
            - 'unbounded_size' (optional): Benchmark size for the unbounded (last) range, such
                as [2048, inf] -> [2048, unbounded_size]. Set based on your expected workload size.
                Default is DEFAULT_RANGE_UPPER_BOUND=65536.
        split_points: List of range endpoints in ascending order for range-based autotuning
        config_patches: Config patches to apply during subgraph compilation for autotuning
                       (e.g., {"coordinate_descent_tuning": True} to enable coordinate descent
                       tuning only for custom op benchmarks without enabling it globally)
        benchmark_with_cudagraphs: If True, capture into CUDA graph before benchmarking.
                                  This can provide more accurate timing for operations that
                                  benefit from CUDA graph replay.
        benchmark_size_strategy: For range-based dispatch, how to pick the representative
                                size for benchmarking each range. Options:
                                - "upper": Use top of range (default) - benchmark at largest size
                                - "lower": Use bottom of range - benchmark at smallest size
                                - "midpoint": Use midpoint of range - balanced approach
        min_speedup_threshold: Minimum speedup ratio required for a decomposition to be
                              selected over the fallback (default implementation). A value of
                              1.1 means the decomposition must be at least 10% faster.
                              Default is 1.0 (any faster decomposition wins).

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
        default_impl: For aten ops, the default implementation function (e.g., torch.mm).
                      Not needed for custom ops as it's extracted automatically.

    Aten Op Example:
        # Register autotuning for aten.mm
        def mm_config_generator(tensors):
            m, k = tensors["mat1"].shape
            n = tensors["mat2"].shape[1]
            configs = [CustomOpConfig(torch.mm)]  # cuBLAS baseline
            if k >= 6144 and n <= 512:
                configs.append(CustomOpConfig(batch1_decompose))
                configs.append(CustomOpConfig(make_dec_k(32)))
            return configs

        register_custom_op_autotuning(
            torch.ops.aten.mm.default,
            config_generator=mm_config_generator,
            dispatch_on={"tensor_name": "mat1", "dim": 0, "range_upper_bound": 16},
            split_points=[4],
            default_impl=torch.mm,
        )
    """
    from torch._library.custom_ops import CustomOpDef

    # Determine if this is a custom op or aten op
    is_aten_op = isinstance(custom_op, torch._ops.OpOverload)
    is_custom_op = isinstance(custom_op, CustomOpDef)

    if not is_aten_op and not is_custom_op:
        raise TypeError(
            f"custom_op must be a CustomOpDef (from @torch.library.custom_op) or "
            f"an OpOverload (e.g., torch.ops.aten.mm.default), got {type(custom_op)}."
        )

    # Validate configs and config_generator are mutually exclusive
    if configs is not None and config_generator is not None:
        raise ValueError(
            "Cannot specify both 'configs' and 'config_generator'. "
            "Use 'config_generator' for shape-dependent configs."
        )

    # Allow range_config_generator as alternative for range-based autotuning
    if configs is None and config_generator is None and range_config_generator is None:
        raise ValueError(
            "Must specify either 'configs', 'config_generator', or 'range_config_generator'"
        )

    # Extract op_overload and default_impl based on op type
    if is_custom_op:
        op_overload = custom_op._opoverload
        impl_fn = custom_op._init_fn
    else:
        # Aten op
        op_overload = custom_op
        if default_impl is None:
            raise ValueError(
                "default_impl must be provided for aten ops "
                "(e.g., default_impl=torch.mm for aten.mm)"
            )
        impl_fn = default_impl

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
        if not isinstance(split_points, list):
            raise ValueError("split_points must be a list of integers")
        if split_points and sorted(split_points) != split_points:
            raise ValueError("split_points must be sorted in ascending order")

    # Create the lowering function
    lowering_fn = _create_autotuning_lowering(
        # pyrefly: ignore [bad-argument-type]
        processed_configs=static_configs,
        default_impl=impl_fn,
        name=name,
        op_overload=op_overload,
        input_gen_fns=input_gen_fns,
        is_range_based=is_range_based,
        config_generator=config_generator,
        range_config_generator=range_config_generator,
        dispatch_on=dispatch_on_tuple,
        split_points=split_points,
        range_upper_bound=range_upper_bound,
        config_patches=config_patches,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
        benchmark_size_strategy=benchmark_size_strategy,
        min_speedup_threshold=min_speedup_threshold,
    )

    # Register the lowering function
    if is_aten_op:
        # For aten ops, store in registry so the op's existing lowering can check it
        _aten_op_autotuning_registry[op_overload] = lowering_fn
        log.info("Registered aten op autotuning for %s", op_overload)
    else:
        # For custom ops, replace the lowering directly
        lowerings[op_overload] = lowering_fn
