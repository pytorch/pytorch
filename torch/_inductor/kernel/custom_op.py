# Owner(s): ["module: inductor"]

import functools
import logging
from collections.abc import Callable
from typing import Any, Optional, Union

import torch
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, ir_node_to_tensor, TensorBox
from torch._inductor.lowering import lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


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
]


def _extract_tensor_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from mixed args/kwargs.
    Separates tensors (for autotuning input_nodes) from non-tensor parameters.
    Non-tensor kwargs are later functools.partial'd into decomposition functions.

    Args:
        args: Positional arguments (mix of tensors and scalars)
        kwargs: Keyword arguments (mix of tensors and scalars)

    Returns:
        Tuple of (tensor_inputs_list, non_tensor_kwargs)
    """
    tensor_inputs = []
    non_tensor_kwargs = {}

    # Process args and kwargs: separate tensor inputs and non tensor args
    for i, arg in enumerate(args):
        if isinstance(arg, (TensorBox, Buffer)):
            tensor_inputs.append(arg)
        else:
            # Add non-tensor positional args to kwargs with generated names
            non_tensor_kwargs[f"arg_{i}"] = arg

    for key, value in kwargs.items():
        if isinstance(value, (TensorBox, Buffer)):
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
            fake_tensor = ir_node_to_tensor(ir_buffer)
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
    range_to_best_impl: dict[tuple[int, Union[int, float]], tuple[Callable, dict, str]],
) -> list[tuple[list[tuple[int, Union[int, float]]], Callable, dict, str]]:
    """Group all ranges by their implementation, even if not adjacent.

    This enables more aggressive optimization: ranges with the same impl are
    grouped together and can use OR predicates in torch.cond.

    Args:
        range_to_best_impl: Mapping from range to (impl, kwargs, name)

    Returns:
        List of (ranges_list, impl, kwargs, name) tuples, sorted by first range start

    Example:
        Input: {
            (1, 64): (impl_a, {}, "a"),
            (65, 128): (impl_b, {}, "b"),
            (129, 256): (impl_a, {}, "a"),  # Same as first!
        }
        Output: [
            ([(1, 64), (129, 256)], impl_a, {}, "a"),  # Grouped!
            ([(65, 128)], impl_b, {}, "b"),
        ]
    """
    from collections import defaultdict

    if not range_to_best_impl:
        return []

    impl_to_ranges = defaultdict(list)

    for range_key, (impl, kwargs, name) in range_to_best_impl.items():
        kwargs_key = frozenset(kwargs.items()) if kwargs else frozenset()
        impl_signature = (id(impl), kwargs_key, name)
        impl_to_ranges[impl_signature].append((range_key, impl, kwargs, name))

    result = []
    for impl_signature, group_items in impl_to_ranges.items():
        group_items.sort(key=lambda x: x[0][0])
        ranges_list = [item[0] for item in group_items]
        impl = group_items[0][1]
        kwargs = group_items[0][2]
        name = group_items[0][3]
        result.append((ranges_list, impl, kwargs, name))

    result.sort(key=lambda x: x[0][0][0])

    original_count = len(range_to_best_impl)
    grouped_count = len(result)

    if grouped_count < original_count:
        log.info(
            "Implementation grouping: reduced from %d ranges to %d impl groups",
            original_count,
            grouped_count,
        )
        for ranges_list, impl, kwargs, name in result:
            if len(ranges_list) > 1:
                ranges_str = ", ".join(
                    "[{}, {}]".format(s, e if e != float("inf") else "inf")
                    for s, e in ranges_list
                )
                log.info(
                    "   Grouped %d ranges for %s: %s",
                    len(ranges_list),
                    name,
                    ranges_str,
                )

    return result


def _split_points_to_ranges(
    split_points: list[int],
) -> list[tuple[int, Union[int, float]]]:
    """Convert split points to inclusive-inclusive ranges.

    Example: split_points=[512, 2048] ->
             [(1, 512), (513, 2048), (2049, float('inf'))]
    """
    ranges = []
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
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create input generator that produces tensor with dimension in range."""

    def constrained_gen_fn(fake_tensor: torch.Tensor) -> torch.Tensor:
        result = base_gen_fn(fake_tensor)
        shape = list(result.shape)

        # Pick middle of range
        if range_end == float("inf"):
            target_dim = int(range_start + 100)
        else:
            target_dim = (int(range_start) + int(range_end)) // 2

        target_dim = max(
            int(range_start),
            min(
                target_dim,
                int(range_end) - 1 if range_end != float("inf") else target_dim,
            ),
        )

        shape[dim_index] = target_dim
        return torch.randn(*shape, dtype=result.dtype, device=result.device)

    return constrained_gen_fn


def _extract_winning_decomposition_index(
    choice_name: str,
    decompositions: list[Callable],
) -> int:
    """Extract the decomposition index from winning SubgraphChoiceCaller's name.

    The choice name format is: "{op_name}_range_{start}_{end}_{decomp_name}_{counter}"
    We parse it to find which decomposition won by matching decomp_name.

    Args:
        choice_name: Name of the winning SubgraphChoiceCaller
        decompositions: List of decomposition functions

    Returns:
        Index into decompositions list (0-based)
    """
    if not choice_name:
        log.warning("Empty choice name, defaulting to first decomposition")
        return 0

    # Try to match decomposition by name
    for i, decomp in enumerate(decompositions):
        decomp_name = decomp.__name__
        if decomp_name in choice_name:
            log.debug(
                "Matched choice '%s' to decomposition[%d] '%s'",
                choice_name,
                i,
                decomp_name,
            )
            return i

    # Fallback: could not determine, use first
    log.warning(
        "Could not determine winning decomposition from choice name '%s', "
        "defaulting to first decomposition",
        choice_name,
    )
    return 0


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

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        # pyrefly: ignore [bad-argument-type]
        decompositions=decompositions,
        input_nodes=list(inputs),
        non_tensor_args=non_tensor_args,
    )

    # Add default implementation as fallback
    if op_overload and hasattr(op_overload, "_op"):
        fallback_name = f"{name}_fallback_default"
        from torch._inductor.select_algorithm import extern_kernels

        # Skip if extern_kernel already registered to avoid duplicate registration error
        if not hasattr(extern_kernels, fallback_name):
            with V.fake_mode:
                fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]
                fallback_kwargs = non_tensor_args[0] if non_tensor_args else {}
                fake_output = op_overload(*fake_inputs, **fallback_kwargs)

            # Convert SymInt to Expr for FixedLayout
            # FixedLayout expects Expr (sympy expressions) or int, not torch.SymInt
            # Extract .node.expr from SymInt to get the underlying sympy.Expr
            def convert_symint_to_expr(val):
                """Convert SymInt to Expr, leave int as is."""
                if isinstance(val, torch.SymInt):
                    return val.node.expr
                return val

            output_size = tuple(convert_symint_to_expr(s) for s in fake_output.shape)
            output_stride = tuple(
                convert_symint_to_expr(s) for s in fake_output.stride()
            )

            fallback_choice = _create_fallback_choice(
                name, op_overload, fake_output, fallback_kwargs
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
            )

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    # Convert user input generation functions to internal format
    input_gen_fns = {}
    if user_input_gen_fns:
        import inspect

        arg_names = (
            list(inspect.signature(decompositions[0]).parameters.keys())
            if decompositions
            else []
        )
        input_gen_fns = _adapt_user_input_gen_fns(inputs, arg_names, user_input_gen_fns)

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
    )

    # Apply inlining for fusion if winning_choice has graph; otherwise return result as-is(default fallback impl)
    if winning_choice.gm is not None:
        log.debug(
            "Inlining winning choice: %s (name=%s)",
            getattr(winning_choice, "name", type(winning_choice).__name__),
            name,
        )
        from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes

        result = inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)
        if return_choice:
            return result, winning_choice
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
) -> list[CustomOpConfig]:
    """Generate configs dynamically based on input tensors at lowering time."""
    import inspect

    sig = inspect.signature(default_impl)
    param_names = list(sig.parameters.keys())

    with V.fake_mode:
        fake_tensors = [ir_node_to_tensor(inp) for inp in tensor_inputs]

    # Only map tensor inputs to parameter names (skip non-tensor params at the end)
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
) -> Any:
    """Standard autotuning lowering function."""
    # Get configs: either generate dynamically or use static configs
    if config_generator is not None:
        configs_to_use = _generate_dynamic_configs(
            tensor_inputs, config_generator, default_impl, name
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

    result = autotune_custom_op(
        name=name,
        decompositions=decompositions,
        inputs=tensor_inputs,
        non_tensor_args=non_tensor_args,
        op_overload=op_overload,
        user_input_gen_fns=input_gen_fns,
    )

    validate_ir(result)
    return result


def _lower_single_impl(
    impl: Callable[..., Any],
    impl_kwargs: dict[str, Any],
    runtime_kwargs: dict[str, Any],
    tensor_inputs: list[Any],
    name: str,
) -> Any:
    """Lower a single implementation by tracing and inlining it."""
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
    result = inline_subgraph_to_ir_nodes(impl_gm, tensor_inputs, name)
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
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
) -> Any:
    """Range-based autotuning lowering function."""
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    from torch.fx.experimental.proxy_tensor import make_fx

    from ..decomposition import select_decomp_table

    log.info("=== Range-based Autotuning for %s ===", name)
    log.info("Dispatch on: %s[%d], Ranges: %s", tensor_name, dim_index, ranges)

    if config_generator is not None:
        configs_to_use = _generate_dynamic_configs(
            tensor_inputs, config_generator, default_impl, name
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

    range_to_best_impl_map = {}

    # Benchmark each range and collect winning implementations
    for range_start, range_end in ranges:
        range_input_gen_fns = None
        if input_gen_fns and tensor_name in input_gen_fns:
            base_gen_fn = input_gen_fns[tensor_name]
            range_gen_fn = _create_range_input_gen_fn(
                base_gen_fn, dim_index, range_start, range_end
            )
            range_input_gen_fns = {**input_gen_fns, tensor_name: range_gen_fn}

        range_name = f"{name}_range_{int(range_start)}_{int(range_end) if range_end != float('inf') else 'inf'}"

        autotuned_result, winning_choice = autotune_custom_op(
            name=range_name,
            decompositions=decompositions,
            inputs=tensor_inputs,
            non_tensor_args=non_tensor_args,
            op_overload=op_overload,
            user_input_gen_fns=range_input_gen_fns,
            return_choice=True,
        )

        winning_impl_idx = _extract_winning_decomposition_index(
            winning_choice.name, decompositions
        )
        winning_impl = decompositions[winning_impl_idx]
        winning_kwargs = non_tensor_args[winning_impl_idx]

        range_to_best_impl_map[(range_start, range_end)] = (
            winning_impl,
            winning_kwargs,
            winning_impl.__name__,
        )

        log.info(
            "   Range [%s, %s] -> %s",
            range_start,
            range_end if range_end != float("inf") else "inf",
            winning_impl.__name__,
        )

    # Group ranges by implementation (more aggressive than adjacent merging)
    impl_groups = _group_ranges_by_impl(range_to_best_impl_map)

    log.info("After grouping by implementation: %d impl groups", len(impl_groups))
    for ranges_list, impl, impl_kwargs, impl_name in impl_groups:
        ranges_str = ", ".join(
            f"[{s}, {e if e != float('inf') else 'inf'}]" for s, e in ranges_list
        )
        log.info("   %s: %s", impl_name, ranges_str)

    # If only one impl group remains, just inline that implementation
    if len(impl_groups) == 1:
        ranges_list, impl, impl_kwargs, impl_name = impl_groups[0]
        log.info("Only one implementation after grouping, directly inlining")
        return _lower_single_impl(
            impl, impl_kwargs, runtime_kwargs, tensor_inputs, name
        )

    # Build dispatch function for N impl groups using recursive nested torch.cond
    # Each impl group may contain multiple non-adjacent ranges (OR predicate)
    def dispatch_fn(*fake_tensors):
        """Main dispatch function that will be traced by make_fx.

        Supports arbitrary number of impl groups with OR predicates for non-adjacent ranges.
        Structure: cond(pred_group1, impl1, cond(pred_group2, impl2, ...))
        where pred_group can be: (dim in range1) OR (dim in range2) OR ...
        """
        num_impl_groups = len(impl_groups)

        if num_impl_groups < 2:
            raise RuntimeError(
                f"dispatch_fn requires at least 2 impl groups, got {num_impl_groups}"
            )

        dispatch_tensor = fake_tensors[0]
        dim_value = dispatch_tensor.size(dim_index)

        def build_range_predicate(
            ranges_list: list[tuple[int, Union[int, float]]],
        ) -> torch.Tensor:
            """Build OR predicate for multiple ranges.

            Args:
                ranges_list: List of (start, end) tuples

            Returns:
                SymBool: (dim in range1) OR (dim in range2) OR ...
            """
            if len(ranges_list) == 1:
                # Single range: start <= dim <= end
                range_start, range_end = ranges_list[0]
                start_int = int(range_start)
                end_int = int(range_end) if range_end != float("inf") else None

                if end_int is None:
                    # [start, inf): dim >= start
                    return dim_value >= start_int
                else:
                    # [start, end]: start <= dim <= end
                    return (dim_value >= start_int) & (dim_value <= end_int)

            # Multiple ranges: OR them together
            predicates = []
            for range_start, range_end in ranges_list:
                start_int = int(range_start)
                end_int = int(range_end) if range_end != float("inf") else None

                if end_int is None:
                    # [start, inf): dim >= start
                    pred = dim_value >= start_int
                else:
                    # [start, end]: start <= dim <= end
                    pred = (dim_value >= start_int) & (dim_value <= end_int)

                predicates.append(pred)

            # Combine with OR
            result = predicates[0]
            for pred in predicates[1:]:
                result = result | pred

            return result

        def build_nested_cond(current_impl_index: int):
            """Recursively build nested torch.cond for impl_groups[current_impl_index:]."""
            if current_impl_index >= num_impl_groups:
                raise RuntimeError(f"Invalid current_impl_index: {current_impl_index}")

            ranges_list, impl, impl_kwargs, impl_name = impl_groups[current_impl_index]
            merged_kwargs = _merge_config_and_runtime_kwargs(
                impl_kwargs, runtime_kwargs
            )

            @torch._dynamo.dont_skip_tracing
            def current_group_fn(*ops):
                return impl(*ops, **merged_kwargs)

            if current_impl_index == num_impl_groups - 1:
                return current_group_fn

            next_branch_fn = build_nested_cond(current_impl_index + 1)

            @torch._dynamo.dont_skip_tracing
            def cond_wrapper(*ops):
                # Compute predicate INSIDE cond_wrapper to avoid closure capture
                # This prevents SymBool from being added to operands
                group_pred = build_range_predicate(ranges_list)
                return torch.cond(
                    pred=group_pred,
                    true_fn=current_group_fn,
                    false_fn=next_branch_fn,
                    operands=list(ops),
                )

            return cond_wrapper

        # Build the nested cond structure starting from group 0
        nested_cond_fn = build_nested_cond(0)

        # Execute the nested cond
        return nested_cond_fn(*fake_tensors)

    # Trace with make_fx using fake mode
    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
        decomposition_table = select_decomp_table()

        log.info("Tracing torch.cond dispatch with symbolic shapes...")

        try:
            dispatch_gm = make_fx(
                dispatch_fn,
                decomposition_table=decomposition_table,
                tracing_mode="symbolic",
            )(*fake_inputs)

            log.info("Successfully traced torch.cond dispatch")

        except Exception:
            log.exception("make_fx tracing FAILED")
            raise

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
    is_range_based: bool = False,
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
    dispatch_on: Optional[tuple[str, int]] = None,
    split_points: Optional[list[int]] = None,
) -> Callable[..., Any]:
    """Create the lowering function for autotuning."""
    if not is_range_based:
        # Standard autotuning path
        @functools.wraps(op_overload)
        def standard_lowering_wrapper(*args: Any, **kwargs: Any) -> Any:
            tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)
            return _standard_lowering_fn(
                processed_configs=processed_configs,
                default_impl=default_impl,
                name=name,
                op_overload=op_overload,
                input_gen_fns=input_gen_fns,
                tensor_inputs=tensor_inputs,
                runtime_kwargs=runtime_kwargs,
                config_generator=config_generator,
            )

        return standard_lowering_wrapper

    # Range-based autotuning path
    tensor_name, dim_index = dispatch_on
    ranges = _split_points_to_ranges(split_points)

    @functools.wraps(op_overload)
    def range_based_lowering_wrapper(*args: Any, **kwargs: Any) -> Any:
        tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)
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
            config_generator=config_generator,
        )

    return range_based_lowering_wrapper


def register_custom_op_autotuning(
    custom_op: torch._library.custom_ops.CustomOpDef,
    configs: Optional[Union[list[CustomOpConfig], list[Callable[..., Any]]]] = None,
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dispatch_on: Optional[tuple[str, int]] = None,
    split_points: Optional[list[int]] = None,
) -> None:
    """Register custom op for autotuning with custom_op configs where each config
    specifies a decomposition implementation function with its parameter values.
    It also supports Range-based autotuning to benchmark per range and generate
    runtime dispatch.

    Args:
        custom_op: Custom operation (decorated function from @torch.library.custom_op)
        configs: List of CustomOpConfig objects for static inputs. Mutually exclusive with config_generator.
        config_generator: Dynamic config generator function that takes a dict mapping
                          parameter names to fake tensors, and returns list[CustomOpConfig]
                          based on input tensor properties. Mutually exclusive with configs.
        name: Operation name (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking

    Example:
        # Static configs
        @torch.library.custom_op("mylib::attention", mutates_args=())
        def my_attention(query, key, value, head_dim=32):
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
            dispatch_on=("x", 1),  # Dispatch on x[1]
            split_points=[512, 2048],  # Creates ranges: [1,512], [513,2048], [2049,inf]
        )
    """
    from torch._library.custom_ops import CustomOpDef

    if not isinstance(custom_op, CustomOpDef):
        raise TypeError(
            f"custom_op must be a CustomOpDef (decorated function from @torch.library.custom_op), "
            f"got {type(custom_op)}."
        )

    # Validate configs and config_generator are mutually exclusive
    if configs is not None and config_generator is not None:
        raise ValueError(
            "Cannot specify both 'configs' and 'config_generator'. "
            "Use 'config_generator' for shape-dependent configs."
        )

    if configs is None and config_generator is None:
        raise ValueError("Must specify either 'configs' or 'config_generator'")

    op_overload = custom_op._opoverload
    default_impl = custom_op._init_fn

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
    if is_range_based:
        if dispatch_on is None or split_points is None:
            raise ValueError(
                "Both dispatch_on and split_points must be specified for range-based autotuning"
            )
        if not isinstance(dispatch_on, tuple) or len(dispatch_on) != 2:
            raise ValueError("dispatch_on must be a tuple of (tensor_name, dim_index)")
        if not isinstance(split_points, list) or len(split_points) == 0:
            raise ValueError("split_points must be a non-empty list of integers")
        if sorted(split_points) != split_points:
            raise ValueError("split_points must be sorted in ascending order")

    # Create and register the lowering function
    lowering_fn = _create_autotuning_lowering(
        processed_configs=static_configs,
        default_impl=default_impl,
        name=name,
        op_overload=op_overload,
        input_gen_fns=input_gen_fns,
        is_range_based=is_range_based,
        config_generator=config_generator,
        dispatch_on=dispatch_on,
        split_points=split_points,
    )

    lowerings[op_overload] = lowering_fn
