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

# ============================================================================
# 🧪 TEST MODE: Force different implementations for different ranges
# ============================================================================
# When enabled, bypasses autotuning and forces each range to use a different
# implementation (cycling through available impls). This is useful for testing
# the dispatch logic without needing to actually benchmark.
_FORCE_DIFFERENT_RANGE_IMPLS = False


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


def _merge_identical_implementations(
    range_to_best_impl: dict[tuple[int, Union[int, float]], tuple[Callable, dict, str]],
) -> dict[tuple[int, Union[int, float]], tuple[Callable, dict, str]]:
    """Merge consecutive ranges using the same implementation."""
    if not range_to_best_impl:
        return {}

    sorted_ranges = sorted(range_to_best_impl.items(), key=lambda x: x[0][0])
    merged = {}
    current_range_start, current_range_end = sorted_ranges[0][0]
    current_impl, current_kwargs, current_name = sorted_ranges[0][1]

    for i in range(1, len(sorted_ranges)):
        (next_start, next_end), (next_impl, next_kwargs, next_name) = sorted_ranges[i]

        if (
            current_impl == next_impl
            and current_kwargs == next_kwargs
            and current_name == next_name
            and next_start == current_range_end + 1
        ):
            current_range_end = next_end
        else:
            merged[(current_range_start, current_range_end)] = (
                current_impl,
                current_kwargs,
                current_name,
            )
            current_range_start, current_range_end = next_start, next_end
            current_impl, current_kwargs, current_name = (
                next_impl,
                next_kwargs,
                next_name,
            )

    merged[(current_range_start, current_range_end)] = (
        current_impl,
        current_kwargs,
        current_name,
    )

    if len(merged) < len(range_to_best_impl):
        log.info(
            "Range merging: reduced from %d to %d ranges",
            len(range_to_best_impl),
            len(merged),
        )

    return merged


@torch._dynamo.allow_in_graph
def _build_torch_cond_dispatch_recursive(
    range_idx: int,
    range_endpoints: list[int],
    impl_wrappers: list[Callable],
    tensors: tuple[torch.Tensor, ...],
    dispatch_dim: torch.SymInt,
) -> torch.Tensor:
    """Build nested torch.cond recursively without using functools.partial.

    This function is designed to work around Dynamo's limitations with functools.partial.
    Instead of using partial, we use simple recursion with index-based selection.

    """
    if range_idx >= len(impl_wrappers):
        raise IndexError(
            f"range_idx {range_idx} out of bounds (len={len(impl_wrappers)})"
        )

    if range_idx == len(impl_wrappers) - 1:
        # Last range: just call the wrapper directly
        return impl_wrappers[range_idx](*tensors)

    # Not last range: create conditional
    endpoint = range_endpoints[range_idx]

    # Define true/false branches as simple nested functions
    # These are NOT closures over tensors - they will receive tensors as args
    def true_fn():
        return impl_wrappers[range_idx](*tensors)

    def false_fn():
        return _build_torch_cond_dispatch_recursive(
            range_idx + 1,
            range_endpoints,
            impl_wrappers,
            tensors,
            dispatch_dim,
        )

    # 🔍 DEBUG: Print what torch.cond actually receives
    print("\n" + "=" * 80)
    print(f"🔍 DEBUG: torch.cond branches at range_idx={range_idx}")
    print("=" * 80)
    print(f"Predicate: dispatch_dim <= {endpoint}")
    print(f"dispatch_dim type: {type(dispatch_dim)}, value: {dispatch_dim}")
    print(f"\ntrue_fn:")
    print(f"  type: {type(true_fn)}")
    print(f"  __name__: {true_fn.__name__}")
    print(f"  __qualname__: {true_fn.__qualname__}")
    print(f"  __module__: {true_fn.__module__}")

    if true_fn.__closure__:
        print(f"  Closure ({len(true_fn.__closure__)} items):")
        for i, cell in enumerate(true_fn.__closure__):
            try:
                content = cell.cell_contents
                print(
                    f"    [{i}] {type(content).__name__}: {content if not callable(content) else f'<function {content.__name__}>'}"
                )
            except:
                print(f"    [{i}] <error accessing>")
    else:
        print(f"  Closure: None")

    print(f"\nfalse_fn:")
    print(f"  type: {type(false_fn)}")
    print(f"  __name__: {false_fn.__name__}")
    print(f"  __qualname__: {false_fn.__qualname__}")
    print(f"  __module__: {false_fn.__module__}")

    if false_fn.__closure__:
        print(f"  Closure ({len(false_fn.__closure__)} items):")
        for i, cell in enumerate(false_fn.__closure__):
            try:
                content = cell.cell_contents
                print(
                    f"    [{i}] {type(content).__name__}: {content if not callable(content) else f'<function {content.__name__}>'}"
                )
            except:
                print(f"    [{i}] <error accessing>")
    else:
        print(f"  Closure: None")

    # Print bytecode
    import dis

    print(f"\ntrue_fn bytecode:")
    dis.dis(true_fn)

    print(f"\nfalse_fn bytecode:")
    dis.dis(false_fn)

    print("=" * 80 + "\n")

    return torch.cond(
        dispatch_dim <= endpoint,
        true_fn,
        false_fn,
    )


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
        # Check if decomposition name appears in choice name
        if decomp_name in choice_name:
            log.debug(
                f"Matched choice '{choice_name}' to decomposition[{i}] '{decomp_name}'"
            )
            return i

    # Fallback: could not determine, use first
    log.warning(
        f"Could not determine winning decomposition from choice name '{choice_name}', "
        f"defaulting to first decomposition"
    )
    return 0


def _extract_tensor_by_name(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    tensor_name: str,
    op_overload: torch._ops.OpOverload,
) -> Optional[Any]:
    """Extract a tensor from args/kwargs by parameter name.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        tensor_name: Name of the parameter to extract
        op_overload: OpOverload to get parameter names

    Returns:
        The tensor (TensorBox/Buffer) if found, None otherwise
    """
    import inspect

    # Get parameter names from the op's signature
    try:
        sig = inspect.signature(op_overload)
        param_names = list(sig.parameters.keys())
    except Exception:
        log.warning("Could not get signature for %s, using fallback", op_overload)
        # Fallback: assume tensor_name matches position or kwargs
        if tensor_name in kwargs:
            return kwargs[tensor_name]
        return None

    # Check if tensor_name is in kwargs
    if tensor_name in kwargs:
        return kwargs[tensor_name]

    # Check if tensor_name is in positional args
    if tensor_name in param_names:
        param_index = param_names.index(tensor_name)
        if param_index < len(args):
            return args[param_index]

    return None


def _get_dimension_value(tensor: Any, dim_index: int) -> Any:
    """Get the dimension value from a tensor IR node.

    Args:
        tensor: TensorBox or Buffer IR node
        dim_index: Dimension index to extract

    Returns:
        Dimension value (may be symbolic or concrete)
    """
    if hasattr(tensor, "get_size"):
        # Buffer has get_size()
        shape = tensor.get_size()
    elif hasattr(tensor, "data") and hasattr(tensor.data, "get_size"):
        # TensorBox wraps data
        shape = tensor.data.get_size()
    else:
        raise RuntimeError(f"Cannot extract shape from {type(tensor)}")

    if dim_index >= len(shape):
        raise IndexError(
            f"dim_index {dim_index} out of range for tensor with {len(shape)} dimensions"
        )

    return shape[dim_index]


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


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
def _create_range_specific_input_gen_fns(
    user_input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    tensor_name: str,
    dim_index: int,
    range_start: Union[int, float],
    range_end: Union[int, float],
) -> Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]]:
    """Create input generators that produce tensors with dimension in specified range.

    Args:
        user_input_gen_fns: Original user-provided input generators
        tensor_name: Name of the tensor parameter to constrain
        dim_index: Dimension index to constrain
        range_start: Start of the range (inclusive)
        range_end: End of the range (exclusive)

    Returns:
        Modified input generators that ensure dimension is in range
    """
    if user_input_gen_fns is None:
        return None

    # Create a modified generator for the target tensor
    modified_gen_fns = user_input_gen_fns.copy()

    if tensor_name in user_input_gen_fns:
        original_gen_fn = user_input_gen_fns[tensor_name]

        def range_constrained_gen_fn(fake_tensor: torch.Tensor) -> torch.Tensor:
            """Generate input tensor with dimension in specified range."""
            # Generate tensor using original function
            result = original_gen_fn(fake_tensor)

            # Adjust the specified dimension to be in range
            current_shape = list(result.shape)

            # Pick a value in the middle of the range
            if range_end == float("inf"):
                # For unbounded range, use range_start + some reasonable offset
                target_dim = int(range_start + 100)
            else:
                # Use middle of the range
                target_dim = int((range_start + range_end) / 2)

            # Ensure it's actually in the range
            target_dim = max(int(range_start) + 1, target_dim)
            if range_end != float("inf"):
                target_dim = min(int(range_end) - 1, target_dim)

            # Recreate tensor with adjusted dimension
            current_shape[dim_index] = target_dim
            return torch.randn(*current_shape, dtype=result.dtype, device=result.device)

        modified_gen_fns[tensor_name] = range_constrained_gen_fn

    return modified_gen_fns


def _benchmark_configs_for_range(
    name: str,
    range_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    op_overload: torch._ops.OpOverload,
    tensor_inputs: list[Any],
    runtime_kwargs: dict[str, Any],
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    tensor_name: str,
    dim_index: int,
    range_start: Union[int, float],
    range_end: Union[int, float],
) -> tuple[Callable[..., Any], dict[str, Any], str]:
    """Benchmark all configs for a specific range and return the best implementation.

    Args:
        name: Base name for the operation
        range_configs: List of configs to benchmark for this range
        default_impl: Default implementation
        op_overload: OpOverload of the custom op
        tensor_inputs: Tensor inputs
        runtime_kwargs: Runtime keyword arguments
        input_gen_fns: Input generators
        tensor_name: Name of the tensor being dispatched on
        dim_index: Dimension index being dispatched on
        range_start: Start of range
        range_end: End of range

    Returns:
        Tuple of (best_decomposition_function, best_kwargs, best_impl_name)
    """
    # Create range-specific input generators for this range
    range_input_gen_fns = _create_range_specific_input_gen_fns(
        input_gen_fns, tensor_name, dim_index, range_start, range_end
    )
=======
def _standard_lowering_fn(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    args: Any,
    kwargs: Any,
) -> Any:
    """Standard autotuning lowering function."""
    tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)
>>>>>>> 68143ba7956 (clean up a bit)

    decompositions = []
    non_tensor_args = []

<<<<<<< HEAD
    for cfg in range_configs:
        decomp = cfg.get_decomposition(default_impl=default_impl)
        decompositions.append(decomp)

        merged_kwargs = _merge_config_and_runtime_kwargs(cfg.params, runtime_kwargs)
        non_tensor_args.append(merged_kwargs)

    # Use autotune_custom_op to benchmark and select the best
    range_name = f"{name}_range_{int(range_start)}_{int(range_end) if range_end != float('inf') else 'inf'}"

    # Run autotuning for this specific range
    autotune_custom_op(
        name=range_name,
=======
    for cfg in processed_configs:
        decomp = cfg.get_decomposition(default_impl=default_impl)
        decompositions.append(decomp)
        merged_kwargs = _merge_config_and_runtime_kwargs(cfg.params, runtime_kwargs)
        non_tensor_args.append(merged_kwargs)

    result = autotune_custom_op(
        name=name,
>>>>>>> 68143ba7956 (clean up a bit)
        decompositions=decompositions,
        inputs=tensor_inputs,
        non_tensor_args=non_tensor_args,
        op_overload=op_overload,
<<<<<<< HEAD
        user_input_gen_fns=range_input_gen_fns,
    )

    # Extract the winning choice from the result
    # The autotune_custom_op inlines the winning choice, so we need to determine
    # which implementation was selected based on the benchmarking results

    # For now, we'll use a heuristic: return the first implementation
    # In a complete implementation, we would extract this from the autotuning cache
    best_impl = decompositions[0]
    best_kwargs = non_tensor_args[0]
    best_impl_name = best_impl.__name__ if hasattr(best_impl, '__name__') else str(best_impl)

    log.info(
        "Range [%s, %s): Selected implementation '%s' after benchmarking %d candidates",
        range_start,
        range_end if range_end != float('inf') else 'inf',
        best_impl_name,
        len(decompositions),
    )

    return best_impl, best_kwargs, best_impl_name


def _generate_range_dispatch_ir(
    range_to_impl: dict[
        tuple[str, int, Union[int, float], Union[int, float]],
        tuple[Callable[..., Any], dict[str, Any], str],
    ],
    tensor_name: str,
    dim_index: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    op_overload: torch._ops.OpOverload,
    default_impl: Callable[..., Any],
) -> Any:
    """Generate torch.cond based dispatch for different ranges.

    Args:
        range_to_impl: Mapping from range to (implementation, kwargs, impl_name)
        tensor_name: Name of tensor to dispatch on
        dim_index: Dimension index to dispatch on
        args: Input arguments
        kwargs: Keyword arguments
        op_overload: OpOverload of the custom op
        default_impl: Default implementation

    Returns:
        Result from the selected implementation
    """
    # Extract tensor inputs
    tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)

    # Get the target tensor
    target_tensor_ir = _extract_tensor_by_name(args, kwargs, tensor_name, op_overload)
    if target_tensor_ir is None:
        raise RuntimeError(f"Could not find tensor '{tensor_name}' in arguments")

    # Get dimension value (may be symbolic or concrete)
    dim_value = _get_dimension_value(target_tensor_ir, dim_index)

    # Sort ranges by start value
    sorted_ranges = sorted(range_to_impl.items(), key=lambda x: x[0][2])

    log.info(
        "Generating torch.cond dispatch for %s[%d] with %d ranges",
        tensor_name,
        dim_index,
        len(sorted_ranges),
    )

    # Convert IR nodes to tensors for the implementations
    tensor_args = [ir_node_to_tensor(inp) for inp in tensor_inputs]

    # Build nested torch.cond dispatch recursively
    def build_cond_tree(range_idx: int) -> torch.Tensor:
        """Recursively build nested torch.cond calls for range dispatch."""
        if range_idx >= len(sorted_ranges):
            # Shouldn't reach here - use last range's impl
            _, (impl, impl_kwargs, _) = sorted_ranges[-1]
            merged_kwargs = {**impl_kwargs, **runtime_kwargs}
            return impl(*tensor_args, **merged_kwargs)

        range_key, (impl, impl_kwargs, impl_name) = sorted_ranges[range_idx]
        _, _, range_start, range_end = range_key
        merged_kwargs = {**impl_kwargs, **runtime_kwargs}

        # Last range - just call the implementation
        if range_idx == len(sorted_ranges) - 1:
            log.debug(
                "  Range [%s, %s): Using %s (final range)",
                range_start,
                "inf" if range_end == float("inf") else range_end,
                impl_name,
            )
            return impl(*tensor_args, **merged_kwargs)

        # Create predicate: dim_value < range_end
        # Handle both concrete and symbolic dimensions
        if isinstance(dim_value, int):
            # Concrete dimension - convert to tensor for torch.cond
            pred = torch.tensor(dim_value < range_end)
        else:
            # Symbolic dimension - create comparison
            # dim_value is a sympy expression or SymInt
            pred = dim_value < range_end

        log.debug(
            "  Range [%s, %s): Checking dim < %s for %s",
            range_start,
            "inf" if range_end == float("inf") else range_end,
            range_end,
            impl_name,
        )

        # Define branches for torch.cond
        def true_fn() -> torch.Tensor:
            """Use this range's implementation."""
            return impl(*tensor_args, **merged_kwargs)

        def false_fn() -> torch.Tensor:
            """Check next range."""
            return build_cond_tree(range_idx + 1)

        # Use torch.cond to create runtime dispatch
        # This will be captured and lowered by Inductor
        result = torch.cond(pred, true_fn, false_fn)

        return result

    # Build the dispatch tree starting from first range
    try:
        result = build_cond_tree(0)
        log.info(
            "Successfully generated torch.cond dispatch tree with %d conditional branches",
            len(sorted_ranges) - 1,
        )
        return result
    except Exception as e:
        # If torch.cond generation fails, fall back to global autotuning
        log.warning(
            "Failed to generate torch.cond dispatch: %s. Falling back to global autotuning.",
            str(e),
        )

        # Fallback: use global autotuning
        all_decompositions = []
        all_non_tensor_args = []

        for range_key, (impl, impl_kwargs, _) in sorted_ranges:
            all_decompositions.append(impl)
            merged_kwargs = {**impl_kwargs, **runtime_kwargs}
            all_non_tensor_args.append(merged_kwargs)

        result = autotune_custom_op(
            name=f"{op_overload._name}_range_dispatch_fallback",
            decompositions=all_decompositions,
            inputs=tensor_inputs,
            non_tensor_args=all_non_tensor_args,
            op_overload=op_overload,
            user_input_gen_fns=None,
        )

        return result


=======
>>>>>>> d4349888545 (update code)
=======
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
    from torch.fx.experimental.proxy_tensor import make_fx
    from ..decomposition import select_decomp_table
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes

    def impl_wrapper(*tensors):
        return impl(*tensors, **{**runtime_kwargs, **impl_kwargs})

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
    args: Any,
    kwargs: Any,
) -> Any:
    """Range-based autotuning lowering function."""
    log.info("=== Range-based Autotuning for %s ===", name)
    log.info("Dispatch on: %s[%d], Ranges: %s", tensor_name, dim_index, ranges)

    tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)

    # Benchmark each range and collect winning implementations
    range_to_best_impl = {}
    decompositions = []
    non_tensor_args = []

    for cfg in processed_configs:
        decomp = cfg.get_decomposition(default_impl=default_impl)
        decompositions.append(decomp)
        merged_kwargs = _merge_config_and_runtime_kwargs(cfg.params, runtime_kwargs)
        non_tensor_args.append(merged_kwargs)

    # TEST MODE: Force different implementations for different ranges
    if _FORCE_DIFFERENT_RANGE_IMPLS:
        for range_idx, (range_start, range_end) in enumerate(ranges):
            # Force each range to use a different implementation
            forced_impl_idx = range_idx % len(decompositions)
            impl = decompositions[forced_impl_idx]
            impl_kwargs = non_tensor_args[forced_impl_idx]

            range_to_best_impl[(range_start, range_end)] = (
                impl,
                impl_kwargs,
                impl.__name__,
            )

            log.warning(
                "   Range [%s, %s] -> FORCED to use impl[%d]: %s",
                range_start,
                range_end if range_end != float("inf") else "inf",
                forced_impl_idx,
                impl.__name__,
            )

        log.warning("=" * 80)
    else:
        # Normal autotuning path
        for range_start, range_end in ranges:
            # Create range-specific input generator
            range_input_gen_fns = None
            if input_gen_fns and tensor_name in input_gen_fns:
                base_gen_fn = input_gen_fns[tensor_name]
                range_gen_fn = _create_range_input_gen_fn(
                    base_gen_fn, dim_index, range_start, range_end
                )
                range_input_gen_fns = {**input_gen_fns, tensor_name: range_gen_fn}

            range_name = f"{name}_range_{int(range_start)}_{int(range_end) if range_end != float('inf') else 'inf'}"

            # Run autotuning for this range
            autotuned_result, winning_choice = autotune_custom_op(
                name=range_name,
                decompositions=decompositions,
                inputs=tensor_inputs,
                non_tensor_args=non_tensor_args,
                op_overload=op_overload,
                user_input_gen_fns=range_input_gen_fns,
                return_choice=True,
            )

            # Extract winning implementation
            choice_name = getattr(winning_choice, "name", "")
            winning_idx = _extract_winning_decomposition_index(
                choice_name, decompositions
            )
            impl = decompositions[winning_idx]
            impl_kwargs = non_tensor_args[winning_idx]

            range_to_best_impl[(range_start, range_end)] = (
                impl,
                impl_kwargs,
                impl.__name__,
            )

            log.info(
                "Range [%s, %s]: Selected %s",
                range_start,
                range_end if range_end != float("inf") else "inf",
                impl.__name__,
            )

    log.info("Completed autotuning for %d ranges", len(range_to_best_impl))

    # Step 2: Merge consecutive ranges with identical implementations
    merged_range_to_best_impl = _merge_identical_implementations(range_to_best_impl)

    log.info(
        "After merging: %d unique implementations across %d ranges",
        len({impl_name for _, _, impl_name in merged_range_to_best_impl.values()}),
        len(merged_range_to_best_impl),
    )

    # Step 3: Check if all ranges merged into one
    if len(merged_range_to_best_impl) == 1:
        log.info(
            "All ranges selected the same implementation - skipping dispatch, using direct inline"
        )
        single_impl, single_kwargs, _ = next(iter(merged_range_to_best_impl.values()))
        return _lower_single_impl(
            single_impl, single_kwargs, runtime_kwargs, tensor_inputs, name
        )

    # Step 4: Create dispatch using torch.cond traced with make_fx
    log.info(
        "Creating torch.cond dispatch for %d ranges", len(merged_range_to_best_impl)
    )

    from torch.fx.experimental.proxy_tensor import make_fx
    from ..decomposition import select_decomp_table
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes

    sorted_ranges = sorted(merged_range_to_best_impl.items())

    # Build dispatch function for 2 ranges
    def dispatch_fn(*fake_tensors):
        """Main dispatch function that will be traced by make_fx.

        Currently supports only 2 ranges.
        For 2 ranges: torch.cond(pred1, impl1, impl2)
        """
        if len(sorted_ranges) != 2:
            raise NotImplementedError(
                f"Range-based dispatch currently supports only 2 ranges, got {len(sorted_ranges)}. "
                f"Support for 3+ ranges requires a different approach to avoid data-dependent branching issues."
            )

        dispatch_tensor = fake_tensors[0]
        dim_value = dispatch_tensor.size(dim_index)

        # Simple case: just one torch.cond for 2 ranges
        (r1_start, r1_end), (impl1, impl1_kwargs, _) = sorted_ranges[0]
        (r2_start, r2_end), (impl2, impl2_kwargs, _) = sorted_ranges[1]

        merged_kwargs1 = {**runtime_kwargs, **impl1_kwargs}
        merged_kwargs2 = {**runtime_kwargs, **impl2_kwargs}

        @torch._dynamo.dont_skip_tracing
        def true_fn(*ops):
            return impl1(*ops, **merged_kwargs1)

        @torch._dynamo.dont_skip_tracing
        def false_fn(*ops):
            return impl2(*ops, **merged_kwargs2)

        return torch.cond(
            pred=dim_value <= r1_end,
            true_fn=true_fn,
            false_fn=false_fn,
            operands=list(fake_tensors),
        )

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
            log.debug(
                f"GraphModule created with {len(list(dispatch_gm.graph.nodes))} nodes"
            )

        except Exception as e:
            log.error("make_fx tracing FAILED: %s", e)
            raise

    # Inline the dispatch graph
    log.info("Inlining torch.cond dispatch graph...")
    result = inline_subgraph_to_ir_nodes(dispatch_gm, tensor_inputs, f"{name}_dispatch")

    log.info(
        "Successfully created torch.cond dispatch for %d ranges", len(sorted_ranges)
    )

    validate_ir(result)
    return result


>>>>>>> 68143ba7956 (clean up a bit)
def _create_autotuning_lowering(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    is_range_based: bool = False,
    dispatch_on: Optional[tuple[str, int]] = None,
    split_points: Optional[list[int]] = None,
) -> Callable[..., Any]:
    """Create the lowering function for autotuning."""
    if not is_range_based:
        # Standard autotuning path
        @functools.wraps(op_overload)
        def standard_lowering_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _standard_lowering_fn(
                processed_configs=processed_configs,
                default_impl=default_impl,
                name=name,
                op_overload=op_overload,
                input_gen_fns=input_gen_fns,
                args=args,
                kwargs=kwargs,
            )

        return standard_lowering_wrapper

    # Range-based autotuning path
    tensor_name, dim_index = dispatch_on
    ranges = _split_points_to_ranges(split_points)

    @functools.wraps(op_overload)
    def range_based_lowering_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _range_based_lowering_fn(
            processed_configs=processed_configs,
            default_impl=default_impl,
            name=name,
            op_overload=op_overload,
            input_gen_fns=input_gen_fns,
            tensor_name=tensor_name,
            dim_index=dim_index,
            ranges=ranges,
            args=args,
            kwargs=kwargs,
        )

<<<<<<< HEAD
        # Save to file for debugging/review
        import os

        output_dir = "/tmp/torch_inductor_range_dispatch"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{name}_dispatch.py")

        with open(output_file, "w") as f:
            f.write(dispatch_func_code)

        log.info("Generated dispatch function saved to: %s", output_file)

        # ========================================
        # Option B: Use make_fx + inline_subgraph_to_ir_nodes
        # ========================================
        log.info("Creating runtime dispatch using make_fx tracing")

        # Use merged ranges for compilation
        sorted_ranges = sorted(merged_range_to_best_impl.items())

        # Build dispatch function for tracing
        def build_dispatch_fn_for_tracing():
            def dispatch_fn(*fake_tensors):
                dispatch_tensor = fake_tensors[0]
                dim_value = dispatch_tensor.size(dim_index)

                # Build nested torch.cond
                def build_cond_recursive(ranges_list, idx=0):
                    if idx >= len(ranges_list):
                        raise RuntimeError("No ranges available")

                    (r_start, r_end), (impl_fn, impl_kwargs, impl_name) = ranges_list[
                        idx
                    ]

                    # Last range - no condition
                    if idx == len(ranges_list) - 1:
                        return impl_fn(
                            *fake_tensors, **{**runtime_kwargs, **impl_kwargs}
                        )

                    # Recursive case with torch.cond
                    return torch.cond(
                        pred=dim_value <= r_end,
                        true_fn=lambda: impl_fn(
                            *fake_tensors, **{**runtime_kwargs, **impl_kwargs}
                        ),
                        false_fn=lambda: build_cond_recursive(ranges_list, idx + 1),
                        operands=[],
                    )

                return build_cond_recursive(sorted_ranges, 0)

            return dispatch_fn

        dispatch_fn = build_dispatch_fn_for_tracing()

        # Trace with make_fx to create GraphModule
        from torch.fx.experimental.proxy_tensor import make_fx
        from ..decomposition import select_decomp_table

        log.debug("Tracing dispatch function with make_fx...")

        with V.fake_mode:
            fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)

            decomposition_table = select_decomp_table()
            dispatch_gm = make_fx(
                dispatch_fn,
                decomposition_table=decomposition_table,
                tracing_mode="symbolic",
            )(*fake_inputs)

        log.debug(
            f"GraphModule created with {len(list(dispatch_gm.graph.nodes))} nodes"
        )

        log.info("Creating SubgraphBuffer with multi-range dispatch capability...")

        from ..ir import FixedLayout, SubgraphBuffer, TensorBox

        range_gms = []

        for (range_start, range_end), (
            impl_fn,
            impl_kwargs,
            perf_time,
        ) in sorted_ranges:
            log.debug(
                f"  Compiling range [{range_start}, {range_end}]: {impl_fn.__name__}"
            )

            # Create wrapper for this specific implementation
            def create_impl_wrapper(fn, kwargs):
                def wrapper(*tensors):
                    return fn(*tensors, **{**runtime_kwargs, **kwargs})

                return wrapper

            impl_wrapper = create_impl_wrapper(impl_fn, impl_kwargs)

            # Trace this implementation independently (no torch.cond!)
            with V.fake_mode:
                impl_gm = make_fx(
                    impl_wrapper,
                    decomposition_table=decomposition_table,
                    tracing_mode="symbolic",
                )(*fake_inputs)

                log.debug(
                    f"    → Generated GraphModule with {len(list(impl_gm.graph.nodes))} nodes"
                )

                # Store (range, GraphModule) tuple
                range_gms.append(((range_start, range_end), impl_gm))

        log.info(f"Compiled {len(range_gms)} range implementations")

        # Step 2: Create unified SubgraphBuffer with multi-range dispatch
        # Passing a list of (range, gm) tuples triggers multi-range mode
        with V.fake_mode:
            fake_output = dispatch_gm(*fake_inputs)
            output_layout = FixedLayout(
                device=fake_output.device,
                dtype=fake_output.dtype,
                size=fake_output.shape,
                stride=fake_output.stride(),
            )

        result = TensorBox.create(
            SubgraphBuffer(
                layout=output_layout,
                input_nodes=tensor_inputs,
                gm=range_gms,  # List of (range, gm) tuples triggers multi-range mode
                example_inputs=list(fake_inputs),
                subgraph_name=f"{name}_autotuned",
                dispatch_dim_index=dim_index,
            )
        )

        log.info(
            f"Created SubgraphBuffer with multi-range dispatch ({len(range_gms)} ranges)"
        )

        validate_ir(result)
        return result

    return range_based_lowering_fn
>>>>>>> a35f24f0b77 (add changes for dynamic range tuning)
=======
    return range_based_lowering_wrapper
>>>>>>> 68143ba7956 (clean up a bit)


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

<<<<<<< HEAD
<<<<<<< HEAD
    Args:
        custom_op: Custom operation (decorated function from @torch.library.custom_op)
        configs: List of CustomOpConfig objects for static inputs. Mutually exclusive with config_generator.
        config_generator: Dynamic config generator function that takes a dict mapping
                          parameter names to fake tensors, and returns list[CustomOpConfig]
                          based on input tensor properties. Mutually exclusive with configs.
=======
    Args:
        custom_op: Custom operation (decorated function from @torch.library.custom_op)
        configs: List of CustomOpConfig objects
>>>>>>> 86ea916f1b0 (cleanup)
        name: Operation name (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking

    Examples:
<<<<<<< HEAD
<<<<<<< HEAD
        # Static configs
=======
        # Standard autotuning
>>>>>>> a35f24f0b77 (add changes for dynamic range tuning)
        @torch.library.custom_op("mylib::attention", mutates_args=())
        def my_attention(query, key, value, head_dim=32):
            ...
=======
    Two modes:
    1. Standard autotuning: Benchmark all configs and select the best globally
    2. Range-based autotuning: Benchmark per range and generate runtime dispatch
>>>>>>> d4349888545 (update code)
=======
        @torch.library.custom_op("mylib::attention", mutates_args=())
        def my_attention(query, key, value, head_dim=32):
            ...
>>>>>>> 86ea916f1b0 (cleanup)

    Standard Example:
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

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
        # Range-based autotuning
        register_custom_op_autotuning(
            my_op,
            configs=[
                # Range [0, 512): test 3 implementations
                CustomOpConfig(impl1, tensor_name='x', dim_index=1, dim_range=(0, 512)),
                CustomOpConfig(impl2, tensor_name='x', dim_index=1, dim_range=(0, 512)),
                CustomOpConfig(impl3, tensor_name='x', dim_index=1, dim_range=(0, 512)),
                # Range [512, inf): test 3 implementations
                CustomOpConfig(impl1, tensor_name='x', dim_index=1, dim_range=(512, float('inf'))),
                CustomOpConfig(impl2, tensor_name='x', dim_index=1, dim_range=(512, float('inf'))),
                CustomOpConfig(impl3, tensor_name='x', dim_index=1, dim_range=(512, float('inf'))),
            ],
>>>>>>> a35f24f0b77 (add changes for dynamic range tuning)
=======
    Range-based Example:
        register_custom_op_autotuning(
            my_op,
            configs=[CustomOpConfig(impl1), CustomOpConfig(impl2), CustomOpConfig(impl3)],
            dispatch_on=("x", 1),  # Dispatch on x[1]
            split_points=[512, 2048],  # Creates ranges: [1,512], [513,2048], [2049,inf]
>>>>>>> d4349888545 (update code)
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

<<<<<<< HEAD
    # Group configs by range and validate
    range_groups = _group_configs_by_range(processed_configs)
    _validate_range_groups(range_groups)

<<<<<<< HEAD
        # Get configs: either generate dynamically or use static configs
        if config_generator is not None:
            configs_to_use = _generate_dynamic_configs(
                tensor_inputs, config_generator, default_impl, name
            )
        else:
            assert static_configs is not None
            configs_to_use = static_configs

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
=======
    # Detect if this is range-based autotuning
    is_range_based = (None, None, None, None) not in range_groups

    if is_range_based:
        log.debug(
            "Detected range-based configs for %s. Using simplified autotuning for all configs.",
            name,
>>>>>>> a35f24f0b77 (add changes for dynamic range tuning)
        )
=======
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
>>>>>>> d4349888545 (update code)

    # Create and register the lowering function
    lowering_fn = _create_autotuning_lowering(
        processed_configs=processed_configs,
        default_impl=default_impl,
        name=name,
        op_overload=op_overload,
        input_gen_fns=input_gen_fns,
        is_range_based=is_range_based,
        dispatch_on=dispatch_on,
        split_points=split_points,
    )

    lowerings[op_overload] = lowering_fn
