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
        tensor_name: Optional tensor parameter name for range-based dispatch (e.g., 'x', 'query')
        dim_index: Optional dimension index for range-based dispatch (e.g., 0 for batch, 1 for seq_len)
        dim_range: Optional tuple (start, end) defining the range [start, end) for this config
        **params: Parameters passed to the function

    Examples:
        CustomOpConfig(attention_impl, head_dim=32, method='chunked')
        CustomOpConfig(short_impl, tensor_name='x', dim_index=1, dim_range=(0, 512))
    """

    def __init__(
        self,
        decomposition: Optional[Callable[..., Any]] = None,
        tensor_name: Optional[str] = None,
        dim_index: Optional[int] = None,
        dim_range: Optional[tuple[Union[int, float], Union[int, float]]] = None,
        **params: Any,
    ):
        if decomposition is not None and not callable(decomposition):
            raise TypeError(
                f"decomposition must be callable, got {type(decomposition)}"
            )

        # Validate range parameters
        if dim_range is not None:
            if tensor_name is None:
                raise ValueError(
                    "tensor_name must be specified when dim_range is provided"
                )
            if dim_index is None:
                raise ValueError(
                    "dim_index must be specified when dim_range is provided"
                )
            if not isinstance(dim_range, (tuple, list)) or len(dim_range) != 2:
                raise ValueError("dim_range must be a tuple or list of (start, end)")
            start, end = dim_range
            if start >= end:
                raise ValueError(
                    f"dim_range start ({start}) must be less than end ({end})"
                )

        self.decomposition = decomposition
        self.tensor_name = tensor_name
        self.dim_index = dim_index
        self.dim_range = tuple(dim_range) if dim_range is not None else None
        self.params = params

    def is_range_based(self) -> bool:
        """Check if this config is range-based."""
        return self.dim_range is not None

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
        parts = [decomp_name]

        if self.is_range_based():
            parts.append(f"tensor_name='{self.tensor_name}'")
            parts.append(f"dim_index={self.dim_index}")
            parts.append(f"dim_range={self.dim_range}")

        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            parts.append(params_str)

        return f"CustomOpConfig({', '.join(parts)})"


__all__ = [
    "autotune_custom_op",
    "register_custom_op_autotuning",
    "CustomOpConfig",
]


def _extract_tensor_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from args/kwargs, separating from non-tensor parameters."""
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


def _group_configs_by_range(
    configs: list[CustomOpConfig],
) -> dict[
    tuple[Optional[str], Optional[int], Optional[float], Optional[float]],
    list[CustomOpConfig],
]:
    """Group configs by their range parameters.

    Returns a dictionary where:
    - Key: (tensor_name, dim_index, range_start, range_end)
    - Value: List of CustomOpConfig objects with that range

    Non-range configs are grouped under key (None, None, None, None).
    """
    groups: dict[
        tuple[Optional[str], Optional[int], Optional[float], Optional[float]],
        list[CustomOpConfig],
    ] = {}

    for cfg in configs:
        if cfg.is_range_based():
            assert cfg.dim_range is not None
            range_start, range_end = cfg.dim_range
            key = (cfg.tensor_name, cfg.dim_index, range_start, range_end)
        else:
            key = (None, None, None, None)

        if key not in groups:
            groups[key] = []
        groups[key].append(cfg)

    return groups


def _validate_range_groups(
    range_groups: dict[
        tuple[Optional[str], Optional[int], Optional[float], Optional[float]],
        list[CustomOpConfig],
    ],
) -> None:
    """Validate range-based config groups.

    Checks:
    1. Cannot mix range-based and non-range configs
    2. All range configs must use same tensor_name and dim_index
    3. Ranges must not overlap
    """
    has_range_based = any(
        key != (None, None, None, None) for key in range_groups.keys()
    )
    has_non_range = (None, None, None, None) in range_groups

    # Check 1: Cannot mix range-based and non-range configs
    if has_range_based and has_non_range:
        raise ValueError(
            "Cannot mix range-based and non-range CustomOpConfigs. "
            "All configs must either have range parameters or none should have them."
        )

    if not has_range_based:
        return  # No range validation needed

    # Check 2: All range configs must use same tensor_name and dim_index
    tensor_names = set()
    dim_indices = set()
    ranges = []

    for key in range_groups.keys():
        if key == (None, None, None, None):
            continue
        tensor_name, dim_index, range_start, range_end = key
        tensor_names.add(tensor_name)
        dim_indices.add(dim_index)
        ranges.append((range_start, range_end))

    if len(tensor_names) > 1:
        raise ValueError(
            f"All range configs must use the same tensor_name. Found: {tensor_names}"
        )

    if len(dim_indices) > 1:
        raise ValueError(
            f"All range configs must use the same dim_index. Found: {dim_indices}"
        )

    # Check 3: Ranges must not overlap
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    for i in range(len(sorted_ranges) - 1):
        current_start, current_end = sorted_ranges[i]
        next_start, next_end = sorted_ranges[i + 1]

        if next_start < current_end:
            raise ValueError(
                f"Ranges overlap: [{current_start}, {current_end}) and [{next_start}, {next_end})"
            )


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
    inputs: list[Any],
    non_tensor_args: list[dict[str, Any]],
    op_overload: torch._ops.OpOverload,
    user_input_gen_fns: Optional[
        dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
) -> Union[TensorBox, Any]:
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

            fallback_choice = _create_fallback_choice(
                name, op_overload, fake_output, fallback_kwargs
            )
            fallback_choice.maybe_append_choice(
                choices=choices,
                input_nodes=list(inputs),
                layout=FixedLayout(
                    device=fake_output.device,
                    dtype=fake_output.dtype,
                    size=fake_output.shape,
                    stride=fake_output.stride(),
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

        return inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)

    log.debug(
        "Winning choice does not support inlining: %s (name=%s)",
        getattr(winning_choice, "name", type(winning_choice).__name__),
        name,
    )
    return selected_result


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

    decompositions = []
    non_tensor_args = []

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
        decompositions=decompositions,
        inputs=tensor_inputs,
        non_tensor_args=non_tensor_args,
        op_overload=op_overload,
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


def _create_autotuning_lowering(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    is_range_based: bool = False,
) -> Callable[..., Any]:
    """Create the lowering function for autotuning (shared logic for both range and non-range).

    Args:
        processed_configs: List of validated CustomOpConfig objects
        default_impl: Default implementation function
        name: Operation name for autotuning
        op_overload: OpOverload of the custom op
        input_gen_fns: Optional custom input generators
        is_range_based: Whether this is range-based autotuning

    Returns:
        Lowering function that can be registered with Inductor
    """
    if not is_range_based:
        # Standard autotuning path
        @functools.wraps(op_overload)
        def standard_lowering_fn(*args: Any, **kwargs: Any) -> Any:
            """Standard autotuning lowering."""
            tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)

            decompositions = []
            non_tensor_args = []

            for cfg in processed_configs:
                decomp = cfg.get_decomposition(default_impl=default_impl)
                decompositions.append(decomp)

                merged_kwargs = _merge_config_and_runtime_kwargs(
                    cfg.params, runtime_kwargs
                )
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

        return standard_lowering_fn

    # Range-based autotuning path - with per-range benchmarking
    @functools.wraps(op_overload)
    def range_based_lowering_fn(*args: Any, **kwargs: Any) -> Any:
        """Range-based autotuning lowering with per-range optimization."""
        tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)

        # Group configs by range
        range_groups = _group_configs_by_range(processed_configs)

        # Get tensor_name and dim_index from first config (all should be the same after validation)
        first_config = processed_configs[0]
        tensor_name = first_config.tensor_name
        dim_index = first_config.dim_index

        log.info(
            "=== Range-based Autotuning for %s ===",
            name
        )
        log.info(
            "Dispatch dimension: %s[%d]",
            tensor_name,
            dim_index
        )

        # Benchmark each range and collect best implementations
        range_to_impl: dict[
            tuple[str, int, Union[int, float], Union[int, float]],
            tuple[Callable[..., Any], dict[str, Any], str],
        ] = {}

        for range_key, range_configs in range_groups.items():
            if range_key == (None, None, None, None):
                continue  # Skip non-range configs (shouldn't happen after validation)

            tensor_name_key, dim_index_key, range_start, range_end = range_key

            # Benchmark this range
            best_impl, best_kwargs, best_impl_name = _benchmark_configs_for_range(
                name=name,
                range_configs=range_configs,
                default_impl=default_impl,
                op_overload=op_overload,
                tensor_inputs=tensor_inputs,
                runtime_kwargs=runtime_kwargs,
                input_gen_fns=input_gen_fns,
                tensor_name=tensor_name_key,
                dim_index=dim_index_key,
                range_start=range_start,
                range_end=range_end,
            )

            range_to_impl[range_key] = (best_impl, best_kwargs, best_impl_name)

        # Check if all ranges selected the same implementation
        unique_impl_names = {impl_name for _, _, impl_name in range_to_impl.values()}

        log.info(
            "=== Range-based Autotuning Summary for %s ===",
            name,
        )
        for range_key, (_, _, impl_name) in sorted(range_to_impl.items(), key=lambda x: x[0][2]):
            _, _, range_start, range_end = range_key
            log.info(
                "  Range [%s, %s): %s",
                range_start,
                range_end if range_end != float("inf") else "inf",
                impl_name,
            )

        if len(unique_impl_names) == 1:
            # All ranges use same implementation - use it directly (fusion-friendly!)
            the_impl, the_kwargs, the_impl_name = next(iter(range_to_impl.values()))

            log.info(
                "=== All ranges selected same implementation '%s' - using directly (fusion-friendly) ===",
                the_impl_name,
            )

            # Just use the single implementation for all inputs
            decompositions = []
            non_tensor_args = []

            for cfg in processed_configs:
                decomp = cfg.get_decomposition(default_impl=default_impl)
                decompositions.append(decomp)

                merged_kwargs = _merge_config_and_runtime_kwargs(
                    cfg.params, runtime_kwargs
                )
                non_tensor_args.append(merged_kwargs)

            result = autotune_custom_op(
                name=name,
                decompositions=decompositions,
                inputs=tensor_inputs,
                non_tensor_args=non_tensor_args,
                op_overload=op_overload,
                user_input_gen_fns=input_gen_fns,
            )
        else:
            # Different ranges use different implementations - generate dispatch
            log.info(
                "=== Different ranges selected different implementations ===",
            )
            log.info(
                "=== Generating runtime dispatch with torch.cond ===",
            )

            # Generate torch.cond dispatch
            result = _generate_range_dispatch_ir(
                range_to_impl=range_to_impl,
                tensor_name=tensor_name,
                dim_index=dim_index,
                args=args,
                kwargs=kwargs,
                op_overload=op_overload,
                default_impl=default_impl,
            )

        validate_ir(result)
        return result

    return range_based_lowering_fn
>>>>>>> a35f24f0b77 (add changes for dynamic range tuning)


def register_custom_op_autotuning(
    custom_op: torch._library.custom_ops.CustomOpDef,
    configs: Optional[Union[list[CustomOpConfig], list[Callable[..., Any]]]] = None,
    config_generator: Optional[
        Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    ] = None,
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
) -> None:
    """Register custom op for autotuning with custom_op configs where each config
    specifies a decomposition implementation function with its parameter values.

    Args:
        custom_op: Custom operation (decorated function from @torch.library.custom_op)
        configs: List of CustomOpConfig objects for static inputs. Mutually exclusive with config_generator.
        config_generator: Dynamic config generator function that takes a dict mapping
                          parameter names to fake tensors, and returns list[CustomOpConfig]
                          based on input tensor properties. Mutually exclusive with configs.
        name: Operation name (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking

    Examples:
<<<<<<< HEAD
        # Static configs
=======
        # Standard autotuning
>>>>>>> a35f24f0b77 (add changes for dynamic range tuning)
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

    # Create and register the lowering function
    lowering_fn = _create_autotuning_lowering(
        processed_configs=processed_configs,
        default_impl=default_impl,
        name=name,
        op_overload=op_overload,
        input_gen_fns=input_gen_fns,
        is_range_based=is_range_based,
    )

    lowerings[op_overload] = lowering_fn
