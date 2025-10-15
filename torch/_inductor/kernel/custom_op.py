# Owner(s): ["module: inductor"]

import functools
import uuid
from typing import Any, Callable, Optional, Union

import torch
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, ir_node_to_tensor, TensorBox
from torch._inductor.lowering import lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)
from torch._inductor.virtualized import V


__all__ = [
    "autotune_custom_op",
    "register_custom_op_autotuning",
]


def _extract_tensor_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from mixed args/kwargs.

    Args:
        args: Positional arguments (mix of tensors and scalars)
        kwargs: Keyword arguments (mix of tensors and scalars)

    Returns:
        Tuple of (tensor_inputs_list, non_tensor_kwargs)
    """
    tensor_inputs = []
    non_tensor_kwargs = {}

    for arg in args:
        if isinstance(arg, (TensorBox, Buffer)) or (
            hasattr(arg, "dtype") and hasattr(arg, "shape")
        ):
            tensor_inputs.append(arg)

    for key, value in kwargs.items():
        if isinstance(value, (TensorBox, Buffer)) or (
            hasattr(value, "dtype") and hasattr(value, "shape")
        ):
            tensor_inputs.append(value)
        else:
            non_tensor_kwargs[key] = value

    return tensor_inputs, non_tensor_kwargs


def _create_user_input_gen_fns(
    inputs: list[Any],
    user_input_gen_fns: dict[int, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[int, Callable[[Any], torch.Tensor]]:
    """Convert user input generators to internal format.

    Args:
        inputs: List of input IR nodes from compilation
        user_input_gen_fns: User-provided input generation functions

    Returns:
        Dict mapping indices to internal input generation functions
    """
    internal_input_gen_fns = {}

    with V.fake_mode:
        fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]

    def create_internal_input_gen_fn(
        user_function: Callable[[torch.Tensor], torch.Tensor],
        template: torch.Tensor,
    ) -> Callable[[Any], torch.Tensor]:
        def internal_input_gen_fn(ir_buffer: Any) -> torch.Tensor:
            fake_tensor_for_user = torch.empty(
                template.shape,
                dtype=template.dtype,
                device="meta",
            )
            return user_function(fake_tensor_for_user)

        return internal_input_gen_fn

    for i, user_gen_fn in user_input_gen_fns.items():
        if i >= len(fake_inputs):
            continue

        fake_template = fake_inputs[i]
        internal_input_gen_fns[i] = create_internal_input_gen_fn(
            user_gen_fn, fake_template
        )

    return internal_input_gen_fns


# Global cache for fallback choices to avoid duplicate creation
_fallback_choice_cache = {}


def _get_or_create_fallback_choice(
    name: str,
    default_impl: Callable[..., Any],
    fake_output: torch.Tensor,
    kwargs: dict[str, Any],
) -> ExternKernelChoice:
    """Get or create fallback choice for default implementation."""
    cache_key = (id(default_impl), name, tuple(sorted(kwargs.items())))

    if cache_key not in _fallback_choice_cache:

        def fallback_wrapper(*args: Any) -> Any:
            return default_impl(*args, **kwargs)

        fallback_name = f"{name}_fallback_{default_impl._name}"
        _fallback_choice_cache[cache_key] = ExternKernelChoice(
            kernel=fallback_wrapper,
            name=fallback_name,
            has_out_variant=False,
            op_overload=default_impl,
            use_fallback_kernel=True,
        )

    return _fallback_choice_cache[cache_key]


def _create_parameter_variants(
    decompositions: list[Callable[..., Any]],
    tuning_knob: dict[str, list[Any]],
) -> list[Any]:  # Returns partial objects which are callable
    """Create parameter variants for decompositions using tuning knob.

    Args:
        decompositions: Base implementation functions
        tuning_knob: Parameter tuning dict with parameter names and value lists

    Returns:
        List of variant functions with all parameter combinations
    """
    # Validate parameter values
    for param_name, param_values in tuning_knob.items():
        if not isinstance(param_values, (list, tuple)):
            raise TypeError(
                f"Parameter values for '{param_name}' must be a list or tuple, got {type(param_values)}"
            )
        if not param_values:
            raise ValueError(
                f"At least one parameter value must be provided for '{param_name}'"
            )

    # Generate all combinations of parameter values using Cartesian product
    import itertools

    param_names = list(tuning_knob.keys())
    param_values_lists = list(tuning_knob.values())
    param_combinations = list(itertools.product(*param_values_lists))

    # Create variants for each decomposition with each parameter combination
    variants = []
    for decomp_fn in decompositions:
        for param_combo in param_combinations:
            # Create kwargs dict for this combination
            param_kwargs = dict(zip(param_names, param_combo))

            # Create partial function with all parameters
            variant = functools.partial(decomp_fn, **param_kwargs)

            # Generate descriptive name
            param_suffix = "_".join(
                f"{name}_{value}" for name, value in param_kwargs.items()
            )
            variant.__name__ = f"{decomp_fn.__name__}_{param_suffix}"  # type: ignore[attr-defined]
            variants.append(variant)

    return variants


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    kwargs: Optional[dict[str, Any]] = None,
    default_impl: Optional[Callable[..., Any]] = None,
    user_input_gen_fns: Optional[
        dict[int, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
) -> Union[TensorBox, Any]:
    """Autotune custom operations by comparing multiple decomposition implementations.

    This function generates multiple implementation choices for a custom operation and
    uses Inductor's autotuning system to select the best performing variant at runtime.

    Args:
        name: Unique identifier for the autotuning operation
        decompositions: List of alternative implementation functions to benchmark
        inputs: Input tensor IR nodes from compilation (TensorBox/Buffer objects)
        kwargs: Non-tensor parameters to pass to decomposition functions
        default_impl: Original custom op implementation used as fallback
        user_input_gen_fns: Optional custom input generators for benchmarking.
                           Maps input indices to functions that take fake tensors
                           and return real tensors for performance measurement.

    Returns:
        IR node representing the optimized operation result

    Raises:
        TypeError: If decompositions is not a list/tuple
        RuntimeError: If no inputs or no valid choices generated
    """
    if kwargs is None:
        kwargs = {}

    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    if not inputs:
        raise RuntimeError(f"Custom op '{name}' requires tensor inputs for autotuning")

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=list(decompositions),
        input_nodes=list(inputs),
        kwargs=kwargs,
    )

    # Add default implementation as fallback
    if default_impl and hasattr(default_impl, "_op"):
        # Get output shape/dtype by calling default implementation with fake inputs
        with V.fake_mode:
            fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]
            fake_output = default_impl(*fake_inputs, **kwargs)

        fallback_choice = _get_or_create_fallback_choice(
            name, default_impl, fake_output, kwargs
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
        input_gen_fns = _create_user_input_gen_fns(inputs, user_input_gen_fns)

    return autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
    )


def register_custom_op_autotuning(
    custom_op: torch._ops.OpOverload,
    decompositions: list[Callable[..., Any]],
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None,
    tuning_knob: Optional[dict[str, list[Any]]] = None,
    max_autotune_configs: Optional[dict[str, list[Any]]] = None,
) -> None:
    """Register custom operation for autotuning with multiple implementations.

    Provides two-tier parameter tuning:
    - tuning_knob: Basic configs, always active
    - max_autotune_configs: Extended configs, active only when config.max_autotune=True

    Args:
        custom_op: Custom operation to register
        decompositions: Implementation functions to benchmark
        name: Operation name for identification
        input_gen_fns: Custom input generators for benchmarking
        tuning_knob: Basic parameter configurations
        max_autotune_configs: Extended parameter configurations for max_autotune mode

    Raises:
        TypeError: If decompositions is not a list/tuple
        ValueError: If no decompositions provided

    Example:
        register_custom_op_autotuning(
            torch.ops.mylib.mm_split_k.default,
            decompositions=[mm_split_k_impl],
            tuning_knob={"k_splits": [32, 64]},
            max_autotune_configs={
                "block_size": [32, 64, 128],
                "num_warps": [2, 4, 8],
            },
        )
    """
    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple, got {type(decompositions)}"
        )

    if not decompositions:
        raise ValueError("At least one decomposition must be provided")

    if name is None:
        name = f"{custom_op._name}_autotuned"

    # Generate final decomposition list with optional parameter variants
    if tuning_knob is None:
        final_decompositions = list(decompositions)
    else:
        final_decompositions = _create_parameter_variants(decompositions, tuning_knob)

    @functools.wraps(custom_op)
    def autotuning_lowering(*args: Any, **kwargs: Any) -> Any:
        """Inductor lowering function that replaces custom op calls with autotuned versions."""
        from torch._inductor import config

        # Extract tensor inputs and non-tensor parameters
        tensor_inputs, non_tensor_kwargs = _extract_tensor_inputs(args, kwargs)

        # Select decompositions based on max_autotune configuration
        if config.max_autotune and max_autotune_configs is not None:
            active_decompositions = _create_parameter_variants(
                decompositions, max_autotune_configs
            )
        else:
            active_decompositions = list(final_decompositions)

        result = autotune_custom_op(
            name=name,
            decompositions=active_decompositions,
            inputs=tensor_inputs,
            kwargs=non_tensor_kwargs,
            default_impl=custom_op,
            user_input_gen_fns=input_gen_fns,
        )

        validate_ir(result)
        return result

    lowerings[custom_op] = autotuning_lowering
