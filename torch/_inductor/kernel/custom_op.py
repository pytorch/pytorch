# Owner(s): ["module: inductor"]
"""
Custom operation autotuning for PyTorch Inductor.

Enables automatic selection of the best performing implementation variant
for custom operations by registering multiple decompositions and tuning knobs.
"""

import functools
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
    "register_parametric_op_autotuning",
]


def _create_fallback_choice(
    name: str,
    default_impl: Callable[..., Any],
    fake_output: torch.Tensor,
    kwargs: dict[str, Any],
) -> ExternKernelChoice:
    """Create fallback ExternKernelChoice for the default implementation.

    Args:
        name: Base name for the fallback choice
        default_impl: Default implementation function
        fake_output: Fake output tensor for layout inference
        kwargs: Keyword arguments to pass to the implementation

    Returns:
        ExternKernelChoice configured for the default implementation
    """
    layout = FixedLayout(
        device=fake_output.device,
        dtype=fake_output.dtype,
        size=fake_output.shape,
        stride=fake_output.stride(),
    )

    def fallback_wrapper(*args: Any) -> Any:
        return default_impl(*args, **kwargs)

    return ExternKernelChoice(
        kernel=fallback_wrapper,
        name=f"{name}_fallback_default",
        has_out_variant=False,
        op_overload=default_impl,
        use_fallback_kernel=True,
    )


def _create_user_input_gen_fns(
    inputs: list[Any],
    user_input_gen_fns: dict[int, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[int, Callable[[Any], torch.Tensor]]:
    """Convert user-friendly input generation functions to internal format.

    User functions take fake tensors (with shape/dtype info but no real data) and return
    real tensors for benchmarking. This function bridges the user API to the internal API.

    Args:
        inputs: List of input IR nodes from compilation
        user_input_gen_fns: User-provided input generation functions

    Returns:
        Dict mapping indices to internal input generation functions
    """
    internal_input_gen_fns = {}

    with V.fake_mode:
        fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]

    for i, user_gen_fn in user_input_gen_fns.items():
        if i >= len(fake_inputs):
            continue

        fake_template = fake_inputs[i]

        def create_internal_input_gen_fn(user_function, template):
            def internal_input_gen_fn(ir_buffer):
                fake_tensor_for_user = torch.empty(
                    template.shape,
                    dtype=template.dtype,
                    device="meta",
                )
                return user_function(fake_tensor_for_user)

            return internal_input_gen_fn

        internal_input_gen_fns[i] = create_internal_input_gen_fn(
            user_gen_fn, fake_template
        )

    return internal_input_gen_fns


def _extract_tensor_inputs(
    args: tuple, kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from mixed args/kwargs for custom op processing.

    Separates tensor inputs (which need to be passed to decompositions) from
    non-tensor parameters (which are passed as keyword arguments).

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

        fallback_choice = _create_fallback_choice(
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
) -> None:
    """Register custom operation for autotuning with multiple implementations.

    Integrates with torch.library.custom_op and register_fake infrastructure.
    The default implementation is automatically included as a fallback.

    Args:
        custom_op: Custom operation to register (e.g., torch.ops.mylib.myop.default)
        decompositions: Alternative implementations to benchmark
        name: Operation name for identification (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking

    Raises:
        TypeError: If decompositions is not a list/tuple
        ValueError: If no decompositions provided

    Example:
        @torch.library.custom_op("mylib::rmsnorm", mutates_args=())
        def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            return torch.nn.functional.rms_norm(x, x.shape[-1:], weight, eps=eps)

        @rmsnorm_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8):
            return torch.empty_like(x)

        register_custom_op_autotuning(
            torch.ops.mylib.rmsnorm.default,
            decompositions=[impl1, impl2, impl3],
            input_gen_fns={
                0: lambda fake: torch.randn_like(fake, device='cuda') * 0.02,
                1: lambda fake: torch.ones_like(fake, device='cuda'),
            }
        )
    """
    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    if not decompositions:
        raise ValueError("At least one decomposition must be provided")

    if name is None:
        name = f"{custom_op._name}_autotuned"

    @functools.wraps(custom_op)
    def autotuning_lowering(*args: Any, **kwargs: Any) -> Any:
        """Inductor lowering function that replaces custom op calls with autotuned versions."""
        # Extract tensor inputs and non-tensor parameters
        tensor_inputs, non_tensor_kwargs = _extract_tensor_inputs(args, kwargs)

        result = autotune_custom_op(
            name=name,
            decompositions=decompositions,
            inputs=tensor_inputs,
            kwargs=non_tensor_kwargs,
            default_impl=custom_op,
            user_input_gen_fns=input_gen_fns,
        )

        validate_ir(result)
        return result

    lowerings[custom_op] = autotuning_lowering


def register_parametric_op_autotuning(
    custom_op: torch._ops.OpOverload,
    implementation_fn: Callable[..., Any],
    parameter_name: str,
    parameter_values: list[Any],
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None,
) -> None:
    """Register custom operation for autotuning with parameter-based variants.

    This function addresses use case 2: autotuning hyperparameters for a custom operator.
    Instead of providing explicit decompositions, users provide a single implementation
    function and specify which parameter should be autotuned with which values.

    Args:
        custom_op: Custom operation to register (e.g., torch.ops.mylib.myop.default)
        implementation_fn: Base implementation function that takes the parameter
        parameter_name: Name of the parameter to autotune
        parameter_values: List of values to try for the parameter
        name: Operation name for identification (default: "{op_name}_parametric_autotuned")
        input_gen_fns: Custom input generators for benchmarking

    Example:
        def parametric_algorithm(x: torch.Tensor, weight: torch.Tensor, method: int = 0) -> torch.Tensor:
            if method == 0:
                return x * weight  # Simple multiplication
            elif method == 1:
                return x / x.norm(dim=-1, keepdim=True) * weight  # Normalized
            elif method == 2:
                return x / torch.sqrt((x*x).mean(dim=-1, keepdim=True)) * weight  # RMS normalized

        @torch.library.custom_op("mylib::parametric_op", mutates_args=())
        def parametric_op(x: torch.Tensor, weight: torch.Tensor, method: int = 0) -> torch.Tensor:
            return parametric_algorithm(x, weight, method)

        register_parametric_op_autotuning(
            torch.ops.mylib.parametric_op.default,
            implementation_fn=parametric_algorithm,
            parameter_name="method",
            parameter_values=[0, 1, 2],
        )
    """
    if not isinstance(parameter_values, (list, tuple)):
        raise TypeError(
            f"parameter_values must be a list or tuple, got {type(parameter_values)}"
        )

    if not parameter_values:
        raise ValueError("At least one parameter value must be provided")

    if name is None:
        name = f"{custom_op._name}_parametric_autotuned"

    # Generate specialized functions for each parameter value using functools.partial
    def make_variant(value):
        """Create a specialized variant that fixes the parameter to a specific value."""
        variant = functools.partial(implementation_fn, **{parameter_name: value})
        variant.__name__ = f"{implementation_fn.__name__}_{parameter_name}_{value}"
        return variant

    decompositions = [make_variant(value) for value in parameter_values]

    register_custom_op_autotuning(
        custom_op=custom_op,
        decompositions=decompositions,
        name=name,
        input_gen_fns=input_gen_fns,
    )
