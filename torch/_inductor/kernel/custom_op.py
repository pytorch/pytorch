# Owner(s): ["module: inductor"]
"""
Custom operation autotuning support for PyTorch Inductor.

This module extends Inductor's autotuning capabilities to support custom operations.
Users can define decompositions for their custom ops and automatically generate
optimized implementations through Inductor's autotuning system.

Example:
    @register_custom_op_autotuning(my_custom_ops.my_op.default)
    def tuned_my_op(input_tensor, other_tensor, param=1.0):
        return autotune_custom_op(
            "my_op_autotuned",
            [impl_variant1, impl_variant2, impl_variant3],
            [input_tensor, other_tensor],
            {"param": param},
        )
"""

import functools
from typing import Any, Callable, Optional, Union

import torch
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, TensorBox
from torch._inductor.lowering import lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)


__all__ = ["autotune_custom_op", "register_custom_op_autotuning"]


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    kwargs: Optional[dict[str, Any]] = None,
    default_impl: Optional[Callable[..., Any]] = None,
) -> Union[TensorBox, Any]:
    """
    Autotune custom operations by comparing multiple decomposition implementations.

    The original custom op implementation is automatically included as a blackbox
    choice when default_impl is provided as an OpOverload, providing a performance
    baseline and fallback option.

    Args:
        name: Operation name for identification
        decompositions: List of decomposition function implementations to compare
        inputs: Input tensors/nodes
        kwargs: Additional arguments for decomposition functions
        default_impl: Default implementation to use as fallback (optional)

    Returns:
        Optimized implementation result
    """
    if kwargs is None:
        kwargs = {}

    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    input_nodes = list(inputs)

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=list(decompositions),
        input_nodes=input_nodes,
        kwargs=kwargs,
        default_impl=None,
    )

    # Handle default custom op implementation as fallback using ExternKernelChoice
    if default_impl and hasattr(default_impl, "_op"):
        from torch._inductor.ir import ir_node_to_tensor
        from torch._inductor.virtualized import V

        with V.fake_mode:
            example_inputs = [ir_node_to_tensor(inp) for inp in input_nodes]
            output = default_impl(*example_inputs, **kwargs)
            layout = FixedLayout(
                device=output.device,
                dtype=output.dtype,
                size=output.shape,
                stride=output.stride(),
            )

        def fallback_wrapper(*args: Any) -> Any:
            return default_impl(*args, **kwargs)

        extern_choice = ExternKernelChoice(
            kernel=fallback_wrapper,
            name=f"{name}_fallback_default",
            has_out_variant=False,
            op_overload=default_impl,
            use_fallback_kernel=True,
        )

        extern_choice.maybe_append_choice(
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
        )

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    def simple_randn_gen(x: Buffer) -> torch.Tensor:
        """Generate real tensor from fake tensor/buffer for autotuning."""
        from torch._inductor.virtualized import V

        return torch.randn(
            [V.graph.sizevars.size_hint(i) for i in x.get_size()],
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    input_gen_fns = dict.fromkeys(range(len(input_nodes)), simple_randn_gen)

    return autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=input_nodes,
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
    )


def register_custom_op_autotuning(
    custom_op: torch._ops.OpOverload,
) -> Callable[..., Any]:
    """Register a custom operation for autotuning with multiple decomposition choices.

    This function provides a clean API for registering custom operation autotuning.
    It registers the operation for inductor lowering so that multiple decomposition
    choices can be autotuned at the inductor level.

    The default implementation from @torch.library.custom_op is automatically included
    as a fallback choice and will be used if all other choices fail.

    Args:
        custom_op: The custom operation to register for autotuning

    Returns:
        Decorator function for registering the autotuning function

    Example:
        @register_custom_op_autotuning(my_ops.rmsnorm.default)
        def tuned_rmsnorm(input_tensor, weight, eps=1e-6):
            return autotune_custom_op(
                "rmsnorm",
                [rmsnorm_impl1, rmsnorm_impl2, rmsnorm_impl3],
                [input_tensor, weight],
                {"eps": eps},
            )
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = fn(*args, **kwargs, default_impl=custom_op)
            validate_ir(out)
            return out

        # Register the wrapped function in the lowerings dictionary
        lowerings[custom_op] = wrapped
        return wrapped

    return decorator
