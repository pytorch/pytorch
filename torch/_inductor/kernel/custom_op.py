# Owner(s): ["module: inductor"]
"""
Custom operation autotuning support for PyTorch Inductor.

This module extends Inductor's autotuning capabilities to support arbitrary custom
operations. Users can define ATen-based decompositions for their custom ops and
automatically generate optimized implementations through Inductor's autotuning system.

Example:
    from torch._inductor.kernel.custom_op import register_custom_op_autotuning

    @register_custom_op_autotuning(custom_ops.rmsnorm.default)
    def tuned_rmsnorm(input_tensor, weight, eps=1e-6):
        return autotune_custom_op(
            "rmsnorm",
            [rmsnorm_decomposition1, rmsnorm_decomposition2],
            [input_tensor, weight],
            {"eps": eps},
        )
"""

import functools
from typing import Any, Callable, Optional, Union

import torch
from torch._inductor.codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, Layout, TensorBox
from torch._inductor.lowering import lowerings, validate_ir
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch.fx.experimental.proxy_tensor import make_fx


class CustomOpTemplate(SubgraphTemplate):
    """Template for generating custom operation choices for autotuning.

    This template manages multiple decomposition implementations of a custom operation
    and generates appropriate choices for the autotuning system.
    """

    def __init__(
        self,
        name: str,
        decompositions: list[Callable[..., Any]],
        kwargs: dict[str, Any],
        default_impl: Callable[..., Any],
    ) -> None:
        super().__init__(name)
        self.decompositions = decompositions
        self.kwargs = kwargs
        self.default_impl = default_impl

    def _infer_output_layout(self, input_nodes: list[Buffer]) -> Layout:
        """Infer output layout for custom ops using the default implementation when available."""
        from torch._inductor.ir import ir_node_to_tensor
        from torch._inductor.virtualized import V

        # If we have a default implementation, use it directly for layout inference
        # since it's guaranteed to work
        with V.fake_mode:
            example_inputs = [ir_node_to_tensor(inp) for inp in input_nodes]
            fn = functools.partial(self.default_impl, **self.kwargs)
            output = fn(*example_inputs)

            return FixedLayout(
                device=output.device,
                dtype=output.dtype,
                size=output.shape,
                stride=output.stride(),
            )

    def generate_choices(self, input_nodes: list[Buffer]) -> list[SubgraphChoiceCaller]:
        """Generate SubgraphChoiceCaller instances for all working decompositions."""
        # Infer correct output layout once, use for all choices
        layout = self._infer_output_layout(input_nodes)

        choices = []
        for i, decomp_fn in enumerate(self.decompositions):
            traced_fn = make_fx(functools.partial(decomp_fn, **self.kwargs))
            func_name = getattr(decomp_fn, "__name__", f"impl_{i}")

            choice = self.generate(
                name=f"{self.name}_{func_name}",
                input_nodes=input_nodes,
                layout=layout,
                description=f"CustomOp {func_name}",
                make_fx_graph=traced_fn,
            )
            choices.append(choice)

        return choices


__all__ = ["autotune_custom_op", "register_custom_op_autotuning"]


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    kwargs: Optional[dict[str, Any]] = None,
    layout: Optional[Layout] = None,
    default_impl: Optional[Callable[..., Any]] = None,
) -> Union[TensorBox, Any]:
    """
    Autotune custom operations by comparing multiple decomposition implementations.

    The default implementation from @torch.library.custom_op is automatically
    added as a fallback choice if provided and not already present in the decompositions list.

    Args:
        name: Operation name for identification
        decompositions: List of decomposition function implementations to compare
        inputs: Input tensors/nodes
        kwargs: Additional arguments for decomposition functions
        layout: Output layout (inferred if None)
        default_impl: Default implementation to use as fallback (optional)

    Returns:
        Optimized implementation result
    """
    if kwargs is None:
        kwargs = {}

    # Validate that decompositions is always a list
    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    decompositions = list(decompositions)
    if not decompositions:
        raise ValueError("decompositions list cannot be empty")

    # Add default implementation as fallback if provided and not already present
    if default_impl and default_impl not in decompositions:
        decompositions.append(default_impl)

    input_nodes = list(inputs)

    # Create template and generate all choices at once
    template = CustomOpTemplate(
        name=name,
        decompositions=decompositions,
        kwargs=kwargs,
        default_impl=default_impl,
    )

    choices = template.generate_choices(input_nodes)

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    # Use the inferred layout from the choices (all choices should have the same layout)
    inferred_layout = layout or choices[0].layout

    return autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=input_nodes,
        layout=inferred_layout,
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
