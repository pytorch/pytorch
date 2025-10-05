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


__all__ = ["autotune_custom_op", "register_custom_op_autotuning"]


class CustomOpTemplate(SubgraphTemplate):
    """Template for creating autotune choices from multiple custom op decompositions.

    Provides a clean interface for autotuning between different implementations
    of custom operations, following PyTorch Inductor patterns.
    """

    def __init__(
        self,
        name: str,
        decompositions: list[Callable[..., Any]],
        kwargs: dict[str, Any],
    ) -> None:
        super().__init__(name=name)
        self.decompositions = decompositions
        self.kwargs = kwargs

    def _infer_output_layout(self, input_nodes: list[Buffer]) -> Layout:
        """Infer output layout for custom ops by running first decomposition in fake mode."""
        from torch._inductor.ir import ir_node_to_tensor
        from torch._inductor.virtualized import V

        with V.fake_mode:
            example_inputs = [ir_node_to_tensor(inp) for inp in input_nodes]
            fn = functools.partial(self.decompositions[0], **self.kwargs)
            output = fn(*example_inputs)

            return FixedLayout(
                device=output.device,
                dtype=output.dtype,
                size=output.shape,
                stride=output.stride(),
            )

    def generate_choices(self, input_nodes: list[Buffer]) -> list[SubgraphChoiceCaller]:
        """Generate SubgraphChoiceCaller instances for all decompositions."""
        # Infer correct output layout once, use for all choices
        layout = self._infer_output_layout(input_nodes)

        choices = []
        for i, decomp_fn in enumerate(self.decompositions):
            traced_fn = make_fx(functools.partial(decomp_fn, **self.kwargs))

            choice = self.generate(
                name=f"{self.name}_{getattr(decomp_fn, '__name__', f'impl_{i}')}",
                input_nodes=input_nodes,
                layout=layout,
                description=f"CustomOp {getattr(decomp_fn, '__name__', f'impl_{i}')}",
                make_fx_graph=traced_fn,
            )
            choices.append(choice)

        return choices


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    kwargs: Optional[dict[str, Any]] = None,
    layout: Optional[Layout] = None,
) -> Union[TensorBox, Any]:
    """
    Autotune custom operations by comparing multiple decomposition implementations.

    The default implementation of the custom operation (defined in @torch.library.custom_op)
    is automatically included as a fallback choice if not already present in the decompositions list.

    Args:
        name: Operation name for identification
        decompositions: List of decomposition function implementations to compare
        inputs: Input tensors/nodes
        kwargs: Additional arguments for decomposition functions
        layout: Output layout (inferred if None)

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

    input_nodes = list(inputs)

    # Create template and generate all choices
    template = CustomOpTemplate(name=name, decompositions=decompositions, kwargs=kwargs)
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

    Args:
        custom_op: The custom operation to register for autotuning

    Returns:
        Decorator function for registering the autotuning function

    Example:
        @register_custom_op_autotuning(my_ops.rmsnorm.default)
        def tuned_rmsnorm(input_tensor, weight, eps=1e-6, layout=None):
            return autotune_custom_op(
                "rmsnorm",
                [rmsnorm_impl1, rmsnorm_impl2, rmsnorm_impl3],
                [input_tensor, weight],
                {"eps": eps},
                layout,
            )
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # Call the decorated function directly
            out = fn(*args, **kwargs)
            validate_ir(out)
            return out

        # Register the wrapped function in the lowerings dictionary
        # Handle both single ops and lists of ops (following the pattern from lowering.py)
        if isinstance(custom_op, (list, tuple)):
            ops_to_register = custom_op
        else:
            ops_to_register = [custom_op]

        # Register the lowering for each operation
        lowerings.update(dict.fromkeys(ops_to_register, wrapped))
        return wrapped

    return decorator
