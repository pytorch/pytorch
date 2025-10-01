# Owner(s): ["module: inductor"]
"""
Custom operation autotuning support for PyTorch Inductor.

This module extends Inductor's autotuning capabilities to support arbitrary custom
operations. Users can define ATen-based decompositions for their custom ops and
automatically generate optimized implementations through Inductor's autotuning system.

Example:
    @register_custom_op_lowering(custom_ops.rmsnorm.default)
    def tuned_rmsnorm(input_tensor, weight, eps=1e-6, *, layout=None):
        return autotune_custom_op(
            "rmsnorm",
            rmsnorm_decomposition,
            [input_tensor, weight],
            {"eps": eps},
            layout=layout,
        )
"""

import functools
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch._inductor.codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, Layout, TensorBox
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch.fx.experimental.proxy_tensor import make_fx


__all__ = ["autotune_custom_op", "register_custom_op_lowering"]


class CustomOpTemplate(SubgraphTemplate):
    """Template for creating autotune choices from multiple custom op decompositions.

    Provides a clean interface for autotuning between different implementations
    of custom operations, following PyTorch Inductor patterns.
    """

    def __init__(
        self,
        name: str,
        decompositions: List[Callable[..., Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(name=name)
        self.decompositions = decompositions
        self.kwargs = kwargs

    def _infer_output_layout(self, input_nodes: List[Buffer]) -> Layout:
        """Infer correct output layout by tracing the first decomposition.

        Uses PyTorch's ir_node_to_tensor for proper example input creation,
        then runs the decomposition to determine the actual output properties.
        """
        if not input_nodes:
            raise RuntimeError("Cannot infer layout without input nodes")

        try:
            # Create example inputs using PyTorch IR utilities
            from torch._inductor.ir import ir_node_to_tensor
            from torch._inductor.virtualized import V

            example_inputs = []
            with V.fake_mode:
                for inp in input_nodes:
                    example_inputs.append(ir_node_to_tensor(inp))

            # Run first decomposition to determine output properties
            with torch.no_grad():
                fn = functools.partial(self.decompositions[0], **self.kwargs)
                output = fn(*example_inputs)

            # Extract layout from actual output tensor
            if isinstance(output, torch.Tensor):
                return FixedLayout(
                    device=output.device,
                    dtype=output.dtype,
                    size=output.shape,
                    stride=output.stride(),
                )

        except Exception:
            pass

        # Fallback to first input layout (guaranteed to exist)
        first_input = input_nodes[0]
        if hasattr(first_input, "get_layout"):
            return first_input.get_layout()

        # Last resort: create a basic layout if nothing else works
        raise RuntimeError("Unable to infer output layout from decomposition or inputs")

    def generate_choices(self, input_nodes: List[Buffer]) -> List[SubgraphChoiceCaller]:
        """Generate SubgraphChoiceCaller instances for all decompositions."""
        # Infer correct output layout once, use for all choices
        layout = self._infer_output_layout(input_nodes)

        choices = []
        for i, decomp_fn in enumerate(self.decompositions):

            def make_fx_graph(*example_inputs, fn=decomp_fn):
                return make_fx(functools.partial(fn, **self.kwargs))(*example_inputs)

            choice = SubgraphChoiceCaller(
                name=f"{self.name}_{getattr(decomp_fn, '__name__', f'impl_{i}')}_{next(SubgraphTemplate.index_counter)}",
                input_nodes=input_nodes,
                layout=layout,
                description=f"CustomOp {getattr(decomp_fn, '__name__', f'impl_{i}')}",
                make_fx_graph=make_fx_graph,
            )
            choices.append(choice)

        return choices


def autotune_custom_op(
    name: str,
    decompositions: Union[Callable[..., Any], List[Callable[..., Any]]],
    inputs: List[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    layout: Optional[Layout] = None,
) -> Union[TensorBox, Any]:
    """
    Autotune custom operations by comparing multiple decomposition implementations.

    Args:
        name: Operation name for identification
        decompositions: Single decomposition function or list of alternative implementations
        inputs: Input tensors/nodes
        kwargs: Additional arguments for decomposition functions
        layout: Output layout (inferred if None)

    Returns:
        Optimized implementation result
    """
    if kwargs is None:
        kwargs = {}

    # Normalize decompositions to list
    if callable(decompositions):
        decompositions = [decompositions]
    elif isinstance(decompositions, (list, tuple)):
        decompositions = list(decompositions)
    else:
        raise TypeError(
            f"Expected callable or sequence of callables, got {type(decompositions)}"
        )

    input_nodes = list(inputs)

    # Create template and generate all choices at once
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


def register_custom_op_lowering(custom_op: torch._ops.OpOverload) -> Callable[..., Any]:
    """Register a custom operation for autotuning lowering.

    Args:
        custom_op: The custom operation to register

    Returns:
        Decorator function for registering the lowering

    Example:
        @register_custom_op_lowering(my_ops.rmsnorm.default)
        def tuned_rmsnorm(input_tensor, weight, eps=1e-6, *, layout=None):
            return autotune_custom_op(
                "rmsnorm",
                rmsnorm_decomposition,
                [input_tensor, weight],
                {"eps": eps},
                layout=layout,
            )
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return register_lowering(custom_op, type_promotion_kind=None)(fn)

    return decorator
