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
    """Template for creating custom operation choices from decomposition functions.

    Args:
        name: Template identifier
        decomposition_fn: Function implementing the operation using ATen ops
    """

    def __init__(self, name: str, decomposition_fn: Callable) -> None:
        super().__init__(name=name)
        self.decomposition_fn = decomposition_fn

    def generate(
        self,
        name: str,
        input_nodes: List[Buffer],
        layout: Layout,
        inputs: List[Any],
        kwargs: Dict[str, Any],
        description: str = "",
        **template_kwargs: Any,
    ) -> SubgraphChoiceCaller:
        """Generate a SubgraphChoiceCaller for this decomposition."""

        def make_fx_graph(*example_inputs):
            return make_fx(functools.partial(self.decomposition_fn, **kwargs))(
                *example_inputs
            )

        return SubgraphChoiceCaller(
            name=f"{name}_{next(SubgraphTemplate.index_counter)}",
            input_nodes=input_nodes,
            layout=layout,
            description=(
                f"CustomOp {self.decomposition_fn.__name__}"
                if description == ""
                else description
            ),
            make_fx_graph=make_fx_graph,
        )

    def maybe_append_choice(self, choices, input_nodes, layout, inputs, kwargs):
        """Maybe append this choice to the choices list"""
        try:
            choice = self.generate(
                name=f"{self.name}_{next(SubgraphTemplate.index_counter)}",
                input_nodes=input_nodes,
                layout=layout,
                inputs=inputs,
                kwargs=kwargs,
            )
            choices.append(choice)
        except Exception as e:
            print(f"Failed to add choice {self.decomposition_fn.__name__}: {e}")


def autotune_custom_op(
    name: str,
    decompositions: Union[Callable, List[Callable]],
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

    Example:
        # Single implementation (heuristic selection)
        result = autotune_custom_op(
            name="rmsnorm",
            decompositions=rmsnorm_impl,
            inputs=[x, w],
            kwargs={"eps": 1e-6}
        )

        # Multiple implementations (benchmarked selection)
        result = autotune_custom_op(
            name="rmsnorm",
            decompositions=[impl1, impl2, impl3],
            inputs=[x, w],
            kwargs={"eps": 1e-6}
        )
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

    # Convert inputs to IR nodes (simplified for current implementation)
    input_nodes = []
    for inp in inputs:
        if hasattr(inp, "get_layout"):
            input_nodes.append(inp)
        else:
            input_nodes.append(inp)

    # Infer layout from first input if not provided
    if layout is None and input_nodes:
        first_input = input_nodes[0]
        if hasattr(first_input, "get_layout"):
            layout = first_input.get_layout()
        elif isinstance(first_input, torch.Tensor):
            layout = FixedLayout(
                device=first_input.device,
                dtype=first_input.dtype,
                size=first_input.shape,
                stride=first_input.stride(),
            )

    # Generate choices from decomposition functions
    choices = []

    for i, decomp_fn in enumerate(decompositions):
        template = CustomOpTemplate(
            name=f"{name}_{getattr(decomp_fn, '__name__', f'impl_{i}')}",
            decomposition_fn=decomp_fn,
        )

        # Use the new maybe_append_choice method
        template.maybe_append_choice(choices, input_nodes, layout, inputs, kwargs)

    # Fallback to direct execution if no choices generated
    if not choices:
        import warnings

        warnings.warn(f"No valid choices generated for {name}, using fallback")
        return decompositions[0](*inputs, **kwargs)

    # Use autotuning system
    try:
        return autotune_select_algorithm(
            name=name,
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
        )
    except Exception:
        import warnings

        warnings.warn(f"Autotuning failed for {name}, using fallback")
        return decompositions[0](*inputs, **kwargs)


def register_custom_op_lowering(custom_op: torch._ops.OpOverload):
    """Register a custom operation for autotuning lowering.

    Args:
        custom_op: The custom operation to register

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

    def decorator(fn):
        return register_lowering(custom_op, type_promotion_kind=None)(fn)

    return decorator
