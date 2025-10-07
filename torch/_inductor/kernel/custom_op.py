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
import torch._inductor.config as config
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, ChoiceCaller, FixedLayout, Layout, TensorBox
from torch._inductor.lowering import fallback_handler, lowerings, validate_ir
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.utils import do_bench_using_profiling


class CustomOpFallbackChoice(ChoiceCaller):
    """
    Wraps the original custom op implementation as a fallback choice for autotuning.

    When a custom op is autotuned, this allows the original implementation (from
    @torch.library.custom_op) to compete against decomposed variants. This provides:
    - Performance baseline for comparison
    - Safe fallback if decompositions fail
    - Direct benchmarking of the original implementation
    """

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
        default_op_overload: torch._ops.OpOverload,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.default_op_overload = default_op_overload
        self.kwargs = kwargs or {}

    def benchmark(self, *args, out=None) -> float:
        """Benchmark the original OpOverload directly."""

        def run_default_op():
            return self.default_op_overload(*args, **self.kwargs)

        if config.profile_bandwidth_with_do_bench_using_profiling:
            result = do_bench_using_profiling(run_default_op)
            if isinstance(result, list):
                return result[0] if result else 0.0
            return result

        if self.layout.device.type == "cpu":
            result = benchmarker.benchmark_cpu(run_default_op)
        else:
            result = benchmarker.benchmark_gpu(run_default_op)

        if isinstance(result, list):
            return result[0] if result else 0.0
        return result

    def to_callable(self) -> Callable[..., Any]:
        """Return the default OpOverload as callable."""
        return lambda *args: self.default_op_overload(*args, **self.kwargs)

    def call_name(self) -> str:
        return f"fallback_{self.default_op_overload.name()}"

    def hash_key(self) -> str:
        return f"fallback-{self.default_op_overload.name()}-{id(self.input_nodes)}"

    def output_node(self) -> TensorBox:
        """Create fallback kernel for the default OpOverload."""
        handler = fallback_handler(self.default_op_overload, add_to_fallback_set=False)
        return handler(*self.input_nodes, **self.kwargs)

    def info_dict(self) -> dict[str, str]:
        return {
            "backend": "fallback_default",
            "kernel_name": self.default_op_overload.name(),
            "description": self.description,
        }

    def autoheuristic_id(self) -> str:
        return f"fallback_{self.default_op_overload.name()}"


__all__ = ["autotune_custom_op", "register_custom_op_autotuning"]


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    kwargs: Optional[dict[str, Any]] = None,
    layout: Optional[Layout] = None,
    default_impl: Optional[Callable[..., Any]] = None,
    input_gen_fns: Optional[dict[int, Callable[[Buffer], torch.Tensor]]] = None,
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

    # Store default OpOverload for fallback choice
    default_op_overload = None
    processed_decompositions = decompositions.copy()
    template_default_impl = None

    # Handle default implementation
    if default_impl is not None:
        # If default_impl is an OpOverload (the custom op itself), store it for fallback use
        if hasattr(default_impl, "_op"):
            default_op_overload = default_impl
        else:
            # default_impl is already a callable function
            if default_impl not in processed_decompositions:
                processed_decompositions.append(default_impl)
            template_default_impl = default_impl

    input_nodes = list(inputs)

    # Generate decomposition choices using SubgraphTemplate's new method
    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=processed_decompositions,
        input_nodes=input_nodes,
        kwargs=kwargs,
        default_impl=template_default_impl,
        input_gen_fns=input_gen_fns,
    )

    # Use the inferred layout from the choices (all choices should have the same layout)
    inferred_layout = layout or (choices[0].layout if choices else None)

    # Add fallback choice automatically if we have an OpOverload
    if default_op_overload is not None:
        if inferred_layout is None:
            # Need to infer layout for fallback choice
            from torch._inductor.ir import ir_node_to_tensor
            from torch._inductor.virtualized import V

            with V.fake_mode:
                example_inputs = [ir_node_to_tensor(inp) for inp in input_nodes]
                output = default_op_overload(*example_inputs, **kwargs)
                inferred_layout = FixedLayout(
                    device=output.device,
                    dtype=output.dtype,
                    size=output.shape,
                    stride=output.stride(),
                )

        fallback_choice = CustomOpFallbackChoice(
            name=f"{name}_fallback_default",
            input_nodes=input_nodes,
            layout=inferred_layout,
            description=f"Default {default_op_overload.name()} implementation (fallback)",
            default_op_overload=default_op_overload,
            kwargs=kwargs,
        )
        choices.append(fallback_choice)

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    return autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=input_nodes,
        layout=inferred_layout,
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
