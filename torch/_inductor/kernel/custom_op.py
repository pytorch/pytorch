# Owner(s): ["module: inductor"]
"""
Custom operation autotuning support for PyTorch Inductor.

This module extends Inductor's autotuning capabilities to support arbitrary custom
operations. Users can define ATen-based decompositions for their custom ops and
automatically generate optimized implementations through Inductor's autotuning system.

Example:
    from torch._inductor.kernel.custom_op import register_custom_op_autotuning

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
from torch._inductor.codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from torch._inductor.ir import Buffer, ChoiceCaller, FixedLayout, Layout, TensorBox
from torch._inductor.lowering import fallback_handler, lowerings, validate_ir
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.utils import do_bench_using_profiling
from torch.fx.experimental.proxy_tensor import make_fx


class BlackboxChoiceCaller(ChoiceCaller):
    """
    Wraps the original custom op implementation as a blackbox choice for autotuning.

    This allows the original custom op implementation to compete against optimized
    decompositions in autotuning, providing a performance baseline and fallback.
    """

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
        original_op_overload: torch._ops.OpOverload,
        kwargs: dict = None,
    ):
        super().__init__(name, input_nodes, layout, description)
        self.original_op_overload = original_op_overload
        self.kwargs = kwargs if kwargs is not None else {}

    def benchmark(self, *args, out=None) -> float:
        """Benchmark the original OpOverload directly"""

        def run_blackbox():
            return self.original_op_overload(*args, **self.kwargs)

        if config.profile_bandwidth_with_do_bench_using_profiling:
            result = do_bench_using_profiling(run_blackbox)
            # Ensure we return a float, not a list
            if isinstance(result, list):
                return result[0] if result else 0.0
            return result

        if self.layout.device.type == "cpu":
            result = benchmarker.benchmark_cpu(run_blackbox)
        else:
            result = benchmarker.benchmark_gpu(run_blackbox)

        # Ensure we return a float, not a list
        if isinstance(result, list):
            return result[0] if result else 0.0
        return result

    def to_callable(self):
        """Return the original OpOverload as callable"""
        return lambda *args: self.original_op_overload(*args, **self.kwargs)

    def call_name(self):
        return f"blackbox_{self.original_op_overload.name()}"

    def hash_key(self):
        return f"blackbox-{self.original_op_overload.name()}-{id(self.input_nodes)}"

    def output_node(self):
        """
        Create fallback kernel for the original OpOverload.
        This ensures that if the blackbox is selected, it can be properly executed.
        """
        handler = fallback_handler(self.original_op_overload, add_to_fallback_set=False)
        return handler(*self.input_nodes, **self.kwargs)

    def info_dict(self):
        return {
            "backend": "blackbox_fallback",
            "kernel_name": self.original_op_overload.name(),
            "description": self.description,
        }

    def autoheuristic_id(self):
        return f"blackbox_{self.original_op_overload.name()}"


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
        if self.default_impl is None:
            # Use the first decomposition as fallback for layout inference
            if not self.decompositions:
                raise RuntimeError(
                    "Cannot infer layout without default_impl or decompositions"
                )
            default_impl = self.decompositions[0]
        else:
            default_impl = self.default_impl

        with V.fake_mode:
            example_inputs = [ir_node_to_tensor(inp) for inp in input_nodes]
            fn = functools.partial(default_impl, **self.kwargs)
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
    include_blackbox: bool = True,
) -> Union[TensorBox, Any]:
    """
    Autotune custom operations by comparing multiple decomposition implementations.

    Args:
        name: Operation name for identification
        decompositions: List of decomposition function implementations to compare
        inputs: Input tensors/nodes
        kwargs: Additional arguments for decomposition functions
        layout: Output layout (inferred if None)
        default_impl: Default implementation to use as fallback (optional)
        include_blackbox: If True, include original custom op as blackbox choice (default: True)

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

    # Allow empty decompositions only if include_blackbox=True and we have a default_impl OpOverload
    if not decompositions:
        if not (
            include_blackbox
            and default_impl is not None
            and hasattr(default_impl, "_op")
        ):
            raise ValueError(
                "decompositions list cannot be empty unless include_blackbox=True with OpOverload default_impl"
            )

    # Store original OpOverload for blackbox choice
    original_op_overload = None
    processed_decompositions = decompositions.copy()
    template_default_impl = None  # What to pass to CustomOpTemplate

    # Add default implementation as fallback decomposition if provided and not using blackbox
    if default_impl is not None:
        # If default_impl is an OpOverload (the custom op itself), handle specially
        if hasattr(default_impl, "_op"):
            # Store for potential blackbox use
            original_op_overload = default_impl

            # Only add as decomposition if not using blackbox
            if not include_blackbox:
                # Create a wrapper function that calls the OpOverload DIRECTLY
                # This should bypass the lowering system to avoid recursive autotuning
                def default_wrapper(*args, **kwargs):
                    # Call the OpOverload directly, bypassing any registered lowerings
                    # This prevents recursive autotuning calls
                    from torch._inductor.ir import ir_node_to_tensor

                    # Convert IR nodes back to tensors if needed
                    tensor_args = []
                    for arg in args:
                        if hasattr(arg, "realize"):
                            # This is an IR node, convert to tensor
                            tensor_args.append(ir_node_to_tensor(arg))
                        else:
                            tensor_args.append(arg)

                    # Call the original OpOverload directly
                    return original_op_overload(*tensor_args, **kwargs)

                # Try to extract a meaningful name from the OpOverload
                try:
                    op_name = default_impl.name().replace("::", "_")
                except:
                    op_name = "custom_op"
                default_wrapper.__name__ = f"{op_name}_default"

                processed_decompositions.append(default_wrapper)
                template_default_impl = default_wrapper  # Use wrapper, NOT OpOverload
            # If include_blackbox=True, don't add OpOverload to decompositions or template
        else:
            # default_impl is already a callable function
            if default_impl not in processed_decompositions:
                processed_decompositions.append(default_impl)
            template_default_impl = default_impl

    input_nodes = list(inputs)

    # Create template and generate decomposition choices
    template = CustomOpTemplate(
        name=name,
        decompositions=processed_decompositions,
        kwargs=kwargs,
        default_impl=template_default_impl,  # Use the correct callable, never OpOverload
    )

    # If we have no decompositions but have an OpOverload, we need to handle layout inference specially
    if (
        not processed_decompositions
        and include_blackbox
        and original_op_overload is not None
    ):
        # Infer layout using the OpOverload directly
        from torch._inductor.ir import ir_node_to_tensor
        from torch._inductor.virtualized import V

        with V.fake_mode:
            example_inputs = [ir_node_to_tensor(inp) for inp in input_nodes]
            output = original_op_overload(*example_inputs, **kwargs)
            inferred_layout = FixedLayout(
                device=output.device,
                dtype=output.dtype,
                size=output.shape,
                stride=output.stride(),
            )
        choices = []  # No decomposition choices
    else:
        choices = template.generate_choices(input_nodes)
        # Use the inferred layout from the choices (all choices should have the same layout)
        inferred_layout = layout or (choices[0].layout if choices else None)

    # Add blackbox choice if requested and we have an OpOverload
    if include_blackbox and original_op_overload is not None:
        if inferred_layout is None:
            # Need to infer layout for blackbox choice
            from torch._inductor.ir import ir_node_to_tensor
            from torch._inductor.virtualized import V

            with V.fake_mode:
                example_inputs = [ir_node_to_tensor(inp) for inp in input_nodes]
                output = original_op_overload(*example_inputs, **kwargs)
                inferred_layout = FixedLayout(
                    device=output.device,
                    dtype=output.dtype,
                    size=output.shape,
                    stride=output.stride(),
                )

        blackbox_choice = BlackboxChoiceCaller(
            name=f"{name}_blackbox_original",
            input_nodes=input_nodes,
            layout=inferred_layout,
            description=f"Original {original_op_overload.name()} implementation (blackbox)",
            original_op_overload=original_op_overload,
            kwargs=kwargs,
        )
        choices.append(blackbox_choice)
        print(f"✅ Added blackbox choice for {original_op_overload.name()}")

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

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
