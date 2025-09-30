"""
Custom Op AutoTuning Support for PyTorch Inductor.

This module extends PyTorch Inductor's autotuning capabilities to support arbitrary
custom operations. Users can define ATen-based decompositions for their custom ops
and automatically generate optimized implementations through Inductor's autotuning
system.

The implementation follows the same patterns as existing autotuning infrastructure
(e.g., DecomposeK for matrix multiplication) to ensure seamless integration with
Inductor's optimization pipeline.

Example usage:
    def rmsnorm_decomposition(input_tensor, weight, eps=1e-6):
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance + eps)
        return normalized * weight

    @register_custom_op_lowering(custom_ops.rmsnorm.default)
    def tuned_rmsnorm(input_tensor, weight, eps=1e-6, *, layout=None):
        return autotune_custom_op(
            op_name="rmsnorm",
            decomposition_fn=rmsnorm_decomposition,
            inputs=[input_tensor, weight],
            kwargs={"eps": eps},
            layout=layout,
        )
"""

import functools
import itertools
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch._inductor import config
from torch._inductor.codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from torch._inductor.ir import Buffer, ExternKernel, Layout, TensorBox
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx


class CustomOpSubgraphTemplate(SubgraphTemplate):
    """
    Template for creating subgraph choices from custom operation decompositions.

    This template allows users to define custom operations through ATen-based
    decomposition functions and automatically generate SubgraphChoiceCaller
    instances for Inductor's autotuning system.

    Similar to DecomposeKSugraphTemplate but generalized for arbitrary custom
    operations beyond matrix multiplication.

    Args:
        name: Unique template name for identification
        decomposition_fn: Function that implements the custom op using ATen operations

    Example:
        def my_custom_norm(x, eps=1e-6):
            return x / (x.norm(dim=-1, keepdim=True) + eps)

        template = CustomOpSubgraphTemplate("custom_norm", my_custom_norm)
    """

    def __init__(self, name: str, decomposition_fn: Callable):
        """Initialize custom op template with decomposition function."""
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
        """
        Generate a SubgraphChoiceCaller for the custom op decomposition.

        Args:
            name: Unique name for this choice
            input_nodes: Input IR nodes
            layout: Output layout
            inputs: Actual input tensors for tracing
            kwargs: Additional arguments for the decomposition function
            description: Human-readable description
        """

        def make_fx_graph(*example_inputs):
            """Create FX graph from the decomposition function"""
            # Create FX graph without fake_mode for now - will be handled by SubgraphChoiceCaller
            return make_fx(functools.partial(self.decomposition_fn, **kwargs))(
                *example_inputs
            )

        return SubgraphChoiceCaller(
            name=f"{name}_{next(SubgraphTemplate.index_counter)}",
            input_nodes=input_nodes,
            layout=layout,
            description=f"{description} (decomposition: {self.decomposition_fn.__name__})",
            make_fx_graph=make_fx_graph,
        )


# Note: CustomOpExternTemplate removed for now to avoid KernelTemplate import issues
# Will be implemented later when extern kernel support is needed


class CustomOpExternChoiceCaller:
    """
    Choice caller that directly invokes the custom op as a blackbox.
    Similar to ExternKernelCaller but for custom ops.
    """

    def __init__(
        self,
        name: str,
        input_nodes: List[Buffer],
        layout: Layout,
        custom_op: torch._ops.OpOverload,
        kwargs: Dict[str, Any],
        description: str = "",
    ):
        self.name = name
        self.input_nodes = input_nodes
        self.layout = layout
        self.custom_op = custom_op
        self.kwargs = kwargs
        self.description = description

    def benchmark(self, *args, out: torch.Tensor) -> float:
        """Benchmark the custom op directly"""
        from torch._inductor.runtime.benchmarking import benchmarker

        def run_custom_op():
            return self.custom_op(*args, **self.kwargs)

        if config.profile_bandwidth_with_do_bench_using_profiling:
            from torch._inductor.utils import do_bench_using_profiling

            return do_bench_using_profiling(run_custom_op)
        return benchmarker.benchmark_gpu(run_custom_op)

    def hash_key(self) -> str:
        return "-".join(
            [
                self.name,
                str(self.custom_op),
                *[str(inp.get_size()) for inp in self.input_nodes],
                *[str(inp.get_stride()) for inp in self.input_nodes],
                str(sorted(self.kwargs.items())),
            ]
        )

    def output_node(self) -> Union[TensorBox, Any]:
        """Create output node for this choice"""
        # For now, we'll use a simple extern kernel approach
        # In practice, this might need more sophisticated IR handling
        from torch._inductor.ir import ExternKernel

        return TensorBox.create(
            ExternKernel.create(
                inputs=self.input_nodes,
                output_layout=self.layout,
                python_kernel_name=f"custom_op_{self.name}",
                cpp_kernel_name=f"custom_op_{self.name}",
            )
        )

    def info_dict(self) -> Dict[str, Any]:
        return {
            "backend": "custom_op_extern",
            "kernel_name": self.name,
            "custom_op": str(self.custom_op),
        }

    def autoheuristic_id(self) -> str:
        return f"custom_op_extern_{self.name}"


def autotune_custom_op(
    op_name: str,
    decomposition_fn: Callable,
    inputs: List[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    layout: Optional[Layout] = None,
    custom_op: Optional[torch._ops.OpOverload] = None,
) -> Union[TensorBox, Any]:
    """
    Main entry point for custom op autotuning.

    Args:
        op_name: Name of the custom operation (e.g., "rmsnorm")
        decomposition_fn: Function that implements the op using ATen operations
        inputs: Input tensors/nodes (can be Tensors or IR Buffers)
        kwargs: Additional arguments for the decomposition
        layout: Output layout (inferred if None)
        custom_op: Optional custom op for extern comparison

    Returns:
        Optimized implementation (choice output or MultiTemplateBuffer)
    """
    if kwargs is None:
        kwargs = {}

    # Convert inputs to proper IR nodes (simplified approach for now)
    input_nodes = []
    for inp in inputs:
        if hasattr(inp, "get_layout"):  # Already an IR Buffer
            input_nodes.append(inp)
        else:
            # For testing purposes, we'll use the tensor directly
            # In real graph lowering, tensors are already converted to IR nodes
            input_nodes.append(inp)

    # Infer layout if not provided (following mm.py pattern)
    if layout is None and input_nodes:
        # Handle different input types
        first_input = input_nodes[0]
        if hasattr(first_input, "get_layout"):
            base_layout = first_input.get_layout()
            layout = base_layout
        elif isinstance(first_input, torch.Tensor):
            # Create a simple FixedLayout from tensor
            from torch._inductor.ir import FixedLayout

            layout = FixedLayout(
                device=first_input.device,
                dtype=first_input.dtype,
                size=first_input.shape,
                stride=first_input.stride(),
            )
        else:
            # Default layout - will be handled by autotune_select_algorithm
            layout = None

    # Collect template choices (following tuned_mm pattern)
    choices = []

    # Choice 1: User decomposition via SubgraphTemplate
    decomp_template = CustomOpSubgraphTemplate(
        name=f"{op_name}_decomposition", decomposition_fn=decomposition_fn
    )

    # Generate choice from template (similar to V.choices.get_template_configs)
    try:
        choice = decomp_template.generate(
            name=f"{op_name}_subgraph",
            input_nodes=input_nodes,
            layout=layout,
            inputs=inputs,
            kwargs=kwargs,
            description=f"Custom op {op_name} decomposition",
        )
        choices.append(choice)
    except Exception as e:
        # Fallback gracefully when choice generation fails
        import warnings

        warnings.warn(f"Custom op autotuning failed for {op_name}: {e}")
        return decomposition_fn(*inputs, **kwargs)

    # Choice 2: Direct custom op execution (if available)
    # TODO: Implement CustomOpExternChoiceCaller when needed

    # Call autotune_select_algorithm (following tuned_mm pattern)
    try:
        return autotune_select_algorithm(
            name=op_name,
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
        )
    except Exception as e:
        import logging

        log = logging.getLogger(__name__)
        log.warning(f"Autotuning failed for {op_name}: {e}")
        # Fallback: return direct decomposition result
        return decomposition_fn(*inputs, **kwargs)


def register_custom_op_lowering(custom_op: torch._ops.OpOverload):
    """
    Decorator to register a custom op for autotuning lowering.

    Usage:
        @register_custom_op_lowering(my_ops.rmsnorm.default)
        def tuned_rmsnorm(input_tensor, weight, eps=1e-6, *, layout=None):
            return autotune_custom_op(
                op_name="rmsnorm",
                decomposition_fn=rmsnorm_decomposition,
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
                layout=layout,
                custom_op=my_ops.rmsnorm.default,  # Optional for extern choice
            )
    """

    def decorator(fn):
        # Register with Inductor's lowering system
        return register_lowering(custom_op, type_promotion_kind=None)(fn)

    return decorator


# Example usage template for users
class CustomOpAutoTuneExample:
    """
    Example showing how to use the custom op autotuning framework.

    This demonstrates the complete flow from decomposition definition
    to registration and usage.
    """

    @staticmethod
    def example_rmsnorm_decomposition(
        input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ):
        """
        Example RMSNorm implementation using ATen operations.
        This function will be traced into an FX graph for autotuning.
        """
        # Compute RMS normalization
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance + eps)
        return normalized * weight

    @staticmethod
    def register_example():
        """
        Example of how a user would register their custom op.

        Note: This assumes you have a custom op defined as:
        # my_custom_ops.rmsnorm.default
        """

        # This is what users would write:
        """
        @register_custom_op_lowering(my_custom_ops.rmsnorm.default)
        def tuned_rmsnorm(input_tensor, weight, eps=1e-6, *, layout=None):
            return autotune_custom_op(
                op_name="rmsnorm",
                decomposition_fn=CustomOpAutoTuneExample.example_rmsnorm_decomposition,
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
                layout=layout,
                custom_op=my_custom_ops.rmsnorm.default,  # For extern comparison
            )
        """
        pass


if __name__ == "__main__":
    print("Custom Op AutoTuning Framework for PyTorch Inductor")
    print("See CustomOpAutoTuneExample for usage patterns.")
