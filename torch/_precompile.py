"""
Precompile API for composable ahead-of-time compilation.

This module provides the torch.Precompile API for precompiling PyTorch models
in a composable manner, allowing separate control over Dynamo tracing,
AOT Autograd transformations, and backend compilation.
"""

from typing import Any, Tuple

import torch


class Precompile:
    """
    Namespace class for precompilation APIs.

    This class provides static methods for composable ahead-of-time compilation
    of PyTorch models. The API allows users to:

    1. Use dynamo() to trace a model and capture the FX graph
    2. Use aot_autograd() to apply AOT Autograd transformations and compile

    Example::

        >>> model = MyModel()
        >>> example_input = torch.randn(2, 3)
        >>> gm, bytecode, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        >>> compiled_fn, guards = torch.Precompile.aot_autograd(gm, guards)
    """

    @staticmethod
    def dynamo(
        model: Any, *example_inputs: Any
    ) -> Tuple[torch.fx.GraphModule, Any, Any, Tuple[Any, ...]]:
        """
        Trace a model using TorchDynamo and capture the FX graph.

        Args:
            model: The PyTorch model to trace
            *example_inputs: Example inputs for tracing

        Returns:
            tuple: (gm, bytecode, guards, example_inputs) where:
                - gm: The captured FX GraphModule
                - bytecode: The captured bytecode (placeholder for future use)
                - guards: The guards for the captured graph (placeholder for future use)
                - example_inputs: The example inputs used for tracing
        """
        # Container to capture the graph module
        captured_gm = None
        captured_guards = None

        def capturing_backend(
            gm: torch.fx.GraphModule, inner_example_inputs: list
        ) -> Any:
            nonlocal captured_gm
            captured_gm = gm
            # Return the graph module's forward for execution
            return gm.forward

        def guard_export_fn(guards: Any) -> None:
            nonlocal captured_guards
            captured_guards = guards

        # Get the function to optimize
        if isinstance(model, torch.nn.Module):
            fn = model.forward
        else:
            fn = model

        # Use dynamo with a capturing backend
        optimized_fn = torch._dynamo.optimize(
            capturing_backend,
            nopython=True,
            guard_export_fn=guard_export_fn,
        )(fn)

        # Run the optimized function to trigger tracing
        if isinstance(model, torch.nn.Module):
            # Bind the method to the module
            import types

            bound_fn = types.MethodType(optimized_fn, model)
            bound_fn(*example_inputs)
        else:
            optimized_fn(*example_inputs)

        if captured_gm is None:
            raise RuntimeError(
                "Failed to capture graph. Dynamo tracing did not produce a GraphModule."
            )

        # Return (gm, bytecode, guards, example_inputs)
        # bytecode is a placeholder for future implementation
        return (captured_gm, None, captured_guards, example_inputs)

    @staticmethod
    def aot_autograd(gm, guards, compiler=None):
        """
        Apply AOT Autograd transformations and compile the graph.

        Args:
            gm: The FX GraphModule from dynamo()
            guards: The guards from dynamo()
            compiler: Optional custom compiler backend (default: inductor)

        Returns:
            tuple: (compiled_fn, guards) where:
                - compiled_fn: The compiled callable that accepts a tuple of inputs
                - guards: The output guards
        """
        from functorch.compile import min_cut_rematerialization_partition
        from torch._functorch.aot_autograd import aot_module_simplified
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._inductor.decomposition import select_decomp_table

        # Extract example inputs from graph module's placeholder nodes
        example_inputs = []
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                if "val" in node.meta:
                    example_inputs.append(node.meta["val"])
                elif "example_value" in node.meta:
                    example_inputs.append(node.meta["example_value"])

        # Use the provided compiler or default to inductor's compile_fx_inner
        if compiler is None:
            fw_compiler = compile_fx_inner
        else:
            fw_compiler = compiler

        # Apply AOT Autograd transformation using aot_module_simplified
        compiled_fn = aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=fw_compiler,
            bw_compiler=fw_compiler,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=select_decomp_table(),
            keep_inference_input_mutations=True,
        )

        # Wrap the compiled function to accept a tuple of inputs
        def wrapped_fn(inputs):
            if isinstance(inputs, tuple):
                return compiled_fn(*inputs)
            else:
                return compiled_fn(inputs)

        return (wrapped_fn, guards)
