"""
torch.Precompile - Composable precompilation API for PyTorch models.

This module provides a phase-by-phase API for precompiling PyTorch models,
exposing each compilation phase (dynamo, aot_autograd, inductor) as separate
callable steps.

Usage:
    dynamo_output = torch.Precompile.dynamo(model, example_input)
    joint_gm, guards = torch.Precompile.aot_autograd(dynamo_output)
    compiled_joint_gm, guards = torch.Precompile.inductor(joint_gm, guards)
    precompiled_artifact = torch.Precompile.precompile(compiled_joint_gm)
    torch.Precompile.save("/tmp/model.pt", precompiled_artifact, bytecode, guards)

    precompiled_artifact = torch.Precompile.load("/tmp/model.pt")
    precompiled_artifact(example_input)
"""

from typing import Any, Callable, Optional, Tuple

import torch

from torch.precompile.types import (
    AOTAutogradOutput,
    DynamoOutput,
    GraphRuntimeEnv,
    InductorOutput,
    PrecompiledArtifact,
    SystemInfo,
)

__all__ = [
    "Precompile",
    "DynamoOutput",
    "AOTAutogradOutput",
    "InductorOutput",
    "PrecompiledArtifact",
    "GraphRuntimeEnv",
    "SystemInfo",
]


class Precompile:
    """
    Composable precompilation API for PyTorch models.

    Provides static methods for each phase of the compilation pipeline:
    - dynamo: Trace the model and capture the FX graph
    - aot_autograd: Apply AOT autograd transformations
    - inductor: Compile with the Inductor backend
    - precompile: Bundle all artifacts
    - save/load: Serialize and deserialize compiled artifacts
    """

    @staticmethod
    def dynamo(
        model: Callable[..., Any],
        example_inputs: Any,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> DynamoOutput:
        """
        Trace a model with Dynamo and capture the FX graph.

        Args:
            model: The model or callable to trace
            example_inputs: Example inputs for tracing (tuple of args)
            kwargs: Optional keyword arguments for tracing

        Returns:
            DynamoOutput containing graph_module, bytecode, guards, example_inputs, and fake_mode
        """
        from torch._dynamo import convert_frame
        from torch._dynamo.utils import get_metrics_context

        # Normalize inputs to tuple
        if not isinstance(example_inputs, tuple):
            example_inputs = (example_inputs,)

        # Capture the graph using fullgraph_capture
        # Wrap in metrics context as required by dynamo internals
        with get_metrics_context():
            capture_output = convert_frame.fullgraph_capture(
                model,
                example_inputs,
                kwargs,
            )

        # Extract components
        graph_capture_output = capture_output.graph_capture_output
        backend_input = capture_output.backend_input

        if backend_input is None:
            raise RuntimeError(
                "Dynamo did not capture any graph. This may happen if the model "
                "has no tensor operations."
            )

        # Extract guards state from output_graph
        guards_state = graph_capture_output.output_graph.dump_guards_state()

        return DynamoOutput(
            graph_module=backend_input.graph_module,
            bytecode=graph_capture_output.bytecode,
            guards=guards_state,
            example_inputs=backend_input.example_inputs,
            fake_mode=backend_input.fake_mode,
        )

    @staticmethod
    def aot_autograd(
        dynamo_output: DynamoOutput,
        *,
        decompositions: Optional[dict[str, Any]] = None,
        trace_joint: bool = True,
    ) -> AOTAutogradOutput:
        """
        Apply AOT autograd transformations to the graph.

        This traces the backward pass and creates a joint forward+backward graph.

        Args:
            dynamo_output: Output from Precompile.dynamo()
            decompositions: Optional decomposition dict for lowering ops.
                            If None, uses Inductor's default decomposition table.
            trace_joint: If True, trace joint forward+backward graph (default).
                         If False, trace only inference graph.

        Returns:
            AOTAutogradOutput containing joint_graph, guards, and metadata
        """
        from torch._functorch.aot_autograd import aot_export_joint_simple
        from torch._inductor.decomposition import select_decomp_table

        # Get graph module and example inputs from dynamo output
        graph_module = dynamo_output.graph_module
        example_inputs = dynamo_output.example_inputs
        guards = dynamo_output.guards

        # Use Inductor's default decomposition table if not specified
        if decompositions is None:
            decompositions = select_decomp_table()

        # Use aot_export_joint_simple to trace the joint graph
        # This function expects a callable and flat args
        joint_graph = aot_export_joint_simple(
            graph_module,
            tuple(example_inputs),
            trace_joint=trace_joint,
            decompositions=decompositions,
        )

        # Build metadata dict to pass to inductor
        metadata = {
            "fake_mode": dynamo_output.fake_mode,
            "example_inputs": example_inputs,
            "trace_joint": trace_joint,
        }

        return AOTAutogradOutput(
            joint_graph=joint_graph,
            guards=guards,
            metadata=metadata,
        )

    @staticmethod
    def inductor(
        aot_autograd_output: AOTAutogradOutput,
    ) -> InductorOutput:
        """
        Compile the joint graph with the Inductor backend.

        Args:
            aot_autograd_output: Output from Precompile.aot_autograd()

        Returns:
            InductorOutput containing compiled_module, guards, and kernel_artifacts
        """
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._subclasses.fake_tensor import FakeTensorMode

        joint_graph = aot_autograd_output.joint_graph
        guards = aot_autograd_output.guards
        metadata = aot_autograd_output.metadata

        # Extract example inputs from graph placeholder metadata
        # The joint graph may have more placeholders than the original inputs
        # (e.g., tangent inputs for backward). We extract the fake tensor
        # values from each placeholder node's metadata.
        example_inputs = []
        for node in joint_graph.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if val is not None:
                    example_inputs.append(val)

        # If no placeholder metadata, fall back to original example inputs
        if not example_inputs:
            example_inputs = list(metadata.get("example_inputs", []))

        # Compile with Inductor
        compiled_module = compile_fx_inner(
            joint_graph,
            example_inputs,
        )

        # Serialize kernel artifacts for portability
        # For now, we use an empty bytes placeholder - full serialization
        # will be implemented in the save/load phase
        kernel_artifacts = b""

        return InductorOutput(
            compiled_module=compiled_module,
            guards=guards,
            kernel_artifacts=kernel_artifacts,
        )

    @staticmethod
    def precompile(
        inductor_output: InductorOutput,
        *,
        runtime_env: Optional[GraphRuntimeEnv] = None,
        signature: Optional[Any] = None,
    ) -> PrecompiledArtifact:
        """
        Bundle all compilation artifacts into a PrecompiledArtifact.

        Args:
            inductor_output: Output from Precompile.inductor()
            runtime_env: Optional runtime environment configuration.
                         If None, uses default GraphRuntimeEnv.
            signature: Optional callable signature for the precompiled function.

        Returns:
            PrecompiledArtifact that can be saved, loaded, and called
        """
        import inspect

        # Use default runtime env if not provided
        if runtime_env is None:
            runtime_env = GraphRuntimeEnv()

        # Capture system info at compile time
        system_info = SystemInfo.current()

        return PrecompiledArtifact(
            inductor_output=inductor_output,
            runtime_env=runtime_env,
            signature=signature,
            system_info=system_info,
        )

    @staticmethod
    def save(
        path: str,
        artifact: Any,
        bytecode: Any,
        guards: Any,
    ) -> None:
        """
        Save a precompiled artifact to disk.

        Args:
            path: File path to save to
            artifact: The PrecompiledArtifact to save
            bytecode: The bytecode from dynamo
            guards: The accumulated guards (OutputGraphGuardsState or bytes)
        """
        import dataclasses
        import io
        import pickle
        from copy import copy

        from torch._dynamo.package import SerializedCode

        # Validate inputs
        if not isinstance(artifact, PrecompiledArtifact):
            raise TypeError(
                f"artifact must be a PrecompiledArtifact, got {type(artifact)}"
            )

        # Prepare the compiled module for serialization
        # The compiled_module may be a CompiledFxGraph which has
        # prepare_for_serialization() to clear non-serializable callables
        compiled_module = artifact.inductor_output.compiled_module
        if hasattr(compiled_module, "prepare_for_serialization"):
            # Make a shallow copy to avoid mutating the original
            serializable_module = copy(compiled_module)
            serializable_module.prepare_for_serialization()
        else:
            serializable_module = compiled_module

        # Create a serialized inductor output with the prepared module
        # Note: We don't include guards here as they're serialized separately
        # The guards in InductorOutput are redundant with the main guards parameter
        serialized_inductor = InductorOutput(
            compiled_module=serializable_module,
            guards=None,  # Guards are serialized separately
            kernel_artifacts=artifact.inductor_output.kernel_artifacts,
        )

        # Serialize bytecode using SerializedCode
        serialized_bytecode = SerializedCode.from_code_object(bytecode)

        # Serialize guards - handle both raw guards and pre-serialized bytes
        if isinstance(guards, bytes):
            serialized_guards = guards
        else:
            # Use the GuardsStatePickler approach
            # This handles complex objects like tensors, modules, etc.
            from torch._dynamo.guards import GuardsStatePickler

            buf = io.BytesIO()
            # GuardsStatePickler needs guard_tree_values, empty_values, missing_values
            # For simplicity, we use empty dicts and let the pickler handle missing objects
            pickler = GuardsStatePickler({}, {}, {}, buf)
            try:
                pickler.dump(guards)
                serialized_guards = buf.getvalue()
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                # If guards can't be serialized with GuardsStatePickler,
                # store a placeholder. Full guard support will be added in load()
                serialized_guards = pickle.dumps({"_guards_not_serializable": str(e)})

        # Bundle everything together
        save_data = {
            "version": 1,
            "inductor_output": serialized_inductor,
            "runtime_env": artifact.runtime_env,
            "signature": artifact.signature,
            "system_info": artifact.system_info,
            "bytecode": serialized_bytecode,
            "guards": serialized_guards,
        }

        # Write to file
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    @staticmethod
    def load(path: str) -> PrecompiledArtifact:
        """
        Load a precompiled artifact from disk.

        Args:
            path: File path to load from

        Returns:
            A callable PrecompiledArtifact
        """
        import pickle

        from torch._inductor.output_code import (
            CompiledFxGraph,
            CompiledFxGraphConstants,
        )

        # Load the serialized data
        with open(path, "rb") as f:
            save_data = pickle.load(f)

        # Validate version
        version = save_data.get("version")
        if version != 1:
            raise ValueError(
                f"Unsupported precompiled artifact version: {version}. "
                "This artifact may have been created with a different version of PyTorch."
            )

        # Extract the serialized components
        serialized_inductor = save_data["inductor_output"]
        runtime_env = save_data["runtime_env"]
        signature = save_data["signature"]
        system_info = save_data["system_info"]

        # Restore the compiled module's callable
        # The compiled_module is a CompiledFxGraph with current_callable = None
        compiled_module = serialized_inductor.compiled_module
        if isinstance(compiled_module, CompiledFxGraph):
            # Use after_deserialization to restore the callable
            # This writes source code to disk and loads it via PyCodeCache
            constants = CompiledFxGraphConstants()
            compiled_module.after_deserialization(constants)

        # Reconstruct the InductorOutput
        inductor_output = InductorOutput(
            compiled_module=compiled_module,
            guards=serialized_inductor.guards,
            kernel_artifacts=serialized_inductor.kernel_artifacts,
        )

        # Reconstruct and return the PrecompiledArtifact
        return PrecompiledArtifact(
            inductor_output=inductor_output,
            runtime_env=runtime_env,
            signature=signature,
            system_info=system_info,
        )
