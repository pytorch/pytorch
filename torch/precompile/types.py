"""
Type definitions for the torch.Precompile API.

This module contains dataclasses for the outputs of each compilation phase.
"""

import inspect
import platform
import types
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch


@dataclass
class DynamoOutput:
    """
    Output from the Precompile.dynamo() phase.

    Contains the traced FX graph and associated metadata needed for subsequent
    compilation phases.

    Attributes:
        graph_module: The FX GraphModule captured by Dynamo
        bytecode: The Python bytecode object for the traced function
        guards: Serializable guards that must be checked at runtime
        example_inputs: The example inputs used during tracing
        fake_mode: The FakeTensorMode used during tracing
    """

    graph_module: torch.fx.GraphModule
    bytecode: types.CodeType
    guards: Any  # GuardsState - serializable guards
    example_inputs: List[torch.Tensor]
    fake_mode: Any  # FakeTensorMode


@dataclass
class AOTAutogradOutput:
    """
    Output from the Precompile.aot_autograd() phase.

    Contains the joint forward+backward graph and associated metadata needed
    for the inductor compilation phase.

    Attributes:
        joint_graph: The FX GraphModule containing joint forward and backward
        guards: Accumulated guards from dynamo and aot_autograd phases
        metadata: AOTConfig containing configuration for inductor
    """

    joint_graph: torch.fx.GraphModule
    guards: Any  # GuardsState - accumulated guards
    metadata: Any  # AOTConfig - configuration for inductor


@dataclass
class InductorOutput:
    """
    Output from the Precompile.inductor() phase.

    Contains the compiled module and associated metadata from Inductor compilation.

    Attributes:
        compiled_module: The compiled FX graph from Inductor (CompiledFxGraph or callable)
        guards: Accumulated guards from all previous phases
        kernel_artifacts: Serialized kernel artifacts (e.g., Triton kernels) as bytes
    """

    compiled_module: Any  # CompiledFxGraph or callable
    guards: Any  # GuardsState - accumulated guards
    kernel_artifacts: bytes  # Serialized kernels for portability


@dataclass
class SystemInfo:
    """
    System information captured at compile time.

    Used for validating compatibility when loading precompiled artifacts.

    Attributes:
        python_version: Python version string (e.g., "3.10.12")
        torch_version: PyTorch version string
        cuda_version: CUDA version string, or None if not available
        platform: Platform identifier (e.g., "Linux-5.15.0-x86_64")
    """

    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    platform: str

    @classmethod
    def current(cls) -> "SystemInfo":
        """Capture current system information."""
        cuda_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda

        return cls(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            platform=platform.platform(),
        )


@dataclass
class GraphRuntimeEnv:
    """
    Runtime environment configuration for the compiled graph.

    Contains settings and configuration needed at runtime to execute
    the precompiled artifact correctly.

    Attributes:
        device: Target device for execution (e.g., "cuda:0", "cpu")
        dtype: Default dtype for tensor operations
        grad_enabled: Whether gradients should be enabled
    """

    device: str = "cpu"
    dtype: Optional[torch.dtype] = None
    grad_enabled: bool = False


@dataclass
class PrecompiledArtifact:
    """
    Bundle of all compilation artifacts from the Precompile pipeline.

    This is the final output from Precompile.precompile() and contains
    everything needed to save, load, and execute a precompiled model.

    Attributes:
        inductor_output: The compiled module and associated artifacts from Inductor
        runtime_env: Runtime environment configuration for execution
        signature: The callable signature of the original function
        system_info: System information captured at compile time
    """

    inductor_output: InductorOutput
    runtime_env: GraphRuntimeEnv = field(default_factory=GraphRuntimeEnv)
    signature: Optional[inspect.Signature] = None
    system_info: SystemInfo = field(default_factory=SystemInfo.current)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the precompiled artifact.

        Args:
            *args: Positional arguments to pass to the compiled function.
                   For models, these should be all traced inputs (params + user inputs)
                   from dynamo_output.example_inputs.
            **kwargs: Keyword arguments to pass to the compiled function

        Returns:
            Output from the compiled function
        """
        compiled_fn = self.inductor_output.compiled_module

        # Check if this is a CompiledFxGraph which expects inputs as a sequence
        # rather than positional arguments
        if hasattr(compiled_fn, "current_callable"):
            # CompiledFxGraph expects a list of inputs
            return compiled_fn(list(args))
        else:
            return compiled_fn(*args, **kwargs)
