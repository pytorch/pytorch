"""
CUDAGraphAdapter for translating CUDA graph configuration to CUDAGraphSetupNode IR nodes.

This module provides functionality to translate CUDA graph configuration settings
from torch.compile and Inductor into the structured CUDAGraphSetupNode IR representation.
The translated IR nodes can then be used by code generation backends to produce
either runtime CUDA graph setup code (gen_binary) or explicit Python code for CUDA
graph capture and replay (gen_python).

CUDA graph configuration includes:
- Graph ID for tracking multiple graphs
- Warmup runs before capture
- Capture mode (thread_local, relaxed, etc.)
- Stream management
- Memory pool configuration
- Static input optimization

The CUDAGraphAdapter translates these into CUDAGraphSetupNode IR nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Sequence, TYPE_CHECKING

from ..ir import CUDAGraphPhase, CUDAGraphSetupNode


if TYPE_CHECKING:
    pass


class CUDACaptureMode(Enum):
    """
    Capture modes for CUDA graph recording.

    THREAD_LOCAL: Default mode, captures operations on current thread's stream
    RELAXED: Allows cross-stream operations during capture
    GLOBAL: Captures operations from all threads
    """

    THREAD_LOCAL = auto()
    RELAXED = auto()
    GLOBAL = auto()

    @classmethod
    def from_string(cls, s: str) -> "CUDACaptureMode":
        """Convert a string to CUDACaptureMode enum."""
        mapping = {
            "thread_local": cls.THREAD_LOCAL,
            "relaxed": cls.RELAXED,
            "global": cls.GLOBAL,
        }
        return mapping.get(s.lower(), cls.THREAD_LOCAL)

    def to_string(self) -> str:
        """Convert CUDACaptureMode to string for code generation."""
        return self.name.lower()


class CUDAGraphTreeMode(Enum):
    """
    Mode for CUDA graph tree memory management.

    DISABLED: No CUDA graph trees, standard CUDA graph behavior
    ENABLED: Use CUDA graph trees for memory pooling across graphs
    """

    DISABLED = auto()
    ENABLED = auto()


@dataclass
class StaticInputInfo:
    """
    Information about a static input to a CUDA graph.

    Static inputs are tensors whose memory addresses do not change between
    graph replays. This allows for optimization by avoiding input copies.

    Attributes:
        input_index: Index of this input in the function's argument list
        is_static: Whether this input is static (address stable)
        is_parameter: Whether this input is a model parameter
        is_buffer: Whether this input is a model buffer
    """

    input_index: int
    is_static: bool = False
    is_parameter: bool = False
    is_buffer: bool = False


@dataclass
class CUDAGraphInfo:
    """
    Comprehensive information for CUDA graph setup and capture.

    This dataclass captures all the essential configuration for CUDA graph
    capture that is needed to generate the CUDAGraphSetupNode IR.

    For training mode, separate CUDAGraphInfo objects are created for forward
    and backward passes. The forward graph info includes saved_tensor_indices
    and num_forward_outputs to manage the saved_tensors buffer that connects
    forward and backward graphs.

    Attributes:
        enabled: Whether CUDA graphs are enabled for this compilation
        graph_id: Unique identifier for this CUDA graph
        capture_mode: The CUDA graph capture mode to use
        warmup_runs: Number of warmup runs before graph capture
        stream_name: Name of the CUDA stream to use for capture
        pool_id: Memory pool identifier for graph allocation
        static_input_indices: Indices of inputs that are static
        static_inputs_info: Detailed info about each static input
        mutated_input_indices: Indices of inputs that are mutated
        use_cudagraph_trees: Whether CUDA graph trees are enabled
        skip_dynamic_graphs: Whether to skip graphs with dynamic shapes
        force_cudagraph_sync: Whether to synchronize after graph replay
        skip_warmup: Whether to skip the warmup phase
        device_index: CUDA device index for graph capture
        is_backward: Whether this is a backward pass graph (training mode)
        backward_graph_id: For forward graphs, the ID of the paired backward graph
        saved_tensor_indices: Indices of tensors to save for backward pass
        num_forward_outputs: Number of actual forward outputs (excluding saved)
    """

    enabled: bool = False
    graph_id: str = "cuda_graph_0"
    capture_mode: CUDACaptureMode = CUDACaptureMode.THREAD_LOCAL
    warmup_runs: int = 1
    stream_name: str = "default"
    pool_id: Optional[str] = None
    static_input_indices: list[int] = field(default_factory=list)
    static_inputs_info: list[StaticInputInfo] = field(default_factory=list)
    mutated_input_indices: list[int] = field(default_factory=list)
    use_cudagraph_trees: bool = True
    skip_dynamic_graphs: bool = False
    force_cudagraph_sync: bool = False
    skip_warmup: bool = False
    device_index: Optional[int] = None
    is_backward: bool = False
    backward_graph_id: Optional[str] = None
    saved_tensor_indices: list[int] = field(default_factory=list)
    num_forward_outputs: Optional[int] = None


class CUDAGraphAdapter:
    """
    Adapter for translating CUDA graph configuration to CUDAGraphSetupNode IR nodes.

    This adapter takes CUDA graph configuration from torch.compile and Inductor
    and produces the corresponding CUDAGraphSetupNode IR nodes that can be processed
    by the pythonify code generation backends.

    The adapter handles several forms of input:
    1. CUDAGraphInfo objects (intermediate representation)
    2. Configuration dictionaries (from CompilationArtifacts.cuda_graph_config)
    3. Inductor config settings (from torch._inductor.config.triton)

    Usage:
        adapter = CUDAGraphAdapter()

        # From CUDAGraphInfo
        info = CUDAGraphInfo(
            enabled=True,
            graph_id="graph_0",
            warmup_runs=2,
            capture_mode=CUDACaptureMode.THREAD_LOCAL,
        )
        node = adapter.translate_cuda_graph_info(info)

        # From configuration dictionary
        config = {
            "graph_id": "graph_0",
            "warmup_runs": 2,
            "capture_mode": "thread_local",
            "static_inputs": True,
        }
        node = adapter.translate_config_dict(config)
    """

    def __init__(self) -> None:
        """Initialize the CUDAGraphAdapter."""
        pass

    def translate_cuda_graph_info(
        self,
        info: CUDAGraphInfo,
    ) -> Optional[CUDAGraphSetupNode]:
        """
        Translate a CUDAGraphInfo object to a CUDAGraphSetupNode.

        This is the primary entry point for translating CUDA graph configuration
        into the pythonify IR. For training mode, this method determines the
        appropriate CUDAGraphPhase based on the is_backward flag and
        backward_graph_id.

        Args:
            info: CUDAGraphInfo with CUDA graph configuration

        Returns:
            CUDAGraphSetupNode if CUDA graphs are enabled, None otherwise
        """
        if not info.enabled:
            return None

        if info.is_backward:
            phase = CUDAGraphPhase.BACKWARD
        elif info.backward_graph_id is not None:
            phase = CUDAGraphPhase.FORWARD
        else:
            phase = CUDAGraphPhase.INFERENCE

        return CUDAGraphSetupNode(
            graph_id=info.graph_id,
            warmup_runs=info.warmup_runs,
            capture_mode=info.capture_mode.to_string(),
            stream_name=info.stream_name,
            pool_id=info.pool_id,
            static_inputs=len(info.static_input_indices) > 0,
            static_input_indices=list(info.static_input_indices),
            phase=phase,
            backward_graph_id=info.backward_graph_id,
            saved_tensor_indices=list(info.saved_tensor_indices),
            num_forward_outputs=info.num_forward_outputs,
            device_index=info.device_index,
            force_cudagraph_sync=info.force_cudagraph_sync,
        )

    def translate_config_dict(
        self,
        config: dict[str, Any],
        enabled: bool = True,
    ) -> Optional[CUDAGraphSetupNode]:
        """
        Translate a CUDA graph configuration dictionary to a CUDAGraphSetupNode.

        This method is useful for translating CompilationArtifacts.cuda_graph_config
        directly into a CUDAGraphSetupNode.

        Args:
            config: Dictionary with CUDA graph configuration. Expected keys:
                - graph_id: str (optional, default "cuda_graph_0")
                - warmup_runs: int (optional, default 1)
                - capture_mode: str (optional, default "thread_local")
                - stream_name: str (optional, default "default")
                - pool_id: str (optional)
                - static_inputs: bool (optional, default False)
                - static_input_indices: list[int] (optional)
            enabled: Whether CUDA graphs are enabled

        Returns:
            CUDAGraphSetupNode if enabled, None otherwise
        """
        if not enabled:
            return None

        info = self._extract_info_from_dict(config)
        info.enabled = enabled
        return self.translate_cuda_graph_info(info)

    def translate_inductor_config(
        self,
        triton_config: Any,
        graph_id: str = "cuda_graph_0",
        static_input_indices: Optional[Sequence[int]] = None,
    ) -> Optional[CUDAGraphSetupNode]:
        """
        Translate Inductor's triton config to a CUDAGraphSetupNode.

        This method extracts CUDA graph settings from torch._inductor.config.triton
        and produces a CUDAGraphSetupNode.

        Args:
            triton_config: The triton config object (torch._inductor.config.triton)
            graph_id: Unique identifier for this graph
            static_input_indices: Indices of static inputs (if known)

        Returns:
            CUDAGraphSetupNode if CUDA graphs are enabled, None otherwise
        """
        enabled = getattr(triton_config, "cudagraphs", False)
        if not enabled:
            return None

        info = CUDAGraphInfo(
            enabled=True,
            graph_id=graph_id,
            warmup_runs=0 if getattr(triton_config, "skip_cudagraph_warmup", False) else 1,
            capture_mode=CUDACaptureMode.THREAD_LOCAL,
            use_cudagraph_trees=getattr(triton_config, "cudagraph_trees", True),
            skip_dynamic_graphs=getattr(triton_config, "cudagraph_skip_dynamic_graphs", False),
            force_cudagraph_sync=getattr(triton_config, "force_cudagraph_sync", False),
            skip_warmup=getattr(triton_config, "skip_cudagraph_warmup", False),
        )

        if static_input_indices is not None:
            info.static_input_indices = list(static_input_indices)

        return self.translate_cuda_graph_info(info)

    def translate_wrapped_function(
        self,
        wrapped_fn: Any,
        graph_id: str = "cuda_graph_0",
    ) -> Optional[CUDAGraphSetupNode]:
        """
        Translate a WrappedFunction to a CUDAGraphSetupNode.

        WrappedFunction is used in CUDA graph trees to wrap functions
        for graph capture and replay.

        Args:
            wrapped_fn: WrappedFunction object from cudagraph_utils
            graph_id: Unique identifier for this graph

        Returns:
            CUDAGraphSetupNode representing this wrapped function's config
        """
        static_indices = getattr(wrapped_fn, "static_input_idxs", [])
        mutated_indices = getattr(wrapped_fn, "mutated_input_idxs", [])

        info = CUDAGraphInfo(
            enabled=True,
            graph_id=graph_id,
            static_input_indices=list(static_indices),
            mutated_input_indices=list(mutated_indices),
        )

        return self.translate_cuda_graph_info(info)

    def extract_static_inputs_info(
        self,
        static_input_indices: Sequence[int],
        parameter_count: int = 0,
        buffer_count: int = 0,
    ) -> list[StaticInputInfo]:
        """
        Extract detailed StaticInputInfo for each static input.

        This provides more detailed information about why each input
        is considered static, useful for debugging and code generation.

        Args:
            static_input_indices: Indices of static inputs
            parameter_count: Number of model parameters
            buffer_count: Number of model buffers

        Returns:
            List of StaticInputInfo objects
        """
        infos = []
        for idx in static_input_indices:
            is_parameter = idx < parameter_count
            is_buffer = (
                not is_parameter
                and idx < parameter_count + buffer_count
            )

            infos.append(StaticInputInfo(
                input_index=idx,
                is_static=True,
                is_parameter=is_parameter,
                is_buffer=is_buffer,
            ))

        return infos

    def _extract_info_from_dict(
        self,
        config: dict[str, Any],
    ) -> CUDAGraphInfo:
        """Extract CUDAGraphInfo from a configuration dictionary."""
        capture_mode_str = config.get("capture_mode", "thread_local")
        capture_mode = CUDACaptureMode.from_string(capture_mode_str)

        static_input_indices = config.get("static_input_indices", [])
        if config.get("static_inputs", False) and not static_input_indices:
            static_input_indices = []

        return CUDAGraphInfo(
            enabled=True,
            graph_id=config.get("graph_id", "cuda_graph_0"),
            warmup_runs=config.get("warmup_runs", 1),
            capture_mode=capture_mode,
            stream_name=config.get("stream_name", "default"),
            pool_id=config.get("pool_id"),
            static_input_indices=list(static_input_indices),
            mutated_input_indices=list(config.get("mutated_input_indices", [])),
            use_cudagraph_trees=config.get("use_cudagraph_trees", True),
            skip_dynamic_graphs=config.get("skip_dynamic_graphs", False),
            force_cudagraph_sync=config.get("force_cudagraph_sync", False),
            skip_warmup=config.get("skip_warmup", False),
            device_index=config.get("device_index"),
            is_backward=config.get("is_backward", False),
            backward_graph_id=config.get("backward_graph_id"),
            saved_tensor_indices=list(config.get("saved_tensor_indices", [])),
            num_forward_outputs=config.get("num_forward_outputs"),
        )


def translate_cuda_graphs(
    config: dict[str, Any],
    enabled: bool = True,
) -> Optional[CUDAGraphSetupNode]:
    """
    Convenience function to translate CUDA graph configuration to IR node.

    This is the main entry point for translating CUDA graph configuration
    from CompilationArtifacts format to a CUDAGraphSetupNode.

    Args:
        config: Dictionary containing CUDA graph configuration
        enabled: Whether CUDA graphs are enabled

    Returns:
        CUDAGraphSetupNode if enabled, None otherwise

    Example:
        node = translate_cuda_graphs({
            "graph_id": "graph_0",
            "warmup_runs": 2,
            "capture_mode": "thread_local",
            "static_inputs": True,
        })
    """
    adapter = CUDAGraphAdapter()
    return adapter.translate_config_dict(config, enabled=enabled)
