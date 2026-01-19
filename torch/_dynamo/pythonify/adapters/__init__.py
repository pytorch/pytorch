"""
Adapters for translating compilation artifacts to IR nodes.

This module contains adapters that bridge between torch.compile's
internal data structures and the pythonify IR nodes:

- guards.py: Translate Dynamo guards to GuardCheckNodes
- aot_autograd.py: Translate AOT compilation results to IR nodes
- cuda_graphs.py: Translate CUDA graph configuration to IR nodes
- inputs.py: Translate input sources to ArgumentExtractionNodes
- graph_serializer.py: Serialize FX graphs to Python source code
"""

from .aot_autograd import (
    AOTAutogradAdapter,
    AOTAutogradGraphSerializer,
    AOTAutogradInfo,
    AOTCompilationMode,
    InputMutationInfo,
    OutputAliasInfo,
    serialize_aot_autograd_graphs,
    SerializedAOTAutogradInfo,
    translate_aot_autograd,
)
from .cuda_graphs import (
    CUDACaptureMode,
    CUDAGraphAdapter,
    CUDAGraphInfo,
    CUDAGraphTreeMode,
    StaticInputInfo,
    translate_cuda_graphs,
)
from .graph_serializer import (
    extract_graph_metadata,
    get_graph_readable,
    GraphSerializer,
    SerializedGraph,
    serialize_graph_to_code,
)
from .guards import (
    create_dynamic_shape_guard,
    DynamoGuardType,
    GuardAdapter,
    GuardInfo,
    TensorGuardInfo,
    translate_guards,
)
from .inputs import (
    extract_input_info_from_source,
    InputAdapter,
    InputInfo,
    InputSourceType,
    translate_inputs,
)


__all__ = [
    # AOT Autograd adapter
    "AOTAutogradAdapter",
    "AOTAutogradGraphSerializer",
    "AOTAutogradInfo",
    "AOTCompilationMode",
    "InputMutationInfo",
    "OutputAliasInfo",
    "serialize_aot_autograd_graphs",
    "SerializedAOTAutogradInfo",
    "translate_aot_autograd",
    # CUDA graph adapter
    "CUDACaptureMode",
    "CUDAGraphAdapter",
    "CUDAGraphInfo",
    "CUDAGraphTreeMode",
    "StaticInputInfo",
    "translate_cuda_graphs",
    # Graph serializer
    "extract_graph_metadata",
    "get_graph_readable",
    "GraphSerializer",
    "SerializedGraph",
    "serialize_graph_to_code",
    # Guard adapter
    "create_dynamic_shape_guard",
    "DynamoGuardType",
    "GuardAdapter",
    "GuardInfo",
    "TensorGuardInfo",
    "translate_guards",
    # Input adapter
    "extract_input_info_from_source",
    "InputAdapter",
    "InputInfo",
    "InputSourceType",
    "translate_inputs",
]
