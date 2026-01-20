"""
AOTAutogradAdapter for translating AOT Autograd compilation results to IR nodes.

This module provides functionality to translate AOT Autograd compilation artifacts
into the structured AOTAutogradWrapperNode IR representation. The translated IR
nodes can then be used by code generation backends to produce either runtime
autograd wrappers (gen_binary) or explicit Python code defining torch.autograd.Function
subclasses (gen_python).

AOT Autograd compilation produces several key artifacts:
- Forward and backward compiled FX graphs
- ViewAndMutationMeta with input/output aliasing and mutation info
- Saved tensors indices for the backward pass
- Subclass metadata for tensor wrapping/unwrapping

The AOTAutogradAdapter translates these into the simpler AOTAutogradWrapperNode IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, TYPE_CHECKING

from ..ir import AOTAutogradWrapperNode


if TYPE_CHECKING:
    from torch.fx import GraphModule


class AOTCompilationMode(Enum):
    """
    Compilation mode of the AOT Autograd compilation.

    INFERENCE: No backward pass compiled (inference-only mode)
    TRAINING: Both forward and backward passes compiled
    LAZY_BACKWARD: Forward compiled, backward will be compiled lazily
    """

    INFERENCE = auto()
    TRAINING = auto()
    LAZY_BACKWARD = auto()


@dataclass
class InputMutationInfo:
    """
    Information about mutations to a single input.

    Attributes:
        input_index: Index of the input being described
        mutates_data: Whether the input's data is mutated
        mutates_metadata: Whether the input's metadata is mutated
        mutation_hidden_from_autograd: Whether mutation is hidden from autograd
        requires_grad: Whether the input requires gradients
        mutation_type: String describing the mutation type
    """

    input_index: int
    mutates_data: bool = False
    mutates_metadata: bool = False
    mutation_hidden_from_autograd: bool = False
    requires_grad: bool = False
    mutation_type: str = "NOT_MUTATED"


@dataclass
class OutputAliasInfo:
    """
    Information about aliasing for a single output.

    Attributes:
        output_index: Index of the output being described
        output_type: Type of output (non_alias, alias_of_input, etc.)
        base_index: Index of the base tensor if aliased (None otherwise)
        requires_grad: Whether the output requires gradients
    """

    output_index: int
    output_type: str = "non_alias"
    base_index: Optional[int] = None
    requires_grad: bool = False


@dataclass
class AOTAutogradInfo:
    """
    Comprehensive information extracted from AOT Autograd compilation.

    This dataclass captures all the essential information from AOT Autograd's
    compilation artifacts that is needed to generate the autograd wrapper.

    Attributes:
        compilation_mode: Whether this is inference, training, or lazy backward
        forward_graph: The compiled forward FX graph (may be None if not available)
        backward_graph: The compiled backward FX graph (None for inference)
        num_inputs: Number of inputs to the forward function
        num_outputs: Number of outputs from the forward function
        num_mutated_inputs: Number of inputs that are mutated
        saved_tensors_indices: Indices of tensors saved for backward pass
        saved_symints_count: Number of symbolic integers saved for backward
        input_mutations: Information about input mutations
        output_aliases: Information about output aliasing
        grad_enabled_mutation: Whether grad mode is mutated during forward
        requires_subclass_dispatch: Whether subclass dispatch is needed
        intermediate_bases_count: Number of intermediate bases in outputs
        forward_graph_str: String representation of forward graph (for logging)
        backward_graph_str: String representation of backward graph (for logging)
        joint_graph_str: String representation of joint graph (for logging)
    """

    compilation_mode: AOTCompilationMode = AOTCompilationMode.INFERENCE
    forward_graph: Optional["GraphModule"] = None
    backward_graph: Optional["GraphModule"] = None
    num_inputs: int = 0
    num_outputs: int = 0
    num_mutated_inputs: int = 0
    saved_tensors_indices: list[int] = field(default_factory=list)
    saved_symints_count: int = 0
    input_mutations: list[InputMutationInfo] = field(default_factory=list)
    output_aliases: list[OutputAliasInfo] = field(default_factory=list)
    grad_enabled_mutation: Optional[bool] = None
    requires_subclass_dispatch: bool = False
    intermediate_bases_count: int = 0
    forward_graph_str: Optional[str] = None
    backward_graph_str: Optional[str] = None
    joint_graph_str: Optional[str] = None


class AOTAutogradAdapter:
    """
    Adapter for translating AOT Autograd compilation results to IR nodes.

    This adapter takes AOT Autograd compilation artifacts and produces
    AOTAutogradWrapperNode IR nodes that can be processed by the pythonify
    code generation backends.

    The adapter handles several forms of input:
    1. AOTAutogradInfo objects (intermediate representation)
    2. ViewAndMutationMeta objects (from AOT Autograd's schemas)
    3. GenericAOTAutogradResult objects (full compilation result)
    4. Dictionary-based artifacts (from CompilationArtifacts)

    Usage:
        adapter = AOTAutogradAdapter()

        # From AOTAutogradInfo
        info = AOTAutogradInfo(
            compilation_mode=AOTCompilationMode.TRAINING,
            forward_graph=fw_graph,
            backward_graph=bw_graph,
            num_inputs=2,
            num_outputs=1,
            saved_tensors_indices=[0, 1],
        )
        node = adapter.translate_aot_info(info)

        # From CompilationArtifacts dictionary format
        artifacts_dict = {
            "forward_graph": fw_graph,
            "backward_graph": bw_graph,
            "num_inputs": 2,
            "num_outputs": 1,
            "saved_tensors_indices": [0, 1],
        }
        node = adapter.translate_artifacts_dict(artifacts_dict)
    """

    def __init__(self, class_name: str = "CompiledFunction") -> None:
        """
        Initialize the AOTAutogradAdapter.

        Args:
            class_name: Default class name for generated autograd.Function
        """
        self.class_name = class_name

    def translate_aot_info(
        self,
        info: AOTAutogradInfo,
        class_name: Optional[str] = None,
    ) -> AOTAutogradWrapperNode:
        """
        Translate an AOTAutogradInfo object to an AOTAutogradWrapperNode.

        This is the primary entry point for translating AOT Autograd compilation
        information into the pythonify IR.

        Args:
            info: AOTAutogradInfo with compilation information
            class_name: Optional class name override for the generated
                autograd.Function (defaults to self.class_name)

        Returns:
            AOTAutogradWrapperNode representing the autograd wrapper
        """
        name = class_name or self.class_name

        metadata = self._build_metadata(info)

        return AOTAutogradWrapperNode(
            class_name=name,
            forward_graph=info.forward_graph,
            backward_graph=info.backward_graph,
            saved_tensors_indices=info.saved_tensors_indices.copy(),
            num_inputs=info.num_inputs,
            num_outputs=info.num_outputs,
            metadata=metadata,
        )

    def translate_artifacts_dict(
        self,
        artifacts_dict: dict[str, Any],
        class_name: Optional[str] = None,
    ) -> AOTAutogradWrapperNode:
        """
        Translate a dictionary of AOT Autograd artifacts to an IR node.

        This method is useful for translating CompilationArtifacts fields
        directly into an AOTAutogradWrapperNode.

        Args:
            artifacts_dict: Dictionary with AOT Autograd artifact fields.
                Expected keys:
                - forward_graph: Optional[GraphModule]
                - backward_graph: Optional[GraphModule]
                - num_inputs: int
                - num_outputs: int
                - saved_tensors_indices: list[int]
                - metadata: dict[str, Any] (optional)

        Returns:
            AOTAutogradWrapperNode representing the autograd wrapper
        """
        info = self._extract_info_from_dict(artifacts_dict)
        return self.translate_aot_info(info, class_name=class_name)

    def translate_view_and_mutation_meta(
        self,
        meta: Any,
        forward_graph: Optional["GraphModule"] = None,
        backward_graph: Optional["GraphModule"] = None,
        class_name: Optional[str] = None,
    ) -> AOTAutogradWrapperNode:
        """
        Translate a ViewAndMutationMeta object to an AOTAutogradWrapperNode.

        ViewAndMutationMeta is the core metadata class from AOT Autograd that
        describes input mutations, output aliasing, and other critical info.

        Args:
            meta: ViewAndMutationMeta object from AOT Autograd
            forward_graph: Optional compiled forward graph
            backward_graph: Optional compiled backward graph
            class_name: Optional class name for generated autograd.Function

        Returns:
            AOTAutogradWrapperNode representing the autograd wrapper
        """
        info = self._extract_info_from_meta(meta, forward_graph, backward_graph)
        return self.translate_aot_info(info, class_name=class_name)

    def translate_aot_result(
        self,
        result: Any,
        class_name: Optional[str] = None,
    ) -> AOTAutogradWrapperNode:
        """
        Translate a GenericAOTAutogradResult to an AOTAutogradWrapperNode.

        GenericAOTAutogradResult is the main result class from AOT Autograd
        compilation containing forward, backward, and all metadata.

        Args:
            result: GenericAOTAutogradResult object
            class_name: Optional class name for generated autograd.Function

        Returns:
            AOTAutogradWrapperNode representing the autograd wrapper
        """
        info = self._extract_info_from_result(result)
        return self.translate_aot_info(info, class_name=class_name)

    def _build_metadata(self, info: AOTAutogradInfo) -> dict[str, Any]:
        """Build metadata dictionary from AOTAutogradInfo."""
        metadata: dict[str, Any] = {
            "compilation_mode": info.compilation_mode.name,
            "num_mutated_inputs": info.num_mutated_inputs,
            "saved_symints_count": info.saved_symints_count,
            "grad_enabled_mutation": info.grad_enabled_mutation,
            "requires_subclass_dispatch": info.requires_subclass_dispatch,
            "intermediate_bases_count": info.intermediate_bases_count,
        }

        if info.input_mutations:
            metadata["input_mutations"] = [
                {
                    "input_index": m.input_index,
                    "mutates_data": m.mutates_data,
                    "mutates_metadata": m.mutates_metadata,
                    "mutation_hidden_from_autograd": m.mutation_hidden_from_autograd,
                    "requires_grad": m.requires_grad,
                    "mutation_type": m.mutation_type,
                }
                for m in info.input_mutations
            ]

        if info.output_aliases:
            metadata["output_aliases"] = [
                {
                    "output_index": a.output_index,
                    "output_type": a.output_type,
                    "base_index": a.base_index,
                    "requires_grad": a.requires_grad,
                }
                for a in info.output_aliases
            ]

        if info.forward_graph_str:
            metadata["forward_graph_str"] = info.forward_graph_str
        if info.backward_graph_str:
            metadata["backward_graph_str"] = info.backward_graph_str
        if info.joint_graph_str:
            metadata["joint_graph_str"] = info.joint_graph_str

        return metadata

    def _extract_info_from_dict(
        self,
        artifacts_dict: dict[str, Any],
    ) -> AOTAutogradInfo:
        """Extract AOTAutogradInfo from a dictionary of artifacts."""
        forward_graph = artifacts_dict.get("forward_graph")
        backward_graph = artifacts_dict.get("backward_graph")

        if backward_graph is not None:
            compilation_mode = AOTCompilationMode.TRAINING
        else:
            compilation_mode = AOTCompilationMode.INFERENCE

        saved_tensors_indices = artifacts_dict.get("saved_tensors_indices", [])
        if saved_tensors_indices is None:
            saved_tensors_indices = []

        metadata = artifacts_dict.get("metadata", {})

        return AOTAutogradInfo(
            compilation_mode=compilation_mode,
            forward_graph=forward_graph,
            backward_graph=backward_graph,
            num_inputs=artifacts_dict.get("num_inputs", 0),
            num_outputs=artifacts_dict.get("num_outputs", 0),
            saved_tensors_indices=list(saved_tensors_indices),
            saved_symints_count=metadata.get("num_symints_saved_for_bw", 0) or 0,
            grad_enabled_mutation=metadata.get("grad_enabled_mutation"),
            requires_subclass_dispatch=metadata.get("requires_subclass_dispatch", False),
            intermediate_bases_count=metadata.get("num_intermediate_bases", 0) or 0,
        )

    def _extract_info_from_meta(
        self,
        meta: Any,
        forward_graph: Optional["GraphModule"],
        backward_graph: Optional["GraphModule"],
    ) -> AOTAutogradInfo:
        """
        Extract AOTAutogradInfo from a ViewAndMutationMeta object.

        ViewAndMutationMeta contains:
        - input_info: list[InputAliasInfo] with mutation info per input
        - output_info: list[OutputAliasInfo] with aliasing info per output
        - num_intermediate_bases: int
        - traced_tangents: list of tangent proxies for backward
        - num_symints_saved_for_bw: Optional[int]
        - grad_enabled_mutation: Optional[bool]
        - And various pre-computed indices...
        """
        input_mutations = []
        if hasattr(meta, "input_info"):
            for i, inp_info in enumerate(meta.input_info):
                mutation_type = "NOT_MUTATED"
                if hasattr(inp_info, "mutation_type"):
                    mutation_type = inp_info.mutation_type.name

                input_mutations.append(
                    InputMutationInfo(
                        input_index=i,
                        mutates_data=getattr(inp_info, "mutates_data", False),
                        mutates_metadata=getattr(inp_info, "mutates_metadata", False),
                        mutation_hidden_from_autograd=getattr(
                            inp_info, "mutations_hidden_from_autograd", False
                        ),
                        requires_grad=getattr(inp_info, "requires_grad", False),
                        mutation_type=mutation_type,
                    )
                )

        output_aliases = []
        if hasattr(meta, "output_info"):
            for i, out_info in enumerate(meta.output_info):
                output_type = "non_alias"
                if hasattr(out_info, "output_type"):
                    output_type = out_info.output_type.name

                output_aliases.append(
                    OutputAliasInfo(
                        output_index=i,
                        output_type=output_type,
                        base_index=getattr(out_info, "base_idx", None),
                        requires_grad=getattr(out_info, "requires_grad", False),
                    )
                )

        num_inputs = len(meta.input_info) if hasattr(meta, "input_info") else 0
        num_outputs = len(meta.output_info) if hasattr(meta, "output_info") else 0
        num_mutated = getattr(meta, "num_mutated_inp_runtime_indices", 0)
        saved_symints = getattr(meta, "num_symints_saved_for_bw", 0) or 0
        intermediate_bases = getattr(meta, "num_intermediate_bases", 0)
        grad_mutation = getattr(meta, "grad_enabled_mutation", None)

        if backward_graph is not None:
            compilation_mode = AOTCompilationMode.TRAINING
        else:
            compilation_mode = AOTCompilationMode.INFERENCE

        subclass_meta_present = (
            hasattr(meta, "subclass_inp_meta") and meta.subclass_inp_meta
        )

        return AOTAutogradInfo(
            compilation_mode=compilation_mode,
            forward_graph=forward_graph,
            backward_graph=backward_graph,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_mutated_inputs=num_mutated,
            input_mutations=input_mutations,
            output_aliases=output_aliases,
            saved_symints_count=saved_symints,
            grad_enabled_mutation=grad_mutation,
            requires_subclass_dispatch=subclass_meta_present,
            intermediate_bases_count=intermediate_bases,
        )

    def _extract_info_from_result(self, result: Any) -> AOTAutogradInfo:
        """
        Extract AOTAutogradInfo from a GenericAOTAutogradResult object.

        GenericAOTAutogradResult contains:
        - compiled_fw: TForward (InductorOutput)
        - compiled_bw: Optional[TBackward]
        - runtime_metadata: ViewAndMutationMeta
        - aot_forward_graph_str: Optional[str]
        - aot_backward_graph_str: Optional[str]
        - aot_joint_graph_str: Optional[str]
        - maybe_subclass_meta: Optional[SubclassMeta]
        - num_fw_outs_saved_for_bw: Optional[int]
        - indices_of_inps_to_detach: list[int]
        """
        meta = getattr(result, "runtime_metadata", None)

        if meta is not None:
            info = self._extract_info_from_meta(
                meta,
                forward_graph=None,
                backward_graph=None,
            )
        else:
            if result.compiled_bw is not None:
                compilation_mode = AOTCompilationMode.TRAINING
            else:
                compilation_mode = AOTCompilationMode.INFERENCE

            info = AOTAutogradInfo(
                compilation_mode=compilation_mode,
            )

        info.forward_graph_str = getattr(result, "aot_forward_graph_str", None)
        info.backward_graph_str = getattr(result, "aot_backward_graph_str", None)
        info.joint_graph_str = getattr(result, "aot_joint_graph_str", None)

        subclass_meta = getattr(result, "maybe_subclass_meta", None)
        info.requires_subclass_dispatch = subclass_meta is not None

        return info

    def extract_saved_tensors_indices(
        self,
        meta: Any,
    ) -> list[int]:
        """
        Extract indices of tensors saved for backward from ViewAndMutationMeta.

        In AOT Autograd, saved tensors are output from the forward graph and
        consumed by the backward graph. This method helps identify which
        forward outputs correspond to saved tensors.

        Args:
            meta: ViewAndMutationMeta object

        Returns:
            List of indices of saved tensors in forward graph outputs
        """
        num_outputs = len(meta.output_info) if hasattr(meta, "output_info") else 0
        intermediate_bases = getattr(meta, "num_intermediate_bases", 0)
        num_mutated = getattr(meta, "num_mutated_inp_runtime_indices", 0)

        base_saved_idx = num_mutated + num_outputs + intermediate_bases

        num_saved = 0
        if hasattr(meta, "traced_tangents"):
            num_saved = len(meta.traced_tangents)

        return list(range(base_saved_idx, base_saved_idx + num_saved))


def translate_aot_autograd(
    artifacts_dict: dict[str, Any],
    class_name: str = "CompiledFunction",
) -> AOTAutogradWrapperNode:
    """
    Convenience function to translate AOT Autograd artifacts to IR node.

    This is the main entry point for translating AOT Autograd compilation
    artifacts from CompilationArtifacts format to an AOTAutogradWrapperNode.

    Args:
        artifacts_dict: Dictionary containing AOT Autograd artifact fields
        class_name: Name for the generated autograd.Function class

    Returns:
        AOTAutogradWrapperNode representing the autograd wrapper

    Example:
        node = translate_aot_autograd({
            "forward_graph": fw_graph,
            "backward_graph": bw_graph,
            "num_inputs": 2,
            "num_outputs": 1,
            "saved_tensors_indices": [0, 1],
        })
    """
    adapter = AOTAutogradAdapter(class_name=class_name)
    return adapter.translate_artifacts_dict(artifacts_dict)


@dataclass
class SerializedAOTAutogradInfo:
    """
    Serialized representation of AOT Autograd compilation results.

    This dataclass contains serialized (string) representations of the
    forward and backward graphs that can be embedded in generated Python
    code and executed to reproduce the original computation.

    Attributes:
        forward_graph_code: Python source code for the forward function
        backward_graph_code: Python source code for the backward function (if any)
        forward_graph_readable: Human-readable graph representation for docs
        backward_graph_readable: Human-readable backward graph representation
        forward_input_names: Names of inputs to the forward graph
        forward_output_names: Names of outputs from the forward graph
        backward_input_names: Names of inputs to the backward graph
        backward_output_names: Names of outputs from the backward graph
        saved_tensors_indices: Indices of tensors saved for backward
        num_inputs: Number of inputs to the compiled function
        num_outputs: Number of outputs from the compiled function
        compilation_mode: The compilation mode (INFERENCE, TRAINING, etc.)
        metadata: Additional metadata from compilation
    """

    forward_graph_code: str = ""
    backward_graph_code: Optional[str] = None
    forward_graph_readable: Optional[str] = None
    backward_graph_readable: Optional[str] = None
    forward_input_names: list[str] = field(default_factory=list)
    forward_output_names: list[str] = field(default_factory=list)
    backward_input_names: list[str] = field(default_factory=list)
    backward_output_names: list[str] = field(default_factory=list)
    saved_tensors_indices: list[int] = field(default_factory=list)
    num_inputs: int = 0
    num_outputs: int = 0
    compilation_mode: str = "INFERENCE"
    metadata: dict[str, Any] = field(default_factory=dict)


class AOTAutogradGraphSerializer:
    """
    Serializer for AOT Autograd forward and backward graphs.

    This class takes AOT Autograd compilation results and serializes the
    forward and backward FX graphs into Python source code that can be
    embedded in pythonify output files.

    The serialization process:
    1. Extracts the forward and backward GraphModules
    2. Serializes each graph to executable Python code
    3. Captures metadata about saved tensors and graph structure
    4. Returns a SerializedAOTAutogradInfo with all the code

    Usage:
        serializer = AOTAutogradGraphSerializer()

        # From AOTAutogradInfo
        info = AOTAutogradInfo(forward_graph=fw_gm, backward_graph=bw_gm)
        serialized = serializer.serialize_from_info(info)

        # From GenericAOTAutogradResult
        result = compiler_output
        serialized = serializer.serialize_from_result(result)

        print(serialized.forward_graph_code)
    """

    def __init__(self, include_readable: bool = True) -> None:
        """
        Initialize the serializer.

        Args:
            include_readable: Whether to include human-readable graph representations
        """
        self._include_readable = include_readable

    def serialize_from_info(
        self,
        info: AOTAutogradInfo,
    ) -> SerializedAOTAutogradInfo:
        """
        Serialize AOT Autograd graphs from an AOTAutogradInfo object.

        Args:
            info: The AOTAutogradInfo containing graphs to serialize

        Returns:
            SerializedAOTAutogradInfo with serialized graph code
        """
        from .graph_serializer import (
            extract_graph_metadata,
            get_graph_readable,
            GraphSerializer,
        )

        serializer = GraphSerializer(include_readable=self._include_readable)

        forward_code = ""
        forward_readable = None
        forward_input_names: list[str] = []
        forward_output_names: list[str] = []

        if info.forward_graph is not None:
            serialized_fw = serializer.serialize(info.forward_graph, "compiled_forward")
            forward_code = serialized_fw.graph_code
            forward_readable = serialized_fw.graph_readable
            forward_input_names = serialized_fw.input_names
            forward_output_names = serialized_fw.output_names
        elif info.forward_graph_str:
            forward_code = f"# Forward graph (from print_readable):\n# {info.forward_graph_str[:500]}"
            forward_readable = info.forward_graph_str

        backward_code = None
        backward_readable = None
        backward_input_names: list[str] = []
        backward_output_names: list[str] = []

        if info.backward_graph is not None:
            serialized_bw = serializer.serialize(
                info.backward_graph, "compiled_backward"
            )
            backward_code = serialized_bw.graph_code
            backward_readable = serialized_bw.graph_readable
            backward_input_names = serialized_bw.input_names
            backward_output_names = serialized_bw.output_names
        elif info.backward_graph_str:
            backward_code = f"# Backward graph (from print_readable):\n# {info.backward_graph_str[:500]}"
            backward_readable = info.backward_graph_str

        metadata: dict[str, Any] = {
            "num_mutated_inputs": info.num_mutated_inputs,
            "saved_symints_count": info.saved_symints_count,
            "grad_enabled_mutation": info.grad_enabled_mutation,
            "requires_subclass_dispatch": info.requires_subclass_dispatch,
            "intermediate_bases_count": info.intermediate_bases_count,
        }

        if info.input_mutations:
            metadata["input_mutations"] = [
                {
                    "input_index": m.input_index,
                    "mutates_data": m.mutates_data,
                    "mutates_metadata": m.mutates_metadata,
                    "requires_grad": m.requires_grad,
                    "mutation_type": m.mutation_type,
                }
                for m in info.input_mutations
            ]

        if info.output_aliases:
            metadata["output_aliases"] = [
                {
                    "output_index": a.output_index,
                    "output_type": a.output_type,
                    "base_index": a.base_index,
                    "requires_grad": a.requires_grad,
                }
                for a in info.output_aliases
            ]

        return SerializedAOTAutogradInfo(
            forward_graph_code=forward_code,
            backward_graph_code=backward_code,
            forward_graph_readable=forward_readable,
            backward_graph_readable=backward_readable,
            forward_input_names=forward_input_names,
            forward_output_names=forward_output_names,
            backward_input_names=backward_input_names,
            backward_output_names=backward_output_names,
            saved_tensors_indices=info.saved_tensors_indices.copy(),
            num_inputs=info.num_inputs,
            num_outputs=info.num_outputs,
            compilation_mode=info.compilation_mode.name,
            metadata=metadata,
        )

    def serialize_from_result(
        self,
        result: Any,
    ) -> SerializedAOTAutogradInfo:
        """
        Serialize AOT Autograd graphs from a GenericAOTAutogradResult.

        This method extracts and serializes graphs from the AOT Autograd
        compilation result object directly.

        Args:
            result: GenericAOTAutogradResult from AOT Autograd compilation

        Returns:
            SerializedAOTAutogradInfo with serialized graph code
        """
        adapter = AOTAutogradAdapter()
        info = adapter._extract_info_from_result(result)

        info.forward_graph_str = getattr(result, "aot_forward_graph_str", None)
        info.backward_graph_str = getattr(result, "aot_backward_graph_str", None)
        info.joint_graph_str = getattr(result, "aot_joint_graph_str", None)

        return self.serialize_from_info(info)

    def serialize_from_artifacts_dict(
        self,
        artifacts_dict: dict[str, Any],
    ) -> SerializedAOTAutogradInfo:
        """
        Serialize AOT Autograd graphs from a CompilationArtifacts dictionary.

        This method is used when processing CompilationArtifacts during
        pythonify code generation.

        Args:
            artifacts_dict: Dictionary with forward_graph, backward_graph, etc.

        Returns:
            SerializedAOTAutogradInfo with serialized graph code
        """
        adapter = AOTAutogradAdapter()
        info = adapter._extract_info_from_dict(artifacts_dict)
        return self.serialize_from_info(info)


def serialize_aot_autograd_graphs(
    info_or_result: Any,
    include_readable: bool = True,
) -> SerializedAOTAutogradInfo:
    """
    Convenience function to serialize AOT Autograd graphs.

    Automatically detects the input type and uses the appropriate
    serialization method.

    Args:
        info_or_result: Either AOTAutogradInfo, GenericAOTAutogradResult,
            or a dictionary with graph information
        include_readable: Whether to include human-readable representations

    Returns:
        SerializedAOTAutogradInfo with serialized graph code

    Example:
        # From AOTAutogradInfo
        info = AOTAutogradInfo(forward_graph=fw_gm)
        serialized = serialize_aot_autograd_graphs(info)

        # From a dictionary
        serialized = serialize_aot_autograd_graphs({
            "forward_graph": fw_gm,
            "backward_graph": bw_gm,
        })
    """
    serializer = AOTAutogradGraphSerializer(include_readable=include_readable)

    if isinstance(info_or_result, AOTAutogradInfo):
        return serializer.serialize_from_info(info_or_result)
    elif isinstance(info_or_result, dict):
        return serializer.serialize_from_artifacts_dict(info_or_result)
    else:
        return serializer.serialize_from_result(info_or_result)
