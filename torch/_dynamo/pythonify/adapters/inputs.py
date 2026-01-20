"""
InputAdapter for translating Dynamo input sources to ArgumentExtractionNode IR nodes.

This module provides functionality to translate the various source types used
by Dynamo's tracing system into ArgumentExtractionNode IR nodes. The adapter
handles:

- LocalSource: Values from frame locals (function arguments)
- GlobalSource: Values from frame globals
- AttrSource: Attribute access chains (e.g., model.layer1.weight)
- Parameters and Buffers from nn.Module

The InputAdapter extracts information from:
1. GraphArg objects from OutputGraph.graphargs
2. Source objects from torch._dynamo.source
3. CompilationArtifacts dictionaries

The translated IR nodes can then be used by code generation backends to produce
either runtime argument extraction (gen_binary) or explicit Python code that
accesses arguments (gen_python).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, TYPE_CHECKING

from ..ir import ArgumentExtractionNode, ArgumentSource


if TYPE_CHECKING:
    from torch._dynamo.source import Source
    from torch._dynamo.variables.builder import GraphArg


class InputSourceType(Enum):
    """
    Enum representing the different source types that Dynamo can generate.

    These correspond to the Source subclasses in torch/_dynamo/source.py.
    The InputAdapter maps these to the simpler ArgumentSource enum in the IR.
    """

    LOCAL = auto()
    GLOBAL = auto()
    ATTRIBUTE = auto()
    CONSTANT = auto()
    PARAMETER = auto()
    BUFFER = auto()
    SYNTHETIC_LOCAL = auto()
    TEMP_LOCAL = auto()
    CLOSURE = auto()
    RANDOM_VALUE = auto()
    BACKWARD_STATE = auto()
    UNKNOWN = auto()


@dataclass
class InputInfo:
    """
    Intermediate representation of input information extracted from Dynamo.

    This captures the essential information needed to generate an
    ArgumentExtractionNode, extracted from Dynamo's Source/GraphArg objects.

    Attributes:
        source_type: The type of input source
        name: Variable name to assign the extracted value to
        access_path: Path to access the value (e.g., "x" for locals, "W" for attrs)
        nested_path: For nested access, list of attribute names to traverse
        base_name: Name of the base object (e.g., "model" for model.W)
        is_tensor: Whether this input is a tensor
        pass_as_tensor: Whether this Python value is passed as a tensor
        example_value: Example value for type/shape inference
        metadata: Additional source-specific metadata
    """

    source_type: InputSourceType
    name: str
    access_path: str
    nested_path: list[str] = field(default_factory=list)
    base_name: str = ""
    is_tensor: bool = True
    pass_as_tensor: bool = False
    example_value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InputAdapter:
    """
    Adapter for translating Dynamo input sources to ArgumentExtractionNode IR nodes.

    This adapter takes source information from Dynamo's tracing system and
    produces the corresponding ArgumentExtractionNode IR nodes that can be
    processed by the pythonify code generation backends.

    The adapter handles several forms of input:
    1. GraphArg objects (from OutputGraph.graphargs)
    2. Source objects (from torch._dynamo.source)
    3. Simplified dictionaries (from CompilationArtifacts)

    Usage:
        adapter = InputAdapter()

        # From a GraphArg
        nodes = adapter.translate_graph_arg(graph_arg, arg_index=0)

        # From a Source object
        nodes = adapter.translate_source(source, variable_name="arg1")

        # From simplified input info
        info = InputInfo(
            source_type=InputSourceType.LOCAL,
            name="arg1",
            access_path="x",
        )
        node = adapter.translate_input_info(info)

    Example:
        # Extracting arguments from a traced function like:
        # def forward(self, x):
        #     return x + self.W
        #
        # Would produce InputInfo objects like:
        # - InputInfo(LOCAL, "arg1", "x")  # for input x
        # - InputInfo(PARAMETER, "arg2", "W", base_name="self")  # for self.W
    """

    def __init__(self, model_name: str = "model") -> None:
        """
        Initialize the InputAdapter.

        Args:
            model_name: The variable name of the model object in the traced
                function's scope. This is used when generating code that
                accesses model attributes. Defaults to "model".
        """
        self.model_name = model_name

    def translate_graph_arg(
        self,
        graph_arg: "GraphArg",
        arg_index: int,
        arg_name: Optional[str] = None,
    ) -> ArgumentExtractionNode:
        """
        Translate a GraphArg to an ArgumentExtractionNode.

        GraphArg objects are stored on FX graph placeholder nodes and contain
        information about how to reconstruct the value at runtime.

        Args:
            graph_arg: The GraphArg object from OutputGraph
            arg_index: The index of this argument (used for naming if no name given)
            arg_name: Optional explicit name for the argument variable

        Returns:
            ArgumentExtractionNode representing this input
        """
        source = graph_arg.source
        name = arg_name or f"arg{arg_index + 1}"

        if source is None:
            return ArgumentExtractionNode(
                name=name,
                source=ArgumentSource.CONSTANT,
                access_path="",
                nested_path=[],
            )

        return self.translate_source(source, name)

    def translate_source(
        self,
        source: "Source",
        variable_name: str,
    ) -> ArgumentExtractionNode:
        """
        Translate a Dynamo Source object to an ArgumentExtractionNode.

        Source objects track where values come from in the original code and
        are used for guard generation and code reconstruction.

        Args:
            source: The Source object from torch._dynamo.source
            variable_name: Name to assign the extracted value to

        Returns:
            ArgumentExtractionNode representing this source
        """
        source_class_name = type(source).__name__
        source_name = getattr(source, "name", str(source))

        if source_class_name == "LocalSource":
            local_name = getattr(source, "local_name", "")
            return ArgumentExtractionNode(
                name=variable_name,
                source=ArgumentSource.F_LOCALS,
                access_path=local_name,
                nested_path=[],
            )

        elif source_class_name == "GlobalSource":
            global_name = getattr(source, "global_name", "")
            return ArgumentExtractionNode(
                name=variable_name,
                source=ArgumentSource.F_GLOBALS,
                access_path=global_name,
                nested_path=[],
            )

        elif source_class_name in ("AttrSource", "ParamBufferSource"):
            return self._translate_attr_source(source, variable_name)

        elif source_class_name == "SyntheticLocalSource":
            local_name = getattr(source, "local_name", "")
            return ArgumentExtractionNode(
                name=variable_name,
                source=ArgumentSource.F_LOCALS,
                access_path=local_name,
                nested_path=[],
            )

        elif source_class_name == "ConstantSource":
            return ArgumentExtractionNode(
                name=variable_name,
                source=ArgumentSource.CONSTANT,
                access_path=source_name,
                nested_path=[],
            )

        elif source_class_name == "UnspecializedParamBufferSource":
            return self._translate_attr_source(source, variable_name)

        else:
            return self._translate_from_source_name(source_name, variable_name)

    def translate_input_info(self, info: InputInfo) -> ArgumentExtractionNode:
        """
        Translate an InputInfo object to an ArgumentExtractionNode.

        This is useful for translating simplified input representations
        from CompilationArtifacts.

        Args:
            info: InputInfo with extracted input information

        Returns:
            ArgumentExtractionNode representing this input
        """
        argument_source = self._map_input_source_type(info.source_type)

        return ArgumentExtractionNode(
            name=info.name,
            source=argument_source,
            access_path=info.access_path,
            nested_path=info.nested_path if info.nested_path else self._parse_nested_path(info.access_path),
        )

    def translate_input_dict(
        self,
        input_dict: dict[str, Any],
        arg_name: Optional[str] = None,
    ) -> ArgumentExtractionNode:
        """
        Translate an input dictionary to an ArgumentExtractionNode.

        This is the primary entry point for translating inputs from the
        CompilationArtifacts format.

        Args:
            input_dict: Dictionary with input information. Expected keys:
                - source_type: String type ("local", "global", "parameter", etc.)
                - access_path: Path to access the value
                - nested_path: (optional) List of attribute names for nested access
                - base_name: (optional) Name of base object for attributes
            arg_name: Optional explicit name for the argument

        Returns:
            ArgumentExtractionNode representing this input
        """
        source_type_str = input_dict.get("source_type", "local").lower()
        access_path = input_dict.get("access_path", "")
        nested_path = input_dict.get("nested_path", [])
        name = arg_name or input_dict.get("name", "arg")

        source_type = self._map_string_to_input_source_type(source_type_str)
        argument_source = self._map_input_source_type(source_type)

        if not nested_path and access_path:
            nested_path = self._parse_nested_path(access_path)

        return ArgumentExtractionNode(
            name=name,
            source=argument_source,
            access_path=access_path,
            nested_path=nested_path,
        )

    def extract_inputs_from_graph_args(
        self,
        graph_args: list["GraphArg"],
    ) -> list[ArgumentExtractionNode]:
        """
        Extract ArgumentExtractionNodes from a list of GraphArgs.

        This is a convenience method for batch processing all inputs
        from an OutputGraph.

        Args:
            graph_args: List of GraphArg objects

        Returns:
            List of ArgumentExtractionNode objects
        """
        nodes = []
        for idx, graph_arg in enumerate(graph_args):
            node = self.translate_graph_arg(graph_arg, idx)
            nodes.append(node)
        return nodes

    def extract_inputs_from_source_list(
        self,
        sources: list["Source"],
    ) -> list[ArgumentExtractionNode]:
        """
        Extract ArgumentExtractionNodes from a list of Source objects.

        Args:
            sources: List of Source objects

        Returns:
            List of ArgumentExtractionNode objects
        """
        nodes = []
        for idx, source in enumerate(sources):
            name = f"arg{idx + 1}"
            node = self.translate_source(source, name)
            nodes.append(node)
        return nodes

    def _translate_attr_source(
        self,
        source: "Source",
        variable_name: str,
    ) -> ArgumentExtractionNode:
        """
        Translate an attribute source (AttrSource, ParamBufferSource, etc.).

        Attribute sources represent access chains like model.layer1.weight.
        We need to extract the full path and determine if this is a
        parameter, buffer, or general model attribute.
        """
        member = getattr(source, "member", "")
        base = getattr(source, "base", None)

        full_path_parts = [member]
        current = base

        while current is not None:
            current_class = type(current).__name__
            if current_class in ("AttrSource", "ParamBufferSource", "GenericAttrSource"):
                full_path_parts.insert(0, getattr(current, "member", ""))
                current = getattr(current, "base", None)
            elif current_class == "LocalSource":
                break
            elif current_class == "GlobalSource":
                break
            else:
                break

        full_path = ".".join(full_path_parts)

        source_class_name = type(source).__name__
        if source_class_name == "ParamBufferSource":
            argument_source = ArgumentSource.PARAMETER
        else:
            argument_source = ArgumentSource.MODEL_ATTRIBUTE

        return ArgumentExtractionNode(
            name=variable_name,
            source=argument_source,
            access_path=full_path,
            nested_path=full_path_parts,
        )

    def _translate_from_source_name(
        self,
        source_name: str,
        variable_name: str,
    ) -> ArgumentExtractionNode:
        """
        Translate from a source name string when the Source type is unknown.

        Source names follow patterns like:
        - L['x'] for locals
        - G['torch'] for globals
        - L['model'].W for model attributes
        """
        local_match = re.match(r"L\['(\w+)'\](?:\.(.+))?", source_name)
        if local_match:
            local_name = local_match.group(1)
            attr_path = local_match.group(2)

            if attr_path:
                nested_path = attr_path.split(".")
                return ArgumentExtractionNode(
                    name=variable_name,
                    source=ArgumentSource.MODEL_ATTRIBUTE,
                    access_path=attr_path,
                    nested_path=nested_path,
                )
            else:
                return ArgumentExtractionNode(
                    name=variable_name,
                    source=ArgumentSource.F_LOCALS,
                    access_path=local_name,
                    nested_path=[],
                )

        global_match = re.match(r"G\['(\w+)'\](?:\.(.+))?", source_name)
        if global_match:
            global_name = global_match.group(1)
            attr_path = global_match.group(2)

            if attr_path:
                full_path = f"{global_name}.{attr_path}"
                return ArgumentExtractionNode(
                    name=variable_name,
                    source=ArgumentSource.F_GLOBALS,
                    access_path=full_path,
                    nested_path=[global_name] + attr_path.split("."),
                )
            else:
                return ArgumentExtractionNode(
                    name=variable_name,
                    source=ArgumentSource.F_GLOBALS,
                    access_path=global_name,
                    nested_path=[],
                )

        return ArgumentExtractionNode(
            name=variable_name,
            source=ArgumentSource.F_LOCALS,
            access_path=source_name,
            nested_path=[],
        )

    def _parse_nested_path(self, path: str) -> list[str]:
        """Parse a dotted path into a list of attribute names."""
        if "." in path:
            return path.split(".")
        return [path] if path else []

    def _map_input_source_type(self, source_type: InputSourceType) -> ArgumentSource:
        """Map InputSourceType to ArgumentSource enum."""
        mapping = {
            InputSourceType.LOCAL: ArgumentSource.F_LOCALS,
            InputSourceType.GLOBAL: ArgumentSource.F_GLOBALS,
            InputSourceType.ATTRIBUTE: ArgumentSource.MODEL_ATTRIBUTE,
            InputSourceType.CONSTANT: ArgumentSource.CONSTANT,
            InputSourceType.PARAMETER: ArgumentSource.PARAMETER,
            InputSourceType.BUFFER: ArgumentSource.BUFFER,
            InputSourceType.SYNTHETIC_LOCAL: ArgumentSource.F_LOCALS,
            InputSourceType.TEMP_LOCAL: ArgumentSource.F_LOCALS,
            InputSourceType.CLOSURE: ArgumentSource.F_LOCALS,
            InputSourceType.RANDOM_VALUE: ArgumentSource.CONSTANT,
            InputSourceType.BACKWARD_STATE: ArgumentSource.CONSTANT,
            InputSourceType.UNKNOWN: ArgumentSource.F_LOCALS,
        }
        return mapping.get(source_type, ArgumentSource.F_LOCALS)

    def _map_string_to_input_source_type(self, type_str: str) -> InputSourceType:
        """Map a string to InputSourceType enum."""
        mapping = {
            "local": InputSourceType.LOCAL,
            "global": InputSourceType.GLOBAL,
            "attribute": InputSourceType.ATTRIBUTE,
            "attr": InputSourceType.ATTRIBUTE,
            "constant": InputSourceType.CONSTANT,
            "parameter": InputSourceType.PARAMETER,
            "param": InputSourceType.PARAMETER,
            "buffer": InputSourceType.BUFFER,
            "synthetic_local": InputSourceType.SYNTHETIC_LOCAL,
            "temp_local": InputSourceType.TEMP_LOCAL,
            "closure": InputSourceType.CLOSURE,
            "random_value": InputSourceType.RANDOM_VALUE,
            "backward_state": InputSourceType.BACKWARD_STATE,
        }
        return mapping.get(type_str.lower(), InputSourceType.UNKNOWN)


def translate_inputs(
    inputs: list[dict[str, Any]],
    model_name: str = "model",
) -> list[ArgumentExtractionNode]:
    """
    Convenience function to translate a list of input dictionaries.

    This is the main entry point for translating inputs from
    CompilationArtifacts to ArgumentExtractionNode IR nodes.

    Args:
        inputs: List of input dictionaries
        model_name: Name of the model variable

    Returns:
        List of ArgumentExtractionNode objects
    """
    adapter = InputAdapter(model_name=model_name)
    nodes = []
    for idx, input_dict in enumerate(inputs):
        name = input_dict.get("name", f"arg{idx + 1}")
        node = adapter.translate_input_dict(input_dict, arg_name=name)
        nodes.append(node)
    return nodes


def extract_input_info_from_source(source: "Source") -> InputInfo:
    """
    Extract InputInfo from a Dynamo Source object.

    This function analyzes a Source object and extracts all relevant
    information into an InputInfo intermediate representation.

    Args:
        source: A Dynamo Source object

    Returns:
        InputInfo with extracted information
    """
    source_class = type(source).__name__
    source_name = getattr(source, "name", str(source))

    if source_class == "LocalSource":
        return InputInfo(
            source_type=InputSourceType.LOCAL,
            name="",
            access_path=getattr(source, "local_name", ""),
            metadata={"is_input": getattr(source, "is_input", False)},
        )

    elif source_class == "GlobalSource":
        return InputInfo(
            source_type=InputSourceType.GLOBAL,
            name="",
            access_path=getattr(source, "global_name", ""),
        )

    elif source_class in ("AttrSource", "ParamBufferSource", "GenericAttrSource"):
        member = getattr(source, "member", "")
        base = getattr(source, "base", None)

        full_path_parts = [member]
        current = base
        base_name = ""

        while current is not None:
            current_class = type(current).__name__
            if current_class in ("AttrSource", "ParamBufferSource", "GenericAttrSource"):
                full_path_parts.insert(0, getattr(current, "member", ""))
                current = getattr(current, "base", None)
            elif current_class == "LocalSource":
                base_name = getattr(current, "local_name", "")
                break
            elif current_class == "GlobalSource":
                base_name = getattr(current, "global_name", "")
                break
            else:
                break

        full_path = ".".join(full_path_parts)
        source_type = (
            InputSourceType.PARAMETER
            if source_class == "ParamBufferSource"
            else InputSourceType.ATTRIBUTE
        )

        return InputInfo(
            source_type=source_type,
            name="",
            access_path=full_path,
            nested_path=full_path_parts,
            base_name=base_name,
        )

    elif source_class == "ConstantSource":
        return InputInfo(
            source_type=InputSourceType.CONSTANT,
            name="",
            access_path=source_name,
        )

    elif source_class == "SyntheticLocalSource":
        return InputInfo(
            source_type=InputSourceType.SYNTHETIC_LOCAL,
            name="",
            access_path=getattr(source, "local_name", ""),
        )

    else:
        return InputInfo(
            source_type=InputSourceType.UNKNOWN,
            name="",
            access_path=source_name,
        )
