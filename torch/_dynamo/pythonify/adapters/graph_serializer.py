"""
FX GraphModule serialization for pythonify.

This module provides utilities to serialize FX GraphModules into executable
Python source code that can be included in pythonify output files. The
serialization produces code that reconstructs the graph and can be executed
to perform the same computation as the original graph.

The serialization has two main modes:
1. print_readable: Uses FX's built-in print_readable() for human-readable code
2. GraphSerializer: Custom serialization that produces more compact, executable code

The serialized code can be included in the generated pythonify output to
allow the forward and backward passes to be run without the original
compiled artifacts.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.fx import GraphModule


@dataclass
class SerializedGraph:
    """
    A serialized representation of an FX GraphModule.

    This dataclass contains all the information needed to reconstruct
    and execute an FX graph from the pythonify output.

    Attributes:
        graph_code: Python source code that defines the graph as a function
        function_name: Name of the generated function
        input_names: Names of the inputs to the graph (in order)
        output_names: Names of the outputs from the graph
        graph_readable: Human-readable print_readable() output for documentation
        constants: Dictionary of constant values referenced by the graph
        metadata: Additional metadata about the graph (e.g., original op count)
    """

    graph_code: str
    function_name: str = "forward"
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    graph_readable: Optional[str] = None
    constants: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphSerializer:
    """
    Serializer for FX GraphModules to executable Python code.

    This class provides methods to convert FX GraphModules into Python source
    code that can be embedded in pythonify output files. The generated code
    can be executed to reproduce the original graph's computation.

    The serialization preserves:
    - Node operations and their arguments
    - Input/output structure
    - Constant values and tensor shapes
    - Control flow and function calls

    Usage:
        serializer = GraphSerializer()
        serialized = serializer.serialize(graph_module)
        print(serialized.graph_code)
    """

    def __init__(
        self,
        include_readable: bool = True,
        include_constants: bool = True,
        indent: str = "    ",
    ) -> None:
        """
        Initialize the GraphSerializer.

        Args:
            include_readable: Whether to include human-readable graph repr
            include_constants: Whether to extract and include constants
            indent: Indentation string to use in generated code
        """
        self._include_readable = include_readable
        self._include_constants = include_constants
        self._indent = indent

    def serialize(
        self,
        graph_module: "GraphModule",
        function_name: str = "forward",
    ) -> SerializedGraph:
        """
        Serialize an FX GraphModule to executable Python code.

        This method converts the graph into Python source code that can be
        executed standalone. The generated code includes:
        1. A function definition with proper input signature
        2. All operations from the graph as Python statements
        3. The return statement with proper output structure

        Args:
            graph_module: The FX GraphModule to serialize
            function_name: Name for the generated function

        Returns:
            SerializedGraph containing the generated code and metadata
        """
        if graph_module is None:
            return SerializedGraph(
                graph_code="# No graph provided",
                function_name=function_name,
                metadata={"error": "No graph provided"},
            )

        graph = graph_module.graph
        input_names: list[str] = []
        output_names: list[str] = []
        constants: dict[str, Any] = {}

        lines: list[str] = []

        placeholder_count = 0
        for node in graph.nodes:
            if node.op == "placeholder":
                input_names.append(node.name)
                placeholder_count += 1
            elif node.op == "output":
                if isinstance(node.args[0], (list, tuple)):
                    output_names = [
                        getattr(arg, "name", str(arg)) for arg in node.args[0]
                    ]
                else:
                    output_names = [getattr(node.args[0], "name", str(node.args[0]))]

        args_str = ", ".join(input_names)
        lines.append(f"def {function_name}({args_str}):")

        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            elif node.op == "output":
                if isinstance(node.args[0], (list, tuple)):
                    returns = ", ".join(
                        getattr(arg, "name", str(arg)) for arg in node.args[0]
                    )
                    lines.append(f"{self._indent}return ({returns},)")
                else:
                    returns = getattr(node.args[0], "name", str(node.args[0]))
                    lines.append(f"{self._indent}return {returns}")
            elif node.op == "call_function":
                func_repr = self._get_function_repr(node.target)
                args_repr = self._format_args(node.args, node.kwargs)
                lines.append(f"{self._indent}{node.name} = {func_repr}({args_repr})")
            elif node.op == "call_method":
                if node.args:
                    self_arg = getattr(node.args[0], "name", str(node.args[0]))
                    other_args = self._format_args(node.args[1:], node.kwargs)
                    if other_args:
                        lines.append(
                            f"{self._indent}{node.name} = {self_arg}.{node.target}({other_args})"
                        )
                    else:
                        lines.append(
                            f"{self._indent}{node.name} = {self_arg}.{node.target}()"
                        )
            elif node.op == "call_module":
                module_name = str(node.target).replace(".", "_")
                args_repr = self._format_args(node.args, node.kwargs)
                lines.append(
                    f"{self._indent}{node.name} = self_{module_name}({args_repr})"
                )
            elif node.op == "get_attr":
                attr_parts = str(node.target).split(".")
                if len(attr_parts) > 1:
                    attr_path = ".".join(attr_parts)
                    lines.append(f"{self._indent}{node.name} = self.{attr_path}")
                else:
                    lines.append(f"{self._indent}{node.name} = self.{node.target}")

        graph_code = "\n".join(lines)

        graph_readable = None
        if self._include_readable:
            try:
                buffer = io.StringIO()
                graph_module.print_readable(print_output=False)
                graph_readable = graph_module.print_readable(print_output=False)
            except Exception:
                graph_readable = str(graph)

        if self._include_constants:
            constants = self._extract_constants(graph_module)

        metadata = {
            "num_nodes": len(list(graph.nodes)),
            "num_inputs": len(input_names),
            "num_outputs": len(output_names),
        }

        return SerializedGraph(
            graph_code=graph_code,
            function_name=function_name,
            input_names=input_names,
            output_names=output_names,
            graph_readable=graph_readable,
            constants=constants,
            metadata=metadata,
        )

    def serialize_to_callable_code(
        self,
        graph_module: "GraphModule",
        function_name: str = "forward",
        include_docstring: bool = True,
    ) -> str:
        """
        Serialize a GraphModule to Python code that can be exec'd and called.

        This produces a complete Python function definition that can be
        executed standalone. The function captures all necessary imports
        and setup.

        Args:
            graph_module: The FX GraphModule to serialize
            function_name: Name for the generated function
            include_docstring: Whether to include a docstring

        Returns:
            Complete Python source code for the function
        """
        serialized = self.serialize(graph_module, function_name)

        lines = []

        lines.append("import torch")
        lines.append("import torch.nn.functional as F")
        lines.append("from torch import ops")
        lines.append("")

        if include_docstring and serialized.graph_readable:
            lines.append('"""')
            lines.append("Original graph (from print_readable):")
            lines.append("")
            for readable_line in serialized.graph_readable.split("\n")[:30]:
                lines.append(readable_line)
            if len(serialized.graph_readable.split("\n")) > 30:
                lines.append("... (truncated)")
            lines.append('"""')
            lines.append("")

        lines.append(serialized.graph_code)

        return "\n".join(lines)

    def _get_function_repr(self, target: Any) -> str:
        """
        Get a string representation of a function target.

        Converts function references to their import paths for
        serialization purposes.
        """
        if callable(target):
            module = getattr(target, "__module__", "")
            name = getattr(target, "__name__", str(target))

            if module and module.startswith("torch"):
                if module == "torch":
                    return f"torch.{name}"
                elif module.startswith("torch.ops"):
                    return f"{module}.{name}".replace("torch.ops.", "ops.")
                elif module.startswith("torch._ops"):
                    return f"ops.aten.{name}"
                elif module.startswith("torch.nn.functional"):
                    return f"F.{name}"
                else:
                    return f"torch.{name}"
            else:
                return name

        return str(target)

    def _format_args(self, args: tuple, kwargs: dict) -> str:
        """
        Format function arguments for serialization.

        Converts args and kwargs to their string representations.
        """
        parts = []

        for arg in args:
            parts.append(self._format_value(arg))

        for key, value in kwargs.items():
            parts.append(f"{key}={self._format_value(value)}")

        return ", ".join(parts)

    def _format_value(self, value: Any) -> str:
        """
        Format a single value for serialization.

        Handles various types including nodes, tensors, and primitives.
        """
        if hasattr(value, "name"):
            return value.name
        elif isinstance(value, (list, tuple)):
            formatted = [self._format_value(v) for v in value]
            if isinstance(value, tuple):
                return f"({', '.join(formatted)})"
            return f"[{', '.join(formatted)}]"
        elif isinstance(value, dict):
            items = [f"{k}: {self._format_value(v)}" for k, v in value.items()]
            return "{" + ", ".join(items) + "}"
        elif isinstance(value, torch.dtype):
            return f"torch.{value}"
        elif isinstance(value, torch.device):
            return f"torch.device('{value}')"
        elif isinstance(value, torch.Tensor):
            return f"torch.tensor({value.tolist()})"
        elif value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return repr(value)
        elif isinstance(value, str):
            return repr(value)
        else:
            return repr(value)

    def _extract_constants(self, graph_module: "GraphModule") -> dict[str, Any]:
        """
        Extract constant values from a GraphModule.

        Finds all get_attr nodes and extracts their values.
        """
        constants: dict[str, Any] = {}

        for name, value in graph_module.named_parameters(recurse=False):
            if isinstance(value, torch.Tensor):
                constants[name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "requires_grad": value.requires_grad,
                }
            else:
                constants[name] = str(value)

        for name, value in graph_module.named_buffers(recurse=False):
            if isinstance(value, torch.Tensor):
                constants[name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            else:
                constants[name] = str(value)

        return constants


def serialize_graph_to_code(
    graph_module: "GraphModule",
    function_name: str = "forward",
) -> str:
    """
    Convenience function to serialize a GraphModule to Python code.

    Args:
        graph_module: The FX GraphModule to serialize
        function_name: Name for the generated function

    Returns:
        Python source code string for the function
    """
    serializer = GraphSerializer()
    return serializer.serialize_to_callable_code(graph_module, function_name)


def get_graph_readable(graph_module: "GraphModule") -> Optional[str]:
    """
    Get the print_readable() output for a GraphModule.

    This is useful for including human-readable graph representations
    in generated code comments.

    Args:
        graph_module: The GraphModule to get readable representation for

    Returns:
        The print_readable() output, or None if it fails
    """
    if graph_module is None:
        return None

    try:
        return graph_module.print_readable(print_output=False)
    except Exception:
        try:
            return str(graph_module.graph)
        except Exception:
            return None


def extract_graph_metadata(graph_module: "GraphModule") -> dict[str, Any]:
    """
    Extract metadata about a GraphModule for logging/debugging.

    Args:
        graph_module: The GraphModule to extract metadata from

    Returns:
        Dictionary with graph metadata
    """
    if graph_module is None:
        return {"error": "No graph provided"}

    graph = graph_module.graph

    op_counts: dict[str, int] = {}
    input_count = 0
    output_count = 0

    for node in graph.nodes:
        op = node.op
        op_counts[op] = op_counts.get(op, 0) + 1

        if op == "placeholder":
            input_count += 1
        elif op == "output":
            if isinstance(node.args[0], (list, tuple)):
                output_count = len(node.args[0])
            else:
                output_count = 1

    return {
        "num_nodes": len(list(graph.nodes)),
        "num_inputs": input_count,
        "num_outputs": output_count,
        "op_counts": op_counts,
    }
