"""
Python code generation backend for torch.compile pythonify feature.

This module implements the gen_python backend, which emits explicit Python source
code that represents all the runtime machinery of torch.compile. The generated
code is readable, well-documented, and can be executed via exec() or saved to
a file.

The PythonCodeGenVisitor traverses the RuntimeWrapper IR and generates Python
source code that:
1. Extracts arguments from model parameters and frame locals
2. Checks guards at runtime with clear assertion messages
3. Defines the AOT Autograd wrapper function
4. Sets up CUDA graphs if enabled
5. Invokes the compiled callable and exposes the result

This backend is used when pythonify is specified in torch.compile().

Object ID Approach for Parameter/Buffer Access
==============================================
When parameters and buffers have their `object_id` field set in the IR,
this code generator emits code using the ctypes-based `obj_from_id` helper:

    import ctypes
    def obj_from_id(obj_id):
        return ctypes.cast(obj_id, ctypes.py_object).value

    # Then for each parameter/buffer:
    arg2 = obj_from_id(140234567890)  # Instead of model.W

This approach has the following characteristics:

ADVANTAGES:
- No need for the model variable in exec() scope
- Works with any model structure (nested modules, dynamic attrs)
- Guarantees exact tensor identity with compiled version

CRITICAL LIMITATIONS:
- PROCESS-LOCAL: Object IDs are memory addresses, only valid within
  the same Python process. The generated file CANNOT be used in a
  different Python process or after interpreter restart.

- LIFETIME: Original tensors must stay alive. If the model is deleted
  or garbage collected, the object IDs become dangling references.

- NOT PERSISTABLE: Do NOT save the generated file for later use.
  It is designed for immediate execution in the same session.

- CRASH RISK: Invalid object IDs will crash or corrupt the interpreter.
  Never manually edit object IDs in generated files.

IMPORTANT: exec() Scoping Limitation
=====================================
Python's exec() function has a known scoping limitation when called with
separate globals and locals dictionaries. When globals != locals:

1. Variable assignments go to the locals dict
2. Variable lookups in nested functions/classes look in the globals dict
3. This breaks closures and nested class definitions like torch.autograd.Function

The requirements specification shows:
    exec(f.read(), frame.f_globals, frame.f_locals)

However, due to this Python limitation, the RECOMMENDED pattern is to use a
merged namespace:

    import inspect
    frame = inspect.currentframe()
    namespace = {**frame.f_globals, **frame.f_locals}
    exec(code, namespace)
    y = namespace["y"]

This merged namespace approach ensures that:
- All variables from both scopes are accessible
- Nested class definitions (like the AOT Autograd function) work correctly
- The result is accessible via namespace["y"]

Note: exec() always returns None. The original requirements example
`y = exec(...)` assigns None to y. The result is accessed via the namespace
dictionary passed to exec(), e.g., namespace["y"] or f_locals["y"].

Example generated code:
    # Argument extraction
    arg1 = model.W
    arg2 = f_locals["x"]

    # Guard evaluation
    assert arg2.shape[0] == 3, "Expected batch size 3"

    # Invoke compiled callable
    result = compiled_fn(arg1, arg2)

    # Expose result
    y = result
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from .ir import (
    ArgumentExtractionNode,
    ArgumentSource,
    AOTAutogradWrapperNode,
    CallableInvocationNode,
    CodeGenVisitor,
    CompiledRegionNode,
    CUDAGraphSetupNode,
    GuardCheckNode,
    GuardType,
    KernelLoadNode,
    KernelType,
    ModelSource,
    MultiRegionDispatchNode,
    RegionExecutionMode,
    ReturnResultNode,
    RuntimeWrapperIR,
)


if TYPE_CHECKING:
    pass


@dataclass
class CodeEmitter:
    """
    Utility class for building Python source code with proper formatting.

    Handles indentation, line management, section headers, and import tracking.
    Provides methods for common Python code patterns to ensure consistency.

    Attributes:
        lines: Accumulated lines of generated code
        indent_level: Current indentation level (in units of indent_str)
        indent_str: String to use for one level of indentation
        imports: Set of import statements to include at the top
    """

    lines: list[str] = field(default_factory=list)
    indent_level: int = 0
    indent_str: str = "    "
    imports: set[str] = field(default_factory=set)

    def add_import(self, import_stmt: str) -> None:
        """
        Add an import statement to the imports set.

        Args:
            import_stmt: Full import statement (e.g., "import torch")
        """
        self.imports.add(import_stmt)

    def emit_line(self, line: str = "") -> None:
        """
        Emit a single line with proper indentation.

        Args:
            line: The line content (without leading indentation)
        """
        if line:
            self.lines.append(f"{self.indent_str * self.indent_level}{line}")
        else:
            self.lines.append("")

    def emit_comment(self, comment: str) -> None:
        """
        Emit a comment line with proper indentation.

        Args:
            comment: The comment text (without the # prefix)
        """
        self.emit_line(f"# {comment}")

    def emit_section_header(self, title: str) -> None:
        """
        Emit a section header with decorative formatting.

        Produces output like:
            # ============================================================
            # Section Title
            # ============================================================

        Args:
            title: The section title
        """
        separator = "=" * 60
        self.emit_line()
        self.emit_comment(separator)
        self.emit_comment(title)
        self.emit_comment(separator)

    def emit_blank_line(self) -> None:
        """Emit an empty line."""
        self.emit_line("")

    def indent(self) -> None:
        """Increase the indentation level by one."""
        self.indent_level += 1

    def dedent(self) -> None:
        """Decrease the indentation level by one."""
        if self.indent_level > 0:
            self.indent_level -= 1

    def get_imports_block(self) -> str:
        """
        Get all imports as a formatted block of code.

        Returns:
            Multi-line string with all import statements, sorted
        """
        if not self.imports:
            return ""
        sorted_imports = sorted(self.imports)
        return "\n".join(sorted_imports)

    def get_code(self) -> str:
        """
        Get the complete generated code as a string.

        Combines imports with the main code body.

        Returns:
            Complete Python source code
        """
        imports_block = self.get_imports_block()
        code_block = "\n".join(self.lines)

        if imports_block:
            return f"{imports_block}\n\n{code_block}"
        return code_block


def escape_string(s: str) -> str:
    """
    Escape a string for use in Python source code.

    Handles special characters like quotes, newlines, and backslashes.

    Args:
        s: The string to escape

    Returns:
        Escaped string suitable for Python source code
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def format_value(value: Any) -> str:
    """
    Format a Python value as source code representation.

    Handles common types like strings, numbers, None, and torch types.

    Special handling for string representations of torch types:
    - "torch.float32" -> torch.float32 (no quotes)
    - "torch.int64" -> torch.int64 (no quotes)

    Args:
        value: The value to format

    Returns:
        String representation suitable for Python source code
    """
    if value is None:
        return "None"
    elif isinstance(value, str):
        if value.startswith("torch.") and not value.startswith("torch._"):
            return value
        return f'"{escape_string(value)}"'
    elif isinstance(value, bool):
        return "True" if value else "False"
    elif isinstance(value, (int, float)):
        return repr(value)
    elif hasattr(value, "__module__") and "torch" in str(value.__module__):
        return repr(value)
    else:
        return repr(value)


class PythonCodeGenVisitor(CodeGenVisitor):
    """
    Code generation visitor that produces Python source code.

    This visitor traverses the RuntimeWrapper IR and generates explicit Python
    code that represents all runtime machinery. The generated code is designed
    to be:
    - Readable and well-documented
    - Executable via exec() with proper f_locals/f_globals
    - Suitable for saving to a file

    The visitor accumulates code in a CodeEmitter as it visits each node.
    After visiting all nodes, call get_code() to retrieve the generated Python
    source code.

    Usage:
        visitor = PythonCodeGenVisitor()
        ir.accept_all(visitor)
        code = visitor.get_code()
        exec(code, globals, locals)
    """

    def __init__(
        self,
        model_var_name: str = "model",
        f_locals_var_name: str = "f_locals",
        f_globals_var_name: str = "f_globals",
        compiled_fn_var_name: str = "compiled_fn",
        expose_result_in_locals: bool = True,
        expose_result_in_globals: bool = False,
        model_source: ModelSource = ModelSource.CLOSURE,
    ) -> None:
        """
        Initialize the visitor with variable name configuration.

        Args:
            model_var_name: Name of the model variable in the generated code
            f_locals_var_name: Name of the f_locals dictionary variable
            f_globals_var_name: Name of the f_globals dictionary variable
            compiled_fn_var_name: Name of the compiled function variable
            expose_result_in_locals: If True, write result to f_locals dict for
                exec() compatibility. This is essential because frame.f_locals
                is read-only in Python, so assignments in exec'd code don't
                persist. Writing to f_locals["y"] makes the result accessible.
            expose_result_in_globals: If True, also write result to f_globals dict.
                This can be useful when using exec(code, globals) without locals.
            model_source: Specifies where the model comes from:
                - ModelSource.CLOSURE: Model is directly accessible as a variable
                  (default, for when passed to exec globals)
                - ModelSource.F_LOCALS: Model is in f_locals["model"]
                  (for when model is a local variable in the calling function)
                - ModelSource.F_GLOBALS: Model is in f_globals["model"]
                  (for when model is a global/module-level variable)
        """
        self._emitter = CodeEmitter()
        self._model_var_name = model_var_name
        self._f_locals_var_name = f_locals_var_name
        self._f_globals_var_name = f_globals_var_name
        self._compiled_fn_var_name = compiled_fn_var_name
        self._expose_result_in_locals = expose_result_in_locals
        self._expose_result_in_globals = expose_result_in_globals
        self._model_source = model_source
        self._argument_names: list[str] = []
        self._has_emitted_arg_section = False
        self._has_emitted_guard_section = False
        self._has_emitted_aot_section = False
        self._has_emitted_cuda_section = False
        self._has_emitted_kernel_section = False
        self._has_emitted_invoke_section = False
        self._has_emitted_return_section = False
        self._has_emitted_model_extraction = False
        self._has_emitted_obj_from_id_helper = False
        self._has_inductor_kernel = False
        self._has_inductor_backward_kernel = False
        self._inductor_uses_list_args = True
        self._inductor_returns_tuple = True

    def get_code(self) -> str:
        """
        Get the generated Python source code.

        Returns:
            Complete Python source code as a string
        """
        return self._emitter.get_code()

    def get_emitter(self) -> CodeEmitter:
        """
        Get the underlying CodeEmitter for advanced access.

        Returns:
            The CodeEmitter instance used for code generation
        """
        return self._emitter

    def prescan_ir(self, ir: "RuntimeWrapperIR") -> None:
        """
        Pre-scan the IR to detect features that affect code generation.

        This method scans all nodes in the IR before visiting them to detect
        features like Inductor kernels that need to be known before generating
        code for earlier nodes (e.g., AOT Autograd forward method needs to know
        if there's an Inductor kernel to use the correct calling convention).

        Detects both forward and backward Inductor kernels for proper autograd
        function generation.

        Call this method before calling ir.accept_all(visitor).

        Args:
            ir: The RuntimeWrapperIR to scan
        """
        for node in ir.nodes:
            if isinstance(node, KernelLoadNode):
                if node.kernel_type == KernelType.INLINE:
                    source_type = node.metadata.get("source", "")
                    is_backward = node.metadata.get("is_backward", False)
                    if source_type == "inductor":
                        if is_backward:
                            self._has_inductor_backward_kernel = True
                        else:
                            self._has_inductor_kernel = True

    def requires_model_access(self, ir: "RuntimeWrapperIR") -> bool:
        """
        Check if the IR contains any nodes that require model object access.

        When all parameters and buffers use OBJECT_ID source, the model object
        is not needed and should not be mentioned in the generated code header.
        This method scans the IR to determine if any nodes still require model
        access (PARAMETER, BUFFER, or MODEL_ATTRIBUTE sources without object_id).

        Args:
            ir: The RuntimeWrapperIR to scan

        Returns:
            True if any node requires model access, False otherwise
        """
        for node in ir.nodes:
            if isinstance(node, ArgumentExtractionNode):
                if node.source in (
                    ArgumentSource.PARAMETER,
                    ArgumentSource.BUFFER,
                    ArgumentSource.MODEL_ATTRIBUTE,
                ):
                    return True
        return False

    def visit_argument_extraction(self, node: ArgumentExtractionNode) -> str:
        """
        Generate code to extract an argument from its source.

        Produces code like:
            model = f_locals["model"]  # if model_source is F_LOCALS
            arg1 = model.W             # for PARAMETER
            arg2 = f_locals["x"]       # for F_LOCALS

        Args:
            node: The ArgumentExtractionNode to process

        Returns:
            The generated line of code
        """
        if not self._has_emitted_arg_section:
            self._emitter.emit_section_header("Argument Extraction")
            self._has_emitted_arg_section = True
            self._emit_model_extraction_if_needed(node)

        if node.source in (
            ArgumentSource.PARAMETER,
            ArgumentSource.BUFFER,
            ArgumentSource.MODEL_ATTRIBUTE,
        ) and not self._has_emitted_model_extraction:
            self._emit_model_extraction_if_needed(node)

        access_code = self._generate_access_code(node)
        line = f"{node.name} = {access_code}"
        self._emitter.emit_line(line)
        self._argument_names.append(node.name)
        return line

    def _emit_model_extraction_if_needed(self, node: ArgumentExtractionNode) -> None:
        """
        Emit code to extract the model from f_locals or f_globals if needed.

        When model_source is F_LOCALS or F_GLOBALS, we need to extract the
        model first before we can access its attributes. This is essential
        for exec() compatibility when the model is a local variable.

        For example, when called with:
            exec(code, frame.f_globals, frame.f_locals)
        and the model is a local variable, we need:
            model = f_locals["model"]
        before we can do:
            arg1 = model.W

        Args:
            node: The ArgumentExtractionNode that triggered this check
        """
        if self._has_emitted_model_extraction:
            return

        if node.source not in (
            ArgumentSource.PARAMETER,
            ArgumentSource.BUFFER,
            ArgumentSource.MODEL_ATTRIBUTE,
        ):
            return

        if self._model_source == ModelSource.F_LOCALS:
            self._emitter.emit_comment(
                f"Extract model from f_locals (model is a local variable)"
            )
            extraction_line = (
                f'{self._model_var_name} = '
                f'{self._f_locals_var_name}["{self._model_var_name}"]'
            )
            self._emitter.emit_line(extraction_line)
            self._emitter.emit_blank_line()
            self._has_emitted_model_extraction = True

        elif self._model_source == ModelSource.F_GLOBALS:
            self._emitter.emit_comment(
                f"Extract model from f_globals (model is a global variable)"
            )
            extraction_line = (
                f'{self._model_var_name} = '
                f'{self._f_globals_var_name}["{self._model_var_name}"]'
            )
            self._emitter.emit_line(extraction_line)
            self._emitter.emit_blank_line()
            self._has_emitted_model_extraction = True

        else:
            self._has_emitted_model_extraction = True

    def _emit_obj_from_id_helper(self) -> None:
        """
        Emit the obj_from_id helper function for retrieving objects by ID.

        This function uses ctypes to retrieve a Python object from its memory
        address (as obtained by id()). This is only safe when:
        1. The object has not been garbage collected
        2. We're in the same Python process where the ID was captured
        3. The generated code is executed before the original object is deleted

        The helper is emitted once, at the top of the argument extraction section,
        before any arguments are extracted that use OBJECT_ID source.
        """
        if self._has_emitted_obj_from_id_helper:
            return

        self._emitter.add_import("import ctypes")
        self._emitter.emit_blank_line()
        self._emitter.emit_comment(
            "Helper function to retrieve Python objects by their ID."
        )
        self._emitter.emit_comment(
            "This uses ctypes to reconstruct the object from its memory address."
        )
        self._emitter.emit_comment(
            "WARNING: Only valid within the same Python process while objects are alive."
        )
        self._emitter.emit_line("def obj_from_id(obj_id):")
        self._emitter.indent()
        self._emitter.emit_line("return ctypes.cast(obj_id, ctypes.py_object).value")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._has_emitted_obj_from_id_helper = True

    def _generate_access_code(self, node: ArgumentExtractionNode) -> str:
        """
        Generate the code to access a value from its source.

        Args:
            node: The ArgumentExtractionNode describing the access

        Returns:
            Python expression to access the value
        """
        if node.source == ArgumentSource.OBJECT_ID:
            if node.object_id is not None:
                self._emit_obj_from_id_helper()
                return f"obj_from_id({node.object_id})"
            else:
                return f'{self._f_locals_var_name}["{node.access_path}"]'
        elif node.source == ArgumentSource.F_LOCALS:
            return f'{self._f_locals_var_name}["{node.access_path}"]'
        elif node.source == ArgumentSource.F_GLOBALS:
            return f'{self._f_globals_var_name}["{node.access_path}"]'
        elif node.source == ArgumentSource.CONSTANT:
            return format_value(node.access_path)
        elif node.source in (
            ArgumentSource.PARAMETER,
            ArgumentSource.BUFFER,
            ArgumentSource.MODEL_ATTRIBUTE,
        ):
            if node.nested_path:
                path_parts = [self._model_var_name] + node.nested_path
                return ".".join(path_parts)
            else:
                return f"{self._model_var_name}.{node.access_path}"
        else:
            return f'{self._f_locals_var_name}["{node.access_path}"]'

    def visit_guard_check(self, node: GuardCheckNode) -> str:
        """
        Generate code to check a guard condition.

        For static guards, produces code like:
            assert arg2.shape[0] == 3, "Expected batch size 3"

        For dynamic guards (is_dynamic=True), produces a comment indicating
        that this dimension is dynamic and not checked:
            # DYNAMIC: arg2.shape[0] can vary (no assertion)

        If the dynamic guard has min/max constraints, those are checked:
            assert arg2.shape[0] >= 1, "Minimum batch size is 1"
            assert arg2.shape[0] <= 100, "Maximum batch size is 100"

        Args:
            node: The GuardCheckNode to process

        Returns:
            The generated assertion line or comment
        """
        if not self._has_emitted_guard_section:
            self._emitter.emit_section_header("Guard Evaluation")
            self._has_emitted_guard_section = True

        if node.is_dynamic:
            return self._generate_dynamic_guard_code(node)

        condition = self._generate_guard_condition(node)
        error_msg = node.error_message or f"Guard failed: {node.condition}"
        line = f'assert {condition}, "{escape_string(error_msg)}"'
        self._emitter.emit_line(line)
        return line

    def _generate_dynamic_guard_code(self, node: GuardCheckNode) -> str:
        """
        Generate code for a dynamic guard (no strict assertion).

        Dynamic guards generate comments instead of assertions, but may
        include min/max bound checks if constraints are specified.

        Args:
            node: The GuardCheckNode marked as dynamic

        Returns:
            The generated comment/code for the dynamic guard
        """
        target = node.target_name
        dim = node.dimension

        comment_line = f"# DYNAMIC: {target}.shape[{dim}] can vary (no assertion)"
        self._emitter.emit_line(comment_line)

        lines_generated = [comment_line]

        if node.min_value is not None:
            min_condition = f"{target}.shape[{dim}] >= {node.min_value}"
            min_error = f"Dynamic dimension {target}.shape[{dim}] must be >= {node.min_value}"
            min_line = f'assert {min_condition}, "{min_error}"'
            self._emitter.emit_line(min_line)
            lines_generated.append(min_line)

        if node.max_value is not None:
            max_condition = f"{target}.shape[{dim}] <= {node.max_value}"
            max_error = f"Dynamic dimension {target}.shape[{dim}] must be <= {node.max_value}"
            max_line = f'assert {max_condition}, "{max_error}"'
            self._emitter.emit_line(max_line)
            lines_generated.append(max_line)

        return lines_generated[0]

    def _generate_guard_condition(self, node: GuardCheckNode) -> str:
        """
        Generate the condition expression for a guard check.

        Args:
            node: The GuardCheckNode describing the guard

        Returns:
            Python boolean expression for the guard condition
        """
        target = node.target_name

        if node.guard_type == GuardType.SHAPE:
            if node.dimension is not None:
                return f"{target}.shape[{node.dimension}] == {node.expected_value}"
            else:
                return f"{target}.shape == {format_value(node.expected_value)}"
        elif node.guard_type == GuardType.DTYPE:
            dtype_repr = format_value(node.expected_value)
            return f"{target}.dtype == {dtype_repr}"
        elif node.guard_type == GuardType.DEVICE:
            if isinstance(node.expected_value, str):
                return f'{target}.device.type == "{node.expected_value}"'
            else:
                return f"{target}.device == {format_value(node.expected_value)}"
        elif node.guard_type == GuardType.VALUE:
            return f"{target} == {format_value(node.expected_value)}"
        elif node.guard_type == GuardType.TYPE:
            type_repr = format_value(node.expected_value)
            return f"type({target}) == {type_repr}"
        elif node.guard_type == GuardType.IDENTITY:
            return f"{target} is {format_value(node.expected_value)}"
        elif node.guard_type == GuardType.TENSOR_MATCH:
            return node.condition if node.condition else "True"
        else:
            return node.condition if node.condition else "True"

    def visit_aot_autograd_wrapper(self, node: AOTAutogradWrapperNode) -> str:
        """
        Generate code for the AOT Autograd function wrapper.

        This method generates a complete, executable torch.autograd.Function class
        with working forward() and backward() static methods. The generated class
        can be used to perform automatic differentiation on the compiled model.

        There are several code generation modes:

        1. No backward graph/kernel: Generate a simple forward-only wrapper that
           delegates to the compiled function.

        2. Inductor backward kernel: Generate a torch.autograd.Function that uses
           compiled_fn for forward and compiled_fn_backward for backward. This is
           the preferred mode for pythonify output with autograd support.

        3. Serialized code provided: Embed the pre-serialized forward/backward
           code directly in the generated output.

        4. Forward/backward graphs provided: Serialize the graphs at generation
           time and embed the resulting code.

        5. Compiled callables provided: Generate code that references external
           compiled callables (compiled_forward, compiled_backward).

        Args:
            node: The AOTAutogradWrapperNode to process

        Returns:
            The generated code section
        """
        if not self._has_emitted_aot_section:
            self._emitter.emit_section_header("AOT Autograd Function Definition")
            self._has_emitted_aot_section = True

        has_backward = (
            node.backward_graph is not None
            or node.serialized_backward_code is not None
            or node.compiled_backward is not None
            or self._has_inductor_backward_kernel
        )

        has_forward_implementation = (
            node.serialized_forward_code is not None
            or self._has_inductor_kernel
        )

        if not has_backward and not has_forward_implementation:
            comment = "# No backward graph - using forward-only compiled function"
            self._emitter.emit_line(comment)
            return comment

        self._emitter.add_import("import torch")

        forward_code = self._get_forward_code(node)
        backward_code = self._get_backward_code(node)

        self._emitter.emit_line(f"class {node.class_name}(torch.autograd.Function):")
        self._emitter.indent()

        self._emit_forward_method(node, forward_code)

        # Always emit a backward method. If there's a backward kernel, it will
        # use that. Otherwise, emit a fallback backward that raises a clear error.
        # This is necessary because without a backward method, PyTorch raises:
        # "NotImplementedError: You must implement either the backward or vjp
        # method for your custom autograd.Function to use it with backward mode AD."
        # A clear error message is more helpful than this generic PyTorch error.
        self._emitter.emit_blank_line()
        self._emit_backward_method(node, backward_code, has_backward=has_backward)

        self._emitter.dedent()

        self._emitter.emit_blank_line()
        self._emitter.emit_line(f"callable = {node.class_name}.apply")

        return f"class {node.class_name}"

    def _get_forward_code(self, node: AOTAutogradWrapperNode) -> str:
        """
        Get the forward implementation code for the autograd function.

        This method determines the best source for forward code and returns
        it ready for embedding in the generated class.

        Priority:
        1. serialized_forward_code (pre-serialized)
        2. forward_graph (serialize at generation time)
        3. Fall back to calling compiled_fn

        Args:
            node: The AOTAutogradWrapperNode

        Returns:
            Python code string for the forward implementation
        """
        if node.serialized_forward_code:
            return node.serialized_forward_code

        if node.forward_graph is not None:
            try:
                from .adapters.graph_serializer import GraphSerializer

                serializer = GraphSerializer(include_readable=False)
                serialized = serializer.serialize(
                    node.forward_graph, "compiled_forward"
                )
                return serialized.graph_code
            except Exception:
                pass

        return None

    def _get_backward_code(self, node: AOTAutogradWrapperNode) -> str:
        """
        Get the backward implementation code for the autograd function.

        This method determines the best source for backward code and returns
        it ready for embedding in the generated class.

        Priority:
        1. serialized_backward_code (pre-serialized)
        2. backward_graph (serialize at generation time)
        3. None (no backward graph available)

        Args:
            node: The AOTAutogradWrapperNode

        Returns:
            Python code string for the backward implementation, or None
        """
        if node.serialized_backward_code:
            return node.serialized_backward_code

        if node.backward_graph is not None:
            try:
                from .adapters.graph_serializer import GraphSerializer

                serializer = GraphSerializer(include_readable=False)
                serialized = serializer.serialize(
                    node.backward_graph, "compiled_backward"
                )
                return serialized.graph_code
            except Exception:
                pass

        return None

    def _emit_forward_method(
        self, node: AOTAutogradWrapperNode, forward_code: str
    ) -> None:
        """
        Emit the forward() static method for the autograd Function.

        Generates code that:
        1. Saves tensors for backward pass (if needed)
        2. Calls the forward computation
        3. Returns the result

        The forward computation is either:
        - Inlined serialized code from the FX graph
        - A call to the external compiled function (compiled_fn)

        When Inductor source code is embedded (self._has_inductor_kernel is True),
        the forward method calls compiled_fn with the correct Inductor calling
        convention:
        - Arguments are passed as a list: compiled_fn(list(args))
        - Inductor returns a tuple: (output, saved_tensors...), so we extract [0]

        When we have an Inductor backward kernel, the forward pass also saves
        the additional return values from Inductor (the tensors saved for backward).

        Args:
            node: The AOTAutogradWrapperNode
            forward_code: Serialized forward implementation code, or None
        """
        self._emitter.emit_line("@staticmethod")
        self._emitter.emit_line("def forward(ctx, *args):")
        self._emitter.indent()

        if node.saved_tensors_indices:
            indices_repr = repr(node.saved_tensors_indices)
            self._emitter.emit_comment(f"Save tensors at indices: {indices_repr}")
            tensors_to_save = ", ".join(
                f"args[{i}]" for i in node.saved_tensors_indices
            )
            self._emitter.emit_line(f"ctx.save_for_backward({tensors_to_save})")
            self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Save context info for backward pass."
        )
        self._emitter.emit_line(f"ctx.num_inputs = {node.num_inputs}")

        if forward_code:
            self._emitter.emit_blank_line()
            self._emitter.emit_comment("Forward computation (from compiled graph):")
            for line in forward_code.strip().split("\n"):
                if line.startswith("def compiled_forward"):
                    continue
                stripped = line.lstrip()
                if stripped:
                    self._emitter.emit_line(stripped)
        elif self._has_inductor_kernel:
            self._emitter.emit_blank_line()
            self._emitter.emit_comment(
                "Forward computation via Inductor-compiled kernel."
            )

            if self._has_inductor_backward_kernel:
                self._emitter.emit_comment(
                    "Inductor's call() returns (output, saved_tensors...) where"
                )
                self._emitter.emit_comment(
                    "saved_tensors are needed for backward. We save them here."
                )
                self._emitter.emit_line(
                    f"_inductor_result = {self._compiled_fn_var_name}(list(args))"
                )
                self._emitter.emit_blank_line()

                self._emitter.emit_comment(
                    "Save the tensors returned by Inductor for backward pass."
                )
                self._emitter.emit_comment(
                    "The first element is the output; remaining are saved tensors."
                )
                self._emitter.emit_line("if len(_inductor_result) > 1:")
                self._emitter.indent()
                self._emitter.emit_line("ctx.save_for_backward(*_inductor_result[1:])")
                self._emitter.dedent()
                self._emitter.emit_blank_line()
                self._emitter.emit_line("return _inductor_result[0]")
            else:
                self._emitter.emit_comment(
                    "Inductor's call() function expects args as a list and"
                )
                self._emitter.emit_comment(
                    "returns (output, saved_tensors...), so we extract [0]."
                )
                if self._inductor_uses_list_args and self._inductor_returns_tuple:
                    self._emitter.emit_line(
                        f"_inductor_result = {self._compiled_fn_var_name}(list(args))"
                    )
                    self._emitter.emit_line("return _inductor_result[0]")
                elif self._inductor_uses_list_args:
                    self._emitter.emit_line(
                        f"return {self._compiled_fn_var_name}(list(args))"
                    )
                elif self._inductor_returns_tuple:
                    self._emitter.emit_line(
                        f"_inductor_result = {self._compiled_fn_var_name}(*args)"
                    )
                    self._emitter.emit_line("return _inductor_result[0]")
                else:
                    self._emitter.emit_line(
                        f"return {self._compiled_fn_var_name}(*args)"
                    )
        else:
            self._emitter.emit_blank_line()
            self._emitter.emit_comment(
                "Forward computation handled by compiled function."
            )
            self._emitter.emit_line(f"return {self._compiled_fn_var_name}(*args)")

        self._emitter.dedent()

    def _emit_backward_method(
        self, node: AOTAutogradWrapperNode, backward_code: str, has_backward: bool = True
    ) -> None:
        """
        Emit the backward() static method for the autograd Function.

        Generates code that:
        1. Retrieves saved tensors from context
        2. Computes gradients using the backward graph/kernel
        3. Returns gradients for each input

        The backward computation is either:
        - Inductor backward kernel (compiled_fn_backward) - preferred when available
        - Inlined serialized code from the FX graph
        - A call to the external compiled backward function
        - A fallback that raises RuntimeError (for inference-compiled models)

        When has_backward is False, a fallback backward method is generated that
        raises a clear RuntimeError explaining that the model was compiled in
        inference mode without a backward kernel. This is more helpful than
        PyTorch's generic "NotImplementedError: You must implement..." message.

        Args:
            node: The AOTAutogradWrapperNode
            backward_code: Serialized backward implementation code, or None
            has_backward: Whether a backward kernel/graph is available. If False,
                generates a fallback that raises RuntimeError.
        """
        self._emitter.emit_line("@staticmethod")
        self._emitter.emit_line("def backward(ctx, *grad_outputs):")
        self._emitter.indent()

        # If no backward kernel is available, emit a clear error message
        if not has_backward:
            self._emitter.emit_comment(
                "No backward kernel was compiled for this model."
            )
            self._emitter.emit_comment(
                "This can happen when the model was compiled with torch.no_grad()"
            )
            self._emitter.emit_comment(
                "or torch.inference_mode(), or with inputs that don't require grad."
            )
            self._emitter.emit_comment(
                "To enable backward pass, recompile with gradient-tracking enabled."
            )
            self._emitter.emit_line(
                'raise RuntimeError('
                '"Backward pass not available: this model was compiled in inference mode "'
                '"without a backward kernel. To enable gradients, recompile the model "'
                '"with inputs that require gradients and outside of torch.no_grad() context."'
                ')'
            )
            self._emitter.dedent()
            return

        self._emitter.emit_comment(
            "Retrieve saved tensors from forward pass."
        )
        self._emitter.emit_line("saved = ctx.saved_tensors")
        self._emitter.emit_line("num_inputs = ctx.num_inputs")
        self._emitter.emit_blank_line()

        if self._has_inductor_backward_kernel:
            self._emitter.emit_comment(
                "Backward computation via Inductor-compiled backward kernel."
            )
            self._emitter.emit_comment(
                "AOT Autograd structures backward inputs as: (*saved_tensors, *tangents)"
            )
            self._emitter.emit_comment(
                "where tangents are the grad_outputs from the autograd engine."
            )
            self._emitter.emit_blank_line()

            self._emitter.emit_comment(
                "Prepare backward inputs: saved tensors followed by grad_outputs."
            )
            self._emitter.emit_line("backward_inputs = list(saved) + list(grad_outputs)")
            self._emitter.emit_blank_line()

            self._emitter.emit_comment(
                "Call the Inductor-compiled backward kernel."
            )
            self._emitter.emit_line("_bw_result = compiled_fn_backward(backward_inputs)")
            self._emitter.emit_blank_line()

            self._emitter.emit_comment(
                "The backward kernel returns gradients as a tuple/list."
            )
            self._emitter.emit_comment(
                "Extract the first element if it's a tuple."
            )
            self._emitter.emit_line("if isinstance(_bw_result, (tuple, list)) and len(_bw_result) > 0:")
            self._emitter.indent()
            self._emitter.emit_line("if isinstance(_bw_result[0], (tuple, list)):")
            self._emitter.indent()
            self._emitter.emit_line("return _bw_result[0]")
            self._emitter.dedent()
            self._emitter.emit_line("return _bw_result")
            self._emitter.dedent()
            self._emitter.emit_line("return _bw_result")
        elif backward_code:
            self._emitter.emit_comment("Backward computation (from compiled graph):")
            for line in backward_code.strip().split("\n"):
                if line.startswith("def compiled_backward"):
                    continue
                stripped = line.lstrip()
                if stripped:
                    self._emitter.emit_line(stripped)
        else:
            self._emitter.emit_comment(
                "Backward computation using gradient formula."
            )
            self._emitter.emit_comment(
                "When a serialized backward graph is available, it will be inlined here."
            )
            self._emitter.emit_comment(
                "For now, return gradients matching the input count."
            )
            self._emitter.emit_blank_line()
            self._emitter.emit_comment(
                "Return None for each input that doesn't require grad."
            )
            self._emitter.emit_line("return (None,) * num_inputs")

        self._emitter.dedent()

    def visit_cuda_graph_setup(self, node: CUDAGraphSetupNode) -> str:
        """
        Generate code for CUDA graph capture and replay setup.

        Args:
            node: The CUDAGraphSetupNode to process

        Returns:
            The generated CUDA graph setup code
        """
        if not self._has_emitted_cuda_section:
            self._emitter.emit_section_header("CUDA Graphs Setup")
            self._has_emitted_cuda_section = True

        self._emitter.add_import("import torch")

        lines = []

        self._emitter.emit_line(f"# CUDA Graph ID: {node.graph_id}")
        self._emitter.emit_line(f"# Warmup runs: {node.warmup_runs}")
        self._emitter.emit_line(f"# Capture mode: {node.capture_mode}")

        graph_var = f"{node.graph_id}_graph"
        self._emitter.emit_line(f"{graph_var} = torch.cuda.CUDAGraph()")

        if node.stream_name != "default":
            stream_var = f"{node.graph_id}_stream"
            self._emitter.emit_line(f"{stream_var} = torch.cuda.Stream()")

        self._emitter.emit_blank_line()
        self._emitter.emit_line(f"# Warmup runs for {node.graph_id}")
        self._emitter.emit_line(f"for _ in range({node.warmup_runs}):")
        self._emitter.indent()
        self._emitter.emit_line("pass  # Warmup would invoke compiled function here")
        self._emitter.dedent()

        self._emitter.emit_blank_line()
        self._emitter.emit_line("# Capture the CUDA graph")
        self._emitter.emit_line(f"with torch.cuda.graph({graph_var}):")
        self._emitter.indent()
        self._emitter.emit_line("pass  # Graph capture would happen here")
        self._emitter.dedent()

        return f"CUDA Graph: {node.graph_id}"

    def visit_callable_invocation(self, node: CallableInvocationNode) -> str:
        """
        Generate code to invoke the compiled callable.

        Produces code like:
            result = compiled_fn(arg1, arg2, arg3)
        or:
            result = compiled_fn([arg1, arg2, arg3])
        or:
            _raw_result = compiled_fn([arg1, arg2])
            result = _raw_result[0]

        If node.args_as_list is True, arguments are wrapped in a list.
        This is needed for Inductor's call() function which takes a
        single args list.

        If node.extract_first_output is True, the first element is
        extracted from the result tuple (Inductor returns tuple).

        Args:
            node: The CallableInvocationNode to process

        Returns:
            The generated invocation line
        """
        if not self._has_emitted_invoke_section:
            self._emitter.emit_section_header("Invoke Compiled Callable")
            self._has_emitted_invoke_section = True

        args_str = ", ".join(node.argument_names)

        if node.is_autograd_function:
            callable_name = "callable"
        else:
            callable_name = node.callable_name

        if node.args_as_list:
            args_call = f"[{args_str}]"
        else:
            args_call = args_str

        if node.extract_first_output:
            raw_result_name = "_raw_result"
            line = f"{raw_result_name} = {callable_name}({args_call})"
            self._emitter.emit_line(line)
            self._emitter.emit_comment(
                "Inductor returns (output, saved_tensors...), extract output"
            )
            extract_line = f"{node.result_name} = {raw_result_name}[0]"
            self._emitter.emit_line(extract_line)
            return extract_line
        else:
            line = f"{node.result_name} = {callable_name}({args_call})"
            self._emitter.emit_line(line)
            return line

    def visit_kernel_load(self, node: KernelLoadNode) -> str:
        """
        Generate code to load a serialized compiled kernel.

        Produces code that loads a kernel from disk or decodes an inline
        kernel, depending on the kernel type. This enables the generated
        Python file to load and use pre-compiled Inductor kernels.

        Args:
            node: The KernelLoadNode to process

        Returns:
            The generated kernel loading code
        """
        if not self._has_emitted_kernel_section:
            self._emitter.emit_section_header("Kernel Loading")
            self._has_emitted_kernel_section = True

        if node.kernel_type == KernelType.CPP:
            return self._generate_cpp_kernel_load(node)
        elif node.kernel_type == KernelType.TRITON:
            return self._generate_triton_kernel_load(node)
        elif node.kernel_type == KernelType.PYTHON:
            return self._generate_python_kernel_load(node)
        elif node.kernel_type == KernelType.INLINE:
            return self._generate_inline_kernel_load(node)
        else:
            self._emitter.emit_comment(
                f"Unknown kernel type: {node.kernel_type.name}"
            )
            return f"# Unknown kernel type: {node.kernel_type.name}"

    def _generate_cpp_kernel_load(self, node: KernelLoadNode) -> str:
        """
        Generate code to load a C++ compiled kernel (.so file).

        The generated code uses ctypes to load the shared library and
        extract the entry point function.
        """
        self._emitter.add_import("import ctypes")
        self._emitter.add_import("from pathlib import Path")

        self._emitter.emit_comment(f"Load C++ kernel: {node.kernel_id}")
        self._emitter.emit_line(
            f'_kernel_path_{node.kernel_id} = Path(__file__).parent / "{node.kernel_path}"'
        )
        self._emitter.emit_line(
            f'if not _kernel_path_{node.kernel_id}.exists():'
        )
        self._emitter.indent()
        self._emitter.emit_line(
            f'raise FileNotFoundError(f"Kernel not found: {{_kernel_path_{node.kernel_id}}}")'
        )
        self._emitter.dedent()
        self._emitter.emit_line(
            f"_lib_{node.kernel_id} = ctypes.CDLL(str(_kernel_path_{node.kernel_id}))"
        )
        self._emitter.emit_line(
            f'{node.variable_name} = getattr(_lib_{node.kernel_id}, "{node.entry_point}")'
        )
        self._emitter.emit_blank_line()

        return f"{node.variable_name} = ctypes.CDLL(...).{node.entry_point}"

    def _generate_triton_kernel_load(self, node: KernelLoadNode) -> str:
        """
        Generate code to load a Triton GPU kernel.

        Triton kernels are loaded via the triton runtime. The generated
        code handles kernel loading and provides a callable interface.
        """
        self._emitter.add_import("import torch")

        self._emitter.emit_comment(f"Load Triton kernel: {node.kernel_id}")
        self._emitter.emit_line("try:")
        self._emitter.indent()
        self._emitter.emit_line("import triton")
        self._emitter.emit_line("import triton.language as tl")
        self._emitter.dedent()
        self._emitter.emit_line("except ImportError:")
        self._emitter.indent()
        self._emitter.emit_line(
            'raise ImportError("Triton is required to load CUDA kernels")'
        )
        self._emitter.dedent()

        if node.kernel_path:
            self._emitter.add_import("from pathlib import Path")
            self._emitter.emit_line(
                f'_triton_kernel_path_{node.kernel_id} = '
                f'Path(__file__).parent / "{node.kernel_path}"'
            )
            self._emitter.emit_comment(
                "Triton kernel would be loaded from cubin/ptx here"
            )

        kernel_hash = node.metadata.get("kernel_hash", node.kernel_id)
        self._emitter.emit_line(
            f"# Kernel hash: {kernel_hash}"
        )
        self._emitter.emit_line(
            f"{node.variable_name} = None  # Triton kernel loading placeholder"
        )
        self._emitter.emit_blank_line()

        return f"{node.variable_name} = triton_kernel(...)"

    def _generate_python_kernel_load(self, node: KernelLoadNode) -> str:
        """
        Generate code to load a Python wrapper module.

        The Python wrapper is loaded as a module and the entry point
        function is extracted.
        """
        self._emitter.add_import("import importlib.util")
        self._emitter.add_import("from pathlib import Path")

        self._emitter.emit_comment(f"Load Python wrapper: {node.kernel_id}")
        self._emitter.emit_line(
            f'_wrapper_path_{node.kernel_id} = Path(__file__).parent / "{node.kernel_path}"'
        )
        self._emitter.emit_line(
            f"_spec_{node.kernel_id} = importlib.util.spec_from_file_location("
        )
        self._emitter.indent()
        self._emitter.emit_line(f'"wrapper_{node.kernel_id}",')
        self._emitter.emit_line(f"_wrapper_path_{node.kernel_id},")
        self._emitter.dedent()
        self._emitter.emit_line(")")
        self._emitter.emit_line(
            f"_module_{node.kernel_id} = importlib.util.module_from_spec(_spec_{node.kernel_id})"
        )
        self._emitter.emit_line(
            f"_spec_{node.kernel_id}.loader.exec_module(_module_{node.kernel_id})"
        )
        self._emitter.emit_line(
            f'{node.variable_name} = getattr(_module_{node.kernel_id}, "{node.entry_point}")'
        )
        self._emitter.emit_blank_line()

        return f"{node.variable_name} = module.{node.entry_point}"

    def _generate_inline_kernel_load(self, node: KernelLoadNode) -> str:
        """
        Generate code to load an inline kernel.

        For Inductor Python source code, the code is embedded directly.
        For binary kernels, they are base64-encoded and decoded at runtime.

        This method detects if the inline content is Python source code
        (from Inductor) and embeds it directly, or base64-encodes binary
        content for decoding at runtime.
        """
        source_type = node.metadata.get("source", "")

        if source_type == "inductor":
            return self._generate_inductor_source_embed(node)

        if not node.inline_content:
            self._emitter.emit_comment("No inline content provided")
            self._emitter.emit_line(f"{node.variable_name} = None")
            self._emitter.emit_blank_line()
            return f"{node.variable_name} = None"

        return self._generate_binary_inline_kernel(node)

    def _generate_inductor_source_embed(self, node: KernelLoadNode) -> str:
        """
        Embed Inductor-generated Python source code directly.

        The Inductor source code is included as-is in the generated file,
        allowing the compiled callable to be used directly.

        This method also sets the _has_inductor_kernel flag so that the
        AOT Autograd forward method knows to use the Inductor calling convention
        (args as list, extract first output from tuple).

        Note: The `if __name__ == "__main__":` block from Inductor is stripped
        because it contains benchmark code that would execute when the generated
        file is run via exec().

        For backward kernels (is_backward=True in metadata), Triton function
        names are renamed with a `_bwd` suffix to avoid collisions with forward
        kernel functions that may have the same names.

        Error Handling:
        - If source code is None/empty, a fallback function is generated
        - If source code fails validation, a warning is emitted and fallback used
        - If the 'call' function is missing, a fallback stub is generated
        """
        is_backward = node.metadata.get("is_backward", False)

        if is_backward:
            self._has_inductor_backward_kernel = True
            self._emitter.emit_section_header("Inductor Compiled Backward Kernel")
        else:
            self._has_inductor_kernel = True
            self._emitter.emit_section_header("Inductor Compiled Kernel")

        graph_str = node.metadata.get("graph_str", "")
        if graph_str:
            preview = graph_str[:200] + "..." if len(graph_str) > 200 else graph_str
            self._emitter.emit_comment(
                "Readable graph representation (for debugging):"
            )
            for line in preview.split("\n")[:5]:
                self._emitter.emit_comment(f"  {line}")
            self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "The following code was generated by Inductor."
        )
        self._emitter.emit_comment(
            "It defines the compiled kernel as 'call' function."
        )
        self._emitter.emit_blank_line()

        source_code = node.inline_content
        if source_code:
            validation_result = self._validate_inductor_source(source_code)

            if validation_result["is_valid"]:
                filtered_source = self._strip_main_block(source_code)

                if is_backward:
                    filtered_source = self._rename_triton_functions_for_backward(
                        filtered_source
                    )

                for line in filtered_source.split("\n"):
                    self._emitter.emit_line(line)
                self._emitter.emit_blank_line()

                self._emitter.emit_comment(
                    "Reference the compiled kernel function."
                )
                self._emitter.emit_line(f"{node.variable_name} = call")
            else:
                self._emit_inductor_fallback(
                    node.variable_name,
                    validation_result["error_message"],
                    validation_result["error_type"],
                )
        else:
            self._emit_inductor_fallback(
                node.variable_name,
                "No Inductor source code provided",
                "empty_source",
            )

        self._emitter.emit_blank_line()

        return f"{node.variable_name} = call"

    def _validate_inductor_source(self, source_code: str) -> dict:
        """
        Validate that Inductor source code is well-formed and usable.

        This method checks for common issues in Inductor-generated code that
        would cause problems when the generated pythonify output is executed.

        Checks performed:
        1. Source code is not empty after stripping whitespace
        2. Source code is valid Python syntax (can be parsed by compile())
        3. Source code defines a 'call' function (Inductor's entry point)
        4. No obvious malformed patterns that would crash at runtime

        Args:
            source_code: The Inductor-generated Python source code

        Returns:
            A dict with keys:
            - is_valid: True if source code passed validation
            - error_message: Human-readable error message (if invalid)
            - error_type: Category of error (if invalid):
              - "empty_source": Source is empty/whitespace only
              - "syntax_error": Invalid Python syntax
              - "missing_call": No 'call' function defined
              - "parse_error": Unexpected error during parsing
        """
        if not source_code or not source_code.strip():
            return {
                "is_valid": False,
                "error_message": "Inductor source code is empty",
                "error_type": "empty_source",
            }

        filtered_source = self._strip_main_block(source_code)

        try:
            compile(filtered_source, "<inductor_source>", "exec")
        except SyntaxError as e:
            return {
                "is_valid": False,
                "error_message": f"Inductor source has syntax error: {e.msg} at line {e.lineno}",
                "error_type": "syntax_error",
            }
        except Exception as e:
            return {
                "is_valid": False,
                "error_message": f"Failed to parse Inductor source: {type(e).__name__}: {e}",
                "error_type": "parse_error",
            }

        if "def call(" not in filtered_source:
            return {
                "is_valid": False,
                "error_message": "Inductor source missing required 'call' function",
                "error_type": "missing_call",
            }

        return {"is_valid": True, "error_message": None, "error_type": None}

    def _emit_inductor_fallback(
        self,
        variable_name: str,
        error_message: str,
        error_type: str,
    ) -> None:
        """
        Emit fallback code when Inductor source is malformed or missing.

        Instead of crashing or producing invalid output, this method generates
        a fallback function that raises a clear error at runtime. This allows
        the pythonify output file to be generated and inspected even when
        the Inductor source code has issues.

        The generated fallback function:
        1. Raises RuntimeError with a clear message about what went wrong
        2. Includes the original error type for debugging
        3. Can be inspected in the generated file

        Args:
            variable_name: The variable name to assign the fallback to
            error_message: Human-readable description of the problem
            error_type: Category of error for programmatic handling
        """
        self._emitter.emit_comment(
            "WARNING: Inductor source code validation failed."
        )
        self._emitter.emit_comment(
            f"Error type: {error_type}"
        )
        self._emitter.emit_comment(
            f"Error: {error_message}"
        )
        self._emitter.emit_blank_line()
        self._emitter.emit_comment(
            "Generating fallback function that raises an error at runtime."
        )
        self._emitter.emit_comment(
            "This allows the file to be inspected even though execution will fail."
        )
        self._emitter.emit_blank_line()

        escaped_message = escape_string(error_message)
        self._emitter.emit_line("def call(args):")
        self._emitter.indent()
        self._emitter.emit_line(
            f'raise RuntimeError('
            f'"Pythonify fallback: {escaped_message}. '
            f'The Inductor-generated kernel could not be embedded."'
            f')'
        )
        self._emitter.dedent()
        self._emitter.emit_blank_line()
        self._emitter.emit_line(f"{variable_name} = call")

    def _strip_main_block(self, source_code: str) -> str:
        """
        Strip the `if __name__ == "__main__":` block from Inductor source.

        This block contains benchmark code that would execute during exec(),
        causing argument parsing errors and unwanted side effects.

        The function finds the start of the if __name__ block and removes
        everything from that point to the end, since the benchmark code
        is always at the end of the Inductor output.
        """
        lines = source_code.split("\n")
        result_lines = []
        in_main_block = False
        main_block_indent = None

        for line in lines:
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)

            if stripped.startswith('if __name__ == "__main__"') or \
               stripped.startswith("if __name__ == '__main__'"):
                in_main_block = True
                main_block_indent = current_indent
                continue

            if in_main_block:
                if stripped == "" or stripped.startswith("#"):
                    continue
                if current_indent > main_block_indent:
                    continue
                else:
                    in_main_block = False
                    main_block_indent = None
                    result_lines.append(line)
            else:
                result_lines.append(line)

        while result_lines and result_lines[-1].strip() == "":
            result_lines.pop()

        return "\n".join(result_lines)

    def _rename_triton_functions_for_backward(self, source_code: str) -> str:
        """
        Rename Triton function names in backward kernel to avoid collisions.

        When both forward and backward Inductor kernels define Triton functions
        with the same name (e.g., `triton_poi_fused_mul_0`), the backward kernel
        definition will overwrite the forward definition when the generated file
        is exec'd. This causes runtime errors because the forward call will use
        the backward kernel signature.

        This method adds a `_bwd` suffix to all Triton function names in the
        backward source code. The patterns that need to be renamed are:

        1. Variable assignment: `triton_xxx = async_compile.triton(...)`
        2. String argument: `async_compile.triton('triton_xxx', ...)`
        3. Function definition: `def triton_xxx(...)`
        4. Function calls: `triton_xxx.run(...)`
        5. Kernel name in metadata: `'kernel_name': 'triton_xxx'`

        Args:
            source_code: The backward kernel Inductor source code

        Returns:
            Source code with all Triton function names suffixed with `_bwd`
        """
        triton_func_names = set(re.findall(r'\bdef (triton_\w+)\(', source_code))

        if not triton_func_names:
            return source_code

        result = source_code
        for name in triton_func_names:
            new_name = f"{name}_bwd"

            result = re.sub(
                rf'^({name})\s*=\s*async_compile\.triton\(',
                rf'{new_name} = async_compile.triton(',
                result,
                flags=re.MULTILINE
            )

            result = re.sub(
                rf"async_compile\.triton\(\s*'{name}'",
                f"async_compile.triton('{new_name}'",
                result
            )

            result = re.sub(
                rf'^def {name}\(',
                f'def {new_name}(',
                result,
                flags=re.MULTILINE
            )

            result = re.sub(
                rf'\b{name}\.run\(',
                f'{new_name}.run(',
                result
            )

            result = re.sub(
                rf"'kernel_name':\s*'{name}'",
                f"'kernel_name': '{new_name}'",
                result
            )

        return result

    def _generate_binary_inline_kernel(self, node: KernelLoadNode) -> str:
        """
        Generate code to decode and load a binary inline (base64-encoded) kernel.

        Binary kernels are embedded as base64 strings, decoded at runtime,
        and loaded via ctypes.
        """
        self._emitter.add_import("import base64")
        self._emitter.add_import("import tempfile")
        self._emitter.add_import("import ctypes")

        self._emitter.emit_comment(f"Decode inline kernel: {node.kernel_id}")

        content_preview = (
            node.inline_content[:50] + "..."
            if len(node.inline_content) > 50
            else node.inline_content
        )
        self._emitter.emit_line(
            f"_kernel_b64_{node.kernel_id} = ("
        )
        self._emitter.indent()
        chunk_size = 76
        content = node.inline_content
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            self._emitter.emit_line(f'"{chunk}"')
        self._emitter.dedent()
        self._emitter.emit_line(")")
        self._emitter.emit_line(
            f"_kernel_bytes_{node.kernel_id} = base64.b64decode(_kernel_b64_{node.kernel_id})"
        )
        self._emitter.emit_line(
            f'_kernel_tmpfile_{node.kernel_id} = tempfile.NamedTemporaryFile(suffix=".so", delete=False)'
        )
        self._emitter.emit_line(
            f"_kernel_tmpfile_{node.kernel_id}.write(_kernel_bytes_{node.kernel_id})"
        )
        self._emitter.emit_line(
            f"_kernel_tmpfile_{node.kernel_id}.close()"
        )
        self._emitter.emit_line(
            f"_lib_{node.kernel_id} = ctypes.CDLL(_kernel_tmpfile_{node.kernel_id}.name)"
        )
        self._emitter.emit_line(
            f'{node.variable_name} = getattr(_lib_{node.kernel_id}, "{node.entry_point}")'
        )

        self._emitter.emit_blank_line()

        return f"{node.variable_name} = decode_inline(...)"

    def visit_return_result(self, node: ReturnResultNode) -> str:
        """
        Generate code to expose the result for exec() compatibility.

        Produces code like:
            y = result
            f_locals["y"] = y  # Makes result accessible after exec()

        This ensures the result is accessible after exec() completes. The key
        issue is that Python's frame.f_locals is read-only, so simple assignments
        in exec'd code don't persist. By writing to f_locals["y"], we make the
        result accessible in the namespace dictionary passed to exec().

        Args:
            node: The ReturnResultNode to process

        Returns:
            The generated assignment line
        """
        if not self._has_emitted_return_section:
            self._emitter.emit_section_header("Expose Result")
            self._has_emitted_return_section = True

        if node.result_name != node.expose_as:
            line = f"{node.expose_as} = {node.result_name}"
            self._emitter.emit_line(line)
        else:
            line = f"# Result is already available as '{node.expose_as}'"
            self._emitter.emit_line(line)

        if self._expose_result_in_locals:
            self._emitter.emit_comment(
                "Write result to f_locals for exec() compatibility."
            )
            self._emitter.emit_comment(
                "This is necessary because frame.f_locals is read-only in Python."
            )
            self._emitter.emit_comment(
                "Simple assignments in exec'd code don't persist to the caller's frame."
            )
            locals_line = (
                f'{self._f_locals_var_name}["{node.expose_as}"] = {node.expose_as}'
            )
            self._emitter.emit_line(locals_line)

        if self._expose_result_in_globals:
            globals_line = (
                f'{self._f_globals_var_name}["{node.expose_as}"] = {node.expose_as}'
            )
            self._emitter.emit_line(globals_line)

        return line

    def visit_compiled_region(self, node: CompiledRegionNode) -> str:
        """
        Generate code for a single compiled region.

        A compiled region is a self-contained unit of compiled code with its
        own guards, kernel loading, and invocation logic. This method generates
        a function that encapsulates the region's execution.

        The generated code structure:
            def _region_0(inputs):
                # Guard checks for this region
                assert x.shape[0] == 3, "..."

                # Execute compiled callable
                result = compiled_fn_0(inputs)
                return result

        Args:
            node: The CompiledRegionNode to process

        Returns:
            The generated function definition
        """
        self._emitter.emit_section_header(f"Compiled Region: {node.region_id}")

        input_args = ", ".join(node.input_names) if node.input_names else "*args"
        self._emitter.emit_line(f"def _{node.region_id}({input_args}):")
        self._emitter.indent()

        if node.guards:
            self._emitter.emit_comment("Region-specific guards")
            for guard in node.guards:
                self.visit_guard_check(guard)
            self._emitter.emit_blank_line()

        if node.ir is not None:
            nested_visitor = PythonCodeGenVisitor(
                model_var_name=self._model_var_name,
                f_locals_var_name=self._f_locals_var_name,
                f_globals_var_name=self._f_globals_var_name,
                compiled_fn_var_name=f"{self._compiled_fn_var_name}_{node.region_id}",
                expose_result_in_locals=False,
                expose_result_in_globals=False,
                model_source=self._model_source,
            )
            node.ir.accept_all(nested_visitor)
            nested_code = nested_visitor.get_code()
            for line in nested_code.strip().split("\n"):
                stripped = line.strip()
                if stripped:
                    self._emitter.emit_line(stripped)
        else:
            self._emitter.emit_comment(
                "Region IR not available - placeholder implementation"
            )
            self._emitter.emit_line("pass")

        self._emitter.dedent()
        self._emitter.emit_blank_line()

        if node.resume_target:
            self._emitter.emit_comment(
                f"Resume target: _{node.resume_target}"
            )

        return f"def _{node.region_id}"

    def visit_multi_region_dispatch(self, node: MultiRegionDispatchNode) -> str:
        """
        Generate code for multi-region dispatch.

        This method generates the control flow logic that executes multiple
        compiled regions. There are two modes:

        1. SEQUENTIAL: Regions execute in order, passing results between them.
           This is the pattern for graph breaks:

               result_0 = _region_0(x)
               result_1 = _region_1(result_0)
               y = result_1

        2. GUARD_DISPATCH: Guards are checked and the first matching region
           is executed:

               def _dispatch(x):
                   if _check_guards_region_0(x):
                       return _region_0(x)
                   elif _check_guards_region_1(x):
                       return _region_1(x)
                   else:
                       raise RuntimeError("No matching region")

        Args:
            node: The MultiRegionDispatchNode to process

        Returns:
            The generated dispatch code
        """
        self._emitter.emit_section_header("Multi-Region Dispatch")

        for region in node.regions:
            self.visit_compiled_region(region)

        if node.execution_mode == RegionExecutionMode.SEQUENTIAL:
            return self._generate_sequential_dispatch(node)
        else:
            return self._generate_guard_dispatch(node)

    def _generate_sequential_dispatch(self, node: MultiRegionDispatchNode) -> str:
        """
        Generate sequential execution code for multiple regions.

        In sequential mode, regions execute in order with the output of
        each region becoming the input to the next. This is the pattern
        when graph breaks occur.

        Args:
            node: The MultiRegionDispatchNode

        Returns:
            The generated sequential execution code
        """
        self._emitter.emit_comment(
            "Sequential execution: regions run in order, passing results"
        )
        self._emitter.emit_blank_line()

        result_var = "result"
        for i, region in enumerate(node.regions):
            if i == 0:
                input_expr = ", ".join(region.input_names) if region.input_names else ""
                self._emitter.emit_line(
                    f"{result_var}_{i} = _{region.region_id}({input_expr})"
                )
            else:
                self._emitter.emit_line(
                    f"{result_var}_{i} = _{region.region_id}({result_var}_{i-1})"
                )

        final_result = f"{result_var}_{len(node.regions) - 1}" if node.regions else "None"
        self._emitter.emit_line(f"result = {final_result}")

        return f"Sequential dispatch with {len(node.regions)} regions"

    def _generate_guard_dispatch(self, node: MultiRegionDispatchNode) -> str:
        """
        Generate guard-based dispatch code for multiple regions.

        In guard dispatch mode, guards are checked for each region in order,
        and the first region whose guards pass is executed.

        Args:
            node: The MultiRegionDispatchNode

        Returns:
            The generated guard dispatch code
        """
        self._emitter.emit_comment(
            "Guard-based dispatch: first matching region is executed"
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_line(f"def {node.dispatch_table_name}(*args):")
        self._emitter.indent()

        for i, region in enumerate(node.regions):
            guard_fn_name = f"_check_guards_{region.region_id}"
            self._emitter.emit_line(f"def {guard_fn_name}():")
            self._emitter.indent()
            self._emitter.emit_line("try:")
            self._emitter.indent()
            if region.guards:
                for guard in region.guards:
                    condition = self._generate_guard_condition(guard)
                    self._emitter.emit_line(f"assert {condition}")
            else:
                self._emitter.emit_line("pass  # No guards for this region")
            self._emitter.emit_line("return True")
            self._emitter.dedent()
            self._emitter.emit_line("except AssertionError:")
            self._emitter.indent()
            self._emitter.emit_line("return False")
            self._emitter.dedent()
            self._emitter.dedent()
            self._emitter.emit_blank_line()

        for i, region in enumerate(node.regions):
            guard_fn_name = f"_check_guards_{region.region_id}"
            keyword = "if" if i == 0 else "elif"
            self._emitter.emit_line(f"{keyword} {guard_fn_name}():")
            self._emitter.indent()
            input_expr = ", ".join(region.input_names) if region.input_names else "*args"
            self._emitter.emit_line(f"return _{region.region_id}({input_expr})")
            self._emitter.dedent()

        if node.fallback_to_eager:
            self._emitter.emit_line("else:")
            self._emitter.indent()
            self._emitter.emit_comment("No matching region - fallback to eager execution")
            self._emitter.emit_line(
                'raise RuntimeError("No compiled region matches current inputs - '
                'would need to fallback to eager execution")'
            )
            self._emitter.dedent()
        else:
            self._emitter.emit_line("else:")
            self._emitter.indent()
            self._emitter.emit_line(
                'raise RuntimeError("No compiled region matches current inputs")'
            )
            self._emitter.dedent()

        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment("Execute the dispatch")
        self._emitter.emit_line(f"result = {node.dispatch_table_name}(*args)")

        return f"Guard dispatch with {len(node.regions)} regions"


def generate_python_code(
    ir: RuntimeWrapperIR,
    model_var_name: str = "model",
    f_locals_var_name: str = "f_locals",
    f_globals_var_name: str = "f_globals",
    compiled_fn_var_name: str = "compiled_fn",
    include_header: bool = True,
    expose_result_in_locals: bool = True,
    expose_result_in_globals: bool = False,
    model_source: ModelSource = ModelSource.CLOSURE,
) -> str:
    """
    Generate Python source code from a RuntimeWrapperIR.

    This is the main entry point for the gen_python backend. It creates a
    PythonCodeGenVisitor, traverses the IR, and returns the generated Python
    source code.

    IMPORTANT: exec() Scoping Limitation
    -------------------------------------
    Python's exec() with separate globals and locals dicts has scoping issues
    for nested class definitions. The generated code includes an AOT Autograd
    torch.autograd.Function class that may not work correctly with:

        exec(code, frame.f_globals, frame.f_locals)  # May have scoping issues

    The RECOMMENDED pattern uses a merged namespace:

        import inspect
        frame = inspect.currentframe()
        namespace = {**frame.f_globals, **frame.f_locals}
        exec(code, namespace)
        y = namespace["y"]

    This ensures all variables are accessible and nested classes work correctly.

    Args:
        ir: The RuntimeWrapperIR to process
        model_var_name: Name of the model variable in generated code
        f_locals_var_name: Name of the f_locals dictionary variable
        f_globals_var_name: Name of the f_globals dictionary variable
        compiled_fn_var_name: Name of the compiled function variable
        include_header: Whether to include a header comment with metadata
        expose_result_in_locals: If True, write result to f_locals dict for
            exec() compatibility. This ensures the result is accessible after
            exec() completes, even though frame.f_locals is read-only.
        expose_result_in_globals: If True, also write result to f_globals dict.
        model_source: Where the model object comes from:
            - ModelSource.CLOSURE: Model is directly in scope (passed to exec globals)
            - ModelSource.F_LOCALS: Model is in f_locals["model"] (local variable)
            - ModelSource.F_GLOBALS: Model is in f_globals["model"] (global variable)

    Returns:
        Generated Python source code as a string

    Example:
        artifacts = CompilationArtifacts(...)
        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        # RECOMMENDED: Use merged namespace for proper scoping
        frame = inspect.currentframe()
        namespace = {**frame.f_globals, **frame.f_locals}
        exec(code, namespace)
        y = namespace["y"]

        # Alternative: Pass model in globals explicitly
        code = generate_python_code(ir, model_source=ModelSource.CLOSURE)
        exec(code, {"model": model, "compiled_fn": fn, "f_locals": f_locals}, {})
    """
    visitor = PythonCodeGenVisitor(
        model_var_name=model_var_name,
        f_locals_var_name=f_locals_var_name,
        f_globals_var_name=f_globals_var_name,
        compiled_fn_var_name=compiled_fn_var_name,
        expose_result_in_locals=expose_result_in_locals,
        expose_result_in_globals=expose_result_in_globals,
        model_source=model_source,
    )

    emitter = visitor.get_emitter()

    visitor.prescan_ir(ir)
    needs_model = visitor.requires_model_access(ir)

    if include_header:
        emitter.emit_comment("Generated by torch.compile(pythonify=...)")
        emitter.emit_comment("This file contains the runtime machinery for the compiled model.")
        emitter.emit_comment("")
        emitter.emit_comment("Variables expected in scope:")
        if needs_model:
            if model_source == ModelSource.CLOSURE:
                emitter.emit_comment(f"  - {model_var_name}: The model object (directly accessible)")
            elif model_source == ModelSource.F_LOCALS:
                emitter.emit_comment(f"  - {f_locals_var_name}[\"{model_var_name}\"]: The model object (from f_locals)")
            else:
                emitter.emit_comment(f"  - {f_globals_var_name}[\"{model_var_name}\"]: The model object (from f_globals)")
        emitter.emit_comment(f"  - {f_locals_var_name}: Frame locals dictionary")
        emitter.emit_comment(f"  - {compiled_fn_var_name}: The compiled callable")

    ir.accept_all(visitor)

    return visitor.get_code()


def write_python_file(
    ir: RuntimeWrapperIR,
    file_path: str,
    model_var_name: str = "model",
    f_locals_var_name: str = "f_locals",
    f_globals_var_name: str = "f_globals",
    compiled_fn_var_name: str = "compiled_fn",
) -> None:
    """
    Generate Python code and write it to a file.

    This is a convenience function that combines generate_python_code()
    with file writing.

    Args:
        ir: The RuntimeWrapperIR to process
        file_path: Path to the output file
        model_var_name: Name of the model variable in generated code
        f_locals_var_name: Name of the f_locals dictionary variable
        f_globals_var_name: Name of the f_globals dictionary variable
        compiled_fn_var_name: Name of the compiled function variable

    Raises:
        OSError: If the file cannot be written
    """
    code = generate_python_code(
        ir,
        model_var_name=model_var_name,
        f_locals_var_name=f_locals_var_name,
        f_globals_var_name=f_globals_var_name,
        compiled_fn_var_name=compiled_fn_var_name,
        include_header=True,
    )

    with open(file_path, "w") as f:
        f.write(code)
