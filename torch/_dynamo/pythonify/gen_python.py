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
    AOTDedupeWrapperNode,
    AOTDispatchSubclassWrapperNode,
    AOTSyntheticBaseWrapperNode,
    ArgumentExtractionNode,
    ArgumentSource,
    AOTAutogradWrapperNode,
    CallableInvocationNode,
    CodeGenVisitor,
    CompiledRegionNode,
    CUDAGraphPhase,
    CUDAGraphSetupNode,
    DebugAssertWrapperNode,
    EffectTokensWrapperNode,
    FakifiedOutWrapperNode,
    FunctionalizedRngRuntimeWrapperNode,
    GuardCheckNode,
    GuardType,
    KernelLoadNode,
    KernelType,
    ModelSource,
    MultiRegionDispatchNode,
    RegionExecutionMode,
    ReturnResultNode,
    RuntimeWrapperIR,
    RuntimeWrapperNode,
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
        emit_debug_prints: bool = False,
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
            emit_debug_prints: If True, emit debug print statements at key points
                in the generated code. This helps trace execution flow, debug
                argument values, and diagnose issues like the "not enough values
                to unpack" error. Debug prints are emitted at:
                - Argument extraction (shows each extracted value)
                - AOT Autograd forward/backward entry/exit
                - Compiled function invocation (shows args and result)
                - Result exposure (shows final output)
        """
        self._emitter = CodeEmitter()
        self._model_var_name = model_var_name
        self._f_locals_var_name = f_locals_var_name
        self._f_globals_var_name = f_globals_var_name
        self._compiled_fn_var_name = compiled_fn_var_name
        self._expose_result_in_locals = expose_result_in_locals
        self._expose_result_in_globals = expose_result_in_globals
        self._model_source = model_source
        self._emit_debug_prints = emit_debug_prints
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
        self._has_emitted_wrapper_section = False
        self._has_inductor_kernel = False
        self._has_inductor_backward_kernel = False
        self._inductor_uses_list_args = True
        self._inductor_returns_tuple = True
        self._cuda_graph_replay_fn: Optional[str] = None
        self._cuda_graph_static_outputs_var: Optional[str] = None
        self._forward_cuda_graph_node: Optional[CUDAGraphSetupNode] = None
        self._backward_cuda_graph_node: Optional[CUDAGraphSetupNode] = None
        self._inference_cuda_graph_node: Optional[CUDAGraphSetupNode] = None
        # Wrapper state tracking
        self._effect_token_count: int = 0
        self._has_subclass_wrapper: bool = False
        self._has_rng_wrapper: bool = False
        self._rng_num_outputs_offset: int = 0
        self._rng_num_forward_returns: int = 0
        self._has_fakified_wrapper: bool = False
        self._has_runtime_wrapper: bool = False
        self._indices_to_detach: list[int] = []
        self._has_dedupe_wrapper: bool = False
        self._dedupe_keep_arg_mask: Optional[list[bool]] = None
        self._dedupe_add_dupe_map: Optional[list[int]] = None
        self._has_synthetic_base_wrapper: bool = False
        self._has_debug_assert_wrapper: bool = False

    @property
    def _has_forward_wrappers(self) -> bool:
        """
        Check if any forward wrappers are active and need to process args.

        Returns True if any wrapper helper function was emitted that needs to
        be called during forward(). This is used to determine whether to use
        _processed_args vs args in the compiled function call.
        """
        return (
            self._effect_token_count > 0 or
            self._has_subclass_wrapper or
            self._has_rng_wrapper or
            self._has_runtime_wrapper or
            self._has_dedupe_wrapper or
            self._has_synthetic_base_wrapper or
            self._has_debug_assert_wrapper
        )

    @property
    def _has_forward_output_wrappers(self) -> bool:
        """
        Check if any forward wrappers need to post-process outputs.

        Returns True if any wrapper helper function was emitted that transforms
        outputs after the compiled function call.
        """
        return (
            self._effect_token_count > 0 or
            self._has_subclass_wrapper or
            self._has_rng_wrapper or
            self._has_fakified_wrapper or
            self._has_synthetic_base_wrapper
        )

    @property
    def cuda_graphs_enabled(self) -> bool:
        """
        Check if CUDA graphs are enabled in the IR being processed.

        This property returns True if any CUDAGraphSetupNode was detected during
        prescan_ir(). This allows code generation logic to make decisions based
        on whether CUDA graphs will be used, such as choosing between direct
        callable invocation vs. graph replay.

        The property checks for CUDA graph nodes in all phases:
        - INFERENCE: Single graph for inference-only computation
        - FORWARD: Forward pass graph for training mode
        - BACKWARD: Backward pass graph for training mode

        Returns:
            True if CUDA graphs are enabled, False otherwise
        """
        return (
            self._inference_cuda_graph_node is not None
            or self._forward_cuda_graph_node is not None
            or self._backward_cuda_graph_node is not None
        )

    def get_cuda_graph_node(
        self, phase: Optional["CUDAGraphPhase"] = None
    ) -> Optional[CUDAGraphSetupNode]:
        """
        Get the CUDAGraphSetupNode for a specific phase, or any phase.

        This method retrieves the CUDA graph node detected during prescan_ir().
        It's useful for code generation logic that needs to access CUDA graph
        configuration when deciding how to emit code.

        Args:
            phase: The specific CUDAGraphPhase to retrieve. If None, returns
                the inference node if present, otherwise the forward node if
                present, otherwise the backward node if present.

        Returns:
            The CUDAGraphSetupNode for the requested phase, or None if not found.
        """
        if phase == CUDAGraphPhase.INFERENCE:
            return self._inference_cuda_graph_node
        elif phase == CUDAGraphPhase.FORWARD:
            return self._forward_cuda_graph_node
        elif phase == CUDAGraphPhase.BACKWARD:
            return self._backward_cuda_graph_node
        else:
            return (
                self._inference_cuda_graph_node
                or self._forward_cuda_graph_node
                or self._backward_cuda_graph_node
            )

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

    def _emit_debug_print(self, message: str, *values: str) -> None:
        """
        Emit a debug print statement if debug mode is enabled.

        This method generates print() calls that help trace execution flow
        through the generated code. The prints show key values at each stage
        of execution, making it easier to diagnose issues like the "not enough
        values to unpack" error.

        Args:
            message: A descriptive message for the debug output
            *values: Optional variable names whose values should be printed.
                These are formatted as f-string expressions in the generated code.

        Example:
            _emit_debug_print("Extracted arg1", "arg1", "arg1.shape")
            # Generates: print("[DEBUG] Extracted arg1: arg1=", arg1, "arg1.shape=", arg1.shape)
        """
        if not self._emit_debug_prints:
            return

        if values:
            parts = []
            for val in values:
                parts.append(f'"{val}=", {val}')
            values_str = ", ".join(parts)
            self._emitter.emit_line(f'print("[DEBUG] {message}:", {values_str})')
        else:
            self._emitter.emit_line(f'print("[DEBUG] {message}")')

    def prescan_ir(self, ir: "RuntimeWrapperIR") -> None:
        """
        Pre-scan the IR to detect features that affect code generation.

        This method scans all nodes in the IR before visiting them to detect
        features like Inductor kernels that need to be known before generating
        code for earlier nodes (e.g., AOT Autograd forward method needs to know
        if there's an Inductor kernel to use the correct calling convention).

        Detects both forward and backward Inductor kernels for proper autograd
        function generation.

        Detects CUDAGraphSetupNodes in all phases:
        - INFERENCE: Single graph for inference-only computation. Tracked in
          _inference_cuda_graph_node for use when generating code that needs
          to know whether CUDA graphs are enabled.
        - FORWARD: Forward pass graph for training mode. Tracked in
          _forward_cuda_graph_node for AOT Autograd wrapper integration.
        - BACKWARD: Backward pass graph for training mode. Tracked in
          _backward_cuda_graph_node for AOT Autograd wrapper integration.

        After calling prescan_ir(), use the cuda_graphs_enabled property to
        check if any CUDA graph nodes were found, or get_cuda_graph_node() to
        retrieve a specific node.

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
            elif isinstance(node, CUDAGraphSetupNode):
                if node.phase == CUDAGraphPhase.INFERENCE:
                    self._inference_cuda_graph_node = node
                elif node.phase == CUDAGraphPhase.FORWARD:
                    self._forward_cuda_graph_node = node
                elif node.phase == CUDAGraphPhase.BACKWARD:
                    self._backward_cuda_graph_node = node

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

    def uses_object_id_fallback(self, ir: "RuntimeWrapperIR") -> bool:
        """
        Check if the IR contains any nodes that use OBJECT_ID source (fallback path).

        The golden path uses PARAMETER/BUFFER sources with nested_path to access
        model attributes directly (e.g., model.layer.weight). The fallback path
        uses OBJECT_ID source with ctypes-based obj_from_id() to retrieve tensors
        by their memory address.

        This method is used to determine which documentation comments to include
        in the generated code header. When using the golden path, we show cleaner
        documentation about module-based access. When using the fallback, we show
        warnings about process-local object IDs and their limitations.

        Args:
            ir: The RuntimeWrapperIR to scan

        Returns:
            True if any node uses OBJECT_ID source, False otherwise
        """
        for node in ir.nodes:
            if isinstance(node, ArgumentExtractionNode):
                if node.source == ArgumentSource.OBJECT_ID and node.object_id is not None:
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

        # Emit legacy fallback warning comment for OBJECT_ID source.
        # OBJECT_ID uses ctypes to retrieve objects by memory address, which is
        # only safe within the same process while the original objects are alive.
        # The preferred approach for nn.Module is module-based access (PARAMETER/BUFFER).
        if node.source == ArgumentSource.OBJECT_ID and node.object_id is not None:
            self._emitter.emit_comment(
                "LEGACY FALLBACK: Using object ID because module context unavailable"
            )

        self._emitter.emit_line(line)
        self._emit_debug_print(
            f"Extracted {node.name}", node.name, f"{node.name}.shape"
        )
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
            "LEGACY FALLBACK: obj_from_id helper for retrieving Python objects by memory ID."
        )
        self._emitter.emit_comment(
            "This ctypes-based approach is a fallback when module-based access is unavailable."
        )
        self._emitter.emit_comment(
            "For nn.Module, prefer the golden path: access parameters via model.attr traversal."
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

    # Wrapper nodes currently no-op in Python codegen; populated in pipeline ordering
    # and handled within AOTAutograd/Python runtime assembly. Kept for visitor parity.

    def visit_effect_tokens_wrapper(self, node: EffectTokensWrapperNode) -> str:
        """
        Generate code for effect tokens wrapper.

        Effect tokens are prepended to args (as None values) before calling the
        compiled function, and then stripped from the beginning of outputs.
        This mirrors EffectTokensWrapper from runtime_wrappers.py.

        When token_count > 0:
        - Before call: args = [None] * token_count + args
        - After call: outputs = outputs[token_count:]
        """
        if node.token_count <= 0:
            return ""

        if not self._has_emitted_wrapper_section:
            self._emitter.emit_section_header("Wrapper Helper Functions")
            self._has_emitted_wrapper_section = True

        self._effect_token_count = node.token_count

        self._emitter.emit_comment(
            f"Effect tokens wrapper: prepend {node.token_count} None tokens to args,"
        )
        self._emitter.emit_comment(
            "then strip them from the beginning of outputs."
        )
        self._emitter.emit_line(f"_effect_token_count = {node.token_count}")
        self._emitter.emit_blank_line()

        self._emitter.emit_line("def _inject_effect_tokens(args):")
        self._emitter.indent()
        self._emitter.emit_line("return [None] * _effect_token_count + list(args)")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("def _strip_effect_tokens(outputs):")
        self._emitter.indent()
        self._emitter.emit_line("if outputs is None:")
        self._emitter.indent()
        self._emitter.emit_line("return None")
        self._emitter.dedent()
        self._emitter.emit_line("return outputs[_effect_token_count:]")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        return f"_effect_token_count = {node.token_count}"

    def visit_aot_dispatch_subclass_wrapper(
        self, node: AOTDispatchSubclassWrapperNode
    ) -> str:
        """
        Generate code for AOT dispatch subclass wrapper.

        This wrapper handles tensor subclass unwrapping/rewrapping. For most
        standard cases (no subclass metadata), this is a no-op. When subclass
        metadata is present, we need to emit code that:
        1. Unwraps tensor subclasses from inputs before the call
        2. Re-wraps outputs as the appropriate subclass type after the call

        The generated code mirrors runtime_wrappers.py AOTDispatchSubclassWrapper:
        - _unwrap_tensor_subclasses: Flattens subclass inputs to inner tensors
        - _wrap_tensor_subclasses: Reconstructs subclass outputs from flat tensors
        """
        if node.maybe_subclass_meta is None:
            return ""

        if not self._has_emitted_wrapper_section:
            self._emitter.emit_section_header("Wrapper Helper Functions")
            self._has_emitted_wrapper_section = True

        self._has_subclass_wrapper = True

        self._emitter.add_import("import torch")
        self._emitter.add_import(
            "from torch.utils._python_dispatch import is_traceable_wrapper_subclass"
        )

        self._emitter.emit_comment(
            "AOT dispatch subclass wrapper: handles tensor subclass unwrapping/rewrapping."
        )
        self._emitter.emit_comment(
            "Unwraps subclass inputs to inner tensors before calling compiled fn,"
        )
        self._emitter.emit_comment(
            "then rewraps outputs back to the appropriate subclass types."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_line(
            f"_num_fw_outs_saved_for_bw = {node.num_fw_outs_saved_for_bw}"
        )
        self._emitter.emit_blank_line()

        self._emit_subclass_unwrap_function(node)
        self._emit_subclass_wrap_function(node)

        return ""

    def _emit_subclass_unwrap_function(
        self, node: AOTDispatchSubclassWrapperNode
    ) -> None:
        """
        Emit the _unwrap_tensor_subclasses function.

        This function takes a list of args (which may contain tensor subclasses)
        and returns a flat list of inner tensors. For each subclass, it uses
        __tensor_flatten__ to extract the inner tensors and appends symints
        for dynamic shapes.
        """
        self._emitter.emit_comment(
            "Unwrap tensor subclasses to their inner tensors."
        )
        self._emitter.emit_comment(
            "Uses __tensor_flatten__ to extract inner tensors from subclasses."
        )
        self._emitter.emit_comment(
            "Also appends symbolic shape info (symints) when available."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_line("def _unwrap_tensor_subclasses(args, subclass_metas=None):")
        self._emitter.indent()

        self._emitter.emit_line("def _flatten_subclass(x, meta, out):")
        self._emitter.indent()
        self._emitter.emit_comment(
            "If not a subclass, append directly and return."
        )
        self._emitter.emit_line("if not is_traceable_wrapper_subclass(x):")
        self._emitter.indent()
        self._emitter.emit_line("out.append(x)")
        self._emitter.emit_line("return out")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Get attrs from __tensor_flatten__ and recursively flatten."
        )
        self._emitter.emit_line("attrs, _ = x.__tensor_flatten__()")
        self._emitter.emit_line("for attr in attrs:")
        self._emitter.indent()
        self._emitter.emit_line("inner = getattr(x, attr)")
        self._emitter.emit_line("inner_meta = meta.get(attr) if meta else None")
        self._emitter.emit_line("_flatten_subclass(inner, inner_meta, out)")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Append symints for outer_size/outer_stride if present in meta."
        )
        self._emitter.emit_line("if meta is not None and hasattr(meta, 'outer_size'):")
        self._emitter.indent()
        self._emitter.emit_line("size = x.size()")
        self._emitter.emit_line("stride = x.stride()")
        self._emitter.emit_line("for s in size:")
        self._emitter.indent()
        self._emitter.emit_line("if isinstance(s, torch.SymInt):")
        self._emitter.indent()
        self._emitter.emit_line("out.append(s)")
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.emit_line("for s in stride:")
        self._emitter.indent()
        self._emitter.emit_line("if isinstance(s, torch.SymInt):")
        self._emitter.indent()
        self._emitter.emit_line("out.append(s)")
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.emit_line("return out")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("unwrapped = []")
        self._emitter.emit_line("for i, arg in enumerate(args):")
        self._emitter.indent()
        self._emitter.emit_line("if isinstance(arg, torch.Tensor):")
        self._emitter.indent()
        self._emitter.emit_line(
            "meta = subclass_metas[i] if subclass_metas and i < len(subclass_metas) else None"
        )
        self._emitter.emit_line("if hasattr(meta, 'attrs'):")
        self._emitter.indent()
        self._emitter.emit_line("_flatten_subclass(arg, meta.attrs, unwrapped)")
        self._emitter.dedent()
        self._emitter.emit_line("else:")
        self._emitter.indent()
        self._emitter.emit_line("_flatten_subclass(arg, None, unwrapped)")
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.emit_line("else:")
        self._emitter.indent()
        self._emitter.emit_line("unwrapped.append(arg)")
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.emit_line("return unwrapped")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

    def _emit_subclass_wrap_function(
        self, node: AOTDispatchSubclassWrapperNode
    ) -> None:
        """
        Emit the _wrap_tensor_subclasses function.

        This function takes a flat list of outputs and reconstructs
        tensor subclasses using __tensor_unflatten__. It uses the
        stored subclass creation metadata to determine class types
        and reconstruction parameters.
        """
        self._emitter.emit_comment(
            "Wrap flat tensor outputs back into tensor subclasses."
        )
        self._emitter.emit_comment(
            "Uses __tensor_unflatten__ to reconstruct subclasses from inner tensors."
        )
        self._emitter.emit_comment(
            "Handles both PlainTensorMeta (pass-through) and SubclassCreationMeta."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_line(
            "def _wrap_tensor_subclasses(unwrapped_outputs, subclass_metas, "
            "num_fw_outs_saved=None):"
        )
        self._emitter.indent()
        self._emitter.emit_line("if subclass_metas is None:")
        self._emitter.indent()
        self._emitter.emit_line("return unwrapped_outputs")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("wrapped = []")
        self._emitter.emit_line("idx = 0")
        self._emitter.emit_blank_line()

        self._emitter.emit_line("for meta in subclass_metas:")
        self._emitter.indent()

        self._emitter.emit_comment(
            "PlainTensorMeta: just copy the tensor at unwrapped_idx."
        )
        self._emitter.emit_line("if hasattr(meta, 'unwrapped_idx'):")
        self._emitter.indent()
        self._emitter.emit_line("wrapped.append(unwrapped_outputs[meta.unwrapped_idx])")
        self._emitter.emit_line("idx += 1")
        self._emitter.emit_line("continue")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "SubclassCreationMeta: use creation_fn to reconstruct."
        )
        self._emitter.emit_line("if hasattr(meta, 'creation_fn'):")
        self._emitter.indent()
        self._emitter.emit_line("subclass = meta.creation_fn(unwrapped_outputs, is_runtime=True)")
        self._emitter.emit_line("wrapped.append(subclass)")
        self._emitter.emit_line("idx += meta.arg_count if hasattr(meta, 'arg_count') else 1")
        self._emitter.emit_line("continue")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Fallback: just append the current output."
        )
        self._emitter.emit_line("if idx < len(unwrapped_outputs):")
        self._emitter.indent()
        self._emitter.emit_line("wrapped.append(unwrapped_outputs[idx])")
        self._emitter.emit_line("idx += 1")
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Append remaining activations saved for backward pass."
        )
        self._emitter.emit_line("if num_fw_outs_saved is not None and idx < len(unwrapped_outputs):")
        self._emitter.indent()
        self._emitter.emit_line("wrapped.extend(unwrapped_outputs[idx:])")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("return tuple(wrapped)")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

    def visit_functionalized_rng_runtime_wrapper(
        self, node: FunctionalizedRngRuntimeWrapperNode
    ) -> str:
        """
        Generate code for RNG functionalization wrapper.

        When is_rng_op_functionalized is True, this wrapper:
        1. Appends CUDA RNG state (seed, offset) to the input args
        2. Extracts the new RNG offset from outputs and updates CUDA RNG state
        3. Strips the RNG offset from returned outputs

        This mirrors FunctionalizedRngRuntimeWrapper from runtime_wrappers.py.
        """
        if not node.is_rng_op_functionalized:
            return ""

        if not self._has_emitted_wrapper_section:
            self._emitter.emit_section_header("Wrapper Helper Functions")
            self._has_emitted_wrapper_section = True

        self._has_rng_wrapper = True
        self._rng_num_outputs_offset = node.num_outputs_rng_offset
        self._rng_num_forward_returns = node.num_forward_returns

        self._emitter.emit_comment(
            "RNG functionalization wrapper: plumbs CUDA RNG state through compiled function."
        )
        self._emitter.emit_comment(
            "Appends (seed, offset) to args, extracts new offset from outputs."
        )
        self._emitter.emit_line(
            f"_rng_num_outputs_offset = {node.num_outputs_rng_offset}"
        )
        self._emitter.emit_line(
            f"_rng_num_forward_returns = {node.num_forward_returns}"
        )
        self._emitter.emit_blank_line()

        self._emitter.add_import("import torch")

        self._emitter.emit_line("def _append_rng_state(args):")
        self._emitter.indent()
        self._emitter.emit_comment(
            "Get current CUDA RNG state (seed, offset) and append to args."
        )
        self._emitter.emit_line(
            "rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None"
        )
        self._emitter.emit_line("if rng_state is not None:")
        self._emitter.indent()
        self._emitter.emit_line(
            "seed_offset = (torch.cuda.initial_seed(), 0)"
        )
        self._emitter.emit_line("return list(args) + list(seed_offset)")
        self._emitter.dedent()
        self._emitter.emit_line("return args")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("def _handle_rng_outputs(outputs):")
        self._emitter.indent()
        self._emitter.emit_comment(
            "Extract RNG offset from outputs and update CUDA RNG state."
        )
        self._emitter.emit_line("if _rng_num_outputs_offset > 0 and outputs is not None:")
        self._emitter.indent()
        self._emitter.emit_comment("Last _rng_num_outputs_offset elements are RNG offsets.")
        self._emitter.emit_line("return outputs[:-_rng_num_outputs_offset]")
        self._emitter.dedent()
        self._emitter.emit_line("return outputs")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        return ""

    def visit_fakified_out_wrapper(self, node: FakifiedOutWrapperNode) -> str:
        """
        Generate code for fakified output wrapper.

        This wrapper re-fakifies outputs using traced metadata and reported strides.
        When fwd_output_strides is provided, outputs whose strides differ from
        expected are corrected using as_strided. This mirrors FakifiedOutWrapper
        from runtime_wrappers.py.

        The generated _fix_output_strides function:
        1. Takes outputs and the expected strides list
        2. For each tensor output, checks if strides match expected
        3. If strides differ, uses as_strided to fix the layout
        4. Returns the corrected outputs

        This is important for layout-sensitive operations where Inductor may
        produce outputs with different strides than what was traced.
        """
        if node.out_metas is None:
            return ""

        if not self._has_emitted_wrapper_section:
            self._emitter.emit_section_header("Wrapper Helper Functions")
            self._has_emitted_wrapper_section = True

        self._has_fakified_wrapper = True

        self._emitter.add_import("import torch")

        self._emitter.emit_comment(
            "Fakified output wrapper: re-fakifies outputs using traced metadata."
        )
        self._emitter.emit_comment(
            "Output strides are corrected to match those recorded during tracing."
        )
        self._emitter.emit_comment(
            "This is needed when Inductor produces outputs with different strides."
        )
        self._emitter.emit_blank_line()

        if node.fwd_output_strides is not None:
            strides_repr = repr(node.fwd_output_strides)
            self._emitter.emit_line(f"_fwd_output_strides = {strides_repr}")
            self._emitter.emit_blank_line()

            self._emitter.emit_line("def _fix_output_strides(outputs, expected_strides=None):")
            self._emitter.indent()
            self._emitter.emit_line("if expected_strides is None:")
            self._emitter.indent()
            self._emitter.emit_line("expected_strides = _fwd_output_strides")
            self._emitter.dedent()
            self._emitter.emit_line("if outputs is None or not expected_strides:")
            self._emitter.indent()
            self._emitter.emit_line("return outputs")
            self._emitter.dedent()

            self._emitter.emit_comment(
                "Convert single output to list for uniform handling."
            )
            self._emitter.emit_line("is_single = not isinstance(outputs, (list, tuple))")
            self._emitter.emit_line("if is_single:")
            self._emitter.indent()
            self._emitter.emit_line("outputs = [outputs]")
            self._emitter.dedent()
            self._emitter.emit_blank_line()

            self._emitter.emit_line("result = list(outputs)")
            self._emitter.emit_line("for i in range(min(len(result), len(expected_strides))):")
            self._emitter.indent()
            self._emitter.emit_line("if not isinstance(result[i], torch.Tensor):")
            self._emitter.indent()
            self._emitter.emit_line("continue")
            self._emitter.dedent()
            self._emitter.emit_line("strides = expected_strides[i]")
            self._emitter.emit_comment(
                "None strides mean Inductor couldn't compute expected strides,"
            )
            self._emitter.emit_comment(
                "so we skip correction for this output."
            )
            self._emitter.emit_line("if strides is None:")
            self._emitter.indent()
            self._emitter.emit_line("continue")
            self._emitter.dedent()

            self._emitter.emit_comment(
                "Check if actual strides match expected strides."
            )
            self._emitter.emit_line("actual_strides = result[i].stride()")
            self._emitter.emit_line("if len(actual_strides) != len(strides):")
            self._emitter.indent()
            self._emitter.emit_line("continue")
            self._emitter.dedent()
            self._emitter.emit_line(
                "if all(a == e for a, e in zip(actual_strides, strides)):"
            )
            self._emitter.indent()
            self._emitter.emit_line("continue")
            self._emitter.dedent()

            self._emitter.emit_comment(
                "Strides differ - use as_strided to correct the layout."
            )
            self._emitter.emit_line(
                "result[i] = result[i].as_strided(result[i].shape, strides)"
            )
            self._emitter.dedent()
            self._emitter.emit_blank_line()

            self._emitter.emit_line("if is_single:")
            self._emitter.indent()
            self._emitter.emit_line("return result[0]")
            self._emitter.dedent()
            self._emitter.emit_line("return result")
            self._emitter.dedent()
            self._emitter.emit_blank_line()

        return ""

    def visit_runtime_wrapper(self, node: RuntimeWrapperNode) -> str:
        """
        Generate code for runtime wrapper (epilogue handling).

        This wrapper handles:
        1. Detaching inputs at specified indices (indices_of_inps_to_detach)
        2. Restoring autocast/grad context if needed
        3. Alias/mutation epilogue for outputs

        This mirrors RuntimeWrapper from runtime_wrappers.py.
        """
        if not node.indices_of_inps_to_detach and not node.disable_amp:
            return ""

        if not self._has_emitted_wrapper_section:
            self._emitter.emit_section_header("Wrapper Helper Functions")
            self._has_emitted_wrapper_section = True

        self._has_runtime_wrapper = True
        self._indices_to_detach = node.indices_of_inps_to_detach

        if node.indices_of_inps_to_detach:
            self._emitter.emit_comment(
                "Runtime wrapper: detaches inputs at specified indices."
            )
            self._emitter.emit_comment(
                "This prevents gradients from flowing back through these inputs."
            )
            indices_repr = repr(node.indices_of_inps_to_detach)
            self._emitter.emit_line(f"_indices_to_detach = {indices_repr}")
            self._emitter.emit_blank_line()

            self._emitter.emit_line("def _detach_inputs(args):")
            self._emitter.indent()
            self._emitter.emit_line("args = list(args)")
            self._emitter.emit_line("for idx in _indices_to_detach:")
            self._emitter.indent()
            self._emitter.emit_line(
                "if hasattr(args[idx], 'detach'):"
            )
            self._emitter.indent()
            self._emitter.emit_line(
                "args[idx] = args[idx].detach()"
            )
            self._emitter.dedent()
            self._emitter.dedent()
            self._emitter.emit_line("return args")
            self._emitter.dedent()
            self._emitter.emit_blank_line()

        return ""

    def visit_aot_dedupe_wrapper(self, node: AOTDedupeWrapperNode) -> str:
        """
        Generate code for AOT dedupe wrapper.

        This wrapper handles duplicate argument removal and reinsertion:
        1. Pre-call: Remove duplicate args using keep_arg_mask
        2. Post-call: Reinsert duplicates using add_dupe_map (for grad outputs)

        This mirrors AOTDedupeWrapper from runtime_wrappers.py.
        """
        if not node.needs_post_compile or not node.keep_arg_mask:
            return ""

        if not self._has_emitted_wrapper_section:
            self._emitter.emit_section_header("Wrapper Helper Functions")
            self._has_emitted_wrapper_section = True

        self._has_dedupe_wrapper = True
        self._dedupe_keep_arg_mask = node.keep_arg_mask
        self._dedupe_add_dupe_map = node.add_dupe_map

        self._emitter.emit_comment(
            "AOT dedupe wrapper: handles duplicate argument removal/reinsertion."
        )
        self._emitter.emit_comment(
            "Duplicate args are removed before calling compiled fn, then"
        )
        self._emitter.emit_comment(
            "gradients are reconstructed using add_dupe_map on backward."
        )

        keep_mask_repr = repr(node.keep_arg_mask)
        add_map_repr = repr(node.add_dupe_map) if node.add_dupe_map else "None"

        self._emitter.emit_line(f"_dedupe_keep_arg_mask = {keep_mask_repr}")
        self._emitter.emit_line(f"_dedupe_add_dupe_map = {add_map_repr}")
        self._emitter.emit_blank_line()

        self._emitter.emit_line("def _remove_dupe_args(args):")
        self._emitter.indent()
        self._emitter.emit_comment(
            "Remove duplicate arguments based on keep_arg_mask."
        )
        self._emitter.emit_line(
            "return [t for t, keep in zip(args, _dedupe_keep_arg_mask) if keep]"
        )
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        if node.add_dupe_map:
            self._emitter.emit_line("def _add_dupe_args(args):")
            self._emitter.indent()
            self._emitter.emit_comment(
                "Reinsert duplicated arguments using add_dupe_map."
            )
            self._emitter.emit_line(
                "return [args[i] for i in _dedupe_add_dupe_map]"
            )
            self._emitter.dedent()
            self._emitter.emit_blank_line()

        return ""

    def visit_aot_synthetic_base_wrapper(
        self, node: AOTSyntheticBaseWrapperNode
    ) -> str:
        """
        Generate code for AOT synthetic base wrapper.

        This wrapper handles synthetic bases for aliased inputs:
        1. Pre-call: Merge aliased inputs into synthetic bases using merge_view_inputs
        2. Post-call: Unpack synthetic bases to original views
        3. Post-call: Apply metadata mutations using as_strided_ for aliased inputs

        The generated code mirrors runtime_wrappers.py AOTSyntheticBaseWrapper:
        - _merge_aliased_inputs_to_synthetic_bases: Merges aliased inputs at runtime
        - _unpack_synthetic_bases: Reconstructs original views from synthetic bases
        - _apply_metadata_mutations: Applies as_strided_ for metadata mutations
        """
        if not node.needs_post_compile or node.synthetic_base_info is None:
            return ""

        if not self._has_emitted_wrapper_section:
            self._emitter.emit_section_header("Wrapper Helper Functions")
            self._has_emitted_wrapper_section = True

        self._has_synthetic_base_wrapper = True

        self._emitter.add_import("import torch")
        self._emitter.add_import("from torch import Tensor")
        self._emitter.add_import(
            "from torch._functorch._aot_autograd.runtime_wrappers import merge_view_inputs"
        )
        self._emitter.add_import(
            "from torch._functorch._aot_autograd.functional_utils import gen_alias_from_base"
        )
        self._emitter.add_import("from torch._functorch import config as functorch_config")

        self._emitter.emit_comment(
            "AOT synthetic base wrapper: handles aliased inputs via synthetic bases."
        )
        self._emitter.emit_comment(
            "When inputs alias each other and one is mutated, they must share a"
        )
        self._emitter.emit_comment(
            "synthetic base so mutations are visible to all aliases."
        )
        self._emitter.emit_blank_line()

        self._emit_synthetic_base_metadata(node)
        self._emit_merge_aliased_inputs_function(node)
        self._emit_unpack_synthetic_bases_function(node)
        self._emit_apply_metadata_mutations_function(node)

        return ""

    def _emit_synthetic_base_metadata(
        self, node: AOTSyntheticBaseWrapperNode
    ) -> None:
        """Emit metadata variables for synthetic base wrapper."""
        aliased_idx = node.aliased_arg_idx_with_metadata_mutations or []
        aliased_idx_repr = repr(aliased_idx)
        self._emitter.emit_line(
            f"_aliased_arg_idx_with_metadata_mutations = {aliased_idx_repr}"
        )
        is_inference = not node.trace_joint
        self._emitter.emit_line(f"_synthetic_base_is_inference = {is_inference}")
        self._emitter.emit_blank_line()

    def _emit_merge_aliased_inputs_function(
        self, node: AOTSyntheticBaseWrapperNode
    ) -> None:
        """
        Emit the _merge_aliased_inputs_to_synthetic_bases function.

        This function takes the original args and merges aliased inputs into
        synthetic bases at runtime using merge_view_inputs. It returns the
        merged args and the synthetic_base_info needed for unpacking.
        """
        self._emitter.emit_comment(
            "Merge aliased inputs into synthetic bases for the compiled function."
        )
        self._emitter.emit_comment(
            "Uses merge_view_inputs to detect aliased inputs and create synthetic bases."
        )
        self._emitter.emit_comment(
            "Returns merged args and synthetic_base_info for later unpacking."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_line(
            "def _merge_aliased_inputs_to_synthetic_bases(args, old_input_info, aot_config=None):"
        )
        self._emitter.indent()
        self._emitter.emit_comment(
            "Call merge_view_inputs to get synthetic bases and mapping info."
        )
        self._emitter.emit_line(
            "args_with_bases, _, synth_info = merge_view_inputs("
        )
        self._emitter.indent()
        self._emitter.emit_line("aot_config, args, None, old_input_info,")
        self._emitter.emit_line("is_inference=_synthetic_base_is_inference")
        self._emitter.dedent()
        self._emitter.emit_line(")")
        self._emitter.emit_blank_line()
        self._emitter.emit_line("return args_with_bases, synth_info")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

    def _emit_unpack_synthetic_bases_function(
        self, node: AOTSyntheticBaseWrapperNode
    ) -> None:
        """
        Emit the _unpack_synthetic_bases function.

        This function takes args that have been merged into synthetic bases
        and reconstructs the original views using synthetic_base_info.
        Each element in synthetic_base_info is either:
        - An int: index into args to use directly
        - A tuple (base_idx, view_tensor): create view from args[base_idx]
        """
        self._emitter.emit_comment(
            "Unpack synthetic bases back to original views for the traced function."
        )
        self._emitter.emit_comment(
            "For each input, either pass through directly (int index) or"
        )
        self._emitter.emit_comment(
            "reconstruct the view from a synthetic base (tuple with base_idx, view_tensor)."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_line("def _unpack_synthetic_bases(primals, synthetic_base_info):")
        self._emitter.indent()

        self._emitter.emit_line("if synthetic_base_info is None:")
        self._emitter.indent()
        self._emitter.emit_line("return list(primals)")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("replay_views = functorch_config.view_replay_for_aliased_outputs")
        self._emitter.emit_line("f_args_inner = []")
        self._emitter.emit_blank_line()

        self._emitter.emit_line("for inner_idx_or_tuple in synthetic_base_info:")
        self._emitter.indent()
        self._emitter.emit_line("if isinstance(inner_idx_or_tuple, int):")
        self._emitter.indent()
        self._emitter.emit_comment(
            "Direct index: just use the primal at that index."
        )
        self._emitter.emit_line("f_args_inner.append(primals[inner_idx_or_tuple])")
        self._emitter.dedent()
        self._emitter.emit_line("else:")
        self._emitter.indent()
        self._emitter.emit_comment(
            "Tuple (base_idx, view_tensor): reconstruct view from base."
        )
        self._emitter.emit_line("inner_base_idx, view_tensor = inner_idx_or_tuple")
        self._emitter.emit_line("base = primals[inner_base_idx]")
        self._emitter.emit_line("view_arg = gen_alias_from_base(")
        self._emitter.indent()
        self._emitter.emit_line("base,")
        self._emitter.emit_line("view_tensor,")
        self._emitter.emit_line("view_tensor.requires_grad,")
        self._emitter.emit_line("replay_views=replay_views,")
        self._emitter.dedent()
        self._emitter.emit_line(")")
        self._emitter.emit_line("f_args_inner.append(view_arg)")
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("return f_args_inner")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

    def _emit_apply_metadata_mutations_function(
        self, node: AOTSyntheticBaseWrapperNode
    ) -> None:
        """
        Emit the _apply_metadata_mutations function.

        When aliased inputs have metadata mutations (size/stride changes),
        the compiled function returns extra outputs representing the mutated
        inputs. This function applies those mutations back to the original
        input tensors using as_strided_.
        """
        aliased_idx = node.aliased_arg_idx_with_metadata_mutations or []
        if not aliased_idx:
            return

        self._emitter.emit_comment(
            "Apply metadata mutations back to aliased inputs after the call."
        )
        self._emitter.emit_comment(
            "The compiled fn returns extra outputs for inputs with metadata mutations."
        )
        self._emitter.emit_comment(
            "We apply as_strided_ to update the original input's size/stride/offset."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_line(
            "def _apply_metadata_mutations(original_args, outputs):"
        )
        self._emitter.indent()

        self._emitter.emit_line(
            "num_metadata_mutations = len(_aliased_arg_idx_with_metadata_mutations)"
        )
        self._emitter.emit_line("if num_metadata_mutations == 0:")
        self._emitter.indent()
        self._emitter.emit_line("return outputs")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Extract the aliased inputs that have metadata mutations."
        )
        self._emitter.emit_line("aliased_inputs = [")
        self._emitter.indent()
        self._emitter.emit_line(
            "original_args[i] for i in _aliased_arg_idx_with_metadata_mutations"
        )
        self._emitter.dedent()
        self._emitter.emit_line("]")
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "The extra outputs are at the end and represent mutated input states."
        )
        self._emitter.emit_line(
            "mutated_metadata_inps = outputs[-num_metadata_mutations:]"
        )
        self._emitter.emit_line(
            "user_outs = outputs[:-num_metadata_mutations]"
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Apply the metadata mutations via as_strided_ to original inputs."
        )
        self._emitter.emit_line(
            "for inp, mutated_inp in zip(aliased_inputs, mutated_metadata_inps):"
        )
        self._emitter.indent()
        self._emitter.emit_line("inp.as_strided_(")
        self._emitter.indent()
        self._emitter.emit_line("mutated_inp.size(),")
        self._emitter.emit_line("mutated_inp.stride(),")
        self._emitter.emit_line("mutated_inp.storage_offset(),")
        self._emitter.dedent()
        self._emitter.emit_line(")")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("return user_outs")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

    def visit_debug_assert_wrapper(self, node: DebugAssertWrapperNode) -> str:
        """
        Generate code for debug assert wrapper.

        When debug asserts are enabled, this wrapper validates that each input's
        requires_grad status matches what was recorded during tracing.

        This mirrors DebugAssertWrapper from runtime_wrappers.py.
        """
        if not node.flat_requires_grad:
            return ""

        if not self._has_emitted_wrapper_section:
            self._emitter.emit_section_header("Wrapper Helper Functions")
            self._has_emitted_wrapper_section = True

        self._has_debug_assert_wrapper = True

        flat_rg_repr = repr(node.flat_requires_grad)
        self._emitter.emit_comment(
            "Debug assert wrapper: validates requires_grad expectations."
        )
        self._emitter.emit_line(f"_expected_requires_grad = {flat_rg_repr}")
        self._emitter.emit_blank_line()

        self._emitter.emit_line("def _assert_requires_grad(args):")
        self._emitter.indent()
        self._emitter.emit_comment(
            "Assert that each tensor's requires_grad matches expected."
        )
        self._emitter.emit_line("for i, (arg, expected) in enumerate(zip(args, _expected_requires_grad)):")
        self._emitter.indent()
        self._emitter.emit_line(
            "if hasattr(arg, 'requires_grad') and arg.requires_grad != expected:"
        )
        self._emitter.indent()
        self._emitter.emit_line(
            'raise AssertionError(f"Input {i}: expected requires_grad={expected}, got {arg.requires_grad}")'
        )
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        return ""

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

        When CUDA graphs are enabled for training (forward/backward phases), the
        generated class uses CUDA graph capture and replay in both forward() and
        backward() methods. Module-level variables are used to store the graph
        state, static buffers, and captured flags.

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
            or node.has_lazy_backward
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

        if self._forward_cuda_graph_node is not None:
            self._emit_cuda_graph_state_variables(self._forward_cuda_graph_node)

        if self._backward_cuda_graph_node is not None:
            self._emit_cuda_graph_state_variables(self._backward_cuda_graph_node)

        # Emit lazy backward state variables if lazy backward is enabled
        if node.has_lazy_backward:
            self._emit_lazy_backward_state_variables(node)

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

    def _emit_forward_wrapper_pre_call(self) -> None:
        """
        Emit wrapper pre-processing code for forward() before calling compiled_fn.

        This method emits calls to wrapper helper functions that transform args
        before they are passed to the compiled function. The wrappers are applied
        in the correct order matching AOTAutograd's post_compile wrapper stack.

        Wrapper application order (inside-out, matching wrapper_audit.md):
        1. DebugAssertWrapper - validates requires_grad expectations
        2. AOTDedupeWrapper - removes duplicate args
        3. AOTSyntheticBaseWrapper - merges aliased inputs to synthetic bases
        4. AOTDispatchSubclassWrapper - unwraps tensor subclasses
        5. FunctionalizedRngRuntimeWrapper - appends RNG state to args
        6. EffectTokensWrapper - injects effect tokens at start of args
        7. RuntimeWrapper - detaches inputs at specified indices

        The processed args are stored in _processed_args for the compiled call.
        """
        has_any_wrapper = (
            self._effect_token_count > 0 or
            self._has_subclass_wrapper or
            self._has_rng_wrapper or
            self._has_runtime_wrapper or
            self._has_dedupe_wrapper or
            self._has_synthetic_base_wrapper or
            self._has_debug_assert_wrapper
        )

        if not has_any_wrapper:
            return

        self._emitter.emit_blank_line()
        self._emitter.emit_comment(
            "Apply wrapper pre-processing to args before compiled function call."
        )
        self._emitter.emit_line("_processed_args = list(args)")

        if self._has_debug_assert_wrapper:
            self._emitter.emit_comment("Validate requires_grad expectations (debug mode).")
            self._emitter.emit_line("_assert_requires_grad(_processed_args)")

        if self._has_dedupe_wrapper and self._dedupe_keep_arg_mask:
            self._emitter.emit_comment("Remove duplicate arguments.")
            self._emitter.emit_line("_processed_args = _remove_dupe_args(_processed_args)")

        if self._has_synthetic_base_wrapper:
            self._emitter.emit_comment("Merge aliased inputs to synthetic bases.")
            self._emitter.emit_line(
                "_processed_args, _synth_info = _merge_aliased_inputs_to_synthetic_bases("
            )
            self._emitter.indent()
            self._emitter.emit_line("_processed_args, None, None")
            self._emitter.dedent()
            self._emitter.emit_line(")")

        if self._has_subclass_wrapper:
            self._emitter.emit_comment("Unwrap tensor subclasses to inner tensors.")
            self._emitter.emit_line(
                "_processed_args = _unwrap_tensor_subclasses(_processed_args)"
            )

        if self._has_rng_wrapper:
            self._emitter.emit_comment("Append RNG state (seed, offset) to args.")
            self._emitter.emit_line("_processed_args = _append_rng_state(_processed_args)")

        if self._effect_token_count > 0:
            self._emitter.emit_comment(
                f"Inject {self._effect_token_count} effect token(s) at start of args."
            )
            self._emitter.emit_line("_processed_args = _inject_effect_tokens(_processed_args)")

        if self._has_runtime_wrapper and self._indices_to_detach:
            self._emitter.emit_comment("Detach inputs at specified indices.")
            self._emitter.emit_line("_processed_args = _detach_inputs(_processed_args)")

        self._emitter.emit_blank_line()

    def _emit_forward_wrapper_post_call(self, output_var: str = "_result") -> None:
        """
        Emit wrapper post-processing code for forward() after calling compiled_fn.

        This method emits calls to wrapper helper functions that transform outputs
        after they are returned from the compiled function. The wrappers are applied
        in reverse order (outside-in) to undo the pre-call transformations.

        Wrapper application order (reverse of pre-call):
        1. EffectTokensWrapper - strips effect tokens from start of outputs
        2. FunctionalizedRngRuntimeWrapper - handles RNG offset outputs
        3. AOTDispatchSubclassWrapper - wraps outputs as tensor subclasses
        4. FakifiedOutWrapper - fixes output strides to match traced
        5. AOTSyntheticBaseWrapper - unpacks synthetic bases to original views

        Args:
            output_var: Name of the variable holding the raw outputs
        """
        has_any_wrapper = (
            self._effect_token_count > 0 or
            self._has_subclass_wrapper or
            self._has_rng_wrapper or
            self._has_fakified_wrapper or
            self._has_synthetic_base_wrapper
        )

        if not has_any_wrapper:
            return

        self._emitter.emit_blank_line()
        self._emitter.emit_comment(
            "Apply wrapper post-processing to outputs."
        )

        if self._effect_token_count > 0:
            self._emitter.emit_comment("Strip effect tokens from start of outputs.")
            self._emitter.emit_line(f"{output_var} = _strip_effect_tokens({output_var})")

        if self._has_rng_wrapper:
            self._emitter.emit_comment("Handle RNG offset outputs.")
            self._emitter.emit_line(f"{output_var} = _handle_rng_outputs({output_var})")

        if self._has_subclass_wrapper:
            self._emitter.emit_comment("Wrap outputs as tensor subclasses.")
            self._emitter.emit_line(
                f"{output_var} = _wrap_tensor_subclasses({output_var}, None, _num_fw_outs_saved_for_bw)"
            )

        if self._has_fakified_wrapper:
            self._emitter.emit_comment("Fix output strides to match traced layout.")
            self._emitter.emit_line(f"{output_var} = _fix_output_strides({output_var})")

        if self._has_synthetic_base_wrapper:
            self._emitter.emit_comment("Unpack synthetic bases to original views.")
            self._emitter.emit_line(f"{output_var} = _unpack_synthetic_bases({output_var}, _synth_info)")
            self._emitter.emit_comment("Apply metadata mutations for aliased inputs.")
            self._emitter.emit_line(f"_apply_metadata_mutations(args, {output_var})")

    def _emit_backward_wrapper_pre_call(self) -> None:
        """
        Emit wrapper pre-processing code for backward() before calling backward kernel.

        For backward, we need to apply fewer wrappers since most are forward-specific.
        The main concern is handling effect tokens and RNG state for backward.
        """
        has_any_wrapper = self._effect_token_count > 0 or self._has_rng_wrapper

        if not has_any_wrapper:
            return

        self._emitter.emit_comment(
            "Apply wrapper pre-processing to backward inputs."
        )

        if self._effect_token_count > 0:
            self._emitter.emit_comment("Inject effect tokens for backward inputs.")
            self._emitter.emit_line("backward_inputs = _inject_effect_tokens(backward_inputs)")

    def _emit_backward_wrapper_post_call(self, output_var: str = "_bw_result") -> None:
        """
        Emit wrapper post-processing code for backward() after calling backward kernel.

        This handles:
        1. Stripping effect tokens from backward outputs
        2. Reinserting duplicate gradients via add_dupe_map
        """
        has_any_wrapper = (
            self._effect_token_count > 0 or
            (self._has_dedupe_wrapper and self._dedupe_add_dupe_map)
        )

        if not has_any_wrapper:
            return

        self._emitter.emit_blank_line()
        self._emitter.emit_comment(
            "Apply wrapper post-processing to backward outputs."
        )

        if self._effect_token_count > 0:
            self._emitter.emit_comment("Strip effect tokens from backward outputs.")
            self._emitter.emit_line(f"{output_var} = _strip_effect_tokens({output_var})")

        if self._has_dedupe_wrapper and self._dedupe_add_dupe_map:
            self._emitter.emit_comment("Reinsert duplicate gradients.")
            self._emitter.emit_line(f"{output_var} = _add_dupe_args({output_var})")

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

        # Add debug print at function entry
        if self._emit_debug_prints:
            self._emitter.emit_line('print("[DEBUG] CompiledFunction.forward() called with:", "len(args)=", len(args))')
            self._emitter.emit_line('print("[DEBUG] Args shapes:", [tuple(a.shape) if hasattr(a, "shape") else type(a) for a in args])')

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

        # Emit wrapper pre-call processing if any wrappers are active
        self._emit_forward_wrapper_pre_call()

        # Determine which args variable to use
        args_var = "_processed_args" if self._has_forward_wrappers else "args"

        if self._forward_cuda_graph_node is not None:
            self._emitter.emit_blank_line()
            self._emit_forward_cuda_graph_logic(self._forward_cuda_graph_node, node)
        elif forward_code:
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
                if self._emit_debug_prints:
                    self._emitter.emit_line(f'print("[DEBUG] Calling compiled_fn with list({args_var}), len=", len({args_var}))')
                self._emitter.emit_line(
                    f"_inductor_result = {self._compiled_fn_var_name}(list({args_var}))"
                )
                if self._emit_debug_prints:
                    self._emitter.emit_line('print("[DEBUG] compiled_fn returned _inductor_result, type=", type(_inductor_result), "len=", len(_inductor_result) if hasattr(_inductor_result, "__len__") else "N/A")')
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

                # Apply wrapper post-processing if needed
                if self._has_forward_output_wrappers:
                    self._emitter.emit_line("_result = _inductor_result[0]")
                    self._emit_forward_wrapper_post_call("_result")
                    self._emitter.emit_line("return _result")
                else:
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
                        f"_inductor_result = {self._compiled_fn_var_name}(list({args_var}))"
                    )
                    if self._has_forward_output_wrappers:
                        self._emitter.emit_line("_result = _inductor_result[0]")
                        self._emit_forward_wrapper_post_call("_result")
                        self._emitter.emit_line("return _result")
                    else:
                        self._emitter.emit_line("return _inductor_result[0]")
                elif self._inductor_uses_list_args:
                    if self._has_forward_output_wrappers:
                        self._emitter.emit_line(
                            f"_result = {self._compiled_fn_var_name}(list({args_var}))"
                        )
                        self._emit_forward_wrapper_post_call("_result")
                        self._emitter.emit_line("return _result")
                    else:
                        self._emitter.emit_line(
                            f"return {self._compiled_fn_var_name}(list({args_var}))"
                        )
                elif self._inductor_returns_tuple:
                    self._emitter.emit_line(
                        f"_inductor_result = {self._compiled_fn_var_name}(*{args_var})"
                    )
                    if self._has_forward_output_wrappers:
                        self._emitter.emit_line("_result = _inductor_result[0]")
                        self._emit_forward_wrapper_post_call("_result")
                        self._emitter.emit_line("return _result")
                    else:
                        self._emitter.emit_line("return _inductor_result[0]")
                else:
                    if self._has_forward_output_wrappers:
                        self._emitter.emit_line(
                            f"_result = {self._compiled_fn_var_name}(*{args_var})"
                        )
                        self._emit_forward_wrapper_post_call("_result")
                        self._emitter.emit_line("return _result")
                    else:
                        self._emitter.emit_line(
                            f"return {self._compiled_fn_var_name}(*{args_var})"
                        )
        else:
            self._emitter.emit_blank_line()
            self._emitter.emit_comment(
                "Forward computation handled by compiled function."
            )
            if self._has_forward_output_wrappers:
                self._emitter.emit_line(f"_result = {self._compiled_fn_var_name}(*{args_var})")
                self._emit_forward_wrapper_post_call("_result")
                self._emitter.emit_line("return _result")
            else:
                self._emitter.emit_line(f"return {self._compiled_fn_var_name}(*{args_var})")

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

        # Add debug print at backward entry
        if self._emit_debug_prints:
            self._emitter.emit_line('print("[DEBUG] CompiledFunction.backward() called with:", "len(grad_outputs)=", len(grad_outputs))')

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

        # Handle lazy backward compilation if enabled
        if node.has_lazy_backward:
            self._emit_lazy_backward_compilation_logic(node)
            # After lazy compilation, we use the compiled backward
            self._emitter.emit_comment(
                "Prepare backward inputs: saved tensors followed by grad_outputs."
            )
            self._emitter.emit_line("backward_inputs = list(saved) + list(grad_outputs)")

            # Apply wrapper pre-processing for backward inputs
            self._emit_backward_wrapper_pre_call()

            if self._emit_debug_prints:
                self._emitter.emit_line('print("[DEBUG] Calling _compiled_bw with backward_inputs, len=", len(backward_inputs))')
            self._emitter.emit_blank_line()

            self._emitter.emit_comment(
                "Call the lazily-compiled backward function."
            )
            self._emitter.emit_line("_bw_result = _compiled_bw(backward_inputs)")

            # Apply wrapper post-processing for backward outputs
            self._emit_backward_wrapper_post_call("_bw_result")

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
        elif self._backward_cuda_graph_node is not None:
            self._emit_backward_cuda_graph_logic(self._backward_cuda_graph_node, node)
        elif self._has_inductor_backward_kernel:
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

            # Apply wrapper pre-processing for backward inputs
            self._emit_backward_wrapper_pre_call()

            if self._emit_debug_prints:
                self._emitter.emit_line('print("[DEBUG] Calling compiled_fn_backward with backward_inputs, len=", len(backward_inputs))')
            self._emitter.emit_blank_line()

            self._emitter.emit_comment(
                "Call the Inductor-compiled backward kernel."
            )
            self._emitter.emit_line("_bw_result = compiled_fn_backward(backward_inputs)")
            if self._emit_debug_prints:
                self._emitter.emit_line('print("[DEBUG] compiled_fn_backward returned, type=", type(_bw_result))')

            # Apply wrapper post-processing for backward outputs
            self._emit_backward_wrapper_post_call("_bw_result")

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

    def _emit_lazy_backward_state_variables(
        self,
        node: AOTAutogradWrapperNode,
    ) -> None:
        """
        Emit module-level state variables for lazy backward compilation.

        When lazy backward is enabled, the backward graph is not compiled until
        backward() is actually called. This defers compilation cost for inference
        workloads that never call backward.

        The generated code stores:
        - _lazy_bw_module: The uncompiled backward GraphModule (via object ID)
        - _lazy_bw_placeholder_list: Placeholder nodes for backward inputs
        - _lazy_bw_saved_context: TracingContext for lazy compilation
        - _lazy_bw_saved_compile_context: CompileContext for lazy compilation
        - _compiled_bw: The compiled backward function (None until first backward)

        These are accessed via nonlocal declarations in the backward method.

        Args:
            node: The AOTAutogradWrapperNode containing lazy backward info
        """
        self._emitter.emit_comment(
            "Lazy backward state variables."
        )
        self._emitter.emit_comment(
            "The backward graph is compiled on first backward() call."
        )
        self._emitter.emit_comment(
            "This defers compilation cost for inference-only workloads."
        )

        # Store the lazy backward module via object ID if available
        if node.lazy_bw_module is not None:
            bw_module_id = id(node.lazy_bw_module)
            self._emitter.emit_line(f"_lazy_bw_module_id = {bw_module_id}")
        else:
            self._emitter.emit_line("_lazy_bw_module_id = None")

        # Store placeholder list via object ID if available
        if node.lazy_bw_placeholder_list is not None:
            placeholder_id = id(node.lazy_bw_placeholder_list)
            self._emitter.emit_line(f"_lazy_bw_placeholder_list_id = {placeholder_id}")
        else:
            self._emitter.emit_line("_lazy_bw_placeholder_list_id = None")

        # Store saved context via object ID if available
        if node.lazy_bw_saved_context is not None:
            ctx_id = id(node.lazy_bw_saved_context)
            self._emitter.emit_line(f"_lazy_bw_saved_context_id = {ctx_id}")
        else:
            self._emitter.emit_line("_lazy_bw_saved_context_id = None")

        # Store saved compile context via object ID if available
        if node.lazy_bw_saved_compile_context is not None:
            compile_ctx_id = id(node.lazy_bw_saved_compile_context)
            self._emitter.emit_line(f"_lazy_bw_saved_compile_context_id = {compile_ctx_id}")
        else:
            self._emitter.emit_line("_lazy_bw_saved_compile_context_id = None")

        # The compiled backward starts as None
        self._emitter.emit_line("_compiled_bw = None")
        self._emitter.emit_blank_line()

    def _emit_lazy_backward_compilation_logic(
        self,
        node: AOTAutogradWrapperNode,
    ) -> None:
        """
        Emit the lazy backward compilation check and compilation code.

        This method generates code that checks if the backward function has been
        compiled yet, and if not, compiles it using the saved contexts and
        backward module.

        The generated code mirrors AOTDispatchAutograd._backward_impl:
        1. Check if _compiled_bw is None (as a module-level variable)
        2. If so, retrieve saved module and contexts via object ID
        3. Restore tracing and compile contexts
        4. Call bw_compiler to compile the backward module
        5. Store result in _compiled_bw for future calls

        Note: Unlike the runtime, we use a module-level variable rather than
        a class attribute because the generated code doesn't have access to
        the class definition at the point of method execution.

        Args:
            node: The AOTAutogradWrapperNode containing lazy backward info
        """
        self._emitter.emit_comment(
            "Lazy backward compilation: compile on first backward call."
        )
        self._emitter.emit_comment(
            "Access module-level _compiled_bw variable."
        )
        self._emitter.emit_line("global _compiled_bw")
        self._emitter.emit_blank_line()

        self._emitter.emit_line("if _compiled_bw is None:")
        self._emitter.indent()

        self._emitter.emit_comment(
            "First backward call - compile the backward graph now."
        )
        self._emitter.emit_comment(
            "LEGACY FALLBACK: Retrieve backward module and contexts via object ID."
        )
        self._emitter.emit_comment(
            "This uses ctypes-based obj_from_id which requires same-process execution."
        )
        self._emitter.emit_line("import copy")
        self._emitter.emit_line("from contextlib import nullcontext")
        self._emitter.emit_blank_line()

        # Retrieve objects via object ID
        self._emitter.emit_line("if _lazy_bw_module_id is not None:")
        self._emitter.indent()
        self._emitter.emit_line("_bw_module = obj_from_id(_lazy_bw_module_id)")
        self._emitter.dedent()
        self._emitter.emit_line("else:")
        self._emitter.indent()
        self._emitter.emit_line(
            'raise RuntimeError("Lazy backward module not available")'
        )
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("if _lazy_bw_placeholder_list_id is not None:")
        self._emitter.indent()
        self._emitter.emit_line("_placeholder_list = obj_from_id(_lazy_bw_placeholder_list_id)")
        self._emitter.dedent()
        self._emitter.emit_line("else:")
        self._emitter.indent()
        self._emitter.emit_line("_placeholder_list = []")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "LEGACY FALLBACK: Retrieve saved contexts for tracing/compilation via object ID."
        )
        self._emitter.emit_line("_saved_context = None")
        self._emitter.emit_line("if _lazy_bw_saved_context_id is not None:")
        self._emitter.indent()
        self._emitter.emit_line("_saved_context = obj_from_id(_lazy_bw_saved_context_id)")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_line("_saved_compile_context = None")
        self._emitter.emit_line("if _lazy_bw_saved_compile_context_id is not None:")
        self._emitter.indent()
        self._emitter.emit_line("_saved_compile_context = obj_from_id(_lazy_bw_saved_compile_context_id)")
        self._emitter.dedent()
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Compile the backward graph using saved contexts."
        )
        self._emitter.emit_comment(
            "This mirrors AOTDispatchAutograd._backward_impl lazy compilation."
        )
        self._emitter.emit_line("from torch._guards import tracing, compile_context")
        self._emitter.emit_blank_line()

        # Emit compilation with context restoration
        self._emitter.emit_line("with (")
        self._emitter.indent()
        self._emitter.emit_line("tracing(_saved_context) if _saved_context is not None else nullcontext(),")
        self._emitter.emit_line("compile_context(_saved_compile_context) if _saved_compile_context is not None else nullcontext(),")
        self._emitter.dedent()
        self._emitter.emit_line("):")
        self._emitter.indent()

        self._emitter.emit_comment(
            "Use torch.compile's backend to compile the backward module."
        )
        self._emitter.emit_comment(
            "The bw_compiler is accessed from the saved compile context."
        )
        self._emitter.emit_line("if _saved_compile_context is not None and hasattr(_saved_compile_context, 'bw_compiler'):")
        self._emitter.indent()
        self._emitter.emit_line("_compiled_bw = _saved_compile_context.bw_compiler(copy.deepcopy(_bw_module), _placeholder_list)")
        self._emitter.dedent()
        self._emitter.emit_line("else:")
        self._emitter.indent()
        self._emitter.emit_comment(
            "Fallback: use inductor as the backward compiler."
        )
        self._emitter.emit_line("from torch._inductor.compile_fx import compile_fx")
        self._emitter.emit_line("_compiled_bw = compile_fx(copy.deepcopy(_bw_module), _placeholder_list)")
        self._emitter.dedent()

        self._emitter.dedent()  # End with block
        self._emitter.dedent()  # End if _compiled_bw is None
        self._emitter.emit_blank_line()

    def _emit_cuda_graph_state_variables(
        self,
        cuda_node: CUDAGraphSetupNode,
    ) -> None:
        """
        Emit module-level state variables for CUDA graph capture/replay.

        When CUDA graphs are used with the AOT Autograd wrapper, we need
        module-level variables to store the graph state, static buffers,
        and a flag indicating whether the graph has been captured.

        These variables are accessed via nonlocal declarations in the forward
        and backward methods, allowing the graph state to persist across calls.

        For forward graphs in training mode, we also emit a static saved tensors
        variable that holds references to the saved_tensors at stable memory
        addresses. This is critical because the backward graph needs to access
        these tensors at the same addresses across replays.

        The generated code structure:
            # CUDA graph state for forward pass
            _forward_graph = None
            _forward_stream = None
            _static_inputs_forward = None
            _static_outputs_forward = None
            _forward_captured = False
            _static_saved_tensors_forward = None  # Only for forward in training

        Args:
            cuda_node: The CUDAGraphSetupNode containing the graph configuration
        """
        graph_id = cuda_node.graph_id
        graph_var = f"_{graph_id}_graph"
        stream_var = f"_{graph_id}_stream"
        static_inputs_var = f"_static_inputs_{graph_id}"
        static_outputs_var = f"_static_outputs_{graph_id}"
        captured_flag_var = f"_{graph_id}_captured"

        phase_name = "forward" if cuda_node.phase == CUDAGraphPhase.FORWARD else (
            "backward" if cuda_node.phase == CUDAGraphPhase.BACKWARD else "inference"
        )

        self._emitter.emit_comment(
            f"CUDA graph state variables for {phase_name} pass."
        )
        self._emitter.emit_comment(
            "These are accessed via nonlocal in the forward/backward methods."
        )
        self._emitter.emit_line(f"{graph_var} = None")
        self._emitter.emit_line(f"{stream_var} = None")
        self._emitter.emit_line(f"{static_inputs_var} = None")
        self._emitter.emit_line(f"{static_outputs_var} = None")
        self._emitter.emit_line(f"{captured_flag_var} = False")

        if cuda_node.phase == CUDAGraphPhase.FORWARD:
            static_saved_tensors_var = f"_static_saved_tensors_{graph_id}"
            self._emitter.emit_comment(
                "Static saved tensors buffer for forward-backward communication."
            )
            self._emitter.emit_comment(
                "The backward graph uses these instead of ctx.saved_tensors to"
            )
            self._emitter.emit_comment(
                "ensure stable memory addresses across graph replays."
            )
            self._emitter.emit_line(f"{static_saved_tensors_var} = None")

        self._emitter.emit_blank_line()

    def _emit_static_buffer_allocation(
        self,
        num_inputs: int,
        static_input_indices: list[int],
        graph_id: str,
    ) -> None:
        """
        Emit code to allocate static input buffers for CUDA graph capture.

        CUDA graphs require that input tensors have stable memory addresses across
        replays. This method generates code that:

        1. Creates a list to hold static input buffers (_static_inputs)
        2. Documents which indices are static (parameters/buffers) vs dynamic
        3. The actual tensor allocation happens during warmup when we see real
           tensor shapes and dtypes

        Static inputs (parameters/buffers) are inputs whose memory addresses don't
        change between calls. These don't need to be copied before replay because
        they're already at stable addresses.

        Dynamic inputs (user inputs like x) need to be copied into pre-allocated
        static buffers before each graph replay, because the user may pass
        different tensor objects with different memory addresses.

        The generated code structure:
            # Static buffer allocation for CUDA graph
            _static_inputs_{graph_id} = [None] * {num_inputs}
            _static_outputs_{graph_id} = None

            # Static input indices (no copy needed): [0, 2]
            # Dynamic input indices (copy before replay): [1, 3]

        Args:
            num_inputs: Total number of inputs to the compiled function
            static_input_indices: List of indices for static inputs (parameters/buffers)
            graph_id: Unique identifier for this CUDA graph (used in variable names)
        """
        static_inputs_var = f"_static_inputs_{graph_id}"
        static_outputs_var = f"_static_outputs_{graph_id}"

        dynamic_indices = [i for i in range(num_inputs) if i not in static_input_indices]

        self._emitter.emit_comment("Static buffer allocation for CUDA graph capture.")
        self._emitter.emit_comment(
            "These buffers hold tensors with stable memory addresses for graph replay."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_line(f"{static_inputs_var} = [None] * {num_inputs}")
        self._emitter.emit_line(f"{static_outputs_var} = None")
        self._emitter.emit_blank_line()

        if static_input_indices:
            self._emitter.emit_comment(
                f"Static input indices (params/buffers, no copy needed): {static_input_indices}"
            )
        if dynamic_indices:
            self._emitter.emit_comment(
                f"Dynamic input indices (user inputs, copy before replay): {dynamic_indices}"
            )

        self._emitter.emit_blank_line()

    def _emit_input_copy_logic(
        self,
        num_inputs: int,
        static_input_indices: list[int],
        graph_id: str,
        inputs_var: str = "inputs",
    ) -> None:
        """
        Emit code to copy dynamic inputs into static buffers before graph replay.

        CUDA graphs require that input tensors have stable memory addresses across
        replays. Static inputs (parameters/buffers) already have stable addresses,
        but dynamic inputs (user-provided tensors like x) may have different memory
        addresses on each call.

        This method generates code that:
        1. Iterates over dynamic input indices (not in static_input_indices)
        2. Copies each dynamic input's data into the pre-allocated static buffer
           using `.copy_()` which is an in-place operation that preserves the
           destination tensor's memory address

        The generated code structure:
            # Copy dynamic inputs into static buffers for CUDA graph replay
            _static_inputs_{graph_id}[1].copy_(inputs[1])
            _static_inputs_{graph_id}[3].copy_(inputs[3])

        Or for all dynamic inputs:
            for _idx in [1, 3]:
                _static_inputs_{graph_id}[_idx].copy_(inputs[_idx])

        Args:
            num_inputs: Total number of inputs to the compiled function
            static_input_indices: List of indices for static inputs (parameters/buffers)
                that don't need to be copied
            graph_id: Unique identifier for this CUDA graph (used in variable names)
            inputs_var: Name of the variable containing the input list/tuple
        """
        static_inputs_var = f"_static_inputs_{graph_id}"
        dynamic_indices = [i for i in range(num_inputs) if i not in static_input_indices]

        if not dynamic_indices:
            self._emitter.emit_comment(
                "No dynamic inputs to copy (all inputs are static parameters/buffers)."
            )
            return

        self._emitter.emit_comment(
            "Copy dynamic inputs into static buffers for CUDA graph replay."
        )
        self._emitter.emit_comment(
            "Static inputs (parameters/buffers) retain stable memory addresses,"
        )
        self._emitter.emit_comment(
            "but dynamic inputs must be copied to preserve graph correctness."
        )
        self._emitter.emit_blank_line()

        if len(dynamic_indices) <= 3:
            for idx in dynamic_indices:
                self._emitter.emit_line(
                    f"{static_inputs_var}[{idx}].copy_({inputs_var}[{idx}])"
                )
        else:
            self._emitter.emit_line(f"for _idx in {dynamic_indices}:")
            self._emitter.indent()
            self._emitter.emit_line(
                f"{static_inputs_var}[_idx].copy_({inputs_var}[_idx])"
            )
            self._emitter.dedent()

        self._emitter.emit_blank_line()

    def _emit_forward_cuda_graph_logic(
        self,
        cuda_node: CUDAGraphSetupNode,
        aot_node: AOTAutogradWrapperNode,
    ) -> None:
        """
        Emit CUDA graph capture and replay logic for the forward method.

        When CUDA graphs are enabled for training mode, the forward method needs to:
        1. Check if the graph has been captured (using a class-level flag)
        2. On first call: allocate static buffers, run warmup, capture the graph
        3. On subsequent calls: copy dynamic inputs and replay the graph
        4. Handle saved_tensors by storing them in static buffers that persist
           between forward and backward

        This generates code that manages the graph capture state using a nonlocal
        variable that persists across method calls. The static buffers for inputs
        and outputs are stored at module level so they can be accessed by both
        forward and backward methods.

        For training with backward pass, the saved tensors are stored in a static
        buffer (_static_saved_tensors_{graph_id}) that the backward graph uses
        instead of ctx.saved_tensors. This ensures stable memory addresses across
        graph replays.

        Args:
            cuda_node: The CUDAGraphSetupNode for the forward pass
            aot_node: The AOTAutogradWrapperNode for context
        """
        graph_id = cuda_node.graph_id
        graph_var = f"_{graph_id}_graph"
        stream_var = f"_{graph_id}_stream"
        static_inputs_var = f"_static_inputs_{graph_id}"
        static_outputs_var = f"_static_outputs_{graph_id}"
        captured_flag_var = f"_{graph_id}_captured"
        static_saved_tensors_var = f"_static_saved_tensors_{graph_id}"

        self._emitter.emit_comment(
            "CUDA graph forward pass with capture/replay optimization."
        )
        self._emitter.emit_comment(
            "On first call: warmup, capture. On subsequent calls: replay."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Declare global references to module-level graph state."
        )
        self._emitter.emit_comment(
            "These variables persist across method calls to enable graph replay."
        )
        nonlocal_vars = f"{graph_var}, {stream_var}, {static_inputs_var}, {static_outputs_var}, {captured_flag_var}"
        if self._has_inductor_backward_kernel:
            nonlocal_vars += f", {static_saved_tensors_var}"
        self._emitter.emit_line(f"global {nonlocal_vars}")
        self._emitter.emit_blank_line()

        self._emitter.emit_line(f"if not {captured_flag_var}:")
        self._emitter.indent()

        self._emitter.emit_comment(
            "First call: Initialize graph, run warmup, then capture."
        )
        self._emitter.emit_line(f"{graph_var} = torch.cuda.CUDAGraph()")
        self._emitter.emit_line(f"{stream_var} = torch.cuda.Stream()")
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Allocate static input buffers by cloning current inputs."
        )
        self._emitter.emit_line(
            f"{static_inputs_var} = [t.clone() if isinstance(t, torch.Tensor) else t for t in args]"
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Warmup run on a separate stream to populate buffers."
        )
        self._emitter.emit_line(f"with torch.cuda.stream({stream_var}):")
        self._emitter.indent()
        self._emitter.emit_line(f"for _warmup_iter in range({cuda_node.warmup_runs}):")
        self._emitter.indent()

        if self._has_inductor_kernel:
            self._emitter.emit_line(
                f"_warmup_result = {self._compiled_fn_var_name}(list({static_inputs_var}))"
            )
        else:
            self._emitter.emit_line(
                f"_warmup_result = {self._compiled_fn_var_name}(*{static_inputs_var})"
            )
        self._emitter.dedent()
        self._emitter.dedent()

        self._emitter.emit_line(f"torch.cuda.current_stream().wait_stream({stream_var})")
        self._emitter.emit_blank_line()

        pool_arg = ""
        if cuda_node.pool_id is not None:
            pool_var = f"_{graph_id}_pool"
            self._emitter.emit_line(f"{pool_var} = torch.cuda.graph_pool_handle()")
            pool_arg = f", pool={pool_var}"

        self._emitter.emit_comment("Capture the forward graph.")
        self._emitter.emit_line(
            f"with torch.cuda.graph({graph_var}, stream={stream_var}{pool_arg}):"
        )
        self._emitter.indent()

        if self._has_inductor_kernel:
            self._emitter.emit_line(
                f"{static_outputs_var} = {self._compiled_fn_var_name}(list({static_inputs_var}))"
            )
        else:
            self._emitter.emit_line(
                f"{static_outputs_var} = {self._compiled_fn_var_name}(*{static_inputs_var})"
            )

        self._emitter.dedent()
        self._emitter.emit_line(f"{captured_flag_var} = True")

        if self._has_inductor_backward_kernel:
            self._emitter.emit_blank_line()
            self._emitter.emit_comment(
                "Store saved tensors in static buffer for backward graph."
            )
            self._emitter.emit_comment(
                "The backward graph will use these static references instead of"
            )
            self._emitter.emit_comment(
                "ctx.saved_tensors to ensure stable memory addresses on replay."
            )
            self._emitter.emit_line(f"if isinstance({static_outputs_var}, (tuple, list)) and len({static_outputs_var}) > 1:")
            self._emitter.indent()
            self._emitter.emit_line(f"{static_saved_tensors_var} = list({static_outputs_var}[1:])")
            self._emitter.dedent()
            self._emitter.emit_line("else:")
            self._emitter.indent()
            self._emitter.emit_line(f"{static_saved_tensors_var} = []")
            self._emitter.dedent()

        self._emitter.dedent()  # End if not captured

        self._emitter.emit_line("else:")
        self._emitter.indent()
        self._emitter.emit_comment("Graph already captured - copy dynamic inputs and replay.")

        dynamic_indices = [
            i for i in range(aot_node.num_inputs)
            if i not in cuda_node.static_input_indices
        ]

        if dynamic_indices:
            if len(dynamic_indices) <= 3:
                for idx in dynamic_indices:
                    self._emitter.emit_line(
                        f"{static_inputs_var}[{idx}].copy_(args[{idx}])"
                    )
            else:
                self._emitter.emit_line(f"for _idx in {dynamic_indices}:")
                self._emitter.indent()
                self._emitter.emit_line(
                    f"{static_inputs_var}[_idx].copy_(args[_idx])"
                )
                self._emitter.dedent()
        else:
            self._emitter.emit_comment("No dynamic inputs to copy.")

        self._emitter.emit_blank_line()
        self._emitter.emit_line(f"{graph_var}.replay()")
        if cuda_node.force_cudagraph_sync:
            self._emitter.emit_line("torch.cuda.synchronize()")
        self._emitter.dedent()  # End else

        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Extract output from static_outputs and handle saved tensors."
        )
        if self._has_inductor_backward_kernel:
            self._emitter.emit_comment(
                "For training: also call save_for_backward with static saved tensors."
            )
            self._emitter.emit_comment(
                "This allows ctx.saved_tensors to work, but backward graph uses"
            )
            self._emitter.emit_comment(
                "the static buffer directly for stable memory addresses."
            )
            self._emitter.emit_line(f"if {static_saved_tensors_var}:")
            self._emitter.indent()
            self._emitter.emit_line(f"ctx.save_for_backward(*{static_saved_tensors_var})")
            self._emitter.dedent()
            self._emitter.emit_line(f"return {static_outputs_var}[0]")
        else:
            if self._inductor_returns_tuple and self._has_inductor_kernel:
                self._emitter.emit_line(f"return {static_outputs_var}[0]")
            else:
                self._emitter.emit_line(f"return {static_outputs_var}")

    def _emit_backward_cuda_graph_logic(
        self,
        cuda_node: CUDAGraphSetupNode,
        aot_node: AOTAutogradWrapperNode,
    ) -> None:
        """
        Emit CUDA graph capture and replay logic for the backward method.

        Similar to forward, but operates on saved tensors and grad_outputs.
        The backward graph is captured separately from forward and uses its
        own static buffers for inputs (saved_tensors + grad_outputs) and
        outputs (gradients).

        For proper CUDA graph integration with saved_tensors:
        1. The saved tensors are accessed from the forward graph's static
           saved tensors buffer (_static_saved_tensors_forward), not from
           ctx.saved_tensors. This ensures stable memory addresses.
        2. On first call: allocate static buffers for grad_outputs only
           (saved tensors are already static from forward)
        3. On replay: only copy grad_outputs into static buffers, not saved
           tensors (they're already at stable addresses from forward graph)

        Args:
            cuda_node: The CUDAGraphSetupNode for the backward pass
            aot_node: The AOTAutogradWrapperNode for context
        """
        graph_id = cuda_node.graph_id
        graph_var = f"_{graph_id}_graph"
        stream_var = f"_{graph_id}_stream"
        static_inputs_var = f"_static_inputs_{graph_id}"
        static_outputs_var = f"_static_outputs_{graph_id}"
        captured_flag_var = f"_{graph_id}_captured"

        forward_graph_id = self._forward_cuda_graph_node.graph_id if self._forward_cuda_graph_node else "forward"
        static_saved_tensors_var = f"_static_saved_tensors_{forward_graph_id}"

        self._emitter.emit_comment(
            "CUDA graph backward pass with capture/replay optimization."
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Declare global references to module-level graph state."
        )
        self._emitter.emit_comment(
            "Note: We also reference the forward's static_saved_tensors for"
        )
        self._emitter.emit_comment(
            "stable memory addresses between forward and backward graphs."
        )
        self._emitter.emit_line(f"global {graph_var}, {stream_var}, {static_inputs_var}, {static_outputs_var}, {captured_flag_var}, {static_saved_tensors_var}")
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Prepare backward inputs using STATIC saved tensors from forward."
        )
        self._emitter.emit_comment(
            "Using static saved tensors ensures stable memory addresses across"
        )
        self._emitter.emit_comment(
            "graph replays. ctx.saved_tensors would give different addresses."
        )
        self._emitter.emit_line(f"_static_saved = {static_saved_tensors_var} if {static_saved_tensors_var} else list(saved)")
        self._emitter.emit_line("backward_inputs = list(_static_saved) + list(grad_outputs)")
        self._emitter.emit_blank_line()

        self._emitter.emit_line(f"if not {captured_flag_var}:")
        self._emitter.indent()

        self._emitter.emit_comment(
            "First backward call: Initialize graph, warmup, and capture."
        )
        self._emitter.emit_line(f"{graph_var} = torch.cuda.CUDAGraph()")
        self._emitter.emit_line(f"{stream_var} = torch.cuda.Stream()")
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Allocate static input buffers for backward."
        )
        self._emitter.emit_comment(
            "For saved tensors, we use the static references directly."
        )
        self._emitter.emit_comment(
            "For grad_outputs, we clone to create static buffers."
        )
        self._emitter.emit_line(
            f"{static_inputs_var} = list(_static_saved) + [t.clone() if isinstance(t, torch.Tensor) else t for t in grad_outputs]"
        )
        self._emitter.emit_blank_line()

        self._emitter.emit_comment("Warmup run for backward graph.")
        self._emitter.emit_line(f"with torch.cuda.stream({stream_var}):")
        self._emitter.indent()
        self._emitter.emit_line(f"for _warmup_iter in range({cuda_node.warmup_runs}):")
        self._emitter.indent()
        self._emitter.emit_line(
            f"_bw_warmup = compiled_fn_backward({static_inputs_var})"
        )
        self._emitter.dedent()
        self._emitter.dedent()

        self._emitter.emit_line(f"torch.cuda.current_stream().wait_stream({stream_var})")
        self._emitter.emit_blank_line()

        pool_arg = ""
        if cuda_node.pool_id is not None:
            pool_var = f"_{graph_id}_pool"
            self._emitter.emit_line(f"{pool_var} = torch.cuda.graph_pool_handle()")
            pool_arg = f", pool={pool_var}"

        self._emitter.emit_comment("Capture the backward graph.")
        self._emitter.emit_line(
            f"with torch.cuda.graph({graph_var}, stream={stream_var}{pool_arg}):"
        )
        self._emitter.indent()
        self._emitter.emit_line(
            f"{static_outputs_var} = compiled_fn_backward({static_inputs_var})"
        )
        self._emitter.dedent()
        self._emitter.emit_line(f"{captured_flag_var} = True")
        self._emitter.dedent()  # End if not captured

        self._emitter.emit_line("else:")
        self._emitter.indent()
        self._emitter.emit_comment(
            "Graph already captured - copy only grad_outputs and replay."
        )
        self._emitter.emit_comment(
            "Saved tensors are already at static addresses from forward graph,"
        )
        self._emitter.emit_comment(
            "so we only need to copy the grad_outputs into static buffers."
        )

        self._emitter.emit_line(f"_num_saved = len(_static_saved)")
        self._emitter.emit_line("for _idx, _grad in enumerate(grad_outputs):")
        self._emitter.indent()
        self._emitter.emit_line(
            f"if isinstance(_grad, torch.Tensor):"
        )
        self._emitter.indent()
        self._emitter.emit_line(
            f"{static_inputs_var}[_num_saved + _idx].copy_(_grad)"
        )
        self._emitter.dedent()
        self._emitter.dedent()

        self._emitter.emit_blank_line()
        self._emitter.emit_line(f"{graph_var}.replay()")
        if cuda_node.force_cudagraph_sync:
            self._emitter.emit_line("torch.cuda.synchronize()")
        self._emitter.dedent()  # End else

        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Return gradients from static outputs."
        )
        self._emitter.emit_line("if isinstance({}, (tuple, list)) and len({}) > 0:".format(
            static_outputs_var, static_outputs_var
        ))
        self._emitter.indent()
        self._emitter.emit_line("if isinstance({}[0], (tuple, list)):".format(static_outputs_var))
        self._emitter.indent()
        self._emitter.emit_line(f"return {static_outputs_var}[0]")
        self._emitter.dedent()
        self._emitter.emit_line(f"return {static_outputs_var}")
        self._emitter.dedent()
        self._emitter.emit_line(f"return {static_outputs_var}")

    def _emit_replay_wrapper_function(
        self,
        node: CUDAGraphSetupNode,
        replay_fn_name: str,
        graph_var: str,
        static_inputs_var: str,
        static_outputs_var: str,
    ) -> None:
        """
        Emit a replay wrapper function that replays the captured CUDA graph.

        The generated function:
        1. Copies dynamic inputs into pre-allocated static buffers (parameters/
           buffers with stable addresses don't need copying)
        2. Replays the captured CUDA graph
        3. Returns the static output tensors

        When skip_dynamic_graphs is True, the function also:
        4. Checks if input shapes match the shapes seen during graph capture
        5. Falls back to direct function call if shapes differ

        This replay function becomes the callable that should be used instead of
        the original compiled function when CUDA graphs are enabled. The graph
        replay is much faster than re-executing the operations because the GPU
        command stream is pre-recorded.

        Generated code structure:
            def _replay_{graph_id}(inputs):
                # [If skip_dynamic_graphs=True] Check shapes match expected
                # Copy dynamic inputs into static buffers for CUDA graph replay.
                # Static inputs (parameters/buffers) retain stable memory addresses,
                # but dynamic inputs must be copied to preserve graph correctness.

                _static_inputs_{graph_id}[1].copy_(inputs[1])

                # Replay the captured CUDA graph. This executes all the GPU
                # operations that were recorded during graph capture, using
                # the static buffers for inputs and outputs.
                _{graph_id}_graph.replay()

                return _static_outputs_{graph_id}

        Args:
            node: The CUDAGraphSetupNode being processed
            replay_fn_name: Name for the generated replay function
            graph_var: Variable name holding the CUDAGraph object
            static_inputs_var: Variable name for static inputs list
            static_outputs_var: Variable name for static outputs
        """
        num_inputs = len(self._argument_names) if self._argument_names else 0

        expected_shapes_var = f"_expected_shapes_{node.graph_id}"
        if node.skip_dynamic_graphs:
            self._emitter.emit_comment(
                "Record expected input shapes for dynamic shape checking."
            )
            self._emitter.emit_comment(
                "CUDA graphs require fixed tensor shapes, so we record the shapes"
            )
            self._emitter.emit_comment(
                "seen during capture and fall back to direct execution if they differ."
            )
            self._emitter.emit_line(
                f"{expected_shapes_var} = [t.shape if hasattr(t, 'shape') else None "
                f"for t in {static_inputs_var}]"
            )
            self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Define the replay function that will be called instead of the"
        )
        self._emitter.emit_comment(
            "original compiled function. Graph replay is much faster because"
        )
        self._emitter.emit_comment(
            "the GPU command stream is pre-recorded."
        )

        self._emitter.emit_line(f"def {replay_fn_name}(inputs):")
        self._emitter.indent()

        if node.skip_dynamic_graphs:
            self._emitter.emit_comment(
                "Check if input shapes match the shapes seen during graph capture."
            )
            self._emitter.emit_comment(
                "CUDA graphs require fixed tensor shapes, so we fall back to"
            )
            self._emitter.emit_comment(
                "direct function execution if any shapes differ."
            )
            self._emitter.emit_line(
                f"_shapes_match = all("
            )
            self._emitter.indent()
            self._emitter.emit_line(
                f"(not hasattr(inputs[i], 'shape') and {expected_shapes_var}[i] is None) or "
            )
            self._emitter.emit_line(
                f"(hasattr(inputs[i], 'shape') and inputs[i].shape == {expected_shapes_var}[i])"
            )
            self._emitter.dedent()
            self._emitter.emit_line(
                f"for i in range(len(inputs))"
            )
            self._emitter.emit_line(")")
            self._emitter.emit_blank_line()
            self._emitter.emit_line("if not _shapes_match:")
            self._emitter.indent()
            self._emitter.emit_comment(
                "Shapes differ from graph capture - fall back to direct execution."
            )
            self._emitter.emit_comment(
                "This is slower but handles dynamic shapes correctly."
            )
            self._emitter.emit_line(
                f"return {self._compiled_fn_var_name}(inputs)"
            )
            self._emitter.dedent()
            self._emitter.emit_blank_line()

        dynamic_indices = [
            i for i in range(num_inputs)
            if i not in node.static_input_indices
        ]

        if dynamic_indices:
            self._emitter.emit_comment(
                "Copy dynamic inputs into static buffers for CUDA graph replay."
            )
            self._emitter.emit_comment(
                "Static inputs (parameters/buffers) retain stable memory addresses,"
            )
            self._emitter.emit_comment(
                "but dynamic inputs must be copied to preserve graph correctness."
            )
            self._emitter.emit_blank_line()

            if len(dynamic_indices) <= 3:
                for idx in dynamic_indices:
                    self._emitter.emit_line(
                        f"{static_inputs_var}[{idx}].copy_(inputs[{idx}])"
                    )
            else:
                self._emitter.emit_line(f"for _idx in {dynamic_indices}:")
                self._emitter.indent()
                self._emitter.emit_line(
                    f"{static_inputs_var}[_idx].copy_(inputs[_idx])"
                )
                self._emitter.dedent()

            self._emitter.emit_blank_line()
        else:
            self._emitter.emit_comment(
                "No dynamic inputs to copy (all inputs are static parameters/buffers)."
            )
            self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Replay the captured CUDA graph. This executes all the GPU"
        )
        self._emitter.emit_comment(
            "operations that were recorded during graph capture, using"
        )
        self._emitter.emit_comment(
            "the static buffers for inputs and outputs."
        )
        self._emitter.emit_line(f"{graph_var}.replay()")

        if node.force_cudagraph_sync:
            self._emitter.emit_blank_line()
            self._emitter.emit_comment(
                "Force synchronization after graph replay for debugging."
            )
            self._emitter.emit_comment(
                "This ensures all GPU operations complete before returning,"
            )
            self._emitter.emit_comment(
                "which is useful for timing, profiling, or debugging scenarios."
            )
            self._emitter.emit_line("torch.cuda.synchronize()")

        self._emitter.emit_blank_line()

        self._emitter.emit_line(f"return {static_outputs_var}")
        self._emitter.dedent()

        self._emitter.emit_blank_line()

    def visit_cuda_graph_setup(self, node: CUDAGraphSetupNode) -> str:
        """
        Generate code for CUDA graph capture and replay setup.

        This method generates executable Python code that:
        1. Creates a CUDA graph object and optional stream
        2. Allocates static input/output buffers
        3. Performs warmup runs to populate buffers with realistic values
        4. Captures GPU operations into the CUDA graph

        The generated warmup code runs on a dedicated CUDA stream to ensure
        isolation from any concurrent GPU work. The warmup runs execute the
        compiled function to populate static buffers before graph capture.

        Note: FORWARD and BACKWARD phase nodes are handled by the AOT Autograd
        wrapper (visit_aot_autograd_wrapper) and should not be processed here.
        These nodes integrate CUDA graph capture/replay into the forward() and
        backward() methods of the generated CompiledFunction class.

        Args:
            node: The CUDAGraphSetupNode to process

        Returns:
            The generated CUDA graph setup code, or empty string for
            FORWARD/BACKWARD nodes which are handled by AOT wrapper
        """
        # FORWARD and BACKWARD phase nodes are processed by visit_aot_autograd_wrapper,
        # not here. These nodes represent training CUDA graphs that need to be
        # integrated into the CompiledFunction's forward/backward methods.
        if node.phase in (CUDAGraphPhase.FORWARD, CUDAGraphPhase.BACKWARD):
            return ""

        if not self._has_emitted_cuda_section:
            self._emitter.emit_section_header("CUDA Graphs Setup")
            self._has_emitted_cuda_section = True

        self._emitter.add_import("import torch")

        graph_var = f"_{node.graph_id}_graph"
        stream_var = f"_{node.graph_id}_stream"
        static_inputs_var = f"_static_inputs_{node.graph_id}"
        static_outputs_var = f"_static_outputs_{node.graph_id}"

        self._emitter.emit_comment(f"CUDA Graph ID: {node.graph_id}")
        self._emitter.emit_comment(f"Warmup runs: {node.warmup_runs}")
        self._emitter.emit_comment(f"Capture mode: {node.capture_mode}")
        self._emitter.emit_blank_line()

        if node.device_index is not None:
            self._emitter.emit_comment(
                "Set the CUDA device for multi-GPU scenarios."
            )
            self._emitter.emit_comment(
                "CUDA graph capture and replay must occur on the same device."
            )
            self._emitter.emit_comment(
                f"This ensures all operations are captured on device {node.device_index}."
            )
            self._emitter.emit_line(f"torch.cuda.set_device({node.device_index})")
            self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Create CUDA graph object that will hold the captured operations."
        )
        self._emitter.emit_line(f"{graph_var} = torch.cuda.CUDAGraph()")
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Create a dedicated stream for warmup and capture."
        )
        self._emitter.emit_comment(
            "Using a separate stream ensures isolation from concurrent GPU work."
        )
        self._emitter.emit_line(f"{stream_var} = torch.cuda.Stream()")
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Static buffer allocation for CUDA graph capture."
        )
        self._emitter.emit_comment(
            "These buffers hold tensors with stable memory addresses for graph replay."
        )
        self._emitter.emit_comment(
            "During warmup, we'll populate these with actual tensor allocations."
        )
        self._emitter.emit_line(f"{static_inputs_var} = None")
        self._emitter.emit_line(f"{static_outputs_var} = None")
        self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Warmup runs: Execute the compiled function to populate static buffers."
        )
        self._emitter.emit_comment(
            "Warmup is necessary because CUDA graphs capture memory operations,"
        )
        self._emitter.emit_comment(
            "so we need buffers to be allocated before capture."
        )
        self._emitter.emit_comment(
            "Use no_grad() since Inductor kernels use out= ops that don't support autograd."
        )
        self._emitter.emit_line("with torch.no_grad():")
        self._emitter.indent()
        self._emitter.emit_line(f"with torch.cuda.stream({stream_var}):")
        self._emitter.indent()

        self._emitter.emit_line(f"for _warmup_iter in range({node.warmup_runs}):")
        self._emitter.indent()

        self._emitter.emit_comment(
            "On first warmup iteration, allocate static input buffers"
        )
        self._emitter.emit_comment(
            "by cloning the original inputs. This ensures buffers have"
        )
        self._emitter.emit_comment(
            "the correct shape, dtype, and device."
        )
        self._emitter.emit_line(f"if {static_inputs_var} is None:")
        self._emitter.indent()

        args_list = ", ".join(self._argument_names) if self._argument_names else ""
        if args_list:
            self._emitter.emit_line(
                f"{static_inputs_var} = [t.clone() if isinstance(t, torch.Tensor) else t "
                f"for t in [{args_list}]]"
            )
        else:
            self._emitter.emit_comment(
                "No arguments captured - static_inputs will be set by caller"
            )
            self._emitter.emit_line(f"{static_inputs_var} = []")

        self._emitter.dedent()

        self._emitter.emit_blank_line()
        self._emitter.emit_comment("Execute compiled function during warmup")
        # NOTE: compiled_fn mutates (clears) its input list. Make a shallow copy so
        # _static_inputs_* retains the original Tensor references across warmup and
        # capture.
        self._emitter.emit_line(
            f"{static_outputs_var} = {self._compiled_fn_var_name}(list({static_inputs_var}))"
        )

        self._emitter.dedent()
        self._emitter.dedent()
        self._emitter.dedent()

        self._emitter.emit_blank_line()
        self._emitter.emit_comment("Synchronize to ensure warmup is complete before capture")
        self._emitter.emit_line("torch.cuda.current_stream().wait_stream(" + stream_var + ")")

        self._emitter.emit_blank_line()

        pool_var = None
        if node.pool_id is not None:
            pool_var = f"_{node.graph_id}_pool"
            self._emitter.emit_comment(
                "Get a memory pool handle for deterministic graph memory allocation."
            )
            self._emitter.emit_comment(
                "Using a pool ensures consistent memory addresses across graph replays"
            )
            self._emitter.emit_comment(
                "and enables memory sharing between graphs using the same pool."
            )
            self._emitter.emit_line(f"{pool_var} = torch.cuda.graph_pool_handle()")
            self._emitter.emit_blank_line()

        self._emitter.emit_comment(
            "Capture the CUDA graph. All GPU operations inside this context"
        )
        self._emitter.emit_comment(
            "are recorded into the graph rather than executed immediately."
        )
        self._emitter.emit_comment(
            "The captured graph can then be replayed efficiently."
        )
        self._emitter.emit_comment(
            "Use no_grad() since Inductor kernels use out= ops that don't support autograd."
        )
        self._emitter.emit_line("with torch.no_grad():")
        self._emitter.indent()
        if pool_var is not None:
            self._emitter.emit_line(
                f"with torch.cuda.graph({graph_var}, stream={stream_var}, pool={pool_var}):"
            )
        else:
            self._emitter.emit_line(f"with torch.cuda.graph({graph_var}, stream={stream_var}):")
        self._emitter.indent()

        self._emitter.emit_comment(
            "Execute the compiled function to capture its GPU operations."
        )
        self._emitter.emit_comment(
            "The static_outputs will hold references to the output tensors"
        )
        self._emitter.emit_comment(
            "whose memory addresses are stable across replays."
        )
        # NOTE: compiled_fn mutates (clears) its input list. Make a shallow copy so
        # _static_inputs_* retains the original Tensor references for replay.
        self._emitter.emit_line(
            f"{static_outputs_var} = {self._compiled_fn_var_name}(list({static_inputs_var}))"
        )

        self._emitter.dedent()
        self._emitter.dedent()

        self._emitter.emit_blank_line()

        replay_fn_name = f"_replay_{node.graph_id}"
        self._emit_replay_wrapper_function(
            node=node,
            replay_fn_name=replay_fn_name,
            graph_var=graph_var,
            static_inputs_var=static_inputs_var,
            static_outputs_var=static_outputs_var,
        )

        self._cuda_graph_replay_fn = replay_fn_name
        self._cuda_graph_static_outputs_var = static_outputs_var

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

        When CUDA graphs are enabled (indicated by _cuda_graph_replay_fn being set),
        the invocation is replaced with a call to the replay function:
            result = _replay_graph_id([arg1, arg2, arg3])

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

        # When we have a backward kernel, we need to invoke via the autograd
        # function to preserve grad_fn for backward propagation. The autograd
        # function's forward() method will handle the computation. Without this,
        # calling the CUDA graph replay function directly bypasses the autograd
        # function and the output has no grad_fn.
        use_autograd_function = (
            self._has_inductor_backward_kernel and node.is_autograd_function
        )

        if self._cuda_graph_replay_fn is not None and not use_autograd_function:
            args_call = f"[{args_str}]"
            self._emitter.emit_comment(
                "CUDA graphs are enabled - use the replay function instead of"
            )
            self._emitter.emit_comment(
                "calling the compiled function directly. The replay function copies"
            )
            self._emitter.emit_comment(
                "dynamic inputs into static buffers and replays the captured graph."
            )
            needs_extract = node.extract_first_output or (
                self._has_inductor_kernel and self._inductor_returns_tuple
            )

            raw_result_name = node.result_name
            if needs_extract:
                raw_result_name = "_raw_result"

            if self._emit_debug_prints:
                self._emitter.emit_line(
                    f'print("[DEBUG] Invoking CUDA graph replay: '
                    f'{self._cuda_graph_replay_fn}({args_call})")'
                )
            line = f"{raw_result_name} = {self._cuda_graph_replay_fn}({args_call})"
            self._emitter.emit_line(line)

            if needs_extract:
                self._emitter.emit_comment(
                    "Inductor returns (output, saved_tensors...), extract output"
                )
                extract_line = f"{node.result_name} = {raw_result_name}[0]"
                self._emitter.emit_line(extract_line)
            if self._emit_debug_prints:
                self._emitter.emit_line(
                    f'print("[DEBUG] {node.result_name}=", '
                    f'{node.result_name}.shape if hasattr({node.result_name}, "shape") '
                    f"else type({node.result_name}))"
                )
            return line

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
            if self._emit_debug_prints:
                self._emitter.emit_line(f'print("[DEBUG] Invoking {callable_name}({args_call})")')
            line = f"{raw_result_name} = {callable_name}({args_call})"
            self._emitter.emit_line(line)
            if self._emit_debug_prints:
                self._emitter.emit_line(f'print("[DEBUG] {raw_result_name}=", type({raw_result_name}), "len=", len({raw_result_name}) if hasattr({raw_result_name}, "__len__") else "N/A")')
            self._emitter.emit_comment(
                "Inductor returns (output, saved_tensors...), extract output"
            )
            extract_line = f"{node.result_name} = {raw_result_name}[0]"
            self._emitter.emit_line(extract_line)
            if self._emit_debug_prints:
                self._emitter.emit_line(f'print("[DEBUG] {node.result_name}=", {node.result_name}.shape if hasattr({node.result_name}, "shape") else type({node.result_name}))')
            return extract_line
        else:
            if self._emit_debug_prints:
                self._emitter.emit_line(f'print("[DEBUG] Invoking {callable_name}({args_call})")')
            line = f"{node.result_name} = {callable_name}({args_call})"
            self._emitter.emit_line(line)
            if self._emit_debug_prints:
                self._emitter.emit_line(f'print("[DEBUG] {node.result_name}=", {node.result_name}.shape if hasattr({node.result_name}, "shape") else type({node.result_name}))')
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

        self._emit_debug_print(
            f"Final result {node.expose_as}",
            node.expose_as,
            f"{node.expose_as}.shape if hasattr({node.expose_as}, 'shape') else type({node.expose_as})",
        )

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
    emit_debug_prints: bool = False,
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
        emit_debug_prints: If True, emit debug print statements at key points
            in the generated code for tracing execution flow.

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
        emit_debug_prints=emit_debug_prints,
    )

    emitter = visitor.get_emitter()

    visitor.prescan_ir(ir)
    needs_model = visitor.requires_model_access(ir)
    uses_fallback = visitor.uses_object_id_fallback(ir)

    if include_header:
        emitter.emit_comment("Generated by torch.compile(pythonify=...)")
        emitter.emit_comment("This file contains the runtime machinery for the compiled model.")
        emitter.emit_comment("")

        if uses_fallback:
            # Fallback path: show warnings about object-id limitations
            emitter.emit_comment("=" * 72)
            emitter.emit_comment("WARNING: PROCESS-LOCAL FILE - DO NOT SAVE OR REUSE")
            emitter.emit_comment("=" * 72)
            emitter.emit_comment("")
            emitter.emit_comment("This file uses object IDs (memory addresses) to retrieve model")
            emitter.emit_comment("parameters and buffers. These IDs have CRITICAL LIMITATIONS:")
            emitter.emit_comment("")
            emitter.emit_comment("1. SAME-PROCESS ONLY: Object IDs are memory addresses that are")
            emitter.emit_comment("   only valid within the SAME Python process. This file CANNOT")
            emitter.emit_comment("   be used in a different process or after interpreter restart.")
            emitter.emit_comment("")
            emitter.emit_comment("2. KEEP MODEL ALIVE: The original model and its tensors MUST")
            emitter.emit_comment("   stay alive in memory. If the model is deleted or garbage")
            emitter.emit_comment("   collected, the object IDs become dangling references.")
            emitter.emit_comment("")
            emitter.emit_comment("3. NOT PERSISTABLE: Do NOT save this file for later use. It is")
            emitter.emit_comment("   designed ONLY for immediate execution in the same session.")
            emitter.emit_comment("   For serializable models, use torch.export() instead.")
            emitter.emit_comment("")
            emitter.emit_comment("4. CRASH RISK: Using invalid object IDs will crash the Python")
            emitter.emit_comment("   interpreter or cause memory corruption. Never manually edit")
            emitter.emit_comment("   object IDs in this file.")
            emitter.emit_comment("")
            emitter.emit_comment("TL;DR: Execute this file immediately via exec() in the same")
            emitter.emit_comment("       process where torch.compile was called. Do NOT save it.")
            emitter.emit_comment("       For a persistable format, use torch.export() instead.")
            emitter.emit_comment("")
            emitter.emit_comment("=" * 72)
        elif needs_model:
            # Golden path: show documentation for module-based access
            emitter.emit_comment("=" * 72)
            emitter.emit_comment("GOLDEN PATH: Module-Based Parameter Access")
            emitter.emit_comment("=" * 72)
            emitter.emit_comment("")
            emitter.emit_comment("This generated code uses the RECOMMENDED module-based approach for")
            emitter.emit_comment("accessing parameters and buffers. Instead of using ctypes-based")
            emitter.emit_comment("object IDs, it accesses tensors via attribute traversal on the")
            emitter.emit_comment("model object that you provide in the exec() namespace.")
            emitter.emit_comment("")
            emitter.emit_comment("HOW IT WORKS:")
            emitter.emit_comment("  Parameters and buffers are accessed as model.attr or")
            emitter.emit_comment("  model.submodule.attr (e.g., model.layer1.weight). This")
            emitter.emit_comment("  requires the 'model' variable to be in scope when executing.")
            emitter.emit_comment("")
            emitter.emit_comment("ADVANTAGES:")
            emitter.emit_comment("  - PROCESS-PORTABLE: Unlike object IDs, attribute access works")
            emitter.emit_comment("    across different Python processes and interpreter restarts.")
            emitter.emit_comment("  - LIVE ACCESS: Always uses current parameter values. If you")
            emitter.emit_comment("    mutate a parameter after compilation, exec() sees the new value.")
            emitter.emit_comment("  - SAFER: No risk of dangling memory references or crashes from")
            emitter.emit_comment("    stale object IDs.")
            emitter.emit_comment("  - NATURAL: Matches typical nn.Module usage patterns.")
            emitter.emit_comment("")
            emitter.emit_comment("USAGE:")
            emitter.emit_comment("  Pass your model in the exec() namespace. Example:")
            emitter.emit_comment("")
            if model_source == ModelSource.CLOSURE:
                emitter.emit_comment("    namespace = {")
                emitter.emit_comment(f'        "{model_var_name}": your_model,')
                emitter.emit_comment(f'        "{compiled_fn_var_name}": compiled_callable,')
                emitter.emit_comment(f'        "{f_locals_var_name}": {{"x": input_tensor}},')
                emitter.emit_comment("    }")
                emitter.emit_comment("    exec(code, namespace)")
                emitter.emit_comment('    result = namespace["y"]')
            elif model_source == ModelSource.F_LOCALS:
                emitter.emit_comment(f"    {f_locals_var_name} = {{")
                emitter.emit_comment(f'        "{model_var_name}": your_model,')
                emitter.emit_comment('        "x": input_tensor,')
                emitter.emit_comment("    }")
                emitter.emit_comment(f"    exec(code, {{{f_locals_var_name!r}: {f_locals_var_name}, {compiled_fn_var_name!r}: compiled_callable}})")
            else:  # F_GLOBALS
                emitter.emit_comment(f"    {f_globals_var_name} = {{")
                emitter.emit_comment(f'        "{model_var_name}": your_model,')
                emitter.emit_comment(f'        "{compiled_fn_var_name}": compiled_callable,')
                emitter.emit_comment("    }")
                emitter.emit_comment(f"    exec(code, {f_globals_var_name}, {{{f_locals_var_name!r}: ...}})")
            emitter.emit_comment("")
            emitter.emit_comment("=" * 72)

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
