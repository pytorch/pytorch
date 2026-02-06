"""
Automatic line-by-line profiler for PyTorch code.

This tool automatically instruments Python functions with record_function
annotations without requiring manual code modifications.

Usage:
    from torch.profiler.auto_instrumenter import auto_profile_module

    # Instrument a module:
    model = auto_profile_module(model, 'forward')

    # Profile it:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        output = model(x)

    # Export trace:
    prof.export_chrome_trace("trace.json")
"""

import ast
import functools
import inspect
import os
import textwrap
import threading
from typing import Callable, List, Optional

import torch
from torch.profiler import record_function

# Annotation marker for line annotations in profiler traces
# Format: {MARKER}:<filename>:<line_number>:<code>
# Example: @@:so2_layers.py:42:x = torch.matmul(a, b)
ANNOTATION_MARKER = "@@"

# Thread-local storage for loop context stack
_loop_context = threading.local()


def _get_loop_stack():
    """Get the current thread's loop iteration stack."""
    if not hasattr(_loop_context, "stack"):
        _loop_context.stack = []
    return _loop_context.stack


def _get_loop_str():
    """Get current loop context as comma-separated string."""
    stack = _get_loop_stack()
    if not stack:
        return ""
    return ",".join(map(str, stack))


class LineInstrumenter(ast.NodeTransformer):
    """AST transformer that wraps each statement with record_function."""

    def __init__(
        self,
        function_name: str,
        source_file: str = "<unknown>",
        start_line_offset: int = 0,
        instrument_loop_body: bool = True,
    ):
        self.function_name = function_name
        self.source_file = source_file
        self.start_line_offset = start_line_offset
        self.line_counter = 0
        self.instrument_loop_body = instrument_loop_body

    def visit_For(self, node: ast.For):
        """Handle for loops - add iteration counter and push to context stack."""
        if not self.instrument_loop_body:
            return node

        counter_var = f"_loop_iter_{self.line_counter}"
        self.line_counter += 1

        # Initialize counter before loop
        counter_init = ast.Assign(
            targets=[ast.Name(id=counter_var, ctx=ast.Store())],
            value=ast.Constant(value=0),
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        # Instrument loop body
        new_body = []

        # Push loop counter to stack
        push_context = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=ast.Name(id="_get_loop_stack", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=counter_var, ctx=ast.Load())],
                keywords=[],
            )
        )
        new_body.append(push_context)

        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                new_body.append(stmt)
                continue

            try:
                code = ast.unparse(stmt)
            except Exception:
                code = f"statement_{self.line_counter}"

            self.line_counter += 1

            stmt_line = getattr(stmt, "lineno", self.line_counter)
            actual_line = stmt_line + self.start_line_offset
            code_desc = code[:60] + "..." if len(code) > 60 else code

            annotation_str = (
                f"{ANNOTATION_MARKER}:{self.source_file}:{actual_line}:{code_desc}"
            )

            with_node = ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Name(id="record_function", ctx=ast.Load()),
                            args=[ast.Constant(value=annotation_str)],
                            keywords=[],
                        ),
                        optional_vars=None,
                    )
                ],
                body=[self.visit(stmt) if stmt else stmt],
                lineno=stmt_line,
                col_offset=node.col_offset,
            )
            new_body.append(with_node)

        # Increment counter
        counter_increment = ast.AugAssign(
            target=ast.Name(id=counter_var, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
            lineno=node.lineno,
            col_offset=node.col_offset,
        )
        new_body.append(counter_increment)

        # Pop loop counter from stack
        pop_context = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=ast.Name(id="_get_loop_stack", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                    attr="pop",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )
        new_body.append(pop_context)

        node.body = new_body
        return [counter_init, node]

    def _wrap_statement(self, stmt: ast.stmt) -> ast.With:
        """Wrap a statement in record_function."""
        try:
            code = ast.unparse(stmt)
        except Exception:
            code = f"statement_{self.line_counter}"

        self.line_counter += 1

        ast_line = getattr(stmt, "lineno", self.line_counter)
        actual_line = ast_line + self.start_line_offset
        code_desc = code[:60] + "..." if len(code) > 60 else code

        annotation_str = (
            f"{ANNOTATION_MARKER}:{self.source_file}:{actual_line}:{code_desc}"
        )

        with_node = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="record_function", ctx=ast.Load()),
                        args=[ast.Constant(value=annotation_str)],
                        keywords=[],
                    ),
                    optional_vars=None,
                )
            ],
            body=[stmt],
            lineno=ast_line,
            col_offset=0,
        )
        return with_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Instrument function body with line-by-line annotations."""
        if node.name != self.function_name:
            return node

        new_body = []

        for stmt in node.body:
            # Skip docstrings
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                new_body.append(stmt)
                continue

            # Handle loops specially
            if isinstance(stmt, ast.For):
                result = self.visit(stmt)
                if isinstance(result, list):
                    new_body.extend(result)
                else:
                    new_body.append(result)
            else:
                new_body.append(self._wrap_statement(stmt))

        node.body = new_body
        return node


def auto_profile_module(
    module: torch.nn.Module,
    method_name: str = "forward",
    recursive_modules: Optional[List[str]] = None,
) -> torch.nn.Module:
    """
    Automatically instrument a module's method with line-by-line profiling.

    Args:
        module: PyTorch module to instrument
        method_name: Method name to instrument (default: 'forward')
        recursive_modules: List of module attributes to recursively instrument

    Returns:
        Module with instrumented method

    Example:
        >>> model = MyModel()
        >>> model = auto_profile_module(model, 'forward')
        >>> with profile(activities=[ProfilerActivity.CPU]) as prof:
        ...     output = model(input)
        >>> prof.export_chrome_trace("trace.json")
    """
    original_method = getattr(module, method_name)

    # Get source information
    source = inspect.getsource(original_method)
    source_file = inspect.getsourcefile(original_method)

    try:
        source_lines, start_line = inspect.getsourcelines(original_method)
    except (OSError, TypeError):
        start_line = 1

    source = textwrap.dedent(source)

    # Extract filename
    if source_file:
        source_file = os.path.basename(source_file)
    else:
        source_file = "<unknown>"

    # Parse and transform AST
    tree = ast.parse(source)
    instrumenter = LineInstrumenter(
        method_name, source_file, start_line_offset=start_line - 1
    )
    new_tree = instrumenter.visit(tree)

    # Adjust line numbers
    line_offset = start_line - 1
    for node in ast.walk(new_tree):
        if hasattr(node, "lineno"):
            node.lineno += line_offset
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            node.end_lineno += line_offset

    ast.fix_missing_locations(new_tree)

    # Compile and execute
    code = compile(new_tree, filename="<auto_profile_module>", mode="exec")
    namespace = {
        "record_function": record_function,
        "torch": torch,
        "_get_loop_stack": _get_loop_stack,
        "_get_loop_str": _get_loop_str,
        **original_method.__globals__,
    }
    exec(code, namespace)

    instrumented_method = namespace[method_name]

    # Bind to module
    setattr(module, method_name, instrumented_method.__get__(module, type(module)))

    # Recursively instrument sub-modules
    if recursive_modules:
        for attr_name in recursive_modules:
            if hasattr(module, attr_name):
                attr = getattr(module, attr_name)
                if isinstance(attr, torch.nn.ModuleList):
                    for submodule in attr:
                        auto_profile_module(submodule, method_name, recursive_modules=[])
                elif isinstance(attr, torch.nn.Module):
                    auto_profile_module(attr, method_name, recursive_modules=[])

    return module
