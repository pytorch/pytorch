"""Dual-path FX CodeGen with zero-overhead profiler support.

Generates two forward paths from FX graph IR:
  - ``_forward_impl``: clean code, zero profiler overhead
  - ``_forward_profiled``: per-node ``_RecordFunctionFast`` wrapping

At runtime, ``forward()`` dispatches based on
``torch.autograd.profiler._is_profiler_enabled``.

Usage::

    from torch.fx.profiler_codegen import ProfilerCodeGen

    gm = torch.export.export(model, args).module()
    gm.graph.set_codegen(ProfilerCodeGen())
    gm.recompile()

    # Without profiler: runs _forward_impl (zero overhead)
    output = gm(input)

    # With profiler: runs _forward_profiled (per-op RecordFunctionFast)
    with torch.profiler.profile() as prof:
        output = gm(input)
"""

import re
from typing import Any

import torch
import torch.fx
from torch.fx.graph import (
    _counter_regexp,
    _Namespace,
    CodeGen,
    PythonCode,
)
from torch.fx.node import Node


class ProfilerCodeGen(CodeGen):
    """CodeGen that emits _forward_impl, _forward_profiled, and a forward dispatcher."""

    def __init__(self) -> None:
        super().__init__()

    def _gen_python_code(
        self,
        nodes,
        root_module: str,
        namespace: _Namespace,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        expanded_def: bool = False,
        record_func: bool = False,
        additional_meta: list[str] | None = None,
    ) -> PythonCode:
        """Generate dual-path source from FX graph nodes.

        Maintains two body lists (``impl_body``, ``profiled_body``) and
        populates both in a single pass over the nodes.  The
        ``record_func`` parameter is ignored.
        """
        ctx = self._create_codegen_context(
            nodes, root_module, namespace, expanded_def=expanded_def
        )

        impl_body: list[str] = []
        profiled_body: list[str] = []

        for i, node in enumerate(nodes):
            impl_body.append(f"# COUNTER: {i}\n")
            profiled_body.append(f"# COUNTER: {i}\n")

            is_recordable = node.op in (
                "call_function",
                "call_method",
                "call_module",
            )

            body_len_before = len(impl_body)
            self._codegen_emit_node(ctx, node, impl_body)
            emitted_lines = impl_body[body_len_before:]

            if is_recordable and emitted_lines:
                label = self._get_profiler_label(node).replace('"', '\\"')
                args_tuple = self._format_args_tuple(node)
                rf_var = f"_rf_{repr(node)}"
                profiled_body.append(
                    f"{rf_var} = torch._C._profiler._RecordFunctionFast("
                    f'"{label}", {args_tuple}); {rf_var}.__enter__()\n'
                )
                profiled_body.extend(emitted_lines)
                self._codegen_delete_unused_values(
                    ctx, node, [impl_body, profiled_body]
                )
                profiled_body.append(f"{rf_var}.__exit__(None, None, None)\n")
            else:
                profiled_body.extend(emitted_lines)
                self._codegen_delete_unused_values(
                    ctx, node, [impl_body, profiled_body]
                )

        if len(impl_body) == 0:
            impl_body.append("pass\n")
            profiled_body.append("pass\n")

        if len(ctx.wrapped_fns) > 0:
            wrap_name = self._codegen_add_global(ctx, "wrap", torch.fx.wrap)
            wrap_stmts = "\n".join(
                [f'{wrap_name}("{name}")' for name in ctx.wrapped_fns]
            )
        else:
            wrap_stmts = ""

        if self._body_transformer:
            impl_body = self._body_transformer(impl_body)
            profiled_body = self._body_transformer(profiled_body)

        for name, value in self.additional_globals():
            self._codegen_add_global(ctx, name, value)

        impl_prologue = self._gen_fn_def_with_name(
            "_forward_impl",
            ctx.free_vars,
            ctx.maybe_return_annotation[0],
            expanded_def=expanded_def,
        )
        impl_code, impl_lineno_map, impl_prologue_start = self._assemble_function(
            impl_prologue, impl_body, ""
        )

        profiled_prologue = self._gen_fn_def_with_name(
            "_forward_profiled",
            ctx.free_vars,
            ctx.maybe_return_annotation[0],
            expanded_def=expanded_def,
        )
        profiled_code, _, _ = self._assemble_function(
            profiled_prologue,
            profiled_body,
            "",
        )

        dispatcher_code = self._generate_dispatcher(ctx.free_vars)

        profiler_import = "import torch.autograd.profiler as _autograd_profiler"
        prefix = f"\n{wrap_stmts}\n\n{profiler_import}\n\n"
        combined = f"{prefix}{impl_code}\n\n{profiled_code}\n\n{dispatcher_code}\n"

        prefix_lines = prefix.count("\n")
        shifted_lineno_map = {k + prefix_lines: v for k, v in impl_lineno_map.items()}

        return PythonCode(
            combined,
            ctx.globals_,
            _lineno_map=shifted_lineno_map,
            _prologue_start=impl_prologue_start + prefix_lines,
        )

    def _gen_fn_def_with_name(
        self,
        func_name: str,
        free_vars: list[str],
        return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        """Generate a ``def`` line with the given function name."""
        vars_copy = list(free_vars)
        if len(vars_copy) == 0 or vars_copy[0] != "self":
            vars_copy.insert(0, "self")
        if expanded_def:
            args_formatted = self._format_multiline_args(vars_copy)
            return f"def {func_name}(\n{args_formatted}){return_annotation}:"
        return f"def {func_name}({', '.join(vars_copy)}){return_annotation}:"

    def _assemble_function(
        self,
        prologue: str,
        body: list[str],
        wrap_stmts: str,
    ) -> tuple[str, dict[int, int | None], int]:
        """Assemble function source from prologue + body.

        Strips ``# COUNTER:`` comments and builds ``lineno_map``.
        """
        lineno_map: dict[int, int | None] = {}
        prologue_len = prologue.count("\n") + 1
        new_lines: list[str] = []
        cur_idx = None
        for line in "".join(body).split("\n"):
            counter = _counter_regexp.search(line)
            if counter is not None:
                cur_idx = int(counter.group(1))
            else:
                lineno_map[len(new_lines) + prologue_len] = cur_idx
                new_lines.append(line)

        code = "\n".join(new_lines).lstrip("\n")
        code = "\n".join("    " + line for line in code.split("\n"))

        fn_code = f"""
{wrap_stmts}

{prologue}
{code}"""
        prologue_start = wrap_stmts.count("\n") + 4
        return fn_code, lineno_map, prologue_start

    def _generate_dispatcher(self, free_vars: list[str]) -> str:
        """Generate ``forward()`` that dispatches on profiler state."""
        vars_copy = list(free_vars)
        if len(vars_copy) == 0 or vars_copy[0] != "self":
            vars_copy.insert(0, "self")
        params = ", ".join(vars_copy)
        call_args = ", ".join(v.split(":")[0].split("=")[0].strip() for v in vars_copy)
        return (
            f"def forward({params}):\n"
            f"    if _autograd_profiler._is_profiler_enabled:\n"
            f"        return _forward_profiled({call_args})\n"
            f"    return _forward_impl({call_args})\n"
        )

    def _get_profiler_label(self, node: Node) -> str:
        """Format: ``"node_name: op_name (file:line)"``."""
        op_name = self._get_op_name(node)
        source_loc = self._get_source_location(node)
        label = f"{node.name}: {op_name}"
        if source_loc:
            label = f"{label} ({source_loc})"
        return label

    def _get_op_name(self, node: Node) -> str:
        if node.op == "call_function":
            if hasattr(node.target, "__name__"):
                return node.target.__name__
            elif hasattr(node.target, "_name"):
                return node.target._name
            return str(node.target).split(".")[-1]
        return str(node.target)

    _SOURCE_LOC_PATTERN = re.compile(r'^File "(.+)", line (\d+), in (.+)$')

    def _get_source_location(self, node: Node) -> str:
        """Extract ``filename:lineno`` from the node's stack trace.

        Skips internal PyTorch and site-packages frames.
        """
        stack_trace = node.meta.get("stack_trace", None)
        if not stack_trace:
            return ""

        lines = stack_trace.strip().split("\n")
        for idx in range(len(lines) - 2, -1, -1):
            line = lines[idx].strip()
            match = self._SOURCE_LOC_PATTERN.match(line)
            if match:
                filepath = match.group(1)
                lineno = match.group(2)
                if "/torch/" in filepath or "/site-packages/" in filepath:
                    continue
                filename = filepath.split("/")[-1]
                return f"{filename}:{lineno}"
        return ""

    def _format_args_tuple(self, node: Node) -> str:
        """Format node's tensor args as a tuple for ``_RecordFunctionFast``."""
        parts: list[str] = []

        def collect_nodes(item: Any) -> None:
            if isinstance(item, Node):
                parts.append(repr(item))
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    collect_nodes(sub_item)

        for arg in node.args:
            collect_nodes(arg)
        for val in node.kwargs.values():
            collect_nodes(val)

        if not parts:
            return "()"
        return f"({', '.join(parts)},)"
