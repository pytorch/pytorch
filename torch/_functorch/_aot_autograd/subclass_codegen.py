"""
Codegen for AOTDispatchSubclassWrapper.

Generates a Python function that replaces the data-driven
runtime_unwrap_tensor_subclasses / wrap_tensor_subclasses loop with
a straight-line function where all metadata (indices, attr names,
subclass types, symint positions) is baked in at compile time.
"""

import functools
import keyword
import logging
from collections.abc import Callable, Iterable

import torch
from torch import SymInt

from .schemas import OpaqueMeta, PlainTensorMeta, SubclassCreationMeta


log = logging.getLogger(__name__)


def _is_symint_placeholder(x: None | int | SymInt) -> bool:
    """Check whether a size/stride entry is symbolic and needs a runtime value.

    Works both before make_runtime_safe() (entries are SymInt) and after
    (symbolic entries replaced with None, nested ints with -1).
    """
    if x is None:
        return True
    if isinstance(x, SymInt) and not x.node.is_nested_int():
        return True
    return False


def _compute_placeholders(outer: Iterable[None | int | SymInt]) -> list[bool]:
    return [_is_symint_placeholder(s) for s in outer]


def _safe_attr_access(var: str, attr: str) -> str:
    if attr.isidentifier() and not keyword.iskeyword(attr):
        return f"{var}.{attr}"
    return f"getattr({var}, {attr!r})"


class _CodegenState:
    """Accumulates lines of generated source and global bindings."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.globals: dict[str, object] = {}
        self._name_counter: int = 0

    def emit(self, line: str, indent: int = 1) -> None:
        self.lines.append("    " * indent + line)

    def fresh_name(self, prefix: str) -> str:
        name = f"{prefix}_{self._name_counter}"
        self._name_counter += 1
        return name

    def add_global(self, name: str, value: object) -> str:
        self.globals[name] = value
        return name


def _codegen_unwrap_subclass(
    state: _CodegenState,
    meta: SubclassCreationMeta,
    var: str,
    indent: int = 1,
    include_symints: bool = True,
) -> None:
    """Emit code to recursively unwrap a single subclass input."""
    for attr, attr_meta in meta.attrs.items():
        match attr_meta:
            case PlainTensorMeta() | OpaqueMeta():
                state.emit(
                    f"unwrapped_args.append({_safe_attr_access(var, attr)})",
                    indent=indent,
                )
            case SubclassCreationMeta():
                inner_var = state.fresh_name("_inner")
                state.emit(
                    f"{inner_var} = {_safe_attr_access(var, attr)}", indent=indent
                )
                _codegen_unwrap_subclass(
                    state,
                    attr_meta,
                    inner_var,
                    indent=indent,
                    include_symints=include_symints,
                )

    # Emit symint extraction
    if include_symints:
        size_placeholders = _compute_placeholders(meta.outer_size)
        stride_placeholders = _compute_placeholders(meta.outer_stride)
        has_size_symints = any(size_placeholders)
        has_stride_symints = any(stride_placeholders)

        if has_size_symints or has_stride_symints:
            size_var = state.fresh_name("_size")
            state.emit(f"{size_var} = {var}.size()", indent=indent)
            for i, is_sym in enumerate(size_placeholders):
                if is_sym:
                    state.emit(f"unwrapped_args.append({size_var}[{i}])", indent=indent)

            stride_var = state.fresh_name("_stride")
            state.emit(f"{stride_var} = {var}.stride()", indent=indent)
            for i, is_sym in enumerate(stride_placeholders):
                if is_sym:
                    state.emit(
                        f"unwrapped_args.append({stride_var}[{i}])", indent=indent
                    )


def _concrete_value(val: None | int | SymInt) -> int:
    """Get the concrete int value for a non-symbolic size/stride entry.

    Used for entries that are NOT symbolic placeholders, meaning they are
    concrete ints or nested ints (represented as -1 after make_runtime_safe).
    """
    if isinstance(val, int):
        return val
    # Before make_runtime_safe: nested ints are SymInts; use -1 as dummy.
    # After make_runtime_safe: they're already -1.
    if isinstance(val, SymInt) and val.node.is_nested_int():
        return -1
    raise AssertionError(f"Expected concrete int, got {type(val)}: {val}")


def _codegen_wrap_subclass(
    state: _CodegenState,
    meta: SubclassCreationMeta,
    out_idx_ref: list[int],
) -> str:
    """Emit code to reconstruct one subclass output. Returns the variable name."""
    inner_dict_var = state.fresh_name("_out_inner")
    entries: list[str] = []

    for attr, attr_meta in meta.attrs.items():
        match attr_meta:
            case PlainTensorMeta() | OpaqueMeta():
                idx = out_idx_ref[0]
                out_idx_ref[0] += 1
                entries.append(f"{attr!r}: unwrapped_outs[{idx}]")
            case SubclassCreationMeta():
                nested_var = _codegen_wrap_subclass(state, attr_meta, out_idx_ref)
                entries.append(f"{attr!r}: {nested_var}")

    state.emit(f"{inner_dict_var} = {{{', '.join(entries)}}}")

    # Reconstruct outer_size and outer_stride
    size_placeholders = _compute_placeholders(meta.outer_size)
    stride_placeholders = _compute_placeholders(meta.outer_stride)

    def _build_tuple(
        outer: Iterable[None | int | SymInt], placeholders: list[bool]
    ) -> str:
        parts: list[str] = []
        for val, is_sym in zip(outer, placeholders):
            if is_sym:
                idx = out_idx_ref[0]
                out_idx_ref[0] += 1
                parts.append(f"unwrapped_outs[{idx}]")
            else:
                parts.append(repr(_concrete_value(val)))
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    size_expr = _build_tuple(meta.outer_size, size_placeholders)
    stride_expr = _build_tuple(meta.outer_stride, stride_placeholders)

    type_name = state.add_global(
        state.fresh_name("_subclass_type"),
        meta.original_subclass_type or type(meta.original_subclass),
    )
    meta_name = state.add_global(state.fresh_name("_meta"), meta.meta)

    result_var = state.fresh_name("_out")
    state.emit(
        f"{result_var} = {type_name}.__tensor_unflatten__("
        f"{inner_dict_var}, {meta_name}, {size_expr}, {stride_expr})"
    )
    return result_var


def _emit_output_wrapping(
    state: _CodegenState,
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
) -> tuple[list[str], int]:
    """Emit wrapping code for output metas.

    Returns (result_exprs, num_args_tallied) where result_exprs are Python
    expression strings referencing each wrapped output.
    """
    out_idx_ref = [0]
    result_exprs: list[str] = []
    num_args_tallied = 0

    for meta in out_metas:
        if isinstance(meta, PlainTensorMeta):
            result_exprs.append(f"unwrapped_outs[{meta.unwrapped_idx}]")
            num_args_tallied += 1
            out_idx_ref[0] = max(out_idx_ref[0], meta.unwrapped_idx + 1)
        else:
            result_var = _codegen_wrap_subclass(state, meta, out_idx_ref)
            result_exprs.append(result_var)
            num_args_tallied += meta.arg_count

    return result_exprs, num_args_tallied


def _emit_input_unwrapping(
    state: _CodegenState,
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    frozen_inp_indices: frozenset[int] = frozenset(),
    include_symints: bool = True,
) -> None:
    """Emit unwrapping code for input metas into unwrapped_args.

    Caller must have already emitted ``unwrapped_args = []``.
    """
    for i, meta in enumerate(inp_metas):
        if isinstance(meta, PlainTensorMeta):
            state.emit(f"unwrapped_args.append(args[{i}])")
        elif i in frozen_inp_indices:
            # Frozen by inductor freezing: constant already baked into graph.
            state.emit("unwrapped_args.append(None)")
        else:
            inp_var = state.fresh_name("_inp")
            type_name = state.add_global(
                state.fresh_name("_expected_type"),
                meta.original_subclass_type or type(meta.original_subclass),
            )
            state.emit(f"{inp_var} = args[{i}]")
            state.emit(
                f"assert type({inp_var}) is {type_name}, "
                f"f'expected {{{type_name}}}, got {{type({inp_var})}}'",
            )
            _codegen_unwrap_subclass(
                state, meta, inp_var, indent=1, include_symints=include_symints
            )


def _codegen_subclass_wrapper_source(
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_indices: list[int] | None = None,
) -> tuple[str, dict[str, object]]:
    """Generate source and globals for a subclass wrapper.

    Returns (source, globals_dict).  The globals_dict will NOT contain
    ``compiled_fn`` — the caller is responsible for adding it before exec.
    """
    state = _CodegenState()

    state.emit("def inner_fn(args):", indent=0)

    # --- Resolve AsyncCollectiveTensors ---
    # ACTs are transient eager-mode wrappers for async collective overlap.
    # Inductor triton kernels bypass __torch_dispatch__, so we must call
    # trigger_wait() before the compiled graph uses the data.
    if act_input_indices:
        for i in act_input_indices:
            state.emit(f"args[{i}] = args[{i}].trigger_wait()")

    # --- Input unwrapping ---
    state.emit("unwrapped_args = []")
    _emit_input_unwrapping(state, inp_metas, frozen_inp_indices=frozen_inp_indices)

    # Pass through any trailing args not covered by inp_metas
    # (e.g. rng seed/offset added by FunctionalizedRngRuntimeWrapper).
    num_inp_metas = len(inp_metas)
    state.emit(f"unwrapped_args.extend(args[{num_inp_metas}:])")
    state.emit("args.clear()")

    # --- Call compiled function ---
    state.emit("unwrapped_outs = compiled_fn(unwrapped_args)")

    # --- Output wrapping ---
    result_exprs, num_args_tallied = _emit_output_wrapping(state, out_metas)
    result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
    if num_fw_outs_saved_for_bw is not None:
        state.emit(
            f"return {result_tuple} + tuple(unwrapped_outs[{num_args_tallied}:])"
        )
    else:
        state.emit(f"return {result_tuple}")

    source = "\n".join(state.lines)
    return source, state.globals


def _codegen_subclass_wrap_source(
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
) -> tuple[str, dict[str, object]]:
    """Generate source for wrapping flat outputs into subclasses.

    Used for the backward epilogue. Shares output-wrapping logic with
    _codegen_subclass_wrapper_source via _emit_output_wrapping.
    """
    state = _CodegenState()
    state.emit("def wrap_fn(unwrapped_outs):", indent=0)
    result_exprs, _ = _emit_output_wrapping(state, out_metas)
    result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
    state.emit(f"return {result_tuple}")
    source = "\n".join(state.lines)
    return source, state.globals


def _compile_and_exec_source(
    source: str,
    globals_dict: dict[str, object],
    fn_name: str,
    artifact_name: str,
    wrapped_fn: Callable[..., object] | None = None,
) -> Callable[..., object]:
    """Compile generated source, exec it, and return the named function.

    If wrapped_fn is provided, applies functools.update_wrapper so that
    __wrapped__ and __dict__ (e.g. _fx_graph_cache_key) propagate to the
    generated function.
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Generated %s:\n%s", artifact_name, source)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: source,
    )

    code = compile(source, f"<{artifact_name}>", "exec")
    local_dict: dict[str, object] = {}
    exec(code, globals_dict, local_dict)  # noqa: S102
    fn = local_dict[fn_name]
    if wrapped_fn is not None:
        functools.update_wrapper(fn, wrapped_fn)  # type: ignore[arg-type]
    return fn  # type: ignore[return-value]


def codegen_backward_subclass_fns(
    grad_input_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = None,
) -> tuple[Callable[..., object], Callable[..., object] | None]:
    """Generate codegen'd unwrap and wrap functions for the backward pass.

    Returns (unwrap_fn, wrap_fn). unwrap_fn is used by the backward prologue
    to unwrap non-tangent subclass inputs (always an identity in AOT dispatch
    since the compiled forward operates on unwrapped inner tensors). wrap_fn
    is used by the backward epilogue to wrap flat grad inputs back into
    subclasses; it is None when grad_input_metas is None.
    """
    source = "def unwrap_fn(args):\n    return list(args)"
    globals_dict: dict[str, object] = {}
    unwrap_fn = _compile_and_exec_source(
        source, globals_dict, "unwrap_fn", "backward_subclass_unwrap"
    )

    wrap_fn = None
    if grad_input_metas is not None:
        wrap_source, wrap_globals = _codegen_subclass_wrap_source(grad_input_metas)
        wrap_fn = _compile_and_exec_source(
            wrap_source, wrap_globals, "wrap_fn", "backward_subclass_wrapper"
        )

    return unwrap_fn, wrap_fn


def codegen_subclass_wrapper(
    compiled_fn: Callable[..., object],
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_indices: list[int] | None = None,
) -> Callable[..., object]:
    """Generate a specialized wrapper function for subclass unwrap/wrap."""
    source, globals_dict = _codegen_subclass_wrapper_source(
        inp_metas,
        out_metas,
        num_fw_outs_saved_for_bw,
        frozen_inp_indices,
        act_input_indices=act_input_indices,
    )
    globals_dict["compiled_fn"] = compiled_fn
    return _compile_and_exec_source(
        source, globals_dict, "inner_fn", "subclass_wrapper", wrapped_fn=compiled_fn
    )
