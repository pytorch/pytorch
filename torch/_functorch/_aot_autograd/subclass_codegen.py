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
from typing import cast, TYPE_CHECKING

import torch
from torch import SymInt

from .schemas import ActInputPaths, OpaqueMeta, PlainTensorMeta, SubclassCreationMeta
from .utils import import_async_collective_tensor_type


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.distributed._functional_collectives import AsyncCollectiveTensor


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


def _maybe_wait_async_collective_tensor(
    x: object,
    AsyncCollectiveTensor: type["AsyncCollectiveTensor"],
) -> object:
    """Wait on ACT values and leave all other runtime inputs unchanged."""
    if isinstance(x, AsyncCollectiveTensor):
        return cast("AsyncCollectiveTensor", x).trigger_wait()
    return x


def _codegen_unwrap_subclass(
    state: _CodegenState,
    meta: SubclassCreationMeta,
    var: str,
    indent: int = 1,
    include_symints: bool = True,
    act_input_paths: set[tuple[str, ...]] | None = None,
    act_wait_fn: str | None = None,
) -> None:
    """Emit code to recursively unwrap a single subclass input."""
    act_input_paths = act_input_paths or set()
    for attr, attr_meta in meta.attrs.items():
        attr_expr = _safe_attr_access(var, attr)
        attr_act_input_paths = {
            path[1:] for path in act_input_paths if path and path[0] == attr
        }
        match attr_meta:
            case PlainTensorMeta() | OpaqueMeta():
                if attr_act_input_paths:
                    if attr_act_input_paths != {()}:
                        raise AssertionError(
                            f"ACT path for {attr} continues past a leaf meta"
                        )
                    if act_wait_fn is None:
                        raise AssertionError("missing ACT wait function")
                    resolved_var = state.fresh_name("_resolved")
                    state.emit(
                        f"{resolved_var} = {act_wait_fn}({attr_expr})",
                        indent=indent,
                    )
                    state.emit(f"unwrapped_args.append({resolved_var})", indent=indent)
                else:
                    state.emit(
                        f"unwrapped_args.append({attr_expr})",
                        indent=indent,
                    )
            case SubclassCreationMeta():
                if () in attr_act_input_paths:
                    raise AssertionError(f"ACT path for {attr} stops at subclass meta")
                inner_var = state.fresh_name("_inner")
                state.emit(f"{inner_var} = {attr_expr}", indent=indent)
                _codegen_unwrap_subclass(
                    state,
                    attr_meta,
                    inner_var,
                    indent=indent,
                    include_symints=include_symints,
                    act_input_paths=attr_act_input_paths,
                    act_wait_fn=act_wait_fn,
                )

    if include_symints:
        size_placeholders = _compute_placeholders(meta.outer_size)
        stride_placeholders = _compute_placeholders(meta.outer_stride)
        if any(size_placeholders) or any(stride_placeholders):
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
) -> str:
    """Emit code to reconstruct one subclass output. Returns the variable name."""
    inner_dict_var = state.fresh_name("_out_inner")
    entries: list[str] = []
    attr_exprs: dict[str, str] = {}

    for attr, attr_meta in meta.attrs.items():
        match attr_meta:
            case PlainTensorMeta() | OpaqueMeta():
                attr_expr = state.fresh_name("_out_attr")
                state.emit(f"{attr_expr} = unwrapped_outs[_out_idx]")
                state.emit("_out_idx += 1")
            case SubclassCreationMeta():
                attr_expr = _codegen_wrap_subclass(state, attr_meta)
        attr_exprs[attr] = attr_expr
        entries.append(f"{attr!r}: {attr_expr}")

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
                sym_expr = state.fresh_name("_out_sym")
                state.emit(f"{sym_expr} = unwrapped_outs[_out_idx]")
                state.emit("_out_idx += 1")
                parts.append(sym_expr)
            else:
                parts.append(repr(_concrete_value(val)))
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    def _consume_placeholders(placeholders: list[bool]) -> None:
        num_placeholders = sum(placeholders)
        if num_placeholders:
            state.emit("if _has_subclass_symint_outputs:")
            state.emit(f"_out_idx += {num_placeholders}", indent=2)

    outer_size_from_attr = meta.outer_size_from_attr
    outer_stride_from_attr = meta.outer_stride_from_attr
    if outer_size_from_attr is not None:
        size_expr = f"{attr_exprs[outer_size_from_attr]}.size()"
        _consume_placeholders(size_placeholders)
    else:
        size_expr = _build_tuple(meta.outer_size, size_placeholders)

    if outer_stride_from_attr is not None:
        stride_expr = f"{attr_exprs[outer_stride_from_attr]}.stride()"
        _consume_placeholders(stride_placeholders)
    else:
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


def _count_output_args(
    meta: PlainTensorMeta | SubclassCreationMeta,
    *,
    include_subclass_symints: bool,
) -> int:
    if isinstance(meta, PlainTensorMeta):
        return 1

    total = 0
    for attr_meta in meta.attrs.values():
        if isinstance(attr_meta, OpaqueMeta):
            total += 1
        else:
            total += _count_output_args(
                attr_meta, include_subclass_symints=include_subclass_symints
            )

    if include_subclass_symints:
        total += sum(_compute_placeholders(meta.outer_size))
        total += sum(_compute_placeholders(meta.outer_stride))
    return total


def _emit_output_wrapping(
    state: _CodegenState,
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
) -> tuple[list[str], int]:
    """Emit wrapping code for output metas.

    Returns (result_exprs, num_args_tallied) where result_exprs are Python
    expression strings referencing each wrapped output.
    """
    result_exprs: list[str] = []
    num_args_tallied = 0
    saved_for_bw = num_fw_outs_saved_for_bw or 0
    expected_with_symints = (
        sum(
            _count_output_args(meta, include_subclass_symints=True)
            for meta in out_metas
        )
        + saved_for_bw
    )
    state.emit("_out_idx = 0")
    state.emit(
        f"_has_subclass_symint_outputs = len(unwrapped_outs) == {expected_with_symints}"
    )

    for meta in out_metas:
        if isinstance(meta, PlainTensorMeta):
            result_exprs.append(f"unwrapped_outs[{meta.unwrapped_idx}]")
            num_args_tallied += 1
            state.emit(f"_out_idx = max(_out_idx, {meta.unwrapped_idx + 1})")
        else:
            result_var = _codegen_wrap_subclass(state, meta)
            result_exprs.append(result_var)
            num_args_tallied += meta.arg_count

    return result_exprs, num_args_tallied


def _emit_input_unwrapping(
    state: _CodegenState,
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    frozen_inp_indices: frozenset[int] = frozenset(),
    include_symints: bool = True,
    act_input_paths_by_input: dict[int, set[tuple[str, ...]]] | None = None,
    act_wait_fn: str | None = None,
) -> None:
    """Emit unwrapping code for input metas into unwrapped_args.

    Caller must have already emitted ``unwrapped_args = []``.
    """
    act_input_paths_by_input = act_input_paths_by_input or {}
    for i, meta in enumerate(inp_metas):
        input_act_paths = act_input_paths_by_input.get(i, set())
        if isinstance(meta, PlainTensorMeta):
            if input_act_paths:
                if input_act_paths != {()}:
                    raise AssertionError(
                        f"ACT path for input {i} continues past a plain meta"
                    )
                if act_wait_fn is None:
                    raise AssertionError("missing ACT wait function")
                state.emit(f"unwrapped_args.append({act_wait_fn}(args[{i}]))")
            else:
                state.emit(f"unwrapped_args.append(args[{i}])")
        elif i in frozen_inp_indices:
            # Frozen by inductor freezing: constant already baked into graph.
            state.emit("unwrapped_args.append(None)")
        else:
            if () in input_act_paths:
                raise AssertionError(f"ACT path for input {i} stops at subclass meta")
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
                state,
                meta,
                inp_var,
                indent=1,
                include_symints=include_symints,
                act_input_paths=input_act_paths,
                act_wait_fn=act_wait_fn,
            )


def _codegen_subclass_wrapper_source(
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_paths: ActInputPaths | None = None,
) -> tuple[str, dict[str, object]]:
    """Generate source and globals for a subclass wrapper.

    Returns (source, globals_dict).  The globals_dict will NOT contain
    ``compiled_fn`` — the caller is responsible for adding it before exec.
    """
    state = _CodegenState()

    state.emit("def inner_fn(args):", indent=0)

    act_input_paths_by_input: dict[int, set[tuple[str, ...]]] = {}
    act_wait_fn = None
    if act_input_paths:
        AsyncCollectiveTensor = import_async_collective_tensor_type()
        act_wait_fn = state.add_global(
            state.fresh_name("_maybe_wait_act"),
            functools.partial(
                _maybe_wait_async_collective_tensor,
                AsyncCollectiveTensor=AsyncCollectiveTensor,
            ),
        )
        for i, attr_path in act_input_paths:
            act_input_paths_by_input.setdefault(i, set()).add(attr_path)

    # --- Input unwrapping ---
    state.emit("unwrapped_args = []")
    _emit_input_unwrapping(
        state,
        inp_metas,
        frozen_inp_indices=frozen_inp_indices,
        act_input_paths_by_input=act_input_paths_by_input,
        act_wait_fn=act_wait_fn,
    )

    # Pass through any trailing args not covered by inp_metas
    # (e.g. rng seed/offset added by FunctionalizedRngRuntimeWrapper).
    num_inp_metas = len(inp_metas)
    state.emit(f"unwrapped_args.extend(args[{num_inp_metas}:])")
    state.emit("args.clear()")

    # --- Call compiled function ---
    state.emit("unwrapped_outs = compiled_fn(unwrapped_args)")

    # --- Output wrapping ---
    result_exprs, _ = _emit_output_wrapping(state, out_metas, num_fw_outs_saved_for_bw)
    result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
    if num_fw_outs_saved_for_bw is not None:
        state.emit(
            f"_activation_start = len(unwrapped_outs) - {num_fw_outs_saved_for_bw}"
        )
        state.emit(f"return {result_tuple} + tuple(unwrapped_outs[_activation_start:])")
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
    result_exprs, _ = _emit_output_wrapping(
        state, out_metas, num_fw_outs_saved_for_bw=None
    )
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

    # Use a path under torch/_functorch/ so the code object is recognized by
    # dynamo's MOD_SKIPLIST. The eval frame hook stays active during the entire
    # torch.compile(fn)(*args) call (to handle graph breaks and resume functions),
    # so codegen'd functions called during backward get intercepted even though
    # no tracing is active. A real path makes them skip automatically.
    code = compile(source, f"{__file__}:codegen({artifact_name})", "exec")
    local_dict: dict[str, object] = {}
    exec(code, globals_dict, local_dict)
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
    act_input_paths: ActInputPaths | None = None,
) -> Callable[..., object]:
    """Generate a specialized wrapper function for subclass unwrap/wrap."""
    source, globals_dict = _codegen_subclass_wrapper_source(
        inp_metas,
        out_metas,
        num_fw_outs_saved_for_bw,
        frozen_inp_indices,
        act_input_paths=act_input_paths,
    )
    globals_dict["compiled_fn"] = compiled_fn
    return _compile_and_exec_source(
        source, globals_dict, "inner_fn", "subclass_wrapper", wrapped_fn=compiled_fn
    )
