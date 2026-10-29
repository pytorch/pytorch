"""Strict executable backend over experimental C++ LoopsIR/KernelIR/TiledIR.

Static single-CUDA-device inference only. Generated Compute covers f32
add (alpha exactly 1), mul, and relu with eager FP boundaries and operand
order preserved. Every reduction (audited ``torch.Tag.reduction``) and every
other pure tensor/custom op is an explicit structured eager extern call with
the real symbolic target. Views are wrapper-representable aliases only
(no hidden copies). ``alpha != 1`` add falls back to an exact aten.add
extern (never mul+add, which could fuse/contract differently).

Numerical policy (``config.experimental_cpp_ir_numerics``, validated only in
this lane via the shared ``cpp_ir_numerics()`` reader): ``strict_eager``
(default) emits explicit rounded arithmetic with no contraction, no FTZ, and
NaN-sensitive ReLU; ``inductor`` follows stock's effective choices (FP fusion
controlled by ``emulate_precision_casts``, FTZ by
``eager_numerics.disable_ftz``) without forcing ``use_fast_math=True``.
Codegen consumes the already-selected TiledIR policy; it never chooses
fusion, layouts, memory reuse, or tiling.

Pipeline (all C++ except Triton emission and runtime dispatch):

1. AOT capture via ``torch._dynamo.backends.common.aot_autograd``
   (fw_compiler only, no Inductor decompositions/optimizations) to obtain
   the functionalized ATen FX graph and lifted params.
2. ``lower_fx_to_cpp(..., executable=True)`` -> verified LoopsIR Program.
3. ``cpp.optimize_program`` -> ``cpp.lower_to_kernels`` -> ``cpp.fuse_kernels``
   (vertical pointwise only; never across externs).
4. ``cpp.lower_to_tiled(KP, TileOptions)`` -> TiledProgram.
5. ``emit_triton`` per tiled function (``@triton.jit`` + normal
   ``@triton_heuristics.fixed_config``, exact function name, ``tl.where``
   ReLU, ``enable_fp_fusion=False``/``disable_ftz=True`` numerics) embedded
   in a file-backed straight-line ``call(args)`` module
   (``codegen/loops_ir_wrapper.py``); kernels resolve through normal
   ``AsyncCompile.triton`` + ``async_compile.wait(globals())`` and launch
   via ``CachingAutotuner.run(*ptrs, grid, 1, 1, stream=raw_stream)`` with
   the function's own static grid; empty grids are skipped.
6. Extern calls run exact eager ops resolved from the symbolic target;
   opaque results come from the real call (never a preallocated copy).
   Intermediate generated results are flat ``torch.empty`` spans on CUDA
   allocated per call; final outputs are ``as_strided`` views with
   ``storage_offset + IR-relative offset`` over their backing buffers, or
   direct aliases when compile-time metadata proves the backing layout
   matches.

Compile-time plan freeze: every C++ IR property (views, descriptors,
CallArgs, launch metadata, output spec) is snapshotted into pure-Python
plan tuples BEFORE the wrapper module is emitted. The per-invocation hot
path touches no C++ IR object -- only ``call``-local tensors, ``torch.*``
ops, preconverted literals, and cached launch handles. No Python callbacks
enter the C++ IR; the plans are wrapper metadata outside the IR.

Do NOT build the C++ extension on import: all ``load_cpp_ir`` calls happen
lazily inside compile execution. This module is an opt-in
``torch.compile(backend=loops_ir_backend)`` callable; no global backend
registration. ``loops_ir_backend`` itself is a thin forwarder to ordinary
``compile_fx`` with ``config.experimental_cpp_ir=True``; the production
seam ``compile_fx_loops_ir_output`` (called by the frontend worker after
post-grad) builds the frozen plan, emits/loads the wrapper module, and
returns the existing ``CompiledFxGraph``.
"""

from __future__ import annotations

import contextvars
import dataclasses
from typing import Any, NamedTuple

import torch

from torch._inductor.loops_ir import load_cpp_ir, LoopsIrUnsupportedError


# ---------------------------------------------------------------------------
# Compilation artifacts (attached to the returned callable for orchestration).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LoopsIrArtifacts:
    """Inspectable compilation record; no delegated codegen inside.

    - ``program_dump`` / ``kernel_dump`` / ``tiled_dump``: verified IR text.
    - ``triton_sources``: function name -> emitted ``@triton.jit`` source.
    - ``triton_ptx``: function name -> PTX string when retrievable.
    - ``num_generated_kernels`` / ``num_extern_calls``: coverage counts.
    - ``function_names``: tiled function names in launch order.
    """

    program_dump: str = ""
    optimized_program_dump: str = ""
    kernels_before_fusion: int = 0
    kernel_dump: str = ""
    tiled_dump: str = ""
    triton_sources: dict[str, str] = dataclasses.field(default_factory=dict)
    triton_ptx: dict[str, str] = dataclasses.field(default_factory=dict)
    num_generated_kernels: int = 0
    num_extern_calls: int = 0
    function_names: list[str] = dataclasses.field(default_factory=list)

    def clear(self) -> None:
        self.program_dump = ""
        self.optimized_program_dump = ""
        self.kernels_before_fusion = 0
        self.kernel_dump = ""
        self.tiled_dump = ""
        self.triton_sources.clear()
        self.triton_ptx.clear()
        self.num_generated_kernels = 0
        self.num_extern_calls = 0
        self.function_names.clear()


@dataclasses.dataclass
class _Collector:
    artifacts: LoopsIrArtifacts | None = None
    dumps: bool = False
    source: bool = False
    ir: bool = False
    statistics: bool = False


# Compile-time artifact scope for the production seam: ``loops_ir_backend``
# sets this around the inner ``compile_fx`` call (ContextVar, never a config
# object, so the FX-cache key is unaffected). ``compile_fx_loops_ir_output``
# reads it to populate artifacts/PTX outside the hot path. On an FX-cache
# hit this seam never runs, so no artifacts are materialized (honest: the
# caller observes the absence rather than stale data).
_current_collector: contextvars.ContextVar[_Collector | None] = contextvars.ContextVar(
    "loops_ir_collector", default=None
)


# ---------------------------------------------------------------------------
# Eager target resolution (symbolic target -> real callable; no callbacks in IR).
# ---------------------------------------------------------------------------


def _resolve_eager_target(dotted: str) -> Any:
    """Resolve 'aten.add.Tensor' style names to the live OpOverload."""
    parts = dotted.split(".")
    if len(parts) < 2:
        raise LoopsIrUnsupportedError(f"unresolvable extern target '{dotted}'")
    namespace, packet = parts[0], parts[1]
    overload = ".".join(parts[2:]) if len(parts) > 2 else "default"
    try:
        ops_ns = getattr(torch.ops, namespace)
        overload_packet = getattr(ops_ns, packet)
        return getattr(overload_packet, overload)
    except Exception as error:
        raise LoopsIrUnsupportedError(
            f"cannot resolve eager target '{dotted}': {error}"
        ) from error


# ---------------------------------------------------------------------------
# Dtype maps.
# ---------------------------------------------------------------------------

_IR_DTYPE_TO_TORCH: dict[str, Any] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i8": torch.int8,
    "u8": torch.uint8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
    "c64": torch.complex64,
    "c128": torch.complex128,
}

_CALLARG_DTYPE_TO_TORCH: dict[str, Any] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

# Allowlisted frozen torch.layout / torch.memory_format literals. Freeze maps
# the validated native text payload to the actual enum object (never eval);
# render maps back to the ``torch.<name>`` attribute spelling.
_CALLARG_LAYOUT_TO_TORCH: dict[str, Any] = {
    "strided": torch.strided,
    "sparse_coo": torch.sparse_coo,
    "sparse_csr": torch.sparse_csr,
    "sparse_csc": torch.sparse_csc,
    "sparse_bsr": torch.sparse_bsr,
    "sparse_bsc": torch.sparse_bsc,
}

_CALLARG_MEMORY_FORMAT_TO_TORCH: dict[str, Any] = {
    "contiguous_format": torch.contiguous_format,
    "channels_last": torch.channels_last,
    "channels_last_3d": torch.channels_last_3d,
    "preserve_format": torch.preserve_format,
}


def _ir_dtype_to_torch(name: str) -> Any:
    try:
        return _IR_DTYPE_TO_TORCH[name]
    except KeyError:
        raise LoopsIrUnsupportedError(f"unsupported IR dtype '{name}'") from None


# ---------------------------------------------------------------------------
# Compile-time plan freeze (pure-Python snapshots; hot path touches no C++ IR).
# ---------------------------------------------------------------------------


class _TensorKey:
    """Marker for a dynamic tensor reference inside a frozen argument tree.

    Static literals (None/int/float/bool/str/dtype/list-vs-tuple) are stored
    verbatim; only tensor leaves become ``_TensorKey`` and are looked up in
    the per-call tensor environment. ``__slots__`` keeps it allocation-light.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"_TensorKey({self.name!r})"


class _KernelStep(NamedTuple):
    """Precomputed generated-kernel launch plan (C++ read once at compile)."""

    name: str
    param_bases: tuple[str, ...]
    grid: tuple[int, int, int]
    num_warps: int
    enable_fp_fusion: bool
    enable_reflect_ftz: bool
    skip_empty: bool
    jit_fn: Any


class _ExternInput(NamedTuple):
    """Precomputed extern input descriptor (input-only filter applied)."""

    key: str  # tensor_env key: unresolved IR name, preserves call order
    base: str  # resolved backing buffer (alias chains composed)
    rel: int  # total IR-relative offset (descriptor + composed view)
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    span: int  # backing buffer element count (for the 1-D span fast path)


class _KnownLayout(NamedTuple):
    """Compile-time frozen tensor layout for the borrow shortcut.

    ``sizes``/``strides`` are the exact static metadata from the authoritative
    spec (example-input boundary tensors or validated extern results).
    ``storage_offset`` is the compile-time base offset: example inputs are
    trusted offset-0 from the direct-API static-CUDA validation (same trust
    the existing guards already place), extern results are data_ptr-relative
    (rel 0). Only exact (sizes, strides) matches borrow the backing tensor;
    any nonzero IR-relative ``rel`` still materializes a fresh view.
    """

    sizes: tuple[int, ...]
    strides: tuple[int, ...]


class _ExternResult(NamedTuple):
    """Precomputed extern result metadata for the eager layout check."""

    name: str
    dtype: Any
    sizes: tuple[int, ...]
    strides: tuple[int, ...]


class _ExternStep(NamedTuple):
    """Precomputed eager extern plan (target resolved once at compile)."""

    name: str
    target_fn: Any
    target_dotted: str
    inputs: tuple[_ExternInput, ...]
    args: tuple[Any, ...]  # frozen tree: literals + _TensorKey leaves
    kwargs: tuple[tuple[str, Any], ...]  # ordered (name, frozen tree)
    results: tuple[_ExternResult, ...]
    # Actual FX return-container kind for this node: "tensor" (bare Tensor)
    # or "sequence" (list/tuple of Tensors, including a one-element
    # List[Tensor]). Tracked from GM ``node.meta['val']`` (schema fallback)
    # so a single List[Tensor] result unpacks element-wise while a bare
    # Tensor binds directly.
    result_container: str = "sequence"


class _OutputPlan(NamedTuple):
    """Precomputed final-output view plan."""

    base: str
    rel: int
    sizes: tuple[int, ...]
    strides: tuple[int, ...]


def _freeze_call_arg(arg: Any) -> Any:
    """Convert one C++ CallArg to a pure-Python literal tree (compile time).

    Preserves None/int64/float/bool/string/dtype payloads and list-vs-tuple
    structure exactly; tensor references become :class:`_TensorKey` markers.
    Every C++ property (``kind_name``/``tensor_name``/``int_value``/``items``
    ...) is read here, never in the hot path.
    """
    kind = arg.kind_name
    if kind == "none":
        return None
    if kind == "tensor":
        return _TensorKey(arg.tensor_name)
    if kind == "int":
        return int(arg.int_value)
    if kind == "float":
        return float(arg.float_value)
    if kind == "bool":
        return bool(arg.bool_value)
    if kind == "string":
        return str(arg.text)
    if kind == "dtype":
        try:
            return _CALLARG_DTYPE_TO_TORCH[arg.text]
        except KeyError:
            raise LoopsIrUnsupportedError(
                f"unsupported dtype arg '{arg.text}'"
            ) from None
    if kind == "layout":
        try:
            return _CALLARG_LAYOUT_TO_TORCH[arg.text]
        except KeyError:
            raise LoopsIrUnsupportedError(
                f"unsupported layout arg '{arg.text}'"
            ) from None
    if kind == "memory_format":
        try:
            return _CALLARG_MEMORY_FORMAT_TO_TORCH[arg.text]
        except KeyError:
            raise LoopsIrUnsupportedError(
                f"unsupported memory_format arg '{arg.text}'"
            ) from None
    if kind == "list":
        return [_freeze_call_arg(item) for item in arg.items]
    if kind == "tuple":
        return tuple(_freeze_call_arg(item) for item in arg.items)
    raise LoopsIrUnsupportedError(f"unsupported extern arg kind '{kind}'")


def _materialize_call_arg(tree: Any, tensors: dict[str, Any]) -> Any:
    """Resolve a frozen argument tree against per-call tensors (hot path).

    Pure Python: no C++ IR access. Static literals pass through untouched;
    ``_TensorKey`` leaves are dictionary lookups; list/tuple structure is
    rebuilt exactly (list stays list, tuple stays tuple).
    """
    if isinstance(tree, _TensorKey):
        try:
            return tensors[tree.name]
        except KeyError:
            raise LoopsIrUnsupportedError(
                f"extern tensor arg '{tree.name}' unavailable"
            ) from None
    if isinstance(tree, list):
        return [_materialize_call_arg(item, tensors) for item in tree]
    if isinstance(tree, tuple):
        return tuple(_materialize_call_arg(item, tensors) for item in tree)
    return tree


def _resolve_view_chains(raw: dict[str, tuple[str, int]]) -> dict[str, tuple[str, int]]:
    """Compose alias chains to (base buffer, total offset) at compile time.

    ``raw`` maps view name -> (source name, relative offset), snapshotted
    from C++ ``View`` objects once. Returns the resolved map; plain buffer
    names are absent (callers default them to ``(name, 0)`` via the buffer
    table). Cycles raise here instead of per invocation.
    """
    resolved: dict[str, tuple[str, int]] = {}

    def rec(name: str, seen: tuple[str, ...]) -> tuple[str, int]:
        if name in resolved:
            return resolved[name]
        entry = raw.get(name)
        if entry is None:
            return (name, 0)
        if name in seen:
            raise LoopsIrUnsupportedError(f"view alias cycle at '{name}'")
        src, off = entry
        base, total = rec(src, seen + (name,))
        out = (base, off + total)
        resolved[name] = out
        return out

    for view_name in raw:
        rec(view_name, ())
    return resolved


def _is_extern_node(node: Any) -> bool:
    """KernelProgram node discrimination: ExternCall vs KernelFunction."""
    return type(node).__name__ == "ExternCall"


def _extern_result_container(gm: Any, node_name: str, num_results: int) -> str:
    """Return 'tensor' vs 'sequence' for an extern node's actual FX result.

    Reads ``node.meta['val']`` for the FX node named ``node_name`` (the C++
    ``ExternCall`` carries the FX node name); falls back to the op schema
    returns, then to the result count. A bare ``torch.Tensor`` val is a
    Tensor result; anything else (list/tuple, including a one-element
    ``List[Tensor]``) is a sequence result unpacked element-wise.
    """
    val: Any = None
    target: Any = None
    try:
        for node in gm.graph.nodes:
            if node.name == node_name:
                val = node.meta.get("val", None)
                if node.op == "call_function":
                    target = node.target
                break
    except Exception:
        val = None
    if isinstance(val, torch.Tensor):
        return "tensor"
    if isinstance(val, (list, tuple)):
        return "sequence"
    schema = getattr(target, "_schema", None)
    if schema is not None:
        try:
            returns = list(schema.returns)
            if len(returns) == 1:
                rtype = str(returns[0].type)
                if "Tensor[]" in rtype or "List[Tensor]" in rtype or "[]" in rtype:
                    return "sequence"
                return "tensor"
            return "sequence"
        except Exception:
            pass
    return "tensor" if num_results == 1 else "sequence"


class _FrozenPlan(NamedTuple):
    """Shared compile-time plan: pure-Python snapshots, no C++ IR access."""

    device_index: int
    num_inputs: int
    input_bases: tuple[str, ...]
    gen_spans: tuple[tuple[str, int], ...]
    # Conservative full-store coverage per gen span, in gen_spans order.
    span_full_coverage: tuple[bool, ...]
    steps: list[tuple[str, Any]]
    out_plans: tuple[_OutputPlan, ...]
    out_spec: Any
    # Normal-path Triton integration (per tiled function, launch order).
    kernel_sources: dict[str, str]
    kernel_names: list[str]
    kernel_params: dict[str, list[str]]
    kernel_grids: dict[str, tuple[int, int, int]]
    # Frozen authoritative layouts for the extern-arg borrow shortcut:
    # input base buffer -> (sizes, strides) from the static example-input
    # boundary tensors, and extern result buffer -> (sizes, strides) from
    # the validated result metadata. The emitter borrows the backing tensor
    # only on exact match with rel == 0 (never inferred from first use).
    # Represented as flat tuples (name, sizes, strides) for picklability.
    input_layouts: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
    result_layouts: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
    # Coverage counts + verified IR text (artifacts/diagnostics only).
    num_generated_kernels: int
    num_extern_calls: int
    kernels_before_fusion: int
    program_dump: str
    optimized_program_dump: str
    kernel_dump: str
    tiled_dump: str
    # Frozen effective policy (compile-time snapshot of the configured
    # numerics mode plus the threaded TileOptions flags). Default keeps old
    # _FrozenPlan(...) fixtures constructing without these fields working.
    numerics: str = "strict_eager"
    enable_fp_fusion: bool = False
    disable_ftz: bool = True


def _check_static_cuda_inputs(example_inputs: list[Any]) -> int:
    """Validate single-CUDA-device static inputs; return the device index."""
    devices: set[str] = set()
    device_index: int | None = None
    for value in example_inputs:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cuda":
                raise LoopsIrUnsupportedError("loops_ir_backend requires CUDA tensors")
            devices.add(str(value.device))
            if device_index is None and value.device.index is not None:
                device_index = value.device.index
            for dim in (*value.shape, *value.stride()):
                if type(dim) is not int:
                    raise LoopsIrUnsupportedError(
                        "loops_ir_backend requires static shapes (no symbolic ints)"
                    )
    if len(devices) > 1:
        raise LoopsIrUnsupportedError("single CUDA device required")
    if device_index is None:
        device_index = torch.cuda.current_device()
    return device_index


def _reject_non_cuda_intermediates(gm: Any, device_index: int) -> None:
    """Reject CPU/off-device tensors anywhere in the executable FX graph.

    The supported lane is static single-CUDA-device only: generated Triton
    kernels take device pointers, so any intermediate or extern-result
    tensor that is not on the compiled CUDA device would hand an illegal
    pointer to a kernel. Placeholders were already validated; this covers
    every ``node.meta['val']`` in the graph (flattened with the existing
    pytree APIs -- scalar/int shape entries are fine, only real tensors
    are checked). CPU opaque outputs are NOT implicitly supported: this
    lane has no host-side staging, so they are rejected here by default.
    """
    from torch.utils._pytree import tree_leaves

    expected = torch.device("cuda", device_index)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        val = node.meta.get("val", None)
        if val is None:
            continue
        for leaf in tree_leaves(val):
            if isinstance(leaf, torch.Tensor):
                if leaf.device != expected:
                    raise LoopsIrUnsupportedError(
                        f"loops_ir path is static-CUDA-only: FX node "
                        f"'{node.name}' carries a '{leaf.device}' tensor "
                        f"(expected '{expected}')"
                    )
            elif isinstance(leaf, torch.SymInt | torch.SymFloat | torch.SymBool):
                raise LoopsIrUnsupportedError(
                    f"loops_ir path requires static shapes: FX node "
                    f"'{node.name}' carries a symbolic value"
                )


def _build_frozen_plan(gm: Any, example_inputs: list[Any]) -> _FrozenPlan:
    """Shared IR compilation + plan freeze (compile time only).

    Runs verified lowering/optimization/tiling, emits per-function Triton
    sources with the normal fixed-config device path, and snapshots every
    C++ IR property the runtime needs into pure-Python tuples. Neither the
    returned plan nor its consumers touch C++ IR at runtime.
    """
    from torch._inductor.codegen.loops_ir_triton import emit_triton

    cpp = load_cpp_ir()
    _reject_backward_graph(gm)

    device_index = _check_static_cuda_inputs(example_inputs)
    _reject_non_cuda_intermediates(gm, device_index)

    from torch._inductor.loops_ir import lower_fx_to_cpp as _lower

    program = _lower(gm, example_inputs, executable=True)
    program.verify()
    optimized = cpp.optimize_program(program)
    optimized.verify()
    kernels = cpp.lower_to_kernels(optimized)
    kernels.verify()
    fused = cpp.fuse_kernels(kernels)
    fused.verify()

    # Thread the validated policy: the Program carries the configured mode
    # (set by the lowerer from the shared reader); the backend consumes the
    # stock-effective compilation flags without choosing fusion/layouts. The
    # effective flags are frozen here so old _FrozenPlan fixtures (which lack
    # the new field) keep working via the NamedTuple default below.
    from torch._inductor import config as inductor_config
    from torch._inductor.fx_passes.cpp_ir_strict import (
        INDUCTOR_NUMERICS,
        cpp_ir_numerics,
    )

    mode = cpp_ir_numerics()
    if mode == INDUCTOR_NUMERICS:
        effective_fp_fusion = not bool(inductor_config.emulate_precision_casts)
        effective_disable_ftz = bool(inductor_config.eager_numerics.disable_ftz)
    else:
        effective_fp_fusion = False
        effective_disable_ftz = True
    tiled = cpp.lower_to_tiled(
        fused,
        cpp.TileOptions(
            256,
            4,
            enable_fp_fusion=effective_fp_fusion,
            disable_ftz=effective_disable_ftz,
        ),
    )
    tiled.verify()

    program_dump = program.dump()
    optimized_dump = optimized.dump()
    kernel_dump = fused.dump()
    tiled_dump = tiled.dump()

    # Snapshot immutable C++ metadata outside the execution hot path.
    tiled_functions = list(tiled.functions)
    tiled_by_name = {fn.name: fn for fn in tiled_functions}
    device = torch.device("cuda", device_index)
    # Emit Triton per tiled function with the normal fixed-config device
    # path; keep sources for the file-backed wrapper module + inspection.
    sources: dict[str, str] = {}
    kernel_params: dict[str, list[str]] = {}
    kernel_grids: dict[str, tuple[int, int, int]] = {}
    kernel_names: list[str] = []
    for fn in tiled_functions:
        src = emit_triton(fn, device=device)
        sources[fn.name] = src
        kernel_names.append(fn.name)
        kernel_grids[fn.name] = (int(fn.grid_size), 1, 1)

    num_extern = sum(1 for n in fused.nodes if _is_extern_node(n))
    num_generated = len(tiled_functions)

    # Buffer metadata: map every buffer name -> (dtype, elements).
    # Read once here; the hot path uses only the frozen plans below.
    buffer_meta: dict[str, tuple[str, int]] = {}
    for buf in fused.inputs:
        buffer_meta[buf.name] = (buf.dtype, buf.elements)
    for node in fused.nodes:
        if _is_extern_node(node):
            results = node.results
        else:
            results = [buf for region in node.regions for buf in region.results]
        for buf in results:
            buffer_meta[buf.name] = (buf.dtype, buf.elements)

    # Views: snapshot (source, offset) once, then resolve full alias chains
    # to (base, total offset) now. Program-level (LoopsIR) views also appear
    # for broadcast/permute aliases; they were inlined by
    # optimize_program/view_inline before tiling unless still live.
    views_raw: dict[str, tuple[str, int]] = {}
    for v in fused.views:
        views_raw[v.name] = (v.source, v.offset)
    try:
        for node in optimized.nodes:
            if hasattr(node, "source") and hasattr(node, "name"):
                if node.name not in views_raw:
                    views_raw[node.name] = (node.source, node.offset)
    except Exception:
        pass
    resolved_views = _resolve_view_chains(views_raw)
    for known in buffer_meta:
        resolved_views.setdefault(known, (known, 0))

    def _lookup(name: str) -> tuple[str, int]:
        try:
            return resolved_views[name]
        except KeyError:
            raise LoopsIrUnsupportedError(f"unknown buffer '{name}'") from None

    # Input bindings: resolved base buffer per positional input.
    input_bases = tuple(_lookup(d.buffer)[0] for d in fused.input_descriptors)

    # Frozen authoritative input layouts from the static example-input
    # boundary tensors (compile-phase static-CUDA validation, same trust as
    # the existing guards). Keyed by resolved BASE buffer so alias chains
    # that resolve to the same backing share one spec; a base reached by
    # several positional inputs keeps the first spec (static shapes make
    # them consistent; the emitter only borrows on exact match anyway).
    # NOTE: example_inputs align positionally with fused.input_descriptors
    # (the lowerer consumes exactly one example per placeholder and errors
    # otherwise), so descriptor order == example order.
    input_layouts: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for descriptor, example in zip(fused.input_descriptors, example_inputs):
        base = _lookup(descriptor.buffer)[0]
        if base in input_layouts:
            continue
        if isinstance(example, torch.Tensor):
            input_layouts[base] = (
                tuple(int(s) for s in example.shape),
                tuple(int(s) for s in example.stride()),
            )

    # Generated intermediate spans in DAG order (fresh allocation per call;
    # opaque extern results are never preallocated). Names colliding with
    # input bases are already bound and skipped at runtime, same as before.
    # The C++ store proof covers multidimensional dense pointwise writes.
    # Holey, overlapping, reduction, zero-size, or otherwise unproven spans
    # keep the safe torch.empty fallback in the emitted wrapper.
    gen_spans: tuple[tuple[str, int], ...] = tuple(
        (buf.name, buf.elements)
        for node in fused.nodes
        if not _is_extern_node(node)
        for region in node.regions
        for buf in region.results
    )
    input_base_set = set(input_bases)
    gen_spans = tuple(
        (name, elements) for name, elements in gen_spans if name not in input_base_set
    )
    gen_span_names = {name for name, _ in gen_spans}
    span_full_coverage: dict[str, bool] = {}
    for node in fused.nodes:
        if _is_extern_node(node):
            continue
        for region in node.regions:
            for buf in region.results:
                if buf.name not in gen_span_names:
                    continue
                span_full_coverage[buf.name] = (
                    buf.dtype == "f32"
                    and buf.elements > 0
                    and cpp.prove_dense_injective_store(region, buf.name)
                )
    for name, _ in gen_spans:
        span_full_coverage.setdefault(name, False)

    # Ordered DAG steps: ("kernel", _KernelStep) or ("extern", _ExternStep).
    # A single list preserves KernelDAG execution order.
    steps: list[tuple[str, Any]] = []
    result_layouts: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for node in fused.nodes:
        if _is_extern_node(node):
            target_fn = _resolve_eager_target(node.target)
            input_names = frozenset(node.inputs)
            extern_inputs: list[_ExternInput] = []
            for d in node.descriptors:
                if d.buffer not in input_names:
                    continue
                base, composed = _lookup(d.buffer)
                try:
                    span = buffer_meta[base][1]
                except KeyError:
                    raise LoopsIrUnsupportedError(
                        f"unknown backing buffer '{base}'"
                    ) from None
                extern_inputs.append(
                    _ExternInput(
                        d.buffer,
                        base,
                        d.offset + composed,
                        tuple(d.sizes),
                        tuple(d.strides),
                        span,
                    )
                )
            frozen_args = tuple(_freeze_call_arg(a) for a in node.arguments)
            frozen_kwargs = tuple(
                (kw.name, _freeze_call_arg(kw.value)) for kw in node.keyword_args
            )
            desc_by_buffer: dict[str, Any] = {}
            for d in node.descriptors:
                desc_by_buffer.setdefault(d.buffer, d)
            extern_results: list[_ExternResult] = []
            for buf in node.results:
                match = desc_by_buffer.get(buf.name)
                if match is None:
                    raise LoopsIrUnsupportedError(
                        f"extern result '{buf.name}' lacks a descriptor"
                    )
                extern_results.append(
                    _ExternResult(
                        buf.name,
                        _ir_dtype_to_torch(buf.dtype),
                        tuple(match.sizes),
                        tuple(match.strides),
                    )
                )
                # Validated result layout: the emitted wrapper checks
                # dtype/shape/stride per result before binding, so later
                # extern args may borrow this tensor on exact match.
                result_layouts[buf.name] = (tuple(match.sizes), tuple(match.strides))
            steps.append(
                (
                    "extern",
                    _ExternStep(
                        node.name,
                        target_fn,
                        node.target,
                        tuple(extern_inputs),
                        frozen_args,
                        frozen_kwargs,
                        tuple(extern_results),
                        _extern_result_container(gm, node.name, len(extern_results)),
                    ),
                )
            )
        else:
            try:
                tiled_fn = tiled_by_name[node.name]
            except KeyError:
                raise LoopsIrUnsupportedError(
                    f"no tiled function for kernel '{node.name}'"
                ) from None
            grid_size = tiled_fn.grid_size
            kernel_params[node.name] = [_lookup(p.name)[0] for p in tiled_fn.parameters]
            steps.append(
                (
                    "kernel",
                    _KernelStep(
                        node.name,
                        tuple(kernel_params[node.name]),
                        (grid_size, 1, 1),
                        tiled_fn.num_warps,
                        tiled_fn.enable_fp_fusion,
                        not tiled_fn.disable_ftz,
                        grid_size == 0,
                        None,
                    ),
                )
            )

    # Final-output view plans + output pytree spec, both captured outside run.
    out_plans = tuple(
        _OutputPlan(
            _lookup(descriptor.buffer)[0],
            descriptor.offset + _lookup(descriptor.buffer)[1],
            tuple(descriptor.sizes),
            tuple(descriptor.strides),
        )
        for descriptor in fused.outputs
    )
    out_spec = getattr(gm, "_loops_ir_out_spec", None)

    return _FrozenPlan(
        device_index=device_index,
        num_inputs=len(input_bases),
        input_bases=input_bases,
        gen_spans=gen_spans,
        span_full_coverage=tuple(span_full_coverage[name] for name, _ in gen_spans),
        steps=steps,
        out_plans=out_plans,
        out_spec=out_spec,
        kernel_sources=sources,
        kernel_names=kernel_names,
        kernel_params=kernel_params,
        kernel_grids=kernel_grids,
        input_layouts=tuple(
            (name, sizes, strides) for name, (sizes, strides) in input_layouts.items()
        ),
        result_layouts=tuple(
            (name, sizes, strides) for name, (sizes, strides) in result_layouts.items()
        ),
        num_generated_kernels=num_generated,
        num_extern_calls=num_extern,
        kernels_before_fusion=sum(not _is_extern_node(n) for n in kernels.nodes),
        program_dump=program_dump,
        optimized_program_dump=optimized_dump,
        kernel_dump=kernel_dump,
        tiled_dump=tiled_dump,
        numerics=mode,
        enable_fp_fusion=effective_fp_fusion,
        disable_ftz=effective_disable_ftz,
    )


def _populate_artifacts(
    plan: _FrozenPlan, collector: _Collector | None, ptx: dict[str, str] | None = None
) -> None:
    """Record compile-time artifacts outside math (never alters numerics)."""
    artifacts = (
        collector.artifacts if collector is not None and collector.artifacts else None
    )
    if artifacts is None:
        return
    artifacts.program_dump = (
        plan.program_dump if collector.dumps or collector.ir else ""
    )
    artifacts.optimized_program_dump = (
        plan.optimized_program_dump if collector.dumps or collector.ir else ""
    )
    artifacts.kernels_before_fusion = plan.kernels_before_fusion
    artifacts.kernel_dump = plan.kernel_dump if collector.dumps or collector.ir else ""
    artifacts.tiled_dump = plan.tiled_dump if collector.dumps or collector.ir else ""
    artifacts.triton_sources = (
        dict(plan.kernel_sources) if collector.source or collector.dumps else {}
    )
    artifacts.num_generated_kernels = plan.num_generated_kernels
    artifacts.num_extern_calls = plan.num_extern_calls
    artifacts.function_names = list(plan.kernel_names)
    if ptx is not None:
        artifacts.triton_ptx.update(ptx)


# ---------------------------------------------------------------------------
# Direct FX compilation (useful for unit tests; AOT path below for torch.compile).
# ---------------------------------------------------------------------------


def _reject_backward_graph(gm: Any) -> None:
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "tangents" in node.name:
            raise LoopsIrUnsupportedError("backward graphs are rejected")
        if node.op == "output":
            from torch.utils._pytree import tree_leaves

            for value in tree_leaves(node.args[0]):
                if not hasattr(value, "op"):
                    raise LoopsIrUnsupportedError("non-tensor graph output rejected")


def _emit_and_load_wrapper(
    plan: _FrozenPlan,
    *,
    collector: _Collector | None = None,
    inference: bool = True,
) -> tuple[Any, dict[str, str] | None]:
    """Emit the straight-line wrapper module and load it file-backed.

    Returns ``(module, ptx_or_None)``. ``async_compile.wait`` runs at import
    inside :func:`load_loops_ir_wrapper`, so kernels are resolved CachingA
    utotuners on return. PTX is captured (compile time, before execution)
    from the resolved ``launchers[0].bin.asm`` only when ``statistics`` collection
    is requested; otherwise ``None`` (never a per-call collector).
    """
    from torch._inductor.codegen.loops_ir_wrapper import (
        capture_triton_ptx,
        emit_loops_ir_wrapper,
        load_loops_ir_wrapper,
    )

    source = emit_loops_ir_wrapper(
        kernel_sources=plan.kernel_sources,
        kernel_order=plan.kernel_names,
        kernel_params=plan.kernel_params,
        kernel_grids=plan.kernel_grids,
        steps=plan.steps,
        input_bases=plan.input_bases,
        gen_spans=plan.gen_spans,
        span_full_coverage=plan.span_full_coverage,
        input_layouts=plan.input_layouts,
        result_layouts=plan.result_layouts,
        out_plans=plan.out_plans,
        num_inputs=plan.num_inputs,
        device_index=plan.device_index,
        inference=inference,
    )
    mod, _key, _path, _linemap = load_loops_ir_wrapper(source)
    if not hasattr(mod, "call"):
        raise LoopsIrUnsupportedError("emitted wrapper module lacks 'call'")
    ptx: dict[str, str] | None = None
    if collector is not None and collector.statistics:
        ptx = capture_triton_ptx(mod, plan.kernel_names)
    _populate_artifacts(plan, collector, ptx)
    return mod, ptx


def compile_fx_loops_ir(
    gm: Any,
    example_inputs: list[Any],
    *,
    collector: _Collector | None = None,
) -> Any:
    """Compile an FX graph to a boxed-output callable (inference).

    Direct-test helper (not the production seam): builds the shared frozen
    plan, emits/loads the file-backed straight-line wrapper module, and
    adapts its flat-tuple production convention to the legacy return
    convention (single tensor unwrapped; captured pytree spec unflattened)
    so existing direct tests keep passing. Raises
    ``LoopsIrUnsupportedError`` for anything outside the strict subset.
    IR/statistics described by ``collector`` flags are recorded outside math
    (they never alter numerics).
    """
    from torch.utils._pytree import tree_unflatten

    plan = _build_frozen_plan(gm, list(example_inputs))
    mod, _ptx = _emit_and_load_wrapper(plan, collector=collector)
    out_spec = plan.out_spec

    def run(boxed_args: list[Any]) -> Any:
        # Test-only adapter outside production: the module's own call()
        # is the straight-line hot path (no C++ IR access); only the
        # legacy return convention is applied here.
        flat = mod.call(boxed_args)
        results_out = list(flat)
        if out_spec is not None:
            return tree_unflatten(results_out, out_spec)
        if len(results_out) == 1:
            return results_out[0]
        return results_out

    run._boxed_call = True  # type: ignore[attr-defined]
    artifacts = (
        collector.artifacts if collector is not None and collector.artifacts else None
    )
    # Attach artifacts for orchestrator inspection (clear() to reset).
    run._loops_ir_artifacts = artifacts  # type: ignore[attr-defined]
    run._loops_ir_sources = plan.kernel_sources  # type: ignore[attr-defined]
    run._loops_ir_tiled_dump = plan.tiled_dump  # type: ignore[attr-defined]
    run._loops_ir_kernel_dump = plan.kernel_dump  # type: ignore[attr-defined]
    run._loops_ir_module = mod  # type: ignore[attr-defined]
    return run


# ---------------------------------------------------------------------------
# Production seam: post-grad FX -> CompiledFxGraph (called by frontend worker).
# ---------------------------------------------------------------------------


def _collect_gm_constants(gm: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Collect residual tensor ``get_attr`` constants from the FX graph.

    Accepted graphs must have none (the strict lowerer rejects ``get_attr``
    explicitly, and lifted params arrive positionally); a non-empty result
    is an explicit error, never silent dead weight in the cache entry.
    """
    constants: dict[str, Any] = {}
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        try:
            value = getattr(gm, str(node.target))
        except AttributeError:
            continue
        if isinstance(value, torch.Tensor):
            constants[str(node.target)] = value
    if constants:
        raise LoopsIrUnsupportedError(
            f"loops_ir path cannot embed residual constants {sorted(constants)}"
        )
    return {}, {}


def _read_original_output_strides(
    gm: Any, plan: _FrozenPlan
) -> list[tuple[str, ...] | None]:
    """Output strides for ``CompiledFxGraph`` from real FX/IR metadata.

    Prefers the frontend-recorded ``original_output_strides`` (written by
    ``_recursive_record_original_output_strides`` before this seam); falls
    back to the verified IR output descriptors. Static-only: any symbolic
    entry is an explicit error.
    """
    try:
        for node in gm.graph.nodes:
            if node.op == "output":
                recorded = node.meta.get("original_output_strides", None)
                if recorded is not None:
                    out: list[tuple[str, ...] | None] = []
                    for entry in recorded:
                        if entry is None:
                            out.append(None)
                        else:
                            out.append(tuple(str(int(s)) for s in entry))
                    return out
                break
    except LoopsIrUnsupportedError:
        raise
    except Exception:
        pass
    return [
        tuple(str(int(s)) for s in out.strides) if out.strides is not None else None
        for out in plan.out_plans
    ]


def compile_fx_loops_ir_output(
    gm: Any,
    example_inputs: Any,
    *,
    graph_kwargs: Any,
    inputs_to_check: Any,
    runnable_graph_str: str,
    inductor_post_grad_graph_str: str,
) -> Any:
    """Production FX->``CompiledFxGraph`` seam for the C++ LoopsIR path.

    Called by the frontend worker from ``codegen_and_compile`` (after
    post-grad passes) when ``config.experimental_cpp_ir`` is set. Builds
    the frozen compile-time plan, emits/loads the straight-line boxed
    ``call(args)`` module, and returns the existing ``CompiledFxGraph``
    (same constructor/``__call__``/``post_compile``/serialization/input
    alignment/output-lifetime machinery as the normal path).

    Inference-only: ``is_inference`` graphs are accepted regardless of
    lifted weights' ``requires_grad`` (provenance comes from the graph
    flag, not the parameter flag); every other mode fails explicitly --
    joint/backward graphs, non-inference forwards (including training FW),
    AOTI/C++ wrapper, FX wrapper, freezing, dynamic shapes, and
    input/storage mutations.
    """
    from torch._inductor import config as inductor_config
    from torch._inductor import metrics
    from torch._inductor.codegen.loops_ir_wrapper import (
        emit_loops_ir_wrapper,
        load_loops_ir_wrapper,
    )
    from torch._inductor.cudagraph_utils import get_input_storage_mutation_info
    from torch._inductor.freezing_utils import has_frozen_params
    from torch._inductor.output_code import CompiledFxGraph
    from torch._inductor.utils import BoxedBool
    from torch._inductor.virtualized import V
    from torch.utils._ordered_set import OrderedSet
    from torch._dynamo.utils import counters

    cudagraphs = graph_kwargs.get("cudagraphs", None)
    static_input_idxs = tuple(graph_kwargs.get("static_input_idxs", ()))
    is_backward = bool(graph_kwargs.get("is_backward", False))
    is_inference = bool(graph_kwargs.get("is_inference", False))
    cpp_wrapper = bool(graph_kwargs.get("cpp_wrapper", False))
    aot_mode = bool(graph_kwargs.get("aot_mode", False) or V.aot_compilation)
    fx_wrapper = bool(graph_kwargs.get("fx_wrapper", False))
    if graph_kwargs.get("extern_node_serializer", None) is not None:
        raise LoopsIrUnsupportedError(
            "loops_ir path does not support extern_node_serializer"
        )

    # Explicit mode rejections (never silent fallback). This lane is
    # inference-only: only is_inference graphs are accepted. Lifted weights
    # with requires_grad=True are fine under is_inference (provenance comes
    # from the graph flag, not the parameter flag); any non-inference
    # forward (including training FW) is rejected rather than silently
    # enabled, since backward/lower-precision/training codegen is deferred.
    if is_backward:
        raise LoopsIrUnsupportedError("loops_ir path rejects backward/joint graphs")
    if not is_inference:
        raise LoopsIrUnsupportedError(
            "loops_ir path is inference-only (is_inference=False graphs rejected)"
        )
    if aot_mode or cpp_wrapper:
        raise LoopsIrUnsupportedError("loops_ir path rejects AOTI/C++ wrapper mode")
    if fx_wrapper or inductor_config.fx_wrapper:
        raise LoopsIrUnsupportedError("loops_ir path rejects FX wrapper mode")
    if inductor_config.cpp_wrapper:
        raise LoopsIrUnsupportedError("loops_ir path rejects C++ wrapper mode")
    if has_frozen_params(gm):
        raise LoopsIrUnsupportedError("loops_ir path rejects freezing")
    # NOTE on graph_partition (default on): stock Inductor only consults
    # partition maps for cudagraph flows (scheduler partition_maps,
    # cudagraph_partition_post_compile). This seam always produces a single
    # flat partition (partition_maps=None, has_uncaptured_partition=False)
    # and disables cudagraphs with an accurate reason below, so the default
    # flag is harmless -- post_compile short-circuits on the disable reason
    # before any partition path. No blanket reject: it would make the
    # default configuration unusable.
    storage_mutation_info = get_input_storage_mutation_info(gm)
    if len(storage_mutation_info.input_idxs) > 0:
        raise LoopsIrUnsupportedError(
            "loops_ir path rejects input storage mutation "
            f"(idxs {sorted(storage_mutation_info.input_idxs)})"
        )

    metrics_helper = metrics.CachedMetricsHelper()
    inductor_counters = counters["inductor"].copy()

    inputs = list(example_inputs)
    plan = _build_frozen_plan(gm, inputs)
    # Accurate kernel accounting: exactly the embedded generated kernels.
    metrics.generated_kernel_count += plan.num_generated_kernels

    source = emit_loops_ir_wrapper(
        kernel_sources=plan.kernel_sources,
        kernel_order=plan.kernel_names,
        kernel_params=plan.kernel_params,
        kernel_grids=plan.kernel_grids,
        steps=plan.steps,
        input_bases=plan.input_bases,
        gen_spans=plan.gen_spans,
        span_full_coverage=plan.span_full_coverage,
        input_layouts=plan.input_layouts,
        result_layouts=plan.result_layouts,
        out_plans=plan.out_plans,
        num_inputs=plan.num_inputs,
        device_index=plan.device_index,
        # inference_mode() provenance follows the graph flag: inference
        # graphs run eager externs under inference mode (no autograd
        # history, matching eager-inference bitwise behavior). Forward
        # training graphs are rejected above (inference-only lane), so this
        # is always True in production; the False path exists only for the
        # diagnostic direct helper default.
        inference=is_inference,
    )
    mod, cache_key, cache_path, cache_linemap = load_loops_ir_wrapper(source)
    try:
        current_callable = mod.call
    except AttributeError as error:
        raise LoopsIrUnsupportedError("emitted wrapper module lacks 'call'") from error

    # Compile-time artifact scope (ContextVar; never affects the cache key).
    # On an FX-cache hit this seam never runs, so nothing is materialized.
    collector = _current_collector.get()
    _populate_artifacts(plan, collector)

    from torch._inductor.codegen.loops_ir_wrapper import LoopsIrLoweringMetadata

    device_index = plan.device_index
    metadata = LoopsIrLoweringMetadata(
        cache_key=cache_key,
        cache_path=cache_path,
        cache_linemap=cache_linemap,
        device_types=OrderedSet(["cuda"]),
        device_idxs=OrderedSet([device_index]),
        mutated_inputs=OrderedSet([]),
        mutated_input_idxs=OrderedSet([]),
        constants={},
        allocated_constant_name={},
        torchbind_constants={},
        opaque_value_type_classes={},
        kernel_free_cudagraph=False,
        partition_maps=None,
        has_uncaptured_partition=False,
    )
    _collect_gm_constants(gm)  # explicit reject on residual constants
    output_strides = _read_original_output_strides(gm, plan)

    if cudagraphs is None:
        cudagraphs = BoxedBool(inductor_config.triton.cudagraphs)
    # Conservative cudagraph disable: normal Inductor CUDA graphs CAN contain
    # allocations and ATen externs, so this reason must NOT claim intrinsic
    # uncapturability. This experimental lane simply has not validated CUDA
    # graph capture yet, so it opts out through the shared runtime instead
    # of recording an unvalidated graph. No cudagraph features or broader
    # partition support are added here.
    disabled_cudagraphs_reason: str | None = (
        "experimental_cpp_ir CUDA graph capture is not yet supported/validated"
    )

    graph = CompiledFxGraph(
        current_callable,
        metadata,
        gm,
        output_strides,
        disabled_cudagraphs_reason,
        metrics_helper.get_deltas(),
        counters["inductor"] - inductor_counters,
        cudagraphs,
        inputs,
        static_input_idxs,
        None,
        dict(graph_kwargs),
        tuple(inputs_to_check),
        runnable_graph_str,
        inductor_post_grad_graph_str,
    )
    if collector is not None and collector.artifacts is not None:
        try:
            graph._loops_ir_artifacts = collector.artifacts  # type: ignore[attr-defined]
        except Exception:
            pass
    return graph


# ---------------------------------------------------------------------------
# torch.compile backend (AOT capture + strict lowering + boxed convention).
# ---------------------------------------------------------------------------


def loops_ir_backend(
    gm: Any,
    example_inputs: list[Any],
    *,
    dumps: bool = False,
    source: bool = False,
    ir: bool = False,
    statistics: bool = False,
    collector: LoopsIrArtifacts | None = None,
) -> Any:
    """Opt-in ``torch.compile`` backend callable (inference only).

    Thin ordinary-``compile_fx`` forwarder selecting
    ``config.experimental_cpp_ir=True``: no standalone AOT adapter, no
    Inductor decompositions bypass -- the normal Dynamo/post-grad/FX-cache
    pipeline runs, and the frontend worker routes the lowering seam to
    :func:`compile_fx_loops_ir_output`. Extra kwargs are capture flags
    (dumps / source / IR / statistics) scoped through a ContextVar (never a
    config object, so the FX-cache key is unaffected) and recorded outside
    math. The returned callable follows the boxed convention
    (``fn([args...])`` with ``_boxed_call=True``).
    """
    from torch._inductor.compile_fx import compile_fx

    artifacts = collector if collector is not None else LoopsIrArtifacts()
    collect = _Collector(
        artifacts=artifacts,
        dumps=dumps,
        source=source,
        ir=ir,
        statistics=statistics,
    )
    token = _current_collector.set(collect)
    try:
        compiled = compile_fx(
            gm,
            example_inputs,
            config_patches={"experimental_cpp_ir": True},
        )
    finally:
        _current_collector.reset(token)
    # Preserve artifact access through the compiled result. On an FX-cache
    # hit the production seam never runs, so artifacts stay honestly empty.
    try:
        compiled._loops_ir_artifacts = artifacts  # type: ignore[attr-defined]
    except Exception:
        pass
    return compiled


def compile_fx_entry(gm: Any, example_inputs: list[Any], **flags: Any) -> Any:
    """Direct-test entry: ``compile_fx_loops_ir`` with flag passthrough."""
    artifacts = flags.pop("collector", None)
    collect = _Collector(
        artifacts=artifacts,
        dumps=bool(flags.pop("dumps", False)),
        source=bool(flags.pop("source", False)),
        ir=bool(flags.pop("ir", False)),
        statistics=bool(flags.pop("statistics", False)),
    )
    if flags:
        raise LoopsIrUnsupportedError(f"unknown compile flags {sorted(flags)}")
    return compile_fx_loops_ir(gm, list(example_inputs), collector=collect)


# Alias useful in direct tests.
compile_fx_loops_ir_entry = compile_fx_entry
compile_fx_entry
