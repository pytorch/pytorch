"""File-backed straight-line wrapper emission for the C++ LoopsIR path.

``torch._inductor.codegen.loops_ir_wrapper`` turns a compile-time frozen
launch plan (built by ``torch._inductor.loops_ir_backend`` from verified
C++ IR -- descriptors, launch metadata, output spec snapshotted BEFORE the
hot path) into a normal Inductor-style Python module defining
``call(args)``.

The emitted module:

- embeds each generated Triton kernel source via normal
  ``async_compile.triton(name, source, device_str='cuda')`` definitions and
  resolves them with ``async_compile.wait(globals())`` at import time;
- launches kernels with the normal fixed-grid calling convention
  ``tuner.run(*ptrs, grid0, grid1, grid2, stream=get_raw_stream(idx))``;
- runs structured eager extern calls under ``torch.inference_mode()`` with
  cheap per-result dtype/shape/stride checks;
- allocates generated intermediate storage per call with the normal
  fast ``empty_strided_cuda`` helper (same module-level alias as the stock
  wrapper) for spans with compile-time proven full-store coverage, with a
  safe ``torch.empty`` fallback otherwise (never retained in globals), and
  returns a flat output tuple.

No C++ IR object, callback, property, or argument tree is read at runtime:
everything dynamic is a module-local tensor variable. Scalar float literals
are emitted bit-exact (f64 bits round-trip through ``struct``) so NaN
payloads and signed zeros survive source rendering.
"""

from __future__ import annotations

import dataclasses
import re
import struct
from typing import Any


__all__ = [
    "LoopsIrLoweringMetadata",
    "capture_triton_ptx",
    "emit_loops_ir_wrapper",
    "float_bits_literal",
    "load_loops_ir_wrapper",
    "render_frozen_arg",
]


# ---------------------------------------------------------------------------
# Bit-exact scalar rendering (compile time only).
# ---------------------------------------------------------------------------


def float_bits_literal(value: float) -> str:
    """Render a Python float as a bit-exact source expression.

    ``repr()`` of NaN/negative-NaN loses payload/sign bits and large/small
    values may round-trip through decimal text; packing the exact f64 bits
    preserves NaN payloads and signed zeros verbatim.
    """
    bits = struct.pack("<d", float(value)).hex()
    return f"struct.unpack('<d', bytes.fromhex('{bits}'))[0]"


_TORCH_DTYPE_ATTRS: dict[str, str] = {
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "float32": "torch.float32",
    "float64": "torch.float64",
    "int8": "torch.int8",
    "uint8": "torch.uint8",
    "int16": "torch.int16",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "bool": "torch.bool",
    "complex64": "torch.complex64",
    "complex128": "torch.complex128",
}

# Validated torch.layout / torch.memory_format tokens, mirroring the native
# CallArg whitelist (ir.h isValidLayoutName / isValidMemoryFormatName) but
# restricted to tokens that resolve to actual torch singletons. Tokens with no
# corresponding torch attribute (e.g. 'sparse_compressed') are absent and
# therefore rejected by the renderer; tokens the native layer rejects (e.g.
# 'mkldnn' without the underscore prefix, or any 'arbitrary_code') never
# arrive through the real pipeline and are rejected here too. Rendering is a
# table lookup emitting a fixed literal -- never repr()/eval() of a payload.
_VALIDATED_LAYOUT_TOKENS: tuple[str, ...] = (
    "strided",
    "sparse_coo",
    "sparse_csr",
    "sparse_csc",
    "sparse_bsr",
    "sparse_bsc",
    "_mkldnn",
)

_VALIDATED_MEMORY_FORMAT_TOKENS: tuple[str, ...] = (
    "contiguous_format",
    "preserve_format",
    "channels_last",
    "channels_last_3d",
)


def _sanitize_identifier(name: str) -> str:
    cleaned = re.sub(r"\W", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def render_frozen_arg(tree: Any, tensor_vars: dict[str, str]) -> str:
    """Render a frozen extern argument tree to a Python expression.

    ``tree`` is the pure-Python snapshot built by
    ``loops_ir_backend._freeze_call_arg``: ``None``/int/float/bool/str/dtype/
    layout/memory_format literals (list stays list, tuple stays tuple) with
    tensor leaves as ``_TensorKey`` markers resolved through ``tensor_vars``.
    No C++ IR is touched; unknown shapes raise instead of guessing.

    Layout/memory-format leaves are actual ``torch.layout`` /
    ``torch.memory_format`` singletons (frozen through the backend allowlist).
    They render by table lookup to a fixed ``torch.<name>`` literal -- never
    ``repr()``/``eval()`` of a payload. Tokens outside the validated tables
    (including any native-only names with no torch singleton) are rejected.
    """
    from torch._inductor.loops_ir_backend import _TensorKey

    if isinstance(tree, _TensorKey):
        try:
            return tensor_vars[tree.name]
        except KeyError:
            raise ValueError(
                f"extern tensor arg '{tree.name}' has no wrapper variable"
            ) from None
    if tree is None:
        return "None"
    if isinstance(tree, bool):
        return "True" if tree else "False"
    if isinstance(tree, int):
        return repr(tree)
    if isinstance(tree, float):
        return float_bits_literal(tree)
    if isinstance(tree, str):
        return repr(tree)
    # torch.dtype instances render as torch.float32 etc.
    import torch

    if isinstance(tree, torch.dtype):
        for text, attr in _TORCH_DTYPE_ATTRS.items():
            if getattr(torch, text, None) is tree or str(tree) == f"torch.{text}":
                return attr
        raise ValueError(f"unsupported dtype arg '{tree}'")
    # Frozen layout/memory-format singletons render as fixed torch.<name>
    # literals. The identity check (``is``) is the allowlist: only the actual
    # singleton for a validated token renders; spoofed lookalikes fall
    # through to the explicit rejection below.
    if isinstance(tree, torch.layout):
        for token in _VALIDATED_LAYOUT_TOKENS:
            singleton = getattr(torch, token, None)
            if singleton is not None and singleton is tree:
                return f"torch.{token}"
        raise ValueError(f"unsupported layout arg '{tree}'")
    if isinstance(tree, torch.memory_format):
        for token in _VALIDATED_MEMORY_FORMAT_TOKENS:
            singleton = getattr(torch, token, None)
            if singleton is not None and singleton is tree:
                return f"torch.{token}"
        raise ValueError(f"unsupported memory_format arg '{tree}'")
    if isinstance(tree, list):
        return "[" + ", ".join(render_frozen_arg(t, tensor_vars) for t in tree) + "]"
    if isinstance(tree, tuple):
        inner = ", ".join(render_frozen_arg(t, tensor_vars) for t in tree)
        if len(tree) == 1:
            inner += ","
        return "(" + inner + ")"
    raise ValueError(f"unsupported frozen arg of type {type(tree).__name__}")


def extern_target_path(dotted: str) -> str:
    """Render a resolved dotted target ('aten.add.Tensor') as source."""
    parts = dotted.split(".")
    if len(parts) < 2:
        raise ValueError(f"unresolvable extern target '{dotted}'")
    return "torch.ops." + ".".join(parts)


# ---------------------------------------------------------------------------
# Wrapper emission.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LoopsIrLoweringMetadata:
    """Concrete ``LoweringOutputMetadata`` for the C++ LoopsIR path.

    Built from the compiled wrapper module (cache key/path/linemap), the FX
    graph (mutation info, constants, output strides), and the verified IR
    descriptors (device sets). No ``GraphLowering`` is constructed.
    """

    cache_key: str
    cache_path: str
    cache_linemap: list[tuple[int, str]]
    device_types: Any
    device_idxs: Any
    mutated_inputs: Any
    mutated_input_idxs: Any
    constants: dict[str, Any]
    allocated_constant_name: dict[str, str]
    torchbind_constants: dict[str, Any]
    opaque_value_type_classes: dict[str, type]
    kernel_free_cudagraph: bool
    partition_maps: Any | None
    has_uncaptured_partition: bool


def emit_loops_ir_wrapper(
    *,
    kernel_sources: dict[str, str],
    kernel_order: list[str],
    kernel_params: dict[str, list[str]],
    kernel_grids: dict[str, tuple[int, int, int]],
    steps: list[tuple[str, Any]],
    input_bases: tuple[str, ...],
    gen_spans: tuple[tuple[str, int], ...],
    # Conservative full-store coverage per span, in gen_spans order: True
    # only when the producing generated kernel provably overwrites every
    # element (single dense pointwise range covering the whole span).
    # Absent (None) or False selects the safe torch.empty fallback.
    span_full_coverage: tuple[bool, ...] | None = None,
    # Frozen authoritative layouts for the extern-arg borrow shortcut, as
    # flat (buffer, sizes, strides) tuples. Only exact (sizes, strides)
    # matches with rel == 0 borrow the backing tensor; nonzero rel or any
    # mismatch materializes a fresh as_strided view. Input layouts come
    # from the static example-input boundary spec; result layouts from the
    # validated extern-result metadata. Nothing is inferred from first use.
    input_layouts: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (),
    result_layouts: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (),
    out_plans: tuple[Any, ...],
    num_inputs: int,
    device_index: int,
    flat_span_dtype: str = "torch.float32",
    inference: bool = True,
) -> str:
    """Emit the straight-line boxed ``call(args)`` module source.

    - ``kernel_sources``: tuner variable name -> full kernel file text
      (already including ``@triton_heuristics.fixed_config`` via
      ``emit_triton(fn, device=...)``). Names must be valid identifiers.
    - ``kernel_order``: launch order (subset of ``kernel_sources`` keys).
    - ``kernel_params``: tuner name -> backing-buffer names in launch order.
    - ``kernel_grids``: tuner name -> static ``(gx, gy, gz)`` grid.
    - ``steps``: ordered ``("kernel", _KernelStep)`` / ``("extern",
      _ExternStep)`` frozen plans from ``loops_ir_backend``. Only the
      ``extern`` payload fields (inputs/args/kwargs/results/container kind)
      are read; kernel launch metadata comes from the explicit dicts above
      so this module never interprets TiledIR at runtime.
    - ``input_bases``/``gen_spans``/``out_plans``: frozen buffer plans.
    - ``out_plans`` entries expose ``.base``/``.rel``/``.sizes``/``.strides``.
    - ``inference``: wrap eager extern calls in ``torch.inference_mode()``
      (production inference graphs). ``False`` emits plain calls so forward
      training graphs produce normal autograd-compatible tensors (the
      backward graph itself is still rejected explicitly). Inference mode
      never changes op math, only tensor provenance.
    """
    from torch._inductor.loops_ir_backend import _TensorKey  # noqa: F401

    lines: list[str] = []
    lines.append("import struct")
    lines.append("")
    lines.append("import torch")
    lines.append("from torch._inductor.async_compile import AsyncCompile")
    lines.append("from torch._inductor.loops_ir import LoopsIrUnsupportedError")
    # Normal inductor fast allocation helper (same module-level alias as the
    # stock Python wrapper): at::detail::empty_strided_cuda, which skips the
    # public aten.empty_cuda deterministic-fill path. Only used for spans
    # with proven full-store coverage; never toggles any global flag.
    lines.append("empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda")
    lines.append("")
    lines.append("async_compile = AsyncCompile()")
    lines.append("try:")
    lines.append("    get_raw_stream = torch._C._cuda_getCurrentRawStream")
    lines.append("except (ImportError, AttributeError):")
    lines.append("    get_raw_stream = None")
    lines.append("")

    for key in kernel_order:
        src = kernel_sources[key]
        var = _sanitize_identifier(key)
        if var != key:
            raise ValueError(f"kernel name {key!r} is not a valid identifier")
        # The kernel source is the exact bytes submitted to the async
        # compiler; dedent/strip parity keeps CompiledTritonKernels keyed
        # on the same string at wrapper time.
        body = "\n" + src.strip() + "\n"
        lines.append(
            f"{var} = async_compile.triton({key!r}, '''{body}''', device_str='cuda')"
        )
    lines.append("")
    lines.append("async_compile.wait(globals())")
    lines.append("del async_compile")
    lines.append("")

    # Resolve eager extern targets once at import, not per call.
    extern_targets: dict[str, str] = {}
    target_counter = 0
    for kind, step in steps:
        if kind != "extern":
            continue
        dotted = step.target_dotted
        if dotted not in extern_targets:
            extern_targets[dotted] = f"_extern_target_{target_counter}"
            target_counter += 1
    for dotted, var in extern_targets.items():
        lines.append(f"{var} = {extern_target_path(dotted)}")
    if extern_targets:
        lines.append("")

    # Buffer name -> module-local variable. Input bases bind to boxed args;
    # every other buffer gets its own variable.
    buf_vars: dict[str, str] = {}
    used_vars: set[str] = set()

    def _buf_var(name: str) -> str:
        if name in buf_vars:
            return buf_vars[name]
        base = "b_" + _sanitize_identifier(name)
        var = base
        counter = 0
        while var in used_vars:
            counter += 1
            var = f"{base}_{counter}"
        used_vars.add(var)
        buf_vars[name] = var
        return var

    for base in input_bases:
        _buf_var(base)
    for gen_name, _ in gen_spans:
        _buf_var(gen_name)
    for _kind, step in steps:
        if _kind != "extern":
            continue
        for res in step.results:
            _buf_var(res.name)

    span_elements = dict(gen_spans)

    def _is_flat_span_alias(
        base: str, sizes: tuple[int, ...], strides: tuple[int, ...], rel: int
    ) -> bool:
        """Compile-time proof that a view is exactly our flat span buffer."""
        if base not in span_elements:
            return False
        if rel != 0:
            return False
        elements = span_elements[base]
        return tuple(sizes) == (elements,) and tuple(strides) == (1,)

    # Borrow tables: backing buffer -> exact frozen (sizes, strides). Input
    # entries come from the static boundary spec, result entries from
    # validated extern-result metadata. A borrow is sound only when the
    # requested descriptor matches exactly with rel == 0: rel 0 is
    # data_ptr-relative (the backing tensor's own storage_offset, zero or
    # not, is preserved by borrowing), while any nonzero rel needs a fresh
    # view. No aliasing semantics are merged and no input is mutated.
    known_layouts: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for layout_name, layout_sizes, layout_strides in input_layouts:
        known_layouts.setdefault(
            layout_name, (tuple(layout_sizes), tuple(layout_strides))
        )
    for layout_name, layout_sizes, layout_strides in result_layouts:
        known_layouts.setdefault(
            layout_name, (tuple(layout_sizes), tuple(layout_strides))
        )

    def _borrowed_var(
        base: str, sizes: tuple[int, ...], strides: tuple[int, ...], rel: int
    ) -> str | None:
        """Backing variable if the descriptor borrows it exactly, else None."""
        if rel != 0:
            return None
        known = known_layouts.get(base)
        if known is None:
            return None
        if tuple(sizes) != known[0] or tuple(strides) != known[1]:
            return None
        return buf_vars[base]

    lines.append("")
    lines.append("def call(args):")
    lines.append("    if not isinstance(args, list):")
    lines.append(
        "        raise LoopsIrUnsupportedError('boxed calling convention requires a list')"
    )
    lines.append(f"    if len(args) != {num_inputs}:")
    lines.append("        raise LoopsIrUnsupportedError('boxed args count mismatch')")
    # Bind inputs to their data_ptr-relative base buffers.
    for pos, base in enumerate(input_bases):
        lines.append(f"    {buf_vars[base]} = args[{pos}]")
    # Device comes from the first boxed tensor; static CUDA was verified.
    # The baked compile-time index is enforced below so spans, streams, and
    # kernels never land on the ambient default device by accident.
    lines.append("    _first = args[0] if args else None")
    lines.append(
        "    _device = _first.device if isinstance(_first, torch.Tensor)"
        f" else torch.device('cuda', {device_index})"
    )
    lines.append("    for _v in args:")
    lines.append("        if isinstance(_v, torch.Tensor) and _v.device != _device:")
    lines.append(
        "            raise LoopsIrUnsupportedError('single CUDA device required')"
    )
    lines.append(f"    if _device.type != 'cuda' or _device.index != {device_index}:")
    lines.append("        raise LoopsIrUnsupportedError('single CUDA device required')")
    # Provenance branch selector, computed once per call: when the caller
    # is already in inference mode, extern calls run directly (no
    # per-extern inference_mode() enter/exit); otherwise each extern keeps
    # its own context for inference graphs. Allocations and kernels stay
    # outside any inference context so generated span tensors keep normal
    # provenance and version-counter behavior.
    lines.append("    _already_inference = torch.is_inference_mode_enabled()")
    # Everything below allocates/launches on the baked device: mirror the
    # stock wrapper's device-guard-enter pattern so empty_strided_cuda (which
    # allocates on the current device) cannot land on the default device.
    lines.append(f"    with torch.cuda._DeviceGuard({device_index}):")
    body: list[str] = []

    def _emit(line: str) -> None:
        body.append(f"        {line}")

    # Fresh per-call intermediate spans (opaque extern results excluded).
    # Proven fully-overwritten dense spans use the normal fast helper;
    # anything unproven (or zero-size) keeps the safe torch.empty fallback.
    if span_full_coverage is None:
        span_full_coverage = tuple(False for _ in gen_spans)
    for (gen_name, elements), proven in zip(gen_spans, span_full_coverage, strict=True):
        if elements == 0 or not proven:
            if elements == 0:
                _emit(
                    f"{buf_vars[gen_name]} = torch.empty(0, "
                    f"dtype={flat_span_dtype}, device=_device)"
                )
            else:
                _emit(
                    f"{buf_vars[gen_name]} = torch.empty({elements}, "
                    f"dtype={flat_span_dtype}, device=_device)"
                )
        else:
            _emit(
                f"{buf_vars[gen_name]} = empty_strided_cuda(({elements},), (1,), "
                f"{flat_span_dtype})"
            )

    needs_stream = any(
        kind == "kernel" and kernel_grids.get(step.name, (0, 0, 0)) != (0, 0, 0)
        for kind, step in steps
    )
    if needs_stream:
        _emit("if get_raw_stream is None:")
        _emit("    raise LoopsIrUnsupportedError('CUDA raw stream unavailable')")
        _emit(f"_stream0 = get_raw_stream({device_index})")

    view_counter = 0

    def _fresh_view() -> str:
        nonlocal view_counter
        name = f"_v{view_counter}"
        view_counter += 1
        return name

    for kind, step in steps:
        if kind == "kernel":
            grid = kernel_grids.get(step.name)
            if grid is None:
                raise ValueError(f"no static grid for kernel '{step.name}'")
            gx, gy, gz = grid
            if gx == 0 and gy == 0 and gz == 0:
                continue
            var = _sanitize_identifier(step.name)
            arg_vars = [buf_vars[base] for base in kernel_params[step.name]]
            call_args = ", ".join(arg_vars + [str(gx), str(gy), str(gz)])
            _emit(f"{var}.run({call_args}, stream=_stream0)")
        else:
            if kind != "extern":
                raise ValueError(f"unknown plan step kind {kind!r}")
            tensor_vars: dict[str, str] = {}
            for inp in step.inputs:
                backing = buf_vars[inp.base]
                if _is_flat_span_alias(inp.base, inp.sizes, inp.strides, inp.rel):
                    tensor_vars[inp.key] = backing
                    continue
                # Borrow shortcut (extern args only): exact frozen layout
                # match with rel == 0 reuses the backing tensor instead of
                # a fresh as_strided view.
                borrowed = _borrowed_var(inp.base, inp.sizes, inp.strides, inp.rel)
                if borrowed is not None:
                    tensor_vars[inp.key] = borrowed
                    continue
                view = _fresh_view()
                _emit(
                    f"{view} = torch.as_strided({backing}, "
                    f"{tuple(inp.sizes)!r}, "
                    f"{tuple(inp.strides)!r}, "
                    f"storage_offset={backing}.storage_offset() "
                    f"+ {int(inp.rel)})"
                )
                tensor_vars[inp.key] = view
            decoded_args = [render_frozen_arg(a, tensor_vars) for a in step.args]
            decoded_kwargs = [
                f"{name}={render_frozen_arg(tree, tensor_vars)}"
                for name, tree in step.kwargs
            ]
            call_expr = ", ".join(decoded_args + decoded_kwargs)
            tgt = extern_targets[step.target_dotted]
            results = list(step.results)
            multi = not (
                getattr(step, "result_container", "sequence") == "tensor"
                and len(results) == 1
            )
            holder = _fresh_view()
            # No per-extern inference_mode() enter/exit: the per-call
            # _already_inference flag selects the provenance branch once.
            # Both branches compute the same op math; only tensor
            # provenance differs (inference tensors vs normal tensors).
            # Views/args above are shared (no duplicated as_strided).
            _emit("if _already_inference:")
            _emit(f"    {holder} = {tgt}({call_expr})")
            _emit("else:")
            if inference:
                _emit("    with torch.inference_mode():")
                _emit(f"        {holder} = {tgt}({call_expr})")
            else:
                _emit(f"    {holder} = {tgt}({call_expr})")
            if multi:
                _emit(f"if not isinstance({holder}, (list, tuple)):")
                _emit(
                    "    raise LoopsIrUnsupportedError("
                    "'extern produced a non-tensor output')"
                )
                _emit(f"if len({holder}) != {len(results)}:")
                _emit(
                    "    raise LoopsIrUnsupportedError('extern output count mismatch')"
                )
                produced = []
                for i in range(len(results)):
                    item = _fresh_view()
                    _emit(f"{item} = {holder}[{i}]")
                    _emit(f"if not isinstance({item}, torch.Tensor):")
                    _emit(
                        "    raise LoopsIrUnsupportedError("
                        "'extern produced a non-tensor output')"
                    )
                    produced.append(item)
            else:
                _emit(f"if not isinstance({holder}, torch.Tensor):")
                _emit(
                    "    raise LoopsIrUnsupportedError("
                    "'extern produced a non-tensor output')"
                )
                produced = [holder]
            for res, item in zip(results, produced):
                _emit(f"if {item}.dtype != {render_frozen_arg(res.dtype, {})}:")
                _emit(
                    "    raise LoopsIrUnsupportedError("
                    f"'eager result layout mismatch for {res.name}')"
                )
                _emit(f"if tuple({item}.shape) != {tuple(res.sizes)!r}:")
                _emit(
                    "    raise LoopsIrUnsupportedError("
                    f"'eager result layout mismatch for {res.name}')"
                )
                _emit(f"if tuple({item}.stride()) != {tuple(res.strides)!r}:")
                _emit(
                    "    raise LoopsIrUnsupportedError("
                    f"'eager result layout mismatch for {res.name}')"
                )
                _emit(f"{buf_vars[res.name]} = {item}")
    out_vars: list[str] = []
    for out in out_plans:
        backing = buf_vars[out.base]
        if _is_flat_span_alias(out.base, out.sizes, out.strides, out.rel):
            out_vars.append(backing)
        else:
            view = _fresh_view()
            _emit(
                f"{view} = torch.as_strided({backing}, "
                f"{tuple(out.sizes)!r}, "
                f"{tuple(out.strides)!r}, "
                f"storage_offset={backing}.storage_offset() + {int(out.rel)})"
            )
            out_vars.append(view)
    if len(out_vars) == 1:
        _emit(f"return ({out_vars[0]},)")
    else:
        _emit(f"return ({', '.join(out_vars)})")
    lines.extend(body)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File-backed loading + optional PTX capture (compile time only).
# ---------------------------------------------------------------------------


def load_loops_ir_wrapper(source: str) -> tuple[Any, str, str, list[tuple[int, str]]]:
    """Write ``source`` to the code cache and import it file-backed.

    Returns ``(module, cache_key, cache_path, cache_linemap)`` using the
    real ``mod.key``/``mod.__file__`` attributes. Import executes the
    embedded ``async_compile.wait(globals())``, so kernels are resolved
    before this returns.
    """
    from torch._inductor.codecache import PyCodeCache

    key, path = PyCodeCache.write(source)
    linemap = [(lineno, path) for lineno in range(1, len(source.splitlines()) + 1)]
    mod = PyCodeCache.load_by_key_path(key, path, linemap)
    real_key = getattr(mod, "key", key)
    real_path = getattr(mod, "__file__", path) or path
    return mod, real_key, real_path, linemap


def capture_triton_ptx(mod: Any, kernel_names: list[str]) -> dict[str, str]:
    """Best-effort PTX capture from resolved ``CachingAutotuner`` state.

    Prefers the prescribed ``launchers[0].bin.asm`` (present on dynamic
    launchers with cubin storage), then falls back to precompile results
    exposing ``kernel.asm`` -- both read compile-only state, never
    executing the kernel and never altering numerics. Missing entries are
    skipped (returned dict is partial). Note: under the default static
    Triton launcher the compiled result keeps only the cubin path, so PTX
    is unavailable before execution and the entry is skipped honestly.
    """
    ptx: dict[str, str] = {}
    for name in kernel_names:
        try:
            tuner = getattr(mod, name)
        except AttributeError:
            continue
        try:
            launchers = getattr(tuner, "launchers", None) or []
            if not launchers and hasattr(tuner, "precompile"):
                tuner.precompile()
                launchers = getattr(tuner, "launchers", None) or []
            for launcher in launchers:
                binary = getattr(launcher, "bin", None)
                asm = getattr(binary, "asm", None) or {}
                text = asm.get("ptx")
                if text:
                    ptx[name] = text
                    break
            if name in ptx:
                continue
            for result in getattr(tuner, "compile_results", None) or []:
                asm = getattr(getattr(result, "kernel", None), "asm", None) or {}
                text = asm.get("ptx")
                if text:
                    ptx[name] = text
                    break
        except Exception:
            continue
    return ptx
