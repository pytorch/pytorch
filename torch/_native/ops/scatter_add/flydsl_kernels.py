"""FlyDSL row scatter-add kernel: ``dst[index[i], :] += src[i, :]``.

One wave per source row. The wave id is made uniform with ``readfirstlane``,
so the row, its destination index and both row bases are wave-uniform. Lanes
then take consecutive 4-byte words of the row (two halves or one fp32), so
every wave-wide load and atomic instruction covers contiguous addresses (see
``flydsl_geometry.py``):

* loads: 32-bit buffer loads of ``src``, issued in batches of
  ``_LOADS_IN_FLIGHT`` steps ahead of their atomics;
* order: each row starts at a different step (``row % steps``, wrapping
  around), so waves that run together work on different parts of the row.
  Starting every wave at column 0 keeps waves that update the same
  destination row on the same addresses at the same time. On gfx950, fp32
  with N=4096 and every row on one destination ran at 0.88x of ATen that
  way and 0.98x rotated, with no change when rows are spread out;
* adds: no-return atomics into ``dst``. Halves use **full** packed
  ``<2 x bf16/f16>`` atomics (two real values per instruction, unlike ATen's
  zero-padded ``fastAtomicAdd``); fp32 uses scalar atomics.

Two atomic forms are compiled on request and benchmarked against each other:

* ``"buffer"`` - ``buffer_atomic_{pk_add_*,add_f32}`` through a buffer resource;
* ``"global"`` - LLVM ``atomicrmw fadd`` on a global pointer (agent scope).
  gfx942 has ``global_atomic_pk_add_bf16`` but no buffer form, so bf16 there
  must use this mode.

Operands are passed as flat 1D views starting at each tensor's first element;
row strides are runtime arguments, so one compiled kernel serves every row
count and stride. The caller (``flydsl_impl.py``) guarantees: the byte span of
every operand fits a signed int32, ``src`` and ``dst`` bases/row strides are
4-byte aligned, and ``N`` is a whole number of 4-byte words. (The conds
currently ask for more: a 16-byte multiple ``N`` and a 16-byte aligned
``src``.)

``flyc.compile`` runs the launcher once on its sample arguments before it
returns. An accumulating kernel must not see live tensors there, so
specializations are compiled against disposable one-row scratch operands
(``_scratch_compile_args``); the real update then runs exactly once.

The compile target is FlyDSL's process-wide target (the ``ARCH`` override or
the first enumerated GPU), not the destination's device. ``compile_target_arch``
exposes it so callers can decline when the two differ.

Index validation matches ATen on ROCm, where ``CUDA_KERNEL_ASSERT`` aborts the
kernel: an index outside ``[0, M_dst)`` traps the wave (``llvm.trap``) before
any offset is formed from it, and that row is never written. If the MLIR
bindings lack the trap builder, the host wrapper instead enqueues a
stream-ordered ``torch._assert_async`` ahead of the launch (same abort
behavior, two extra small launches).
"""

import contextlib
import functools
import os
from typing import cast

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import is_rdna_arch

import torch
from torch._native.flydsl.cache import CachedCompile, CacheInfo
from torch._native.flydsl.compile_args import make_compile_arg, read_only_tensor
from torch._native.flydsl_utils import _get_flydsl_device_arch
from torch._native.instrumentation import instrumented_flydsl_cache

from .flydsl_geometry import lane_elems, MAX_STEPS_PER_LANE, steps_per_lane


_WAVES_PER_BLOCK = 4
# Source loads issued per lane before their atomics, so several loads are in
# flight (atomics may alias the source, so they are not reordered past them).
_LOADS_IN_FLIGHT = 8

_DTYPES = {
    torch.float32: ("f32", fx.Float32, 32),
    torch.float16: ("f16", fx.Float16, 16),
    torch.bfloat16: ("bf16", fx.BFloat16, 16),
}

ATOMIC_MODES = ("buffer", "global")

# Archs that have no buffer_atomic_pk_add_bf16 but do have the global form.
_NO_BUFFER_PK_ADD_BF16_ARCHS = ("gfx942",)

_device_trap = getattr(llvm, "intr_trap", None)
HAS_DEVICE_TRAP = _device_trap is not None


# Environment variables FlyDSL 0.3.4.1 rereads on every target query:
# FLYDSL_COMPILE_BACKEND picks the backend; RocmBackend.detect_target() uses
# ARCH if set, else get_rocm_arch(), which reads FLYDSL_GPU_ARCH, then
# HSA_OVERRIDE_GFX_VERSION, and only then its cached hardware probe.
_TARGET_ENV_VARS = (
    "FLYDSL_COMPILE_BACKEND",
    "ARCH",
    "FLYDSL_GPU_ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
)


@functools.lru_cache(maxsize=8)
def _compile_target_arch_for(env: tuple[str | None, ...]) -> str | None:
    del env  # cache key only
    try:
        from flydsl.compiler.backends import current_target

        return str(current_target().arch).split(":", 1)[0].lower()
    except Exception:
        return None


def compile_target_arch() -> str | None:
    """The gfx name FlyDSL compiles for in this process, or None if unknown.

    FlyDSL re-reads its environment on every query, which costs a few
    microseconds per dispatch. Apart from its cached hardware probe, the
    result depends only on ``_TARGET_ENV_VARS``, so it is cached per value of
    those variables; changing any of them is seen on the next call.
    """
    return _compile_target_arch_for(
        tuple(os.environ.get(name) for name in _TARGET_ENV_VARS)
    )


def wave_size(arch: str) -> int:
    return 32 if is_rdna_arch(arch) else 64


def default_atomic_mode(dtype: torch.dtype, arch: str) -> str:
    if dtype == torch.bfloat16 and arch in _NO_BUFFER_PK_ADD_BF16_ARCHS:
        return "global"
    return "buffer"


def atomic_mode_supported(mode: str, dtype: torch.dtype, arch: str) -> bool:
    if mode not in ATOMIC_MODES:
        return False
    if mode == "buffer" and dtype == torch.bfloat16:
        return arch not in _NO_BUFFER_PK_ADD_BF16_ARCHS
    return True


def _build_row_scatter_add_module(
    n: int,
    dtype_str: str,
    index_bits: int,
    atomic_mode: str,
    device_trap: bool,
    arch: str,
):
    torch_dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[
        dtype_str
    ]
    _, elem_dtype, elem_bits = _DTYPES[torch_dtype]
    is_f32 = dtype_str == "f32"
    wave = wave_size(arch)
    vec = lane_elems(torch_dtype.itemsize)
    vecs_per_row = n // vec
    steps = steps_per_lane(n, torch_dtype.itemsize, wave)
    # Every step but the last covers whole waves; the last may be partial.
    last_partial = vecs_per_row % wave != 0
    full_steps = steps - 1 if last_partial else steps
    block_threads = wave * _WAVES_PER_BLOCK
    if n % vec or steps > MAX_STEPS_PER_LANE:
        raise ValueError(f"row scatter-add: unsupported N={n} for {dtype_str}")
    if atomic_mode not in ATOMIC_MODES:
        raise ValueError(f"row scatter-add: unknown atomic mode {atomic_mode!r}")
    if index_bits not in (32, 64):
        raise ValueError(f"row scatter-add: unsupported index width {index_bits}")

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def row_scatter_add_kernel(
        dst: fx.Tensor,
        src: fx.Tensor,
        index: fx.Tensor,
        m_src: fx.Int32,
        m_dst: fx.Int32,
        dst_row_stride: fx.Int32,
        src_row_stride: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % fx.Int32(wave)
        wave_id = fx.Int32(
            rocdl.readfirstlane(T.i32, (tid // fx.Int32(wave)).ir_value())
        )
        row = bid * fx.Int32(_WAVES_PER_BLOCK) + wave_id

        src_buf = fx.rocdl.make_buffer_tensor(src)
        index_buf = fx.rocdl.make_buffer_tensor(index)
        dst_buf = fx.rocdl.make_buffer_tensor(dst)
        dst_iter = fx.get_iter(dst)
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), elem_bits)
        if const_expr(is_f32):
            add_atom = fx.make_copy_atom(
                fx.rocdl.BufferAtomicAdd(fx.Float32), fx.Float32
            )
            frag = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        else:
            add_atom = fx.make_copy_atom(
                fx.rocdl.BufferAtomicPkAdd(elem_dtype), elem_dtype
            )
            frag = fx.make_rmem_tensor(fx.make_layout(2, 1), elem_dtype)

        def add_vector(vals, dst_off):
            if const_expr(is_f32):
                for e in range_constexpr(vec):
                    if const_expr(atomic_mode == "global"):
                        fx.atomic_add(
                            dst_iter + dst_off + e,
                            vals[e],
                            syncscope=fx.rocdl.SyncScope.Agent,
                        )
                    else:
                        frag.store(fx.Vector.from_elements([vals[e]], fx.Float32))
                        fx.copy_atom_call(
                            add_atom,
                            frag,
                            fx.make_view(
                                fx.get_iter(dst_buf) + dst_off + e,
                                fx.make_layout(1, 1),
                            ),
                        )
            else:
                for p in range_constexpr(vec // 2):
                    pair = fx.Vector.from_elements(
                        [vals[2 * p], vals[2 * p + 1]], elem_dtype
                    )
                    if const_expr(atomic_mode == "global"):
                        # A raw !llvm.ptr: fx.atomic_add only type-checks
                        # fly pointers, whose element type is the scalar.
                        fx.atomic_add(
                            (dst_iter + dst_off + 2 * p).llvm_ptr,
                            pair,
                            syncscope=fx.rocdl.SyncScope.Agent,
                        )
                    else:
                        frag.store(pair)
                        fx.copy_atom_call(
                            add_atom,
                            frag,
                            fx.make_view(
                                fx.get_iter(dst_buf) + dst_off + 2 * p,
                                fx.make_layout(2, 1),
                            ),
                        )

        def step_col(step, rot):
            # Rotated step index of this row; full steps only, the tail stays last.
            if const_expr(full_steps > 1):
                return (fx.Int32(step) + rot) % fx.Int32(full_steps)
            return fx.Int32(step)

        def load_word(v, src_base):
            r = fx.make_rmem_tensor(vec, elem_dtype)
            fx.copy_atom_call(
                load_atom,
                fx.make_view(
                    fx.get_iter(src_buf) + src_base + v * fx.Int32(vec),
                    fx.make_layout(vec, 1),
                ),
                r,
            )
            return r.load()

        if row < m_src:
            dst_index = fx.Int64(index_buf[row])
            in_range = (dst_index >= fx.Int64(0)) & (dst_index < fx.Int64(m_dst))
            if const_expr(device_trap):
                out_of_range = (dst_index < fx.Int64(0)) | (
                    dst_index >= fx.Int64(m_dst)
                )
                if out_of_range:
                    _device_trap()
            if in_range:
                # Only formed after the bounds check, so the narrowing is exact.
                dst_base = fx.Int32(dst_index) * dst_row_stride
                src_base = row * src_row_stride
                rot = row % fx.Int32(max(full_steps, 1))
                for first in range_constexpr(0, full_steps, _LOADS_IN_FLIGHT):
                    last = min(first + _LOADS_IN_FLIGHT, full_steps)
                    vs = [
                        lane + step_col(s, rot) * fx.Int32(wave)
                        for s in range(first, last)
                    ]
                    words = [load_word(v, src_base) for v in vs]
                    for v, w in zip(vs, words):
                        add_vector(w, dst_base + v * fx.Int32(vec))
                if const_expr(last_partial):
                    v = lane + fx.Int32(full_steps * wave)
                    if v < fx.Int32(vecs_per_row):
                        w = load_word(v, src_base)
                        add_vector(w, dst_base + v * fx.Int32(vec))

    @flyc.jit
    def launch_row_scatter_add(
        dst: fx.Tensor,
        src: fx.Tensor,
        index: fx.Tensor,
        m_src: fx.Int32,
        m_dst: fx.Int32,
        dst_row_stride: fx.Int32,
        src_row_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        num_blocks = (m_src + _WAVES_PER_BLOCK - 1) // _WAVES_PER_BLOCK
        launcher = row_scatter_add_kernel(
            dst, src, index, m_src, m_dst, dst_row_stride, src_row_stride
        )
        launcher.launch(
            grid=(num_blocks, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_row_scatter_add


@instrumented_flydsl_cache(
    "aten::scatter_add",
    key_fn=lambda n, dtype, index_bits, mode, trap, arch, backend, dev, *a, **k: (
        f"row N={n} {dtype} idx{index_bits} {mode} trap={trap} {arch} "
        f"backend={backend} device={dev}"
    ),
)
def _compile_row_scatter_add(
    n: int,
    dtype: str,
    index_bits: int,
    atomic_mode: str,
    device_trap: bool,
    arch: str,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    # backend and device_index are cache keys only: the returned callable holds
    # context-local module/function handles, so it is cached per device.
    del backend, device_index
    # compile_args is a zero-argument callable so scratch operands are only
    # allocated on a miss; flyc.compile launches the kernel once on them.
    dst_flat, src_flat, index_1d, m_src, m_dst, sd, ss, stream = compile_args()
    launch = _build_row_scatter_add_module(
        n, dtype, index_bits, atomic_mode, device_trap, arch
    )
    return flyc.compile(
        launch,
        make_compile_arg(dst_flat),
        make_compile_arg(src_flat, read_only=True),
        make_compile_arg(index_1d, read_only=True),
        m_src,
        m_dst,
        sd,
        ss,
        stream,
    )


def _flat_span(t2d: torch.Tensor) -> torch.Tensor:
    """1D view over the bytes a (rows, N) view with unit inner stride touches."""
    rows, n = t2d.shape
    span = (rows - 1) * t2d.stride(0) + n if rows else 0
    return t2d.as_strided((span,), (1,))


def _scratch_compile_args(
    n: int, dtype: torch.dtype, index_dtype: torch.dtype, device, stream
):
    """One-row operands for ``flyc.compile``'s sample launch (adds 0 to 0)."""
    dst = torch.zeros(n, dtype=dtype, device=device)
    src = torch.zeros(n, dtype=dtype, device=device)
    index = torch.zeros(1, dtype=index_dtype, device=device)
    return (dst, src, index, 1, 1, n, n, stream)


def row_scatter_add_(
    dst_2d: torch.Tensor,
    index_1d: torch.Tensor,
    src_2d: torch.Tensor,
    *,
    atomic_mode: str | None = None,
) -> torch.Tensor:
    """``dst_2d[index_1d[i], :] += src_2d[i, :]`` in place; returns ``dst_2d``.

    Preconditions (checked by ``flydsl_impl.py``, not here): CUDA/HIP tensors
    on one device; ``dst_2d`` (M_dst, N) and ``src_2d`` (M_src, N) with unit
    inner stride, matching dtype; ``index_1d`` contiguous int32/int64 of length
    M_src; spans/alignment as described in the module docstring.
    """
    m_src, n = src_2d.shape
    m_dst = dst_2d.shape[0]
    if m_src == 0:
        return dst_2d
    arch = _get_flydsl_device_arch(dst_2d.device.index)
    if arch is None or arch != compile_target_arch():
        raise ValueError(
            f"FlyDSL compiles for {compile_target_arch()!r}, but the destination "
            f"is on {arch!r}"
        )
    if atomic_mode is not None and not atomic_mode_supported(
        atomic_mode, dst_2d.dtype, arch
    ):
        raise ValueError(
            f"atomic mode {atomic_mode!r} unsupported for {dst_2d.dtype} on {arch}"
        )
    launch_rows(
        _flat_span(dst_2d),
        index_1d,
        _flat_span(src_2d),
        m_src,
        m_dst,
        n,
        dst_2d.stride(0),
        src_2d.stride(0),
        atomic_mode=atomic_mode,
    )
    return dst_2d


def launch_rows(
    dst_flat: torch.Tensor,
    index_1d: torch.Tensor,
    src_flat: torch.Tensor,
    m_src: int,
    m_dst: int,
    n: int,
    dst_row_stride: int,
    src_row_stride: int,
    *,
    atomic_mode: str | None = None,
) -> None:
    """Launch on flat 1D spans, with no compile-target check.

    For callers that have already checked ``compile_target_arch()`` against
    the destination device (the dispatch conds do); otherwise use
    ``row_scatter_add_``. ``m_src`` must be positive.
    """
    device = dst_flat.device
    device_index = device.index
    arch = _get_flydsl_device_arch(device_index)
    dtype = dst_flat.dtype
    mode = atomic_mode or default_atomic_mode(dtype, arch)
    index_bits = index_1d.element_size() * 8
    switch = device_index != torch.cuda.current_device()
    with torch.cuda.device(device) if switch else contextlib.nullcontext():
        stream = torch.cuda.current_stream(device)
        if not HAS_DEVICE_TRAP:
            lo, hi = torch.aminmax(index_1d)
            torch._assert_async(
                (lo >= 0) & (hi < m_dst),
                "scatter_add/index_add: index out of bounds",
            )
        compiled = _compile_row_scatter_add(
            n,
            _DTYPES[dtype][0],
            index_bits,
            mode,
            HAS_DEVICE_TRAP,
            arch,
            flyc.compile_backend_name(),
            device_index,
            compile_args=lambda: _scratch_compile_args(
                n, dtype, index_1d.dtype, device, stream
            ),
        )
        compiled(
            dst_flat,
            read_only_tensor(src_flat),
            read_only_tensor(index_1d),
            m_src,
            m_dst,
            dst_row_stride,
            src_row_stride,
            stream,
        )


def clear_row_scatter_add_cache() -> None:
    cast(CachedCompile, _compile_row_scatter_add).cache_clear()


def row_scatter_add_cache_info() -> CacheInfo:
    return cast(CachedCompile, _compile_row_scatter_add).cache_info()
