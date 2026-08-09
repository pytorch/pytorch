"""CuTeDSL inner-tree sum reduction kernels.

Bitwise port of the inner-tree reduction kernels in
``aten/src/ATen/native/cuda/ReduceSumProdKernel.cu`` (with helpers from
``ReduceInnerTree.cuh`` / ``WarpReduce.cuh``). Each output element is the
sum of ``N`` contiguous input elements; ``N`` is baked in at ``cute.compile``
time so the per-row vector tree, streaming tree, and launch planning all
unroll at trace time and the accumulation order matches the CUDA kernel
exactly.

Three execution shapes, selected on the host exactly as
``try_inner_tree_reduction`` does:

* Multirow (``inner_tree_reduction_multirow_kernel``): one thread per row,
  used when ``N <= vec_size * kMultiRowMaxLoads``.
* Looped (``inner_tree_reduction_looped_kernel``): warps cooperate per row,
  batches accumulated inside the kernel; used when ``num_batches <= 3``.
* Two-kernel (``inner_tree_reduction_kernel`` +
  ``inner_tree_accumulate_multirow_kernel``): a partials kernel followed by a
  linear per-row accumulate; used otherwise.

Bitwise-critical ordering reproduced here:

* ``vec_size = 16 / itemsize`` grouping; the in-vec tree order
  (``inner_tree_reduce``) for vec_size >= 4, else linear. Zero-padded tails
  are bitwise identical between tree and linear (``x + 0 == x``), so a single
  uniform choice covers both the full and tail cases.
* Warp butterfly with ASCENDING shuffle-XOR offsets ``1,2,4,...`` (matches
  ``WarpReduceDirection::ASCENDING``); we hand-roll it because
  ``cute.arch.warp_reduction`` uses descending offsets, which is a different
  (and thus bitwise-different) order.
* Streaming inner-tree carry across loads (``streaming_inner_tree_step``,
  merge count = trailing-zero bits of ``load + 1``).
* Accumulation dtype: fp32 for fp16/bf16/fp32, fp64 for fp64 -- matches
  ``reduce_dispatch`` so rounding is identical. The inner-tree bitwise tests
  only exercise fp32 and fp64 (they upcast narrower data first); fp16/bf16
  native reductions are functionally supported but not bitwise-validated.

Because ``N`` is compile-time, a warp/batch that owns fewer than the unrolled
number of loads contributes zeros for the extra loads; zeros are additive
identities, so this reproduces the ragged CUDA loop bit-for-bit.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]  # noqa: TC002

import cutlass
import cutlass.cute as cute

# fp64 is part of the bitwise contract; import Float64 directly so an
# unsupported runtime surfaces a clear error rather than silently dropping it.
from cutlass import BFloat16, const_expr, Float16, Float32, Float64, Int32

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from .inner_tree_plan import (
    _ceil_div,
    _K_MULTIROW_MAX_LOADS,
    _K_TWO_KERNEL_THRESHOLD,
    _next_power_of_2,
    _WARP_SIZE,
    compute_inner_tree_params,
    InnerTreeParams,
    vec_size as _vec_size,
)


_MULTIROW_THREADS = 128
_MULTIROW_MAX_DEPTH = 6  # kMaxDepth in the multirow kernel
_ACCUM_THREADS = 128  # kAccumulateThreads

_TORCH_TO_CUTE = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.float64: Float64,
}


def _acc_for(in_dt):
    return Float64 if in_dt is Float64 else Float32


# Torch accumulator dtype for the two-kernel partials buffer (fp32 for
# fp16/bf16/fp32, fp64 for fp64) -- keeps cross-batch accumulation in the
# accumulator dtype, matching ATen.
_ACC_TORCH_DTYPE = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


# ---------------------------------------------------------------------------
# Trace-time reduction helpers. These run during ``cute.compile`` tracing and
# emit the exact add DAG; ``vals`` / ``tree`` are Python lists of cute
# ``Numeric`` register values.
# ---------------------------------------------------------------------------


def _linear_reduce(vals):
    acc = vals[0]
    for i in range(1, len(vals)):
        acc = acc + vals[i]
    return acc


def _inner_tree_reduce(vals):
    v = list(vals)
    n = len(v)
    stride = 1
    while stride < n:
        i = 0
        while i + stride < n:
            v[i] = v[i] + v[i + stride]
            i += stride * 2
        stride *= 2
    return v[0]


def _reduce_vec(vals, vec_size: int):
    return _inner_tree_reduce(vals) if vec_size >= 4 else _linear_reduce(vals)


def _streaming_push(tree, val, load: int, max_depth: int) -> None:
    """Port of ``streaming_inner_tree_step``: merge count is the number of
    trailing zero bits of ``load + 1`` (``__ffs(load + 1) - 1``), capped at
    ``max_depth``. ``carry = tree_accs[--top] + carry`` keeps the existing
    accumulator on the left."""
    trailing_zeros = ((load + 1) & -(load + 1)).bit_length() - 1
    carry = val
    for _ in range(min(trailing_zeros, max_depth)):
        carry = tree.pop() + carry
    tree.append(carry)


def _warp_butterfly(val):
    """Ascending-offset shuffle-XOR butterfly add reduce (matches the CUDA
    ``warp_reduce`` ASCENDING direction)."""
    offset = 1
    while offset < _WARP_SIZE:
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=offset, mask_and_clamp=_WARP_SIZE - 1
        )
        offset <<= 1
    return val


def _load_vec_static(mIn, row, base: int, N: int, vec_size: int, acc, zero):
    """Load + reduce one vector group with compile-time ``base`` (multirow)."""
    vals = []
    for i in range(vec_size):
        off = base + i
        vals.append(acc(mIn[row, off]) if off < N else zero)
    return _reduce_vec(vals, vec_size)


@cute.jit
def _load_vec_dyn(
    mIn,
    row,
    base,
    N: cutlass.Constexpr,
    vec_size: cutlass.Constexpr,
    acc: cutlass.Constexpr,
    zero,
):
    """Load + reduce one vector group with a runtime ``base`` (warp paths).

    ``@cute.jit`` so the per-element bounds check is lowered by the DSL
    instead of evaluated as a Python bool at trace time (the kernel body's
    AST transform does not reach into plain helpers). Each element is guarded
    against the row width so out-of-row reads never happen and tail elements
    contribute zero.
    """
    vals = []
    for i in cutlass.range_constexpr(vec_size):
        off = base + Int32(i)
        v = zero
        if off < Int32(N):
            v = acc(mIn[row, off])
        vals.append(v)
    return _reduce_vec(vals, vec_size)


# ---------------------------------------------------------------------------
# Kernel builders. Each returns a ``@cute.jit`` launcher specialized on the
# compile-time parameters.
# ---------------------------------------------------------------------------


def _make_multirow(in_dt, acc, out_dt, N: int, vec_size: int):
    num_loads = _next_power_of_2(_ceil_div(N, vec_size))

    @cute.kernel
    def _kernel(mIn: cute.Tensor, mOut: cute.Tensor, num_outputs: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row = bx * Int32(_MULTIROW_THREADS) + tx
        if row < num_outputs:
            zero = acc(0.0)
            tree = []
            for load in cutlass.range_constexpr(num_loads):
                val = _load_vec_static(
                    mIn, row, load * vec_size, N, vec_size, acc, zero
                )
                _streaming_push(tree, val, load, _MULTIROW_MAX_DEPTH)
            mOut[row] = out_dt(tree[0])

    @cute.jit
    def _launch(
        mIn: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_outputs: Int32,
        grid: Int32,
    ):
        _kernel(mIn, mOut, num_outputs).launch(
            grid=[grid, 1, 1], block=[_MULTIROW_THREADS, 1, 1], stream=stream
        )

    return _launch


def _make_looped(in_dt, acc, out_dt, N: int, vec_size: int, p: InnerTreeParams):
    wpr = p.num_warps  # warps_per_reduction == num_warps in launch_looped
    rows_per_block = p.rows_per_block
    total_warps = wpr * rows_per_block
    wle = _WARP_SIZE * vec_size

    # Per-batch geometry is fully compile-time (num_batches <= 3, fixed N).
    batches = []
    for b in range(p.num_batches):
        batch_offset = b * p.batch_total_elements
        remaining = min(p.batch_total_elements, N - batch_offset)
        lpw = _ceil_div(remaining, wpr * wle)
        if lpw > 1:
            lpw = _next_power_of_2(lpw)
        batches.append((batch_offset, remaining, lpw, lpw * wle))

    @cute.kernel
    def _kernel(mIn: cute.Tensor, mOut: cute.Tensor, num_outputs: Int32):
        tx, ty, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        lane = tx
        global_warp_id = ty
        row_in_block = global_warp_id // Int32(wpr)
        warp_id = global_warp_id % Int32(wpr)
        row = bx * Int32(rows_per_block) + row_in_block
        row_active = row < num_outputs

        warp_writes = None  # bound for type checker; used only when wpr > 1
        if const_expr(wpr > 1):
            smem = cutlass.utils.SmemAllocator()
            warp_writes = smem.allocate_tensor(
                acc, cute.make_layout(total_warps), byte_alignment=8
            )

        zero = acc(0.0)
        final_sum = zero
        for b in cutlass.range_constexpr(p.num_batches):
            batch_offset, remaining_c, lpw, warp_chunk = batches[b]
            rem = Int32(0)
            if row_active:
                rem = Int32(remaining_c)
            warp_off = warp_id * Int32(warp_chunk)
            if warp_off > rem:
                warp_off = rem
            warp_start = Int32(batch_offset) + warp_off
            this_warp_elements = Int32(warp_chunk)
            tail = rem - warp_off
            if this_warp_elements > tail:
                this_warp_elements = tail
            this_batch_loads = (this_warp_elements + Int32(wle - 1)) // Int32(wle)

            tree = []
            for load in cutlass.range_constexpr(lpw):
                base = warp_start + Int32(load * wle) + lane * Int32(vec_size)
                v = zero
                if Int32(load) < this_batch_loads:
                    v = _warp_butterfly(
                        _load_vec_dyn(mIn, row, base, N, vec_size, acc, zero)
                    )
                _streaming_push(tree, v, load, p.depth)
            warp_acc = tree[0]

            if const_expr(wpr > 1):
                assert warp_writes is not None  # noqa: S101
                if lane == Int32(0):
                    warp_writes[row_in_block * Int32(wpr) + warp_id] = warp_acc
                cute.arch.barrier()
                merged = zero
                if lane < Int32(wpr):
                    merged = warp_writes[row_in_block * Int32(wpr) + lane]
                warp_acc = _warp_butterfly(merged)
                if const_expr(b + 1 < p.num_batches):
                    cute.arch.barrier()

            final_sum = final_sum + warp_acc

        if row_active:
            if lane == Int32(0):
                if warp_id == Int32(0):
                    mOut[row] = out_dt(final_sum)

    @cute.jit
    def _launch(
        mIn: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_outputs: Int32,
        grid: Int32,
    ):
        _kernel(mIn, mOut, num_outputs).launch(
            grid=[grid, 1, 1], block=[_WARP_SIZE, total_warps, 1], stream=stream
        )

    return _launch


def _make_two_partial(in_dt, acc, out_dt, N: int, vec_size: int, p: InnerTreeParams):
    wpr = p.num_warps
    num_batches = p.num_batches
    wle = _WARP_SIZE * vec_size
    eff = p.effective_loads

    # Only two distinct per-batch widths: full batches and the (shorter) last
    # batch. Precompute both; the kernel selects on ``batch_idx`` at runtime.
    full_remaining = p.batch_total_elements
    last_remaining = N - (num_batches - 1) * p.batch_total_elements
    warp_chunk_full = eff * wle

    def _lpw(remaining: int) -> int:
        lpw = _ceil_div(remaining, wpr * wle)
        return _next_power_of_2(lpw) if lpw > 1 else lpw

    warp_chunk_last = _lpw(last_remaining) * wle

    @cute.kernel
    def _kernel(mIn: cute.Tensor, mPartials: cute.Tensor):
        # Grid is exactly num_outputs * num_batches, so no row bounds check.
        tx, ty, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        lane = tx
        warp_id = ty
        row = bx // Int32(num_batches)
        batch_idx = bx % Int32(num_batches)

        remaining = Int32(full_remaining)
        warp_chunk = Int32(warp_chunk_full)
        if batch_idx == Int32(num_batches - 1):
            remaining = Int32(last_remaining)
            warp_chunk = Int32(warp_chunk_last)
        batch_offset = batch_idx * Int32(p.batch_total_elements)

        warp_writes = None  # bound for type checker; used only when wpr > 1
        if const_expr(wpr > 1):
            smem = cutlass.utils.SmemAllocator()
            warp_writes = smem.allocate_tensor(
                acc, cute.make_layout(wpr), byte_alignment=8
            )

        warp_off = warp_id * warp_chunk
        if warp_off > remaining:
            warp_off = remaining
        warp_start = batch_offset + warp_off
        this_warp_elements = warp_chunk
        tail = remaining - warp_off
        if this_warp_elements > tail:
            this_warp_elements = tail
        this_batch_loads = (this_warp_elements + Int32(wle - 1)) // Int32(wle)

        zero = acc(0.0)
        tree = []
        for load in cutlass.range_constexpr(eff):
            base = warp_start + Int32(load * wle) + lane * Int32(vec_size)
            v = zero
            if Int32(load) < this_batch_loads:
                v = _warp_butterfly(
                    _load_vec_dyn(mIn, row, base, N, vec_size, acc, zero)
                )
            _streaming_push(tree, v, load, p.depth)
        warp_acc = tree[0]

        if const_expr(wpr > 1):
            assert warp_writes is not None  # noqa: S101
            if lane == Int32(0):
                warp_writes[warp_id] = warp_acc
            cute.arch.barrier()
            merged = zero
            if lane < Int32(wpr):
                merged = warp_writes[lane]
            warp_acc = _warp_butterfly(merged)

        if lane == Int32(0):
            if warp_id == Int32(0):
                # Store partials in the accumulator dtype so the cross-batch
                # sum stays in fp32 for fp16/bf16 (matches ATen's
                # accumulate-in-fp32-round-once); downcast happens only in the
                # accumulate kernel's final write.
                mPartials[row * Int32(num_batches) + batch_idx] = acc(warp_acc)

    @cute.jit
    def _launch(
        mIn: cute.Tensor,
        mPartials: cute.Tensor,
        stream: cuda.CUstream,
        grid: Int32,
    ):
        _kernel(mIn, mPartials).launch(
            grid=[grid, 1, 1], block=[_WARP_SIZE, wpr, 1], stream=stream
        )

    return _launch


def _make_two_accum(acc, out_dt, num_batches: int):
    @cute.kernel
    def _kernel(mPartials: cute.Tensor, mOut: cute.Tensor, num_outputs: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row = bx * Int32(_ACCUM_THREADS) + tx
        if row < num_outputs:
            base = row * Int32(num_batches)
            # Partials are in the accumulator dtype; accumulate in that dtype
            # and downcast to the output dtype only on the final store.
            s = mPartials[base]
            # Runtime loop, NOT range_constexpr: num_batches grows with N
            # (thousands for very large reductions), so unrolling would make
            # cute.compile blow up. Linear accumulation order matches the CUDA
            # kernel regardless of unroll.
            for b in cutlass.range(1, num_batches):
                s = s + mPartials[base + b]
            mOut[row] = out_dt(s)

    @cute.jit
    def _launch(
        mPartials: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_outputs: Int32,
        grid: Int32,
    ):
        _kernel(mPartials, mOut, num_outputs).launch(
            grid=[grid, 1, 1], block=[_ACCUM_THREADS, 1, 1], stream=stream
        )

    return _launch


# ---------------------------------------------------------------------------
# Compile cache + host dispatch.
# ---------------------------------------------------------------------------


def _fake_2d(dt, N: int):
    return cute.runtime.make_fake_tensor(
        dt, (cute.sym_int(), N), stride=(cute.sym_int64(), 1)
    )


def _fake_1d(dt):
    return cute.runtime.make_fake_tensor(
        dt, (cute.sym_int(),), stride=(cute.sym_int64(),)
    )


def _fake_1d_contig(dt):
    return cute.runtime.make_fake_tensor(dt, (cute.sym_int(),), stride=(1,))


@instrumented_cutedsl_cache(
    "aten::sum",
    key_fn=lambda torch_dtype, N: f"multirow {torch_dtype} N={N}",
)
def _compile_multirow(torch_dtype: torch.dtype, N: int):
    in_dt = _TORCH_TO_CUTE[torch_dtype]
    acc = _acc_for(in_dt)
    launcher = _make_multirow(in_dt, acc, in_dt, N, _vec_size(torch_dtype.itemsize))
    return cute.compile(
        launcher,
        _fake_2d(in_dt, N),
        _fake_1d(in_dt),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


@instrumented_cutedsl_cache(
    "aten::sum",
    key_fn=lambda torch_dtype, N, num_warps, _bte, num_batches, *_rest: (
        f"looped {torch_dtype} N={N} warps={num_warps} batches={num_batches}"
    ),
)
def _compile_looped(
    torch_dtype: torch.dtype,
    N: int,
    num_warps: int,
    batch_total_elements: int,
    num_batches: int,
    depth: int,
    rows_per_block: int,
    effective_loads: int,
):
    in_dt = _TORCH_TO_CUTE[torch_dtype]
    acc = _acc_for(in_dt)
    p = InnerTreeParams(
        num_warps,
        batch_total_elements,
        num_batches,
        depth,
        rows_per_block,
        effective_loads,
    )
    launcher = _make_looped(in_dt, acc, in_dt, N, _vec_size(torch_dtype.itemsize), p)
    return cute.compile(
        launcher,
        _fake_2d(in_dt, N),
        _fake_1d(in_dt),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


@instrumented_cutedsl_cache(
    "aten::sum",
    key_fn=lambda torch_dtype, N, num_warps, _bte, num_batches, *_rest: (
        f"two_partial {torch_dtype} N={N} warps={num_warps} batches={num_batches}"
    ),
)
def _compile_two_partial(
    torch_dtype: torch.dtype,
    N: int,
    num_warps: int,
    batch_total_elements: int,
    num_batches: int,
    depth: int,
    effective_loads: int,
):
    in_dt = _TORCH_TO_CUTE[torch_dtype]
    acc = _acc_for(in_dt)
    p = InnerTreeParams(
        num_warps,
        batch_total_elements,
        num_batches,
        depth,
        1,
        effective_loads,
    )
    launcher = _make_two_partial(in_dt, acc, acc, N, _vec_size(torch_dtype.itemsize), p)
    return cute.compile(
        launcher,
        _fake_2d(in_dt, N),
        _fake_1d_contig(acc),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        options="--enable-tvm-ffi",
    )


@instrumented_cutedsl_cache(
    "aten::sum",
    key_fn=lambda torch_dtype, num_batches: (
        f"two_accum {torch_dtype} batches={num_batches}"
    ),
)
def _compile_two_accum(torch_dtype: torch.dtype, num_batches: int):
    out_dt = _TORCH_TO_CUTE[torch_dtype]
    acc = _acc_for(out_dt)
    launcher = _make_two_accum(acc, out_dt, num_batches)
    return cute.compile(
        launcher,
        _fake_1d_contig(acc),
        _fake_1d(out_dt),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


def inner_tree_sum_into(out: torch.Tensor, src: torch.Tensor) -> None:
    """Compute ``out[r] = sum(src[r, :])`` for every row ``r``.

    ``src`` is a 2D ``(M, N)`` view with inner-dim stride 1 (outer row stride
    may differ from N); ``out`` is a 1D ``(M,)`` view (its stride may differ
    from 1). Both carry their strides into the kernel via the cute tensor, so
    strided outer inputs and strided outputs are handled directly.
    """
    m, n = src.shape
    dtype = src.dtype
    vec_size = _vec_size(dtype.itemsize)

    if n <= vec_size * _K_MULTIROW_MAX_LOADS:
        compiled = _compile_multirow(dtype, n)
        grid = _ceil_div(m, _MULTIROW_THREADS)
        compiled(src, out, m, grid)
        return

    p = compute_inner_tree_params(n, m, vec_size)
    if p.num_batches <= _K_TWO_KERNEL_THRESHOLD:
        compiled = _compile_looped(
            dtype,
            n,
            p.num_warps,
            p.batch_total_elements,
            p.num_batches,
            p.depth,
            p.rows_per_block,
            p.effective_loads,
        )
        grid = _ceil_div(m, p.rows_per_block)
        compiled(src, out, m, grid)
        return

    partials = torch.empty(
        m * p.num_batches, dtype=_ACC_TORCH_DTYPE[dtype], device=src.device
    )
    c1 = _compile_two_partial(
        dtype,
        n,
        p.num_warps,
        p.batch_total_elements,
        p.num_batches,
        p.depth,
        p.effective_loads,
    )
    c1(src, partials, m * p.num_batches)
    c2 = _compile_two_accum(dtype, p.num_batches)
    c2(partials, out, m, _ceil_div(m, _ACCUM_THREADS))
