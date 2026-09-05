"""Row statistics and the softmax-gradient transform in one kernel.

Replaces the five eager passes the chunked loop makes over its ``(Bc, V)``
logits buffer -- row max, subtract, gather, ``exp_``, row sum -- plus the
separate gradient-transform kernel, with a single launch that reads the raw
logits twice and writes the dense gradient-of-logits once.

Contract, per row ``n`` of the chunk, with ``m_n = max_v z[n, v]`` and
``l_n = sum_v exp(z[n, v] - m_n)``::

    g[n, v] = exp(z[n, v] - m_n) * (s_n / l_n) - s_n * [v == T_hat_n]
    term[n] = s_n * ((m_n + log(l_n)) - z[n, T_hat_n])

``g`` is what makes both parameter gradients plain GEMMs (see
``grad_logits_kernel``); the per-row loss contribution
``s_n * (lse_n - z_target_n)`` is formed in the kernel from the raw logits,
not the shifted ones -- that difference is shift invariant.

The logits stay raw: nothing here mutates them, so the eager loop's in-place
shift and ``exp_`` disappear along with their traffic, and the buffer can be
read a second time instead of being reconstructed.

Geometry: one block per row, columns split across its threads. A row's own
maximum and sum must be complete before any element of that row can be
written, and a block is the largest unit that can share them without a
grid-wide barrier -- which is why this cannot use the column-parallel grid of
the unfused transform, whose statistics arrive precomputed. Parallelism is
therefore the chunk's row count, not its element count, so blocks are wide by
default to keep enough threads in flight per row.

Pass 1 keeps a per-thread running maximum and the sum of ``exp(z - m)``
against it, so each element costs one exponential and only a maximum update
pays a rescale of the running sum. The per-thread pairs are then combined
block-wide: maximum first, then the sums rescaled to it.

Traffic on the ``(Bc, V)`` footprint: two fp32 reads and one output write,
against roughly thirty bytes per element for the sequence this replaces.

Numerics follow the unfused path: ``exp(z - m)`` in fp32, one fp32 divide per
row for ``s / l``, and the one-hot term subtracted before the downcast so the
target column rounds once, from ``(p - 1) * s``. The exponentials use the
hardware approximation (``ex2.approx``), where the eager path used ``exp_``.

The logits buffer's dtype is the caller's choice and is part of the compile
key: fp32 is eager parity, and a low-precision buffer halves both reads at the
cost of rounding the softmax input. Either way every element is widened to fp32
on load and all arithmetic here is fp32.

``g`` may share the logits buffer's storage: the same bytes hold fp32 logits
on the way in and (half as many) low-precision gradients on the way out, so a
chunk costs one buffer rather than two, with no value rounding anywhere. Pass 2
is ordered for that case unconditionally -- it costs nothing measurable when the
buffers are distinct, and one mode is one thing to reason about.

Why the ordering is needed: ``g[n, j]`` occupies the bytes of ``z[n, j / 2]``,
so an unsynchronized write would destroy a logit another thread has yet to read.
Pass 2 therefore stages a group of column tiles in registers, synchronizes, and
only then writes them. That is safe for any group size: a group's writes reach
at most halfway into the columns it just read, so they can only land on logits
this block has already consumed, and never on the columns a later group will
read. The loop runs the same number of iterations in every thread --
out-of-range lanes re-read column zero rather than exiting -- because a thread
that left early would hang the others on the barrier.
"""

import operator

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._vendor.quack.reduce import block_reduce


# Defaults for the kernel's two shape knobs, which a caller may override (see
# `fused_grad_logits_into`). Measured on H100 over threads x tiles in
# {256, 512, 1024} x {1, 4, 8, 16, 32} at 20 (Bc, V) chunk shapes and both
# buffer layouts: no setting wins everywhere. A wide block pays off only on
# long rows (V >= 32000); staging peaks around 8 tiles there and degrades past
# it, steeply on short rows. (512, 4) is the middle of both axes -- the worst
# setting at no shape measured, at most 1.39x behind the per-shape best -- and
# the stake is small either way: where the split can be measured cleanly the
# chunked loop spends ~85% of its time in its three cuBLAS GEMMs and under 10%
# here, so the best setting at the loop's own chunk shapes is worth under 2%
# end to end.
#
# Legal ranges, as opposed to preferences: `threads_per_block` must be a
# multiple of 32 and at most 1024, both because a CUDA block stops there and
# because the cross-warp combine reads one warp's worth of per-warp partials, so
# more than 32 warps per row would drop the surplus. `tiles_per_stage` may be
# any positive integer -- the write-ordering argument below holds for every
# group size.
_DEFAULT_THREADS_PER_BLOCK = 512
_DEFAULT_TILES_PER_STAGE = 4

_TORCH_TO_CUTE = {
    torch.float32: Float32,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
}


def _make_kernel(out_dtype, threads_per_block, tiles_per_stage):
    warps_per_block = threads_per_block // 32

    @cute.kernel
    def _kernel(
        mZ: cute.Tensor,
        mS: cute.Tensor,
        mTarget: cute.Tensor,
        mG: cute.Tensor,
        mTerm: cute.Tensor,
        V: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        # One slot per warp for each of the two block-wide combines; separate
        # buffers so the second combine cannot overwrite what a warp still
        # reads from the first.
        buf_max = smem.allocate_tensor(
            Float32, cute.make_layout((1, warps_per_block)), byte_alignment=4
        )
        buf_sum = smem.allocate_tensor(
            Float32, cute.make_layout((1, warps_per_block)), byte_alignment=4
        )

        threads = Int32(threads_per_block)
        target = Int32(mTarget[row])

        m = Float32(-Float32.inf)
        l = Float32(0.0)
        col = tidx
        while col < V:
            z = Float32(mZ[row, col])
            if z > m:
                # Rescale the sum to the new maximum. A thread that has seen
                # nothing yet has m = -inf and l = 0, so this yields l = 1.
                l = l * cute.math.exp(m - z, fastmath=True) + Float32(1.0)
                m = z
            else:
                l = l + cute.math.exp(z - m, fastmath=True)
            col = col + threads

        row_max = block_reduce(
            cute.arch.warp_reduction_max(m),
            cute.arch.fmax,
            buf_max,
            init_val=-Float32.inf,
        )
        l = l * cute.math.exp(m - row_max, fastmath=True)
        row_sum = block_reduce(
            cute.arch.warp_reduction_sum(l),
            operator.add,
            buf_sum,
            init_val=Float32(0.0),
        )

        s = mS[row]
        factor = s / row_sum
        if tidx == 0:
            # The loss needs only this combination of the row's two statistics,
            # so it is formed here: emitting `lse` and the target logit
            # separately cost the caller a subtract, a multiply and a reduction
            # per chunk on (Bc,) data, which is launch-bound at every size.
            lse = row_max + cute.math.log(row_sum, fastmath=True)
            mTerm[row] = s * (lse - Float32(mZ[row, target]))

        # This read of the target logit has to be ordered against the writes
        # below, which may occupy its bytes.
        cute.arch.barrier()
        stage = Int32(tiles_per_stage)
        tiles = (V + threads - Int32(1)) // threads
        groups = (tiles + stage - Int32(1)) // stage
        group = Int32(0)
        while group < groups:
            base = group * stage * threads + tidx
            staged = []
            for j in cutlass.range_constexpr(tiles_per_stage):
                col = base + Int32(j) * threads
                col_read = col
                if col >= V:
                    col_read = Int32(0)
                staged.append(
                    cute.math.exp(Float32(mZ[row, col_read]) - row_max, fastmath=True)
                    * factor
                )
            cute.arch.barrier()
            for j in cutlass.range_constexpr(tiles_per_stage):
                col = base + Int32(j) * threads
                if col < V:
                    value = staged[j]
                    if col == target:
                        value = value - s
                    mG[row, col] = out_dtype(value)
            group = group + Int32(1)

    @cute.jit
    def _launch(
        mZ: cute.Tensor,
        mS: cute.Tensor,
        mTarget: cute.Tensor,
        mG: cute.Tensor,
        mTerm: cute.Tensor,
        stream: cuda.CUstream,
        V: Int32,
        num_rows: Int32,
    ):
        _kernel(mZ, mS, mTarget, mG, mTerm, V).launch(
            grid=[num_rows, 1, 1],
            block=[threads_per_block, 1, 1],
            stream=stream,
        )

    return _launch


@instrumented_cutedsl_cache(
    "torch_nn::_linear_cross_entropy_batch_chunked",
    key_fn=lambda logits_torch_dtype, out_torch_dtype, threads, tiles: (
        f"fused_grad_logits logits={logits_torch_dtype} out={out_torch_dtype}"
        f" threads={threads} tiles={tiles}"
    ),
)
def _compile_fused_grad_logits(
    logits_torch_dtype: torch.dtype,
    out_torch_dtype: torch.dtype,
    threads: int,
    tiles: int,
):
    # V and the row count stay runtime arguments: the kernel loops over the
    # columns and takes the rows from the grid, so one compile per (dtype pair,
    # block width, staging depth) serves every chunk shape.
    launcher = _make_kernel(_TORCH_TO_CUTE[out_torch_dtype], threads, tiles)

    def logits_2d():
        return cute.runtime.make_fake_tensor(
            _TORCH_TO_CUTE[logits_torch_dtype],
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int64(), 1),
        )

    def f32_1d():
        return cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,))

    # A fresh fake tensor per parameter: reusing one object for two parameters
    # makes the traced signature drop arguments, and the host call then shifts
    # its positionals.
    return cute.compile(
        launcher,
        logits_2d(),
        f32_1d(),
        cute.runtime.make_fake_tensor(Int64, (cute.sym_int(),), stride=(1,)),
        cute.runtime.make_fake_tensor(
            _TORCH_TO_CUTE[out_torch_dtype],
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int64(), 1),
        ),
        f32_1d(),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


def fused_grad_logits_into(
    g: torch.Tensor,
    term: torch.Tensor,
    logits: torch.Tensor,
    row_scale: torch.Tensor,
    target: torch.Tensor,
    **meta: int,
) -> None:
    """Writes ``g`` and the per-row loss term from the raw ``logits``.

    ``logits`` is (Bc, V) with unit inner stride, fp32 or a low-precision
    buffer dtype. When ``g`` is a separate buffer the logits are left untouched;
    ``g`` is a distinct (Bc, V) buffer at the input dtype.

    When ``g`` is a view of that storage instead -- typically
    ``logits.view(g.dtype).narrow(1, 0, V)``, so its rows carry the leading
    dimension of the rows they overwrite -- the logits are CONSUMED: after the
    call that memory holds ``g``. The kernel orders its writes against its reads
    either way (see the module docstring), so the caller chooses the layout and
    nothing else changes.

    ``term``, ``row_scale`` are fp32 (Bc,) and ``target`` is int64 (Bc,), all
    contiguous. ``term`` is the row's loss contribution,
    ``row_scale * (lse - z_target)`` -- the only combination of the row
    statistics the loss needs, so neither is emitted separately.

    ``meta`` carries the kernel's shape knobs, so a tuner -- or a caller who has
    measured its own shapes -- can choose them without editing this file:

    ``threads_per_block``
        Block width, default 512. A multiple of 32, at most 1024.
    ``tiles_per_stage``
        Column tiles staged per barrier in pass 2, default 4. At least 1.

    Each combination compiles its own kernel, so a caller sweeping them pays one
    compile per point and hits the cache thereafter.
    """
    threads = meta.pop("threads_per_block", _DEFAULT_THREADS_PER_BLOCK)
    tiles = meta.pop("tiles_per_stage", _DEFAULT_TILES_PER_STAGE)
    if meta:
        raise ValueError(
            f"unknown meta parameters {sorted(meta)}; this kernel takes"
            " threads_per_block and tiles_per_stage"
        )
    if threads % 32 or not 32 <= threads <= 1024:
        raise ValueError(
            f"threads_per_block must be a multiple of 32 in [32, 1024], got {threads}"
        )
    if tiles < 1:
        raise ValueError(f"tiles_per_stage must be at least 1, got {tiles}")
    num_rows, V = logits.shape
    compiled = _compile_fused_grad_logits(logits.dtype, g.dtype, threads, tiles)
    compiled(logits, row_scale, target, g, term, V, num_rows)
