"""Row statistics and the softmax-gradient transform in one kernel.

Replaces the chunked loop's passes over its ``(Bc, V)`` logits buffer -- row
max, subtract, gather, ``exp_``, row sum, and the scale that turns the softmax
into the gradient -- with one launch that reads the raw logits twice and writes
the gradient-of-logits once. Per row ``n``, with ``m_n = max_v z[n, v]`` and
``l_n = sum_v exp(z[n, v] - m_n)``::

    g[n, v] = exp(z[n, v] - m_n) * (s_n / l_n) - s_n * [v == T_hat_n]
    log_row_sum[n] = log(l_n)
    shifted_target_logit[n] = z[n, T_hat_n] - m_n

The caller forms the loss term as ``s_n * (log_row_sum_n -
shifted_target_logit_n)``. Both statistics are shifted by ``m_n`` because the
unshifted ``m_n + log(l_n)`` and ``z[n, T_hat_n]`` cancel catastrophically in
fp32 at large row offsets: logits of 2**24 turn a loss of log(2) into 0.

One block per row, since a row's max and sum must be complete before any of its
elements is written; parallelism is therefore the chunk's row count, so blocks
are wide by default. Pass 1 is an online softmax (a per-thread running max and
rescaled sum, then a block-wide combine); pass 2 writes ``g``. All arithmetic is
fp32 -- every element is widened on load, whatever the logits buffer's dtype,
which is a compile key. ``exp`` is the hardware ``ex2.approx``, and the one-hot
term is subtracted before the downcast so the target column rounds once.

``g`` may alias the logits storage, so a chunk costs one buffer: ``g[n, j]``
occupies the bytes of ``z[n, j / r]``, with ``r`` gradient elements per logit (2
for an fp32 buffer, 1 at ``g``'s width). At ``r = 2`` a write can land on a
column another thread has not read yet, so pass 2 stages a group of column tiles
in registers, synchronizes, then writes them. A group writes columns
``j / r <= j`` and never reaches the columns a later group reads. At ``r = 1``
the barrier is unnecessary but kept, so there is one code path. Every thread
runs the same number of iterations -- out-of-range lanes re-read column 0 --
because a thread that exited early would hang the others at the barrier.
"""

import operator

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as cute_testing
from cutlass import BFloat16, Float16, Float32, Int32, Int64

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._vendor.quack.reduce import block_reduce


# Defaults for the kernel's two shape knobs, which a caller may override (see
# `fused_grad_logits_into`), chosen from an H100 sweep over chunk shapes the
# loop uses: https://github.com/pytorch/pytorch/pull/195829
_DEFAULT_THREADS_PER_BLOCK = 512
_DEFAULT_TILES_PER_STAGE = 4

_LOG2E = 1.4426950408889634

_TORCH_TO_CUTE = {
    torch.float32: Float32,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
}


def _exp(x):
    """``exp(x)`` as the hardware ``ex2.approx.ftz.f32``. The kernel evaluates
    ``exp(-inf)`` and lets NaN logits propagate to the output, so this must not
    carry ``fastmath``'s ``nnan|ninf`` assumptions, under which both are
    undefined."""
    return cute.math.exp2(x * Float32(_LOG2E), approx=True, ftz=True)


def _make_kernel(out_dtype, threads_per_block, tiles_per_stage):
    warps_per_block = threads_per_block // 32

    @cute.kernel
    def _kernel(
        mZ: cute.Tensor,
        mS: cute.Tensor,
        mTarget: cute.Tensor,
        mG: cute.Tensor,
        mLogRowSum: cute.Tensor,
        mShiftedTargetLogit: cute.Tensor,
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
        # Traps on an out-of-range target, like eager's index assert (live under
        # `--enable-assertions`, below). Checked at int64, since narrowing first
        # would read 2**32 as 0; a target that passes is below V, which
        # `_kernel_eligible` keeps within int32, so narrowing it is exact.
        target = Int64(mTarget[row])
        where = "linear_cross_entropy: target"
        cute_testing.assert_(target >= Int64(0), where + " < 0")
        cute_testing.assert_(target < Int64(V), where + " >= num_classes")
        target_read = Int32(target)

        m = Float32(-Float32.inf)
        l = Float32(0.0)
        col = tidx
        while col < V:
            z = Float32(mZ[row, col])
            if z > m:
                # Rescale the sum to the new maximum. A thread that has seen
                # nothing yet has m = -inf and l = 0, so this yields l = 1.
                l = l * _exp(m - z) + Float32(1.0)
                m = z
            elif z != Float32(-Float32.inf):
                # Skip -inf: it contributes 0, but on a thread that has seen
                # nothing yet m is -inf too and exp(z - m) would be NaN. The
                # rescale below turns the skipped l = 0 into the same 0. A NaN
                # logit still takes this branch and poisons the row, as in eager.
                l = l + _exp(z - m)
            col = col + threads

        row_max = block_reduce(
            cute.arch.warp_reduction_max(m),
            cute.arch.fmax,
            buf_max,
            init_val=-Float32.inf,
        )
        l = l * _exp(m - row_max)
        row_sum = block_reduce(
            cute.arch.warp_reduction_sum(l),
            operator.add,
            buf_sum,
            init_val=Float32(0.0),
        )

        s = mS[row]
        target_logit = Float32(mZ[row, target_read])
        factor = s / row_sum
        if tidx == 0:
            # Shifted, both of them -- see the module docstring for why the
            # unshifted pair cannot be subtracted in fp32.
            mLogRowSum[row] = cute.math.log(row_sum)
            mShiftedTargetLogit[row] = target_logit - row_max

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
                staged.append(_exp(Float32(mZ[row, col_read]) - row_max) * factor)
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
        mLogRowSum: cute.Tensor,
        mShiftedTargetLogit: cute.Tensor,
        stream: cuda.CUstream,
        V: Int32,
        num_rows: Int32,
    ):
        _kernel(mZ, mS, mTarget, mG, mLogRowSum, mShiftedTargetLogit, V).launch(
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

    # Each parameter needs its own fake tensor: the tracer collapses a repeated
    # object into one argument, and the host call's positionals then shift.
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
        f32_1d(),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi --enable-assertions",
    )


def fused_grad_logits_into(
    g: torch.Tensor,
    log_row_sum: torch.Tensor,
    shifted_target_logit: torch.Tensor,
    logits: torch.Tensor,
    row_scale: torch.Tensor,
    target: torch.Tensor,
    **meta: int,
) -> None:
    """Writes ``g`` and the row statistics from the raw ``logits``.

    ``logits`` is (Bc, V) with unit inner stride, fp32 or a low-precision dtype.
    ``g`` is (Bc, V) at the input dtype: either a separate buffer, which leaves
    the logits untouched, or a view of their storage -- typically
    ``logits.view(g.dtype).narrow(1, 0, V)``, whose rows overlay the logit rows
    they overwrite -- which consumes them: after the call that memory holds
    ``g``. The kernel is correct for either layout (see the module docstring).

    ``log_row_sum``, ``shifted_target_logit`` and ``row_scale`` are fp32 (Bc,),
    ``target`` is int64 (Bc,), all contiguous.

    ``meta`` takes the kernel's shape knobs; each combination compiles once:

    ``threads_per_block``
        Block width, default 512. A multiple of 32, at most 1024.
    ``tiles_per_stage``
        Column tiles staged per barrier in pass 2, default 4. At least 1.
    """
    threads = meta.pop("threads_per_block", _DEFAULT_THREADS_PER_BLOCK)
    tiles = meta.pop("tiles_per_stage", _DEFAULT_TILES_PER_STAGE)
    if meta:
        raise ValueError(
            f"unknown meta parameters {sorted(meta)}; this kernel takes"
            " threads_per_block and tiles_per_stage"
        )
    # The block-wide combine reduces one partial per warp within a single warp,
    # so a block holds at most 32 warps.
    if threads % 32:
        raise ValueError(f"threads_per_block must be a multiple of 32, got {threads}")
    if not 32 <= threads <= 1024:
        raise ValueError(f"threads_per_block must be in [32, 1024], got {threads}")
    if tiles < 1:
        raise ValueError(f"tiles_per_stage must be at least 1, got {tiles}")
    num_rows, V = logits.shape
    # A launch on the NULL stream, PyTorch's default, goes to the current device
    # rather than the tensors', so make theirs current. The compiled kernel is
    # context-independent, so one compile serves every device.
    with torch.accelerator.device_index(logits.device.index):
        compiled = _compile_fused_grad_logits(logits.dtype, g.dtype, threads, tiles)
        compiled(
            logits, row_scale, target, g, log_row_sum, shifted_target_logit, V, num_rows
        )
