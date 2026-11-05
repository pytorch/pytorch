# mypy: allow-untyped-decorators
"""Triton finalizer for the partial sums of a mix-order reduction.

A mix-order reduction kernel leaves each of its column accumulators as an
``[nsplit, d]`` fp32 block of partial sums, stored back to back in one
workspace. ``colsum_finalize`` reduces up to ``MAX_OUTPUTS`` consecutive blocks
in one launch and casts each result to its output's dtype on store. Generated
wrapper code calls it when ``config.triton.mix_order_reduction_triton_finalize``
is set.
"""

import torch

from .triton_compat import tl, triton


# Number of output pointers of the kernel, i.e. accumulators per launch.
MAX_OUTPUTS = 4
# The kernel does not split over rows: longer workspaces reduce faster in ATen.
MAX_NSPLIT = 8192


@triton.jit
def _colsum_finalize_kernel(
    ws_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    nsplit,
    d,
    NUM_OUTPUTS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    k = tl.program_id(1)
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < d
    base = ws_ptr + k.to(tl.int64) * nsplit * d
    acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)
    for s0 in range(0, nsplit, BLOCK_S):
        rows = (s0 + tl.arange(0, BLOCK_S)).to(tl.int64)
        mask = (rows[:, None] < nsplit) & col_mask[None, :]
        acc += tl.load(base + rows[:, None] * d + cols[None, :], mask=mask, other=0.0)
    result = tl.sum(acc, axis=0)
    if k == 0:
        tl.store(out0_ptr + cols, result.to(out0_ptr.dtype.element_ty), mask=col_mask)
    if NUM_OUTPUTS > 1:
        if k == 1:
            tl.store(
                out1_ptr + cols, result.to(out1_ptr.dtype.element_ty), mask=col_mask
            )
    if NUM_OUTPUTS > 2:
        if k == 2:
            tl.store(
                out2_ptr + cols, result.to(out2_ptr.dtype.element_ty), mask=col_mask
            )
    if NUM_OUTPUTS > 3:
        if k == 3:
            tl.store(
                out3_ptr + cols, result.to(out3_ptr.dtype.element_ty), mask=col_mask
            )


def colsum_finalize(
    ws: torch.Tensor, nsplit: int, d: int, outs: list[torch.Tensor]
) -> None:
    """``outs[k] = ws[k * nsplit * d : (k + 1) * nsplit * d].view(nsplit, d).sum(0)``.

    ``outs`` are 1 to ``MAX_OUTPUTS`` contiguous tensors of ``d`` elements each.
    """
    num_outputs = len(outs)
    if not 1 <= num_outputs <= MAX_OUTPUTS:
        raise ValueError(f"expected 1 to {MAX_OUTPUTS} outputs, got {num_outputs}")
    if nsplit > MAX_NSPLIT:
        for k, out in enumerate(outs):
            block = ws[k * nsplit * d : (k + 1) * nsplit * d].view(nsplit, d)
            out.copy_(block.sum(dim=0).view(out.shape))
        return
    # Tuned on B200 at nsplit 503: one 512-row step per program, and column
    # blocks narrow enough that even d = 64 spreads over several SMs.
    block_s, block_d = (256, 32) if d * num_outputs >= 8192 else (512, 16)
    padded = outs + [outs[0]] * (MAX_OUTPUTS - num_outputs)
    _colsum_finalize_kernel[(triton.cdiv(d, block_d), num_outputs)](
        ws,
        *padded,
        nsplit,
        d,
        NUM_OUTPUTS=num_outputs,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        num_warps=4,
    )
