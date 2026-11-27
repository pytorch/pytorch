"""Device kernels shared by native CuTeDSL operators."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32


@cute.kernel
def reduce_rows(
    partial: cute.Tensor,
    out: cute.Tensor,
    cols: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Sum float32 partial rows into a vector with a static column count.

    cols divides 32; threads is a multiple of 32. Launch with block=[threads, 1, 1]
    and grid=[ceil_div(out.shape[0], cols), 1, 1].
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane, warp = tidx % 32, tidx // 32
    col = bidx * cols + tidx % cols
    acc = Float32(0)
    for row in cutlass.range(tidx // cols, partial.shape[0], threads // cols):
        if cutlass.const_expr(out.shape[0] % cols == 0):
            acc += partial[row, col]
        elif col < out.shape[0]:
            acc += partial[row, col]
    for i in cutlass.range_constexpr((32 // cols).bit_length() - 1):
        acc += cute.arch.shuffle_sync_bfly(acc, offset=(1 << i) * cols)
    smem = cutlass.utils.SmemAllocator()
    sums = smem.allocate_tensor(
        Float32, cute.make_layout((threads // 32, cols), stride=(cols, 1))
    )
    if lane < cols:
        sums[warp, lane] = acc
    cute.arch.barrier()
    if tidx < cols:
        for i in cutlass.range_constexpr(1, threads // 32):
            acc += sums[i, tidx]
        if cutlass.const_expr(out.shape[0] % cols == 0):
            out[bidx * cols + tidx] = out.element_type(acc)
        elif col < out.shape[0]:
            out[col] = out.element_type(acc)
