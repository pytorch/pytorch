# The ORDERED sum/prod kernels in the shape the inner-tree override expects, writing rows of a
# canonical (M, N) input into a 1-D output. Eligibility, canonicalisation, keepdim and out=
# stay in cutedsl_impl, which makes this a drop-in kernel swap. The bit pattern is asserted
# rather than assumed: test_inner_tree_order pins 112 golden hashes against the reference DAG.
# Measured 1.08-3.00x of that reference on fp32 sum at a fixed 256 MiB footprint.

import cutlass

import torch

from .._cutedsl import traits as T
from . import kernel_rowtile as rt


def _acc(dtype):
    # The accumulator is PART OF THE BIT PATTERN, not a performance knob: fp64 accumulates in
    # fp64 and every other supported dtype in fp32, matching the reference kernel.
    return cutlass.Float64 if dtype is torch.float64 else cutlass.Float32


def _layout_ok(out, src):
    # This order's wrap declares a COMPACT input and a unit-stride output, so a gapped outer stride
    # or a strided output keeps the reference kernel -- same order, same bits. ALIGNMENT is not
    # gated: a compact input at a non-zero storage offset drops to the unstaged form of the same
    # plan, which is bit-neutral, rather than falling back.
    return src.stride(0) == src.shape[1] and out.stride(0) == 1


def sum_into(out: torch.Tensor, src: torch.Tensor) -> None:
    if _layout_ok(out, src) and rt.reduce_row_itree(
        T.SumOps(acc=_acc(src.dtype)), "itree_sum", src, out
    ):
        return
    from .inner_tree_kernel import inner_tree_sum_into

    inner_tree_sum_into(out, src)


def prod_into(out: torch.Tensor, src: torch.Tensor) -> None:
    if _layout_ok(out, src) and rt.reduce_row_itree(
        T.ProdOps(acc=_acc(src.dtype)), "itree_prod", src, out
    ):
        return
    from .inner_tree_kernel import inner_tree_prod_into

    inner_tree_prod_into(out, src)
