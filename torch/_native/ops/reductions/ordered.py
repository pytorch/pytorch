# The ORDERED sum/prod kernels, in the shape the inner-tree override expects:
# `into(out_1d, in_2d)`, writing rows of a canonical (M, N) input into a 1-D output.
#
# Eligibility, the (M, N) canonicalisation, keepdim and the out= variants are NOT redone here:
# cutedsl_impl owns them and this module only chooses the kernel. That is deliberate -- reusing
# that logic makes this a drop-in kernel swap, so exactly the same calls are served, with the
# same bit pattern, and nothing about the override's contract moves.
#
# The bit pattern is the point, so it is asserted, not assumed: the order reproduces the
# reference kernel's add DAG exactly (test_inner_tree_order pins 112 golden hashes across 14
# shapes x 4 dtypes x sum/prod, and the reference's own suite runs unchanged against this).
#
# MEASURED on B200, fp32 sum over the last dim at a fixed 256 MiB footprint (device us):
#
#   shape             reference   this   speedup
#   (4194304,    16)      127.6   42.6     3.00x
#   ( 524288,   128)       73.6   68.3     1.08x
#   (  65536,  1024)       91.1   41.0     2.22x
#   (  32768,  2048)       97.4   38.4     2.54x
#   (  16384,  4096)       72.9   39.1     1.86x
#   (   8192,  8192)       64.3   39.3     1.64x
#   (    671,100000)       65.4   43.1     1.52x
#   (    256,262144)       67.7   40.8     1.66x

import cutlass

import torch

from .._cutedsl import traits as T
from . import kernel_rowtile as rt


def _acc(dtype):
    # The accumulator is PART OF THE BIT PATTERN, not a performance knob: fp64 accumulates in
    # fp64 and every other supported dtype in fp32, matching the reference kernel.
    return cutlass.Float64 if dtype is torch.float64 else cutlass.Float32


def _layout_ok(out, src):
    # This order's wrap declares a COMPACT (M, N) input and a unit-stride output. A gapped outer
    # stride (a sliced input) or a strided output (a keepdim out= view) is still eligible for the
    # override, so those keep the reference kernel: same order, same bits, no coverage change.
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
