# Ordered sum/prod adapter for canonical (M, N) -> 1-D; cutedsl_impl retains eligibility,
# canonicalisation, keepdim, and out=. 112 hashes pin the reference DAG's bits.
# Measured 1.08-3.00x of that reference on fp32 sum at a fixed 256 MiB footprint.

import cutlass

import torch

from . import kernel_rowtile as rt, traits as T


def _acc(dtype):
    # Accumulator affects bits: fp64 uses fp64; all other supported dtypes use fp32.
    return cutlass.Float64 if dtype is torch.float64 else cutlass.Float32


def _layout_ok(out, src):
    # Require compact input and unit-stride output; otherwise use the same-bit reference.
    # Misalignment selects a bit-neutral unstaged plan rather than falling back.
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
