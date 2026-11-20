# Ordered sum/prod adapter for canonical (M, N) -> 1-D; cutedsl_impl retains eligibility,
# canonicalisation, keepdim, and out=. Pinned hashes preserve the reference DAG's bits.

from typing import Any

import cutlass

import torch

from . import kernel_rowtile as rt, traits as T


def _acc(dtype: torch.dtype) -> Any:
    # Accumulator affects bits: fp64 uses fp64; all other supported dtypes use fp32.
    return cutlass.Float64 if dtype is torch.float64 else cutlass.Float32


def sum_into(out: torch.Tensor, src: torch.Tensor) -> None:
    if not rt.reduce_row_itree(T.SumOps(acc=_acc(src.dtype)), "itree_sum", src, out):
        raise AssertionError("eligible sum reduction has no inner-tree plan")


def prod_into(out: torch.Tensor, src: torch.Tensor) -> None:
    if not rt.reduce_row_itree(T.ProdOps(acc=_acc(src.dtype)), "itree_prod", src, out):
        raise AssertionError("eligible prod reduction has no inner-tree plan")
