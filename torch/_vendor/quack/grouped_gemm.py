# Copyright (c) 2025, Tri Dao.
# mypy: allow-untyped-defs
"""Small QuACK GEMM facade used by torch._native grouped GEMM overrides."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from torch._native.instrumentation import instrument_cutedsl_compile

from .cache import jit_cache
from .cute_dsl_utils import get_device_capacity, get_max_active_clusters
from .gemm_config import GemmConfig
from .gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm120
from .gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    get_dtypes,
    get_majors,
    make_fake_gemm_tensors,
    make_fake_varlen_args,
    make_fake_scheduler_args,
    make_scheduler_args,
    make_varlen_args,
)


def _sm120_default_config() -> GemmConfig:
    return GemmConfig(
        tile_m=128,
        tile_n=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=True,
        device_capacity=12,
        use_tma_gather=False,
    )


@instrument_cutedsl_compile("aten::_grouped_mm")
@jit_cache
def _compile_sm120_grouped_mm_varlen(
    a_dtype,
    b_dtype,
    d_dtype,
    a_major,
    b_major,
    d_major,
    varlen_m,
    varlen_k,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    device_capacity,
):
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        None,
        a_major,
        b_major,
        d_major,
        None,
        varlen_m=varlen_m,
        varlen_k=varlen_k,
        gather_A=False,
    )
    epi_args = GemmDefaultEpiMixin.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=False,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_fake_scheduler_args(
        is_dynamic_persistent and device_capacity[0] <= 9,
        False,
        l,
    )
    varlen_args = make_fake_varlen_args(varlen_m, varlen_k, False, None)
    return compile_gemm_kernel(
        GemmDefaultSm120,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        False,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
    )


class _GroupedGemmPlan(NamedTuple):
    compiled_fn: object
    epi_static: object
    scheduler_static: object | None
    max_active_clusters: int
    max_swizzle_size: int
    scheduler_uses_semaphore: bool


_plan_cache: dict[tuple, _GroupedGemmPlan] = {}


def _empty_grouped_mm_output(
    out_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    alignment = 16 // dtype.itemsize
    size_padded = (out_shape[-1] + alignment - 1) // alignment * alignment
    if len(out_shape) == 2:
        out_stride = (size_padded, 1)
    else:
        out_stride = (out_shape[1] * size_padded, size_padded, 1)
    return torch.empty_strided(out_shape, out_stride, dtype=dtype, device=device)


def _plan(
    A: Tensor,
    B_lower: Tensor,
    out: Tensor,
    *,
    varlen_m: bool,
    varlen_k: bool,
) -> _GroupedGemmPlan:
    config = _sm120_default_config()
    device_capacity = get_device_capacity(A.device)
    a_major, b_major, d_major, _ = get_majors(A, B_lower, out, None)
    a_dtype, b_dtype, d_dtype, _ = get_dtypes(A, B_lower, out, None)
    key = (
        A.device,
        A.dtype,
        B_lower.dtype,
        out.dtype,
        a_major,
        b_major,
        d_major,
        varlen_m,
        varlen_k,
        device_capacity,
    )
    cached = _plan_cache.get(key)
    if cached is not None:
        return cached

    compiled_fn = _compile_sm120_grouped_mm_varlen(
        a_dtype,
        b_dtype,
        d_dtype,
        a_major,
        b_major,
        d_major,
        varlen_m,
        varlen_k,
        (config.tile_m, config.tile_n),
        (config.cluster_m, config.cluster_n, config.cluster_k),
        config.pingpong,
        True,
        config.is_dynamic_persistent,
        device_capacity,
    )
    max_active_clusters = get_max_active_clusters(
        config.cluster_m * config.cluster_n * config.cluster_k,
        device_capacity=device_capacity,
    )
    plan = _GroupedGemmPlan(
        compiled_fn=compiled_fn,
        epi_static=GemmDefaultEpiMixin.EpilogueArguments(
            alpha=None,
            beta=None,
            mRowVecBroadcast=None,
            mColVecBroadcast=None,
            add_to_output=None,
            rounding_mode=None,
            sr_seed=None,
        ),
        scheduler_static=make_scheduler_args(
            max_active_clusters,
            config.max_swizzle_size,
            None,
            None,
        ),
        max_active_clusters=max_active_clusters,
        max_swizzle_size=config.max_swizzle_size,
        scheduler_uses_semaphore=False,
    )
    _plan_cache[key] = plan
    return plan


def grouped_mm_sm12x(A: Tensor, B: Tensor, cu_seqlens: Tensor) -> Tensor:
    B_lower = B.transpose(-1, 0)
    varlen_k = B.dim() == 2
    varlen_m = B.dim() == 3

    if varlen_k:
        out_shape = (cu_seqlens.numel() - 1, A.shape[0], B.shape[-1])
        varlen_args = make_varlen_args(None, cu_seqlens, None)
    else:
        out_shape = (A.shape[0], B.shape[-1])
        varlen_args = make_varlen_args(cu_seqlens, None, None)

    out = _empty_grouped_mm_output(out_shape, A.dtype, A.device)
    out_lower = out.permute(1, 2, 0) if varlen_k else out
    plan = _plan(A, B_lower, out_lower, varlen_m=varlen_m, varlen_k=varlen_k)
    plan.compiled_fn(
        A,
        B_lower,
        out_lower,
        None,
        plan.epi_static,
        plan.scheduler_static,
        varlen_args,
        None,
    )
    return out
