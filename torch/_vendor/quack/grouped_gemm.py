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
    launch_gemm,
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
        pingpong=True,
        is_dynamic_persistent=True,
        device_capacity=12,
        use_tma_gather=False,
    )


@instrument_cutedsl_compile("aten::_grouped_mm")
@jit_cache
def _compile_sm120_grouped_mm_varlen_m(
    a_dtype,
    b_dtype,
    d_dtype,
    a_major,
    b_major,
    d_major,
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
        varlen_m=True,
        varlen_k=False,
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
    varlen_args = make_fake_varlen_args(True, False, False, None)
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
    is_sm100_family: bool
    epi_static: object
    scheduler_static: object | None
    max_active_clusters: int
    max_swizzle_size: int
    scheduler_uses_semaphore: bool


_plan_cache: dict[tuple, _GroupedGemmPlan] = {}


def _plan(A: Tensor, B_lower: Tensor, out: Tensor) -> _GroupedGemmPlan:
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
        device_capacity,
        config,
    )
    cached = _plan_cache.get(key)
    if cached is not None:
        return cached

    cluster_shape_mnk = (config.cluster_m, config.cluster_n, config.cluster_k)
    compiled_fn = _compile_sm120_grouped_mm_varlen_m(
        a_dtype,
        b_dtype,
        d_dtype,
        a_major,
        b_major,
        d_major,
        (config.tile_m, config.tile_n),
        cluster_shape_mnk,
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
        is_sm100_family=False,
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


def grouped_mm_sm120_varlen_m(A: Tensor, B: Tensor, cu_seqlens_m: Tensor) -> Tensor:
    """Run grouped GEMM for A[cu_seqlens_m[i]:cu_seqlens_m[i+1]] @ B[i]."""
    B_lower = B.permute(2, 1, 0)
    out = torch.empty((A.shape[0], B.shape[-1]), dtype=A.dtype, device=A.device)
    plan = _plan(A, B_lower, out)
    launch_gemm(
        plan,
        A,
        B_lower,
        out,
        None,
        plan.epi_static,
        plan.scheduler_static,
        make_varlen_args(cu_seqlens_m, None, None),
    )
    return out
