# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM100 RMSNorm kernels for Oink's CuteDSL Blackwell path."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

# Vendored/adapted from Quack's SM100 RMSNorm with Oink-specific B200 tuning.

from ._cutedsl_cache import ensure_versioned_cutedsl_cache_dir

ensure_versioned_cutedsl_cache_dir()

try:
    import cutlass  # type: ignore  # noqa: F401
except Exception as e:
    raise ImportError(
        "kernelagent_oink.blackwell.rmsnorm requires CuTeDSL's Python package "
        "(`cutlass`, typically provided by `nvidia-cutlass-dsl`)."
    ) from e

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

import cuda.bindings.driver as cuda  # provided by NVIDIA cuda-python  # noqa: E402

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass import Float32, Int32, const_expr  # noqa: E402
from cutlass.cute import runtime as rt  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

# Shared fast-launch helpers.
from .fast_launch import (  # noqa: E402
    GenericFastLaunch as _GenericFastLaunch,
    StableF32Arg as _StableF32Arg,
    StableI32Arg as _StableI32Arg,
    _env_flag,
    build_fast_launcher as _build_generic_fast_launcher,
)
from ._rmsnorm_smallm_cuda import (  # noqa: E402
    try_rmsnorm_smallm_noweight_cuda,
)
from ._rmsnorm_simple_weightonly import (  # noqa: E402
    try_simple_weightonly_rmsnorm_forward,
)
from . import lite_quack as qutils  # noqa: E402
from .lite_quack import (  # noqa: E402
    _KERNEL_ACCEPTS_LAYOUT_ARGS,
    TORCH2CUTE_DTYPE,
    RMSNormBackward as BaseRMSNormBackward,
    convert_from_dlpack as convert_from_dlpack_cute,
    get_sm_count,
    row_reduce,
)

# Simple compile cache declared early so direct execution works
_PTR_COMPILE_CACHE = {}


# Cached fp32 ones row for GEMM-based partial reductions.
_DW_REDUCE_ONES_CACHE: dict[tuple[int, int], Tensor] = {}


def _get_dw_reduce_ones(device_index: int, sm_count: int) -> Tensor:
    key = (int(device_index), int(sm_count))
    ones = _DW_REDUCE_ONES_CACHE.get(key)
    if ones is None or ones.shape != (1, sm_count) or ones.device.index != device_index:
        ones = torch.ones(
            (1, sm_count),
            device=torch.device("cuda", device_index),
            dtype=torch.float32,
        )
        _DW_REDUCE_ONES_CACHE[key] = ones
    return ones


def _reduce_partial_sum_fp32(partial: Tensor, *, device_index: int) -> Tensor:
    """Reduce a (sm_count, N) fp32 partial buffer into an (N,) fp32 result."""
    assert partial.dtype is torch.float32
    assert partial.dim() == 2
    ones = _get_dw_reduce_ones(device_index, int(partial.shape[0]))
    return torch.mm(ones, partial).squeeze(0)


# Fused-add schedule knobs; set env vars before importing to override.
_DIRECT_GMEM_POLICY = (
    os.environ.get("OINK_RMSNORM_DIRECT_GMEM", "auto").strip().lower() or "auto"
)
_COPY_BITS_POLICY = (
    os.environ.get("OINK_RMSNORM_COPY_BITS", "auto").strip().lower() or "auto"
)
_ENABLE_STAGE2 = _env_flag("OINK_RMSNORM_ENABLE_STAGE2", default=False)

# Forward dispatch control.
_FORCE_RMSNORM_STAGE2_FWD = _env_flag(
    "KERNELAGENT_OINK_FORCE_RMSNORM_STAGE2", default=False
)


def _direct_gmem_from_policy(*, default: bool) -> bool:
    """Resolve the direct-GMEM schedule flag from the (import-time) policy string."""
    if _DIRECT_GMEM_POLICY in {"0", "false", "no", "off"}:
        return False
    if _DIRECT_GMEM_POLICY in {"1", "true", "yes", "on"}:
        return True
    return default


def _copy_bits_from_policy(*, default: int, can_use_256: bool) -> int:
    """Resolve copy width (in bits) from the (import-time) policy string."""
    if _COPY_BITS_POLICY == "64":
        return 64
    if _COPY_BITS_POLICY == "128":
        return 128
    if _COPY_BITS_POLICY == "256" and can_use_256:
        return 256
    return default


def _weight_assumed_align(
    *,
    default: int,
    weight: Tensor | None,
    weight_dtype: type[cutlass.Numeric] | None,
) -> int:
    if (
        weight is not None
        and weight_dtype is not None
        and weight_dtype.width == 32
        and (weight.data_ptr() % 32) == 0
    ):
        return 32
    return int(default)


@dataclass(frozen=True)
class _ForwardLaunchConfig:
    direct_gmem: bool
    use_async: bool
    copy_bits: int
    assumed_align: int
    weight_assumed_align: int
    stage: int
    tpr_override: int | None
    nt_override: int | None
    cluster_n_override: int | None


def _resolve_forward_launch_config(
    *,
    M: int,
    N: int,
    dtype: type[cutlass.Numeric],
    x: Tensor,
    weight: Tensor | None,
    weight_dtype: type[cutlass.Numeric] | None,
    aligned_tensors: tuple[Tensor | None, ...],
) -> _ForwardLaunchConfig:
    direct_gmem_default = bool(
        dtype.width == 16 and N in {128, 512, 4096, 6144, 7168, 8192}
    )
    if (
        dtype.width == 16
        and N == 1536
        and weight_dtype is not None
        and weight_dtype.width == 16
    ):
        direct_gmem_default = True
    if weight_dtype is not None and weight_dtype.width == 32 and N == 7168:
        direct_gmem_default = False
    direct_gmem = _direct_gmem_from_policy(default=direct_gmem_default)
    use_async = not direct_gmem

    can_use_256 = bool(
        dtype.width == 16
        and (weight_dtype is None or weight_dtype.width == 16)
        and (x.data_ptr() % 32) == 0
        and all(t is None or (t.data_ptr() % 32) == 0 for t in aligned_tensors)
    )
    default_copy_bits = 256 if can_use_256 else 128
    if dtype.width == 16 and N == 128:
        default_copy_bits = 128
    if dtype.width == 16 and N in {512, 1536, 4096}:
        default_copy_bits = 128
    if dtype.width == 16 and weight_dtype is not None and weight_dtype.width == 32:
        default_copy_bits = 128 if N == 4096 else 64

    copy_bits = _copy_bits_from_policy(
        default=default_copy_bits, can_use_256=can_use_256
    )
    # cp.async supports at most 128 bits per instruction.  The copy atom clamps
    # async copies to 128b, so keep the TV layout's vector width in sync with the
    # emitted copy width; otherwise shapes such as DSv4 N=1536 can leave half of
    # each logical vector tile uninitialized.
    if use_async and copy_bits > 128:
        copy_bits = 128
    if use_async and copy_bits < 128:
        use_async = False

    assumed_align = 32 if copy_bits >= 256 else 16
    weight_assumed_align = _weight_assumed_align(
        default=assumed_align,
        weight=weight,
        weight_dtype=weight_dtype,
    )
    tpr_override, nt_override, cluster_n_override = _forward_launch_overrides(
        M=int(M),
        N=int(N),
        dtype=dtype,
        weight_dtype=weight_dtype,
        direct_gmem=bool(direct_gmem),
    )

    stage = 1
    if (
        _ENABLE_STAGE2
        and dtype.width == 16
        and N == 7168
        and (not direct_gmem)
        and M >= 4096
    ):
        stage = 2

    return _ForwardLaunchConfig(
        direct_gmem=bool(direct_gmem),
        use_async=bool(use_async),
        copy_bits=int(copy_bits),
        assumed_align=int(assumed_align),
        weight_assumed_align=int(weight_assumed_align),
        stage=int(stage),
        tpr_override=tpr_override,
        nt_override=nt_override,
        cluster_n_override=cluster_n_override,
    )


def _should_force_stage2_forward(
    *,
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    residual: Tensor | None,
    store_rstd: bool,
) -> bool:
    if bias is not None or residual is not None or store_rstd:
        return False
    if weight is None or x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if weight.dtype not in (x.dtype, torch.float32):
        return False

    M, N = int(x.shape[0]), int(x.shape[1])

    if weight.dtype is torch.float32:
        if N == 7168 and M >= 16384:
            return True
        if N in {6144, 8192} and M >= 65536:
            return True
        return False
    return False


def _forward_launch_overrides(
    *,
    M: int,
    N: int,
    dtype: type[cutlass.Numeric],
    weight_dtype: type[cutlass.Numeric] | None,
    direct_gmem: bool,
) -> tuple[int | None, int | None, int | None]:
    """Return optional `(threads_per_row, num_threads, cluster_n)` overrides."""
    tpr_default: int | None = None
    nt_default: int | None = None
    cluster_n_default: int | None = None

    if (
        dtype.width == 16
        and weight_dtype is not None
        and weight_dtype.width == 16
        and N == 1536
        and direct_gmem
        and M >= 4096
    ):
        tpr_default = 32
        nt_default = 32
    if (
        dtype.width == 16
        and weight_dtype is not None
        and weight_dtype.width == 32
        and N == 7168
        and (not direct_gmem)
        and M < 16384
    ):
        tpr_default = 128
        nt_default = 128

    def _env_int(name: str) -> int | None:
        val = os.environ.get(name, "").strip()
        return int(val) if val else None

    tpr = _env_int("OINK_RMSNORM_TPR")
    nt = _env_int("OINK_RMSNORM_NT")
    cluster_n = _env_int("OINK_RMSNORM_CLUSTER_N")
    return (
        tpr_default if tpr is None else tpr,
        nt_default if nt is None else nt,
        cluster_n_default if cluster_n is None else cluster_n,
    )


def _force_stage2_forward_launch_config(
    launch_cfg: _ForwardLaunchConfig,
    *,
    M: int,
    N: int,
    dtype: type[cutlass.Numeric],
    weight: Tensor | None,
    weight_dtype: type[cutlass.Numeric] | None,
) -> _ForwardLaunchConfig:
    if (
        dtype.width != 16
        or weight is None
        or weight_dtype is None
        or weight_dtype.width != 32
    ):
        return launch_cfg
    if N == 7168 and M >= 16384:
        stage = 1
        tpr_override, nt_override = 128, 128
    elif M >= 65536 and N in {6144, 8192}:
        stage = 2
        tpr_override, nt_override = 128, (128 if N == 6144 else 256)
    else:
        return launch_cfg
    return replace(
        launch_cfg,
        direct_gmem=False,
        use_async=True,
        copy_bits=128,
        assumed_align=16,
        weight_assumed_align=_weight_assumed_align(
            default=16,
            weight=weight,
            weight_dtype=weight_dtype,
        ),
        stage=stage,
        tpr_override=tpr_override,
        nt_override=nt_override,
        cluster_n_override=None,
    )


def _override_simple_weight_only_forward_launch_config(
    launch_cfg: _ForwardLaunchConfig,
    *,
    M: int,
    N: int,
    dtype: type[cutlass.Numeric],
    weight: Tensor | None,
    weight_dtype: type[cutlass.Numeric] | None,
) -> _ForwardLaunchConfig:
    if (
        _DIRECT_GMEM_POLICY != "auto"
        or _COPY_BITS_POLICY != "auto"
        or os.environ.get("OINK_RMSNORM_TPR", "").strip()
        or os.environ.get("OINK_RMSNORM_NT", "").strip()
        or os.environ.get("OINK_RMSNORM_CLUSTER_N", "").strip()
        or _ENABLE_STAGE2
    ):
        return launch_cfg
    if (
        dtype.width != 16
        or weight is None
        or weight_dtype is None
        or weight_dtype.width != 16
        or M < 4096
    ):
        return launch_cfg

    if N == 7168:
        if M >= 262144:
            return launch_cfg
        stage = 2
        if M == 131072:
            tpr_override, nt_override = 64, 64
        elif M >= 65536:
            tpr_override, nt_override = 128, 128
        else:
            tpr_override, nt_override = None, None
    elif N == 8192 and M >= 65536:
        stage = 1
        tpr_override, nt_override = 128, 128
    else:
        return launch_cfg

    return replace(
        launch_cfg,
        direct_gmem=False,
        use_async=True,
        copy_bits=128,
        assumed_align=16,
        weight_assumed_align=_weight_assumed_align(
            default=16,
            weight=weight,
            weight_dtype=weight_dtype,
        ),
        stage=stage,
        tpr_override=tpr_override,
        nt_override=nt_override,
        cluster_n_override=None,
    )


def _make_gmem_ptr(
    dtype: type[cutlass.Numeric],
    device_ptr: int,
    *,
    assumed_align: int,
):
    return rt.make_ptr(
        dtype,
        int(device_ptr),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=int(assumed_align),
    )


def _make_optional_gmem_ptr(
    tensor: Tensor | None,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int,
):
    return (
        None
        if tensor is None
        else _make_gmem_ptr(dtype, tensor.data_ptr(), assumed_align=assumed_align)
    )


def _apply_launch_overrides(
    op: object,
    *,
    tpr_override: int | None,
    nt_override: int | None,
    cluster_n_override: int | None,
) -> None:
    if tpr_override is not None:
        op._tpr_override = tpr_override  # type: ignore[attr-defined]
    if nt_override is not None:
        op._nt_override = nt_override  # type: ignore[attr-defined]
    if cluster_n_override is not None:
        op._cluster_n_override = cluster_n_override  # type: ignore[attr-defined]


def _specialized_bf16_threads(N: int, copy_bits: int) -> tuple[int, int] | None:
    if N == 128:
        return 16, 128
    if N == 1536:
        return 96, 96
    if N == 7168:
        return 224, 224
    if N == 6144:
        threads = 192 if copy_bits >= 256 else 256
        return threads, threads
    if N == 8192:
        return 256, 256
    return None


def _current_stream_handle(device_index: int) -> int:
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    return int(torch.cuda.current_stream().cuda_stream)


def _make_rmsnorm_op(
    N: int,
    dtype: type[cutlass.Numeric],
    *,
    stage: int,
    copy_bits: int,
    use_async: bool,
    direct_gmem: bool,
    tpr_override: int | None = None,
    nt_override: int | None = None,
    cluster_n_override: int | None = None,
):
    op = RMSNormSM100(
        N,
        dtype,
        stage=stage,
        copy_bits=copy_bits,
        use_async=use_async,
        direct_gmem=direct_gmem,
    )
    _apply_launch_overrides(
        op,
        tpr_override=tpr_override,
        nt_override=nt_override,
        cluster_n_override=cluster_n_override,
    )
    return op


def _match_stride(tensor: Tensor | None, ref: Tensor | None) -> Tensor | None:
    if tensor is None or ref is None or tensor.stride() == ref.stride():
        return tensor
    out = torch.empty_strided(
        ref.shape, ref.stride(), device=ref.device, dtype=ref.dtype
    )
    out.copy_(tensor)
    return out


def _rmsnorm_ref_outputs(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    residual: Tensor | None,
    eps: float,
    store_rstd: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    y = _match_stride(rmsnorm_ref(x, weight, bias, residual, eps), x)
    rstd = None
    if store_rstd:
        xf = x.float()
        if residual is not None:
            xf = xf + residual.float()
        rstd = torch.rsqrt(xf.square().mean(dim=-1) + eps).to(torch.float32)
    residual_out = None
    if residual is not None:
        residual_out = _match_stride(
            (x.float() + residual.float()).to(x.dtype), residual
        )
    return y, rstd, residual_out


def _make_forward_ptrs(
    *,
    x: Tensor,
    out: Tensor,
    dtype: type[cutlass.Numeric],
    launch_cfg: _ForwardLaunchConfig,
    weight: Tensor | None = None,
    weight_dtype: type[cutlass.Numeric] | None = None,
    bias: Tensor | None = None,
    residual: Tensor | None = None,
    residual_out: Tensor | None = None,
    rstd: Tensor | None = None,
):
    return (
        _make_gmem_ptr(dtype, x.data_ptr(), assumed_align=launch_cfg.assumed_align),
        _make_optional_gmem_ptr(
            weight,
            weight_dtype or dtype,
            assumed_align=launch_cfg.weight_assumed_align,
        ),
        _make_optional_gmem_ptr(bias, dtype, assumed_align=launch_cfg.assumed_align),
        _make_optional_gmem_ptr(
            residual, dtype, assumed_align=launch_cfg.assumed_align
        ),
        _make_gmem_ptr(dtype, out.data_ptr(), assumed_align=launch_cfg.assumed_align),
        _make_optional_gmem_ptr(
            residual_out,
            dtype,
            assumed_align=launch_cfg.assumed_align,
        ),
        _make_optional_gmem_ptr(rstd, cutlass.Float32, assumed_align=4),
    )


def _make_fused_add_ptrs(
    *,
    x: Tensor,
    weight: Tensor,
    residual: Tensor,
    dtype: type[cutlass.Numeric],
    assumed_align: int,
):
    return (
        _make_gmem_ptr(dtype, x.data_ptr(), assumed_align=assumed_align),
        _make_gmem_ptr(dtype, weight.data_ptr(), assumed_align=assumed_align),
        _make_gmem_ptr(dtype, residual.data_ptr(), assumed_align=assumed_align),
    )


def _apply_backward_launch_overrides(
    op: object,
    *,
    atomic_dw: bool,
    N: int,
    dtype: type[cutlass.Numeric],
    weight_dtype: type[cutlass.Numeric],
) -> None:
    if atomic_dw:
        return
    if (
        N == 4096
        and dtype.width == 16
        and (weight_dtype.width == 16 or weight_dtype is cutlass.Float32)
    ):
        op._tpr_override = 256  # type: ignore[attr-defined]
        op._nt_override = 256  # type: ignore[attr-defined]
        return
    if N == 6144 and dtype is cutlass.Float16 and weight_dtype is cutlass.Float32:
        op._tpr_override = 128  # type: ignore[attr-defined]
        op._nt_override = 256  # type: ignore[attr-defined]


def _make_bwd_ptrs(
    *,
    x: Tensor,
    weight: Tensor,
    dout: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Tensor,
    dtype: type[cutlass.Numeric],
    weight_dtype: type[cutlass.Numeric],
    assumed_align_x: int,
    assumed_align_w: int,
    assumed_align_dw: int,
):
    return (
        _make_gmem_ptr(dtype, x.data_ptr(), assumed_align=assumed_align_x),
        _make_gmem_ptr(weight_dtype, weight.data_ptr(), assumed_align=assumed_align_w),
        _make_gmem_ptr(dtype, dout.data_ptr(), assumed_align=assumed_align_x),
        _make_gmem_ptr(cutlass.Float32, rstd.data_ptr(), assumed_align=assumed_align_x),
        _make_gmem_ptr(dtype, dx.data_ptr(), assumed_align=assumed_align_x),
        _make_gmem_ptr(
            cutlass.Float32,
            dw_partial.data_ptr(),
            assumed_align=assumed_align_dw,
        ),
    )


def _make_rmsnorm_fallback_launch(
    *,
    compiled: object,
    stream: cuda.CUstream,
    assumed_align: int,
    assumed_align_w: int,
    weight_dtype: type[cutlass.Numeric] | None,
):
    def _fallback_launch(
        *,
        x: Tensor,
        weight: Tensor | None,
        out: Tensor,
        M: int,
        N: int,
        ld: int,
        eps: float,
        **_: object,
    ) -> None:
        dtype = TORCH2CUTE_DTYPE[x.dtype]
        ptr_x = _make_gmem_ptr(dtype, x.data_ptr(), assumed_align=assumed_align)
        ptr_out = _make_gmem_ptr(dtype, out.data_ptr(), assumed_align=assumed_align)
        ptr_w = _make_optional_gmem_ptr(
            weight,
            weight_dtype or dtype,
            assumed_align=assumed_align_w,
        )
        compiled(
            ptr_x,
            ptr_w,
            None,
            None,
            ptr_out,
            None,
            None,
            Int32(M),
            Int32(N),
            Int32(ld),
            stream,
            Float32(eps),
        )

    return _fallback_launch


def _make_fused_add_fallback_launch(
    *,
    compiled: object,
    stream: cuda.CUstream,
    assumed_align: int,
):
    def _fallback_launch(
        *,
        x: Tensor,
        weight: Tensor,
        residual: Tensor,
        M: int,
        N: int,
        ld_x: int,
        eps: float,
        **_: object,
    ) -> None:
        dtype = TORCH2CUTE_DTYPE[x.dtype]
        ptr_x, ptr_w, ptr_res = _make_fused_add_ptrs(
            x=x,
            weight=weight,
            residual=residual,
            dtype=dtype,
            assumed_align=assumed_align,
        )
        compiled(
            ptr_x,
            ptr_w,
            ptr_res,
            Int32(M),
            Int32(N),
            Int32(ld_x),
            stream,
            Float32(eps),
        )

    return _fallback_launch


def _make_rmsnorm_bwd_fallback_launch(
    *,
    compiled: object,
    stream: cuda.CUstream,
    assumed_align_x: int,
    assumed_align_w: int,
    assumed_align_dw: int,
    weight_dtype: type[cutlass.Numeric] | None,
):
    def _fallback_launch(
        *,
        x: Tensor,
        weight: Tensor | None,
        dout: Tensor,
        rstd: Tensor,
        dx: Tensor,
        dw_partial: Tensor | None,
        M: int,
        N: int,
        ld: int,
        sm_count: int,
        **_: object,
    ) -> None:
        dtype = TORCH2CUTE_DTYPE[x.dtype]
        ptr_x = _make_gmem_ptr(dtype, x.data_ptr(), assumed_align=assumed_align_x)
        ptr_dout = _make_gmem_ptr(dtype, dout.data_ptr(), assumed_align=assumed_align_x)
        ptr_dx = _make_gmem_ptr(dtype, dx.data_ptr(), assumed_align=assumed_align_x)
        ptr_rstd = _make_gmem_ptr(
            TORCH2CUTE_DTYPE[rstd.dtype],
            rstd.data_ptr(),
            assumed_align=assumed_align_x,
        )
        ptr_w = _make_optional_gmem_ptr(
            weight,
            weight_dtype or dtype,
            assumed_align=assumed_align_w,
        )
        ptr_dw_partial = _make_optional_gmem_ptr(
            dw_partial,
            TORCH2CUTE_DTYPE[dw_partial.dtype]
            if dw_partial is not None
            else cutlass.Float32,
            assumed_align=assumed_align_dw,
        )
        compiled(
            ptr_x,
            ptr_w,
            ptr_dout,
            ptr_rstd,
            ptr_dx,
            ptr_dw_partial,
            Int32(M),
            Int32(N),
            Int32(ld),
            Int32(sm_count),
            stream,
        )

    return _fallback_launch


def _get_fast_ptr_rmsnorm_bwd_launcher(
    *,
    compiled: object,
    dtype: type[cutlass.Numeric],
    weight_dtype: type[cutlass.Numeric] | None,
    N: int,
    device_index: int,
    stream_handle: int,
    has_weight: bool,
    has_dw_partial: bool,
    assumed_align_x: int,
    assumed_align_w: int,
    assumed_align_dw: int,
) -> _GenericFastLaunch | None:
    assumed_align_x = int(assumed_align_x)
    assumed_align_w = int(assumed_align_w)
    assumed_align_dw = int(assumed_align_dw)
    key = (
        "ptr_bwd_fast",
        id(compiled),
        N,
        dtype,
        weight_dtype,
        device_index,
        int(stream_handle),
        has_weight,
        has_dw_partial,
        assumed_align_x,
        assumed_align_w,
        assumed_align_dw,
    )
    ptr_x = _make_gmem_ptr(dtype, 0, assumed_align=assumed_align_x)
    ptr_w = (
        _make_gmem_ptr(weight_dtype or dtype, 0, assumed_align=assumed_align_w)
        if has_weight
        else None
    )
    ptr_dout = _make_gmem_ptr(dtype, 0, assumed_align=assumed_align_x)
    ptr_rstd = _make_gmem_ptr(cutlass.Float32, 0, assumed_align=assumed_align_x)
    ptr_dx = _make_gmem_ptr(dtype, 0, assumed_align=assumed_align_x)
    ptr_dw_partial = (
        _make_gmem_ptr(cutlass.Float32, 0, assumed_align=assumed_align_dw)
        if has_dw_partial
        else None
    )
    arg_m = _StableI32Arg(0)
    arg_n = _StableI32Arg(N)
    arg_ld = _StableI32Arg(N)
    arg_sm_count = _StableI32Arg(0)
    return _build_generic_fast_launcher(
        key=key,
        compiled=compiled,
        device_index=device_index,
        stream_handle=stream_handle,
        execution_args_builder=lambda stream: (
            ptr_x,
            ptr_w,
            ptr_dout,
            ptr_rstd,
            ptr_dx,
            ptr_dw_partial,
            arg_m,
            arg_n,
            arg_ld,
            arg_sm_count,
            stream,
        ),
        keepalive_items=(
            ptr_x,
            ptr_w,
            ptr_dout,
            ptr_rstd,
            ptr_dx,
            ptr_dw_partial,
            arg_m,
            arg_n,
            arg_ld,
            arg_sm_count,
        ),
        ptr_slots=(
            (ptr_x, "x"),
            (ptr_w, "weight"),
            (ptr_dout, "dout"),
            (ptr_rstd, "rstd"),
            (ptr_dx, "dx"),
            (ptr_dw_partial, "dw_partial"),
        ),
        scalar_slots=(
            (arg_m, "M", -1),
            (arg_ld, "ld", -1),
            (arg_sm_count, "sm_count", -1),
        ),
        fallback_launch_builder=lambda stream: _make_rmsnorm_bwd_fallback_launch(
            compiled=compiled,
            stream=stream,
            assumed_align_x=assumed_align_x,
            assumed_align_w=assumed_align_w,
            assumed_align_dw=assumed_align_dw,
            weight_dtype=weight_dtype if has_weight else None,
        ),
    )


def _get_fast_ptr_rmsnorm_launcher(
    *,
    compiled: object,
    dtype: type[cutlass.Numeric],
    weight_dtype: type[cutlass.Numeric] | None = None,
    N: int,
    device_index: int,
    stream_handle: int,
    has_weight: bool,
    assumed_align: int = 16,
    assumed_align_w: int | None = None,
    eps: float,
) -> _GenericFastLaunch | None:
    assumed_align = int(assumed_align)
    assumed_align_w = int(assumed_align if assumed_align_w is None else assumed_align_w)
    key = (
        "ptr_fast",
        id(compiled),
        N,
        dtype,
        weight_dtype,
        device_index,
        int(stream_handle),
        has_weight,
        assumed_align,
        assumed_align_w,
    )
    ptr_x = _make_gmem_ptr(dtype, 0, assumed_align=assumed_align)
    ptr_out = _make_gmem_ptr(dtype, 0, assumed_align=assumed_align)
    ptr_w = (
        _make_gmem_ptr(weight_dtype or dtype, 0, assumed_align=assumed_align_w)
        if has_weight
        else None
    )
    arg_m = _StableI32Arg(0)
    arg_n = _StableI32Arg(N)
    arg_ld = _StableI32Arg(N)
    arg_eps = _StableF32Arg(eps)
    return _build_generic_fast_launcher(
        key=key,
        compiled=compiled,
        device_index=device_index,
        stream_handle=stream_handle,
        execution_args_builder=lambda stream: (
            ptr_x,
            ptr_w,
            None,
            None,
            ptr_out,
            None,
            None,
            arg_m,
            arg_n,
            arg_ld,
            stream,
            arg_eps,
        ),
        keepalive_items=(ptr_x, ptr_w, ptr_out, arg_m, arg_n, arg_ld, arg_eps),
        ptr_slots=((ptr_x, "x"), (ptr_w, "weight"), (ptr_out, "out")),
        scalar_slots=(
            (arg_m, "M", -1),
            (arg_ld, "ld", -1),
            (arg_eps, "eps", float("nan")),
        ),
        fallback_launch_builder=lambda stream: _make_rmsnorm_fallback_launch(
            compiled=compiled,
            stream=stream,
            assumed_align=assumed_align,
            assumed_align_w=assumed_align_w,
            weight_dtype=weight_dtype if has_weight else None,
        ),
    )


def _get_fast_ptr_fused_add_rmsnorm_launcher(
    *,
    compiled: object,
    dtype: type[cutlass.Numeric],
    N: int,
    device_index: int,
    stream_handle: int,
    copy_bits: int,
    use_async: bool,
    tpr: int,
    direct_gmem: bool,
    assumed_align: int,
    eps: float,
) -> _GenericFastLaunch | None:
    assumed_align = int(assumed_align)
    key = (
        "ptr_fused_add_fast",
        id(compiled),
        N,
        dtype,
        device_index,
        int(stream_handle),
        int(copy_bits),
        bool(use_async),
        int(tpr),
        bool(direct_gmem),
        assumed_align,
    )
    ptr_x = _make_gmem_ptr(dtype, 0, assumed_align=assumed_align)
    ptr_res = _make_gmem_ptr(dtype, 0, assumed_align=assumed_align)
    ptr_w = _make_gmem_ptr(dtype, 0, assumed_align=assumed_align)
    arg_m = _StableI32Arg(0)
    arg_n = _StableI32Arg(N)
    arg_ld_x = _StableI32Arg(N)
    arg_eps = _StableF32Arg(eps)
    return _build_generic_fast_launcher(
        key=key,
        compiled=compiled,
        device_index=device_index,
        stream_handle=stream_handle,
        execution_args_builder=lambda stream: (
            ptr_x,
            ptr_w,
            ptr_res,
            arg_m,
            arg_n,
            arg_ld_x,
            stream,
            arg_eps,
        ),
        keepalive_items=(ptr_x, ptr_w, ptr_res, arg_m, arg_n, arg_ld_x, arg_eps),
        ptr_slots=((ptr_x, "x"), (ptr_w, "weight"), (ptr_res, "residual")),
        scalar_slots=(
            (arg_m, "M", -1),
            (arg_ld_x, "ld_x", -1),
            (arg_eps, "eps", float("nan")),
        ),
        fallback_launch_builder=lambda stream: _make_fused_add_fallback_launch(
            compiled=compiled,
            stream=stream,
            assumed_align=assumed_align,
        ),
    )


# -------------------------
# Copy helpers (allow up to 256b)
# -------------------------


@cute.jit
def get_copy_atom_bw(
    dtype: type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False
) -> cute.CopyAtom:
    # cp.async (SIMT) supports up to 128b per op; use 256b for sync when possible
    max_bits = const_expr(128 if is_async else 256)
    num_copy_bits = const_expr(min(max_bits, num_copy_elems * dtype.width))
    from cutlass.cute.nvgpu import cpasync

    # Prefer GLOBAL cache policy for bulk streaming reads at large M.
    copy_op = (
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
        if is_async
        else cute.nvgpu.CopyUniversalOp()
    )
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@cute.jit
def copy_tiled(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> None:
    atom = get_copy_atom_bw(src.element_type, num_copy_elems, is_async)
    cute.copy(atom, src, dst, pred=pred)


# -------------------------
# RMSNorm Kernel (SM100)
# -------------------------


class RMSNormSM100:
    def __init__(
        self,
        N: int,
        dtype: type[cutlass.Numeric],
        stage: int | None = None,
        *,
        copy_bits: int = 128,
        use_async: bool = True,
        direct_gmem: bool = False,
    ):
        self.N = N
        self.dtype = dtype
        # Match Quack default for RMSNorm: stage = 1 unless explicitly overridden
        self.stage = 1 if stage is None else stage
        self.reduction_dtype = cutlass.Float32
        self.copy_bits = int(copy_bits)
        self.use_async = bool(use_async)
        self.direct_gmem = bool(direct_gmem)

    def _threads_per_row(self) -> int:
        tpr = getattr(self, "_tpr_override", None)
        if tpr is not None:
            return int(tpr)

        if self.dtype.width == 16:
            special = _specialized_bf16_threads(self.N, self.copy_bits)
            if special is not None:
                return special[0]

        N = self.N
        if N <= 1024:
            return 32
        if N <= 4096:
            return 128
        if N <= 8192:
            return 128
        return 256

    def _cluster_n(self) -> int:
        cn = getattr(self, "_cluster_n_override", None)
        if cn is not None:
            return int(cn)
        N = self.N
        if N <= 8192:
            return 1
        limits = (
            ((16 * 1024, 2), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8))
            if const_expr(self.dtype.width == 16)
            else ((32 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8))
        )
        for bound, cluster_n in limits:
            if N <= bound:
                return cluster_n
        return 16

    def _num_threads(self) -> int:
        nt = getattr(self, "_nt_override", None)
        if nt is not None:
            return int(nt)
        if self.dtype.width == 16:
            special = _specialized_bf16_threads(self.N, self.copy_bits)
            if special is not None:
                return special[1]
        if self.N <= 1024:
            return 32
        return 128 if self.N <= 16384 else 256

    def _tv_layout(self, num_copy_bits: int = 256) -> tuple[cute.Shape, cute.Layout]:
        vecsize = num_copy_bits // self.dtype.width
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        tpr = self._threads_per_row()
        cluster_n = self._cluster_n()
        num_cols_vec = cute.ceil_div(self.N, vecsize)
        num_blocks_N = cute.ceil_div(num_cols_vec, tpr * cluster_n)
        cols_per_block = num_threads // tpr
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * tpr)
        tv_layout = cute.make_layout(
            ((tpr, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * tpr),
            ),
        )
        return tiler_mn, tv_layout

    def _smem_bytes(self, tiler_mn, num_warps) -> int:
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + self.stage
            * num_warps
            * self._cluster_n()
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRes: cute.Tensor | None,
        mO: cute.Tensor,
        mResO: cute.Tensor | None,
        mRstd: cute.Tensor | None,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        # Make last dim static (N)
        semistatic_shape = (*mX.shape[:-1], self.N)

        def new_stride(t):
            return (
                cute.assume(t.stride[0], divby=256 // t.element_type.width),
                t.stride[1],
            )

        mX, mRes, mO, mResO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            if const_expr(t is not None)
            else None
            for t in (mX, mRes, mO, mResO)
        ]
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype

        copy_bits = int(self.copy_bits)
        tiler_mn, tv_layout = self._tv_layout(num_copy_bits=copy_bits)
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        threads_per_row = (
            tv_layout.shape[0][0]
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._threads_per_row()
        )
        warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
        cluster_n = self._cluster_n()

        if const_expr(mW is not None):
            mW = cute.make_tensor(
                mW.iterator,
                cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
            )
        if const_expr(mB is not None):
            mB = cute.make_tensor(
                mB.iterator,
                cute.prepend(mB.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
            )
        if const_expr(mRstd is not None):
            mRstd = cute.make_tensor(
                mRstd.iterator,
                cute.append(mRstd.layout, cute.make_layout((self.N,), stride=(0,))),
            )

        # No SMEM reload mode switch; overlap is controlled in the K-loop path

        # Compute smem usage considering staged buffers.
        #
        # In direct-gmem mode, we skip the gmem->smem tiles entirely and only
        # keep the reduction buffers in shared memory.
        stage_bufs = 2 if self.stage > 1 else 1
        tile_bytes_x = (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * stage_bufs
            if const_expr(not self.direct_gmem)
            else 0
        )
        tile_bytes_res = (
            cute.size_in_bytes(mRes.element_type, cute.make_layout(tiler_mn))
            * stage_bufs
            if const_expr(mRes is not None and not self.direct_gmem)
            else 0
        )
        red_bytes = (
            self.stage * num_warps * cluster_n * (self.reduction_dtype.width // 8)
        )
        # mbarriers are only allocated/used for cluster_n>1. Some CuTeDSL builds
        # require mbarrier state to be 16B-aligned in shared memory; account for
        # the alignment padding when computing dynamic smem bytes.
        smem_bytes = tile_bytes_x + tile_bytes_res + red_bytes
        if cluster_n > 1:
            # Align up to 16B before placing the mbarrier array.
            smem_bytes = ((smem_bytes + 15) // 16) * 16
            smem_bytes += self.stage * (cutlass.Int64.width // 8)

        kernel = (
            self.kernel(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                const_expr(cluster_n),
                const_expr(num_warps),
                const_expr(warps_per_row),
                const_expr(threads_per_row),
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
            )
        )
        kernel.launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=([1, cluster_n, 1] if cluster_n > 1 else None),
            smem=smem_bytes,
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_x: cute.Pointer,
        ptr_w: cute.Pointer | None,
        ptr_b: cute.Pointer | None,
        ptr_res: cute.Pointer | None,
        ptr_out: cute.Pointer,
        ptr_res_out: cute.Pointer | None,
        ptr_rstd: cute.Pointer | None,
        M: Int32,
        N_dyn: Int32,
        ld: Int32,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        """Pointer-based entrypoint to reuse the existing RMSNorm schedule.

        This reconstructs cute.Tensor views from raw pointers plus sizes,
        avoiding any DLPack conversions at the Python boundary.
        """
        # Use a dynamic N for the leading-dimension stride so that the
        # subsequent cute.assume(...) in __call__ sees a dynamic expression
        # rather than a plain Python int.
        # The compile-time N for the kernel (self.N) is still used to
        # specialize the schedule.
        # Assume row-major [M, N] with an arbitrary leading-dimension stride
        # (common for padded-row / packed-attention layouts).
        layout_mn = cute.make_layout((M, N_dyn), stride=(ld, 1))
        layout_n = cute.make_layout((N_dyn,), stride=(1,))
        layout_m = cute.make_layout((M,), stride=(1,))

        mX = cute.make_tensor(ptr_x, layout_mn)
        mO = cute.make_tensor(ptr_out, layout_mn)

        mRes = (
            cute.make_tensor(ptr_res, layout_mn)
            if const_expr(ptr_res is not None)
            else None
        )
        mResO = (
            cute.make_tensor(ptr_res_out, layout_mn)
            if const_expr(ptr_res_out is not None)
            else None
        )
        mW = (
            cute.make_tensor(ptr_w, layout_n) if const_expr(ptr_w is not None) else None
        )
        mB = (
            cute.make_tensor(ptr_b, layout_n) if const_expr(ptr_b is not None) else None
        )
        mRstd = (
            cute.make_tensor(ptr_rstd, layout_m)
            if const_expr(ptr_rstd is not None)
            else None
        )

        # Reuse the main JIT entry to launch the scheduled kernel.
        self.__call__(mX, mW, mB, mRes, mO, mResO, mRstd, stream, eps)

    @cute.jit
    def launch_from_ptrs_fused_add_inplace(
        self,
        ptr_x: cute.Pointer,
        ptr_w: cute.Pointer,
        ptr_res: cute.Pointer,
        M: Int32,
        N_dyn: Int32,
        ld_x: Int32,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        """Pointer-based entrypoint for vLLM-style fused_add_rms_norm semantics.

        This specialized entrypoint supports:
        - `x` / output with an arbitrary leading-dimension stride (`ld_x`), and
        - `residual` / residual-out as a contiguous [M, N] tensor (ld_res = N).

        Both `x` and `residual` are updated in-place:
          residual <- x + residual
          x <- RMSNorm(residual) * weight
        """
        layout_x = cute.make_layout((M, N_dyn), stride=(ld_x, 1))
        layout_res = cute.make_layout((M, N_dyn), stride=(N_dyn, 1))
        layout_n = cute.make_layout((N_dyn,), stride=(1,))

        mX = cute.make_tensor(ptr_x, layout_x)
        mO = cute.make_tensor(ptr_x, layout_x)
        mRes = cute.make_tensor(ptr_res, layout_res)
        mResO = cute.make_tensor(ptr_res, layout_res)
        mW = cute.make_tensor(ptr_w, layout_n)

        self.__call__(
            mX,
            mW,
            None,  # bias
            mRes,
            mO,
            mResO,
            None,  # rstd
            stream,
            eps,
        )

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRes: cute.Tensor | None,
        mO: cute.Tensor,
        mResO: cute.Tensor | None,
        mRstd: cute.Tensor | None,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        cluster_n: cutlass.Constexpr[int],
        num_warps: cutlass.Constexpr[int],
        warps_per_row: cutlass.Constexpr[int],
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cluster_n > 1):
            cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        else:
            cta_rank_in_cluster = const_expr(0)
        n_off = cta_rank_in_cluster * tiler_mn[1]

        smem = cutlass.utils.SmemAllocator()
        # Allocate one or two SMEM buffers depending on stage depth
        sX0 = (
            smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(not self.direct_gmem)
            else None
        )
        sX1 = (
            smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(self.stage > 1 and not self.direct_gmem)
            else None
        )
        sRes0 = (
            smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(mRes is not None and not self.direct_gmem)
            else None
        )
        sRes1 = (
            smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(mRes is not None and self.stage > 1 and not self.direct_gmem)
            else None
        )

        # Reduction buffers + mbar for cluster reduce (reused by row_reduce helper)
        red_layout = cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype, red_layout, byte_alignment=4
        )
        if const_expr(cluster_n > 1):
            # Some CuTeDSL builds appear sensitive to the shared-memory alignment of
            # mbarrier state. `SmemAllocator.allocate_array` does not currently
            # expose an alignment parameter, so allocate an Int64 tensor with an
            # explicit alignment and pass its iterator as the pointer.
            mbar_tensor = smem.allocate_tensor(
                cutlass.Int64,
                cute.make_layout((self.stage,), stride=(1,)),
                byte_alignment=16,
            )
            mbar_ptr = mbar_tensor.iterator
        else:
            mbar_ptr = None

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        limit_k = shape[1] - n_off

        # Tiled copy setup
        num_copy_elems_X = tv_layout.shape[1][0]
        use_async = const_expr(
            self.use_async and self.N >= 1024 and not self.direct_gmem
        )
        copy_atom = get_copy_atom_bw(
            mX.element_type, num_copy_elems_X, is_async=use_async
        )
        thr_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn).get_slice(tidx)

        # Tail predicate for the N dimension (when tile width > N). Reuse this
        # for W/B loads so we never read past the end of those 1D tensors.
        is_even_N_wb = const_expr(shape[1] == tiler_mn[1] * cluster_n)
        if const_expr(not is_even_N_wb):
            cX0 = cute.local_tile(idX, tiler_mn, (0, 0))
            tXp_wb = qutils.predicate_k(thr_copy.partition_S(cX0), limit=limit_k)
        else:
            tXp_wb = None

        # Weight/bias loads:
        #
        # - Direct-GMEM schedule: load weight/bias up front to hide latency.
        # - Staged SMEM schedule: loading after the reduction reduces register
        #   pressure during the long-scoreboard reduction phase (better for large-M),
        #   but it measurably hurts small-M latency for the non-fused (no residual,
        #   no bias) case. For that specific case, prefetch weight up front as well.
        tXrW = None
        tXrB = None
        prefetch_w_early = bool(
            mW is not None and (self.direct_gmem or (mRes is None and mB is None))
        )
        if const_expr(prefetch_w_early):
            gW = cute.local_tile(
                qutils.domain_offset_i64((0, n_off), mW), tiler_mn, (0, 0)
            )
            tXgW = thr_copy.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if const_expr(not is_even_N_wb):
                tXrW.fill(0)
            cute.copy(
                get_copy_atom_bw(mW.element_type, num_copy_elems_X, is_async=False),
                tXgW,
                tXrW,
                pred=tXp_wb,
            )
        if const_expr(self.direct_gmem and mB is not None):
            gB = cute.local_tile(
                qutils.domain_offset_i64((0, n_off), mB), tiler_mn, (0, 0)
            )
            tXgB = thr_copy.partition_S(gB)
            tXrB = cute.make_fragment_like(tXgB)
            if const_expr(not is_even_N_wb):
                tXrB.fill(0)
            cute.copy(
                get_copy_atom_bw(mB.element_type, num_copy_elems_X, is_async=False),
                tXgB,
                tXrB,
                pred=tXp_wb,
            )

        # Non-persistent per-CTA execution (one tile in M)
        self._init_cluster(tidx, mbar_ptr)

        mX_i, mRes_i, mO_i, mResO_i = [
            qutils.domain_offset_i64((bidx * tiler_mn[0], 0), t)
            if t is not None
            else None
            for t in (mX, mRes, mO, mResO)
        ]
        mX_i, mRes_i, mO_i, mResO_i = [
            qutils.domain_offset_i64((0, n_off), t) if t is not None else None
            for t in (mX_i, mRes_i, mO_i, mResO_i)
        ]
        gX_i = cute.local_tile(mX_i, tiler_mn, (0, 0))
        gO_i = cute.local_tile(mO_i, tiler_mn, (0, 0))
        gRes_i = (
            cute.local_tile(mRes_i, tiler_mn, (0, 0))
            if const_expr(mRes is not None)
            else None
        )
        gResO_i = (
            cute.local_tile(mResO_i, tiler_mn, (0, 0))
            if const_expr(mResO is not None)
            else None
        )
        gRstd_i = (
            cute.local_tile(mRstd, tiler_mn, (bidx, 0))
            if const_expr(mRstd is not None)
            else None
        )
        cX_i = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Common identity/row index partitions reused by both default and K-loop paths
        tXcX_i = thr_copy.partition_S(cX_i)[(0, None), None, None]
        row_i = tXcX_i[0][0]
        tXgRstd_i = (
            thr_copy.partition_D(gRstd_i) if const_expr(mRstd is not None) else None
        )

        # Stage-2 intra-row K-loop cp.async path for the large-`N` Blackwell
        # rows where the legacy stage-2 fallback was still measurably ahead of
        # the generic one-row schedule.
        if const_expr(
            self.stage > 1
            and not self.direct_gmem
            and use_async
            and cluster_n == 1
            and (shape[1] == 6144 or shape[1] == 7168 or shape[1] == 8192)
        ):
            vecsize = tv_layout.shape[1][0]
            tpr = threads_per_row
            target_tile_n = const_expr(4096 if shape[1] != 8192 else 8192)
            tile_factor = const_expr(target_tile_n // (vecsize * tpr))
            if const_expr(tile_factor > 0):
                tile_n = vecsize * tpr * tile_factor
                num_tiles = cute.ceil_div(shape[1], tile_n)

                tiler_mn_tile = (tiler_mn[0], tile_n)
                sX0_tile = cute.local_tile(sX0, tiler_mn_tile, (0, 0))
                sX1_tile = cute.local_tile(sX1, tiler_mn_tile, (0, 0))
                sRes0_tile = (
                    cute.local_tile(sRes0, tiler_mn_tile, (0, 0))
                    if const_expr(mRes is not None)
                    else None
                )
                sRes1_tile = (
                    cute.local_tile(sRes1, tiler_mn_tile, (0, 0))
                    if const_expr(mRes is not None)
                    else None
                )

                tv_layout_tile = cute.make_layout(
                    ((tpr, tiler_mn[0]), (vecsize, tile_factor)),
                    stride=(
                        (vecsize * tiler_mn[0], 1),
                        (tiler_mn[0], tiler_mn[0] * vecsize * tpr),
                    ),
                )
                thr_copy_tile = cute.make_tiled_copy(
                    copy_atom, tv_layout_tile, tiler_mn_tile
                ).get_slice(tidx)

                # Accumulate per-thread partial sums across tiles; reduce once.
                sum_sq_thread = cute.Float32(0.0)

                # Preload tile 0 into sX0/sRes0.
                k_off0 = const_expr(0) * tile_n
                gX_0 = cute.local_tile(
                    qutils.domain_offset_i64((0, k_off0), mX_i), tiler_mn_tile, (0, 0)
                )
                tXgX_0 = thr_copy_tile.partition_S(gX_0)
                tXsX_0 = thr_copy_tile.partition_D(sX0_tile)
                cX_0 = cute.local_tile(
                    cute.domain_offset((0, k_off0), cX_i), tiler_mn_tile, (0, 0)
                )
                tXc_0 = thr_copy_tile.partition_S(cX_0)
                tXp_0 = qutils.predicate_k(tXc_0, limit=limit_k)

                tXp_ping = tXp_0
                tXp_pong = tXp_0

                if row_i < shape[0]:
                    copy_tiled(
                        tXgX_0,
                        tXsX_0,
                        num_copy_elems=vecsize,
                        is_async=True,
                        pred=tXp_0,
                    )
                    if const_expr(mRes is not None):
                        gRes_0 = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off0), mRes_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgRes_0 = thr_copy_tile.partition_S(gRes_0)
                        tXsRes_0 = thr_copy_tile.partition_D(sRes0_tile)
                        copy_tiled(
                            tXgRes_0,
                            tXsRes_0,
                            num_copy_elems=vecsize,
                            is_async=True,
                            pred=tXp_0,
                        )
                cute.arch.cp_async_commit_group()

                for t in cutlass.range_constexpr(num_tiles):
                    next_t = t + 1
                    if next_t < num_tiles:
                        k_off_n = next_t * tile_n
                        gX_n = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off_n), mX_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgX_n = thr_copy_tile.partition_S(gX_n)
                        cX_n = cute.local_tile(
                            cute.domain_offset((0, k_off_n), cX_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXc_n = thr_copy_tile.partition_S(cX_n)
                        tXp_n = qutils.predicate_k(tXc_n, limit=limit_k)

                        if const_expr((t % 2) == 0):
                            tXsX_n = thr_copy_tile.partition_D(sX1_tile)
                            tXsRes_n = (
                                thr_copy_tile.partition_D(sRes1_tile)
                                if const_expr(mRes is not None)
                                else None
                            )
                            tXp_pong = tXp_n
                        else:
                            tXsX_n = thr_copy_tile.partition_D(sX0_tile)
                            tXsRes_n = (
                                thr_copy_tile.partition_D(sRes0_tile)
                                if const_expr(mRes is not None)
                                else None
                            )
                            tXp_ping = tXp_n

                        if row_i < shape[0]:
                            copy_tiled(
                                tXgX_n,
                                tXsX_n,
                                num_copy_elems=vecsize,
                                is_async=True,
                                pred=tXp_n,
                            )
                            if const_expr(mRes is not None):
                                gRes_n = cute.local_tile(
                                    qutils.domain_offset_i64((0, k_off_n), mRes_i),
                                    tiler_mn_tile,
                                    (0, 0),
                                )
                                tXgRes_n = thr_copy_tile.partition_S(gRes_n)
                                copy_tiled(
                                    tXgRes_n,
                                    tXsRes_n,
                                    num_copy_elems=vecsize,
                                    is_async=True,
                                    pred=tXp_n,
                                )
                        cute.arch.cp_async_commit_group()

                    cute.arch.cp_async_wait_group(1 if next_t < num_tiles else 0)

                    # Current tile buffer (ping/pong).
                    if const_expr((t % 2) == 0):
                        tXsX_cur = thr_copy_tile.partition_D(sX0_tile)
                        tXsRes_cur = (
                            thr_copy_tile.partition_D(sRes0_tile)
                            if const_expr(mRes is not None)
                            else None
                        )
                        pred_cur = tXp_ping
                    else:
                        tXsX_cur = thr_copy_tile.partition_D(sX1_tile)
                        tXsRes_cur = (
                            thr_copy_tile.partition_D(sRes1_tile)
                            if const_expr(mRes is not None)
                            else None
                        )
                        pred_cur = tXp_pong

                    k_off = t * tile_n
                    gX_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mX_i),
                        tiler_mn_tile,
                        (0, 0),
                    )
                    tXgX_t = thr_copy_tile.partition_S(gX_t)
                    tXrX_t = cute.make_fragment_like(tXgX_t)
                    cute.autovec_copy(tXsX_cur, tXrX_t)
                    x_t = tXrX_t.load().to(cute.Float32)
                    if const_expr(mRes is not None):
                        gRes_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mRes_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgRes_t = thr_copy_tile.partition_S(gRes_t)
                        tXrRes_t = cute.make_fragment_like(tXgRes_t)
                        cute.autovec_copy(tXsRes_cur, tXrRes_t)
                        x_t += tXrRes_t.load().to(cute.Float32)

                    if const_expr(mResO is not None):
                        gResO_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mResO_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgResO_t = thr_copy_tile.partition_D(gResO_t)
                        tXrResO_t = cute.make_fragment_like(tXgResO_t)
                        tXrResO_t.store(x_t.to(tXrResO_t.element_type))
                        if row_i < shape[0]:
                            copy_tiled(
                                tXrResO_t,
                                tXgResO_t,
                                num_copy_elems=vecsize,
                                is_async=False,
                                pred=pred_cur,
                            )

                    sum_sq_thread = sum_sq_thread + (x_t * x_t).reduce(
                        cute.ReductionOp.ADD,
                        init_val=0.0,
                        reduction_profile=0,
                    )

                sum_sq = row_reduce(
                    sum_sq_thread,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, 0],
                    mbar_ptr,
                    init_val=0.0,
                )
                rstd = cute.math.rsqrt(sum_sq / shape[1] + eps, fastmath=True)

                if const_expr(mRstd is not None):
                    if tXcX_i[0][1] == 0 and row_i < shape[0]:
                        tXgRstd_i[0] = rstd

                for t in cutlass.range_constexpr(num_tiles):
                    k_off = t * tile_n
                    cX_t = cute.local_tile(
                        cute.domain_offset((0, k_off), cX_i), tiler_mn_tile, (0, 0)
                    )
                    tXc_t = thr_copy_tile.partition_S(cX_t)
                    tXp_t = qutils.predicate_k(tXc_t, limit=limit_k)

                    if const_expr((t % 2) == 0):
                        tXsX_cur = thr_copy_tile.partition_D(sX0_tile)
                        tXsRes_cur = (
                            thr_copy_tile.partition_D(sRes0_tile)
                            if const_expr(mRes is not None)
                            else None
                        )
                    else:
                        tXsX_cur = thr_copy_tile.partition_D(sX1_tile)
                        tXsRes_cur = (
                            thr_copy_tile.partition_D(sRes1_tile)
                            if const_expr(mRes is not None)
                            else None
                        )

                    gX_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mX_i),
                        tiler_mn_tile,
                        (0, 0),
                    )
                    tXgX_t = thr_copy_tile.partition_S(gX_t)
                    tXrX_t = cute.make_fragment_like(tXgX_t)
                    cute.autovec_copy(tXsX_cur, tXrX_t)
                    x_t = tXrX_t.load().to(cute.Float32)
                    if const_expr(mRes is not None):
                        gRes_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mRes_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgRes_t = thr_copy_tile.partition_S(gRes_t)
                        tXrRes_t = cute.make_fragment_like(tXgRes_t)
                        cute.autovec_copy(tXsRes_cur, tXrRes_t)
                        x_t += tXrRes_t.load().to(cute.Float32)

                    y_t = x_t * rstd
                    if const_expr(mW is not None):
                        gW_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mW),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tWgW_t = thr_copy_tile.partition_S(gW_t)
                        tWrW_t = cute.make_fragment_like(tWgW_t)
                        copy_tiled(
                            tWgW_t,
                            tWrW_t,
                            num_copy_elems=vecsize,
                            is_async=False,
                            pred=tXp_t,
                        )
                        y_t = y_t * tWrW_t.load().to(cute.Float32)
                    if const_expr(mB is not None):
                        gB_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mB),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tWgB_t = thr_copy_tile.partition_S(gB_t)
                        tWrB_t = cute.make_fragment_like(tWgB_t)
                        copy_tiled(
                            tWgB_t,
                            tWrB_t,
                            num_copy_elems=vecsize,
                            is_async=False,
                            pred=tXp_t,
                        )
                        y_t = y_t + tWrB_t.load().to(cute.Float32)

                    gO_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mO_i),
                        tiler_mn_tile,
                        (0, 0),
                    )
                    tXgO_t = thr_copy_tile.partition_D(gO_t)
                    tXrO_t = cute.make_fragment_like(tXgO_t)
                    tXrO_t.store(y_t.to(tXrO_t.element_type))
                    if row_i < shape[0]:
                        copy_tiled(
                            tXrO_t,
                            tXgO_t,
                            num_copy_elems=vecsize,
                            is_async=False,
                            pred=tXp_t,
                        )

                return

        # Single-stage path: one-row-per-CTA
        tXgX_i = thr_copy.partition_S(gX_i)
        tXgRes_i = (
            thr_copy.partition_S(gRes_i) if const_expr(mRes is not None) else None
        )
        tXgO_i = thr_copy.partition_D(gO_i)
        tXgResO_i = (
            thr_copy.partition_D(gResO_i) if const_expr(mResO is not None) else None
        )
        # tXgRstd_i / tXcX_i / row_i prepared above
        is_even_N_i = const_expr(shape[1] == tiler_mn[1] * cluster_n)
        tXpX_i = (
            qutils.predicate_k(thr_copy.partition_S(cX_i), limit=limit_k)
            if not is_even_N_i
            else None
        )

        tXrX = cute.make_fragment_like(tXgX_i)
        tXrRes = (
            cute.make_fragment_like(tXgRes_i) if const_expr(mRes is not None) else None
        )
        if const_expr(self.direct_gmem):
            if const_expr(not is_even_N_i):
                tXrX.fill(0)
                if const_expr(tXrRes is not None):
                    tXrRes.fill(0)
            if row_i < shape[0]:
                cute.copy(copy_atom, tXgX_i, tXrX, pred=tXpX_i)
                if const_expr(tXrRes is not None):
                    cute.copy(copy_atom, tXgRes_i, tXrRes, pred=tXpX_i)
        else:
            # If N is not a multiple of the tile width, the predicated gmem->smem
            # copies leave out-of-bounds lanes uninitialized. Clear the SMEM tile
            # so masked lanes read as 0 for reduction/output.
            if const_expr(not is_even_N_i):
                thr_copy.partition_D(sX0).fill(0)
                if const_expr(mRes is not None):
                    thr_copy.partition_D(sRes0).fill(0)

            if row_i < shape[0]:
                cute.copy(copy_atom, tXgX_i, thr_copy.partition_D(sX0), pred=tXpX_i)
                if const_expr(mRes is not None):
                    cute.copy(
                        copy_atom, tXgRes_i, thr_copy.partition_D(sRes0), pred=tXpX_i
                    )
            if const_expr(use_async):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(thr_copy.partition_D(sX0), tXrX)
            if const_expr(tXrRes is not None):
                cute.autovec_copy(thr_copy.partition_D(sRes0), tXrRes)
        x_red = tXrX.load().to(cute.Float32)
        if const_expr(tXrRes is not None):
            x_red += tXrRes.load().to(cute.Float32)

        if const_expr(mResO is not None):
            tXrResO = cute.make_fragment_like(tXgResO_i)
            tXrResO.store(x_red.to(tXrResO.element_type))
            if row_i < shape[0]:
                cute.copy(
                    get_copy_atom_bw(
                        tXrResO.element_type, num_copy_elems_X, is_async=False
                    ),
                    tXrResO,
                    tXgResO_i,
                    pred=tXpX_i,
                )

        sum_sq = row_reduce(
            x_red * x_red,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq / shape[1] + eps, fastmath=True)

        if const_expr(mRstd is not None):
            if (
                tXcX_i[0][1] == 0
                and row_i < shape[0]
                and (cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXgRstd_i[0] = rstd

        if const_expr(not self.direct_gmem and (mRes is not None or mB is not None)):
            # Load weight/bias after the reduction so they don't inflate register
            # pressure during the long-scoreboard reduction phase (helping occupancy
            # when registers are the limiting factor).
            if const_expr(mW is not None):
                gW = cute.local_tile(
                    qutils.domain_offset_i64((0, n_off), mW), tiler_mn, (0, 0)
                )
                tXgW = thr_copy.partition_S(gW)
                tXrW = cute.make_fragment_like(tXgW)
                if const_expr(not is_even_N_wb):
                    tXrW.fill(0)
                cute.copy(
                    get_copy_atom_bw(mW.element_type, num_copy_elems_X, is_async=False),
                    tXgW,
                    tXrW,
                    pred=tXp_wb,
                )
            if const_expr(mB is not None):
                gB = cute.local_tile(
                    qutils.domain_offset_i64((0, n_off), mB), tiler_mn, (0, 0)
                )
                tXgB = thr_copy.partition_S(gB)
                tXrB = cute.make_fragment_like(tXgB)
                if const_expr(not is_even_N_wb):
                    tXrB.fill(0)
                cute.copy(
                    get_copy_atom_bw(mB.element_type, num_copy_elems_X, is_async=False),
                    tXgB,
                    tXrB,
                    pred=tXp_wb,
                )

        # Reuse `x_red` (x + residual, in fp32) for the output path so we don't
        # keep both `tXrX` and `tXrRes` fragments live across the reduction.
        y = x_red * rstd
        if const_expr(mW is not None):
            y = y * tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y = y + tXrB.load().to(cute.Float32)

        tXrO = cute.make_fragment_like(tXgO_i)
        tXrO.store(y.to(tXrO.element_type))
        if row_i < shape[0]:
            cute.copy(
                get_copy_atom_bw(tXrO.element_type, num_copy_elems_X, is_async=False),
                tXrO,
                tXgO_i,
                pred=tXpX_i,
            )

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mB: cute.Tensor | None,
            mRes: cute.Tensor | None,
            mO: cute.Tensor,
            mResO: cute.Tensor | None,
            mRstd: cute.Tensor | None,
            eps: Float32,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
            cluster_n: cutlass.Constexpr[int],
            num_warps: cutlass.Constexpr[int],
            warps_per_row: cutlass.Constexpr[int],
            threads_per_row: cutlass.Constexpr[int],
        ):
            self._kernel_impl(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                cluster_n,
                num_warps,
                warps_per_row,
                threads_per_row,
            )
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mB: cute.Tensor | None,
            mRes: cute.Tensor | None,
            mO: cute.Tensor,
            mResO: cute.Tensor | None,
            mRstd: cute.Tensor | None,
            eps: Float32,
        ):
            copy_bits = int(self.copy_bits)
            tiler_mn, tv_layout = self._tv_layout(num_copy_bits=copy_bits)
            num_threads = self._num_threads()
            num_warps = num_threads // cute.arch.WARP_SIZE
            threads_per_row = self._threads_per_row()
            warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
            cluster_n = self._cluster_n()
            self._kernel_impl(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                const_expr(cluster_n),
                const_expr(num_warps),
                const_expr(warps_per_row),
                const_expr(threads_per_row),
            )

    @cute.jit
    def _init_cluster(self, tidx: cutlass.Int32, mbar_ptr: cute.Pointer | None):
        if const_expr(mbar_ptr is not None):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()


def _can_use_ptr_path(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    residual: Tensor | None,
) -> bool:
    """Return whether the pointer-path forward entry is safe."""
    if x.stride(1) != 1:
        return False
    # All participating tensors are interpreted as the same element type
    # (derived from x.dtype) in the pointer-based path. If dtypes differ,
    # we'd read the wrong bit patterns and silently produce incorrect output.
    if residual is not None and residual.dtype != x.dtype:
        return False
    if weight is not None and weight.dtype != x.dtype:
        # Allow the common "Quack-style" API where weights are fp32 even when
        # activations are bf16/fp16. The pointer path constructs a weight tensor
        # view with the correct element type (fp32) inside the compiled graph.
        if weight.dtype is not torch.float32:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
    if bias is not None and bias.dtype != x.dtype:
        return False
    # The kernel assumes `ld` satisfies a divisibility constraint used by
    # cute.assume(..., divby=...) for vectorization.
    elem_bits = TORCH2CUTE_DTYPE[x.dtype].width
    divby = 256 // elem_bits
    if (x.stride(0) % divby) != 0:
        return False
    # The kernel uses 128-bit vectorized copies (16B). Require at least 16B
    # alignment on all participating tensors to avoid misaligned global loads.
    if (x.data_ptr() % 16) != 0:
        return False
    if residual is not None and residual.stride(1) != 1:
        return False
    if residual is not None and residual.stride(0) != x.stride(0):
        return False
    if residual is not None and (residual.data_ptr() % 16) != 0:
        return False
    if weight is not None and not weight.is_contiguous():
        return False
    if bias is not None and not bias.is_contiguous():
        return False
    if weight is not None:
        # For fp32 weights we use 256b universal copies (32B) by default.
        # Require 32B alignment so the compiler can safely vectorize loads.
        if weight.dtype is torch.float32:
            if (weight.data_ptr() % 32) != 0:
                return False
        else:
            if (weight.data_ptr() % 16) != 0:
                return False
    if bias is not None and (bias.data_ptr() % 16) != 0:
        return False
    return True


def _can_use_ptr_path_fused_add_inplace(
    x: Tensor,
    weight: Tensor,
    residual: Tensor,
) -> bool:
    """Return whether the pointer-path fused-add entry is safe."""
    if x.stride(1) != 1:
        return False
    if residual.dtype != x.dtype:
        return False
    if weight.dtype != x.dtype:
        return False
    if residual.stride(1) != 1:
        return False
    if not residual.is_contiguous():
        return False
    if not weight.is_contiguous():
        return False

    dtype = TORCH2CUTE_DTYPE[x.dtype]
    divby = 256 // dtype.width
    if (x.stride(0) % divby) != 0:
        return False
    if (residual.stride(0) % divby) != 0:
        return False

    if (x.data_ptr() % 16) != 0:
        return False
    if (residual.data_ptr() % 16) != 0:
        return False
    if (weight.data_ptr() % 16) != 0:
        return False
    return True


def _can_use_ptr_path_bwd(
    x: Tensor,
    weight: Tensor | None,
    dout: Tensor,
    rstd: Tensor,
) -> bool:
    """Return whether the pointer-path backward entry is safe."""
    if x.dim() != 2 or dout.dim() != 2:
        return False
    if rstd.dim() != 1:
        return False
    if x.shape != dout.shape:
        return False
    if rstd.numel() != x.shape[0]:
        return False
    # SM100 backward kernel assumes N is divisible by 8 (for 256b fp32 stores
    # into dw_partial rows).
    if (x.shape[1] % 8) != 0:
        return False
    if x.stride(1) != 1 or dout.stride(1) != 1:
        return False
    if dout.stride(0) != x.stride(0):
        return False
    if dout.dtype != x.dtype:
        return False
    if rstd.dtype != torch.float32 or not rstd.is_contiguous():
        return False
    if weight is None:
        return False
    if weight.dim() != 1 or weight.shape[0] != x.shape[1]:
        return False
    if not weight.is_contiguous():
        return False
    if weight.dtype != x.dtype:
        if weight.dtype is not torch.float32:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False

    dtype = TORCH2CUTE_DTYPE[x.dtype]
    divby = 256 // dtype.width
    if (x.stride(0) % divby) != 0:
        return False

    if (x.data_ptr() % 16) != 0:
        return False
    if (dout.data_ptr() % 16) != 0:
        return False
    # Torch CUDA allocations are typically >=256B aligned, but keep the check
    # explicit so we never assume tighter alignment than is true.
    if (rstd.data_ptr() % 4) != 0:
        return False
    if (weight.data_ptr() % (32 if weight.dtype is torch.float32 else 16)) != 0:
        return False
    return True


def _rmsnorm_forward_ptr(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    residual: Tensor | None,
    eps: float,
    store_rstd: bool,
    *,
    force_stage2: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Pointer-path RMSNorm forward that bypasses DLPack."""
    assert x.is_cuda
    assert x.dim() == 2, "Use (M, N) tensor; flatten batch/seq beforehand."
    M, N = x.shape

    # Preserve padded-row layouts by matching the input stride.
    out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    residual_out: Tensor | None = None
    rstd: Tensor | None = None

    if residual is not None:
        residual_out = torch.empty_strided(
            residual.shape,
            residual.stride(),
            device=residual.device,
            dtype=residual.dtype,
        )
    if store_rstd:
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    _rmsnorm_forward_ptr_into(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        out=out,
        residual_out=residual_out,
        rstd=rstd,
        eps=eps,
        force_stage2=force_stage2,
    )
    return out, rstd, residual_out


def _rmsnorm_forward_ptr_into(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    residual: Tensor | None,
    out: Tensor,
    residual_out: Tensor | None,
    rstd: Tensor | None,
    eps: float,
    force_stage2: bool = False,
) -> None:
    """Launch pointer-path forward into preallocated outputs."""
    assert x.is_cuda
    assert x.dim() == 2, "Use (M, N) tensor; flatten batch/seq beforehand."
    M, N = x.shape
    device_index = x.get_device()
    dtype = TORCH2CUTE_DTYPE[x.dtype]

    if bias is None and residual is None and residual_out is None and rstd is None:
        stream_handle = _current_stream_handle(device_index)
        has_weight = weight is not None

        if not has_weight and try_rmsnorm_smallm_noweight_cuda(x, out, float(eps)):
            return

        if has_weight and try_simple_weightonly_rmsnorm_forward(
            x,
            weight,
            out,
            float(eps),
        ):
            return

        weight_dtype = TORCH2CUTE_DTYPE[weight.dtype] if has_weight else None
        launch_cfg = _resolve_forward_launch_config(
            M=int(M),
            N=int(N),
            dtype=dtype,
            x=x,
            weight=weight,
            weight_dtype=weight_dtype,
            aligned_tensors=(out, weight),
        )
        launch_cfg = _override_simple_weight_only_forward_launch_config(
            launch_cfg,
            M=int(M),
            N=int(N),
            dtype=dtype,
            weight=weight,
            weight_dtype=weight_dtype,
        )
        if force_stage2:
            launch_cfg = _force_stage2_forward_launch_config(
                launch_cfg,
                M=int(M),
                N=int(N),
                dtype=dtype,
                weight=weight,
                weight_dtype=weight_dtype,
            )

        compiled_key = (
            "ptr",
            N,
            dtype,
            weight_dtype,
            False,
            has_weight,
            False,
            False,
            False,
            launch_cfg.stage,
            launch_cfg.copy_bits,
            launch_cfg.use_async,
            launch_cfg.direct_gmem,
            launch_cfg.assumed_align,
            launch_cfg.weight_assumed_align,
            launch_cfg.tpr_override,
            launch_cfg.nt_override,
            launch_cfg.cluster_n_override,
            device_index,
        )
        compiled = _PTR_COMPILE_CACHE.get(compiled_key)
        if compiled is None:
            op = _make_rmsnorm_op(
                N,
                dtype,
                stage=launch_cfg.stage,
                copy_bits=launch_cfg.copy_bits,
                use_async=launch_cfg.use_async,
                direct_gmem=launch_cfg.direct_gmem,
                tpr_override=launch_cfg.tpr_override,
                nt_override=launch_cfg.nt_override,
                cluster_n_override=launch_cfg.cluster_n_override,
            )
            ld_val = int(x.stride(0))
            ptr_x, ptr_w, _, _, ptr_out, _, _ = _make_forward_ptrs(
                x=x,
                weight=weight,
                out=out,
                dtype=dtype,
                weight_dtype=weight_dtype,
                launch_cfg=launch_cfg,
            )
            stream = cuda.CUstream(stream_handle)
            ld = Int32(ld_val)
            compiled = cute.compile(
                op.launch_from_ptrs,
                ptr_x,
                ptr_w,
                None,
                None,
                ptr_out,
                None,
                None,
                Int32(M),
                Int32(N),
                ld,
                stream,
                Float32(eps),
            )
            _PTR_COMPILE_CACHE[compiled_key] = compiled

        launcher = _get_fast_ptr_rmsnorm_launcher(
            compiled=compiled,
            dtype=dtype,
            weight_dtype=weight_dtype,
            N=N,
            device_index=device_index,
            stream_handle=stream_handle,
            has_weight=has_weight,
            assumed_align=launch_cfg.assumed_align,
            assumed_align_w=launch_cfg.weight_assumed_align,
            eps=eps,
        )
        ld_val = int(x.stride(0))
        if launcher is not None:
            launcher.launch(x=x, weight=weight, out=out, M=M, N=N, ld=ld_val, eps=eps)
            return

        ptr_x, ptr_w, _, _, ptr_out, _, _ = _make_forward_ptrs(
            x=x,
            weight=weight,
            out=out,
            dtype=dtype,
            weight_dtype=weight_dtype,
            launch_cfg=launch_cfg,
        )
        stream = cuda.CUstream(stream_handle)
        ld = Int32(ld_val)
        compiled(
            ptr_x,
            ptr_w,
            None,
            None,
            ptr_out,
            None,
            None,
            Int32(M),
            Int32(N),
            ld,
            stream,
            Float32(eps),
        )
        return

    weight_dtype = TORCH2CUTE_DTYPE[weight.dtype] if weight is not None else None
    launch_cfg = _resolve_forward_launch_config(
        M=int(M),
        N=int(N),
        dtype=dtype,
        x=x,
        weight=weight,
        weight_dtype=weight_dtype,
        aligned_tensors=(out, weight, bias, residual, residual_out),
    )
    if force_stage2:
        launch_cfg = _force_stage2_forward_launch_config(
            launch_cfg,
            M=int(M),
            N=int(N),
            dtype=dtype,
            weight=weight,
            weight_dtype=weight_dtype,
        )

    stream_handle = _current_stream_handle(device_index)
    key = (
        "ptr",
        N,
        dtype,
        weight_dtype,
        residual is not None,
        weight is not None,
        bias is not None,
        residual_out is not None,
        rstd is not None,
        launch_cfg.stage,
        launch_cfg.copy_bits,
        launch_cfg.use_async,
        launch_cfg.direct_gmem,
        launch_cfg.assumed_align,
        launch_cfg.weight_assumed_align,
        launch_cfg.tpr_override,
        launch_cfg.nt_override,
        launch_cfg.cluster_n_override,
        device_index,
    )
    compiled = _PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = _make_rmsnorm_op(
            N,
            dtype,
            stage=launch_cfg.stage,
            copy_bits=launch_cfg.copy_bits,
            use_async=launch_cfg.use_async,
            direct_gmem=launch_cfg.direct_gmem,
            tpr_override=launch_cfg.tpr_override,
            nt_override=launch_cfg.nt_override,
            cluster_n_override=launch_cfg.cluster_n_override,
        )
        ptr_x, ptr_w, ptr_b, ptr_res, ptr_out, ptr_res_out, ptr_rstd = (
            _make_forward_ptrs(
                x=x,
                weight=weight,
                bias=bias,
                residual=residual,
                out=out,
                residual_out=residual_out,
                rstd=rstd,
                dtype=dtype,
                weight_dtype=weight_dtype,
                launch_cfg=launch_cfg,
            )
        )
        stream = cuda.CUstream(stream_handle)
        ld = Int32(int(x.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_w,
            ptr_b,
            ptr_res,
            ptr_out,
            ptr_res_out,
            ptr_rstd,
            Int32(M),
            Int32(N),
            ld,
            stream,
            Float32(eps),
        )
        _PTR_COMPILE_CACHE[key] = compiled
    ptr_x, ptr_w, ptr_b, ptr_res, ptr_out, ptr_res_out, ptr_rstd = _make_forward_ptrs(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        out=out,
        residual_out=residual_out,
        rstd=rstd,
        dtype=dtype,
        weight_dtype=weight_dtype,
        launch_cfg=launch_cfg,
    )
    stream = cuda.CUstream(stream_handle)
    ld = Int32(int(x.stride(0)))
    compiled(
        ptr_x,
        ptr_w,
        ptr_b,
        ptr_res,
        ptr_out,
        ptr_res_out,
        ptr_rstd,
        Int32(M),
        Int32(N),
        ld,
        stream,
        Float32(eps),
    )


def _fused_add_rmsnorm_forward_ptr_inplace(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> None:
    """Pointer-based fused_add_rmsnorm that updates `x` and `residual` in-place."""
    assert x.is_cuda
    assert x.dim() == 2
    assert residual.is_cuda
    assert residual.dim() == 2
    assert x.shape == residual.shape

    M, N = x.shape
    device_index = x.get_device()
    dtype = TORCH2CUTE_DTYPE[x.dtype]
    stage = 1

    stream_handle = _current_stream_handle(device_index)

    copy_bits = 128
    # DSv3 fused-add prefers direct GMEM on smaller M and staged loads on larger M.
    direct_gmem = _direct_gmem_from_policy(
        default=bool(dtype.width == 16 and N == 7168 and M <= 16384)
    )
    use_async = not direct_gmem
    tpr_override: int | None = None
    nt_override: int | None = None

    if _ENABLE_STAGE2 and dtype.width == 16 and N == 7168 and M >= 4096:
        stage = 2
        direct_gmem = False
        use_async = True

    can_use_256 = bool(
        direct_gmem
        and dtype.width == 16
        and (x.data_ptr() % 32) == 0
        and (residual.data_ptr() % 32) == 0
        and (weight.data_ptr() % 32) == 0
    )
    assumed_align = 32 if can_use_256 else 16
    if can_use_256:
        copy_bits = 256

    copy_bits = _copy_bits_from_policy(default=copy_bits, can_use_256=can_use_256)
    if copy_bits == 128:
        assumed_align = 16
    elif copy_bits == 256 and can_use_256:
        assumed_align = 32
    else:
        copy_bits = 128
        assumed_align = 16

    key = (
        "ptr_fused_add_inplace",
        N,
        dtype,
        stage,
        device_index,
        copy_bits,
        use_async,
        tpr_override,
        nt_override,
        direct_gmem,
        None,
    )
    compiled = _PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = _make_rmsnorm_op(
            N,
            dtype,
            stage=stage,
            copy_bits=copy_bits,
            use_async=use_async,
            direct_gmem=direct_gmem,
            tpr_override=tpr_override,
            nt_override=nt_override,
            cluster_n_override=None,
        )
        ptr_x, ptr_w, ptr_res = _make_fused_add_ptrs(
            x=x,
            weight=weight,
            residual=residual,
            dtype=dtype,
            assumed_align=assumed_align,
        )
        stream = cuda.CUstream(stream_handle)
        ld_x = Int32(int(x.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs_fused_add_inplace,
            ptr_x,
            ptr_w,
            ptr_res,
            Int32(M),
            Int32(N),
            ld_x,
            stream,
            Float32(eps),
        )
        _PTR_COMPILE_CACHE[key] = compiled
    launcher = _get_fast_ptr_fused_add_rmsnorm_launcher(
        compiled=compiled,
        dtype=dtype,
        N=N,
        device_index=device_index,
        stream_handle=stream_handle,
        copy_bits=copy_bits,
        use_async=use_async,
        tpr=tpr_override or 0,
        direct_gmem=direct_gmem,
        assumed_align=assumed_align,
        eps=eps,
    )
    if launcher is not None:
        launcher.launch(
            x=x,
            weight=weight,
            residual=residual,
            M=M,
            N=N,
            ld_x=int(x.stride(0)),
            eps=eps,
        )
        return

    # Fall back to the regular compiled call if fast-launch is unavailable.
    ptr_x, ptr_w, ptr_res = _make_fused_add_ptrs(
        x=x,
        weight=weight,
        residual=residual,
        dtype=dtype,
        assumed_align=assumed_align,
    )
    stream = cuda.CUstream(stream_handle)
    ld_x = Int32(int(x.stride(0)))
    compiled(ptr_x, ptr_w, ptr_res, Int32(M), Int32(N), ld_x, stream, Float32(eps))


# -------------------------
# Public API (forward + verify)
# -------------------------


def rmsnorm_forward(
    x: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    residual: Tensor | None = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    assert x.is_cuda
    assert x.dim() == 2, "Use (M, N) tensor; flatten batch/seq beforehand."
    force_stage2 = _FORCE_RMSNORM_STAGE2_FWD or _should_force_stage2_forward(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        store_rstd=store_rstd,
    )

    use_ptr = _can_use_ptr_path(x, weight, bias, residual)

    if use_ptr:
        return _rmsnorm_forward_ptr(
            x,
            weight,
            bias,
            residual,
            eps,
            store_rstd,
            force_stage2=force_stage2,
        )
    return _rmsnorm_ref_outputs(x, weight, bias, residual, eps, store_rstd)


def rmsnorm_ref(
    x: Tensor,
    w: Tensor | None = None,
    b: Tensor | None = None,
    residual: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd
    if w is not None:
        y = y * w.float()
    if b is not None:
        y = y + b.float()
    return y.to(x.dtype)


def fused_add_rmsnorm_forward(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Return `(RMSNorm(x + residual), x + residual)` with vLLM semantics."""
    assert x.is_cuda and residual.is_cuda
    assert x.shape == residual.shape
    assert x.dtype == residual.dtype

    orig_shape = x.shape
    N = orig_shape[-1]

    x_2d = x.view(-1, N)
    res_2d = residual.view(-1, N)

    y_2d, _rstd, z_2d = rmsnorm_forward(
        x_2d,
        weight=weight,
        bias=None,
        residual=res_2d,
        eps=eps,
        store_rstd=False,
    )

    y = y_2d.view(orig_shape)
    z = z_2d.view(orig_shape)
    return y, z


def fused_add_rmsnorm_forward_inplace(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """In-place fused-add RMSNorm returning `(x, residual)` for vLLM."""
    fused_add_rmsnorm_inplace_(x, residual, weight, eps=eps)
    return x, residual


def fused_add_rmsnorm_inplace_(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> None:
    """Lowest-overhead in-place fused-add RMSNorm entrypoint."""
    assert x.is_cuda and residual.is_cuda
    assert x.shape == residual.shape
    assert x.dtype == residual.dtype

    N = x.shape[-1]
    x_2d = x if x.dim() == 2 else x.view(-1, N)
    res_2d = residual if residual.dim() == 2 else residual.view(-1, N)

    # Fast path: vLLM-compatible layout where x may be strided/padded but
    # residual is contiguous. This updates both tensors in-place without
    # additional allocations.
    if _can_use_ptr_path_fused_add_inplace(x_2d, weight, res_2d):
        _fused_add_rmsnorm_forward_ptr_inplace(x_2d, res_2d, weight, eps)
        return None

    # Fallback: allocate via the regular fused path, then copy results into
    # the user-provided buffers so that semantics remain identical.
    y, z = fused_add_rmsnorm_forward(x, residual, weight, eps)
    x.copy_(y)
    residual.copy_(z)
    return None


# -------------------------
# Backward kernel (SM100)
# -------------------------


class RMSNormBackwardSM100(BaseRMSNormBackward):
    """SM100-tuned wrapper around `lite_quack.RMSNormBackward`."""

    def __init__(self, dtype: cutlass.Numeric, N: int):
        super().__init__(dtype, N)

    def _get_num_threads(self) -> int:
        nt = getattr(self, "_nt_override", None)
        if nt is not None:
            return int(nt)
        return super()._get_num_threads()

    def _calculate_threads_per_row(self) -> int:
        tpr = getattr(self, "_tpr_override", None)
        if tpr is not None:
            return int(tpr)
        return super()._calculate_threads_per_row()

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_x: cute.Pointer,
        ptr_w: cute.Pointer,
        ptr_dout: cute.Pointer,
        ptr_rstd: cute.Pointer,
        ptr_dx: cute.Pointer,
        ptr_dw_partial: cute.Pointer,
        M: Int32,
        N_dyn: Int32,
        ld: Int32,
        sm_count: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Pointer-based backward entry that bypasses DLPack."""
        # `dw_partial` stores use 256b fp32 vectors, so require N to stay /8 aligned.
        N_assumed = cute.assume(N_dyn, divby=8)

        layout_mn = cute.make_layout((M, N_assumed), stride=(ld, 1))
        layout_n = cute.make_layout((N_assumed,), stride=(1,))
        layout_m = cute.make_layout((M,), stride=(1,))
        # Default: write a full (sm_count, N) partial buffer (Quack-style),
        # then reduce on the host with `torch.sum(dim=0)`.
        #
        # Optional: atomic-reduce directly into a single (N,) buffer by using
        # a broadcasted leading dimension (stride0 = 0). This avoids the extra
        # reduction kernel launch and is primarily used for tiny-M regimes.
        if const_expr(self.atomic_dw):
            layout_partial = cute.make_layout((sm_count, N_assumed), stride=(0, 1))
        else:
            layout_partial = cute.make_layout(
                (sm_count, N_assumed), stride=(N_assumed, 1)
            )

        mX = cute.make_tensor(ptr_x, layout_mn)
        mW = cute.make_tensor(ptr_w, layout_n)
        mdO = cute.make_tensor(ptr_dout, layout_mn)
        mRstd = cute.make_tensor(ptr_rstd, layout_m)
        mdX = cute.make_tensor(ptr_dx, layout_mn)
        mdW = cute.make_tensor(ptr_dw_partial, layout_partial)

        self.__call__(
            mX,
            mW,
            mdO,
            None,  # dresidual_out
            mRstd,
            mdX,
            mdW,
            None,  # dresidual
            None,  # db_partial
            sm_count,
            stream,
        )

    def _get_num_threads(self) -> int:
        # Keep 128 threads only up to N=4k; use 256 for larger rows to ensure
        # threads_per_row <= num_threads across buckets.
        nt = getattr(self, "_nt_override", None)
        if nt is not None:
            return int(nt)
        return 128 if self.N <= 4096 else 256

    def _calculate_threads_per_row(self) -> int:
        tpr = getattr(self, "_tpr_override", None)
        if tpr is not None:
            return int(tpr)
        # Match Quack's backward tiling: use 256 threads/row for N > 4096.
        #
        # The earlier "mirror forward" policy (128 threads/row for N<=8192)
        # regresses DSv3 backward at N=6144/7168/8192 on SM100.
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self) -> None:
        # Reuse the SM100 forward cluster growth policy so large-N shapes can
        # fan out across multiple CTAs in the same row.
        cn = getattr(self, "_cluster_n_override", None)
        if cn is not None:
            self.cluster_n = int(cn)
            return

        N = self.N
        if N <= 8192:
            cluster_n = 1
        elif self.dtype.width == 16:
            if N <= 16 * 1024:
                cluster_n = 2
            elif N <= 32 * 1024:
                cluster_n = 2
            elif N <= 64 * 1024:
                cluster_n = 4
            elif N <= 128 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        else:
            if N <= 32 * 1024:
                cluster_n = 1
            elif N <= 64 * 1024:
                cluster_n = 2
            elif N <= 128 * 1024:
                cluster_n = 4
            elif N <= 256 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        self.cluster_n = cluster_n

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mdResO: cute.Tensor | None,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdRes: cute.Tensor | None,
        mdB: cute.Tensor | None,
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        # Match forward's 32B alignment on the leading dimension to unlock
        # wider vectorization when legal.
        semistatic_shape = (*mX.shape[:-1], self.N)

        def new_stride(t):
            return (
                cute.assume(t.stride[0], divby=256 // t.element_type.width),
                t.stride[1],
            )

        mX, mdO, mdResO, mdX, mdRes = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            if const_expr(t is not None)
            else None
            for t in (mX, mdO, mdResO, mdX, mdRes)
        ]

        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                mX.element_type.width,
                mdO.element_type.width,
                mdX.element_type.width,
                mdResO.element_type.width if mdResO is not None else 0,
                mdRes.element_type.width if mdRes is not None else 0,
            )
        )
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=128 // largest_dtype_width * mX.element_type.width
        )
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mW is not None):
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW = cute.make_tensor(mW.iterator, mW_expanded_layout)

        num_blocks = sm_count
        kernel = (
            self.kernel(
                mX, mW, mdO, mdResO, mRstd, mdX, mdW, mdB, mdRes, tv_layout, tiler_mn
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mX, mW, mdO, mdResO, mRstd, mdX, mdW, mdB, mdRes)
        )
        kernel.launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            smem=self._smem_size_in_bytes(
                tiler_mn, num_warps, do_dtype=mdO.element_type
            ),
            stream=stream,
        )


_BWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_BWD_PTR_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _rmsnorm_bwd_sm100(
    x: Tensor,
    weight: Tensor | None,
    dout: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Tensor | None,
    db_partial: Tensor | None = None,
    dresidual_out: Tensor | None = None,
    dresidual: Tensor | None = None,
    sm_count: int | None = None,
) -> None:
    """SM100-specific RMSNorm backward dispatch.

    Mirrors Quack's `quack.rmsnorm._rmsnorm_bwd`, but instantiates
    `RMSNormBackwardSM100` (SM100-tuned heuristics).
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

    if weight is not None:
        assert weight.dim() == 1
        assert x.shape[-1] == weight.shape[0]
        assert weight.is_cuda
        assert weight.dtype in (torch.float32, torch.bfloat16, torch.float16)
    if dresidual_out is not None:
        assert dresidual_out.shape == x.shape
        assert dresidual_out.is_cuda
        assert dresidual_out.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if dresidual is not None:
        assert dresidual.shape == x.shape
        assert dresidual.is_cuda
        assert dresidual.dtype in (torch.float16, torch.bfloat16, torch.float32)

    M, N = x.size(0), x.size(1)
    if dw_partial is None and db_partial is None:
        assert sm_count is not None
    else:
        sm_count = (
            dw_partial.shape[0] if dw_partial is not None else db_partial.shape[0]
        )

    # Match Quack's conversion strategy for activations/gradients: keep the
    # (M, N) layout dynamic without enforcing additional compact-shape
    # constraints. This reduces per-call Python overhead for small-M shapes.
    def _convert_layout_dynamic(t: Tensor) -> cute.Tensor:
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )

    x_tensor, dout_tensor, dres_out_tensor, dx_tensor, dres_tensor = [
        _convert_layout_dynamic(t) if t is not None else None
        for t in (x, dout, dresidual_out, dx, dresidual)
    ]

    if weight is not None:
        weight_dtype = TORCH2CUTE_DTYPE[weight.dtype]
        weight_tensor = convert_from_dlpack_cute(
            weight.detach(),
            leading_dim=0,
            divisibility=128 // weight_dtype.width,
        )
    else:
        weight_tensor = None

    dw_partial_tensor = (
        from_dlpack(dw_partial, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if dw_partial is not None
        else None
    )
    db_partial_tensor = (
        from_dlpack(db_partial, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if db_partial is not None
        else None
    )
    rstd_tensor = from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        M,
        N,
        x_tensor.element_type,
        weight_tensor.element_type if weight is not None else None,
        db_partial.dtype if db_partial is not None else None,
        dresidual.dtype if dresidual is not None else None,
        dresidual_out.dtype if dresidual_out is not None else None,
    )
    kernel = _BWD_COMPILE_CACHE.get(compile_key)
    if kernel is None:
        op = RMSNormBackwardSM100(x_tensor.element_type, N)

        # Shape-specific tuning overrides for DSv3-style N=8192 rows.
        if isinstance(op, RMSNormBackwardSM100) and N == 8192:
            if M >= 65536:
                op._tpr_override = 256  # type: ignore[attr-defined]
                op._nt_override = 256  # type: ignore[attr-defined]
            elif M >= 16384:
                op._tpr_override = 256  # type: ignore[attr-defined]

        kernel = cute.compile(
            op,
            x_tensor,
            weight_tensor,
            dout_tensor,
            dres_out_tensor,
            rstd_tensor,
            dx_tensor,
            dw_partial_tensor,
            dres_tensor,
            db_partial_tensor,
            Int32(sm_count if sm_count is not None else 0),
            current_stream,
        )
        _BWD_COMPILE_CACHE[compile_key] = kernel

    kernel(
        x_tensor,
        weight_tensor,
        dout_tensor,
        dres_out_tensor,
        rstd_tensor,
        dx_tensor,
        dw_partial_tensor,
        dres_tensor,
        db_partial_tensor,
        Int32(sm_count if sm_count is not None else 0),
        current_stream,
    )


def _rmsnorm_bwd_sm100_ptr(
    x: Tensor,
    weight: Tensor,
    dout: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Tensor,
    sm_count: int,
    *,
    atomic_dw: bool = False,
) -> None:
    """Pointer-path SM100 RMSNorm backward launch."""
    assert _can_use_ptr_path_bwd(x, weight, dout, rstd)
    assert dx.shape == x.shape
    assert dx.dtype == x.dtype
    assert dw_partial.dtype == torch.float32

    M, N = x.size(0), x.size(1)
    if atomic_dw:
        assert dw_partial.dim() == 1 and dw_partial.numel() == N
        assert dw_partial.is_contiguous()
    else:
        assert dw_partial.dim() == 2 and dw_partial.shape[1] == N
    device_index = x.get_device()
    dtype = TORCH2CUTE_DTYPE[x.dtype]
    weight_dtype = TORCH2CUTE_DTYPE[weight.dtype]
    assumed_align_x = 16
    assumed_align_w = 32 if weight.dtype is torch.float32 else 16
    assumed_align_dw = 32
    assert (dw_partial.data_ptr() % assumed_align_dw) == 0

    stream_handle = _current_stream_handle(device_index)
    stream = cuda.CUstream(stream_handle)

    ld_val = int(x.stride(0))
    key = (
        "bwd_ptr",
        N,
        dtype,
        weight_dtype,
        int(assumed_align_x),
        int(assumed_align_w),
        int(assumed_align_dw),
        device_index,
        bool(atomic_dw),
    )
    compiled = _BWD_PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = RMSNormBackwardSM100(dtype, N)
        op.atomic_dw = bool(atomic_dw)
        _apply_backward_launch_overrides(
            op,
            atomic_dw=bool(atomic_dw),
            N=N,
            dtype=dtype,
            weight_dtype=weight_dtype,
        )
        ptr_x, ptr_w, ptr_dout, ptr_rstd, ptr_dx, ptr_dw = _make_bwd_ptrs(
            x=x,
            weight=weight,
            dout=dout,
            rstd=rstd,
            dx=dx,
            dw_partial=dw_partial,
            dtype=dtype,
            weight_dtype=weight_dtype,
            assumed_align_x=assumed_align_x,
            assumed_align_w=assumed_align_w,
            assumed_align_dw=assumed_align_dw,
        )
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_w,
            ptr_dout,
            ptr_rstd,
            ptr_dx,
            ptr_dw,
            Int32(M),
            Int32(N),
            Int32(ld_val),
            Int32(int(sm_count)),
            stream,
        )
        _BWD_PTR_COMPILE_CACHE[key] = compiled

    launcher = _get_fast_ptr_rmsnorm_bwd_launcher(
        compiled=compiled,
        dtype=dtype,
        weight_dtype=weight_dtype,
        N=N,
        device_index=device_index,
        stream_handle=stream_handle,
        has_weight=True,
        has_dw_partial=True,
        assumed_align_x=assumed_align_x,
        assumed_align_w=assumed_align_w,
        assumed_align_dw=assumed_align_dw,
    )
    if launcher is not None:
        launcher.launch(
            x=x,
            weight=weight,
            dout=dout,
            rstd=rstd,
            dx=dx,
            dw_partial=dw_partial,
            M=M,
            N=N,
            ld=ld_val,
            sm_count=int(sm_count),
        )
        return

    ptr_x, ptr_w, ptr_dout, ptr_rstd, ptr_dx, ptr_dw = _make_bwd_ptrs(
        x=x,
        weight=weight,
        dout=dout,
        rstd=rstd,
        dx=dx,
        dw_partial=dw_partial,
        dtype=dtype,
        weight_dtype=weight_dtype,
        assumed_align_x=assumed_align_x,
        assumed_align_w=assumed_align_w,
        assumed_align_dw=assumed_align_dw,
    )
    compiled(
        ptr_x,
        ptr_w,
        ptr_dout,
        ptr_rstd,
        ptr_dx,
        ptr_dw,
        Int32(M),
        Int32(N),
        Int32(ld_val),
        Int32(int(sm_count)),
        stream,
    )


def rmsnorm_backward(
    x: Tensor,
    weight: Tensor | None,
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Tensor | None = None,
    has_bias: bool = False,
    has_residual: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Public SM100 RMSNorm backward entry matching Quack's signature."""
    device = x.device
    M, N = x.size(0), x.size(1)
    dx = torch.empty_like(x)
    if dresidual_out is not None and dresidual_out.dtype != dx.dtype:
        dresidual = torch.empty_like(x, dtype=dresidual_out.dtype)
    else:
        dresidual = None

    sm_count = get_sm_count(N, device, M=M, dtype=x.dtype)

    # Clamp small N=4096 cases back to Quack's baseline to avoid allocator churn.
    if N == 4096 and M <= 8192 and x.dtype in (torch.float16, torch.bfloat16):
        num_sms = qutils.get_num_sms(device)
        sm_count = min(int(sm_count), int(num_sms) * 2)

    use_atomic_dw = False
    # Large fp32-weight DSv3 backward prefers atomically accumulating dW.
    if (
        weight is not None
        and (not has_bias)
        and (not has_residual)
        and dresidual_out is None
        and dresidual is None
        and N == 8192
        and weight.dtype is torch.float32
        and M >= 65536
        and x.dtype in (torch.float16, torch.bfloat16)
        and _can_use_ptr_path_bwd(x, weight, dout, rstd)
    ):
        use_atomic_dw = True

    if weight is not None:
        if use_atomic_dw:
            dw_partial = torch.zeros(N, device=device, dtype=torch.float32)
        else:
            dw_partial = torch.empty(sm_count, N, device=device, dtype=torch.float32)
    else:
        dw_partial = None
    db_partial = (
        torch.empty(sm_count, N, device=device, dtype=torch.float32)
        if has_bias
        else None
    )

    if (
        weight is not None
        and dw_partial is not None
        and (not has_bias)
        and (not has_residual)
        and dresidual_out is None
        and dresidual is None
        and _can_use_ptr_path_bwd(x, weight, dout, rstd)
    ):
        _rmsnorm_bwd_sm100_ptr(
            x=x,
            weight=weight,
            dout=dout,
            rstd=rstd,
            dx=dx,
            dw_partial=dw_partial,
            sm_count=int(sm_count),
            atomic_dw=bool(use_atomic_dw),
        )
    else:
        _rmsnorm_bwd_sm100(
            x,
            weight,
            dout,
            rstd,
            dx,
            dw_partial,
            db_partial,
            dresidual_out,
            dresidual,
            sm_count,
        )

    if weight is not None and dw_partial is not None:
        if use_atomic_dw:
            dw_fp32 = dw_partial
        else:
            dw_fp32 = _reduce_partial_sum_fp32(dw_partial, device_index=x.get_device())
        dw = dw_fp32 if weight.dtype is torch.float32 else dw_fp32.to(weight.dtype)
    else:
        dw = None
    db = db_partial.sum(dim=0).to(weight.dtype) if has_bias else None
    if has_residual and dresidual is None:
        dresidual = dx
    return dx, dw, db, dresidual


rmsnorm_bwd = rmsnorm_backward
