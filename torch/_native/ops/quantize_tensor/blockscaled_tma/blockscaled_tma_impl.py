"""PyTorch-facing implementation for TMA-based block-scaled quantization."""

from functools import cache
from typing import NamedTuple, TypeAlias

import torch

from .blockscaled_tma_kernels import (
    _compile_blockscaled_tma,
    _QUANT_ORIENTATION_DIM_K,
    _QUANT_ORIENTATION_DIM_KM,
    _QUANT_ORIENTATION_DIM_M,
)
from .blockscaled_tma_plan import BlockscaledTmaPlan, select_blockscaled_tma_plan
from .utils import _ceil_div


_INT32_MAX = 2**31 - 1
_CUDA_GRID_X_MAX = _INT32_MAX
_CUDA_GRID_Y_MAX = 2**16 - 1
_INPUT_ALIGNMENT_BYTES = 16

_TwoTensorOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor]
_FourTensorOutput: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_BlockscaledTmaOutput: TypeAlias = _TwoTensorOutput | _FourTensorOutput


class _BlockscaledTmaSpec(NamedTuple):
    """Validated, normalized host-side description of one quantization request."""

    M: int
    K: int
    quant_orientation: str
    quant_orientation_id: int
    do_dim_k: bool
    do_dim_m: bool
    is_square_scaling: bool
    is_scale_swizzled: bool
    nrb_k: int | None
    ncb_k: int | None
    nrb_m: int | None
    ncb_m: int | None


class _PreparedBlockscaledTmaLaunch(NamedTuple):
    """Host-side launch plan, allocated outputs, and normalized optional operands."""

    plan: BlockscaledTmaPlan | None
    output_k: torch.Tensor | None
    scale_k: torch.Tensor | None
    output_m: torch.Tensor | None
    scale_m: torch.Tensor | None


@cache
def _cuda_capability(device: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _validate_and_normalize_blockscaled_tma(
    input: torch.Tensor,
    *,
    quant_orientation: str,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _BlockscaledTmaSpec:
    """Validate public arguments and return their canonical host-side form."""

    if quant_orientation not in ("dim_k", "dim_m", "dim_km"):
        raise ValueError(f"unsupported quant_orientation: {quant_orientation}")
    do_dim_k = quant_orientation != "dim_m"
    do_dim_m = quant_orientation != "dim_k"

    if not is_scale_swizzled:
        if quant_orientation != "dim_k":
            raise ValueError("compact scales currently support only dim-k output")
        if is_square_scaling:
            raise ValueError("compact scales do not support square scaling")

    if input.dim() != 2:
        raise ValueError(
            f"blockscaled TMA requires a 2D input; got {input.dim()} dimensions"
        )
    if not input.is_contiguous():
        raise ValueError("blockscaled TMA requires a contiguous input")
    if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("blockscaled TMA supports only bf16, fp16, and fp32 input")
    if input.data_ptr() % _INPUT_ALIGNMENT_BYTES != 0:
        raise ValueError("blockscaled TMA requires a 16-byte-aligned input")

    if is_square_scaling:
        if quant_orientation != "dim_k":
            raise ValueError("square scaling currently supports only dim-k output")
    M, K = input.shape
    if M > _INT32_MAX or K > _INT32_MAX:
        raise ValueError(
            "blockscaled TMA requires each logical dimension to fit in signed int32; "
            f"got shape ({M}, {K})"
        )
    if is_square_scaling and M % 32 != 0:
        raise ValueError("32x32 v2 requires M % 32 == 0")
    if do_dim_m:
        if M % 32 != 0:
            raise ValueError("v2 dim-M requires M % 32 == 0")
        if K % 16 != 0:
            raise ValueError("v2 dim-M requires K % 16 == 0")
        if do_dim_k and K % 32 != 0:
            raise ValueError("v2 dim-K requires K % 32 == 0")
    elif K % 32 != 0:
        raise ValueError("v2 requires K % 32 == 0")

    nrb_k = ncb_k = None
    if do_dim_k:
        nrb_k = _ceil_div(M, 128)
        ncb_k = _ceil_div(K, 128)

    nrb_m = ncb_m = None
    if do_dim_m:
        nrb_m = _ceil_div(K, 128)
        ncb_m = _ceil_div(M, 128)

    quant_orientation_id = {
        "dim_k": _QUANT_ORIENTATION_DIM_K,
        "dim_m": _QUANT_ORIENTATION_DIM_M,
        "dim_km": _QUANT_ORIENTATION_DIM_KM,
    }[quant_orientation]

    return _BlockscaledTmaSpec(
        M=M,
        K=K,
        quant_orientation=quant_orientation,
        quant_orientation_id=quant_orientation_id,
        do_dim_k=do_dim_k,
        do_dim_m=do_dim_m,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
        nrb_k=nrb_k,
        ncb_k=ncb_k,
        nrb_m=nrb_m,
        ncb_m=ncb_m,
    )


def _prepare_blockscaled_tma_launch(
    input: torch.Tensor,
    spec: _BlockscaledTmaSpec,
) -> _PreparedBlockscaledTmaLaunch:
    """Select the launch plan and allocate its output storage."""

    plan = None
    if spec.M != 0 and spec.K != 0:
        plan = select_blockscaled_tma_plan(
            spec.M,
            spec.K,
            input_dtype=input.dtype,
            quant_orientation=spec.quant_orientation,
            is_square_scaling=spec.is_square_scaling,
        )
    if plan is not None and (
        plan.grid_k > _CUDA_GRID_X_MAX or plan.grid_m > _CUDA_GRID_Y_MAX
    ):
        raise ValueError(
            "blockscaled TMA launch grid exceeds CUDA limits: "
            f"grid=({plan.grid_k}, {plan.grid_m}, 1), "
            f"maximum=({_CUDA_GRID_X_MAX}, {_CUDA_GRID_Y_MAX}, 65535)"
        )

    output_m = scale_m = None
    if spec.do_dim_m:
        if spec.nrb_m is None:
            raise AssertionError(f"expected nrb_m, got {spec.nrb_m}")
        if spec.ncb_m is None:
            raise AssertionError(f"expected ncb_m, got {spec.ncb_m}")
        output_m = torch.empty(
            spec.K,
            spec.M,
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        scale_m = torch.empty(
            spec.nrb_m * spec.ncb_m * 32 * 16,
            dtype=torch.uint8,
            device=input.device,
        )

    output_k = scale_k = None
    if spec.do_dim_k:
        if spec.nrb_k is None:
            raise AssertionError(f"expected nrb_k, got {spec.nrb_k}")
        if spec.ncb_k is None:
            raise AssertionError(f"expected ncb_k, got {spec.ncb_k}")
        output_k = torch.empty(
            spec.M,
            spec.K,
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        # Every slot is written by the kernel, so zero-initialization would launch a redundant
        # memset.
        if spec.is_scale_swizzled:
            scale_k = torch.empty(
                spec.nrb_k * spec.ncb_k * 32 * 16,
                dtype=torch.uint8,
                device=input.device,
            )
        else:
            scale_k = torch.empty(
                spec.M * (spec.K // 32),
                dtype=torch.uint8,
                device=input.device,
            )

    return _PreparedBlockscaledTmaLaunch(
        plan=plan,
        output_k=output_k,
        scale_k=scale_k,
        output_m=output_m,
        scale_m=scale_m,
    )


def _format_blockscaled_tma_output(
    spec: _BlockscaledTmaSpec,
    launch: _PreparedBlockscaledTmaLaunch,
) -> _BlockscaledTmaOutput:
    """Apply public dtypes and shapes to the allocated output storage."""

    output_k, scale_k = launch.output_k, launch.scale_k
    output_m, scale_m = launch.output_m, launch.scale_m

    if spec.do_dim_m:
        if output_m is None:
            raise AssertionError("expected output_m, got None")
        if scale_m is None:
            raise AssertionError("expected scale_m, got None")
        if spec.nrb_m is None:
            raise AssertionError(f"expected nrb_m, got {spec.nrb_m}")
        if spec.ncb_m is None:
            raise AssertionError(f"expected ncb_m, got {spec.ncb_m}")
        scale_m = scale_m.view(spec.nrb_m, spec.ncb_m, 32, 16).view(
            torch.float8_e8m0fnu
        )
    if spec.do_dim_k:
        if output_k is None:
            raise AssertionError("expected output_k, got None")
        if scale_k is None:
            raise AssertionError("expected scale_k, got None")
        if spec.nrb_k is None:
            raise AssertionError(f"expected nrb_k, got {spec.nrb_k}")
        if spec.ncb_k is None:
            raise AssertionError(f"expected ncb_k, got {spec.ncb_k}")
        if spec.is_scale_swizzled:
            scale_k = scale_k.view(spec.nrb_k, spec.ncb_k, 32, 16)
        else:
            scale_k = scale_k.view(spec.M, spec.K // 32)
        scale_k = scale_k.view(torch.float8_e8m0fnu)

    if spec.quant_orientation == "dim_k":
        if output_k is None:
            raise AssertionError("expected output_k, got None")
        if scale_k is None:
            raise AssertionError("expected scale_k, got None")
        return output_k, scale_k
    if spec.quant_orientation == "dim_m":
        if output_m is None:
            raise AssertionError("expected output_m, got None")
        if scale_m is None:
            raise AssertionError("expected scale_m, got None")
        return output_m, scale_m
    if output_k is None:
        raise AssertionError("expected output_k, got None")
    if scale_k is None:
        raise AssertionError("expected scale_k, got None")
    if output_m is None:
        raise AssertionError("expected output_m, got None")
    if scale_m is None:
        raise AssertionError("expected scale_m, got None")
    return output_k, scale_k, output_m, scale_m


def _blockscaled_tma_impl_on_current_device(
    input: torch.Tensor,
    quant_orientation: str,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _BlockscaledTmaOutput:
    spec = _validate_and_normalize_blockscaled_tma(
        input,
        quant_orientation=quant_orientation,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
    )
    launch = _prepare_blockscaled_tma_launch(input, spec)

    # CUDA cannot launch a zero-sized grid. The preparation step still allocates
    # correctly shaped empty outputs so result formatting remains shared.
    if launch.plan is None:
        return _format_blockscaled_tma_output(spec, launch)

    fn = _compile_blockscaled_tma(
        input.dtype,
        launch.plan.tile_m_size,
        launch.plan.tile_k_size,
        launch.plan.cluster_k,
        launch.plan.needs_boundary_masking,
        spec.quant_orientation_id,
        spec.is_square_scaling,
        spec.is_scale_swizzled,
    )
    fn(
        input,
        launch.output_k,
        launch.scale_k,
        launch.output_m,
        launch.scale_m,
        spec.M,
        spec.K,
        launch.plan.grid_m,
        launch.plan.grid_k,
    )
    return _format_blockscaled_tma_output(spec, launch)


def _blockscaled_tma_impl(
    input: torch.Tensor,
    quant_orientation: str,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    **kwargs: object,
) -> _BlockscaledTmaOutput:
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise ValueError(f"unexpected keyword arguments: {unexpected}")
    if not isinstance(input, torch.Tensor):
        raise ValueError("blockscaled TMA input must be a torch.Tensor")
    if input.device.type != "cuda":
        raise ValueError("blockscaled TMA requires a CUDA input")

    device = input.get_device()
    capability = _cuda_capability(device)
    if capability < (10, 0):
        raise RuntimeError(
            "blockscaled TMA requires CUDA capability 10.0 or newer; "
            f"device {input.device} has capability {capability[0]}.{capability[1]}"
        )

    def launch_on_current_device() -> _BlockscaledTmaOutput:
        return _blockscaled_tma_impl_on_current_device(
            input,
            quant_orientation=quant_orientation,
            is_square_scaling=is_square_scaling,
            is_scale_swizzled=is_scale_swizzled,
        )

    if device == torch.cuda.current_device():
        return launch_on_current_device()
    with torch.cuda.device(device):
        return launch_on_current_device()
