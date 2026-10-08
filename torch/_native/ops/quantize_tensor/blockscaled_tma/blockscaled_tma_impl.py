"""PyTorch-facing implementation for TMA-based block-scaled quantization."""

from typing import NamedTuple, TypeAlias

import torch

from ..utils import _device_capability
from .blockscaled_tma_config import RoundingVariant
from .blockscaled_tma_kernels import (
    _compile_blockscaled_tma,
    _QUANT_ORIENTATION_DIM_K,
    _QUANT_ORIENTATION_DIM_KM,
    _QUANT_ORIENTATION_DIM_M,
)
from .blockscaled_tma_plan import BlockscaledTmaPlan, select_blockscaled_tma_plan
from .utils import _ceil_div


_CUDA_GRID_X_MAX = 2**31 - 1
_CUDA_GRID_Y_MAX = 2**16 - 1
_CUDA_TMA_MAX_DIM = 2**32
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
    rounding_variant: RoundingVariant
    is_square_scaling: bool
    is_scale_swizzled: bool
    s_num_row_blk_k: int | None
    s_num_col_blk_k: int | None
    s_num_row_blk_m: int | None
    s_num_col_blk_m: int | None


class _PreparedBlockscaledTmaLaunch(NamedTuple):
    """Host-side launch plan, allocated outputs, and normalized optional operands."""

    plan: BlockscaledTmaPlan | None
    output_k: torch.Tensor | None
    scale_k: torch.Tensor | None
    output_m: torch.Tensor | None
    scale_m: torch.Tensor | None


class _PhiloxLaunch(NamedTuple):
    seed: torch.Tensor | None = None
    offset: torch.Tensor | None = None
    seed_scalar: int | None = None
    offset_words_scalar: int | None = None
    intragraph_offset_words: int | None = None


def _prepare_philox_launch(
    rounding_variant: RoundingVariant,
    random_key: torch.Tensor | None,
    device: int,
) -> _PhiloxLaunch:
    if rounding_variant == RoundingVariant.RTNE:
        return _PhiloxLaunch()
    if rounding_variant == RoundingVariant.STATELESS_SR:
        if random_key is None:
            raise AssertionError("stateless rounding requires random_key")
        return _PhiloxLaunch(seed=random_key.reshape(-1).view(torch.int64))

    generator = torch.cuda.default_generators[device]
    seed, offset, intragraph_offset = generator.philox_state(4)
    if rounding_variant == RoundingVariant.STATEFUL_SR_CAPTURE:
        if not seed.is_cuda or not offset.is_cuda:
            raise RuntimeError(
                "CUDA graph capture requires device-resident Philox state"
            )
        return _PhiloxLaunch(
            seed=seed,
            offset=offset,
            intragraph_offset_words=int(intragraph_offset.item()),
        )
    if seed.is_cuda or offset.is_cuda:
        raise RuntimeError("eager execution requires host-resident Philox state")
    return _PhiloxLaunch(
        seed_scalar=int(seed.item()),
        offset_words_scalar=int(offset.item()),
    )


def _validate_and_normalize_blockscaled_tma(
    input: torch.Tensor,
    *,
    quant_orientation: str,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    stochastic_rounding: bool,
    random_key: torch.Tensor | None,
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

    if not stochastic_rounding:
        if random_key is not None:
            raise ValueError("RTNE does not use random_key")
        rounding_variant = RoundingVariant.RTNE
    else:
        if not is_scale_swizzled or is_square_scaling:
            raise ValueError("stochastic rounding requires swizzled 1x32 scales")
        if random_key is not None:
            if random_key.dtype != torch.uint64 or random_key.numel() != 2:
                raise ValueError("random_key must be a two-element uint64 tensor")
            if random_key.device != input.device:
                raise ValueError("random_key and input must be on the same device")
            rounding_variant = RoundingVariant.STATELESS_SR
        else:
            rounding_variant = (
                RoundingVariant.STATEFUL_SR_CAPTURE
                if torch.cuda.is_current_stream_capturing()
                else RoundingVariant.STATEFUL_SR_EAGER
            )

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
    if is_square_scaling and M % 32 != 0:
        raise ValueError("32x32 requires M % 32 == 0")
    if do_dim_m:
        if M % 32 != 0:
            raise ValueError("dim-M requires M % 32 == 0")
        if K % 16 != 0:
            raise ValueError("dim-M requires K % 16 == 0")
        if do_dim_k and K % 32 != 0:
            raise ValueError("dim-K requires K % 32 == 0")
    elif K % 32 != 0:
        raise ValueError("requires K % 32 == 0")

    s_num_row_blk_k = s_num_col_blk_k = None
    if do_dim_k:
        s_num_row_blk_k = _ceil_div(M, 128)
        s_num_col_blk_k = _ceil_div(K, 128)

    s_num_row_blk_m = s_num_col_blk_m = None
    if do_dim_m:
        s_num_row_blk_m = _ceil_div(K, 128)
        s_num_col_blk_m = _ceil_div(M, 128)

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
        rounding_variant=rounding_variant,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
        s_num_row_blk_k=s_num_row_blk_k,
        s_num_col_blk_k=s_num_col_blk_k,
        s_num_row_blk_m=s_num_row_blk_m,
        s_num_col_blk_m=s_num_col_blk_m,
    )


def _prepare_blockscaled_tma_launch(
    input: torch.Tensor,
    spec: _BlockscaledTmaSpec,
) -> _PreparedBlockscaledTmaLaunch:
    """Select the launch plan and allocate its output storage."""

    plan = None
    if spec.M != 0 and spec.K != 0:
        if spec.M > _CUDA_TMA_MAX_DIM or spec.K > _CUDA_TMA_MAX_DIM:
            raise ValueError(
                "blockscaled TMA requires each nonempty input dimension to be at most "
                f"{_CUDA_TMA_MAX_DIM}; got shape ({spec.M}, {spec.K})"
            )
        plan = select_blockscaled_tma_plan(
            spec.M,
            spec.K,
            input_dtype=input.dtype,
            quant_orientation=spec.quant_orientation,
            is_square_scaling=spec.is_square_scaling,
            is_stochastic_qdata_rounding=spec.rounding_variant != RoundingVariant.RTNE,
        )
    if plan is not None and (
        # K <= 2**32 and tile_k_size >= 32 keep grid_k below the X limit.
        # Large M can still exceed the Y limit.
        plan.grid_k > _CUDA_GRID_X_MAX or plan.grid_m > _CUDA_GRID_Y_MAX
    ):
        raise ValueError(
            "blockscaled TMA launch grid exceeds CUDA limits: "
            f"grid=({plan.grid_k}, {plan.grid_m}, 1), "
            f"maximum=({_CUDA_GRID_X_MAX}, {_CUDA_GRID_Y_MAX}, 65535)"
        )

    output_m = scale_m = None
    if spec.do_dim_m:
        if spec.s_num_row_blk_m is None:
            raise AssertionError(
                f"expected s_num_row_blk_m, got {spec.s_num_row_blk_m}"
            )
        if spec.s_num_col_blk_m is None:
            raise AssertionError(
                f"expected s_num_col_blk_m, got {spec.s_num_col_blk_m}"
            )
        output_m = torch.empty(
            spec.K,
            spec.M,
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        scale_m = torch.empty(
            spec.s_num_row_blk_m * spec.s_num_col_blk_m * 32 * 16,
            dtype=torch.uint8,
            device=input.device,
        )

    output_k = scale_k = None
    if spec.do_dim_k:
        if spec.s_num_row_blk_k is None:
            raise AssertionError(
                f"expected s_num_row_blk_k, got {spec.s_num_row_blk_k}"
            )
        if spec.s_num_col_blk_k is None:
            raise AssertionError(
                f"expected s_num_col_blk_k, got {spec.s_num_col_blk_k}"
            )
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
                spec.s_num_row_blk_k * spec.s_num_col_blk_k * 32 * 16,
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
        if spec.s_num_row_blk_m is None:
            raise AssertionError(
                f"expected s_num_row_blk_m, got {spec.s_num_row_blk_m}"
            )
        if spec.s_num_col_blk_m is None:
            raise AssertionError(
                f"expected s_num_col_blk_m, got {spec.s_num_col_blk_m}"
            )
        scale_m = scale_m.view(spec.s_num_row_blk_m, spec.s_num_col_blk_m, 32, 16).view(
            torch.float8_e8m0fnu
        )
    if spec.do_dim_k:
        if output_k is None:
            raise AssertionError("expected output_k, got None")
        if scale_k is None:
            raise AssertionError("expected scale_k, got None")
        if spec.s_num_row_blk_k is None:
            raise AssertionError(
                f"expected s_num_row_blk_k, got {spec.s_num_row_blk_k}"
            )
        if spec.s_num_col_blk_k is None:
            raise AssertionError(
                f"expected s_num_col_blk_k, got {spec.s_num_col_blk_k}"
            )
        if spec.is_scale_swizzled:
            scale_k = scale_k.view(spec.s_num_row_blk_k, spec.s_num_col_blk_k, 32, 16)
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
    stochastic_rounding: bool,
    random_key: torch.Tensor | None,
) -> _BlockscaledTmaOutput:
    spec = _validate_and_normalize_blockscaled_tma(
        input,
        quant_orientation=quant_orientation,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
        stochastic_rounding=stochastic_rounding,
        random_key=random_key,
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
        spec.rounding_variant,
        spec.is_square_scaling,
        spec.is_scale_swizzled,
    )
    philox = _prepare_philox_launch(
        spec.rounding_variant, random_key, input.get_device()
    )
    fn(
        input,
        launch.output_k,
        launch.scale_k,
        launch.output_m,
        launch.scale_m,
        philox.seed,
        philox.offset,
        philox.seed_scalar,
        philox.offset_words_scalar,
        philox.intragraph_offset_words,
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
    stochastic_rounding: bool = False,
    random_key: torch.Tensor | None = None,
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
    capability = _device_capability(device)
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
            stochastic_rounding=stochastic_rounding,
            random_key=random_key,
        )

    if device == torch.cuda.current_device():
        return launch_on_current_device()
    with torch.cuda.device(device):
        return launch_on_current_device()
