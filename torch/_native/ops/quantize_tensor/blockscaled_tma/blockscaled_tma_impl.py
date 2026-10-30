"""PyTorch-facing implementation for TMA-based block-scaled quantization."""

from typing import cast, NamedTuple, TypeAlias

import torch

from .blockscaled_tma_kernels import _compile_blockscaled_tma
from .blockscaled_tma_plan import (
    _QUANT_ORIENTATION_DIM_K,
    _QUANT_ORIENTATION_DIM_KM,
    _QUANT_ORIENTATION_DIM_M,
    BlockscaledTmaPlan,
    select_blockscaled_tma_plan,
)
from .utils import _ceil_div


_CUDA_GRID_X_MAX = 2**31 - 1
_CUDA_GRID_Y_MAX = 2**16 - 1
_CUDA_GRID_Z_MAX = 2**16 - 1
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
_SingleQuantDim: TypeAlias = tuple[int, int]


class _BlockscaledTmaSpec(NamedTuple):
    """Validated, normalized host-side description of one quantization request."""

    M: int
    K: int
    B: int
    is_batched: bool
    quant_orientation_id: int
    do_dim_k: bool
    do_dim_m: bool
    input_transposed: bool
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


def _validate_and_normalize_blockscaled_tma(
    input: torch.Tensor,
    *,
    dim: _SingleQuantDim,
    dual: bool,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _BlockscaledTmaSpec:
    """Validate public arguments and return their canonical host-side form."""

    ndim = input.dim()
    if ndim not in (2, 3):
        raise ValueError(
            f"blockscaled TMA requires a 2D or 3D input; got {ndim} dimensions"
        )
    if not isinstance(dim, tuple) or len(dim) != 2:
        raise ValueError(f"unsupported quantization dim: {dim}")
    first, last = dim
    if type(first) is not int or type(last) is not int:
        raise ValueError(f"unsupported quantization dim: {dim}")
    matrix_axes = (-2, -1, ndim - 2, ndim - 1)
    if first not in matrix_axes or last not in matrix_axes:
        raise ValueError(f"unsupported quantization dim: {dim}")
    normalized_dim = (first % ndim - (ndim - 2), last % ndim - (ndim - 2))
    if normalized_dim not in ((0, 1), (1, 0)):
        raise ValueError(f"expected two distinct quantization axes, got {dim}")
    input_transposed = not input.is_contiguous()
    if input_transposed:
        if not input.transpose(-2, -1).is_contiguous():
            raise ValueError(
                "blockscaled TMA requires a contiguous or transpose-contiguous input"
            )
        # The TMA input is row-major; swap logical axes for transpose-contiguous views.
        normalized_dim = (1 - normalized_dim[0], 1 - normalized_dim[1])
    row_axis, col_axis = (
        (ndim - 1, ndim - 2) if input_transposed else (ndim - 2, ndim - 1)
    )
    do_dim_k = dual or normalized_dim == (0, 1)
    do_dim_m = dual or normalized_dim == (1, 0)

    if do_dim_k and do_dim_m:
        if not is_scale_swizzled:
            raise ValueError("dual quantization does not support compact scales")
        if is_square_scaling:
            raise ValueError("dual quantization does not support square scaling")

    if not is_scale_swizzled:
        if do_dim_m:
            raise ValueError(
                f"compact scales require quantization along contiguous input axis {col_axis}; "
                f"got dim={dim}"
            )
        if is_square_scaling:
            raise ValueError("compact scales do not support square scaling")

    if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("blockscaled TMA supports only bf16, fp16, and fp32 input")
    if input.data_ptr() % _INPUT_ALIGNMENT_BYTES != 0:
        raise ValueError("blockscaled TMA requires a 16-byte-aligned input")

    if is_square_scaling:
        if do_dim_m:
            raise ValueError(
                f"square scaling requires quantization along contiguous input axis {col_axis}; "
                f"got dim={dim}"
            )
    tma_input = input.transpose(-2, -1) if input_transposed else input
    M, K = tma_input.shape[-2:]
    B = input.shape[0] if ndim == 3 else 1
    if is_square_scaling and M % 32 != 0:
        raise ValueError(
            f"32x32 scaling requires input axis {row_axis} to be divisible by 32; "
            f"got size {M}"
        )
    if do_dim_m:
        if M % 32 != 0:
            raise ValueError(
                f"quantized axis {row_axis} must be divisible by 32; got size {M}"
            )
        if K % 16 != 0:
            raise ValueError(
                f"input axis {col_axis} must be divisible by 16 when quantizing "
                f"axis {row_axis}; got size {K}"
            )
    if do_dim_k and K % 32 != 0:
        raise ValueError(
            f"quantized axis {col_axis} must be divisible by 32; got size {K}"
        )

    s_num_row_blk_k = s_num_col_blk_k = None
    if do_dim_k:
        s_num_row_blk_k = _ceil_div(M, 128)
        s_num_col_blk_k = _ceil_div(K, 128)

    s_num_row_blk_m = s_num_col_blk_m = None
    if do_dim_m:
        s_num_row_blk_m = _ceil_div(K, 128)
        s_num_col_blk_m = _ceil_div(M, 128)

    if do_dim_k and do_dim_m:
        quant_orientation_id = _QUANT_ORIENTATION_DIM_KM
    elif do_dim_k:
        quant_orientation_id = _QUANT_ORIENTATION_DIM_K
    else:
        quant_orientation_id = _QUANT_ORIENTATION_DIM_M

    return _BlockscaledTmaSpec(
        M=M,
        K=K,
        B=B,
        is_batched=ndim == 3,
        quant_orientation_id=quant_orientation_id,
        do_dim_k=do_dim_k,
        do_dim_m=do_dim_m,
        input_transposed=input_transposed,
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
    if spec.B != 0 and spec.M != 0 and spec.K != 0:
        if spec.B > _CUDA_GRID_Z_MAX:
            raise ValueError(
                f"blockscaled TMA batch size exceeds CUDA grid.z limit: "
                f"B={spec.B}, maximum={_CUDA_GRID_Z_MAX}"
            )
        if spec.M > _CUDA_TMA_MAX_DIM or spec.K > _CUDA_TMA_MAX_DIM:
            raise ValueError(
                "blockscaled TMA requires each nonempty input dimension to be at most "
                f"{_CUDA_TMA_MAX_DIM}; got shape ({spec.M}, {spec.K})"
            )
        plan = select_blockscaled_tma_plan(
            spec.M,
            spec.K,
            input_dtype=input.dtype,
            quant_orientation_id=spec.quant_orientation_id,
            is_square_scaling=spec.is_square_scaling,
        )
    if plan is not None and (
        # K <= 2**32 and tile_k_size >= 32 keep grid_k below the X limit.
        # Large M can still exceed the Y limit.
        plan.grid_k > _CUDA_GRID_X_MAX or plan.grid_m > _CUDA_GRID_Y_MAX
    ):
        raise ValueError(
            "blockscaled TMA launch grid exceeds CUDA limits: "
            f"grid=({plan.grid_k}, {plan.grid_m}, {spec.B}), "
            f"maximum=({_CUDA_GRID_X_MAX}, {_CUDA_GRID_Y_MAX}, {_CUDA_GRID_Z_MAX})"
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
            (spec.B, spec.K, spec.M) if spec.is_batched else (spec.K, spec.M),
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        scale_m = torch.empty(
            spec.B * spec.s_num_row_blk_m * spec.s_num_col_blk_m * 32 * 16,
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
            (spec.B, spec.M, spec.K) if spec.is_batched else (spec.M, spec.K),
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        # Every slot is written by the kernel, so zero-initialization would launch a redundant
        # memset.
        if spec.is_scale_swizzled:
            scale_k = torch.empty(
                spec.B * spec.s_num_row_blk_k * spec.s_num_col_blk_k * 32 * 16,
                dtype=torch.uint8,
                device=input.device,
            )
        else:
            scale_k = torch.empty(
                spec.B * spec.M * (spec.K // 32),
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
        scale_m_shape = (1, spec.s_num_row_blk_m * spec.s_num_col_blk_m * 32 * 16)
        scale_m = scale_m.view(
            (spec.B, *scale_m_shape) if spec.is_batched else scale_m_shape
        ).view(torch.float8_e8m0fnu)
        scale_m = scale_m.transpose(-2, -1)
        output_m = output_m.transpose(-2, -1)
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
            scale_k_shape = (1, spec.s_num_row_blk_k * spec.s_num_col_blk_k * 32 * 16)
        else:
            scale_k_shape = (spec.M, spec.K // 32)
        scale_k = scale_k.view(
            (spec.B, *scale_k_shape) if spec.is_batched else scale_k_shape
        )
        scale_k = scale_k.view(torch.float8_e8m0fnu)

    if spec.input_transposed:
        if output_k is not None:
            output_k = output_k.transpose(-2, -1)
            if scale_k is not None:
                scale_k = scale_k.transpose(-2, -1)
        if output_m is not None:
            output_m = output_m.transpose(-2, -1)
            if scale_m is not None:
                scale_m = scale_m.transpose(-2, -1)

    if spec.do_dim_k and not spec.do_dim_m:
        if output_k is None:
            raise AssertionError("expected output_k, got None")
        if scale_k is None:
            raise AssertionError("expected scale_k, got None")
        return output_k, scale_k
    if spec.do_dim_m and not spec.do_dim_k:
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
    if spec.input_transposed:
        return output_m, scale_m, output_k, scale_k
    return output_k, scale_k, output_m, scale_m


def _blockscaled_tma_impl_on_current_device(
    input: torch.Tensor,
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    *,
    dual: bool,
) -> _BlockscaledTmaOutput:
    spec = _validate_and_normalize_blockscaled_tma(
        input,
        dim=dim,
        dual=dual,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
    )
    tma_input = input.transpose(-2, -1) if spec.input_transposed else input
    launch = _prepare_blockscaled_tma_launch(tma_input, spec)

    # CUDA cannot launch a zero-sized grid. The preparation step still allocates
    # correctly shaped empty outputs so result formatting remains shared.
    if launch.plan is None:
        return _format_blockscaled_tma_output(spec, launch)

    fn = _compile_blockscaled_tma(
        tma_input.dtype,
        launch.plan.tile_m_size,
        launch.plan.tile_k_size,
        launch.plan.cluster_k,
        launch.plan.needs_boundary_masking,
        spec.quant_orientation_id,
        spec.is_square_scaling,
        spec.is_scale_swizzled,
        spec.is_batched,
    )
    output_k, output_m = launch.output_k, launch.output_m
    if spec.is_batched:
        tma_input = tma_input.permute(1, 2, 0)
        output_k = output_k.permute(1, 2, 0) if output_k is not None else None
        output_m = output_m.permute(1, 2, 0) if output_m is not None else None
    fn(
        tma_input,
        output_k,
        launch.scale_k,
        output_m,
        launch.scale_m,
        spec.M,
        spec.K,
        launch.plan.grid_m,
        launch.plan.grid_k,
    )
    return _format_blockscaled_tma_output(spec, launch)


def _run_blockscaled_tma(
    input: torch.Tensor,
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    *,
    dual: bool,
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
    capability = torch.cuda.get_device_capability(device)
    if capability < (10, 0):
        raise RuntimeError(
            "blockscaled TMA requires CUDA capability 10.0 or newer; "
            f"device {input.device} has capability {capability[0]}.{capability[1]}"
        )

    def launch_on_current_device() -> _BlockscaledTmaOutput:
        return _blockscaled_tma_impl_on_current_device(
            input,
            dim=dim,
            is_square_scaling=is_square_scaling,
            is_scale_swizzled=is_scale_swizzled,
            dual=dual,
        )

    if device == torch.cuda.current_device():
        return launch_on_current_device()
    with torch.cuda.device(device):
        return launch_on_current_device()


def _blockscaled_tma_impl(
    input: torch.Tensor,
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    **kwargs: object,
) -> _TwoTensorOutput:
    r"""_blockscaled_tma_impl(input, dim, is_square_scaling, is_scale_swizzled, **kwargs) -> tuple[Tensor, Tensor]

    Quantize a 2D or 3D tensor to MXFP8 with block scaling along ``dim[-1]``.

    Args:
        input (Tensor): Contiguous or transpose-contiguous CUDA tensor of shape
            ``(M, K)`` or ``(B, M, K)`` on an SM100+ GPU, with dtype float16,
            bfloat16, or float32.
        dim (tuple[int, int]): Ordered pair of distinct positive or negative
            axes among the last two dimensions. The first is the row axis;
            the second is the quantized axis and determines the first. For 2D
            input, ``(0, 1)`` or ``(-2, -1)`` quantizes columns, while ``(1, 0)``
            or ``(-1, -2)`` quantizes rows. For 3D input use ``(1, 2)`` or
            ``(2, 1)`` instead of ``(0, 1)`` or ``(1, 0)``. Lists are not accepted.
        is_square_scaling (bool): Use 32x32 instead of 1x32 scale blocks.
            Requires swizzled scales and a contiguous quantized input axis.
        is_scale_swizzled (bool): Return swizzled scales if ``True``; compact
            scales otherwise.
        **kwargs: No additional keyword arguments are accepted.

    Returns:
        tuple[Tensor, Tensor]: ``(qdata, scales)``. ``qdata`` has dtype
            ``torch.float8_e4m3fn`` and the input shape, with matrix strides
            ``(K, 1)`` for the last axis or ``(1, M)`` for the second-to-last.
            ``scales`` have dtype
            ``torch.float8_e8m0fnu`` and, when swizzled, shape ``(1, N)`` for
            the last axis or ``(N, 1)`` for the second-to-last, where
            ``N = ceil(M / 128) * ceil(K / 128) * 512``. Compact scales have
            shape ``(M, K // 32)`` when quantizing the last axis of contiguous
            input or ``(M // 32, K)`` when quantizing the second-to-last axis
            of transpose-contiguous input. For 3D input, prepend ``B`` to the
            scale shape.

    Examples::
        >>> x = torch.zeros((128, 256), dtype=torch.bfloat16, device="cuda")
        >>> qdata, scales = _blockscaled_tma_impl(x, (1, 0), False, True)
        >>> qdata.stride()
        (1, 128)
    """
    return cast(
        _TwoTensorOutput,
        _run_blockscaled_tma(
            input, dim, is_square_scaling, is_scale_swizzled, dual=False, **kwargs
        ),
    )


def _blockscaled_tma_impl_dual(input: torch.Tensor) -> _FourTensorOutput:
    r"""_blockscaled_tma_impl_dual(input) -> tuple[Tensor, Tensor, Tensor, Tensor]

    Quantize both matrix axes of a 2D or 3D tensor in one kernel launch.

    Uses 1x32 scaling blocks and swizzled scales.

    Args:
        input (Tensor): 2D or 3D CUDA tensor with the layout and dtypes
            described by :func:`_blockscaled_tma_impl`. Both matrix dimensions
            must be divisible by 32.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: ``(q_k, s_k, q_m, s_m)``.
            The first pair quantizes the logical last axis (``(-2, -1)``), and
            the second pair quantizes the second-to-last (``(-1, -2)``),
            regardless of input layout. Output strides and scale shapes follow
            :func:`_blockscaled_tma_impl` for each axis.

    Examples::
        >>> x = torch.zeros((128, 256), dtype=torch.bfloat16, device="cuda")
        >>> q_k, s_k, q_m, s_m = _blockscaled_tma_impl_dual(x)
    """
    return cast(
        _FourTensorOutput,
        _run_blockscaled_tma(
            input,
            (-2, -1),
            False,
            True,
            dual=True,
        ),
    )
