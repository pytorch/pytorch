"""PyTorch-facing implementation for TMA-based block-scaled quantization."""

from typing import cast, NamedTuple, TypeAlias

import torch

from ..utils import _device_capability
from .blockscaled_tma_kernels import _compile_blockscaled_tma
from .blockscaled_tma_layout import (
    _BlockscaledTmaSpec,
    _build_blockscaled_tma_spec,
    _canonicalize_matrix_input,
    _checked_blockscaled_tma_plan,
    _QuantDim,
    _restore_matrix_output,
    _SingleQuantDim,
)
from .blockscaled_tma_plan import BlockscaledTmaPlan


_INPUT_ALIGNMENT_BYTES = 16

_TwoTensorOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor]
_FourTensorOutput: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_BlockscaledTmaOutput: TypeAlias = _TwoTensorOutput | _FourTensorOutput


class _PreparedBlockscaledTmaLaunch(NamedTuple):
    """Host-side launch plan, allocated outputs, and normalized optional operands."""

    plan: BlockscaledTmaPlan | None
    output_k: torch.Tensor | None
    scale_k: torch.Tensor | None
    output_m: torch.Tensor | None
    scale_m: torch.Tensor | None


def _prepare_blockscaled_tma_launch(
    input: torch.Tensor,
    spec: _BlockscaledTmaSpec,
) -> _PreparedBlockscaledTmaLaunch:
    """Select the launch plan and allocate its output storage."""

    plan = _checked_blockscaled_tma_plan(input, spec)

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
            (spec.B_product, spec.K, spec.M) if spec.is_batched else (spec.K, spec.M),
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        scale_m = torch.empty(
            spec.B_product * spec.s_num_row_blk_m * spec.s_num_col_blk_m * 32 * 16,
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
            (spec.B_product, spec.M, spec.K) if spec.is_batched else (spec.M, spec.K),
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        # Every slot is written by the kernel, so zero-initialization would launch a redundant
        # memset.
        if spec.is_scale_swizzled:
            scale_k = torch.empty(
                spec.B_product * spec.s_num_row_blk_k * spec.s_num_col_blk_k * 32 * 16,
                dtype=torch.uint8,
                device=input.device,
            )
        else:
            scale_k = torch.empty(
                spec.B_product * spec.M * (spec.K // 32),
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
            (spec.B_product, *scale_m_shape) if spec.is_batched else scale_m_shape
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
            (spec.B_product, *scale_k_shape) if spec.is_batched else scale_k_shape
        )
        scale_k = scale_k.view(torch.float8_e8m0fnu)

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
    return output_k, scale_k, output_m, scale_m


def _blockscaled_tma_impl_on_current_device(
    input: torch.Tensor,
    dim: _QuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _BlockscaledTmaOutput:
    canonical = _canonicalize_matrix_input(
        input, dim, is_square_scaling, is_scale_swizzled
    )
    input, dim = canonical.input, canonical.dim
    if input.data_ptr() % _INPUT_ALIGNMENT_BYTES:
        input = input.clone()
    spec = _build_blockscaled_tma_spec(
        input,
        dim=dim,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
    )
    launch = _prepare_blockscaled_tma_launch(input, spec)

    # CUDA cannot launch a zero-sized grid. The preparation step still allocates
    # correctly shaped empty outputs so result formatting remains shared.
    if launch.plan is None:
        return _restore_matrix_output(
            _format_blockscaled_tma_output(spec, launch), canonical
        )

    fn = _compile_blockscaled_tma(
        input.dtype,
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
    tma_input = input
    if spec.is_batched:
        B0, _, B1, _ = input.shape
        tma_input = input.permute(1, 3, 2, 0)
        output_k = (
            output_k.view(B0, B1, spec.M, spec.K).permute(2, 3, 1, 0)
            if output_k is not None
            else None
        )
        output_m = (
            output_m.view(B0, B1, spec.K, spec.M).permute(2, 3, 1, 0)
            if output_m is not None
            else None
        )
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
    return _restore_matrix_output(
        _format_blockscaled_tma_output(spec, launch), canonical
    )


def _run_blockscaled_tma(
    input: torch.Tensor,
    dim: _QuantDim,
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
    capability = _device_capability(device)
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

    Quantize a tensor to MXFP8 with block scaling along ``dim[-1]``.

    Args:
        input (Tensor): CUDA tensor with at least two dimensions on an SM100+
            GPU, with dtype float16, bfloat16, or float32. Layouts unsuitable
            for a direct TMA load are materialized.
        dim (tuple[int, int]): Ordered pair of distinct positive or negative
            axes. The first is the row axis; the second is the quantized axis.
            All other axes are batch axes. For 2D input, ``(0, 1)`` or
            ``(-2, -1)`` quantizes columns, while ``(1, 0)`` or ``(-1, -2)``
            quantizes rows. Lists are not accepted.
        is_square_scaling (bool): Use 32x32 instead of 1x32 scale blocks.
            Requires swizzled scales and a dim-k kernel layout.
        is_scale_swizzled (bool): Return swizzled scales if ``True``; compact
            scales otherwise. Compact scales require a dim-k kernel layout.
        **kwargs: No additional keyword arguments are accepted.

    Returns:
        tuple[Tensor, Tensor]: ``(qdata, scales)``. ``qdata`` has dtype
            ``torch.float8_e4m3fn`` and the input shape, but need not retain
            its strides. For 2D input of shape ``(M, K)``, qdata strides are
            ``(K, 1)`` for axis 1 or ``(1, M)`` for axis 0. ``scales`` have
            dtype ``torch.float8_e8m0fnu`` and, when swizzled, shape
            ``(1, N)`` for axis 1 or ``(N, 1)`` for axis 0, where
            ``N = ceil(M / 128) * ceil(K / 128) * 512``. When supported,
            compact scales have shape ``(M, K // 32)`` for axis 1 or
            ``(M // 32, K)`` for axis 0. For higher-rank input, batch axes
            retain their positions in both outputs.

    Examples::
        >>> x = torch.zeros((128, 256), dtype=torch.bfloat16, device="cuda")
        >>> qdata, scales = _blockscaled_tma_impl(x, (1, 0), False, True)
        >>> qdata.stride()
        (1, 128)
    """
    if isinstance(dim, tuple) and len(dim) == 2 and isinstance(dim[0], tuple):
        raise ValueError("dual quantization dims require the dual entrypoint")
    return cast(
        _TwoTensorOutput,
        _run_blockscaled_tma(
            input, dim, is_square_scaling, is_scale_swizzled, **kwargs
        ),
    )


def _blockscaled_tma_impl_dual(
    input: torch.Tensor,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    *,
    dim: tuple[_SingleQuantDim, _SingleQuantDim] = ((-2, -1), (-1, -2)),
    **kwargs: object,
) -> _FourTensorOutput:
    r"""_blockscaled_tma_impl_dual(input, is_square_scaling, is_scale_swizzled, *, dim=((-2, -1), (-1, -2)), **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]

    Quantize both matrix axes of a tensor in one kernel launch.

    Args:
        input (Tensor): CUDA tensor with the layout and dtypes described by
            :func:`_blockscaled_tma_impl`. Both matrix axes must be divisible by 32.
        is_square_scaling (bool): Must be ``False``.
        is_scale_swizzled (bool): Must be ``True``.
        dim (tuple[tuple[int, int], tuple[int, int]], optional): Ordered axis
            pair followed by its reverse. The first pair must list its axes
            in ascending order. Default: ``((-2, -1), (-1, -2))``.
        **kwargs: No additional keyword arguments are accepted.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: ``(q_k, s_k, q_m, s_m)``.
            The first pair quantizes ``dim[0][-1]`` and the second quantizes
            ``dim[1][-1]``, regardless of input layout. With the default
            ``dim``, the logical last axis is first. Output strides and scale
            shapes follow :func:`_blockscaled_tma_impl` for each axis.

    Examples::
        >>> x = torch.zeros((128, 256), dtype=torch.bfloat16, device="cuda")
        >>> q_k, s_k, q_m, s_m = _blockscaled_tma_impl_dual(x, False, True)
    """
    return cast(
        _FourTensorOutput,
        _run_blockscaled_tma(
            input,
            dim,
            is_square_scaling,
            is_scale_swizzled,
            **kwargs,
        ),
    )
