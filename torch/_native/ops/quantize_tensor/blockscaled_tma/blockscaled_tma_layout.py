"""Axis and output layout planning for block-scaled TMA quantization."""

from math import prod
from typing import cast, NamedTuple, TypeAlias

import torch

from .blockscaled_tma_plan import BlockscaledTmaPlan, select_blockscaled_tma_plan


_SingleQuantDim: TypeAlias = tuple[int, int]
_QuantDim: TypeAlias = _SingleQuantDim | tuple[_SingleQuantDim, _SingleQuantDim]
_BlockscaledTmaOutput: TypeAlias = (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)


_QUANT_ORIENTATION_DIM_K = 0
_QUANT_ORIENTATION_DIM_M = 1
_QUANT_ORIENTATION_DIM_KM = 2
_CUDA_GRID_X_MAX = 2**31 - 1
_CUDA_GRID_Y_MAX = 2**16 - 1
_CUDA_GRID_Z_MAX = 2**16 - 1
_CUDA_TMA_MAX_DIM = 2**32


def _layout_ceil_div(num: int, den: int) -> int:
    return num // den + (num % den != 0)


class _BlockscaledTmaSpec(NamedTuple):
    """Validated, normalized host-side description of one quantization request."""

    M: int
    K: int
    B_product: int
    is_batched: bool
    quant_orientation_id: int
    do_dim_k: bool
    do_dim_m: bool
    is_square_scaling: bool
    is_scale_swizzled: bool
    s_num_row_blk_k: int | None
    s_num_col_blk_k: int | None
    s_num_row_blk_m: int | None
    s_num_col_blk_m: int | None


def _build_blockscaled_tma_spec(
    input: torch.Tensor,
    *,
    dim: _QuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _BlockscaledTmaSpec:
    r"""Build a launch spec and check kernel limits for canonical 2D/4D input."""

    ndim = input.dim()
    is_batched = ndim == 4
    dim_k = (1, 3) if is_batched else (-2, -1)
    is_dual = isinstance(dim[0], tuple)
    do_dim_k = is_dual or dim == dim_k
    do_dim_m = is_dual or dim == dim_k[::-1]
    row_axis, col_axis = (1, 3) if is_batched else (0, 1)

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

    if is_square_scaling:
        if do_dim_m:
            raise ValueError(
                f"square scaling requires quantization along contiguous input axis {col_axis}; "
                f"got dim={dim}"
            )
    M, K = (input.shape[1], input.shape[3]) if is_batched else input.shape[-2:]
    B_product = input.shape[0] * input.shape[2] if is_batched else 1
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
        s_num_row_blk_k = _layout_ceil_div(M, 128)
        s_num_col_blk_k = _layout_ceil_div(K, 128)

    s_num_row_blk_m = s_num_col_blk_m = None
    if do_dim_m:
        s_num_row_blk_m = _layout_ceil_div(K, 128)
        s_num_col_blk_m = _layout_ceil_div(M, 128)

    if do_dim_k and do_dim_m:
        quant_orientation_id = _QUANT_ORIENTATION_DIM_KM
    elif do_dim_k:
        quant_orientation_id = _QUANT_ORIENTATION_DIM_K
    else:
        quant_orientation_id = _QUANT_ORIENTATION_DIM_M

    return _BlockscaledTmaSpec(
        M=M,
        K=K,
        B_product=B_product,
        is_batched=is_batched,
        quant_orientation_id=quant_orientation_id,
        do_dim_k=do_dim_k,
        do_dim_m=do_dim_m,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
        s_num_row_blk_k=s_num_row_blk_k,
        s_num_col_blk_k=s_num_col_blk_k,
        s_num_row_blk_m=s_num_row_blk_m,
        s_num_col_blk_m=s_num_col_blk_m,
    )


def _checked_blockscaled_tma_plan(
    input: torch.Tensor, spec: _BlockscaledTmaSpec
) -> BlockscaledTmaPlan | None:
    """Enforce launch limits and select a plan for nonempty inputs."""
    plan = None
    if spec.B_product != 0 and spec.M != 0 and spec.K != 0:
        if spec.B_product > _CUDA_GRID_Z_MAX:
            raise ValueError(
                f"blockscaled TMA batch size exceeds CUDA grid.z limit: "
                f"B={spec.B_product}, maximum={_CUDA_GRID_Z_MAX}"
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
            do_dim_k=spec.do_dim_k,
            do_dim_m=spec.do_dim_m,
            is_square_scaling=spec.is_square_scaling,
        )
    if plan is not None and (
        # K <= 2**32 and tile_k_size >= 32 keep grid_k below the X limit.
        # Large M can still exceed the Y limit.
        plan.grid_k > _CUDA_GRID_X_MAX or plan.grid_m > _CUDA_GRID_Y_MAX
    ):
        raise ValueError(
            "blockscaled TMA launch grid exceeds CUDA limits: "
            f"grid=({plan.grid_k}, {plan.grid_m}, {spec.B_product}), "
            f"maximum=({_CUDA_GRID_X_MAX}, {_CUDA_GRID_Y_MAX}, {_CUDA_GRID_Z_MAX})"
        )
    return plan


def normalize_quantize_dim(dim: object, ndim: int) -> _SingleQuantDim:
    if (
        ndim < 2
        or not isinstance(dim, (tuple, list))
        or len(dim) != 2
        or any(type(axis) is not int or axis < -ndim or axis >= ndim for axis in dim)
    ):
        raise ValueError(f"unsupported quantization dim: {dim} for {ndim}D input")
    first, second = (axis % ndim for axis in dim)
    if first == second:
        raise ValueError(f"expected two distinct quantization axes, got {dim}")
    return first, second


class _CanonicalMatrixInput(NamedTuple):
    """Canonical matrix view/copy and metadata for restoring output axes."""

    input: torch.Tensor
    dim: _QuantDim
    batch_shape: tuple[int, ...]
    inverse_permutation: tuple[int, ...]
    swap_dual_outputs: bool


def _canonicalize_matrix_input(
    input: torch.Tensor,
    dim: _QuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _CanonicalMatrixInput:
    r"""Fold batch axes into a 2D or 4D matrix view for the TMA kernel.

    When a trailing batch axis or incompatible strides prevent a TMA view,
    materialize the matrices with batch axes leading. The returned permutation
    restores logical output axes, including packed scale axes.
    """
    ndim = input.ndim
    if ndim < 2:
        raise ValueError(f"blockscaled TMA requires at least 2 dimensions; got {ndim}")
    if not isinstance(dim, tuple) or len(dim) != 2:
        raise ValueError(f"unsupported quantization dim: {dim}")
    pairs: tuple[_SingleQuantDim, ...] = (
        cast(tuple[_SingleQuantDim, _SingleQuantDim], dim)
        if isinstance(dim[0], tuple)
        else (cast(_SingleQuantDim, dim),)
    )
    normalized = [normalize_quantize_dim(pair, ndim) for pair in pairs]
    if len(normalized) == 2 and normalized[1] != normalized[0][::-1]:
        raise ValueError(f"unsupported quantization dim order: {dim}")
    matrix_axes = tuple(sorted(normalized[0]))
    if len(normalized) == 2 and normalized[0] != matrix_axes:
        raise ValueError(f"dual quantization must start with dim-k: {dim}")
    physical_axes = tuple(sorted(range(ndim), key=lambda axis: -input.stride(axis)))
    physical_input = (
        input if physical_axes == tuple(range(ndim)) else input.permute(physical_axes)
    )
    physical_matrix_axes = tuple(axis for axis in physical_axes if axis in matrix_axes)
    fast_axis = physical_matrix_axes[-1]
    quant_axis = normalized[0][-1]
    physical_dim_k = quant_axis == fast_axis
    use_tma_view = (
        physical_input.is_contiguous()
        and physical_axes[-1] == fast_axis
        and (is_scale_swizzled or physical_dim_k)
        and (not is_square_scaling or physical_dim_k)
        and (physical_dim_k or input.shape[fast_axis] % 16 == 0)
    )
    if use_tma_view:
        slow_axis = physical_matrix_axes[0]
        slow_pos = physical_axes.index(slow_axis)
        fast_pos = physical_axes.index(fast_axis)
        outer_batch = prod(input.shape[axis] for axis in physical_axes[:slow_pos])
        inner_batch = prod(
            input.shape[axis] for axis in physical_axes[slow_pos + 1 : fast_pos]
        )
        batch_axes = tuple(axis for axis in physical_axes if axis not in matrix_axes)
        permutation = (*batch_axes, *physical_matrix_axes)
        if batch_axes:
            matrix_input = physical_input.view(
                outer_batch, input.shape[slow_axis], inner_batch, input.shape[fast_axis]
            )
            single_dim: _SingleQuantDim = (1, 3) if physical_dim_k else (3, 1)
        else:
            matrix_input = physical_input
            single_dim = (-2, -1) if physical_dim_k else (-1, -2)
        swap_dual_outputs = (
            len(normalized) == 2 and normalized[0] != physical_matrix_axes
        )
    else:
        batch_axes = tuple(axis for axis in range(ndim) if axis not in matrix_axes)
        permutation = (*batch_axes, *matrix_axes)
        matrix_input = (
            input if permutation == tuple(range(ndim)) else input.permute(permutation)
        ).contiguous()
        if batch_axes:
            matrix_input = matrix_input.view(
                prod(input.shape[axis] for axis in batch_axes),
                matrix_input.shape[-2],
                1,
                matrix_input.shape[-1],
            )
            single_dim = (1, 3) if normalized[0] == matrix_axes else (3, 1)
        else:
            single_dim = (-2, -1) if normalized[0] == matrix_axes else (-1, -2)
        swap_dual_outputs = False
    if len(normalized) == 2:
        canonical_dim: _QuantDim = (
            ((1, 3), (3, 1)) if matrix_input.ndim == 4 else ((-2, -1), (-1, -2))
        )
    else:
        canonical_dim = single_dim
    inverse = tuple(permutation.index(axis) for axis in range(ndim))
    batch_shape = tuple(input.shape[axis] for axis in batch_axes)
    return _CanonicalMatrixInput(
        matrix_input, canonical_dim, batch_shape, inverse, swap_dual_outputs
    )


def _restore_matrix_output(
    output: _BlockscaledTmaOutput, canonical: _CanonicalMatrixInput
) -> _BlockscaledTmaOutput:
    """Unfold batch axes, restore logical axis order, and order dual outputs."""
    if (
        len(canonical.batch_shape) <= 1
        and canonical.inverse_permutation
        == tuple(range(len(canonical.inverse_permutation)))
        and not canonical.swap_dual_outputs
    ):
        return output
    restored = []
    for tensor in output:
        if canonical.batch_shape:
            if tensor.shape[0] == 0:
                # unflatten infers contiguous strides for empty tensors.
                batch_strides = tuple(
                    tensor.stride(0) * prod(canonical.batch_shape[axis + 1 :])
                    for axis in range(len(canonical.batch_shape))
                )
                tensor = tensor.as_strided(
                    (*canonical.batch_shape, *tensor.shape[1:]),
                    (*batch_strides, *tensor.stride()[1:]),
                )
            else:
                tensor = tensor.unflatten(0, canonical.batch_shape)
        restored.append(tensor.permute(canonical.inverse_permutation))
    if canonical.swap_dual_outputs:
        restored = restored[2:] + restored[:2]
    return cast(_BlockscaledTmaOutput, tuple(restored))


def infer_blockscaled_tma_outputs(
    input: torch.Tensor,
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    dual: bool,
) -> list[torch.Tensor]:
    """Mirror kernel output allocation and layout without importing CuTeDSL."""
    ordered = (min(dim), max(dim))
    kernel_dim: _QuantDim = (ordered, (ordered[1], ordered[0])) if dual else dim
    canonical = _canonicalize_matrix_input(
        input, kernel_dim, is_square_scaling, is_scale_swizzled
    )
    spec = _build_blockscaled_tma_spec(
        canonical.input,
        dim=canonical.dim,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
    )
    _checked_blockscaled_tma_plan(canonical.input, spec)
    batch = (spec.B_product,) if spec.is_batched else ()
    result_k = result_m = None
    if spec.do_dim_k:
        if spec.s_num_row_blk_k is None or spec.s_num_col_blk_k is None:
            raise AssertionError("expected dim-k scale tile counts")
        qdata_k = torch.empty(
            (*batch, spec.M, spec.K), device=input.device, dtype=torch.float8_e4m3fn
        )
        scale_shape = (
            (1, spec.s_num_row_blk_k * spec.s_num_col_blk_k * 512)
            if is_scale_swizzled
            else (spec.M, spec.K // 32)
        )
        scale_k = torch.empty(
            (*batch, *scale_shape), device=input.device, dtype=torch.float8_e8m0fnu
        )
        result_k = (qdata_k, scale_k)
    if spec.do_dim_m:
        if spec.s_num_row_blk_m is None or spec.s_num_col_blk_m is None:
            raise AssertionError("expected dim-m scale tile counts")
        qdata_m = torch.empty(
            (*batch, spec.K, spec.M), device=input.device, dtype=torch.float8_e4m3fn
        ).transpose(-2, -1)
        scale_m = torch.empty(
            (*batch, 1, spec.s_num_row_blk_m * spec.s_num_col_blk_m * 512),
            device=input.device,
            dtype=torch.float8_e8m0fnu,
        ).transpose(-2, -1)
        result_m = (qdata_m, scale_m)
    if dual:
        if result_k is None or result_m is None:
            raise AssertionError("dual quantization requires both axis outputs")
        outputs = (*result_k, *result_m)
    else:
        outputs = result_k if result_k is not None else result_m
        if outputs is None:
            raise AssertionError("quantization requires an axis output")
    restored = _restore_matrix_output(outputs, canonical)
    if dual and dim != ordered:
        restored = (*restored[2:], *restored[:2])
    return list(restored)
