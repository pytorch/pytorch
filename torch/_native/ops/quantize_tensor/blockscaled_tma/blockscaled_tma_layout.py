"""Axis and output layout planning for block-scaled TMA quantization."""

from math import prod
from typing import cast, NamedTuple, TypeAlias

import torch

from .blockscaled_tma_plan import (
    _QUANT_ORIENTATION_DIM_K,
    _QUANT_ORIENTATION_DIM_KM,
    _QUANT_ORIENTATION_DIM_M,
    BlockscaledTmaPlan,
    select_blockscaled_tma_plan,
)


_SingleQuantDim: TypeAlias = tuple[int, int]
_BlockscaledTmaOutput: TypeAlias = (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)


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
    dim: _SingleQuantDim,
    dual: bool,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _BlockscaledTmaSpec:
    r"""Build a launch spec from validated canonical 2D/4D input."""

    ndim = input.dim()
    is_batched = ndim == 4
    dim_k = (1, 3) if is_batched else (-2, -1)
    do_dim_k = dual or dim == dim_k
    do_dim_m = dual or dim == dim_k[::-1]
    M, K = (input.shape[1], input.shape[3]) if is_batched else input.shape[-2:]
    B_product = input.shape[0] * input.shape[2] if is_batched else 1

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


def _plan_blockscaled_tma_launch(
    input: torch.Tensor,
    batch_axes: tuple[int, ...],
    matrix_axes: _SingleQuantDim,
    quant_orientation_id: int,
    is_square_scaling: bool,
) -> BlockscaledTmaPlan | None:
    r"""Validate the selected TMA launch using caller axes before copying input."""

    if input.numel() == 0:
        return None

    caller_batch_axes = tuple(sorted(batch_axes))
    batch_sizes = tuple(input.shape[axis] for axis in caller_batch_axes)
    batch_count = prod(batch_sizes)
    if batch_count > _CUDA_GRID_Z_MAX:
        raise ValueError(
            f"blockscaled TMA batch axes {caller_batch_axes} have sizes {batch_sizes}; "
            f"product {batch_count} exceeds CUDA grid.z limit {_CUDA_GRID_Z_MAX}"
        )
    for axis in matrix_axes:
        size = input.shape[axis]
        if size > _CUDA_TMA_MAX_DIM:
            raise ValueError(
                f"blockscaled TMA input axis {axis} has size {size}; "
                f"maximum TMA dimension size is {_CUDA_TMA_MAX_DIM}"
            )

    M, K = (input.shape[axis] for axis in matrix_axes)
    plan = select_blockscaled_tma_plan(
        M,
        K,
        input_dtype=input.dtype,
        quant_orientation_id=quant_orientation_id,
        is_square_scaling=is_square_scaling,
    )
    if plan.grid_k > _CUDA_GRID_X_MAX or plan.grid_m > _CUDA_GRID_Y_MAX:
        raise ValueError(
            f"blockscaled TMA launch grid exceeds CUDA limits for input matrix axes "
            f"{matrix_axes} with sizes ({M}, {K}): "
            f"grid=({plan.grid_k}, {plan.grid_m}, {batch_count}), "
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
    r"""TMA input and metadata for returning results in the caller's axis order.

    A view-compatible batched input keeps batch axes around the matrix axes in
    ``(B_outer, M, B_inner, K)`` (or with ``M`` and ``K`` reversed). When a view
    is unavailable, a batched ``[*B0, M, *B1, K, *B2]`` input moves its batch
    axes ahead of ``(M, K)`` in physical stride order and folds them into
    ``(B_product, M, 1, K)`` after materialization.

    Attributes:
        input (Tensor): Canonical 2D or 4D matrix input, either a view or a copy.
        dim (_SingleQuantDim): Quantization axis pair in ``input`` coordinates;
          ``(1, 3)`` quantizes the last matrix axis of a 4D input, while
          ``(3, 1)`` quantizes the first.
        batch_shape (tuple[int, ...]): Sizes of batch axes in physical stride
          order, used to unfold the output's batch dimension.
        inverse_permutation (tuple[int, ...]): Permutation that restores the
          original logical axis order after the batch axes are unfolded.
        swap_dual_outputs (bool): Whether the physical matrix-axis order
          reversed the two dual-quantization result pairs.
        plan (BlockscaledTmaPlan, optional): Launch plan validated before any
          input materialization; ``None`` for empty input or meta inference.
    """

    input: torch.Tensor
    dim: _SingleQuantDim
    batch_shape: tuple[int, ...]
    inverse_permutation: tuple[int, ...]
    swap_dual_outputs: bool
    plan: BlockscaledTmaPlan | None


def _validate_blockscaled_tma_input(
    input: torch.Tensor,
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    *,
    dual: bool,
) -> None:
    r"""Validate caller-visible input metadata before materializing the tensor."""

    if not is_scale_swizzled and is_square_scaling:
        raise ValueError("compact scales do not support square scaling")

    if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("blockscaled TMA supports only bf16, fp16, and fp32 input")

    row_axis, quant_axis = dim
    row_size, quant_size = input.shape[row_axis], input.shape[quant_axis]
    if is_square_scaling and row_size % 32 != 0:
        raise ValueError(
            f"32x32 scaling requires input axis {row_axis} to be divisible by 32; "
            f"got size {row_size}"
        )
    if dual and row_size % 32 != 0:
        raise ValueError(
            f"quantized axis {row_axis} must be divisible by 32; got size {row_size}"
        )
    if quant_size % 32 != 0:
        raise ValueError(
            f"quantized axis {quant_axis} must be divisible by 32; got size {quant_size}"
        )


def _canonicalize_matrix_input(
    input: torch.Tensor,
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    *,
    dual: bool,
    check_launch_limits: bool = False,
) -> _CanonicalMatrixInput:
    r"""Arrange matrix and batch axes for the 2D or 4D TMA kernel.

    Use a 2D or 4D TMA view when the physical strides and recipe permit. For
    batched input ``[*B0, M, *B1, K, *B2]``, a 4D view has shape
    ``(B_outer, M, B_inner, K)`` (or with ``M`` and ``K`` reversed), preserving
    batch axes around the matrix axes. Otherwise, move batch axes ahead of the
    matrix axes in physical stride order and materialize contiguous storage.
    Batched inputs then fold those axes into ``(B_product, M, 1, K)``; 2D
    inputs remain 2D. Trailing ``*B2`` axes can prevent a view. For single
    quantization, materialization places the requested quantized axis last
    (dim-k); dual quantization keeps both matrix orientations. Return
    normalized axes and metadata to undo the permutation.

    Args:
        input (Tensor): Input whose two matrix axes are specified by ``dim``.
        dim (_SingleQuantDim): Ordered matrix-axis pair. Dual quantization also
          quantizes the reverse pair.
        is_square_scaling (bool): Whether square-block scaling is requested.
        is_scale_swizzled (bool): Whether swizzled scale storage is requested.
        dual (bool): Whether to quantize both matrix axes.
        check_launch_limits (bool): Check CUDA launch limits before copying for
          eager execution; meta inference does not impose device launch limits.
    """

    ndim = input.ndim
    if ndim < 2:
        raise ValueError(f"blockscaled TMA requires at least 2 dimensions; got {ndim}")
    if not isinstance(dim, tuple) or len(dim) != 2:
        raise ValueError(f"unsupported quantization dim: {dim}")
    first, last = dim
    if type(first) is not int or type(last) is not int:
        raise ValueError(f"unsupported quantization dim: {dim}")
    if first < -ndim or first >= ndim or last < -ndim or last >= ndim:
        raise ValueError(f"unsupported quantization dim: {dim}")
    normalized = (first % ndim, last % ndim)
    if normalized[0] == normalized[1]:
        raise ValueError(f"expected two distinct quantization axes, got {dim}")
    matrix_axes = tuple(sorted(normalized))
    _validate_blockscaled_tma_input(
        input, normalized, is_square_scaling, is_scale_swizzled, dual=dual
    )
    physical_axes = tuple(sorted(range(ndim), key=lambda axis: -input.stride(axis)))
    physical_matrix_axes = tuple(axis for axis in physical_axes if axis in matrix_axes)
    fast_axis = physical_matrix_axes[-1]
    fast_pos = physical_axes.index(fast_axis)
    trailing_batch_axes = physical_axes[fast_pos + 1 :]
    if trailing_batch_axes and all(
        input.shape[axis] == 1 for axis in trailing_batch_axes
    ):
        # Move trailing singletons before the fast matrix axis without
        # changing their order relative to the other batch axes. This lets
        # a shape such as (M, K, B) with B == 1 take the fast path.
        before_fast = physical_axes[:fast_pos]
        insert_at = 1 + max(
            (pos for pos, axis in enumerate(before_fast) if axis not in matrix_axes),
            default=-1,
        )
        physical_axes = (
            *before_fast[:insert_at],
            *trailing_batch_axes,
            *before_fast[insert_at:],
            fast_axis,
        )
    physical_input = (
        input if physical_axes == tuple(range(ndim)) else input.permute(physical_axes)
    )
    batch_axes = tuple(axis for axis in physical_axes if axis not in matrix_axes)
    quant_axis = normalized[-1]
    physical_dim_k = quant_axis == fast_axis

    # Determine if we can take a no-copy fast path (unaligned pointer can still
    # copy later)
    use_tma_view = (
        # Physical input must be contiguous and innermost dim must be one of the
        # requested quantization dims.
        physical_input.is_contiguous()
        and physical_axes[-1] == fast_axis
        # Unswizzled and square scales only support kernel dim-k. If this
        # view requires dim-m, single quantization materializes dim-k.
        and (is_scale_swizzled or physical_dim_k)
        and (not is_square_scaling or physical_dim_k)
        # Dim-m kernel currently requires K % 16 == 0
        and (physical_dim_k or input.shape[fast_axis] % 16 == 0)
    )
    plan = None
    if check_launch_limits:
        kernel_axes = (
            (physical_matrix_axes[0], physical_matrix_axes[1])
            if use_tma_view
            else normalized
        )
        if dual:
            orientation = _QUANT_ORIENTATION_DIM_KM
        elif use_tma_view and not physical_dim_k:
            orientation = _QUANT_ORIENTATION_DIM_M
        else:
            orientation = _QUANT_ORIENTATION_DIM_K
        plan = _plan_blockscaled_tma_launch(
            input, batch_axes, kernel_axes, orientation, is_square_scaling
        )

    if use_tma_view:
        # No-copy fast path, TMA tiles the two matrix axes; batch axes between
        # them become slice coordinates.
        slow_axis = physical_matrix_axes[0]
        slow_pos = physical_axes.index(slow_axis)
        fast_pos = physical_axes.index(fast_axis)
        outer_batch = prod(input.shape[axis] for axis in physical_axes[:slow_pos])
        inner_batch = prod(
            input.shape[axis] for axis in physical_axes[slow_pos + 1 : fast_pos]
        )
        permutation = (*batch_axes, *physical_matrix_axes)
        if batch_axes:
            matrix_input = physical_input.view(
                outer_batch, input.shape[slow_axis], inner_batch, input.shape[fast_axis]
            )
            single_dim: _SingleQuantDim = (1, 3) if physical_dim_k else (3, 1)
        else:
            matrix_input = physical_input
            single_dim = (-2, -1) if physical_dim_k else (-1, -2)
        swap_dual_outputs = dual and normalized != physical_matrix_axes
    else:
        # Materialization fallback, place the requested quantized axis last
        # for the dim-k kernel.
        permutation = (*batch_axes, *normalized)
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
            single_dim = (1, 3)
        else:
            single_dim = (-2, -1)
        swap_dual_outputs = False
    if dual:
        canonical_dim: _SingleQuantDim = (1, 3) if matrix_input.ndim == 4 else (-2, -1)
    else:
        canonical_dim = single_dim
    inverse = tuple(permutation.index(axis) for axis in range(ndim))
    batch_shape = tuple(input.shape[axis] for axis in batch_axes)
    return _CanonicalMatrixInput(
        matrix_input,
        canonical_dim,
        batch_shape,
        inverse,
        swap_dual_outputs,
        plan,
    )


def _restore_matrix_output(
    output: _BlockscaledTmaOutput, canonical: _CanonicalMatrixInput
) -> _BlockscaledTmaOutput:
    r"""Restore result axes after quantizing the canonical matrix input.

    For contiguous ``[*B0, M, *B1, K, *B2]``, unfold each qdata result's
    leading ``B_product`` axis from ``(B_product, M, K)`` to
    ``[*B0, *B1, *B2, M, K]``, then permute it to ``[*B0, M, *B1, K, *B2]``.
    Scale tensors undergo the same axis permutation, but their last two axes
    hold scale data, not necessarily ``M`` and ``K`` elements; swizzled scales
    use packed storage. Swap the dual result pairs if the physical matrix axes
    changed their order.
    This restores logical axes, not the input's strides.

    Args:
        output (_BlockscaledTmaOutput): Results with folded batch dimensions.
        canonical (_CanonicalMatrixInput): Axis order and batch shape to restore.
    """

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
    kernel_dim = ordered if dual else dim
    canonical = _canonicalize_matrix_input(
        input, kernel_dim, is_square_scaling, is_scale_swizzled, dual=dual
    )
    spec = _build_blockscaled_tma_spec(
        canonical.input,
        dim=canonical.dim,
        dual=dual,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
    )
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
