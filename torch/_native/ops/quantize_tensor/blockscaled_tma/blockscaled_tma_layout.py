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


class _MatrixInputLayout(NamedTuple):
    r"""Planned canonical shape, axis order, and output restoration metadata.

    ``matrix_axes`` are in caller coordinates and ordered as the kernel's M/K
    axes. ``input_permutation`` keeps interleaved batches in place for a TMA
    view, but moves batches ahead of the matrix axes for a copy.
    """

    shape: tuple[int, ...]
    dim: _SingleQuantDim
    matrix_axes: _SingleQuantDim
    input_permutation: tuple[int, ...]
    use_tma_view: bool
    batch_shape: tuple[int, ...]
    inverse_permutation: tuple[int, ...]
    swap_dual_outputs: bool


class _CanonicalMatrixInput(NamedTuple):
    r"""TMA input and metadata for returning results in the caller's axis order.

    A view-compatible batched input keeps batch axes around the matrix axes in
    ``(B_outer, M, B_inner, K)`` (or with ``M`` and ``K`` reversed). When a view
    is unavailable, a batched ``[*B0, M, *B1, K, *B2]`` input moves its batch
    axes ahead of ``(M, K)`` in physical stride order and folds them into
    ``(B_product, M, 1, K)`` after materialization.

    Attributes:
        input (Tensor): Canonical 2D or 4D matrix input, either a view or a copy.
        layout (_MatrixInputLayout): Metadata for restoring the input's logical
          axis order after quantization.
    """

    input: torch.Tensor
    layout: _MatrixInputLayout


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


def _plan_blockscaled_tma_launch(
    input: torch.Tensor,
    layout: _MatrixInputLayout,
    spec: _BlockscaledTmaSpec,
) -> BlockscaledTmaPlan | None:
    r"""Validate the selected TMA launch using caller axes before copying input."""

    if input.numel() == 0:
        return None

    caller_batch_axes = tuple(
        axis for axis in range(input.ndim) if axis not in layout.matrix_axes
    )
    batch_sizes = tuple(input.shape[axis] for axis in caller_batch_axes)
    batch_count = spec.B_product
    if batch_count > _CUDA_GRID_Z_MAX:
        raise ValueError(
            f"blockscaled TMA batch axes {caller_batch_axes} have sizes {batch_sizes}; "
            f"product {batch_count} exceeds CUDA grid.z limit {_CUDA_GRID_Z_MAX}"
        )
    for axis in layout.matrix_axes:
        size = input.shape[axis]
        if size > _CUDA_TMA_MAX_DIM:
            raise ValueError(
                f"blockscaled TMA input axis {axis} has size {size}; "
                f"maximum TMA dimension size is {_CUDA_TMA_MAX_DIM}"
            )

    plan = select_blockscaled_tma_plan(
        spec.M,
        spec.K,
        input_dtype=input.dtype,
        quant_orientation_id=spec.quant_orientation_id,
        is_square_scaling=spec.is_square_scaling,
    )
    if plan.grid_k > _CUDA_GRID_X_MAX or plan.grid_m > _CUDA_GRID_Y_MAX:
        raise ValueError(
            f"blockscaled TMA launch grid exceeds CUDA limits for input matrix axes "
            f"{layout.matrix_axes} with sizes ({spec.M}, {spec.K}): "
            f"grid=({plan.grid_k}, {plan.grid_m}, {batch_count}), "
            f"maximum=({_CUDA_GRID_X_MAX}, {_CUDA_GRID_Y_MAX}, {_CUDA_GRID_Z_MAX})"
        )
    return plan


def _plan_matrix_input(
    input: torch.Tensor,
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    *,
    dual: bool,
) -> _MatrixInputLayout:
    r"""Plan matrix and batch axes for the 2D or 4D TMA kernel without copying.

    Choose a 2D or 4D TMA view when the physical strides and recipe permit. For
    batched input ``[*B0, M, *B1, K, *B2]``, a 4D view has shape
    ``(B_outer, M, B_inner, K)`` (or with ``M`` and ``K`` reversed), preserving
    batch axes around the matrix axes. Otherwise, plan a contiguous copy with
    batch axes ahead of the matrix axes in physical stride order. Batched
    inputs then fold those axes into ``(B_product, M, 1, K)``; 2D inputs
    remain 2D. Trailing ``*B2`` axes can prevent a view. For single
    quantization, the copy places the requested quantized axis last (dim-k);
    dual quantization keeps both matrix orientations. Return canonical shape,
    quantization axes and metadata to undo the permutation.

    Args:
        input (Tensor): Input whose two matrix axes are specified by ``dim``.
        dim (_SingleQuantDim): Ordered matrix-axis pair. Dual quantization also
          quantizes the reverse pair.
        is_square_scaling (bool): Whether square-block scaling is requested.
        is_scale_swizzled (bool): Whether swizzled scale storage is requested.
        dual (bool): Whether to quantize both matrix axes.
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
        slow_size, fast_size = input.shape[slow_axis], input.shape[fast_axis]
        if batch_axes:
            matrix_shape = (outer_batch, slow_size, inner_batch, fast_size)
            single_dim: _SingleQuantDim = (1, 3) if physical_dim_k else (3, 1)
        else:
            matrix_shape = (slow_size, fast_size)
            single_dim = (-2, -1) if physical_dim_k else (-1, -2)
        input_permutation = physical_axes
        kernel_axes = (physical_matrix_axes[0], physical_matrix_axes[1])
        swap_dual_outputs = dual and normalized != physical_matrix_axes
    else:
        # Materialization fallback, place the requested quantized axis last
        # for the dim-k kernel.
        permutation = (*batch_axes, *normalized)
        if batch_axes:
            matrix_shape = (
                prod(input.shape[axis] for axis in batch_axes),
                input.shape[normalized[0]],
                1,
                input.shape[normalized[1]],
            )
            single_dim = (1, 3)
        else:
            matrix_shape = (input.shape[normalized[0]], input.shape[normalized[1]])
            single_dim = (-2, -1)
        input_permutation = permutation
        kernel_axes = normalized
        swap_dual_outputs = False
    if dual:
        canonical_dim: _SingleQuantDim = (1, 3) if batch_axes else (-2, -1)
    else:
        canonical_dim = single_dim
    inverse = tuple(permutation.index(axis) for axis in range(ndim))
    batch_shape = tuple(input.shape[axis] for axis in batch_axes)
    return _MatrixInputLayout(
        matrix_shape,
        canonical_dim,
        kernel_axes,
        input_permutation,
        use_tma_view,
        batch_shape,
        inverse,
        swap_dual_outputs,
    )


def _canonicalize_matrix_input(
    input: torch.Tensor, layout: _MatrixInputLayout
) -> _CanonicalMatrixInput:
    r"""Apply a planned TMA view or copy to the input."""

    matrix_input = (
        input
        if layout.input_permutation == tuple(range(input.ndim))
        else input.permute(layout.input_permutation)
    )
    if layout.use_tma_view:
        if layout.batch_shape:
            matrix_input = matrix_input.view(layout.shape)
    else:
        matrix_input = matrix_input.contiguous()
        if layout.batch_shape:
            matrix_input = matrix_input.view(layout.shape)
    return _CanonicalMatrixInput(matrix_input, layout)


def _restore_matrix_output(
    output: _BlockscaledTmaOutput, layout: _MatrixInputLayout
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
        layout (_MatrixInputLayout): Axis order and batch shape to restore.
    """

    if (
        len(layout.batch_shape) <= 1
        and layout.inverse_permutation == tuple(range(len(layout.inverse_permutation)))
        and not layout.swap_dual_outputs
    ):
        return output
    restored = []
    for tensor in output:
        if layout.batch_shape:
            if tensor.shape[0] == 0:
                # unflatten infers contiguous strides for empty tensors.
                batch_strides = tuple(
                    tensor.stride(0) * prod(layout.batch_shape[axis + 1 :])
                    for axis in range(len(layout.batch_shape))
                )
                tensor = tensor.as_strided(
                    (*layout.batch_shape, *tensor.shape[1:]),
                    (*batch_strides, *tensor.stride()[1:]),
                )
            else:
                tensor = tensor.unflatten(0, layout.batch_shape)
        restored.append(tensor.permute(layout.inverse_permutation))
    if layout.swap_dual_outputs:
        restored = restored[2:] + restored[:2]
    return cast(_BlockscaledTmaOutput, tuple(restored))


def _build_blockscaled_tma_spec(
    layout: _MatrixInputLayout,
    *,
    dual: bool,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _BlockscaledTmaSpec:
    r"""Build a launch spec from the planned canonical 2D/4D layout."""

    is_batched = len(layout.shape) == 4
    dim_k = (1, 3) if is_batched else (-2, -1)
    do_dim_k = dual or layout.dim == dim_k
    do_dim_m = dual or layout.dim == dim_k[::-1]
    M, K = (layout.shape[1], layout.shape[3]) if is_batched else layout.shape
    B_product = layout.shape[0] * layout.shape[2] if is_batched else 1

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
    layout = _plan_matrix_input(
        input, kernel_dim, is_square_scaling, is_scale_swizzled, dual=dual
    )
    spec = _build_blockscaled_tma_spec(
        layout,
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
    restored = _restore_matrix_output(outputs, layout)
    if dual and dim != ordered:
        restored = (*restored[2:], *restored[:2])
    return list(restored)
