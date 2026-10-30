"""PyTorch-facing implementation for TMA-based block-scaled quantization."""

from math import prod
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


class _PreparedBlockscaledTmaLaunch(NamedTuple):
    """Host-side launch plan, allocated outputs, and normalized optional operands."""

    plan: BlockscaledTmaPlan | None
    output_k: torch.Tensor | None
    scale_k: torch.Tensor | None
    output_m: torch.Tensor | None
    scale_m: torch.Tensor | None


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
    r"""Apply a planned TMA view or copy and align its input pointer."""

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
    if matrix_input.data_ptr() % _INPUT_ALIGNMENT_BYTES:
        matrix_input = matrix_input.clone()
    return _CanonicalMatrixInput(matrix_input, layout)


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
        canonical (_CanonicalMatrixInput): Input and layout metadata to restore.
    """

    layout = canonical.layout
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


def _prepare_blockscaled_tma_launch(
    input: torch.Tensor,
    spec: _BlockscaledTmaSpec,
    plan: BlockscaledTmaPlan | None,
) -> _PreparedBlockscaledTmaLaunch:
    r"""Allocate output storage for a prevalidated launch plan."""

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
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    *,
    dual: bool,
) -> _BlockscaledTmaOutput:
    layout = _plan_matrix_input(
        input, dim, is_square_scaling, is_scale_swizzled, dual=dual
    )
    spec = _build_blockscaled_tma_spec(
        layout,
        dual=dual,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
    )
    plan = _plan_blockscaled_tma_launch(input, layout, spec)
    canonical = _canonicalize_matrix_input(input, layout)
    input = canonical.input
    launch = _prepare_blockscaled_tma_launch(input, spec, plan)

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

    Quantize an N-D tensor to MXFP8 with block scaling along ``dim[-1]``.

    ``dim`` defines a logical 2D matrix plane; all other axes are batch axes.
    Layouts admitting a 2D or 4D TMA view can avoid copying. Other layouts,
    including trailing batch axes that prevent such a view, are made
    contiguous. For otherwise view-compatible inputs, compact or square
    scaling copies when ``dim[-1]`` is the physically slower matrix axis.
    Swizzled scaling also copies in that orientation if the faster matrix
    axis has size not divisible by 16. Canonical storage with an unaligned
    pointer is cloned to meet the 16-byte alignment requirement.

    .. note::
        These copy conditions describe the current kernel. Whether an input
        uses a view or is materialized is an implementation detail, not a
        public API guarantee. The output properties below apply to either path.

    Args:
        input (Tensor): CUDA tensor with at least two dimensions on an SM100+
            GPU, with dtype float16, bfloat16, or float32.
        dim (tuple[int, int]): Ordered pair of distinct positive or negative
            axes, outer then inner in the logical matrix plane. The second
            selects the block-scaled axis and must have size divisible by 32.
            For 2D input, ``(0, 1)`` or ``(-2, -1)`` quantizes columns, while
            ``(1, 0)`` or ``(-1, -2)`` quantizes rows. Lists are not accepted.
        is_square_scaling (bool): Use 32x32 instead of 1x32 scale blocks.
            Requires swizzled scales and both selected axes divisible by 32.
        is_scale_swizzled (bool): Return swizzled scales if ``True``; compact
            scales otherwise.
        **kwargs: No additional keyword arguments are accepted.

    Returns:
        tuple[Tensor, Tensor]: ``(qdata, scales)``. ``qdata`` has dtype
            ``torch.float8_e4m3fn`` and the input's logical shape and axis
            order, but need not retain its strides. For 2D input of shape
            ``(M, K)``, qdata strides are ``(K, 1)`` for ``dim=(0, 1)`` or
            ``(1, M)`` for ``dim=(1, 0)``. ``scales`` have dtype
            ``torch.float8_e8m0fnu`` and, when swizzled, shape ``(1, N)`` or
            ``(N, 1)``, respectively, where
            ``N = ceil(M / 128) * ceil(K / 128) * 512``. When supported,
            compact scales have shape ``(M, K // 32)`` or ``(M // 32, K)``.
            Batch axes retain their positions in both outputs: for input
            ``(B0, M, B1, K, B2)`` and ``dim=(1, 3)``, swizzled scales have
            shape ``(B0, 1, B1, N, B2)``.

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


def _blockscaled_tma_impl_dual(
    input: torch.Tensor,
    *,
    dim: _SingleQuantDim = (-2, -1),
) -> _FourTensorOutput:
    r"""_blockscaled_tma_impl_dual(input, *, dim=(-2, -1)) -> tuple[Tensor, Tensor, Tensor, Tensor]

    Quantize both matrix axes of an N-D tensor in one kernel launch.

    Uses 1x32 scaling blocks and swizzled scales.

    Args:
        input (Tensor): CUDA tensor with the layout and dtypes described by
            :func:`_blockscaled_tma_impl`. Both selected axes must have sizes
            divisible by 32. Inputs not admitting a 2D or 4D TMA view are
            materialized as described by :func:`_blockscaled_tma_impl`.
        dim (tuple[int, int], optional): Ordered matrix-axis pair. Both this
            pair and its reverse are quantized. All other axes are batch axes.
            Default: ``(-2, -1)``.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: ``(q_first, s_first, q_reverse,
            s_reverse)``. The first pair quantizes ``dim[-1]``; the reverse
            pair quantizes ``dim[0]``. Results retain the input's logical axis
            order, with batch axes in their original positions, but need not retain its
            strides. Scale shapes follow :func:`_blockscaled_tma_impl` for
            each axis, regardless of input layout.

    Examples::
        >>> x = torch.zeros((128, 256), dtype=torch.bfloat16, device="cuda")
        >>> q_k, s_k, q_m, s_m = _blockscaled_tma_impl_dual(x)
    """
    return cast(
        _FourTensorOutput,
        _run_blockscaled_tma(
            input,
            dim,
            False,
            True,
            dual=True,
        ),
    )
