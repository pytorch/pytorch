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


class _CanonicalMatrixInput(NamedTuple):
    r"""TMA input and metadata for returning results in the caller's axis order.

    For contiguous ``[*B0, M, *B1, K, B2]``, canonicalization moves all batch
    axes ahead of ``(M, K)`` and folds them into ``B_product``. Batch axes retain
    their physical stride order even when their logical positions change.

    Attributes:
        input (Tensor): Canonical 2D or 4D matrix input, either a view or a copy.
        dim (_SingleQuantDim): Quantization axis pair in ``input`` coordinates;
          ``(1, 3)`` quantizes the last matrix axis of a 4D input, while
          ``(3, 1)`` quantizes the first.
        input_matrix_axes (tuple[int, int]): Caller axes corresponding to the
          canonical input's ``M`` and ``K`` dimensions, in that order.
        batch_shape (tuple[int, ...]): Sizes of batch axes in canonical order,
          before folding them into ``input``.
        inverse_permutation (tuple[int, ...]): Permutation that restores the
          original logical axis order after the batch axes are unfolded.
        swap_dual_outputs (bool): Whether the physical matrix-axis order
          reversed the two dual-quantization result pairs.
    """

    input: torch.Tensor
    dim: _SingleQuantDim
    input_matrix_axes: _SingleQuantDim
    batch_shape: tuple[int, ...]
    inverse_permutation: tuple[int, ...]
    swap_dual_outputs: bool


def _canonicalize_matrix_input(
    input: torch.Tensor,
    dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    *,
    dual: bool,
) -> _CanonicalMatrixInput:
    r"""Arrange matrix and batch axes for the 2D or 4D TMA kernel.

    For contiguous ``[*B0, M, *B1, K, B2]``, the trailing ``B2`` prevents
    TMA from directly loading an ``(M, K)`` tile. Permute to
    ``[*B0, *B1, B2, M, K]``, materialize contiguous storage, and fold the
    batch axes into ``(B_product, M, 1, K)``. Keep batch axes in physical stride
    order when materializing, even if their logical order differs. For single
    quantization, materialization places the requested quantized axis last
    (dim-k); dual quantization keeps both matrix orientations. When the physical
    matrix axes permit a direct TMA load, instead use a view shaped like
    ``(B_outer, M, B_inner, K)`` (or with ``M`` and ``K`` reversed), without
    materializing. Return normalized axes and metadata to undo the permutation.

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
    if dual and normalized != matrix_axes:
        raise ValueError(f"dual quantization must start with dim-k: {dim}")
    physical_axes = tuple(sorted(range(ndim), key=lambda axis: -input.stride(axis)))
    physical_input = (
        input if physical_axes == tuple(range(ndim)) else input.permute(physical_axes)
    )
    physical_matrix_axes = tuple(axis for axis in physical_axes if axis in matrix_axes)
    batch_axes = tuple(axis for axis in physical_axes if axis not in matrix_axes)
    fast_axis = physical_matrix_axes[-1]
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
        # view requires dim-m, single quantization materializes dim-k;
        # dual quantization is rejected by the spec.
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
    if matrix_input.data_ptr() % _INPUT_ALIGNMENT_BYTES:
        matrix_input = matrix_input.clone()
    batch_shape = tuple(input.shape[axis] for axis in batch_axes)
    return _CanonicalMatrixInput(
        matrix_input,
        canonical_dim,
        (permutation[-2], permutation[-1]),
        batch_shape,
        inverse,
        swap_dual_outputs,
    )


def _restore_matrix_output(
    output: _BlockscaledTmaOutput, canonical: _CanonicalMatrixInput
) -> _BlockscaledTmaOutput:
    r"""Restore result axes after quantizing the canonical matrix input.

    For contiguous ``[*B0, M, *B1, K, B2]``, unfold each qdata result's
    leading ``B_product`` axis from ``(B_product, M, K)`` to
    ``[*B0, *B1, B2, M, K]``, then permute it to ``[*B0, M, *B1, K, B2]``.
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


def _build_blockscaled_tma_spec(
    input: torch.Tensor,
    *,
    dim: _SingleQuantDim,
    dual: bool,
    input_matrix_axes: _SingleQuantDim,
    requested_dim: _SingleQuantDim,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
) -> _BlockscaledTmaSpec:
    r"""Build a launch spec and check kernel limits for canonical 2D/4D input."""

    ndim = input.dim()
    is_batched = ndim == 4
    dim_k = (1, 3) if is_batched else (-2, -1)
    do_dim_k = dual or dim == dim_k
    do_dim_m = dual or dim == dim_k[::-1]
    row_axis, col_axis = input_matrix_axes

    if do_dim_k and do_dim_m:
        if not is_scale_swizzled:
            raise ValueError("dual quantization does not support compact scales")
        if is_square_scaling:
            raise ValueError("dual quantization does not support square scaling")

    if not is_scale_swizzled:
        if do_dim_m:
            raise ValueError(
                f"compact scales require quantization along input axis {col_axis} "
                f"(the kernel's contiguous axis); got dim={requested_dim}"
            )
        if is_square_scaling:
            raise ValueError("compact scales do not support square scaling")

    if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("blockscaled TMA supports only bf16, fp16, and fp32 input")

    if is_square_scaling:
        if do_dim_m:
            raise ValueError(
                f"square scaling requires quantization along input axis {col_axis} "
                f"(the kernel's contiguous axis); got dim={requested_dim}"
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
) -> _PreparedBlockscaledTmaLaunch:
    """Select the launch plan and allocate its output storage."""

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
            f"grid=({plan.grid_k}, {plan.grid_m}, {spec.B_product}), "
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
    canonical = _canonicalize_matrix_input(
        input, dim, is_square_scaling, is_scale_swizzled, dual=dual
    )
    spec = _build_blockscaled_tma_spec(
        canonical.input,
        dim=canonical.dim,
        dual=dual,
        input_matrix_axes=canonical.input_matrix_axes,
        requested_dim=dim,
        is_square_scaling=is_square_scaling,
        is_scale_swizzled=is_scale_swizzled,
    )
    input = canonical.input
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
    contiguous. Canonical storage with an unaligned pointer is cloned to
    meet the 16-byte alignment requirement.

    Args:
        input (Tensor): CUDA tensor with at least two dimensions on an SM100+
            GPU, with dtype float16, bfloat16, or float32.
        dim (tuple[int, int]): Ordered pair of distinct positive or negative
            axes, outer then inner in the logical matrix plane. The second
            selects the block-scaled axis and must have size divisible by 32.
            For 2D input, ``(0, 1)`` or ``(-2, -1)`` quantizes columns, while
            ``(1, 0)`` or ``(-1, -2)`` quantizes rows. Lists are not accepted.
        is_square_scaling (bool): Use 32x32 instead of 1x32 scale blocks.
            Requires swizzled scales and a dim-k kernel layout after
            canonicalization.
        is_scale_swizzled (bool): Return swizzled scales if ``True``; compact
            scales otherwise. Compact scales require a dim-k kernel layout.
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
            materialized as described above.
        dim (tuple[int, int], optional): Ordered matrix-axis pair. Both this
            pair and its reverse are quantized. The pair must be in ascending
            order after normalizing negative axes; all other axes are batch
            axes. Default: ``(-2, -1)``.

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
