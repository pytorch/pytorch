import functools
from collections.abc import Sequence
from typing import cast, NamedTuple

import torch
from torch import Tensor
from torch._meta_registrations import ceil_div, round_up
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._native.utils.capability import device_ok, is_traced, on_current_device
from torch._native.utils.tensor import const_data_ptr
from torch.nn.functional import ScalingType, SwizzleType

from .scaled_grouped_mm_blockscaled_heuristics import (
    _select_kernel_config_fp4,
    _select_kernel_config_fp8,
)


class _BlockScaledFormat(NamedTuple):
    name: str
    ab_dtype: torch.dtype
    scale_dtypes: tuple[torch.dtype, ...]
    scale_recipes: tuple[int, ...]
    scale_ab_vec_size: int
    logical_vals_per_elem: int

    def matches(
        self,
        mat_a: Tensor,
        mat_b: Tensor,
        scale_a: Sequence[Tensor],
        scale_b: Sequence[Tensor],
        recipe_a: tuple[int, ...],
        recipe_b: tuple[int, ...],
    ) -> bool:
        return (
            mat_a.dtype == self.ab_dtype
            and mat_b.dtype == self.ab_dtype
            and tuple(scale.dtype for scale in scale_a) == self.scale_dtypes
            and tuple(scale.dtype for scale in scale_b) == self.scale_dtypes
            and recipe_a == self.scale_recipes
            and recipe_b == self.scale_recipes
        )


_BLOCKSCALED_FORMATS = {
    fmt.name: fmt
    for fmt in (
        _BlockScaledFormat(
            name="mxfp8",
            ab_dtype=torch.float8_e4m3fn,
            scale_dtypes=(torch.float8_e8m0fnu,),
            scale_recipes=(ScalingType.BlockWise1x32.value,),
            scale_ab_vec_size=32,
            logical_vals_per_elem=1,
        ),
        _BlockScaledFormat(
            name="mxfp4",
            ab_dtype=torch.float4_e2m1fn_x2,
            scale_dtypes=(torch.float8_e8m0fnu,),
            scale_recipes=(ScalingType.BlockWise1x32.value,),
            scale_ab_vec_size=32,
            logical_vals_per_elem=2,
        ),
        _BlockScaledFormat(
            name="nvfp4",
            ab_dtype=torch.float4_e2m1fn_x2,
            scale_dtypes=(torch.float8_e4m3fn, torch.float32),
            scale_recipes=(
                ScalingType.BlockWise1x16.value,
                ScalingType.TensorWise.value,
            ),
            scale_ab_vec_size=16,
            logical_vals_per_elem=2,
        ),
    )
}


def _get_blockscaled_format(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: Sequence[Tensor],
    scale_b: Sequence[Tensor],
    scale_recipe_a: Sequence[int],
    scale_recipe_b: Sequence[int],
) -> _BlockScaledFormat | None:
    recipe_a = tuple(scale_recipe_a)
    recipe_b = tuple(scale_recipe_b)
    return next(
        (
            fmt
            for fmt in _BLOCKSCALED_FORMATS.values()
            if fmt.matches(mat_a, mat_b, scale_a, scale_b, recipe_a, recipe_b)
        ),
        None,
    )


def _allocate_output(
    mat_a: Tensor, mat_b: Tensor, out_dtype: torch.dtype, ngroups: int
) -> Tensor:
    M, N = mat_a.size(0), mat_b.size(-1)
    alignment = 128 // torch.finfo(out_dtype).bits
    N_padded = round_up(N, alignment)
    if mat_b.dim() == 2:
        return torch.empty_strided(
            (ngroups, M, N),
            (M * N_padded, N_padded, 1),
            device=mat_a.device,
            dtype=out_dtype,
        )
    return torch.empty_strided(
        (M, N), (N_padded, 1), device=mat_a.device, dtype=out_dtype
    )


@instrumented_cutedsl_cache("aten::_scaled_grouped_mm_v2")
def _compile_scaled_grouped_mm_blockscaled(
    max_active_clusters: int,
    uniform_mn_groups: bool,
    mma_tile_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    transpose_ab: bool,
    ab_dtype: torch.dtype,
    scale_ab_dtype: torch.dtype,
    scale_ab_vec_size: int,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from torch._native.cutedsl.dtypes import torch2cute
    from torch._native.cutedsl.launch import compile_kernel, fake_compact, sym

    from .scaled_grouped_mm_blockscaled_kernel import Sm100GroupedBlockScaledGemmKernel

    m = sym()
    n = sym(16)
    k = sym(scale_ab_vec_size)
    a_stride_m = sym(16)
    b_stride_n = sym(16)

    cute_ab_dtype = torch2cute[ab_dtype]
    cute_scale_ab_dtype = torch2cute[scale_ab_dtype]
    c_dtype = torch2cute[torch.bfloat16]
    c_stride_0 = sym(128 // c_dtype.width)

    fake_a = make_fake_tensor(cute_ab_dtype, (m, k, 1), stride=(a_stride_m, 1, 0))
    if uniform_mn_groups:
        fake_b = make_fake_tensor(cute_ab_dtype, (n, k, 1), stride=(b_stride_n, 1, 0))
    else:
        fake_b = make_fake_tensor(
            cute_ab_dtype, (n, k, sym()), stride=(b_stride_n, 1, sym(16))
        )
    if transpose_ab:
        # In transpose mode, C is passed as (N, M, 1) with transpose strides.
        fake_c = make_fake_tensor(c_dtype, (n, m, 1), stride=(1, c_stride_0, 0))
    else:
        fake_c = make_fake_tensor(c_dtype, (m, n, 1), stride=(c_stride_0, 1, 0))
    fake_scale_a = make_fake_tensor(
        cute_scale_ab_dtype, (sym(), sym(), 1), stride=(sym(), sym(), 0)
    )
    fake_scale_b = make_fake_tensor(
        cute_scale_ab_dtype, (sym(), sym(), 1), stride=(sym(), sym(), 0)
    )

    g = sym()
    fake_problem = fake_compact(cutlass.Int32, (g, 4), stride_order=(1, 0))
    fake_total_clusters = fake_compact(cutlass.Int32, (1,))
    fake_tensormap = fake_compact(
        cutlass.Int64,
        (
            g,
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
        ),
        stride_order=(2, 1, 0),
    )
    fake_offs = fake_compact(cutlass.Int32, (g,))
    fake_group_offsets = fake_compact(cutlass.Int32, (g, 2), stride_order=(1, 0))

    grouped_gemm = Sm100GroupedBlockScaledGemmKernel(
        sf_vec_size=scale_ab_vec_size,
        mma_tiler_mn=mma_tile_mn,
        cluster_shape_mn=cluster_shape_mn,
        transpose_ab=transpose_ab,
        uniform_mn_groups=uniform_mn_groups,
    )

    group_count = 0
    estimate_total_num_clusters = cutlass.Int32(1)
    sf_act_rows = cutlass.Int32(1)
    dims_mnk = tuple(cute.sym_int(32) for _ in range(3))
    base_ptrs = tuple(cute.sym_int(64) for _ in range(7))
    strides = tuple(
        tuple(cute.sym_int(64) for _ in range(rank)) for rank in (2, 3, 2, 3, 2, 2, 2)
    )
    return compile_kernel(
        grouped_gemm,
        fake_a,
        fake_b,
        fake_c,
        fake_scale_a,
        fake_scale_b,
        group_count,
        fake_problem,
        estimate_total_num_clusters,
        fake_total_clusters,
        fake_tensormap,
        fake_offs,
        fake_group_offsets,
        sf_act_rows,
        dims_mnk,
        base_ptrs,
        strides,
        max_active_clusters,
        make_fake_stream(),
        options="--enable-assertions",
    )


@functools.cache
def _get_max_active_clusters(cluster_size: int, device_id: int) -> int:
    import cutlass

    return cutlass.utils.HardwareInfo(device_id).get_max_active_clusters(cluster_size)


def _with_l_dim(t: Tensor) -> Tensor:
    # Kernel expects L as the last dimension.  Use a stride-0 L to
    # avoid ambiguous layouts (multiple stride==1) for
    # mark_layout_dynamic.
    return t.as_strided((*t.size(), 1), (*t.stride(), 0))


def _get_cluster_tile_shape_mn(
    mma_tile_mn: tuple[int, int], cluster_shape_mn: tuple[int, int]
) -> tuple[int, int]:
    cta_tile_m = mma_tile_mn[0]
    if mma_tile_mn[0] == 256:
        cta_tile_m //= 2
    return cta_tile_m * cluster_shape_mn[0], mma_tile_mn[1] * cluster_shape_mn[1]


def _estimate_total_clusters_for_launch(
    *,
    b_is_2d: bool,
    transpose_ab: bool,
    ngroups: int,
    M: int,
    N: int,
    cluster_tile_m: int,
    cluster_tile_n: int,
) -> int:
    if b_is_2d:
        if transpose_ab:
            return ngroups * ceil_div(N, cluster_tile_m) * ceil_div(M, cluster_tile_n)
        return ngroups * ceil_div(M, cluster_tile_m) * ceil_div(N, cluster_tile_n)

    if transpose_ab:
        grouped_tiles = ceil_div(M, cluster_tile_n) + max(ngroups - 1, 0)
        return ceil_div(N, cluster_tile_m) * grouped_tiles

    grouped_tiles = ceil_div(M, cluster_tile_m) + max(ngroups - 1, 0)
    return grouped_tiles * ceil_div(N, cluster_tile_n)


@functools.cache
def _alloc_aux_tensors(
    device_index: int, cap: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from .scaled_grouped_mm_blockscaled_kernel import Sm100GroupedBlockScaledGemmKernel

    device = torch.device("cuda", device_index)
    problem_sizes = torch.empty((cap, 4), device=device, dtype=torch.int32)
    total_num_clusters = torch.empty((1,), device=device, dtype=torch.int32)
    tensormaps = torch.empty(
        (
            cap,
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
        ),
        device=device,
        dtype=torch.int64,
    )
    group_offsets = torch.empty((cap, 2), device=device, dtype=torch.int32)
    return problem_sizes, total_num_clusters, tensormaps, group_offsets


@functools.cache
def _get_aux_tensors(
    ngroups: int, device_index: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Re-allocated when ngroups exceeds current capacity (rounded to
    # next power of 2).
    cap = max(64, 1 << (ngroups - 1).bit_length())
    problem_sizes, total_num_clusters, tensormaps, group_offsets = _alloc_aux_tensors(
        device_index, cap
    )
    return (
        problem_sizes[:ngroups],
        total_num_clusters,
        tensormaps[:ngroups],
        group_offsets[:ngroups],
    )


def _validated_blockscaled_format(mat_a: Tensor, scale_a: list[Tensor]):
    if mat_a.dtype == torch.float8_e4m3fn:
        return _BLOCKSCALED_FORMATS["mxfp8"]
    return _BLOCKSCALED_FORMATS["nvfp4" if len(scale_a) == 2 else "mxfp4"]


def scaled_grouped_mm_blockscaled(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: list[Tensor],
    scale_b: list[Tensor],
    offs: Tensor,
) -> Tensor:
    # Assumes _should_use_cutedsl_scaled_grouped_mm_blockscaled accepted the inputs.
    from torch._native.cutedsl.launch import read_only, stream

    def _is_transposed_layout(t: Tensor) -> bool:
        end_dim = t.dim() - 1
        if t.stride(end_dim - 1) == 1 and t.stride(end_dim) >= max(
            1, t.size(end_dim - 1)
        ):
            return True
        if t.stride(end_dim) == 1 and t.stride(end_dim - 1) >= max(1, t.size(end_dim)):
            return False
        raise ValueError(
            f"Invalid strides/sizes, got {t.stride()} for strides and "
            f"{t.size()} for sizes"
        )

    b_is_2d = mat_b.dim() == 2
    fmt = _validated_blockscaled_format(mat_a, scale_a)

    device = mat_a.device
    if mat_b.device != device:
        raise ValueError("mat_a and mat_b must be on the same device")
    if any(scale.device != device for scale in scale_a):
        raise ValueError("scale_a must be on the same device as mat_a")
    if any(scale.device != device for scale in scale_b):
        raise ValueError("scale_b must be on the same device as mat_a")
    if offs.device != device:
        raise ValueError("offs must be on the same device as mat_a")

    ngroups = int(offs.numel())
    mat_a_m = int(mat_a.size(0))
    mat_a_physical_k = int(mat_a.size(-1))
    mat_b_n = int(mat_b.size(-1))
    mat_b_physical_k = int(mat_b.size(0 if b_is_2d else -2))
    global_scale_ptrs = (0, 0)
    if fmt.name == "nvfp4":
        if scale_a[1].numel() != ngroups or scale_b[1].numel() != ngroups:
            raise ValueError(
                "NVFP4 global scales must have numel equal to offs.numel()"
            )
        global_scale_a = scale_a[1].contiguous()
        global_scale_b = scale_b[1].contiguous()
        global_scale_ptrs = (
            const_data_ptr(global_scale_a),
            const_data_ptr(global_scale_b),
        )
    out = _allocate_output(mat_a, mat_b, torch.bfloat16, ngroups)
    if ngroups == 0:
        return out

    if const_data_ptr(mat_a) % 16 != 0:
        raise ValueError("expected data_ptr to be aligned to 16 bytes")
    if _is_transposed_layout(mat_a):
        raise ValueError("expected mat_a to not be transposed")

    if const_data_ptr(mat_b) % 16 != 0:
        raise ValueError("expected data_ptr to be aligned to 16 bytes")
    if not _is_transposed_layout(mat_b):
        raise ValueError("expected mat_b to be transposed")

    logical_k_a = mat_a_physical_k * fmt.logical_vals_per_elem
    logical_k_b = mat_b_physical_k * fmt.logical_vals_per_elem
    if b_is_2d and logical_k_a != logical_k_b:
        raise ValueError("for 2d/2d grouped gemm, total K dimensions must match")
    if not b_is_2d:
        if logical_k_a != logical_k_b:
            raise ValueError("contraction dimension of mat_a and mat_b must match")
        if ngroups != int(mat_b.size(0)):
            raise ValueError(
                "for 2d/3d grouped gemm, offs size must match mat_b.size(0)"
            )

    device_id = device.index
    m_for_heuristic = mat_a_m if b_is_2d else (mat_a_m // max(ngroups, 1))
    k_for_heuristic = logical_k_a // max(ngroups, 1) if b_is_2d else logical_k_a
    if fmt.name in ("mxfp4", "nvfp4"):
        config = _select_kernel_config_fp4(m_for_heuristic, mat_b_n, k_for_heuristic)
    else:
        config = _select_kernel_config_fp8(m_for_heuristic, mat_b_n, k_for_heuristic)

    cluster_size = config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
    max_active_clusters = _get_max_active_clusters(cluster_size, device_id)
    cluster_tile_m, cluster_tile_n = _get_cluster_tile_shape_mn(
        config.mma_tile_mn, config.cluster_shape_mn
    )
    estimate_total_num_clusters = _estimate_total_clusters_for_launch(
        b_is_2d=b_is_2d,
        transpose_ab=config.transpose_ab,
        ngroups=ngroups,
        M=mat_a_m,
        N=mat_b_n,
        cluster_tile_m=cluster_tile_m,
        cluster_tile_n=cluster_tile_n,
    )

    scaled_grouped_mm_blockscaled_compiled = _compile_scaled_grouped_mm_blockscaled(
        max_active_clusters,
        b_is_2d,
        config.mma_tile_mn,
        config.cluster_shape_mn,
        config.transpose_ab,
        fmt.ab_dtype,
        fmt.scale_dtypes[0],
        fmt.scale_ab_vec_size,
    )

    problem_sizes, total_num_clusters, tensormaps, group_offsets = _get_aux_tensors(
        ngroups, device_id
    )

    scale_a0 = scale_a[0]
    scale_b0 = scale_b[0]

    mat_a_ptr = const_data_ptr(mat_a)
    mat_b_ptr = const_data_ptr(mat_b)
    out_ptr = out.data_ptr()
    scale_a_ptr = const_data_ptr(scale_a0)
    scale_b_ptr = const_data_ptr(scale_b0)
    mat_a_stride = mat_a.stride()
    mat_b_stride = mat_b.stride()
    out_stride = out[0].stride() if b_is_2d else out.stride()
    scale_a_stride = scale_a0.stride()
    scale_b_stride = scale_b0.stride()

    if b_is_2d:
        if fmt.logical_vals_per_elem > 1:
            stride_b_logical = (0, 1, logical_k_a)
        else:
            stride_b_logical = (0, mat_b_stride[0], mat_b_stride[1])
    elif fmt.logical_vals_per_elem > 1:
        stride_b_logical = (mat_b_stride[0], 1, logical_k_a)
    else:
        stride_b_logical = mat_b_stride

    metadata_strides = (
        mat_a_stride,
        (0, mat_b_stride[0], mat_b_stride[1]) if b_is_2d else mat_b_stride,
        ((logical_k_a, 1) if fmt.logical_vals_per_elem > 1 else mat_a_stride),
        stride_b_logical,
        out_stride,
        (
            (
                round_up(logical_k_a // fmt.scale_ab_vec_size, 4),
                1,
            )
            if (not b_is_2d) and fmt.name == "nvfp4"
            else scale_a_stride
        ),
        scale_b_stride,
    )

    scaled_grouped_mm_blockscaled_compiled(
        read_only(_with_l_dim(mat_a)),
        read_only(
            _with_l_dim(mat_b.transpose(0, 1)) if b_is_2d else mat_b.permute(2, 1, 0)
        ),
        _with_l_dim(
            (out[0].transpose(0, 1) if config.transpose_ab else out[0])
            if b_is_2d
            else (out.transpose(0, 1) if config.transpose_ab else out)
        ),
        read_only(_with_l_dim(scale_a0)),
        read_only(_with_l_dim(scale_b0)),
        ngroups,
        problem_sizes,
        estimate_total_num_clusters,
        total_num_clusters,
        tensormaps,
        read_only(offs),
        group_offsets,
        scale_a0.numel() // round_up(logical_k_a // fmt.scale_ab_vec_size, 4),
        (mat_a_m, mat_b_n, logical_k_a),
        (
            mat_a_ptr,
            mat_b_ptr,
            out_ptr,
            scale_a_ptr,
            scale_b_ptr,
            *global_scale_ptrs,
        ),
        metadata_strides,
        stream(device_id),
    )
    return out


def _should_use_cutedsl_scaled_grouped_mm_blockscaled(
    mat_a: object,
    mat_b: object,
    scale_a: object,
    scale_recipe_a: object,
    swizzle_a: object,
    scale_b: object,
    scale_recipe_b: object,
    swizzle_b: object,
    offs: object,
    bias: object,
    out_dtype: object,
    contraction_dim: object,
    use_fast_accum: object,
) -> bool:
    if not isinstance(mat_a, Tensor) or not isinstance(mat_b, Tensor):
        return False
    if not isinstance(scale_a, list) or not isinstance(scale_b, list):
        return False
    if is_traced(mat_a) or not device_ok(mat_a, (10,)):
        return False
    if not on_current_device(mat_a) or mat_b.device != mat_a.device:
        return False
    if mat_a.dim() != 2 or mat_b.dim() not in (2, 3):
        return False
    # The kernel needs at least one K tile to produce an accumulator.
    if mat_a.size(-1) == 0:
        return False
    if out_dtype not in (None, torch.bfloat16):
        return False
    if not isinstance(contraction_dim, Sequence):
        return False
    if (
        mat_b.dim() == 3
        and contraction_dim
        and tuple(cast(Sequence[int], contraction_dim)) != (-1, -2)
    ):
        return False
    if not isinstance(scale_recipe_a, list) or not isinstance(scale_recipe_b, list):
        return False
    if not isinstance(swizzle_a, list) or not isinstance(swizzle_b, list):
        return False
    swizzle = [SwizzleType.SWIZZLE_32_4_4.value]
    if swizzle_a != swizzle or swizzle_b != swizzle:
        return False
    fmt = _get_blockscaled_format(
        mat_a,
        mat_b,
        cast(list[Tensor], scale_a),
        cast(list[Tensor], scale_b),
        cast(list[int], scale_recipe_a),
        cast(list[int], scale_recipe_b),
    )
    if fmt is None:
        return False
    if offs is None:
        return False
    if bias is not None:
        return False
    if bool(use_fast_accum):
        return False
    return True


__all__ = [
    "scaled_grouped_mm_blockscaled",
    "_should_use_cutedsl_scaled_grouped_mm_blockscaled",
]
