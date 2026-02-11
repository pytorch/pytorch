import functools
import importlib
from collections.abc import Sequence
from typing import NamedTuple, Optional

import torch
from torch import Tensor
from torch._cutedsl._compile_with_safe_names import _compile_with_safe_names
from torch._cutedsl.scaled_grouped_mm_prepare_metadata import (
    _compile_scaled_grouped_mm_prepare_metadata,
)
from torch.nn.functional import ScalingType, SwizzleType


class _KernelConfig(NamedTuple):
    mma_tile_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]


@functools.cache
def _cutedsl_unavailable_reason() -> Optional[str]:
    deps = [
        ("nvidia-cutlass-dsl", "cutlass"),
        ("cuda-bindings", "cuda.bindings.driver"),
    ]
    for package_name, module_name in deps:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            return (
                f"missing optional dependency `{package_name}` "
                f"(import `{module_name}` failed: {exc})"
            )
    return None


def assert_cutedsl_runtime_available() -> None:
    reason = _cutedsl_unavailable_reason()
    if reason is None:
        return
    raise RuntimeError(
        "scaled_grouped_mm CuTeDSL path requires optional Python packages "
        "`nvidia-cutlass-dsl` and `cuda-bindings` (from NVIDIA cuda-python); "
        f"{reason}"
    )


def _select_kernel_config(M: int, N: int, K: int) -> _KernelConfig:
    # This came from MSLK heuristics, but it collapses to two configs
    # for our non-transpose path (transpose specialization did not
    # show benefit).
    config_medium = _KernelConfig((128, 128), (2, 1))
    config_large = _KernelConfig((256, 256), (2, 1))
    del N, K  # Kept in signature for eventual future use.
    return config_medium if M <= 128 else config_large


def _round_up(a: int, b: int) -> int:
    return ((a + b - 1) // b) * b


def _allocate_output(mat_a: Tensor, mat_b: Tensor, out_dtype: torch.dtype) -> Tensor:
    M, N = mat_a.size(0), mat_b.size(-1)
    alignment = 128 // torch.finfo(out_dtype).bits
    N_padded = _round_up(N, alignment)
    return torch.empty_strided(
        (M, N), (N_padded, 1), device=mat_a.device, dtype=out_dtype
    )


def _check_valid_strides_and_return_transposed(mat: Tensor) -> bool:
    strides = mat.stride()
    sizes = mat.size()
    end_dim = mat.dim() - 1
    alignment = 16 // mat.element_size()
    if mat.device.type != "cpu" and mat.data_ptr() % 16 != 0:
        raise ValueError("expected data_ptr to be aligned to 16 bytes")
    if strides[end_dim - 1] == 1 and strides[end_dim] >= max(1, sizes[end_dim - 1]):
        if strides[end_dim] % alignment != 0:
            raise ValueError("strides should be multiple of 16 bytes")
        return True
    if strides[end_dim] == 1 and strides[end_dim - 1] >= max(1, sizes[end_dim]):
        if strides[end_dim - 1] % alignment != 0:
            raise ValueError("strides should be multiple of 16 bytes")
        return False
    raise ValueError(
        f"Invalid strides/sizes, got {mat.stride()} for strides and {mat.size()} for sizes"
    )


def _check_scales_blocked(mat: Tensor, scale: Tensor, arg_idx: int) -> None:
    blocksize = 32
    if mat.dim() == 2:
        if scale.dim() != mat.dim():
            raise ValueError(
                f"for block-scaled, scale must have same number of dimensions as "
                f"parent tensor, but got mat.dim() = {mat.dim()} and scale.dim() = "
                f"{scale.dim()} for arg {arg_idx}"
            )
        scale_dim_to_check = 0
        mat_dim_to_check = 0 if arg_idx == 0 else 1
        if scale.size(scale_dim_to_check) < mat.size(mat_dim_to_check):
            raise ValueError(
                f"for block-scaled, arg {arg_idx} tensor shape ({mat.size(0)}, {mat.size(1)}) "
                f"must have scale.shape[{scale_dim_to_check}] >= {mat.size(mat_dim_to_check)} "
                f"but got scale.shape=({scale.size(0)}, {scale.size(1)})"
            )
    else:
        G = mat.size(0)
        K = mat.size(1)
        N = mat.size(2)
        blocked_scale_k = _round_up(K // blocksize, 4)
        blocked_scale_n = _round_up(N, 128)
        if scale.dim() != mat.dim() - 1:
            raise ValueError(
                "for block-scaled 2d-3d grouped GEMM, the 3d tensor must have "
                "a 2d scale of shape (G, blocked_scale_K * blocked_scale_N), "
                f"but scale is {scale.dim()}D for arg {arg_idx}"
            )
        if scale.size(0) != G or scale.size(1) != blocked_scale_k * blocked_scale_n:
            raise ValueError(
                f"for block-scaled grouped GEMM, the tensor shape ({G}, {K}, {N}) must have "
                f"scale shape ({G}, {blocked_scale_k}, {blocked_scale_n}) for arg {arg_idx}, "
                f"got: {scale.size(0)}, {scale.size(1)}"
            )


@functools.cache
def _compile_scaled_grouped_mm_mxfp8(
    sm_count: int,
    max_active_clusters: int,
    mma_tile_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from torch._cutedsl.scaled_grouped_mm_mxfp8_kernel import (
        Sm100GroupedBlockScaledGemmKernel,
    )

    m = cute.sym_int()
    n = cute.sym_int()
    k = cute.sym_int()
    a_stride_m = cute.sym_int()
    b_stride_n = cute.sym_int()
    c_stride_m = cute.sym_int()
    sf_a_m = cute.sym_int()
    sf_a_k = cute.sym_int()
    sf_b_n = cute.sym_int()
    sf_b_k = cute.sym_int()
    sf_a_stride_m = cute.sym_int()
    sf_a_stride_k = cute.sym_int()
    sf_b_stride_n = cute.sym_int()
    sf_b_stride_k = cute.sym_int()

    fake_a = make_fake_tensor(
        cutlass.Float8E4M3FN,
        (m, k, 1),
        stride=(a_stride_m, 1, 0),
    )
    fake_b = make_fake_tensor(
        cutlass.Float8E4M3FN,
        (n, k, 1),
        stride=(b_stride_n, 1, 0),
    )
    fake_c = make_fake_tensor(
        cutlass.BFloat16,
        (m, n, 1),
        stride=(c_stride_m, 1, 0),
    )
    fake_scale_a = make_fake_tensor(
        cutlass.Float8E8M0FNU,
        (sf_a_m, sf_a_k, 1),
        stride=(sf_a_stride_m, sf_a_stride_k, 0),
    )
    fake_scale_b = make_fake_tensor(
        cutlass.Float8E8M0FNU,
        (sf_b_n, sf_b_k, 1),
        stride=(sf_b_stride_n, sf_b_stride_k, 0),
    )

    g = cute.sym_int()
    fake_problem = make_fake_tensor(cutlass.Int32, (g, 4), stride=(4, 1))
    fake_strides = make_fake_tensor(cutlass.Int32, (g, 3, 2), stride=(6, 2, 1))
    fake_ptrs_abc = make_fake_tensor(cutlass.Int64, (g, 3), stride=(3, 1))
    fake_ptrs_scale = make_fake_tensor(cutlass.Int64, (g, 2), stride=(2, 1))
    fake_total_clusters = make_fake_tensor(cutlass.Int32, (1,), stride=(1,))

    tensormap_stride1 = Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
    tensormap_stride0 = (
        tensormap_stride1 * Sm100GroupedBlockScaledGemmKernel.num_tensormaps
    )
    fake_tensormap = make_fake_tensor(
        cutlass.Int64,
        (
            sm_count,
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
        ),
        stride=(tensormap_stride0, tensormap_stride1, 1),
    )
    fake_stream = make_fake_stream()

    grouped_gemm = Sm100GroupedBlockScaledGemmKernel(
        sf_vec_size=32,
        mma_tiler_mn=mma_tile_mn,
        cluster_shape_mn=cluster_shape_mn,
    )

    compiled = _compile_with_safe_names(
        lambda: cute.compile(
            grouped_gemm,
            initial_a=fake_a,
            initial_b=fake_b,
            initial_c=fake_c,
            initial_sfa=fake_scale_a,
            initial_sfb=fake_scale_b,
            group_count=0,
            problem_shape_mnkl=fake_problem,
            strides_abc=fake_strides,
            tensor_address_abc=fake_ptrs_abc,
            tensor_address_sfasfb=fake_ptrs_scale,
            estimate_total_num_clusters=max_active_clusters,
            total_num_clusters=fake_total_clusters,
            tensormap_cute_tensor=fake_tensormap,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            options="--enable-assertions",
        )
    )
    return compiled, grouped_gemm.cluster_tile_shape_mnk


@functools.lru_cache(maxsize=1)
def _get_hardware_info(device_id: int):
    import cutlass

    return cutlass.utils.HardwareInfo(device_id)


@functools.cache
def _get_schedule_meta(cluster_size: int, device_id: int) -> tuple[int, int]:
    hw = _get_hardware_info(device_id)
    sm_count = hw.get_max_active_clusters(1)
    max_active_clusters = hw.get_max_active_clusters(cluster_size)
    return sm_count, max_active_clusters


@functools.cache
def _alloc_aux_tensors(
    device_index: int, cap: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = torch.device("cuda", device_index)
    ptrs_abc = torch.empty((cap, 3), device=device, dtype=torch.int64)
    ptrs_scale = torch.empty((cap, 2), device=device, dtype=torch.int64)
    problem_sizes = torch.empty((cap, 4), device=device, dtype=torch.int32)
    strides_abc = torch.empty((cap, 3, 2), device=device, dtype=torch.int32)
    total_num_clusters = torch.empty((1,), device=device, dtype=torch.int32)
    return ptrs_abc, ptrs_scale, problem_sizes, strides_abc, total_num_clusters


def _get_aux_tensors(
    ngroups: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # Re-allocated when ngroups exceeds current capacity (rounded to
    # next power of 2).
    cap = max(64, 1 << (ngroups - 1).bit_length())
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    ptrs_abc, ptrs_scale, problem_sizes, strides_abc, total_num_clusters = (
        _alloc_aux_tensors(device_index, cap)
    )
    return (
        ptrs_abc[:ngroups],
        ptrs_scale[:ngroups],
        problem_sizes[:ngroups],
        strides_abc[:ngroups],
        total_num_clusters,
    )


@functools.cache
def _alloc_tensormap(device_index: int, sm_count: int) -> Tensor:
    from torch._cutedsl.scaled_grouped_mm_mxfp8_kernel import (
        Sm100GroupedBlockScaledGemmKernel,
    )

    device = torch.device("cuda", device_index)
    shape = (
        sm_count,
        Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
        Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
    )
    return torch.empty(shape, device=device, dtype=torch.int64)


def _get_tensormap(sm_count: int, device: torch.device) -> Tensor:
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return _alloc_tensormap(device_index, sm_count)


def scaled_grouped_mm_mxfp8(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: list[Tensor],
    scale_b: list[Tensor],
    scale_recipe_a: list[ScalingType],
    scale_recipe_b: list[ScalingType],
    swizzle_a: list[SwizzleType],
    swizzle_b: list[SwizzleType],
    offs: Optional[Tensor],
    output_dtype: Optional[torch.dtype] = None,
    contraction_dim: Sequence[int] = (),
    use_fast_accum: bool = False,
    bias: Optional[Tensor] = None,
) -> Tensor:
    if mat_a.size(-1) % 32 != 0:
        raise ValueError("K dimension must be divisible by 32 for MXFP8 block scaling")

    assert_cutedsl_runtime_available()

    if mat_a.device.type != "cuda":
        raise ValueError("scaled grouped MM is only supported on CUDA")

    if bias is not None:
        raise ValueError("bias is not supported for scaled grouped MM")
    if offs is None:
        raise ValueError("offs must be provided for scaled grouped MM")
    if mat_a.dim() != 2 or mat_b.dim() != 3:
        raise ValueError("expected mat_a to be 2D and mat_b to be 3D")
    if mat_a.dtype != torch.float8_e4m3fn or mat_b.dtype != torch.float8_e4m3fn:
        raise ValueError("expected mat_a and mat_b to be float8_e4m3fn")
    if mat_a.device != mat_b.device:
        raise ValueError("expected mat_a and mat_b to be on the same device")
    if _check_valid_strides_and_return_transposed(mat_a):
        raise ValueError("expected mat_a to be transposed")
    if not _check_valid_strides_and_return_transposed(mat_b):
        raise ValueError("expected mat_b to be transposed")
    if output_dtype is not None and output_dtype != torch.bfloat16:
        raise ValueError("only bfloat16 output is supported for scaled grouped MM")
    if use_fast_accum:
        raise ValueError("use_fast_accum is not supported for scaled grouped MM")
    if contraction_dim and tuple(contraction_dim) != (-1, -2):
        raise ValueError("contraction_dim must be (-1, -2) if provided")
    if contraction_dim:
        if mat_a.size(contraction_dim[0]) != mat_b.size(contraction_dim[1]):
            raise ValueError("contraction dimension of mat_a and mat_b must match")
    else:
        if mat_a.size(-1) != mat_b.size(-2):
            raise ValueError("contraction dimension of mat_a and mat_b must match")
    if len(scale_a) != 1 or len(scale_b) != 1:
        raise ValueError("scale_a and scale_b must contain a single tensor")
    if (
        scale_a[0].dtype != torch.float8_e8m0fnu
        or scale_b[0].dtype != torch.float8_e8m0fnu
    ):
        raise ValueError("scale_a and scale_b must be float8_e8m0fnu")
    if scale_a[0].device != mat_a.device or scale_b[0].device != mat_a.device:
        raise ValueError("scale_a and scale_b must be on the same device as mat_a")
    if offs.dtype != torch.int32:
        raise ValueError("offs must be int32")
    if offs.dim() != 1:
        raise ValueError("offs must be 1D")
    if offs.numel() != mat_b.size(0):
        raise ValueError("offs size must match mat_b batch dimension")
    if len(scale_recipe_a) != 1 or len(scale_recipe_b) != 1:
        raise ValueError("scale_recipe_a and scale_recipe_b must be singleton lists")
    if scale_recipe_a[0] != ScalingType.BlockWise1x32:
        raise ValueError("scale_recipe_a must be BlockWise1x32 for MXFP8")
    if scale_recipe_b[0] != ScalingType.BlockWise1x32:
        raise ValueError("scale_recipe_b must be BlockWise1x32 for MXFP8")
    if len(swizzle_a) != 1 or len(swizzle_b) != 1:
        raise ValueError("swizzle_a and swizzle_b must be singleton lists")
    if swizzle_a[0] != SwizzleType.SWIZZLE_32_4_4:
        raise ValueError("swizzle_a must be SWIZZLE_32_4_4 for MXFP8")
    if swizzle_b[0] != SwizzleType.SWIZZLE_32_4_4:
        raise ValueError("swizzle_b must be SWIZZLE_32_4_4 for MXFP8")

    _check_scales_blocked(mat_a, scale_a[0], 0)
    _check_scales_blocked(mat_b, scale_b[0], 1)

    out_dtype = output_dtype or torch.bfloat16
    out = _allocate_output(mat_a, mat_b, out_dtype)
    ngroups = int(offs.numel())
    if ngroups == 0:
        return out
    device = mat_a.device

    max_threads = torch.cuda.get_device_properties(device).max_threads_per_block
    threads_per_block = min(ngroups, max_threads)
    num_blocks = (ngroups + threads_per_block - 1) // threads_per_block

    M_per_group_avg = mat_a.size(0) // max(ngroups, 1)
    config = _select_kernel_config(M_per_group_avg, mat_b.size(-1), mat_a.size(-1))

    cluster_size = config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
    device_id = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    sm_count, max_active_clusters = _get_schedule_meta(cluster_size, device_id)

    scaled_grouped_mm_mxfp8_compiled, cluster_tile_shape_mnk = (
        _compile_scaled_grouped_mm_mxfp8(
            sm_count,
            max_active_clusters,
            config.mma_tile_mn,
            config.cluster_shape_mn,
        )
    )

    ptrs_abc, ptrs_scale, problem_sizes, strides_abc, total_num_clusters = (
        _get_aux_tensors(ngroups, device)
    )

    try:
        cluster_tile_m = int(cluster_tile_shape_mnk[0])
        cluster_tile_n = int(cluster_tile_shape_mnk[1])
    except Exception:
        cta_tile_m = config.mma_tile_mn[0]
        if config.mma_tile_mn[0] == 256:
            cta_tile_m //= 2
        cluster_tile_m = cta_tile_m * config.cluster_shape_mn[0]
        cluster_tile_n = config.mma_tile_mn[1] * config.cluster_shape_mn[1]
    scaled_grouped_mm_prepare_metadata_compiled = (
        _compile_scaled_grouped_mm_prepare_metadata()
    )

    import cuda.bindings.driver as cuda_driver

    stream = cuda_driver.CUstream(int(torch.cuda.current_stream().cuda_stream))

    scaled_grouped_mm_prepare_metadata_compiled(
        ngroups,
        int(mat_a.size(0)),
        int(mat_b.size(2)),
        int(mat_a.size(1)),
        int(mat_a.data_ptr()),
        int(mat_b.data_ptr()),
        int(out.data_ptr()),
        int(scale_a[0].data_ptr()),
        int(scale_b[0].data_ptr()),
        offs,
        tuple(map(int, mat_a.stride())),
        tuple(map(int, mat_b.stride())),
        tuple(map(int, out.stride())),
        tuple(map(int, scale_a[0].stride())),
        tuple(map(int, scale_b[0].stride())),
        cluster_tile_m,
        cluster_tile_n,
        problem_sizes,
        ptrs_abc,
        ptrs_scale,
        strides_abc,
        total_num_clusters,
        num_blocks,
        threads_per_block,
        stream,
    )

    tensormap = _get_tensormap(sm_count, device)

    def _with_l_dim(t: Tensor) -> Tensor:
        # Kernel expects L as the last dimension. Use a stride-0 L to
        # avoid ambiguous layouts (multiple stride==1) for
        # mark_layout_dynamic.
        sizes = t.size()
        strides = t.stride()
        return t.as_strided((*sizes, 1), (*strides, 0))

    def _pick_leading_dim(t: Tensor, preferred: int) -> int:
        ones = [i for i, s in enumerate(t.stride()) if s == 1]
        if preferred in ones:
            return preferred
        if len(ones) == 1:
            return ones[0]
        if ones:
            return ones[0]
        return 0

    def _as_cute_tensor(t: Tensor, preferred: int):
        from cutlass.cute.runtime import from_dlpack as _from_dlpack

        leading_dim = _pick_leading_dim(t, preferred)
        return _from_dlpack(t, assumed_align=16).mark_layout_dynamic(
            leading_dim=leading_dim
        )

    scaled_grouped_mm_mxfp8_compiled(
        _as_cute_tensor(_with_l_dim(mat_a), 1),
        _as_cute_tensor(_with_l_dim(mat_b[0].transpose(0, 1)), 1),
        _as_cute_tensor(_with_l_dim(out), 1),
        _as_cute_tensor(_with_l_dim(scale_a[0]), 1),
        _as_cute_tensor(_with_l_dim(scale_b[0]), 1),
        ngroups,
        problem_sizes,
        strides_abc,
        ptrs_abc,
        ptrs_scale,
        total_num_clusters,
        tensormap,
        stream,
    )
    return out


__all__ = ["scaled_grouped_mm_mxfp8"]
