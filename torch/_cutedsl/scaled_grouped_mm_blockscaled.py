import functools
import importlib
from collections.abc import Sequence
from typing import Any, cast, NamedTuple, Protocol

import torch
from torch import Tensor
from torch._C import (
    _ScalingType as ScalingType,  # pyrefly: ignore [missing-module-attribute]
    _SwizzleType as SwizzleType,  # pyrefly: ignore [missing-module-attribute]
)
from torch._cutedsl._compile_with_safe_names import _compile_with_safe_names
from torch._cutedsl.scaled_grouped_mm_prepare_metadata import (
    _compile_scaled_grouped_mm_prepare_metadata,
)
from torch.library import Library


class _KernelConfig(NamedTuple):
    mma_tile_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]
    transpose_ab: bool


class _BlockScaledFormat(NamedTuple):
    name: str
    torch_ab_dtype: torch.dtype
    torch_scale_ab_dtype: torch.dtype
    scale_ab_recipe: ScalingType
    scale_ab_vec_size: int
    logical_vals_per_elem: int
    cutlass_ab_dtype_name: str
    cutlass_scale_ab_dtype_name: str
    torch_global_scale_dtype: torch.dtype | None = None


_TORCH_TO_CUTLASS_DTYPE_NAME = {
    torch.bfloat16: "BFloat16",
    torch.float16: "Float16",
    torch.float32: "Float32",
    torch.float8_e4m3fn: "Float8E4M3FN",
    torch.float8_e5m2: "Float8E5M2",
}


_BLOCKSCALED_FORMATS = (
    _BlockScaledFormat(
        name="mxfp8",
        torch_ab_dtype=torch.float8_e4m3fn,
        torch_scale_ab_dtype=torch.float8_e8m0fnu,
        scale_ab_recipe=ScalingType.BlockWise1x32,
        scale_ab_vec_size=32,
        logical_vals_per_elem=1,
        cutlass_ab_dtype_name="Float8E4M3FN",
        cutlass_scale_ab_dtype_name="Float8E8M0FNU",
    ),
    _BlockScaledFormat(
        name="mxfp4",
        torch_ab_dtype=torch.float4_e2m1fn_x2,
        torch_scale_ab_dtype=torch.float8_e8m0fnu,
        scale_ab_recipe=ScalingType.BlockWise1x32,
        scale_ab_vec_size=32,
        logical_vals_per_elem=2,
        cutlass_ab_dtype_name="Float4E2M1FN",
        cutlass_scale_ab_dtype_name="Float8E8M0FNU",
    ),
    _BlockScaledFormat(
        name="nvfp4",
        torch_ab_dtype=torch.float4_e2m1fn_x2,
        torch_scale_ab_dtype=torch.float8_e4m3fn,
        scale_ab_recipe=ScalingType.BlockWise1x16,
        scale_ab_vec_size=16,
        logical_vals_per_elem=2,
        cutlass_ab_dtype_name="Float4E2M1FN",
        cutlass_scale_ab_dtype_name="Float8E4M3FN",
        torch_global_scale_dtype=torch.float32,
    ),
)


def _get_blockscaled_format(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: Sequence[Tensor],
    scale_b: Sequence[Tensor],
    scale_recipe_a: Sequence[ScalingType | int],
    scale_recipe_b: Sequence[ScalingType | int],
) -> _BlockScaledFormat | None:
    def _recipe_matches(value: ScalingType | int, expected: ScalingType) -> bool:
        return value == expected or value == expected.value

    for fmt in _BLOCKSCALED_FORMATS:
        if fmt.torch_global_scale_dtype is None:
            if len(scale_a) != 1 or len(scale_b) != 1:
                continue
            if len(scale_recipe_a) != 1 or len(scale_recipe_b) != 1:
                continue
            if (
                mat_a.dtype == fmt.torch_ab_dtype
                and mat_b.dtype == fmt.torch_ab_dtype
                and scale_a[0].dtype == fmt.torch_scale_ab_dtype
                and scale_b[0].dtype == fmt.torch_scale_ab_dtype
                and _recipe_matches(scale_recipe_a[0], fmt.scale_ab_recipe)
                and _recipe_matches(scale_recipe_b[0], fmt.scale_ab_recipe)
            ):
                return fmt
            continue
        if len(scale_a) != 2 or len(scale_b) != 2:
            continue
        if len(scale_recipe_a) != 2 or len(scale_recipe_b) != 2:
            continue
        if (
            mat_a.dtype == fmt.torch_ab_dtype
            and mat_b.dtype == fmt.torch_ab_dtype
            and scale_a[0].dtype == fmt.torch_scale_ab_dtype
            and scale_b[0].dtype == fmt.torch_scale_ab_dtype
            and scale_a[1].dtype == fmt.torch_global_scale_dtype
            and scale_b[1].dtype == fmt.torch_global_scale_dtype
            and _recipe_matches(scale_recipe_a[0], fmt.scale_ab_recipe)
            and _recipe_matches(scale_recipe_a[1], ScalingType.TensorWise)
            and _recipe_matches(scale_recipe_b[0], fmt.scale_ab_recipe)
            and _recipe_matches(scale_recipe_b[1], ScalingType.TensorWise)
        ):
            return fmt
    return None


@functools.cache
def _cutedsl_unavailable_reason() -> str | None:
    deps = [
        ("nvidia-cutlass-dsl", "cutlass"),
        ("apache-tvm-ffi", "tvm_ffi"),
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
        "`nvidia-cutlass-dsl`, `apache-tvm-ffi`, and `cuda-bindings` "
        "(from NVIDIA cuda-python); "
        f"{reason}"
    )


@functools.cache
def _is_blackwell_device(device_id: int) -> bool:
    try:
        major, _ = torch.cuda.get_device_capability(device_id)
    except Exception:
        return False
    return major == 10


def _select_kernel_config_fp8(M: int, N: int, K: int) -> _KernelConfig:
    # Port of MSLK mx8mx8bf16_grouped::get_kernel_via_heuristics()
    cfg_256_64_ba = _KernelConfig((256, 64), (2, 1), True)
    cfg_256_128_ba = _KernelConfig((256, 128), (2, 1), True)
    cfg_256_256_ab = _KernelConfig((256, 256), (2, 1), False)
    cfg_256_256_ba = _KernelConfig((256, 256), (2, 1), True)

    M = int(M)
    N = int(N)
    K = int(K)

    # CuTeDSL-tuned overrides for large per-group workloads where the
    # MSLK heuristic tends to over-select transpose_ab=True.
    if M >= 2048 and N >= 6144 and K >= 2048:
        return cfg_256_256_ab
    if N == 5120 and K == 8192 and M <= 4096:
        return cfg_256_256_ab

    if M <= 64:
        return cfg_256_64_ba
    elif M <= 128:
        if N <= 1024:
            if K <= 5120:
                return cfg_256_64_ba
            else:
                return cfg_256_128_ba
        else:
            return cfg_256_128_ba
    elif M <= 1024:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            elif K <= 6144:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 2048:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 4096:
            if K <= 5120:
                return cfg_256_256_ba
            elif K <= 6144:
                return cfg_256_256_ab
            elif K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 5120:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 6144:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 6144:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 7168:
            if K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 4096:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 2048:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 2048:
            return cfg_256_256_ba
        elif N <= 4096:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 4096:
                return cfg_256_256_ba
            elif K <= 7168:
                return cfg_256_256_ab
            elif K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 6144:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 6144:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 7168:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 4096:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 4096:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 2048:
            if K <= 2048:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            elif K <= 6144:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 4096:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 5120:
            if K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 6144:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 4096:
                return cfg_256_256_ba
            elif K <= 5120:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 7168:
            if K <= 2048:
                return cfg_256_256_ba
            elif K <= 4096:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 4096:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 5120:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 2048:
            return cfg_256_256_ba
        elif N <= 4096:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 5120:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 6144:
                return cfg_256_256_ba
            elif K <= 7168:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 6144:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 7168:
            if K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 6144:
        if N <= 1024:
            return cfg_256_256_ba
        elif N <= 2048:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 4096:
            return cfg_256_256_ba
        elif N <= 5120:
            if K <= 7168:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 6144:
            return cfg_256_256_ba
        elif N <= 7168:
            if K <= 4096:
                return cfg_256_256_ba
            elif K <= 5120:
                return cfg_256_256_ab
            elif K <= 6144:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 1024:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 7168:
        if N <= 2048:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 4096:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 6144:
            return cfg_256_256_ba
        elif N <= 7168:
            if K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 2048:
                return cfg_256_256_ba
            elif K <= 5120:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 8192:
        if N <= 1024:
            return cfg_256_256_ba
        elif N <= 2048:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 8192:
            return cfg_256_256_ba
        else:
            if K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    else:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        else:
            return cfg_256_256_ba


def _select_kernel_config_fp4(M: int, N: int, K: int) -> _KernelConfig:
    # Port of MSLK f4f4bf16_grouped::get_kernel_via_heuristics().
    # The FP4 kernel family carries its K-tiling in the compiled
    # dtype-specific kernel, so the MSLK 256x256x128 vs 256x256x256
    # distinction collapses to a single 256x256 choice here.
    cfg_128_64 = _KernelConfig((128, 64), (1, 1), True)
    cfg_256_64 = _KernelConfig((256, 64), (2, 1), True)
    cfg_128_128 = _KernelConfig((128, 128), (1, 1), False)
    cfg_256_128 = _KernelConfig((256, 128), (2, 1), False)
    cfg_256_256 = _KernelConfig((256, 256), (2, 1), False)

    M = int(M)
    N = int(N)
    K = int(K)

    if M <= 1:
        return cfg_256_64
    if M <= 64:
        if N <= 1024:
            return cfg_256_64
        if N <= 2048:
            return cfg_128_64 if K <= 2048 else cfg_256_64
        if N <= 7168:
            return cfg_256_64
        if N <= 8192:
            if K <= 6144:
                return cfg_256_64
            if K <= 7168:
                return cfg_128_64
            return cfg_256_64
        return cfg_256_64
    if M <= 128:
        if N <= 8192:
            return cfg_256_128
        return cfg_256_128 if K <= 8192 else cfg_128_128
    return cfg_256_256


def _round_up(a: int, b: int) -> int:
    return ((a + b - 1) // b) * b


def _allocate_output(
    mat_a: Tensor, mat_b: Tensor, out_dtype: torch.dtype, ngroups: int
) -> Tensor | None:
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    M, N = mat_a.size(0), mat_b.size(-1)
    alignment = 128 // torch.finfo(out_dtype).bits
    N_padded = _round_up(N, alignment)
    if a_is_2d and b_is_2d:
        return torch.empty_strided(
            (ngroups, M, N),
            (M * N_padded, N_padded, 1),
            device=mat_a.device,
            dtype=out_dtype,
        )
    if a_is_2d and not b_is_2d:
        return torch.empty_strided(
            (M, N), (N_padded, 1), device=mat_a.device, dtype=out_dtype
        )
    return None


@functools.cache
def _compile_scaled_grouped_mm_blockscaled(
    sm_count: int,
    max_active_clusters: int,
    estimate_total_num_clusters: int,
    mma_tile_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    transpose_ab: bool,
    ab_dtype_name: str,
    scale_ab_dtype_name: str,
    scale_ab_vec_size: int,
    c_dtype_name: str,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from torch._cutedsl.scaled_grouped_mm_blockscaled_kernel import (
        Sm100GroupedBlockScaledGemmKernel,
    )

    m = cute.sym_int()
    n = cute.sym_int(divisibility=16)
    k = cute.sym_int(divisibility=scale_ab_vec_size)
    a_stride_m = cute.sym_int(divisibility=16)
    b_stride_n = cute.sym_int(divisibility=16)
    scale_a_m = cute.sym_int()
    scale_a_k = cute.sym_int()
    scale_b_n = cute.sym_int()
    scale_b_k = cute.sym_int()
    scale_a_stride_m = cute.sym_int()
    scale_a_stride_k = cute.sym_int()
    scale_b_stride_n = cute.sym_int()
    scale_b_stride_k = cute.sym_int()

    ab_dtype = getattr(cutlass, ab_dtype_name)
    scale_ab_dtype = getattr(cutlass, scale_ab_dtype_name)
    c_dtype = getattr(cutlass, c_dtype_name)
    c_stride_0 = cute.sym_int(divisibility=max(1, 16 * 8 // c_dtype.width))

    fake_a = make_fake_tensor(
        ab_dtype,
        (m, k, 1),
        stride=(a_stride_m, 1, 0),
    )
    fake_b = make_fake_tensor(
        ab_dtype,
        (n, k, 1),
        stride=(b_stride_n, 1, 0),
    )
    if transpose_ab:
        # In transpose mode, C is passed as (N, M, 1) with transpose strides.
        fake_c = make_fake_tensor(
            c_dtype,
            (n, m, 1),
            stride=(1, c_stride_0, 0),
        )
    else:
        fake_c = make_fake_tensor(
            c_dtype,
            (m, n, 1),
            stride=(c_stride_0, 1, 0),
        )
    fake_scale_a = make_fake_tensor(
        scale_ab_dtype,
        (scale_a_m, scale_a_k, 1),
        stride=(scale_a_stride_m, scale_a_stride_k, 0),
    )
    fake_scale_b = make_fake_tensor(
        scale_ab_dtype,
        (scale_b_n, scale_b_k, 1),
        stride=(scale_b_stride_n, scale_b_stride_k, 0),
    )

    g = cute.sym_int()
    fake_problem = make_fake_tensor(cutlass.Int32, (g, 4), stride=(4, 1))
    fake_strides = make_fake_tensor(cutlass.Int64, (g, 3, 2), stride=(6, 2, 1))
    fake_ptrs_abc = make_fake_tensor(cutlass.Int64, (g, 3), stride=(3, 1))
    fake_ptrs_scale = make_fake_tensor(cutlass.Int64, (g, 2), stride=(2, 1))
    fake_global_scale_ptrs = make_fake_tensor(cutlass.Int64, (g,), stride=(1,))
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
        sf_vec_size=scale_ab_vec_size,
        mma_tiler_mn=mma_tile_mn,
        cluster_shape_mn=cluster_shape_mn,
        transpose_ab=transpose_ab,
    )

    compiled = _compile_with_safe_names(
        lambda: cute.compile(
            grouped_gemm,
            initial_a=fake_a,
            initial_b=fake_b,
            initial_c=fake_c,
            initial_sfa=fake_scale_a,
            initial_sfb=fake_scale_b,
            tensor_addr_global_scale=fake_global_scale_ptrs,
            group_count=0,
            problem_shape_mnkl=fake_problem,
            strides_abc=fake_strides,
            tensor_address_abc=fake_ptrs_abc,
            tensor_address_sfasfb=fake_ptrs_scale,
            estimate_total_num_clusters=estimate_total_num_clusters,
            total_num_clusters=fake_total_clusters,
            tensormap_cute_tensor=fake_tensormap,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            options="--enable-assertions --enable-tvm-ffi",
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
def _get_max_threads_per_block(device_id: int) -> int:
    props = torch.cuda.get_device_properties(device_id)
    return int(getattr(props, "max_threads_per_block", 1024))


def _ceil_div_int(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_cluster_tile_shape_mn(
    mma_tile_mn: tuple[int, int], cluster_shape_mn: tuple[int, int]
) -> tuple[int, int]:
    cta_tile_m = mma_tile_mn[0]
    if mma_tile_mn[0] == 256:
        cta_tile_m //= 2
    return cta_tile_m * cluster_shape_mn[0], mma_tile_mn[1] * cluster_shape_mn[1]


def _estimate_total_clusters_for_launch(
    *,
    a_is_2d: bool,
    b_is_2d: bool,
    transpose_ab: bool,
    ngroups: int,
    M: int,
    N: int,
    cluster_tile_m: int,
    cluster_tile_n: int,
) -> int:
    if a_is_2d and b_is_2d:
        if transpose_ab:
            return (
                ngroups
                * _ceil_div_int(N, cluster_tile_m)
                * _ceil_div_int(M, cluster_tile_n)
            )
        return (
            ngroups
            * _ceil_div_int(M, cluster_tile_m)
            * _ceil_div_int(N, cluster_tile_n)
        )

    if transpose_ab:
        grouped_tiles = _ceil_div_int(M, cluster_tile_n) + max(ngroups - 1, 0)
        return _ceil_div_int(N, cluster_tile_m) * grouped_tiles

    grouped_tiles = _ceil_div_int(M, cluster_tile_m) + max(ngroups - 1, 0)
    return grouped_tiles * _ceil_div_int(N, cluster_tile_n)


@functools.cache
def _alloc_aux_tensors(
    device_index: int, cap: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = torch.device("cuda", device_index)
    ptrs_abc = torch.empty((cap, 3), device=device, dtype=torch.int64)
    ptrs_scale = torch.empty((cap, 2), device=device, dtype=torch.int64)
    ptrs_global_scale = torch.empty((cap,), device=device, dtype=torch.int64)
    problem_sizes = torch.empty((cap, 4), device=device, dtype=torch.int32)
    strides_abc = torch.empty((cap, 3, 2), device=device, dtype=torch.int64)
    total_num_clusters = torch.empty((1,), device=device, dtype=torch.int32)
    return (
        ptrs_abc,
        ptrs_scale,
        ptrs_global_scale,
        problem_sizes,
        strides_abc,
        total_num_clusters,
    )


def _get_aux_tensors(
    ngroups: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # Re-allocated when ngroups exceeds current capacity (rounded to
    # next power of 2).
    cap = max(64, 1 << (ngroups - 1).bit_length())
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    (
        ptrs_abc,
        ptrs_scale,
        ptrs_global_scale,
        problem_sizes,
        strides_abc,
        total_num_clusters,
    ) = _alloc_aux_tensors(device_index, cap)
    return (
        ptrs_abc[:ngroups],
        ptrs_scale[:ngroups],
        ptrs_global_scale[:ngroups],
        problem_sizes[:ngroups],
        strides_abc[:ngroups],
        total_num_clusters,
    )


@functools.cache
def _alloc_unit_global_scales(device_index: int, cap: int) -> Tensor:
    device = torch.device("cuda", device_index)
    return torch.ones((cap,), device=device, dtype=torch.float32)


def _get_unit_global_scales(ngroups: int, device: torch.device) -> Tensor:
    cap = max(64, 1 << (ngroups - 1).bit_length())
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return _alloc_unit_global_scales(device_index, cap)[:ngroups]


@functools.cache
def _alloc_tensormap(device_index: int, sm_count: int) -> Tensor:
    from torch._cutedsl.scaled_grouped_mm_blockscaled_kernel import (
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


def scaled_grouped_mm_blockscaled(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: list[Tensor],
    scale_b: list[Tensor],
    scale_recipe_a: list[ScalingType],
    scale_recipe_b: list[ScalingType],
    swizzle_a: list[SwizzleType],
    swizzle_b: list[SwizzleType],
    offs: Tensor | None,
    output_dtype: torch.dtype | None = None,
    contraction_dim: Sequence[int] = (),
    use_fast_accum: bool = False,
    bias: Tensor | None = None,
) -> Tensor:
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

    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    if not a_is_2d:
        raise ValueError(
            "CuTeDSL blockscaled path currently supports only 2d/2d and 2d/3d"
        )

    # tvm-ffi validates shape/dtype/layout constraints at runtime.
    if offs is None:
        raise ValueError("offs must be provided for scaled grouped MM")
    fmt = _get_blockscaled_format(
        mat_a, mat_b, scale_a, scale_b, scale_recipe_a, scale_recipe_b
    )
    if fmt is None:
        raise ValueError(
            "CuTeDSL blockscaled path currently supports only MXFP8, MXFP4, and NVFP4"
        )

    assert_cutedsl_runtime_available()

    if mat_a.device.type != "cuda":
        raise ValueError("scaled grouped MM blockscaled is only supported on CUDA")
    if mat_a.device != mat_b.device:
        raise ValueError("mat_a and mat_b must be on the same device")
    if any(scale.device != mat_a.device for scale in scale_a):
        raise ValueError("scale_a must be on the same device as mat_a")
    if any(scale.device != mat_a.device for scale in scale_b):
        raise ValueError("scale_b must be on the same device as mat_a")
    if offs.device != mat_a.device:
        raise ValueError("offs must be on the same device as mat_a")

    if bias is not None:
        raise ValueError("bias is not supported for scaled grouped MM")

    ngroups = int(offs.numel())
    mat_a_m = int(mat_a.size(0))
    mat_a_physical_k = int(mat_a.size(-1))
    mat_b_n = int(mat_b.size(-1))
    mat_b_physical_k = int(mat_b.size(0 if b_is_2d else -2))
    global_scales = None
    if fmt.torch_global_scale_dtype is not None:
        if scale_a[1].numel() != ngroups or scale_b[1].numel() != ngroups:
            raise ValueError(
                "NVFP4 global scales must have numel equal to offs.numel()"
            )
        global_scales = scale_a[1].reshape(-1).mul(scale_b[1].reshape(-1))
    requested_out_dtype = output_dtype or torch.bfloat16
    out = _allocate_output(mat_a, mat_b, requested_out_dtype, ngroups)
    if out is None:
        raise ValueError(
            "CuTeDSL blockscaled path currently supports only 2d/2d and 2d/3d"
        )
    if ngroups == 0:
        return out

    if mat_a.data_ptr() % 16 != 0:
        raise ValueError("expected data_ptr to be aligned to 16 bytes")
    if _is_transposed_layout(mat_a):
        raise ValueError("expected mat_a to not be transposed")

    if mat_b.data_ptr() % 16 != 0:
        raise ValueError("expected data_ptr to be aligned to 16 bytes")
    if not _is_transposed_layout(mat_b):
        raise ValueError("expected mat_b to be transposed")

    if use_fast_accum:
        raise ValueError("use_fast_accum is not supported for scaled grouped MM")
    logical_k_a = mat_a_physical_k * fmt.logical_vals_per_elem
    logical_k_b = mat_b_physical_k * fmt.logical_vals_per_elem
    if a_is_2d and not b_is_2d:
        if contraction_dim and tuple(contraction_dim) != (-1, -2):
            raise ValueError("contraction_dim must be (-1, -2) if provided")
        if logical_k_a != logical_k_b:
            raise ValueError("contraction dimension of mat_a and mat_b must match")

    if len(swizzle_a) != 1 or len(swizzle_b) != 1:
        raise ValueError("swizzle_a and swizzle_b must be singleton lists")
    if swizzle_a[0] != SwizzleType.SWIZZLE_32_4_4:
        raise ValueError(f"swizzle_a must be SWIZZLE_32_4_4 for {fmt.name}")
    if swizzle_b[0] != SwizzleType.SWIZZLE_32_4_4:
        raise ValueError(f"swizzle_b must be SWIZZLE_32_4_4 for {fmt.name}")

    if a_is_2d and b_is_2d and logical_k_a != logical_k_b:
        raise ValueError("for 2d/2d grouped gemm, total K dimensions must match")
    if a_is_2d and not b_is_2d and ngroups != int(mat_b.size(0)):
        raise ValueError("for 2d/3d grouped gemm, offs size must match mat_b.size(0)")

    device = mat_a.device
    device_id = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    max_threads = _get_max_threads_per_block(device_id)
    threads_per_block = min(ngroups, max_threads)
    num_blocks = (ngroups + threads_per_block - 1) // threads_per_block

    m_for_heuristic = mat_a_m if b_is_2d else (mat_a_m // max(ngroups, 1))
    k_for_heuristic = logical_k_a // max(ngroups, 1) if b_is_2d else logical_k_a
    if fmt.name in ("mxfp4", "nvfp4"):
        config = _select_kernel_config_fp4(m_for_heuristic, mat_b_n, k_for_heuristic)
    else:
        config = _select_kernel_config_fp8(m_for_heuristic, mat_b_n, k_for_heuristic)

    cluster_size = config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
    sm_count, max_active_clusters = _get_schedule_meta(cluster_size, device_id)
    cluster_tile_m, cluster_tile_n = _get_cluster_tile_shape_mn(
        config.mma_tile_mn, config.cluster_shape_mn
    )
    estimate_total_num_clusters = _estimate_total_clusters_for_launch(
        a_is_2d=a_is_2d,
        b_is_2d=b_is_2d,
        transpose_ab=config.transpose_ab,
        ngroups=ngroups,
        M=mat_a_m,
        N=mat_b_n,
        cluster_tile_m=cluster_tile_m,
        cluster_tile_n=cluster_tile_n,
    )

    scaled_grouped_mm_blockscaled_compiled, cluster_tile_shape_mnk = (
        _compile_scaled_grouped_mm_blockscaled(
            sm_count,
            max_active_clusters,
            estimate_total_num_clusters,
            config.mma_tile_mn,
            config.cluster_shape_mn,
            config.transpose_ab,
            fmt.cutlass_ab_dtype_name,
            fmt.cutlass_scale_ab_dtype_name,
            fmt.scale_ab_vec_size,
            _TORCH_TO_CUTLASS_DTYPE_NAME[requested_out_dtype],
        )
    )

    (
        ptrs_abc,
        ptrs_scale,
        ptrs_global_scale,
        problem_sizes,
        strides_abc,
        total_num_clusters,
    ) = _get_aux_tensors(ngroups, device)

    try:
        cluster_tile_m = int(cluster_tile_shape_mnk[0])
        cluster_tile_n = int(cluster_tile_shape_mnk[1])
    except Exception:
        cluster_tile_m, cluster_tile_n = _get_cluster_tile_shape_mn(
            config.mma_tile_mn, config.cluster_shape_mn
        )
    scaled_grouped_mm_prepare_metadata_compiled = (
        _compile_scaled_grouped_mm_prepare_metadata(a_is_2d, b_is_2d, threads_per_block)
    )

    import cuda.bindings.driver as cuda_driver

    stream = cuda_driver.CUstream(int(torch.cuda.current_stream().cuda_stream))

    if global_scales is None:
        global_scales = _get_unit_global_scales(ngroups, device)

    scale_a0 = scale_a[0]
    scale_b0 = scale_b[0]
    mat_a_ptr = int(mat_a.data_ptr())
    mat_b_ptr = int(mat_b.data_ptr())
    out_ptr = int(out.data_ptr())
    scale_a_ptr = int(scale_a0.data_ptr())
    scale_b_ptr = int(scale_b0.data_ptr())
    global_scale_ptr = int(global_scales.data_ptr())
    mat_a_stride = tuple(map(int, mat_a.stride()))
    mat_b_stride = tuple(map(int, mat_b.stride()))
    out_stride = tuple(map(int, out[0].stride() if b_is_2d else out.stride()))
    scale_a_stride = tuple(map(int, scale_a0.stride()))
    scale_b_stride = tuple(map(int, scale_b0.stride()))
    mat_a_element_size = int(mat_a.element_size())
    scale_a_element_size = int(scale_a0.element_size())
    out_element_size = int(out.element_size())

    if b_is_2d:
        if a_is_2d and fmt.logical_vals_per_elem > 1:
            stride_b_logical = (0, 1, logical_k_a)
        else:
            stride_b_logical = (0, mat_b_stride[0], mat_b_stride[1])
    elif fmt.logical_vals_per_elem > 1:
        stride_b_logical = (mat_b_stride[0], 1, logical_k_a)
    else:
        stride_b_logical = mat_b_stride

    scaled_grouped_mm_prepare_metadata_compiled(
        ngroups,
        mat_a_m,
        mat_b_n,
        logical_k_a,
        mat_a_ptr,
        mat_b_ptr,
        out_ptr,
        scale_a_ptr,
        scale_b_ptr,
        global_scale_ptr,
        offs,
        fmt.logical_vals_per_elem,
        fmt.scale_ab_vec_size,
        mat_a_element_size,
        scale_a_element_size,
        out_element_size,
        4,
        mat_a_stride,
        (0, mat_b_stride[0], mat_b_stride[1]) if b_is_2d else mat_b_stride,
        (
            (logical_k_a, 1)
            if a_is_2d and fmt.logical_vals_per_elem > 1
            else mat_a_stride
        ),
        stride_b_logical,
        out_stride,
        (
            (
                _round_up(logical_k_a // fmt.scale_ab_vec_size, 4),
                1,
            )
            if (not b_is_2d) and fmt.name == "nvfp4"
            else scale_a_stride
        ),
        scale_b_stride,
        int(config.transpose_ab),
        cluster_tile_m,
        cluster_tile_n,
        problem_sizes,
        ptrs_abc,
        ptrs_scale,
        ptrs_global_scale,
        strides_abc,
        total_num_clusters,
        num_blocks,
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

    scaled_grouped_mm_blockscaled_compiled(
        _with_l_dim(mat_a),
        _with_l_dim(mat_b.transpose(0, 1) if b_is_2d else mat_b[0].transpose(0, 1)),
        _with_l_dim(
            (out[0].transpose(0, 1) if config.transpose_ab else out[0])
            if b_is_2d
            else (out.transpose(0, 1) if config.transpose_ab else out)
        ),
        _with_l_dim(scale_a0),
        _with_l_dim(scale_b0),
        ptrs_global_scale,
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


_NN_LIB: Library | None = None


class _BoxedKernel(Protocol):
    def call_boxed(self, keyset: Any, *args: Any, **kwargs: Any) -> Any: ...


_ORIGINAL_SCALED_GROUPED_MM_V2_KERNEL: _BoxedKernel | None = None


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
    use_fast_accum: object,
) -> bool:
    if _cutedsl_unavailable_reason() is not None:
        return False
    if not isinstance(mat_a, Tensor) or not isinstance(mat_b, Tensor):
        return False
    if not isinstance(scale_a, list) or not isinstance(scale_b, list):
        return False
    if mat_a.device.type != "cuda" or mat_b.device.type != "cuda":
        return False
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    b_dim = mat_b.dim()
    if not a_is_2d or (not b_is_2d and b_dim != 3):
        return False
    device_id = (
        mat_a.device.index
        if mat_a.device.index is not None
        else torch.cuda.current_device()
    )
    if not _is_blackwell_device(device_id):
        return False
    if not isinstance(scale_recipe_a, list) or not isinstance(scale_recipe_b, list):
        return False
    if not isinstance(swizzle_a, list) or not isinstance(swizzle_b, list):
        return False
    if len(swizzle_a) != 1 or len(swizzle_b) != 1:
        return False
    if swizzle_a[0] != SwizzleType.SWIZZLE_32_4_4.value:
        return False
    if swizzle_b[0] != SwizzleType.SWIZZLE_32_4_4.value:
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


def _scaled_grouped_mm_v2_conditional_cuda_impl(dispatch_keys, *args, **kwargs):
    kernel = _ORIGINAL_SCALED_GROUPED_MM_V2_KERNEL
    if kernel is None:
        raise RuntimeError(
            "scaled_grouped_mm_blockscaled_register_kernels() must initialize "
            "the original aten::_scaled_grouped_mm_v2 CUDA kernel first"
        )
    if kwargs:
        return kernel.call_boxed(dispatch_keys, *args, **kwargs)
    # Schema (13 args total) has trailing optional defaults:
    # offs=None, bias=None, out_dtype=None, contraction_dim=[], use_fast_accum=False.
    # Callers may omit some/all trailing optionals, so accept positional arity 8..13.
    if len(args) < 8 or len(args) > 13:
        return kernel.call_boxed(dispatch_keys, *args)
    if len(args) < 13:
        trailing_defaults = (None, None, None, [], False)
        args = (*args, *trailing_defaults[len(args) - 8 :])

    (
        mat_a,
        mat_b,
        scale_a,
        scale_recipe_a,
        swizzle_a,
        scale_b,
        scale_recipe_b,
        swizzle_b,
        offs,
        bias,
        output_dtype,
        contraction_dim,
        use_fast_accum,
    ) = args

    if _should_use_cutedsl_scaled_grouped_mm_blockscaled(
        mat_a,
        mat_b,
        scale_a,
        scale_recipe_a,
        swizzle_a,
        scale_b,
        scale_recipe_b,
        swizzle_b,
        offs,
        bias,
        use_fast_accum,
    ):
        mat_a_t = cast(Tensor, mat_a)
        mat_b_t = cast(Tensor, mat_b)
        scale_a_t = cast(list[Tensor], scale_a)
        scale_b_t = cast(list[Tensor], scale_b)
        scale_recipe_a_t = cast(list[int], scale_recipe_a)
        scale_recipe_b_t = cast(list[int], scale_recipe_b)
        swizzle_a_t = cast(list[int], swizzle_a)
        swizzle_b_t = cast(list[int], swizzle_b)
        offs_t = cast(Tensor | None, offs)
        bias_t = cast(Tensor | None, bias)
        output_dtype_t = cast(torch.dtype | None, output_dtype)
        contraction_dim_t = cast(Sequence[int], contraction_dim)
        use_fast_accum_t = cast(bool, use_fast_accum)

        cutedsl_call = scaled_grouped_mm_blockscaled
        if torch.compiler.is_compiling():
            import torch._dynamo as torch_dynamo

            cutedsl_call = torch_dynamo.disable(cutedsl_call)
        return cutedsl_call(
            mat_a_t,
            mat_b_t,
            scale_a_t,
            scale_b_t,
            [ScalingType(v) for v in scale_recipe_a_t],
            [ScalingType(v) for v in scale_recipe_b_t],
            [SwizzleType(swizzle_a_t[0])],
            [SwizzleType(swizzle_b_t[0])],
            offs_t,
            output_dtype_t,
            contraction_dim_t,
            use_fast_accum_t,
            bias=bias_t,
        )

    return kernel.call_boxed(dispatch_keys, *args)


def scaled_grouped_mm_blockscaled_register_kernels() -> Library:
    global _NN_LIB, _ORIGINAL_SCALED_GROUPED_MM_V2_KERNEL
    if _NN_LIB is not None:
        return _NN_LIB

    # Capture original CUDA kernel before installing override.
    if _ORIGINAL_SCALED_GROUPED_MM_V2_KERNEL is None:
        dispatch_get = (
            torch._C._dispatch_get_computed_kernel_for_dispatch_key
        )  # pyrefly: ignore [missing-module-attribute]
        _ORIGINAL_SCALED_GROUPED_MM_V2_KERNEL = dispatch_get(
            "aten::_scaled_grouped_mm_v2", "CUDA"
        )

    lib = Library("aten", "IMPL", "CUDA")
    lib.impl(
        "_scaled_grouped_mm_v2",
        _scaled_grouped_mm_v2_conditional_cuda_impl,
        "CUDA",
        with_keyset=True,
    )
    _NN_LIB = lib
    return lib


__all__ = [
    "scaled_grouped_mm_blockscaled",
    "scaled_grouped_mm_blockscaled_register_kernels",
]
