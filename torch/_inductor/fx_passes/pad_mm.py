import functools
import itertools
import operator
import typing
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._inductor.runtime.runtime_utils
from torch import Tensor
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor import utils
from torch._inductor.autoheuristic.autoheuristic import (
    AHContext,
    AutoHeuristic,
    LocalFeedback,
)
from torch._inductor.autoheuristic.autoheuristic_utils import (
    context_add_strides,
    context_add_using_tf32,
    pad_mm_operations,
    pad_mm_precondition,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._mode_utils import no_dispatch

from ...utils._triton import has_triton
from ..pattern_matcher import (
    fwd_only,
    gen_register_replacement,
    joint_fwd_bwd,
    Match,
    ReplaceFn,
    SearchFn,
)


aten = torch.ops.aten


# This flag is only used for testing purpose.
# Changing it to True will ignore comparing do_bench times
# between original pattern and padded one.
_skip_do_bench_times = False


def fetch_fake_tensors(match: Match, kwarg_names: Sequence[str]) -> list[Tensor]:
    kwargs = match.kwargs
    return [kwargs[name].meta["val"] for name in kwarg_names]


def unwrap_fake_args(
    *arg_names: str,
) -> Callable[[Callable[..., Any]], Callable[[Match], Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[[Match], Any]:
        def wrapper(match: Match) -> Any:
            fake_tensors = fetch_fake_tensors(match, arg_names)
            return func(*fake_tensors)

        return wrapper

    return decorator


def get_alignment_size(x: Tensor) -> int:
    return get_alignment_size_dtype(x.dtype)


def get_alignment_size_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.float16 or dtype == torch.half or dtype == torch.bfloat16:
        return 8
    elif dtype == torch.float32 or dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor) -> bool:
    return (a.is_cuda and b.is_cuda) or (a.is_xpu and b.is_xpu)


def check_dtype(a: Tensor, b: Tensor) -> bool:
    return a.is_floating_point() and b.is_floating_point()


def realize_symbols(
    ds: torch.Size | tuple[torch.SymInt, ...],
) -> list[int]:
    """Helper to convert symbolic dimensions to their concrete hint values."""
    return [d if isinstance(d, int) else d.node.hint for d in ds]


def can_pad(
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    """
    Determines if an operation CAN be padded (safety checks).
    All logic related to whether it's safe to pad should be here.
    """

    # It's fine we have symbolic shapes or strides as long as they
    # have hints. Later, we will make sure we only pad non-symbolic dimensions.
    def valid_shape_and_stride(t: Tensor | None) -> bool:
        if t is None:
            return True

        symbolic_cnt = 0
        for x in t.size():
            if isinstance(x, int):
                continue
            elif utils.is_symbolic(x):
                # pyrefly: ignore [missing-attribute]
                if not x.node.has_hint():
                    return False
                symbolic_cnt += 1
            else:
                return False
        # filter out cases where all dimensions are symbolic
        if symbolic_cnt == len(t.size()):
            return False
        return all(
            # pyrefly: ignore [missing-attribute]
            isinstance(x, int) or (utils.is_symbolic(x) and x.node.has_hint())
            for x in t.stride()
        )

    # Basic safety checks
    if not torch._inductor.config.shape_padding:
        return False

    if not check_device(mat1, mat2):
        return False

    if not check_dtype(mat1, mat2):
        return False

    if not all(valid_shape_and_stride(t) for t in (mat1, mat2, input)):
        return False

    # Check for zero dimensions - not safe to pad
    if any(
        dim == 0
        for dim in itertools.chain(
            realize_symbols(mat1.shape), realize_symbols(mat2.shape)
        )
    ):
        return False

    # Calculate padding lengths to check if padding is needed
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
        else:
            return False

        k_padded_length = get_padded_length(k, get_alignment_size(mat1))
        n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        m_padded_length = get_padded_length(m, get_alignment_size(mat1))

        # No padding needed - can't pad if there's nothing to pad
        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False

    # In deterministic mode, we can use heuristics instead of benchmarking
    # Check this after other basic checks so force_shape_pad can override
    if torch._inductor.config.deterministic:
        heuristic = torch._inductor.config.pad_mm_heuristic
        # If heuristic is "none", disable padding in deterministic mode (legacy behavior)
        if heuristic == "none" and not torch._inductor.config.force_shape_pad:
            return False
        # Otherwise, allow padding - _should_pad will use heuristics

    # Triton availability check - required for padding to work
    if not has_triton():
        return False

    return True


def get_padded_length(x: int | torch.SymInt, alignment_size: int) -> int:
    # we don't pad x if it is symbolic
    if isinstance(x, torch.SymInt) or alignment_size == 0 or x % alignment_size == 0:
        return 0

    # ignore dim that can be squeezed away
    if x == 1:
        return 0

    return int((x // alignment_size + 1) * alignment_size) - x


def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    return torch.cat([x, pad], dim=dim)


def addmm_pattern(
    input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float
) -> Tensor:
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def should_pad_addmm(match: Match) -> bool:
    mat1, mat2, input = fetch_fake_tensors(match, ("mat1", "mat2", "input"))
    return should_pad(match, mat1, mat2, torch.ops.aten.addmm, input=input)


def pad_addmm(
    input: Tensor | None,
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    beta: float = 1.0,
    alpha: float = 1.0,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    # for paddings, dim order is reversed for some reasons
    # and for every dim, we need to specify left and right padding
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )

    # the add broadcasts, so we only pad if the dimension != 1
    if input is not None:
        if n_padded_length != 0:
            if input.dim() == 2 and input.shape[1] != 1:
                input = pad_dim(input, n_padded_length, 1)
            elif input.dim() == 1 and input.shape[0] != 1:
                input = pad_dim(input, n_padded_length, 0)
        if m_padded_length != 0 and input.dim() == 2 and input.shape[0] != 1:
            input = pad_dim(input, m_padded_length, 0)

    res = aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def addmm_replace(
    input: Tensor | None,
    mat1: Tensor,
    mat2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    return pad_addmm(
        input,
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        beta,
        alpha,
    )


def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    denominator = M * K + N * K + M * N
    if denominator == 0:
        return False
    arithmetic_intensity = (M * N * K) / denominator

    # we have experienced some large perf hits in this case, even in bandwidth bound regimes
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and (torch.xpu.is_available() or torch.cuda.get_device_capability() < (9, 0))
    ):  # doesn't repro on h100s:
        return True

    # Fails with AMD
    try:
        machine_balance = (
            1000 * utils.get_device_tflops(dtype)
        ) / utils.get_gpu_dram_gbps()
    except Exception:
        return True

    # dram_gbps might be underestimating bandwidth because of cache.
    # if we estimate machine balance too low we might miss some speedups,
    # if we estimate too high there will be unnecessary compilation time increase.
    # TODO - finetune coefficient here. As a reference point, Triton mm model assumes
    # 80% of reads are in cache and cache is 4x faster than dram_gbps
    machine_balance = machine_balance * 0.5

    return arithmetic_intensity > machine_balance


def _get_dtype_bytes(dtype: torch.dtype) -> int:
    """Return the size in bytes for a given dtype."""
    if dtype in (torch.float16, torch.bfloat16, torch.half):
        return 2
    elif dtype in (torch.float32, torch.float):
        return 4
    elif dtype in (torch.float64, torch.double):
        return 8
    elif dtype == torch.int8:
        return 1
    elif dtype == torch.int16:
        return 2
    elif dtype == torch.int32:
        return 4
    elif dtype == torch.int64:
        return 8
    else:
        # Default to 4 bytes for unknown types
        return 4


@functools.cache
def _get_alignment_efficiency_table() -> dict[tuple[int, int], dict[int, float]]:
    """
    Return alignment efficiency lookup table based on compute capability and alignment.

    The efficiency represents the fraction of peak throughput achieved when matrix
    dimensions are not aligned to optimal boundaries. This is based on:
    - Memory coalescing efficiency for unaligned accesses
    - Tensor core utilization (for SM >= 7.0)
    - Bank conflict penalties

    Keys: (major, minor) compute capability
    Values: dict mapping misalignment (in elements) to efficiency factor

    The efficiency factors are derived from empirical measurements and NVIDIA
    documentation on memory access patterns. Unaligned accesses can cause:
    - Extra memory transactions (128-byte alignment for L1)
    - Reduced tensor core utilization
    - Bank conflicts in shared memory

    NOTE: These values are for fp16/bf16 tensor core operations. float32 operations
    do not benefit from padding on modern GPUs (see _should_skip_padding_for_dtype).
    """
    # Default efficiency for various misalignment scenarios
    # Misalignment of 0 means perfectly aligned (efficiency = 1.0)
    # Higher misalignment generally means lower efficiency

    # Volta (SM 7.0, 7.2) - First gen tensor cores, sensitive to alignment
    volta_efficiency = {
        0: 1.0,    # Perfectly aligned
        1: 0.70,   # 1-element misalignment
        2: 0.75,   # 2-element misalignment
        3: 0.70,   # 3-element misalignment
        4: 0.85,   # 4-element misalignment (half-warp aligned)
        5: 0.70,
        6: 0.75,
        7: 0.70,
    }

    # Ampere (SM 8.0, 8.6, 8.7) - Better handling of misalignment
    ampere_efficiency = {
        0: 1.0,
        1: 0.80,
        2: 0.85,
        3: 0.80,
        4: 0.90,
        5: 0.80,
        6: 0.85,
        7: 0.80,
    }

    # Hopper (SM 9.0) - Much better misalignment tolerance
    hopper_efficiency = {
        0: 1.0,
        1: 0.90,
        2: 0.92,
        3: 0.90,
        4: 0.95,
        5: 0.90,
        6: 0.92,
        7: 0.90,
    }

    # Blackwell (SM 10.0) - Fitted from benchmark measurements on NVIDIA B200
    # Misaligned K dimension causes severe performance degradation for fp16/bf16
    # These values represent padded_time / unpadded_time ratio
    blackwell_efficiency = {
        0: 1.0,     # Perfectly aligned
        1: 0.14,    # 1-element misalignment - severe penalty
        2: 0.14,    # Interpolated
        3: 0.14,    # Interpolated
        4: 0.14,    # Interpolated
        5: 0.14,    # Interpolated
        6: 0.14,    # Interpolated
        7: 0.13,    # 7-element misalignment - severe penalty
    }

    return {
        (7, 0): volta_efficiency,
        (7, 2): volta_efficiency,
        (7, 5): volta_efficiency,  # Turing
        (8, 0): ampere_efficiency,
        (8, 6): ampere_efficiency,
        (8, 7): ampere_efficiency,
        (8, 9): ampere_efficiency,  # Ada Lovelace
        (9, 0): hopper_efficiency,
        (10, 0): blackwell_efficiency,
    }


def _should_skip_padding_for_dtype(dtype: torch.dtype) -> bool:
    """
    Check if padding should be skipped for the given dtype.

    Based on benchmark measurements:
    - float32 operations do NOT benefit from padding on modern GPUs (SM 10.0+)
    - float16/bfloat16 operations benefit significantly from padding
    """
    if not torch.cuda.is_available():
        return False

    try:
        capability = torch.cuda.get_device_capability()
    except Exception:
        return False

    # On Blackwell (SM 10.0+), float32 does not benefit from padding
    if capability[0] >= 10 and dtype in (torch.float32, torch.float):
        return True

    return False


# Minimum problem size (M * K * N) below which padding overhead dominates
# Based on benchmark measurements: small matrices don't benefit from padding
# even with severe alignment penalties because the padding overhead is too high
# On B200: 512^3 doesn't benefit, 1024^3 does benefit
_MIN_PROBLEM_SIZE_FOR_PADDING = 512 * 512 * 512  # ~134M elements


def _is_matrix_column_major(mat: Tensor, is_bmm: bool = False) -> bool:
    """
    Check if a matrix is column-major (transposed).

    For 2D matrix: stride = (K, 1) is row-major, stride = (1, M) is column-major
    For 3D batch matrix: similar logic for last two dimensions

    Returns True if the matrix is column-major (the first non-batch dimension
    has stride 1, meaning that's the fast-moving dimension in memory).
    """
    strides = mat.stride()
    if is_bmm:
        # For batched: check if dim 1 (M dim) has smaller stride than dim 2 (K dim)
        return strides[1] < strides[2]
    else:
        # For 2D: check if dim 0 (M dim) has smaller stride than dim 1 (K dim)
        return strides[0] < strides[1]


def _get_alignment_efficiency(
    m: int, k: int, n: int, dtype: torch.dtype,
    mat1_col_major: bool = False, mat2_col_major: bool = False
) -> float:
    """
    Estimate the efficiency factor for unaligned matmul based on dimension alignment.

    This estimates what fraction of peak FLOPS the unpadded matmul will achieve
    due to alignment issues. Returns a value between 0 and 1.
    """
    if not torch.cuda.is_available():
        return 1.0

    try:
        capability = torch.cuda.get_device_capability()
    except Exception:
        return 1.0

    alignment_size = get_alignment_size_dtype(dtype)
    if alignment_size == 0:
        return 1.0

    # Calculate misalignment for each dimension
    m_misalign = m % alignment_size
    k_misalign = k % alignment_size
    n_misalign = n % alignment_size

    efficiency_table = _get_alignment_efficiency_table()
    cc = (capability[0], capability[1])

    # Find the closest matching compute capability
    if cc not in efficiency_table:
        # Fall back to closest known capability
        if capability[0] >= 10:
            cc = (10, 0)
        elif capability[0] >= 9:
            cc = (9, 0)
        elif capability[0] >= 8:
            cc = (8, 0)
        else:
            cc = (7, 0)

    eff_table = efficiency_table.get(cc, efficiency_table[(8, 0)])

    # Get efficiency for each dimension's misalignment
    m_eff = eff_table.get(m_misalign, 0.75)
    k_eff = eff_table.get(k_misalign, 0.75)
    n_eff = eff_table.get(n_misalign, 0.75)

    # On Blackwell (SM 10.0+), M misalignment doesn't cause performance issues
    # Based on benchmarks: M-only misalignment shows no benefit from padding
    # while K and N misalignment show severe penalties (up to 7x slower)
    if capability[0] >= 10:
        m_eff = 1.0

    # When mat1 (A) is column-major (TN layout), K misalignment doesn't hurt
    # performance. This is because in column-major A, the K dimension is the
    # slow-moving dimension (stride > 1), so memory access pattern doesn't
    # depend on K being aligned. Benchmark measurements confirm this:
    # TN layout shows NO alignment penalty for K, while NN/NT/TT layouts do.
    if mat1_col_major:
        k_eff = 1.0

    # Combined efficiency - K dimension affects both A and B matrices
    # M affects output and A, N affects output and B
    # Use geometric mean weighted by importance (K weighted double)
    combined_efficiency = (m_eff * k_eff * k_eff * n_eff) ** 0.25

    return combined_efficiency


def _estimate_copy_time_ns(
    num_elements: int,
    dtype: torch.dtype,
) -> float:
    """
    Estimate the time to copy num_elements at peak memory bandwidth.
    Returns time in nanoseconds.
    """
    try:
        bandwidth_gbps = utils.get_gpu_dram_gbps()
    except Exception:
        # Default to reasonable bandwidth if unavailable
        bandwidth_gbps = 1000.0

    bytes_to_copy = num_elements * _get_dtype_bytes(dtype)
    # bandwidth_gbps is in GB/s, convert to bytes/ns (same ratio)
    time_ns = bytes_to_copy / bandwidth_gbps
    return time_ns


def _estimate_gemm_time_ns(
    m: int, k: int, n: int, dtype: torch.dtype, efficiency: float = 1.0
) -> float:
    """
    Estimate GEMM time using roofline model at peak FLOPS.
    Returns time in nanoseconds.

    Args:
        m, k, n: Matrix dimensions
        dtype: Data type
        efficiency: Efficiency factor (0-1) accounting for alignment etc.
    """
    try:
        tflops = utils.get_device_tflops(dtype)
    except Exception:
        # Default to reasonable TFLOPS if unavailable
        tflops = 100.0

    # GEMM requires 2*M*N*K FLOPs (multiply-add)
    flops = 2.0 * m * n * k

    # tflops is in TFLOPS (10^12 FLOPS)
    # Convert to FLOPS/ns: TFLOPS * 10^12 / 10^9 = TFLOPS * 10^3
    flops_per_ns = tflops * 1000.0

    # Apply efficiency factor
    effective_flops_per_ns = flops_per_ns * efficiency

    if effective_flops_per_ns == 0:
        return float("inf")

    time_ns = flops / effective_flops_per_ns
    return time_ns


def should_pad_heuristic(
    m: int,
    k: int,
    n: int,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    dtype: torch.dtype,
    mat1_col_major: bool = False,
    mat2_col_major: bool = False,
) -> bool:
    """
    Heuristic-based decision for whether padding is beneficial.

    This is used in deterministic mode when benchmarking is not allowed.
    The decision is based on:
    1. Cost: Extra copies for padding A and B matrices
    2. Benefit: Improved GEMM efficiency from aligned dimensions

    The cost is estimated at peak memory bandwidth.
    The GEMM time is estimated using roofline model with peak FLOPS,
    applying an efficiency factor for unaligned access.

    Args:
        mat1_col_major: True if mat1 (A) is column-major (TN/TT layout)
        mat2_col_major: True if mat2 (B) is column-major (NT/TT layout)
    """
    # Skip padding for dtypes that don't benefit (e.g., float32 on Blackwell)
    if _should_skip_padding_for_dtype(dtype):
        return False

    # Skip padding for small matrices where overhead dominates
    problem_size = m * k * n
    if problem_size < _MIN_PROBLEM_SIZE_FOR_PADDING:
        return False

    # Calculate padded dimensions
    m_padded = m + m_padded_length
    k_padded = k + k_padded_length
    n_padded = n + n_padded_length

    # Estimate copy overhead for padding
    # Padding A: need to copy m*k elements and zero out padding
    # Padding B: need to copy k*n elements and zero out padding
    # In practice, constant_pad_nd does a copy with zeros for padding

    # Cost of padding operations
    # A: original is m x k, padded is m_padded x k_padded
    # B: original is k x n, padded is k_padded x n_padded
    # We need to write out the full padded tensors
    a_pad_elements = m_padded * k_padded
    b_pad_elements = k_padded * n_padded

    # Also need to slice the output (minor cost, but include it)
    output_slice_elements = m * n if (m_padded_length > 0 or n_padded_length > 0) else 0

    total_copy_elements = a_pad_elements + b_pad_elements + output_slice_elements
    copy_time = _estimate_copy_time_ns(total_copy_elements, dtype)

    # Estimate GEMM times
    # Unpadded GEMM with alignment penalty
    unpadded_efficiency = _get_alignment_efficiency(
        m, k, n, dtype, mat1_col_major, mat2_col_major
    )
    unpadded_gemm_time = _estimate_gemm_time_ns(m, k, n, dtype, unpadded_efficiency)

    # Padded GEMM at full efficiency (aligned)
    padded_gemm_time = _estimate_gemm_time_ns(
        m_padded, k_padded, n_padded, dtype, efficiency=1.0
    )

    # Total time comparison
    unpadded_total = unpadded_gemm_time
    padded_total = padded_gemm_time + copy_time

    # Apply a small safety margin (5%) - prefer not padding if marginal
    return padded_total * 1.05 < unpadded_total


def _nvmatmul_heuristics_available() -> bool:
    """Check if nvMatmulHeuristics is available."""
    try:
        import nvMatmulHeuristics  # noqa: F401

        return True
    except ImportError:
        return False


@functools.cache
def _get_nvmatmul_heuristics_interface(precision: str):
    """
    Get a cached nvMatmulHeuristics interface for the given precision.

    The interface is cached to avoid repeated initialization overhead.
    """
    from nvMatmulHeuristics import (
        NvMatmulHeuristicsInterface,
        NvMatmulHeuristicsTarget,
    )

    return NvMatmulHeuristicsInterface(
        backend=NvMatmulHeuristicsTarget.GENERIC,
        precision=precision,
    )


def _dtype_to_nvmatmul_precision(dtype: torch.dtype) -> str:
    """
    Convert torch dtype to nvMatmulHeuristics precision string.

    Precision strings follow the format: [A dtype][B dtype][C/D dtype]
    Common values:
    - 'HSS': Half input, Single accumulate, Single output
    - 'HHS': Half input, Half accumulate, Single output
    - 'HHH': Half input, Half accumulate, Half output
    - 'SSS': Single input, Single accumulate, Single output
    """
    if dtype in (torch.float16, torch.half):
        return "HSS"  # Half compute, Single accumulate
    elif dtype == torch.bfloat16:
        return "BSS"  # BFloat16 compute, Single accumulate
    elif dtype in (torch.float32, torch.float):
        return "SSS"  # Single precision
    elif dtype in (torch.float64, torch.double):
        return "DDD"  # Double precision
    else:
        return "SSS"  # Default to single


def _get_nvmatmul_estimated_runtime(
    m: int, k: int, n: int, dtype: torch.dtype
) -> float:
    """
    Get estimated runtime from nvMatmulHeuristics.

    Returns the estimated runtime in seconds for the best kernel configuration,
    or float('inf') on error.
    """
    try:
        from nvMatmulHeuristics import NvMatmulHeuristicsMatmulLayout

        precision = _dtype_to_nvmatmul_precision(dtype)
        heuristics = _get_nvmatmul_heuristics_interface(precision)

        # Query for best kernel configuration
        # Note: nvMatmulHeuristics uses (M, N, K) order
        configs = heuristics.get_with_mnk(
            m, n, k,
            NvMatmulHeuristicsMatmulLayout.NN_ROW_MAJOR,
            count=1,
        )

        if configs:
            return float(configs[0]["runtime"])
        return float("inf")
    except Exception:
        return float("inf")


def should_pad_nvmatmul_heuristic(
    m: int,
    k: int,
    n: int,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    dtype: torch.dtype,
    mat1_col_major: bool = False,
    mat2_col_major: bool = False,
) -> bool:
    """
    Heuristic-based decision using NVIDIA's nvMatmulHeuristics library.

    This function uses nvMatmulHeuristics to estimate kernel execution times
    for both padded and unpadded matrix configurations. The library provides
    performance models trained on real hardware measurements.

    The decision compares:
    1. Estimated GEMM runtime for unpadded dimensions
    2. Estimated GEMM runtime for padded dimensions + padding copy overhead

    Falls back to roofline heuristic if nvMatmulHeuristics is not available.

    Args:
        mat1_col_major: True if mat1 (A) is column-major (TN/TT layout)
        mat2_col_major: True if mat2 (B) is column-major (NT/TT layout)
    """
    if not _nvmatmul_heuristics_available():
        # Fall back to roofline heuristic
        return should_pad_heuristic(
            m, k, n,
            m_padded_length, k_padded_length, n_padded_length,
            dtype,
            mat1_col_major, mat2_col_major,
        )

    # Calculate padded dimensions
    m_padded = m + m_padded_length
    k_padded = k + k_padded_length
    n_padded = n + n_padded_length

    # Get estimated runtime for unpadded and padded configurations
    unpadded_runtime = _get_nvmatmul_estimated_runtime(m, k, n, dtype)
    padded_runtime = _get_nvmatmul_estimated_runtime(m_padded, k_padded, n_padded, dtype)

    # If we couldn't get valid estimates, fall back to roofline
    if unpadded_runtime == float("inf") or padded_runtime == float("inf"):
        return should_pad_heuristic(
            m, k, n,
            m_padded_length, k_padded_length, n_padded_length,
            dtype,
            mat1_col_major, mat2_col_major,
        )

    # Estimate padding copy overhead
    # Padding requires writing padded A and B tensors
    copy_time = _estimate_copy_time_ns(
        m_padded * k_padded + k_padded * n_padded, dtype
    )
    # Add output slice time if M or N changed
    if m_padded_length > 0 or n_padded_length > 0:
        copy_time += _estimate_copy_time_ns(m * n, dtype)

    # Convert copy_time from ns to seconds (runtime is in seconds)
    copy_time_s = copy_time * 1e-9

    # Total time comparison
    unpadded_total = unpadded_runtime
    padded_total = padded_runtime + copy_time_s

    # Apply safety margin (5%) - prefer not padding if marginal
    return padded_total * 1.05 < unpadded_total


@functools.cache
def get_pad_cache() -> torch._inductor.codecache.LocalCache:
    return torch._inductor.codecache.LocalCache()


def get_cached_should_pad(key: str) -> bool:
    return get_pad_cache().lookup(key)  # type: ignore[return-value]


def set_cached_should_pad(key: str, value: bool) -> None:
    return get_pad_cache().set_value(key, value=value)


def get_cached_base_mm_benchmark_time(key: str) -> float:
    return get_pad_cache().lookup(key)  # type: ignore[return-value]


def set_cached_base_mm_benchmark_time(key: str, value: float) -> None:
    return get_pad_cache().set_value(key, value=value)


def should_pad_bench_key(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
    is_base_time_key: bool = False,
) -> str:
    def tensor_key(t: Tensor) -> tuple[torch.Size, tuple[int, ...], torch.dtype]:
        return (t.shape, t.stride(), t.dtype)

    tf32_key = (
        None
        if mat1.dtype != torch.float32
        else torch.backends.cuda.matmul.allow_tf32 or torch.backends.mkldnn.allow_tf32
    )

    def fmt_pad(name: str) -> str | None:
        if is_base_time_key:
            return None
        return f"exclude_pad:{should_exclude_padding_time(match, name)}"

    key = (
        tensor_key(mat1),
        tensor_key(mat2),
        fmt_pad("mat1"),
        fmt_pad("mat2"),
        op,
        input if input is None else tensor_key(input),
        tf32_key,
    )

    key = str(key)
    if is_base_time_key:
        key = f"base mm time: {key}"
    return key


def get_non_view_def(node: torch.fx.Node) -> torch.fx.Node:
    if node.op is operator.getitem:
        return get_non_view_def(node.args[0])  # type: ignore[arg-type]

    if (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and utils.is_view(node.target)
    ):
        return get_non_view_def(node.all_input_nodes[0])

    return node


def should_exclude_padding_time(match: Match, arg_name: str) -> bool:
    node_def = get_non_view_def(match.kwargs[arg_name])

    # constant padding converts tensors to contiguous so even if the input tensor
    # can be planned layout transform is not free. TODO - way to pad and preserve layout ?
    if not fetch_fake_tensors(match, (arg_name,))[0].is_contiguous():
        return False

    # TODO - see issue https://github.com/pytorch/pytorch/issues/128889
    # We would only able to completely plan these out if we were only doing
    # first dimension padding. non-first we would still need a copy
    # because these outputs are fixed dense.
    cannot_plan_output = [
        aten.mm.default,
        aten.convolution.default,
        aten.convolution_backward.default,
        aten.bmm.default,
        aten.addmm.default,
        aten._scaled_dot_product_flash_attention.default,
        aten._scaled_dot_product_efficient_attention.default,
    ]

    if node_def.target in cannot_plan_output:
        return False

    if (
        node_def.target is aten.cat.default
        and len(node_def.all_input_nodes)
        > torch._inductor.config.max_pointwise_cat_inputs
    ):
        return False

    # optimistically assume we should be able to memory plan away
    # all non inputs
    return node_def.op != "placeholder"


def is_padded_faster(key: str, ori_time: float, pad_time: float) -> bool:
    """
    Determines if padding is beneficial by comparing benchmark times.
    Helper function that applies a multiplier to account for memory ops overhead.
    """
    multiplier = 1.1
    # Shape padding introduces additional memory ops. Based on microbenchmarks, 1.1x represents a reasonable
    # tradeoff between performance improvement from shape padding and overhead from additional memory ops
    # TODO: Build a learned model which would be better than this heuristic
    if "shape_padding_multiplier" in torch._inductor.config.post_grad_fusion_options:
        multiplier = torch._inductor.config.post_grad_fusion_options[
            "shape_padding_multiplier"
        ].get("value", 1.1)
        counters["inductor"]["shape_padding_multiplier"] += 1
    padded_is_faster = _skip_do_bench_times or ori_time > pad_time * multiplier
    set_cached_should_pad(key, padded_is_faster)
    return padded_is_faster


def should_pad_mm_bf16(dtype: torch.dtype, M: int, N: int, K: int) -> bool:
    # always force pad for mm with bf16 when the following are satisfied to avoid perf regression
    large_k_threshold_to_pad = torch._inductor.config.post_grad_fusion_options[
        "pad_aten_mm_pass"
    ].get("k_threshold_to_pad", 8388608)
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and N % 2 == 1
        and K >= large_k_threshold_to_pad
        and (torch.xpu.is_available() or torch.cuda.get_device_capability() < (9, 0))
    ):  # doesn't repro on h100s:
        return True
    return False


def should_pad(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    _can_pad = can_pad(mat1, mat2, op, input)
    with dynamo_timed(
        "pad_mm_benchmark",
        log_pt2_compile_event=False,
        dynamo_compile_column_us="compile_time_autotune_time_us",
    ):
        return _can_pad and _should_pad(match, mat1, mat2, op, input)


def get_do_bench() -> Callable[[Callable[[], Any]], float]:
    with dynamo_timed("pad_mm_benchmark_get_do_bench"):
        return functools.partial(
            # pyrefly: ignore [bad-argument-type]
            torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
            warmup=5,
        )


def _should_pad(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    """
    Determines if an operation SHOULD be padded (performance checks).
    All logic related to whether padding would be performant should be here.
    """
    do_bench = get_do_bench()

    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        else:
            return False

        # Force padding when explicitly requested - performance override
        if torch._inductor.config.force_shape_pad:
            return True

        # Performance heuristic for bf16 large K scenarios
        if (
            "pad_aten_mm_pass" in torch._inductor.config.post_grad_fusion_options
            and should_pad_mm_bf16(mat1.dtype, m, n, k)
        ):
            return True

        # Check if operation is compute bound (performance check)
        if not is_mm_compute_bound(m, k, n, mat1.dtype):
            return False

        # In deterministic mode, use heuristics instead of benchmarking
        if torch._inductor.config.deterministic:
            heuristic = torch._inductor.config.pad_mm_heuristic
            if heuristic == "roofline":
                # Convert symbolic dimensions to concrete values for heuristic
                m_val = int(m) if isinstance(m, int) else m.node.hint
                k_val = int(k) if isinstance(k, int) else k.node.hint
                n_val = int(n) if isinstance(n, int) else n.node.hint
                # Detect matrix layout from strides
                is_bmm = op is torch.ops.aten.bmm
                mat1_col_major = _is_matrix_column_major(mat1, is_bmm)
                mat2_col_major = _is_matrix_column_major(mat2, is_bmm)
                result = should_pad_heuristic(
                    m_val, k_val, n_val,
                    m_padded_length, k_padded_length, n_padded_length,
                    mat1.dtype,
                    mat1_col_major, mat2_col_major,
                )
                # Cache the heuristic result for consistency
                key = should_pad_bench_key(match, mat1, mat2, op, input)
                set_cached_should_pad(key, result)
                return result
            elif heuristic == "nvmatmul":
                # Use cuBLASLt heuristics for GEMM time estimation
                m_val = int(m) if isinstance(m, int) else m.node.hint
                k_val = int(k) if isinstance(k, int) else k.node.hint
                n_val = int(n) if isinstance(n, int) else n.node.hint
                # Detect matrix layout from strides
                is_bmm = op is torch.ops.aten.bmm
                mat1_col_major = _is_matrix_column_major(mat1, is_bmm)
                mat2_col_major = _is_matrix_column_major(mat2, is_bmm)
                result = should_pad_nvmatmul_heuristic(
                    m_val, k_val, n_val,
                    m_padded_length, k_padded_length, n_padded_length,
                    mat1.dtype,
                    mat1_col_major, mat2_col_major,
                )
                key = should_pad_bench_key(match, mat1, mat2, op, input)
                set_cached_should_pad(key, result)
                return result
            # If heuristic is "none", we shouldn't reach here (can_pad returns False)
            # but handle it gracefully by falling through to benchmarking

        # We don't want to look up the cache for cases that are trivially false
        # since it does file io
        key = should_pad_bench_key(match, mat1, mat2, op, input)

        cached_pad = get_cached_should_pad(key)
        if cached_pad is not None:
            return cached_pad

        def realize_tensor(t):
            if isinstance(t, FakeTensor):
                size_hints = realize_symbols(t.size())
                # pyrefly: ignore [bad-argument-type]
                stride_hint = realize_symbols(t.stride())
                real_size = (
                    sum((d - 1) * s for d, s in zip(size_hints, stride_hint)) + 1
                )
                real_t = torch.randn(real_size, dtype=t.dtype, device=t.device)
                return torch.as_strided(real_t, size_hints, stride_hint)
            else:
                return torch.randn_like(t)

        mat1 = realize_tensor(mat1)
        mat2 = realize_tensor(mat2)

        # since we key on whether or not the inputs can be memory planned, set cache for the
        # original time which is unaffected by whether or not the input can be planned
        ori_time_key = should_pad_bench_key(
            match, mat1, mat2, op, input, is_base_time_key=True
        )
        ori_time = get_cached_base_mm_benchmark_time(ori_time_key)
        if ori_time is None and op is torch.ops.aten.addmm and input is not None:
            # realize bias for addmm
            input = realize_tensor(input)

        mat1_pad = mat1
        mat2_pad = mat2

        is_bmm = op is torch.ops.aten.bmm

        mat1_pre_padded = should_exclude_padding_time(match, "mat1")
        fns = []
        if mat1_pre_padded and (m_padded_length or k_padded_length):
            mat1_pad = pad_mat1(
                mat1_pad,
                m_padded_length=m_padded_length,
                k_padded_length=k_padded_length,
                is_bmm=is_bmm,
            )

            def write_pad():
                if is_bmm:
                    mat1_pad[:, -m_padded_length:, -k_padded_length:].fill_(0)
                else:
                    mat1_pad[-m_padded_length:, -k_padded_length:].fill_(0)

            fns.append(write_pad)

        mat2_pre_padded = should_exclude_padding_time(match, "mat2")
        if mat2_pre_padded and (k_padded_length or n_padded_length):
            mat2_pad = pad_mat2(
                mat2_pad,
                k_padded_length=k_padded_length,
                n_padded_length=n_padded_length,
                is_bmm=is_bmm,
            )

            def write_pad():
                if is_bmm:
                    mat2_pad[:, -k_padded_length:, -n_padded_length:].fill_(0)
                else:
                    mat2_pad[-k_padded_length:, -n_padded_length:].fill_(0)

            fns.append(write_pad)

        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and (input.is_cuda or input.is_xpu):
                input_pad = torch.randn_like(input)
            fns.append(
                lambda: pad_addmm(
                    input_pad,
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )
        elif op is torch.ops.aten.mm:
            fns.append(
                lambda: pad_mm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )
        else:
            fns.append(
                lambda: pad_bmm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )

        def orig_bench_fn():
            if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
                op(mat1, mat2)
            else:
                op(input, mat1, mat2)

        def pad_bench_fn():
            for fn in fns:
                fn()

        if (
            torch._inductor.config.run_autoheuristic("pad_mm")
            and op is torch.ops.aten.mm
        ):
            ah_should_pad = run_autoheuristic(
                mat1,
                mat2,
                orig_bench_fn,
                pad_bench_fn,
                m_padded_length,
                k_padded_length,
                n_padded_length,
                do_bench,
                mat1_pre_padded,
                mat2_pre_padded,
                ori_time,
                ori_time_key,
                key,
            )
            if ah_should_pad is not None:
                return ah_should_pad

        if ori_time is None:
            ori_time = do_bench(orig_bench_fn)
            set_cached_base_mm_benchmark_time(ori_time_key, ori_time)

        pad_time = do_bench(pad_bench_fn)

        counters["inductor"]["pad_mm_bench"] += 1
        return is_padded_faster(key, ori_time, pad_time)


def get_context(
    mat1: Tensor,
    mat2: Tensor,
    mat1_pre_padded: bool,
    mat2_pre_padded: bool,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
) -> AHContext:
    context = AHContext()

    context.add_feature("m", mat1.shape[0])
    context.add_feature("k", mat1.shape[1])
    context.add_feature("n", mat2.shape[1])

    context_add_strides(context, "mat1", mat1.stride())
    context_add_strides(context, "mat2", mat2.stride())

    context.add_feature("m_padded_length", m_padded_length)
    context.add_feature("k_padded_length", k_padded_length)
    context.add_feature("n_padded_length", n_padded_length)

    context.add_feature("mat1_align_size", get_alignment_size(mat1))
    context.add_feature("mat2_align_size", get_alignment_size(mat2))

    context.add_feature("mat1_dtype", mat1.dtype, is_categorical=True)
    context.add_feature("mat2_dtype", mat2.dtype, is_categorical=True)

    context.add_feature("prepadded_mat1", mat1_pre_padded, is_categorical=True)
    context.add_feature("prepadded_mat2", mat2_pre_padded, is_categorical=True)

    context_add_using_tf32(context, mat1.dtype)
    return context


def run_autoheuristic(
    mat1: Tensor,
    mat2: Tensor,
    orig_bench_fn: Callable[[], None],
    pad_bench_fn: Callable[[], None],
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    do_bench: Callable[[Callable[[], Any]], float],
    mat1_pre_padded: bool,
    mat2_pre_padded: bool,
    ori_time: float,
    ori_time_key: str,
    key: str,
) -> bool | None:
    def feedback_fn(
        choice: str,
    ) -> float | None:
        if choice == orig_choice:
            return do_bench(orig_bench_fn)
        elif choice == pad_choice:
            return do_bench(pad_bench_fn)
        return None

    def fallback() -> str:
        return "autotune"

    orig_choice = "orig"
    pad_choice = "pad"
    choices = [orig_choice, pad_choice]
    feedback = LocalFeedback(feedback_fn)  # type: ignore[arg-type]
    context = get_context(
        mat1,
        mat2,
        mat1_pre_padded,
        mat2_pre_padded,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )
    name = "pad_mm"
    autoheuristic = AutoHeuristic(
        fallback=fallback,
        choices=choices,
        feedback=feedback,
        context=context,
        name=name,
        augment_context=pad_mm_operations(),
        precondition=pad_mm_precondition,
    )
    choice = autoheuristic.get_choice()
    choice2should_pad = {orig_choice: False, pad_choice: True, "autotune": None}
    ah_should_pad = choice2should_pad.get(choice)

    if torch._inductor.config.collect_autoheuristic(name):
        ah_ori_time = autoheuristic.get_collected_feedback(orig_choice)
        ah_pad_time = autoheuristic.get_collected_feedback(pad_choice)

        # if precondition is not satisfied, autoheuristic does not collect data
        if ah_ori_time is not None and ah_pad_time is not None:
            if ori_time is None:
                set_cached_base_mm_benchmark_time(ori_time_key, ah_ori_time)
            return is_padded_faster(key, ah_ori_time, ah_pad_time)
    if ah_should_pad is not None:
        set_cached_should_pad(key, ah_should_pad)
    return ah_should_pad


def mm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.mm(mat1, mat2)


def should_pad_mm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.mm)


def pad_mat1(
    mat1: Tensor, *, m_padded_length: int, k_padded_length: int, is_bmm: bool = False
) -> Tensor:
    if k_padded_length != 0 or m_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, k_padded_length, 0, m_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat1, pad_arg)
    else:
        return mat1


def pad_mat2(
    mat2: Tensor, *, k_padded_length: int, n_padded_length: int, is_bmm: bool = False
) -> Tensor:
    if k_padded_length != 0 or n_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, n_padded_length, 0, k_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat2, pad_arg)
    else:
        return mat2


def pad_mm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )
    res = aten.mm(mat1, mat2)
    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def mm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    return pad_mm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


def bmm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.bmm(mat1, mat2)


def should_pad_bmm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.bmm)


def pad_bmm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1,
            m_padded_length=m_padded_length,
            k_padded_length=k_padded_length,
            is_bmm=True,
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2,
            k_padded_length=k_padded_length,
            n_padded_length=n_padded_length,
            is_bmm=True,
        )
    res = aten.bmm(mat1, mat2)
    if m_padded_length != 0:
        res = res[:, :-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :, :-n_padded_length]
    return res


def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    return pad_bmm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


@functools.cache
def _pad_mm_init() -> None:
    from .joint_graph import patterns

    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    elif torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"

    # sizes/values dont actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds

    dim2a = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim2b = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)

    dim3a = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim3b = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)

    dim1a = functools.partial(torch.empty, (4), device=device, requires_grad=True)

    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    rep = {"beta": 0.213377, "alpha": 0.113377}

    for pattern, replacement, args, workaround, extra_check in [
        (
            typing.cast(SearchFn, mm_pattern),
            typing.cast(ReplaceFn, mm_replace),
            [dim2a(), dim2b()],
            {},
            should_pad_mm,
        ),
        (
            typing.cast(SearchFn, bmm_pattern),
            typing.cast(ReplaceFn, bmm_replace),
            [dim3a(), dim3b()],
            {},
            should_pad_bmm,
        ),
        (
            typing.cast(SearchFn, addmm_pattern),
            typing.cast(ReplaceFn, addmm_replace),
            [dim1a(), dim2a(), dim2b()],
            rep,
            should_pad_addmm,
        ),
    ]:
        assert isinstance(workaround, dict)  # mypy is unable to infer the type properly
        name = pattern.__name__

        gen_register_replacement(
            f"{name}_training",
            pattern,
            replacement,
            args,
            # pyrefly: ignore [bad-argument-type]
            joint_fwd_bwd,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )

        gen_register_replacement(
            f"{name}_inference",
            pattern,
            replacement,
            args,
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )
