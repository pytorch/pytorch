# Copyright (c) 2025, Tri Dao.
"""nvMatmulHeuristics-based config selection for GEMM.

Queries NVIDIA's analytic heuristic library to pick tile/cluster dims based on
problem shape, then selects swap_ab by comparing estimated runtimes for both
orientations.
"""

import logging
import torch

from .gemm_config import GemmConfig

logger = logging.getLogger(__name__)

_nvmmh_available = None
_iface = None
_hw_descriptors = {}  # gpu_enum -> hw descriptor


def _get_iface():
    """Lazily initialize the nvMatmulHeuristics interface."""
    global _nvmmh_available, _iface
    if _nvmmh_available is not None:
        return _iface
    try:
        from nvMatmulHeuristics import (
            NvMatmulHeuristicsInterface,
            NvMatmulHeuristicsTarget,
        )

        _iface = NvMatmulHeuristicsInterface(
            backend=NvMatmulHeuristicsTarget.CUTLASS3,
            precision="BSB",  # overridden per-call
        )
        _nvmmh_available = True
    except Exception as e:
        logger.debug(f"nvMatmulHeuristics not available: {e}")
        _nvmmh_available = False
        _iface = None
    return _iface


def _get_hw(device_capacity):
    """Get or create a hardware descriptor for the given SM version."""
    global _hw_descriptors
    if device_capacity in _hw_descriptors:
        return _hw_descriptors[device_capacity]
    try:
        from nvMatmulHeuristics import (
            NvMatmulHeuristicsNvidiaGpu,
            NvMatmulHeuristicsMatmulLayout,
        )

        iface = _get_iface()
        if iface is None:
            return None
        gpu_map = {
            9: NvMatmulHeuristicsNvidiaGpu.H100_SXM,
            10: NvMatmulHeuristicsNvidiaGpu.B200,
        }
        gpu = gpu_map.get(device_capacity)
        if gpu is None:
            return None
        hw = iface.createHardwareDescriptor()
        iface.setHardwarePredefinedGpu(hw, gpu)
        # Load discovery sets for TN_ROW_MAJOR and TN_COL_MAJOR
        for layout in [
            NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR,
            NvMatmulHeuristicsMatmulLayout.TN_COL_MAJOR,
        ]:
            iface.loadInternalDiscoverySet(layout, hw)
        _hw_descriptors[device_capacity] = hw
        return hw
    except Exception as e:
        logger.debug(f"Failed to create hardware descriptor: {e}")
        _hw_descriptors[device_capacity] = None
        return None


_TORCH_DTYPE_TO_NVMMH_PRECISION = {
    torch.bfloat16: "BSB",
    torch.float16: "HSH",
    torch.float32: "SSS",
}


def _query_top1(iface, hw, m, n, k, layout, precision):
    """Query nvMMH for top-1 config. Returns (tile_m, tile_n, cl_m, cl_n, est_runtime) or None."""
    try:
        original_precision = iface.precision
        iface.precision = precision
        results = iface.get_with_mnk(
            m=m,
            n=n,
            k=k,
            matmulLayout=layout,
            count=1,
            hardware_descriptor=hw,
        )
        iface.precision = original_precision
        if not results:
            return None
        cfg = results[0]["kernel"]
        return cfg.cta_tile_m, cfg.cta_tile_n, cfg.cluster_m, cfg.cluster_n, results[0]["runtime"]
    except Exception:
        return None


def nvmmh_default_config(A, B, device_capacity):
    """Use nvMatmulHeuristics to pick a GemmConfig based on problem shape.

    Queries both normal (M,N,K) with row-major output and swapped (N,M,K) with
    col-major output, picks the orientation with lower estimated runtime.

    Returns None if nvMatmulHeuristics is unavailable, letting the caller fall
    back to the hardcoded default.
    """
    from nvMatmulHeuristics import NvMatmulHeuristicsMatmulLayout

    iface = _get_iface()
    if iface is None:
        return None
    hw = _get_hw(device_capacity)
    if hw is None:
        return None

    precision = _TORCH_DTYPE_TO_NVMMH_PRECISION.get(A.dtype)
    if precision is None:
        return None

    # Extract M, N, K from tensor shapes
    # A: (M, K) or (L, M, K), B: (K, N) or (L, K, N)
    m = A.shape[-2] if A.ndim >= 2 else A.shape[0]
    k = A.shape[-1]
    n = B.shape[-1]

    # Query normal orientation: D(M,N) row-major
    normal = _query_top1(iface, hw, m, n, k, NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR, precision)
    # Query swapped orientation: D(N,M) col-major
    swapped = _query_top1(
        iface, hw, n, m, k, NvMatmulHeuristicsMatmulLayout.TN_COL_MAJOR, precision
    )

    if normal is None and swapped is None:
        return None

    # Pick orientation with lower estimated runtime
    normal_rt = normal[4] if normal else float("inf")
    swapped_rt = swapped[4] if swapped else float("inf")

    if swapped_rt < normal_rt and swapped is not None:
        tile_m, tile_n, cl_m, cl_n = swapped[:4]
        swap_ab = True
    else:
        tile_m, tile_n, cl_m, cl_n = normal[:4]
        swap_ab = False

    # SM90: pingpong only works with tile_m <= 128
    # SM100: no pingpong
    pingpong = (device_capacity == 9) and (tile_m <= 128)

    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        pingpong=pingpong,
        cluster_m=cl_m,
        cluster_n=cl_n,
        swap_ab=swap_ab,
        max_swizzle_size=8,
        device_capacity=device_capacity,
    )
