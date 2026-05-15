import functools
import logging
import math
import operator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import sympy

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.operator_schemas import normalize_function

from . import ir
from .utils import get_dtype_size, snode_args_kwargs, sympy_product
from .virtualized import V


log = logging.getLogger(__name__)


class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
    ALL_TO_ALL = 3
    UNSUPPORTED = 4
    P2P = 5


class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2
    BLACKWELL = 3


class InterconnectType(IntEnum):
    NVLINK_V100 = 0
    NVLINK_A100 = 1
    NVLINK_H100 = 2
    NVLINK_B200 = 3
    IB_HDR = 4  # 200 Gbps InfiniBand
    IB_NDR = 5  # 400 Gbps InfiniBand / RoCE
    PCIE = 6  # PCIe (no NVLink)


@dataclass(frozen=True)
class InterconnectProfile:
    """Empirically calibrated interconnect parameters for saturation estimation."""

    bus_bw_GBps: float  # effective bus bandwidth at large messages (GB/s)
    latency_per_hop_us: float  # per-ring-step latency (us)
    base_latency_us: float  # fixed startup latency (us)
    min_saturation_bytes: int = 0  # floor for small group sizes


_MB = 1024 * 1024

# Empirically calibrated per-interconnect profiles for saturation estimation.
# NVLink bus_bw from https://github.com/NVIDIA/nccl/blob/master/src/graph/topo.h
# IB hop_lat/base_lat from MAST sweep profiles on H100 GrandTeton RoCE
# (see agent_space/nccl_model_validation.md for calibration data).
# IB min_saturation_bytes floors the estimate for small group sizes where the
# ring-step formula underestimates.
INTERCONNECT_PROFILES: dict[InterconnectType, InterconnectProfile] = {
    InterconnectType.NVLINK_H100: InterconnectProfile(36.0, 40.0, 10.0),
    InterconnectType.NVLINK_A100: InterconnectProfile(22.0, 45.0, 10.0),
    InterconnectType.NVLINK_B200: InterconnectProfile(60.0, 30.0, 10.0),
    InterconnectType.NVLINK_V100: InterconnectProfile(14.0, 50.0, 10.0),
    InterconnectType.IB_NDR: InterconnectProfile(
        22.0, 12.0, 30.0, min_saturation_bytes=100 * _MB
    ),
    InterconnectType.IB_HDR: InterconnectProfile(
        11.0, 14.0, 30.0, min_saturation_bytes=75 * _MB
    ),
    # PCIe: ~10 GB/s effective NCCL bus BW (GPU->CPU->GPU ring steps)
    InterconnectType.PCIE: InterconnectProfile(10.0, 60.0, 15.0),
}

_GPU_TO_INTRA: dict[NVIDIA_GPU_TYPE, InterconnectType] = {
    NVIDIA_GPU_TYPE.VOLTA: InterconnectType.NVLINK_V100,
    NVIDIA_GPU_TYPE.AMPERE: InterconnectType.NVLINK_A100,
    NVIDIA_GPU_TYPE.HOPPER: InterconnectType.NVLINK_H100,
    NVIDIA_GPU_TYPE.BLACKWELL: InterconnectType.NVLINK_B200,
}
_GPU_TO_INTER: dict[NVIDIA_GPU_TYPE, InterconnectType] = {
    NVIDIA_GPU_TYPE.VOLTA: InterconnectType.IB_HDR,
    NVIDIA_GPU_TYPE.AMPERE: InterconnectType.IB_HDR,
    NVIDIA_GPU_TYPE.HOPPER: InterconnectType.IB_NDR,
    NVIDIA_GPU_TYPE.BLACKWELL: InterconnectType.IB_NDR,
}


@functools.lru_cache
def _has_nvlink() -> bool:
    """Detect NVLink via nvidia-smi topology, falling back to peer access check."""
    import subprocess

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return True  # Single GPU: interconnect irrelevant
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Check GPU-GPU section (before Legend) for "NV" links
            topo = result.stdout.split("Legend")[0]
            return "NV" in topo
    except (subprocess.TimeoutExpired, OSError):
        pass
    # Fallback: peer access generally implies NVLink on datacenter GPUs
    try:
        return torch.cuda.can_device_access_peer(0, 1)
    except (AssertionError, RuntimeError):
        return True


@functools.lru_cache
def get_gpu_type() -> NVIDIA_GPU_TYPE:
    # Prefer compute capability (works for all NVIDIA GPUs, including H200, L40, etc.)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        major = torch.cuda.get_device_properties(0).major
        if major >= 10:
            return NVIDIA_GPU_TYPE.BLACKWELL
        if major >= 9:
            return NVIDIA_GPU_TYPE.HOPPER
        if major >= 8:
            return NVIDIA_GPU_TYPE.AMPERE
        if major >= 7:
            return NVIDIA_GPU_TYPE.VOLTA
    # Fallback: nvidia-smi string matching
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run) or ""
    if "V100" in gpu_info:
        return NVIDIA_GPU_TYPE.VOLTA
    elif "A100" in gpu_info:
        return NVIDIA_GPU_TYPE.AMPERE
    elif "H100" in gpu_info:
        return NVIDIA_GPU_TYPE.HOPPER
    elif any(gpu in gpu_info for gpu in ("B100", "B200", "B300")):
        return NVIDIA_GPU_TYPE.BLACKWELL
    else:
        return NVIDIA_GPU_TYPE.AMPERE


def detect_interconnect(group_size: int) -> InterconnectType:
    """Auto-detect interconnect type from GPU generation and group topology."""
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 8
    gpu_gen = get_gpu_type()
    if math.ceil(group_size / gpus_per_node) == 1:
        if not _has_nvlink():
            return InterconnectType.PCIE
        return _GPU_TO_INTRA.get(gpu_gen, InterconnectType.NVLINK_A100)
    return _GPU_TO_INTER.get(gpu_gen, InterconnectType.IB_HDR)


def get_collective_type_from_kernel_name(kernel_name: str) -> NCCL_COLL:
    assert kernel_name is not None
    if "all_reduce" in kernel_name:
        return NCCL_COLL.ALL_REDUCE
    elif "all_gather" in kernel_name:
        return NCCL_COLL.ALL_GATHER
    elif "reduce_scatter" in kernel_name:
        return NCCL_COLL.REDUCE_SCATTER
    elif any(comm in kernel_name for comm in ("all_to_all", "alltoall")):
        return NCCL_COLL.ALL_TO_ALL
    elif any(comm in kernel_name for comm in ("isend", "irecv", "batch_p2p")):
        return NCCL_COLL.P2P
    else:
        return NCCL_COLL.UNSUPPORTED


def get_collective_type(node: ir.IRNode) -> NCCL_COLL:
    if not isinstance(node, ir._CollectiveKernel):
        raise ValueError(f"node is not a collective kernel: {node}")

    name = node.python_kernel_name
    assert name is not None
    return get_collective_type_from_kernel_name(name)


def get_ir_node_size_numel(size: torch.Size, fallback: int = 4096 * 4096) -> int:
    numel = sympy_product(size)
    if isinstance(numel, sympy.Integer):
        return int(numel)
    return V.graph.sizevars.optimization_hint(numel, fallback=fallback)


def get_fx_node_size_numel(size: torch.Size, fallback: int = 4096 * 4096) -> int:
    numel = functools.reduce(operator.mul, size, 1)
    result = optimization_hint(numel, fallback=fallback)
    return result


def get_collective_input_size_bytes(node: ir.IRNode) -> int:
    sz_bytes = 0
    for inp in node.inputs:  # type: ignore[attr-defined]
        numel = get_ir_node_size_numel(inp.layout.size)
        sz_bytes += numel * get_dtype_size(inp.layout.dtype)
    return sz_bytes


def get_collective_group_size(node: ir.IRNode) -> int:
    if isinstance(node, ir._CollectiveKernel) and not isinstance(node, ir._WaitKernel):
        from torch.distributed.distributed_c10d import _get_group_size_by_name

        return _get_group_size_by_name(node.constant_args[-1])
    else:
        raise TypeError(f"Unsupported collective type: {node}")


####################################################################################################################
# The following code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
####################################################################################################################


class NCCL_HW(IntEnum):
    NVLINK = 0
    PCI = 1
    NET = 2


class NCCL_ALGO(IntEnum):
    TREE = 0
    RING = 1


class NCCL_PROTO(IntEnum):
    # Matches NCCL devcomm.h protocol enum
    LL = 0  # Low-latency
    LL128 = 1  # Low-latency 128-byte
    SIMPLE = 2


_NUM_ALGOS = 2  # TREE, RING
_NUM_PROTOS = 3  # LL, LL128, SIMPLE

# Base latencies in us — [algo][proto]
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
baseLat = [
    # Tree
    [6.8, 14.0, 8.4],
    # Ring
    [6.6, 14.0, 8.4],
]

# Hardware latencies in us — [hw][algo][proto]
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
hwLat = [
    # NVLINK
    [
        [0.6, 1.25, 4.0],  # Tree
        [0.6, 1.9, 3.4],  # Ring
    ],
    # PCI
    [
        [1.0, 1.9, 4.0],  # Tree
        [1.0, 2.5, 5.7],  # Ring
    ],
    # NET
    [
        [5.0, 8.5, 14.0],  # Tree
        [2.7, 4.0, 14.0],  # Ring
    ],
]


# LL max bus BW — [compCapIndex][scaleIndex(N1/N2/N4+)]
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
llMaxBws = [
    [39.0, 39.0, 20.4],  # Volta
    [87.7, 22.5, 19.0],  # Ampere
    [141.0, 45.0, 35.0],  # Hopper
    [282.0, 90.0, 70.0],  # Blackwell
]

# Per-channel max BW caps — [compCapIndex][scaleIndex]
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
perChMaxRingLL128Bws = [
    [20.0, 20.0, 20.0],  # Volta
    [20.0, 20.0, 20.0],  # Ampere
    [36.7, 36.7, 36.7],  # Hopper
    [40.0, 40.0, 40.0],  # Blackwell
]

# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
perChMaxTreeLL128Bws = [
    [20.0, 20.0, 20.0],  # Volta
    [20.0, 20.0, 20.0],  # Ampere
    [36.7, 36.7, 29.0],  # Hopper
    [55.6, 31.67, 20.0],  # Blackwell
]

# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
perChMaxTreeBws = [
    [26.5, 18.5, 10.0],  # Volta
    [24.0, 23.6, 17.8],  # Ampere
    [38.7, 41.4, 36.0],  # Hopper
    [70.0, 42.8, 24.0],  # Blackwell
]

# Tree correction factor for medium message sizes — [proto][logSize]
# logSize = log2(nBytes >> 6), indices 0..23 map to 64B..256MB
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
treeCorrectionFactor = [
    # LL
    [
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.8,
        0.7,
        0.7,
        0.7,
        0.7,
        0.6,
        0.5,
        0.4,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    # LL128
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.8,
        0.8,
        0.8,
        0.7,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.8,
        0.9,
        0.9,
        0.9,
        0.9,
        1.0,
        1.0,
        1.0,
    ],
    # SIMPLE
    [
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.8,
        0.7,
        0.6,
        0.6,
        0.5,
        0.5,
        0.5,
        0.5,
        0.6,
        0.7,
        0.8,
        0.7,
        0.7,
        0.8,
        0.9,
        0.9,
        0.9,
    ],
]

# Heuristic nChannels per GPU generation.
# NCCL selects dynamically via topology search (up to MAXCHANNELS/2=32 for ring).
# These approximate standard DGX/HGX topologies; see nccl/src/graph/search.cc.
_GPU_NCHANNELS = {
    NVIDIA_GPU_TYPE.VOLTA: 2,
    NVIDIA_GPU_TYPE.AMPERE: 8,
    NVIDIA_GPU_TYPE.HOPPER: 16,
    NVIDIA_GPU_TYPE.BLACKWELL: 32,
}

# NVLink bus bandwidth per GPU (GB/s).
# https://github.com/NVIDIA/nccl/blob/master/src/graph/topo.h (link speed constants)
_GPU_NVLINK_BW: dict[NVIDIA_GPU_TYPE, float] = {
    NVIDIA_GPU_TYPE.VOLTA: 120.0,
    NVIDIA_GPU_TYPE.AMPERE: 240.0,
    NVIDIA_GPU_TYPE.HOPPER: 370.0,
    NVIDIA_GPU_TYPE.BLACKWELL: 720.0,
}

# PCIe effective intra-node bandwidth (GB/s) for NCCL ring.
# Conservative: GPU->CPU->GPU ring steps limit effective BW well below link rate.
_PCIE_INTRA_NODE_BW = 24.0

# Per-GPU inter-node bandwidth (GB/s), assuming 1:1 GPU:NIC ratio.
# HDR: 200Gbps=25GB/s, NDR: 400Gbps=50GB/s per NIC.
_GPU_INTER_NODE_BW: dict[NVIDIA_GPU_TYPE, float] = {
    NVIDIA_GPU_TYPE.VOLTA: 12.0,
    NVIDIA_GPU_TYPE.AMPERE: 25.0,
    NVIDIA_GPU_TYPE.HOPPER: 50.0,
    NVIDIA_GPU_TYPE.BLACKWELL: 50.0,
}


def get_intra_node_bw() -> float:
    """Return intra-node bandwidth in GB/s. Config overrides auto-detection."""
    override = torch._inductor.config.intra_node_bw
    if override is not None:
        return float(override)
    if not _has_nvlink():
        return _PCIE_INTRA_NODE_BW
    return _GPU_NVLINK_BW.get(get_gpu_type(), 240.0)


def get_inter_node_bw() -> float:
    """Return inter-node (IB/RoCE) bandwidth in GB/s. Config overrides auto-detection."""
    override = torch._inductor.config.inter_node_bw
    if override is not None:
        return float(override)
    return _GPU_INTER_NODE_BW.get(get_gpu_type(), 25.0)


def _log2i(n: int) -> int:
    """Integer log2, matching NCCL's log2i (floor of log2)."""
    if n <= 0:
        return 0
    r = 0
    while n > 1:
        n >>= 1
        r += 1
    return r


def estimate_nccl_collective_runtime_nccl_estimator(snode) -> float | None:  # type: ignore[no-untyped-def]
    kernel = snode.node
    assert kernel is not None
    py_kernel_name = getattr(kernel, "python_kernel_name", "")
    pg_name = kernel.constant_args[-1]  # type: ignore[attr-defined]
    from torch.distributed.distributed_c10d import _resolve_process_group

    pg = _resolve_process_group(pg_name)
    rank: int = torch.distributed.get_rank(pg)
    # TODO(ivankobzarev): Figure out how we can use time estimations,
    # without cuda allocations.
    device = torch.device(f"cuda:{rank}")

    fn = eval(py_kernel_name)
    args, kwargs = snode_args_kwargs(snode)

    # TODO(ivankobzarev): fix out variants snode_args_kwargs
    if "all_gather_into_tensor_out" in py_kernel_name:
        args = args[1:] + args[0]

    with torch.distributed._time_estimator(group=pg, device=device) as time_estimator:
        w = fn(*args, **kwargs)
        torch.ops._c10d_functional.wait_tensor.default(w)

    est_time_us = time_estimator.estimated_time
    # -1000 constant is NCCL return in case of error during estimations.
    # Observed it for all_to_all estimations.
    if est_time_us < 0:
        return None
    est_time_ms = est_time_us / 1e3
    return est_time_ms


def _nccl_algo_time(
    nBytes: int,
    group_size: int,
    coll: NCCL_COLL,
    algo: int,
    proto: int,
) -> float:
    """Compute NCCL estimated time in us for a given (algo, proto) pair.

    Mirrors ncclTopoTuneModel bandwidth/latency computation and
    ncclTopoGetAlgoTime from NCCL tuning.cc. Returns -1 if the
    (algo, proto) combination is disabled for this configuration.
    """
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 8
    nNodes = math.ceil(group_size / gpus_per_node)
    nRanks = group_size
    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2

    # Total bus bandwidth (not per-channel). Auto-detected from GPU
    # generation, or overridden by config. nChannels is only used for
    # per-channel BW cap table lookups.
    bwIntra = get_intra_node_bw()
    bwInter = get_inter_node_bw()
    nChannels = _GPU_NCHANNELS.get(compCapIndex, 8)

    # Detect interconnect for IB-specific corrections
    interconnect = detect_interconnect(group_size)
    is_ib = interconnect in (InterconnectType.IB_NDR, InterconnectType.IB_HDR)

    # LL128: disable for inter-node IB (BW tables calibrated for NVLink only)
    # and for large-scale multi-node (>2 nodes)
    if proto == NCCL_PROTO.LL128:
        if nNodes > 2:
            return -1.0
        if nNodes > 1 and is_ib:
            return -1.0

    # --- Bandwidth computation (mirrors ncclTopoTuneModel) ---
    # For IB interconnect, always use inter-node BW (bwIntra only for NVLink).
    # Original NCCL tuning.cc uses bwIntra for nNodes<=2, designed for NVSwitch;
    # doesn't apply to IB.
    if nNodes == 1:
        bw = bwIntra
    elif is_ib:
        bw = bwInter
    else:
        bw = bwIntra if nNodes <= 2 else bwInter

    # IB corrections calibrated against H100 GrandTeton RoCE profiles
    # (see agent_space/nccl_model_validation.md for calibration data):
    if is_ib and nNodes > 1:
        # NIC efficiency: fewer nodes → fewer NICs utilized → lower effective BW.
        # 2-node (single inter-hop): ~0.80×; 8+ nodes: ~1.0×
        nic_efficiency = 0.77 + 0.23 * min(1.0, (nNodes - 1) / 7)
        bw *= nic_efficiency
        # Multi-rail: larger messages utilize more NICs (8 NICs per H100 node).
        # Power-law fit (exponent 0.17) to non-coalesced profile BW vs size.
        if nNodes > 2:
            multi_rail = (max(nBytes, 50_000_000) / 50_000_000) ** 0.17
            bw *= multi_rail

    busBw = bw

    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    if algo == NCCL_ALGO.RING and proto == NCCL_PROTO.LL:
        busBw = min(llMaxBw, busBw * 0.5)
    elif algo == NCCL_ALGO.RING and proto == NCCL_PROTO.LL128:
        busBw = min(
            busBw * (120.0 / 128.0),
            nChannels * perChMaxRingLL128Bws[compCapIndex][index2],
        )
    elif algo == NCCL_ALGO.TREE and proto == NCCL_PROTO.LL:
        busBw = min(busBw * (1.0 / 3.8), llMaxBw)
    elif algo == NCCL_ALGO.TREE and proto == NCCL_PROTO.LL128:
        factor = 7.0 / 9.0 if nNodes == 1 else 120.0 / 128.0
        busBw = min(
            busBw * factor,
            nChannels * perChMaxTreeLL128Bws[compCapIndex][index2],
        )
    elif algo == NCCL_ALGO.TREE and proto == NCCL_PROTO.SIMPLE:
        if coll == NCCL_COLL.ALL_REDUCE:
            busBw = min(busBw * 0.92, nChannels * perChMaxTreeBws[compCapIndex][index2])
    # Ring+SIMPLE: no extra cap beyond busBw

    if busBw <= 0:
        return -1.0

    # nsteps for the collective
    if coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1
    elif coll == NCCL_COLL.ALL_TO_ALL:
        nsteps = 2 * (nRanks - 1)
    elif coll == NCCL_COLL.P2P:
        nsteps = 1
    else:
        return -1.0

    # Bus BW to algorithm BW conversion
    # Tree allreduce uses 0.5 ratio; Ring uses nRanks/nsteps
    if algo == NCCL_ALGO.TREE and coll == NCCL_COLL.ALL_REDUCE:
        bandwidth = busBw * 0.5
    elif algo == NCCL_ALGO.RING:
        bandwidth = busBw * (1.0 * nRanks / nsteps)
    else:
        bandwidth = busBw * 0.5

    if bandwidth <= 0:
        return -1.0

    # --- Latency computation (mirrors ncclTopoTuneModel) ---
    intraHw = NCCL_HW.PCI if interconnect == InterconnectType.PCIE else NCCL_HW.NVLINK
    lat = baseLat[algo][proto]
    intraLat = hwLat[intraHw][algo][proto]
    interLat = hwLat[NCCL_HW.NET][algo][proto]

    netOverhead = 1.0 if nNodes > 1 else 0.0
    if proto == NCCL_PROTO.SIMPLE and nNodes > 1:
        netOverhead *= 3.0
    intraLat = max(intraLat, netOverhead)

    if algo == NCCL_ALGO.RING:
        nInterSteps = (
            0
            if nNodes == 1
            else (2 * (nNodes - 1) if coll == NCCL_COLL.ALL_REDUCE else nNodes - 1)
        )
        lat += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat
    elif algo == NCCL_ALGO.TREE:
        if coll == NCCL_COLL.ALL_REDUCE:
            lat += 2 * (
                (nRanks // max(nNodes, 1) - 1) * intraLat + _log2i(nNodes) * interLat
            )

    # --- ncclTopoGetAlgoTime: time = lat + nBytes / (1000 * bandwidth) ---
    # Tree correction factor for medium messages
    if algo == NCCL_ALGO.TREE and coll == NCCL_COLL.ALL_REDUCE:
        logSize = _log2i(nBytes >> 6) if nBytes >= 64 else 0
        if 0 <= logSize < 24:
            bandwidth *= treeCorrectionFactor[proto][logSize]

    time_us = lat + nBytes / (1000.0 * bandwidth)
    return time_us


def _nccl_best_algo_time(
    nBytes: int,
    group_size: int,
    coll: NCCL_COLL,
) -> tuple[float, int, int]:
    """Find the best (algo, proto) for a given collective and return (time_us, algo, proto).

    Iterates all supported (algo, proto) combinations and picks the one with
    minimum estimated time, mirroring NCCL's algorithm selection.
    """
    best_time = float("inf")
    best_algo = NCCL_ALGO.RING
    best_proto = NCCL_PROTO.LL

    for algo in (NCCL_ALGO.TREE, NCCL_ALGO.RING):
        # Tree only supports allreduce in our model
        if algo == NCCL_ALGO.TREE and coll != NCCL_COLL.ALL_REDUCE:
            continue
        for proto in (NCCL_PROTO.LL, NCCL_PROTO.LL128, NCCL_PROTO.SIMPLE):
            t = _nccl_algo_time(nBytes, group_size, coll, algo, proto)
            if 0 <= t < best_time:
                best_time = t
                best_algo = algo
                best_proto = proto

    return best_time, best_algo, best_proto


def estimate_nccl_collective_runtime_impl(
    tensor_storage_size_bytes: int, group_size: int, coll: NCCL_COLL
) -> float:
    """Returns estimated NCCL collective runtime in milliseconds (ms).

    Uses a multi-algorithm, multi-protocol model aligned with NCCL's
    ncclTopoTuneModel and ncclTopoGetAlgoTime (tuning.cc). Evaluates
    Ring and Tree algorithms across LL, LL128, and SIMPLE protocols,
    selecting the (algo, proto) pair that minimizes estimated time.
    """
    if group_size <= 1:
        return 0

    if coll == NCCL_COLL.UNSUPPORTED:
        return 0

    time_us, _, _ = _nccl_best_algo_time(tensor_storage_size_bytes, group_size, coll)
    if time_us < 0 or time_us == float("inf"):
        return 0
    return time_us / 1000.0


def compute_min_saturation_bytes(
    group_size: int,
    coll: NCCL_COLL,
    target_efficiency: float = 0.9,
) -> int:
    """Compute min message size for target BW efficiency using empirical profiles.

    Uses per-interconnect profiles calibrated to real NCCL behavior (algorithm,
    protocol, channel selection) rather than the analytical model.
    """
    _SUPPORTED = (NCCL_COLL.ALL_GATHER, NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_REDUCE)
    assert coll in _SUPPORTED, (
        f"Unsupported collective {coll}, expected one of {_SUPPORTED}"
    )

    if group_size <= 1:
        return 0

    profile = INTERCONNECT_PROFILES[detect_interconnect(group_size)]

    if coll in (NCCL_COLL.ALL_GATHER, NCCL_COLL.REDUCE_SCATTER):
        nsteps = group_size - 1
    elif coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (group_size - 1)
    else:
        return 0

    total_latency_us = profile.base_latency_us + nsteps * profile.latency_per_hop_us
    latency_s = total_latency_us * 1e-6
    bw_bytes_per_s = profile.bus_bw_GBps * 1e9
    eff_ratio = target_efficiency / (1.0 - target_efficiency)
    min_bytes = int(eff_ratio * latency_s * bw_bytes_per_s)
    return max(min_bytes, profile.min_saturation_bytes)


################################################################################################################
# The above code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
################################################################################################################


def estimate_nccl_collective_runtime(node: ir.IRNode) -> float:
    """Returns estimated NCCL collective runtime in milliseconds (ms).

    Uses the multi-algorithm, multi-protocol analytical model aligned with
    NCCL's tuning.cc to select the best (algo, proto) pair for the given
    collective size and topology.
    """
    tensor_storage_size_bytes = get_collective_input_size_bytes(node)
    group_size = get_collective_group_size(node)
    coll = get_collective_type(node)
    return estimate_nccl_collective_runtime_impl(
        tensor_storage_size_bytes, group_size, coll
    )


def estimate_fx_collective_size(fx_node: torch.fx.Node) -> int:
    """Estimate the size of a collective operation in bytes, including inputs and outputs."""
    input_bytes = None

    args, kwargs = fx_node.args, fx_node.kwargs
    kwargs = dict(kwargs)

    # dont double count pre-allocated buffer passed in
    kwargs.pop("out", None)

    def tensor_bytes(t: torch.Tensor) -> int:
        return get_fx_node_size_numel(t.size()) * get_dtype_size(t.dtype)

    def add_inp_bytes(inp: torch.fx.Node):
        inp_val = inp.meta.get("val", None)
        if not isinstance(inp_val, torch.Tensor):
            return

        nonlocal input_bytes
        if input_bytes is None:
            input_bytes = 0
        input_bytes += tensor_bytes(inp_val)

    pytree.tree_map_only(
        torch.fx.Node,
        add_inp_bytes,
        (args, kwargs),
    )

    output_val = fx_node.meta.get("val", None)

    if input_bytes is None or output_val is None:
        return 0

    # Coalesced collectives return a list of tensors
    if isinstance(output_val, (list, tuple)):
        output_bytes = sum(
            tensor_bytes(t) for t in output_val if isinstance(t, torch.Tensor)
        )
    elif isinstance(output_val, torch.Tensor):
        output_bytes = tensor_bytes(output_val)
    else:
        return 0

    return input_bytes + output_bytes


def estimate_fx_collective_memory_footprint(fx_node: torch.fx.Node) -> int:
    """Estimate the memory footprint of a collective operation in bytes.

    This returns the total bytes that need to be live concurrently in memory.
    For all_reduce, we divide by 2 since it can be done in-place.
    """
    from torch._inductor.fx_passes.bucketing import (
        is_all_reduce_tensor as is_all_reduce,
    )

    size = estimate_fx_collective_size(fx_node)
    return size if not is_all_reduce(fx_node) else size // 2


def estimate_nccl_collective_runtime_from_fx_node(
    fx_node: torch.fx.Node,
    override_size: int | None = None,
    use_nccl_estimator: bool = True,
) -> float:
    """Returns estimated NCCL collective runtime in milliseconds (ms).

    Tries the NCCL simulator first (if available and enabled), falls back
    to the multi-algo/proto analytical model from tuning.cc.
    """
    from torch.distributed.distributed_c10d import _get_group_size_by_name

    if fx_node.target is torch.ops._c10d_functional.all_to_all_single.default:
        # TODO(ivankobzarev): Temporarily disabled - NCCL estimator returns internal error.
        # for all_to_all during inductor compilation. Falls back to heuristic estimation.
        use_nccl_estimator = False

    if override_size is None:
        tensor_storage_size_bytes = estimate_fx_collective_size(fx_node)
    else:
        tensor_storage_size_bytes = override_size

    assert not isinstance(fx_node.target, str)
    opt_args_kwargs = normalize_function(
        fx_node.target,
        args=fx_node.args,
        kwargs=fx_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt_args_kwargs is not None
    args, kwargs = opt_args_kwargs

    from torch._inductor.fx_passes.bucketing import _resolve_group_name

    group_name = _resolve_group_name(kwargs["group_name"])
    group_size = _get_group_size_by_name(group_name)
    assert isinstance(fx_node.target, torch._ops.OpOverload)
    coll = get_collective_type_from_kernel_name(fx_node.target.name())

    def _nccl_estimate() -> float | None:
        # TODO: Refactor with estimate_nccl_collective_runtime_nccl_estimator
        from torch.distributed.distributed_c10d import _resolve_process_group, Backend

        pg = _resolve_process_group(group_name)
        if torch.distributed.distributed_c10d.get_backend(pg) == Backend.FAKE:
            # nccl estimator requires real process group
            return None

        device = torch.device("cuda")
        try:
            backend = pg._get_backend(device)
        except RuntimeError:
            return None
        if not backend.supports_time_estimate:
            return None

        flat_args, flat_args_pytree_spec = pytree.tree_flatten((args, kwargs))

        def _tensor(size, dtype, device) -> torch.Tensor:  # type: ignore[no-untyped-def]
            return torch.empty(
                size if override_size is None else [override_size],
                dtype=dtype,
                device=device,
            )

        def to_real_tensor(e: Any) -> Any:
            if isinstance(e, torch.fx.Node):
                return to_real_tensor(e.meta["val"])
            if isinstance(e, torch.Tensor):
                return _tensor([get_fx_node_size_numel(e.size())], e.dtype, e.device)
            return e

        flat_args = [to_real_tensor(a) for a in flat_args]
        real_args, real_kwargs = pytree.tree_unflatten(flat_args, flat_args_pytree_spec)

        fn = fx_node.target
        assert isinstance(fn, torch._ops.OpOverload)
        with torch.distributed._time_estimator(
            group=pg, device=device
        ) as time_estimator:
            w = fn(*real_args, **real_kwargs)
            # Coalesced collectives return a list of tensors
            if isinstance(w, (list, tuple)):
                for t in w:
                    torch.ops._c10d_functional.wait_tensor.default(t)
            else:
                torch.ops._c10d_functional.wait_tensor.default(w)
        est_time_us = time_estimator.estimated_time
        # -1000 constant is NCCL return in case of error during estimations.
        # Observed it for all_to_all estimations.
        if est_time_us < 0:
            return None
        est_time_ms = est_time_us / 1e3
        return est_time_ms

    if use_nccl_estimator:
        est_time_ms = _nccl_estimate()
        if est_time_ms is not None:
            return est_time_ms

    return estimate_nccl_collective_runtime_impl(
        tensor_storage_size_bytes, group_size, coll
    )
