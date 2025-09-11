import functools
import logging
import math
from enum import IntEnum
from typing import Optional

import sympy

import torch

from . import ir
from .utils import get_dtype_size, snode_args_kwargs, sympy_product
from .virtualized import V


log = logging.getLogger(__name__)


class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
    ALL_TO_ALL = 3


class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2


@functools.lru_cache
def get_gpu_type() -> NVIDIA_GPU_TYPE:
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run) or ""
    if "V100" in gpu_info:
        return NVIDIA_GPU_TYPE.VOLTA
    elif "A100" in gpu_info:
        return NVIDIA_GPU_TYPE.AMPERE
    elif "H100" in gpu_info:
        return NVIDIA_GPU_TYPE.HOPPER
    else:
        # for other gpu types, assume Ampere
        return NVIDIA_GPU_TYPE.AMPERE


def get_collective_type(node: ir.IRNode) -> NCCL_COLL:
    if not isinstance(node, ir._CollectiveKernel):
        raise ValueError(f"node is not a collective kernel: {node}")

    kernel_name = node.python_kernel_name
    assert kernel_name is not None
    if "all_reduce" in kernel_name:
        return NCCL_COLL.ALL_REDUCE
    elif "all_gather" in kernel_name:
        return NCCL_COLL.ALL_GATHER
    elif "reduce_scatter" in kernel_name:
        return NCCL_COLL.REDUCE_SCATTER
    elif "torch.ops._dtensor.shard_dim_alltoall.default" in kernel_name:
        return NCCL_COLL.ALL_TO_ALL
    else:
        raise ValueError(f"Unsupported collective kernel: {kernel_name}")


def get_collective_input_size_bytes(node: ir.IRNode) -> int:
    sz_bytes = 0
    for inp in node.inputs:  # type: ignore[attr-defined]
        numel = sympy_product(inp.layout.size)
        if isinstance(numel, sympy.Integer):
            # For ease of testing
            numel = int(numel)
        else:
            numel = V.graph.sizevars.size_hint(numel, fallback=0)
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
    # The ordering and enum values here matches original in
    # https://github.com/NVIDIA/nccl/blob/0b083e52096c387bad7a5c5c65b26a9dca54de8c/src/include/devcomm.h#L28
    # For difference between these protocols, see https://github.com/NVIDIA/nccl/issues/281#issuecomment-571816990
    LL = 0  # Low-latency
    # LL128 = 1   # Low-latency 128-byte
    # SIMPLE = 2


# Latencies in us
# len(NCCL_ALGO) x len(NCCL_PROTO)
# NOTE: use array instead of tensor to prevent incompatibility with fake mode
baseLat = [
    # Tree
    [
        6.8,  # LL
    ],
    # Ring
    [
        6.6,  # LL
    ],
]

# Latencies in us
# len(NCCL_HW) x len(NCCL_ALGO) x len(NCCL_PROTO)
hwLat = [
    # NVLINK
    [
        [0.6],  # Tree (LL)
        [0.6],  # Ring (LL)
    ],
    # PCI
    [
        [1.0],  # Tree (LL)
        [1.0],  # Ring (LL)
    ],
    # NET
    [
        [5.0],  # Tree (LL)
        [2.7],  # Ring (LL)
    ],
]


# LL128 max BW per channel
llMaxBws = [
    # Volta-N1/Intel-N2/Intel-N4
    [
        39.0,
        39.0,
        20.4,
    ],
    # Ampere-N1/AMD-N2/AMD-N4
    [
        87.7,
        22.5,  # avg of ring & tree
        19.0,
    ],
    # Hopper-N1/AMD-N2/AMD-N4
    [
        87.7,
        22.5,  # avg of ring & tree
        19.0,
    ],
]


def estimate_nccl_collective_runtime_nccl_estimator(snode) -> Optional[float]:  # type: ignore[no-untyped-def]
    kernel = snode.node
    assert kernel is not None
    py_kernel_name = getattr(kernel, "python_kernel_name", "")
    if not ("all_gather" in py_kernel_name or "reduce_scatter" in py_kernel_name):
        # NCCL of version 2.27 sometimes unrecoverably fail for all_to_all, all_reduce
        return None

    from torch.distributed.distributed_c10d import _resolve_process_group

    pg_name = kernel.constant_args[-1]  # type: ignore[attr-defined]
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

    try:
        with torch.distributed._time_estimator(
            group=pg, device=device
        ) as time_estimator:
            w = fn(*args, **kwargs)
            torch.ops._c10d_functional.wait_tensor.default(w)
    except Exception as e:
        # NCCL estimator can fail
        log.info(e)
        return None

    est_time_us = time_estimator.estimated_time
    # -1000 constant is NCCL return in case of error during estimations.
    # Observed it for all_to_all estimations.
    if est_time_us < 0:
        return None
    est_time_ms = est_time_us / 1e3
    return est_time_ms


def estimate_nccl_collective_runtime(node: ir.IRNode) -> float:
    """
    Returns estimated NCCL collective runtime in milliseconds (ms).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    - collective is one of: allreduce, reducescatter, allgather
    """
    tensor_storage_size_bytes = get_collective_input_size_bytes(node)
    # Convert bytes to GB
    tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024

    # Currently assumes each node has 8 gpus. And when >1 node is used, assumes each node uses all 8 gpus.
    # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    num_gpus_per_node = 8
    group_size = get_collective_group_size(node)
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    if nRanks <= 1:
        return 0

    # Assumes ring algorithm
    nccl_algo = NCCL_ALGO.RING
    nccl_proto = NCCL_PROTO.LL
    coll = get_collective_type(node)

    # =============== bandwidth computation ===============
    # First compute bandwidth in GB/s; then at the end, convert it to GB/ns

    bwIntra = torch._inductor.config.intra_node_bw
    bwInter = torch._inductor.config.inter_node_bw

    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2
    # LL: for single node, we look at GPU type; for multi-node, we look at CPU type
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    # NOTE: each step of ring algorithm is synchronized,
    # and is bottlenecked by the slowest link which is the inter-node interconnect.
    # hence when nNodes >= 2, bw is inter-node bandwidth.
    # NOTE: the original code in https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc
    # have this as `if nNodes <= 2` which seems wrong. Corrected it here.
    bw = bwIntra if nNodes == 1 else bwInter
    nChannels = 2  # Assume # channels is 2
    busBw = nChannels * bw

    # Various model refinements
    busBw = min(
        llMaxBw,
        busBw
        * (1.0 / 4.0 if (nNodes > 1 or coll == NCCL_COLL.ALL_REDUCE) else 1.0 / 3.0),
    )

    if coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (nRanks - 1)
    elif coll == NCCL_COLL.ALL_TO_ALL:
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1

    # Convert bus BW to algorithm BW (tensor bytes / algoBW = actual execution time)
    ratio = (1.0 * nRanks) / nsteps  # type: ignore[possibly-undefined]
    bandwidth = busBw * ratio
    # Convert GB/s to GB/ns
    bandwidth_GB_per_ns = bandwidth / 1e9

    # =============== latency computation ===============
    intraHw = NCCL_HW.NVLINK

    if coll == NCCL_COLL.ALL_REDUCE:
        if nNodes > 1:
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER, NCCL_COLL.ALL_TO_ALL):
        nInterSteps = nNodes - 1

    # First compute latency in us; then at the end, convert it to ns
    latency = baseLat[nccl_algo][nccl_proto]
    intraLat = hwLat[intraHw][nccl_algo][nccl_proto]
    interLat = hwLat[NCCL_HW.NET][nccl_algo][nccl_proto]

    # Inter-node rings still have to launch nsteps * net overhead.
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0  # getNetOverhead(comm);
    intraLat = max(intraLat, netOverhead)
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat  # type: ignore[possibly-undefined]
    # Convert us to ns
    latency_ns = latency * 1e3

    # =============== final result ===============
    transport_ns = tensor_storage_size_GB / bandwidth_GB_per_ns
    ns = transport_ns + latency_ns
    ms = ns / 1e6
    return ms


################################################################################################################
# The above code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
################################################################################################################
