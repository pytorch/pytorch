import math
from enum import IntEnum

from typing import TYPE_CHECKING

import torch
from . import ir

from .utils import get_dtype_size, sympy_product
from .virtualized import V

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode


class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2


class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2


def get_gpu_type() -> NVIDIA_GPU_TYPE:
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run)
    if "V100" in gpu_info:
        return NVIDIA_GPU_TYPE.VOLTA
    elif "A100" in gpu_info:
        return NVIDIA_GPU_TYPE.AMPERE
    elif "H100" in gpu_info:
        return NVIDIA_GPU_TYPE.HOPPER
    else:
        # for other gpu types, assume Ampere
        return NVIDIA_GPU_TYPE.AMPERE


def get_collective_type(snode: "BaseSchedulerNode") -> NCCL_COLL:
    if isinstance(snode.node, (ir.AllReduce, ir.AllReduceCoalesced)):
        return NCCL_COLL.ALL_REDUCE
    elif isinstance(
        snode.node, (ir.AllGatherIntoTensor, ir.AllGatherIntoTensorCoalesced)
    ):
        return NCCL_COLL.ALL_GATHER
    elif isinstance(
        snode.node, (ir.ReduceScatterTensor, ir.ReduceScatterTensorCoalesced)
    ):
        return NCCL_COLL.REDUCE_SCATTER
    else:
        raise Exception(f"Unsupported collective type: {snode.node}")


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
baseLat = torch.tensor(
    [
        # Tree
        [
            6.8,  # LL
        ],
        # Ring
        [
            6.6,  # LL
        ],
    ]
)

# Latencies in us
# len(NCCL_HW) x len(NCCL_ALGO) x len(NCCL_PROTO)
hwLat = torch.tensor(
    [
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
)


# LL128 max BW per channel
llMaxBws = torch.tensor(
    [
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
)


def estimate_nccl_collective_runtime(snode: "BaseSchedulerNode") -> float:
    """
    Returns estimated NCCL collective runtime in nanoseconds (ns).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    - collective is one of: allreduce, reducescatter, allgather
    """
    tensor_numel = V.graph.sizevars.size_hint(sympy_product(snode.node.layout.size))
    tensor_dtype = snode.node.layout.dtype
    tensor_storage_size_bytes = tensor_numel * get_dtype_size(tensor_dtype)
    # Convert bytes to GB
    tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024

    # Currently assumes each node has 8 gpus. And when >1 node is used, assumes each node uses all 8 gpus.
    # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    num_gpus_per_node = 8
    _, _, group_size = snode.node.constant_args  # type: ignore[attr-defined]
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    if nRanks <= 1:
        return 0

    # Assumes ring algorithm
    nccl_algo = NCCL_ALGO.RING
    nccl_proto = NCCL_PROTO.LL
    coll = get_collective_type(snode)

    # =============== bandwidth computation ===============
    # First compute bandwidth in GB/s; then at the end, convert it to GB/ns

    bwIntra = torch._inductor.config.intra_node_bw
    bwInter = torch._inductor.config.inter_node_bw

    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2
    # LL: for single node, we look at GPU type; for multi-node, we look at CPU type
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2].item()

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
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1

    # Convert bus BW to algorithm BW (tensor bytes / algoBW = actual execution time)
    ratio = (1.0 * nRanks) / nsteps
    bandwidth = busBw * ratio
    # Convert GB/s to GB/ns
    bandwidth_GB_per_ns = bandwidth / 1e9

    # =============== latency computation ===============
    intraHw = NCCL_HW.NVLINK
    hw = intraHw if nNodes == 1 else NCCL_HW.NET

    if coll == NCCL_COLL.ALL_REDUCE:
        if nNodes > 1:
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nInterSteps = nNodes - 1

    # First compute latency in us; then at the end, convert it to ns
    latency = baseLat[nccl_algo][nccl_proto].item()
    intraLat = hwLat[intraHw][nccl_algo][nccl_proto].item()
    interLat = hwLat[NCCL_HW.NET][nccl_algo][nccl_proto].item()

    # Inter-node rings still have to launch nsteps * net overhead.
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0  # getNetOverhead(comm);
    intraLat = max(intraLat, netOverhead)
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat
    # Convert us to ns
    latency_ns = latency * 1e3

    # =============== final result ===============
    transport_ns = tensor_storage_size_GB / bandwidth_GB_per_ns
    return transport_ns + latency_ns


################################################################################################################
# The above code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
################################################################################################################
