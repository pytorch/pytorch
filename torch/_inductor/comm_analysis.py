import functools
import logging
import math
import operator
from enum import IntEnum
from typing import Any, Optional

import sympy

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.symbolic_shapes import size_hint
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


class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2
    BLACKWELL = 3


@functools.lru_cache
def get_gpu_type() -> NVIDIA_GPU_TYPE:
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
        # for other gpu types, assume Ampere
        return NVIDIA_GPU_TYPE.AMPERE


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
    return V.graph.sizevars.size_hint(numel, fallback=fallback)


def get_fx_node_size_numel(size: torch.Size, fallback: int = 4096 * 4096) -> int:
    numel = functools.reduce(operator.mul, size, 1)
    result = size_hint(numel, fallback=fallback)
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
        141.0,
        45.0,  # avg of ring & tree
        35.0,
    ],
    # Blackwell-N1/AMD-N2/AMD-N4
    [
        282.0,
        90.0,  # avg of ring & tree
        70.0,
    ],
]


def estimate_nccl_collective_runtime_nccl_estimator(snode) -> Optional[float]:  # type: ignore[no-untyped-def]
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


def estimate_nccl_collective_runtime_impl(
    tensor_storage_size_bytes: int, group_size: int, coll: NCCL_COLL
) -> float:
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
    # Convert bytes to GB
    tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024

    # Currently assumes each node has 8 gpus. And when >1 node is used, assumes each node uses all 8 gpus.
    # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    num_gpus_per_node = 8
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    if nRanks <= 1:
        return 0

    # Assumes ring algorithm
    nccl_algo = NCCL_ALGO.RING
    nccl_proto = NCCL_PROTO.LL

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


def estimate_nccl_collective_runtime(node: ir.IRNode) -> float:
    """
    Returns estimated NCCL collective runtime in nanoseconds (ms).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    - collective is one of: allreduce, reducescatter, allgather
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

    if input_bytes is None or not isinstance(output_val, torch.Tensor):
        return 0

    output_bytes = tensor_bytes(output_val)

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
    override_size: Optional[int] = None,
    use_nccl_estimator: bool = True,
) -> float:
    """
    Returns estimated NCCL collective runtime in nanoseconds (ms).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    - collective is one of: allreduce, reducescatter, allgather
    """
    from torch.distributed.distributed_c10d import _get_group_size_by_name

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

    group_name = kwargs["group_name"]
    group_size = _get_group_size_by_name(group_name)
    assert isinstance(fx_node.target, torch._ops.OpOverload)
    coll = get_collective_type_from_kernel_name(fx_node.target.name())

    def _nccl_estimate() -> Optional[float]:
        # TODO: Refactor with estimate_nccl_collective_runtime_nccl_estimator
        from torch.distributed.distributed_c10d import (
            _get_pg_default_device,
            _resolve_process_group,
        )

        pg = _resolve_process_group(group_name)
        if torch.distributed.distributed_c10d.get_backend(pg) == "fake":
            # nccl estimator requires real process group
            return None

        device = _get_pg_default_device(pg)
        backend = pg._get_backend(device)
        if not backend.supports_time_estimate:
            return None

        flat_args, flat_args_pytree_spec = pytree.tree_flatten((args, kwargs))

        def _tensor(size, dtype, device) -> torch.Tensor:  # type: ignore[no-untyped-def]
            return torch.empty(
                size if override_size is None else [override_size],
                dtype=dtype,
                device=device,
            )

        def try_size_hint(s: sympy.Expr) -> int:
            return V.graph.sizevars.size_hint(s, fallback=0)

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
        with torch.distributed._time_estimator(group=pg) as time_estimator:
            w = fn(*real_args, **real_kwargs)
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
