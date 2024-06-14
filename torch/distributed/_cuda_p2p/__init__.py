# mypy: allow-untyped-defs
from collections import defaultdict
from contextlib import contextmanager

from functools import partial
from typing import Callable, cast, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.distributed._functional_collectives as funcol

import torch.distributed.distributed_c10d as c10d

if TYPE_CHECKING:
    from torch._C._distributed_c10d import _DistributedBackendOptions, Backend


"""
This file contains the registration logic and Python APIs for
``ProcessGroupCudaP2P`` (experimental).

``ProcessGroupCudaP2P`` is a thin wrapper around ``ProcessGroupNCCL``. By
default, it routes all collectives to the underlying ``ProcessGroupNCCL``. In
addition, ``ProcessGroupCudaP2P`` initializes a P2P workspace that allows
direct GPU memory access among the members. The workspace can be used in Python
to optimize intra-node communication patterns or to create custom intra-node
collectives in CUDA.

``ProcessGroupCudaP2P`` aims to bridge the gap where certain important patterns
can be better optimized via fine-grained P2P memory access than with
collectives in the latest version of NCCL. It is meant to complement NCCL
rather than replacing it.

Usage:

    # Using ProcessGroupCudaP2P
    dist.init_process_group(backend="cuda_p2p", ...)

    # Using ProcessGroupCudaP2P while specifying ProcessGroupCudaP2P.Options
    pg_options = ProcessGroupCudaP2P.Options()
    dist.init_process_group(backend="cuda_p2p", pg_options=pg_options, ...)

    # Using ProcessGroupCudaP2P while specifying ProcessGroupNCCL.Options
    pg_options = ProcessGroupNCCL.Options()
    dist.init_process_group(backend="cuda_p2p", pg_options=pg_options, ...)

    # Using ProcessGroupCudaP2P while specifying both
    # ProcessGroupCudaP2P.Options and ProcessGroupNCCL.Options
    pg_options = ProcessGroupCudaP2P.Options()
    pg_options.nccl_options = ProcessGroupNCCL.Options()
    dist.init_process_group(backend="cuda_p2p", pg_options=pg_options, ...)

    # Down-casting the backend to access p2p buffers for cuda_p2p specific
    # optimizations
    if is_cuda_p2p_group(group):
        backend = get_cuda_p2p_backend(group)
        if required_p2p_buffer_size > backend.get_buffer_size():
            # fallback
        p2p_buffer = backend.get_p2p_buffer(...)
    else:
        # fallback
"""


def _create_cuda_p2p_group(
    dist_backend_opts: "_DistributedBackendOptions",
    options: Union[
        "c10d.ProcessGroupCudaP2P.Options", "c10d.ProcessGroupNCCL.Options", None
    ],
) -> "Backend":
    if not c10d.is_nccl_available():
        raise RuntimeError("The cuda_p2p backend is not available")
    if options is None:
        options = c10d.ProcessGroupCudaP2P.Options()
        options.nccl_options = c10d.ProcessGroupNCCL.Options()
    elif isinstance(options, c10d.ProcessGroupNCCL.Options):
        nccl_options = options
        options = c10d.ProcessGroupCudaP2P.Options()
        options.nccl_options = nccl_options
    elif isinstance(options, c10d.ProcessGroupCudaP2P.Options):
        if options.nccl_options is None:
            options.nccl_options = c10d.ProcessGroupNCCL.Options()
    else:
        raise TypeError(
            "options for cuda_p2p must be ProcessGroupCudaP2P.Options "
            f"or ProcessGroupNCCL.Options (got: {type(options)})"
        )

    return c10d.ProcessGroupCudaP2P(
        dist_backend_opts.store,
        dist_backend_opts.group_rank,
        dist_backend_opts.group_size,
        options,
    )


def is_cuda_p2p_group(group: c10d.ProcessGroup) -> bool:
    if _test_with_non_cuda_p2p_group:
        return True
    if not c10d.is_nccl_available():
        return False
    try:
        backend = group._get_backend(torch.device("cuda"))
    except Exception:
        return False
    return isinstance(backend, c10d.ProcessGroupCudaP2P) and backend.is_p2p_available()


def get_cuda_p2p_backend(group: c10d.ProcessGroup) -> "c10d.ProcessGroupCudaP2P":
    if not is_cuda_p2p_group(group):
        raise TypeError("group is not a cuda_p2p process group.")
    return cast(
        c10d.ProcessGroupCudaP2P,
        group._get_backend(torch.device("cuda")),
    )


def get_p2p_buffer_size(group: c10d.ProcessGroup) -> int:
    if not is_cuda_p2p_group(group):
        return 0
    backend = get_cuda_p2p_backend(group)
    return backend.get_buffer_size()


c10d.Backend.register_backend(
    "cuda_p2p",
    _create_cuda_p2p_group,
    extended_api=True,
    devices=["cuda"],
)


_test_with_non_cuda_p2p_group: bool = False


@contextmanager
def test_with_non_cuda_p2p_group():
    """
    Force ops in this file to work with non-cuda_p2p groups for testing
    purposes. Not thread safe.
    """
    global _test_with_non_cuda_p2p_group
    prev = _test_with_non_cuda_p2p_group
    try:
        _test_with_non_cuda_p2p_group = True
        yield
    finally:
        _test_with_non_cuda_p2p_group = prev


_current_p2p_usage_counter: Optional[Dict[str, int]] = None


@contextmanager
def p2p_usage_counter():
    """
    Record the number of ops that utilized p2p capability for testing purposes.
    Fallbacks are excluded.
    """
    global _current_p2p_usage_counter
    prev = _current_p2p_usage_counter
    try:
        _current_p2p_usage_counter = defaultdict(int)
        yield _current_p2p_usage_counter
    finally:
        _current_p2p_usage_counter = prev


def _pipelined_all_gather_and_consume(
    shard: torch.Tensor,
    shard_consumer: Callable[[torch.Tensor, int], None],
    ag_out: torch.Tensor,
    group: c10d.ProcessGroup,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        tensor = all_gather_tensor(shard, gather_dim=1, group=group)
        chunks = tensor.chunk(group.size())
        for src_rank, chunk in enumerate(chunks):
            shard_consumer(chunk, src_rank)

    NOTE:
    - The shard passed to shard consumer will always be contiguous.
    """
    p2p_buf_sz_req = shard.numel() * shard.element_size()
    if get_p2p_buffer_size(group) < p2p_buf_sz_req:
        # We preferred the caller to handle fallback so that the computation
        # doesn't need to be decomposed.
        raise RuntimeError(
            f"_pipelined_all_gather_and_consume on input with shape={shard.shape} "
            f"and dtype={shard.dtype} requires {p2p_buf_sz_req} bytes of p2p buffers "
            f"(got {get_p2p_buffer_size(group)} bytes)."
        )

    backend = get_cuda_p2p_backend(group)
    group_size = group.size()
    rank = group.rank()

    backend.stream().wait_stream(torch.cuda.current_stream())
    local_p2p_buf = backend.get_p2p_buffer(rank, shard.shape, shard.dtype)

    chunks = ag_out.chunk(group.size())

    # While consuming local shard, copy it to the local p2p buffer
    # in another stream.
    shard_consumer(shard, rank)
    chunks[rank].copy_(shard)

    with torch.cuda.stream(backend.stream()):
        local_p2p_buf.copy_(shard)
        work = backend.intra_node_barrier()
    work.wait()

    # At this point, all ranks have copied their local shard to
    # their local p2p buffer. Each rank can now copy and consume
    # remote shards.
    for i in range(1, group_size):
        if i % 2 == 0:
            stream = torch.cuda.current_stream()
        else:
            stream = backend.stream()
        remote_rank = (i + rank) % group_size
        remote_p2p_buf = backend.get_p2p_buffer(remote_rank, shard.shape, shard.dtype)
        with torch.cuda.stream(stream):
            chunks[remote_rank].copy_(remote_p2p_buf)
            shard_consumer(chunks[remote_rank], remote_rank)

    torch.cuda.current_stream().wait_stream(backend.stream())

    with torch.cuda.stream(backend.stream()):
        work = backend.intra_node_barrier()
    work.wait()


def _pipelined_produce_and_all2all(
    chunk_producer: Callable[[int, torch.Tensor], None],
    output: torch.Tensor,
    group: c10d.ProcessGroup,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        chunks = [
            chunk_producer(dst_rank, chunks[dst_rank])
            for dst_rank in range(group.size()):
        ]
        dist.all_to_all_single(output=output, input=torch.cat(chunks))
    """
    group_size = group.size()
    rank = group.rank()

    out_chunks = output.chunk(group_size)
    p2p_buf_sz_req = out_chunks[0].numel() * out_chunks[0].element_size() * 2
    if get_p2p_buffer_size(group) < p2p_buf_sz_req:
        # We preferred the caller to handle fallback so that the computation
        # doesn't need to be decomposed.
        raise RuntimeError(
            f"_pipelined_produce_and_all2all on output with shape={output.shape} "
            f"and dtype={output.dtype} requires {p2p_buf_sz_req} bytes of p2p buffers "
            f"(got {get_p2p_buffer_size(group)} bytes)."
        )

    backend = get_cuda_p2p_backend(group)
    backend.stream().wait_stream(torch.cuda.current_stream())

    def get_p2p_buf(rank: int, idx: int) -> torch.Tensor:
        assert idx in (0, 1)
        offset = 0 if idx == 0 else out_chunks[0].numel()
        return backend.get_p2p_buffer(
            rank, out_chunks[0].shape, out_chunks[0].dtype, offset
        )

    # Prepare two local p2p buffers, so that a remote rank can pull the result
    # of step [i] in one p2p buffer while the local rank can compute the
    # result of step [i+1] and write it directly the other p2p buffer.
    local_p2p_buf_0 = get_p2p_buf(rank, 0)
    local_p2p_buf_1 = get_p2p_buf(rank, 1)

    # Directly write the local result to the destination.
    # No need to go through the p2p buffers.
    chunk_producer(rank, out_chunks[rank])

    with torch.cuda.stream(backend.stream()):
        chunk_producer((rank + 1) % group_size, local_p2p_buf_0)
        backend.intra_node_barrier()
        remote_p2p_buf = get_p2p_buf((rank - 1) % group_size, 0)
        out_chunks[(rank - 1) % group_size].copy_(remote_p2p_buf)

    for step in range(2, group_size):
        remote_rank = (rank - step) % group_size
        if step % 2 == 0:
            stream = torch.cuda.current_stream()
            p2p_buf = local_p2p_buf_1
            remote_p2p_buf = get_p2p_buf(remote_rank, 1)
        else:
            stream = backend.stream()
            p2p_buf = local_p2p_buf_0
            remote_p2p_buf = get_p2p_buf(remote_rank, 0)
        with torch.cuda.stream(stream):
            chunk_producer((rank + step) % group_size, p2p_buf)
            backend.intra_node_barrier()
            out_chunks[remote_rank].copy_(remote_p2p_buf)

    torch.cuda.current_stream().wait_stream(backend.stream())
    backend.intra_node_barrier()


lib = torch.library.Library("cuda_p2p", "DEF")  # noqa: TOR901
lib.define(
    "fused_all_gather_matmul(Tensor A, Tensor[] Bs, int gather_dim, str group_name) -> (Tensor, Tensor[])"
)
lib.define(
    "fused_matmul_reduce_scatter(Tensor A, Tensor B, str reduce_op, int scatter_dim, str group_name) -> Tensor"
)


@torch.library.impl(lib, "fused_all_gather_matmul", "Meta")
def _fused_all_gather_matmul_fallback(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    group_size = c10d._get_group_size_by_name(group_name)
    A = torch.ops._c10d_functional.all_gather_into_tensor(
        A_shard.contiguous(), group_size, group_name
    )
    A = torch.ops._c10d_functional.wait_tensor(A)
    A = A.view(group_size, *A_shard.shape).movedim(gather_dim + 1, 1).flatten(0, 1)
    return A.movedim(0, gather_dim), [
        torch.matmul(A, B).movedim(0, gather_dim) for B in Bs
    ]


@torch.library.impl(lib, "fused_all_gather_matmul", "CUDA")
def _fused_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        all_gather_tensor(A_shard, gather_dim, group_name) @ B
    """
    if A_shard.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    for B in Bs:
        if B.dim() != 2:
            raise ValueError("B must be a matrix")
    if gather_dim < 0 or gather_dim >= A_shard.dim():
        raise ValueError("Invalid gather_dim")

    group = c10d._resolve_process_group(group_name)
    p2p_buf_sz_req = A_shard.numel() * A_shard.element_size()
    if (
        _test_with_non_cuda_p2p_group
        or get_p2p_buffer_size(group) < p2p_buf_sz_req
        # Pipelining a mamtul with split-k is not supported
        or gather_dim == len(A_shard.shape) - 1
    ):
        return _fused_all_gather_matmul_fallback(A_shard, Bs, gather_dim, group_name)

    if _current_p2p_usage_counter is not None:
        _current_p2p_usage_counter["fused_all_gather_matmul"] += 1

    # Move the gather_dim to the front and flatten the tensor into a 2D matrix.
    # The flattened tensor doesn't need to be contiguous (for computation
    # efficiency), as _pipelined_all_gather_and_consume guarantees that shards
    # passed to shard_consumer are contiguous.
    x = A_shard.movedim(gather_dim, 0)
    leading_dims = [group.size()] + list(x.shape[:-1])
    x = x.flatten(0, -2)

    # Helper function for reverting the above transformation
    def unflatten(t):
        return t.view(*leading_dims, -1).flatten(0, 1).movedim(0, gather_dim)

    ag_out = x.new_empty(
        x.shape[0] * group.size(),
        x.shape[1],
    )
    outputs = [
        x.new_empty(
            x.shape[0] * group.size(),
            B.shape[1],
        )
        for B in Bs
    ]
    output_shards = [output.chunk(group.size()) for output in outputs]

    # Computing block-wise matmul along the first dim of A
    def shard_consumer(shard: torch.Tensor, rank: int) -> None:
        for idx, B in enumerate(Bs):
            torch.mm(shard, B, out=output_shards[idx][rank])

    _pipelined_all_gather_and_consume(
        x,
        shard_consumer,
        ag_out,
        group,
    )
    return unflatten(ag_out), [unflatten(output) for output in outputs]


@torch.library.impl(lib, "fused_matmul_reduce_scatter", "Meta")
def _fused_matmul_reduce_scatter_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    res = funcol.reduce_scatter_tensor(A @ B, reduce_op, scatter_dim, group_name)
    res = funcol.wait_tensor(res)
    return res


@torch.library.impl(lib, "fused_matmul_reduce_scatter", "CUDA")
def _fused_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        reduce_scatter_tensor(A @ B, reduce_op, scatter_dim, group_name)

    NOTE:
    - The K dim across ranks are currently accumulated with bf16 with results
      in accuracy loss.
    """
    if A.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    if scatter_dim < 0 or scatter_dim >= A.dim():
        raise ValueError("Invalid gather_dim")
    if B.dim() != 2:
        raise ValueError("B must be a matrix")
    if reduce_op == "sum":
        reduce_fn = partial(torch.sum, dim=0)
    elif reduce_op == "avg":
        reduce_fn = partial(torch.mean, dim=0)
    else:
        raise ValueError("reduce_op must be sum or avg")

    group = c10d._resolve_process_group(group_name)
    out_shape = [*A.shape[:-1], B.shape[1]]
    out_shape[scatter_dim] //= group.size()
    p2p_buf_sz_req = torch.Size(out_shape).numel() * A.element_size() * 2
    if _test_with_non_cuda_p2p_group or get_p2p_buffer_size(group) < p2p_buf_sz_req:
        return _fused_matmul_reduce_scatter_fallback(
            A, B, reduce_op, scatter_dim, group_name
        )

    if _current_p2p_usage_counter is not None:
        _current_p2p_usage_counter["fused_matmul_reduce_scatter"] += 1

    # Move the gather_dim to the front and flatten the tensor into a 2D matrix
    x = A.movedim(scatter_dim, 0)
    leading_dims = [group.size()] + list(x.shape[:-1])
    leading_dims[1] //= group.size()
    x = x.flatten(0, -2)
    shards = x.chunk(group.size())

    # Computing block-wise matmul along the first dim of A
    def chunk_producer(rank: int, out: torch.Tensor) -> None:
        torch.matmul(shards[rank], B, out=out)

    stacked_partials = x.new_empty(x.shape[0], B.shape[1])

    _pipelined_produce_and_all2all(
        chunk_producer,
        stacked_partials,
        group,
    )
    # Ensures that the transpose and reduction produce contiguous result
    # in a single reduction kernel.
    return reduce_fn(
        stacked_partials.view(*leading_dims, -1)
        .movedim(1, scatter_dim + 1)
        .movedim(0, scatter_dim),
        dim=scatter_dim,
    )
