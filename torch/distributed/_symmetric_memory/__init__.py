import socket
import uuid

from contextlib import contextmanager
from functools import partial
from typing import Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import _SymmetricMemory

_group_name_to_store: Dict[str, c10d.Store] = {}


def enable_symm_mem_for_group(group_name: str) -> None:
    """
    Enables symmetric memory for a process group.

    Args:
        group_name (str): the name of the process group.
    """
    if group_name in _group_name_to_store:
        return

    group = c10d._resolve_process_group(group_name)
    store = c10d.PrefixStore(
        "symmetric_memory",
        c10d._get_process_group_store(group),
    )
    # Use one store-based broadcast to bootstrap a file store from the process
    # and simultaneously verify that all ranks are on the same host.
    hostname = socket.gethostname()
    if group.rank() == 0:
        uid = str(uuid.uuid4())
        msg = f"{hostname}/{uid}"
        store.set("init", msg)
    else:
        msg = store.get("init").decode("utf-8")
        tokens = msg.split("/")
        assert len(tokens) == 2, tokens
        rank_0_hostname, uid = tokens
        if hostname != rank_0_hostname:
            raise RuntimeError(
                "init_symmetric_memory_for_process_group() failed for "
                f'group "{group_name}". Rank 0 and rank {group.rank()} '
                f"are on different hosts ({rank_0_hostname} and {hostname})"
            )
    store = torch._C._distributed_c10d.FileStore(f"/tmp/{uid}", group.size())
    # TODO: check device connectiivity
    _group_name_to_store[group_name] = store
    _SymmetricMemory.set_group_info(
        group_name,
        group.rank(),
        group.size(),
        store,
    )


_is_test_mode: bool = False


@contextmanager
def _test_mode() -> Generator[None, None, None]:
    """
    Forces ``is_symm_mem_enabled_for_group()`` to return ``True`` and the ops
    defined in the ``symm_mem`` namespace to use fallback implementations.

    The context manager is not thread safe.
    """
    global _is_test_mode
    prev = _is_test_mode
    try:
        _is_test_mode = True
        yield
    finally:
        _is_test_mode = prev


def is_symm_mem_enabled_for_group(group_name: str) -> bool:
    """
    Check if symmetric memory is enabled for a process group.

    Args:
        group_name (str): the name of the process group.
    """
    return _is_test_mode or group_name in _group_name_to_store


_group_name_to_workspace_tensor: Dict[str, Optional[torch.Tensor]] = {}


def get_symm_mem_workspace(group_name: str, min_size: int) -> _SymmetricMemory:
    """
    Get the symmetric memory workspace associated with the process group. If
    ``min_size`` is greater than the workspace associated with ``group_name``,
    the workspace will be re-allocated and re-rendezvous'd.

    Args:
        group_name (str): the name of the process group.
        min_size (int): the size requirement for the workspace in bytes.

    Returns:
        _SymmetricMemory: the symmetric memory workspace associated with the
        group.
    """
    tensor = _group_name_to_workspace_tensor.get(group_name)
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    if tensor is None or size < min_size:
        tensor = _SymmetricMemory.empty_strided_p2p(
            (max(size, min_size),),
            [1],
            torch.uint8,
            torch.device(f"cuda:{torch.cuda.current_device()}"),
            group_name,
        )
        _group_name_to_workspace_tensor[group_name] = tensor
    return _SymmetricMemory.rendezvous(tensor)


_backend_stream: Optional[torch.cuda.Stream] = None


def _get_backend_stream() -> torch.cuda.Stream:
    global _backend_stream
    if _backend_stream is None:
        _backend_stream = torch.cuda.Stream()
    return _backend_stream


def _pipelined_all_gather_and_consume(
    shard: torch.Tensor,
    shard_consumer: Callable[[torch.Tensor, int], None],
    ag_out: torch.Tensor,
    group_name: str,
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
    p2p_workspace_size_req = shard.numel() * shard.element_size()
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(torch.cuda.current_stream())
    local_p2p_buf = symm_mem.get_buffer(rank, shard.shape, shard.dtype)

    chunks = ag_out.chunk(group_size)

    # While consuming local shard, copy it to the local p2p buffer
    # in another stream.
    shard_consumer(shard, rank)
    chunks[rank].copy_(shard)

    with torch.cuda.stream(backend_stream):
        local_p2p_buf.copy_(shard)
        symm_mem.barrier(channel=0)
    torch.cuda.current_stream().wait_stream(backend_stream)

    # At this point, all ranks have copied their local shard to
    # their local p2p buffer. Each rank can now copy and consume
    # remote shards.
    for step in range(1, group_size):
        if step % 2 == 0:
            stream = torch.cuda.current_stream()
        else:
            stream = backend_stream
        remote_rank = (step + rank) % group_size
        remote_p2p_buf = symm_mem.get_buffer(remote_rank, shard.shape, shard.dtype)
        with torch.cuda.stream(stream):
            chunks[remote_rank].copy_(remote_p2p_buf)
            shard_consumer(chunks[remote_rank], remote_rank)

    with torch.cuda.stream(backend_stream):
        symm_mem.barrier(channel=group_size % 2)
    torch.cuda.current_stream().wait_stream(backend_stream)


def _pipelined_produce_and_all2all(
    chunk_producer: Callable[[int, torch.Tensor], None],
    output: torch.Tensor,
    group_name: str,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        chunks = [
            chunk_producer(dst_rank, chunks[dst_rank])
            for dst_rank in range(group_size):
        ]
        dist.all_to_all_single(output=output, input=torch.cat(chunks))
    """
    out_chunks = output.chunk(c10d._get_group_size_by_name(group_name))
    p2p_workspace_size_req = out_chunks[0].numel() * out_chunks[0].element_size() * 2
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(torch.cuda.current_stream())

    def get_p2p_buf(rank: int, idx: int) -> torch.Tensor:
        assert idx in (0, 1)
        offset = 0 if idx == 0 else out_chunks[0].numel()
        return symm_mem.get_buffer(
            rank, out_chunks[0].shape, out_chunks[0].dtype, offset
        )

    # Prepare two local p2p buffers, so that a remote rank can pull the result
    # of step [i] in one p2p buffer while the local rank can compute the
    # result of step [i+1] and write it directly the other p2p buffer.
    local_p2p_buf_0 = get_p2p_buf(rank, 0)
    local_p2p_buf_1 = get_p2p_buf(rank, 1)

    for step in range(1, group_size):
        remote_rank = (rank - step) % group_size
        if step % 2 == 0:
            stream = torch.cuda.current_stream()
            other_stream = backend_stream
            p2p_buf = local_p2p_buf_1
            remote_p2p_buf = get_p2p_buf(remote_rank, 1)
        else:
            stream = backend_stream
            other_stream = torch.cuda.current_stream()
            p2p_buf = local_p2p_buf_0
            remote_p2p_buf = get_p2p_buf(remote_rank, 0)
        with torch.cuda.stream(stream):
            chunk_producer((rank + step) % group_size, p2p_buf)
            symm_mem.barrier(channel=step % 2)
            # Make the other stream to wait for the barrier on the current
            # stream to finish before chunk_producer to avoid the compute
            # delaying the barrier.
            other_stream.wait_stream(stream)
            out_chunks[remote_rank].copy_(remote_p2p_buf)

    chunk_producer(rank, out_chunks[rank])
    torch.cuda.current_stream().wait_stream(backend_stream)


lib = torch.library.Library("symm_mem", "DEF")  # noqa: TOR901
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

    Optimal stride order for A_shard - if A_shard.movedim(scatter_dim, 0) is
    contiguous, no extra copy is required for input layout transformation.
    Otherwise A_shard needs to be copied once.

    """
    if _is_test_mode:
        return _fused_all_gather_matmul_fallback(A_shard, Bs, gather_dim, group_name)
    if A_shard.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    for B in Bs:
        if B.dim() != 2:
            raise ValueError("B must be a matrix")
    if gather_dim < 0 or gather_dim >= A_shard.dim():
        raise ValueError("Invalid gather_dim")

    group = c10d._resolve_process_group(group_name)

    with torch.profiler.record_function("fused_all_gather_matmul"):
        # Move the gather_dim to the front and flatten the tensor into a 2D matrix.
        # The flattened tensor doesn't need to be contiguous (for computation
        # efficiency), as _pipelined_all_gather_and_consume guarantees that shards
        # passed to shard_consumer are contiguous.
        x = A_shard.movedim(gather_dim, 0)
        leading_dims = [group.size()] + list(x.shape[:-1])
        x = x.flatten(0, -2)

        # Helper function for reverting the above transformation
        def unflatten(t: torch.Tensor) -> torch.Tensor:
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
            group_name,
        )
        return unflatten(ag_out), [unflatten(output) for output in outputs]


def make_contiguous_for_perm(
    t: torch.Tensor,
    perm: List[int],
) -> torch.Tensor:
    """
    Restride `t` such that `t.permute(perm)` is contiguous.
    """
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return t.permute(perm).contiguous().permute(inv_perm)


def restride_A_shard_for_fused_all_gather_matmul(
    t: torch.Tensor,
    scatter_dim: int,
) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_all_gather_matmul` for optimal perf.
    See the doc for `fused_all_gather_matmul` for detail.
    """
    perm = list(range(len(t.shape)))
    perm.insert(0, perm.pop(scatter_dim))
    return make_contiguous_for_perm(t, perm)


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

    Optimal stride order for A - if A.movedim(scatter_dim, 0) is contiguous, no
    extra copy is required for input layout transformation. Otherwise A needs
    to be copied once.

    NOTE:
    - The K dim across ranks are currently accumulated with bf16 which results
      in accuracy loss.
    """
    if _is_test_mode:
        return _fused_matmul_reduce_scatter_fallback(
            A, B, reduce_op, scatter_dim, group_name
        )
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

    with torch.profiler.record_function("fused_matmul_reduce_scatter"):
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
            group_name,
        )
        # Ensures that the transpose and reduction produce contiguous result
        # in a single reduction kernel.
        return reduce_fn(
            stacked_partials.view(*leading_dims, -1)
            .movedim(1, scatter_dim + 1)
            .movedim(0, scatter_dim),
            dim=scatter_dim,
        )


def restride_A_for_fused_matmul_reduce_scatter(
    t: torch.Tensor,
    gather_dim: int,
) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_matmul_reduce_scatter` for optimal
    perf. See the doc for `fused_matmul_reduce_scatter` for detail.
    """
    perm = list(range(len(t.shape)))
    perm.insert(0, perm.pop(gather_dim))
    return make_contiguous_for_perm(t, perm)
