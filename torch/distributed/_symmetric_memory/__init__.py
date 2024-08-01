# mypy: allow-untyped-decorators
import socket
import uuid
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import _SymmetricMemory, Work as _Work


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
    global_ranks = sorted(c10d._world.pg_group_ranks[group].keys())
    # Different subgroups with the same name should use different stores
    global_ranks_str = "_".join(map(str, global_ranks))
    store = c10d.PrefixStore(
        f"symmetric_memory-{global_ranks_str}",
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
    "fused_all_gather_scaled_matmul("
    "Tensor A, Tensor[] Bs, Tensor A_scale, Tensor[] B_scales, "
    "ScalarType?[] out_dtypes, int gather_dim, str group_name) -> (Tensor, Tensor[])"
)
lib.define(
    "fused_matmul_reduce_scatter(Tensor A, Tensor B, str reduce_op, int scatter_dim, str group_name) -> Tensor"
)
lib.define(
    "fused_scaled_matmul_reduce_scatter("
    "Tensor A, Tensor B, Tensor A_scale, Tensor B_scale, "
    "ScalarType out_dtype, str reduce_op, int scatter_dim, str group_name) -> Tensor"
)
lib.define("_low_contention_all_gather(Tensor tensor, str group_name) -> Tensor")
lib.define(
    "_low_contention_reduce_scatter(Tensor tensor, str reduce_op, str group_name) -> Tensor"
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

    Optimal stride order for A_shard - if A_shard.movedim(gather_dim, 0) is
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
    gather_dim: int,
) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_all_gather_matmul` for optimal perf.
    See the doc for `fused_all_gather_matmul` for detail.
    """
    perm = list(range(len(t.shape)))
    perm.insert(0, perm.pop(gather_dim))
    return make_contiguous_for_perm(t, perm)


def _fused_matmul_reduce_scatter_impl(
    mm_out_op: torch._ops.OpOverload,
    A: torch.Tensor,
    B: torch.Tensor,
    kwargs: Dict[str, Any],
    out_dtype: Optional[torch.dtype],
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
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

    # Move the gather_dim to the front and flatten the tensor into a 2D matrix
    x = A.movedim(scatter_dim, 0)
    leading_dims = [group.size()] + list(x.shape[:-1])
    leading_dims[1] //= group.size()
    x = x.flatten(0, -2)
    shards = x.chunk(group.size())

    # Computing block-wise matmul along the first dim of A
    def chunk_producer(rank: int, out: torch.Tensor) -> None:
        mm_out_op(shards[rank], B, **kwargs, out=out)

    stacked_partials = x.new_empty(x.shape[0], B.shape[1], dtype=out_dtype or A.dtype)

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
    """
    if _is_test_mode:
        return _fused_matmul_reduce_scatter_fallback(
            A, B, reduce_op, scatter_dim, group_name
        )

    with torch.profiler.record_function("fused_matmul_reduce_scatter"):
        return _fused_matmul_reduce_scatter_impl(
            mm_out_op=torch.ops.aten.mm.out,
            A=A,
            B=B,
            kwargs={},
            out_dtype=A.dtype,
            reduce_op=reduce_op,
            scatter_dim=scatter_dim,
            group_name=group_name,
        )


@torch.library.impl(lib, "fused_scaled_matmul_reduce_scatter", "Meta")
def _fused_scaled_matmul_reduce_scatter_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out_dtype: Optional[torch.dtype],
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    C = torch._scaled_mm(
        A.flatten(0, -2).contiguous(),
        B,
        A_scale,
        B_scale,
        out_dtype=out_dtype,
    )
    C = C.view(*A.shape[:-1], B.shape[1])
    res = funcol.reduce_scatter_tensor(
        C,
        reduce_op,
        scatter_dim,
        group_name,
    )
    res = funcol.wait_tensor(res)
    return res


@torch.library.impl(lib, "fused_scaled_matmul_reduce_scatter", "CUDA")
def _fused_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out_dtype: Optional[torch.dtype],
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    if _is_test_mode:
        return _fused_scaled_matmul_reduce_scatter_fallback(
            A, B, A_scale, B_scale, out_dtype, reduce_op, scatter_dim, group_name
        )
    with torch.profiler.record_function("fused_matmul_reduce_scatter"):
        return _fused_matmul_reduce_scatter_impl(
            mm_out_op=torch.ops.aten._scaled_mm.out,
            A=A,
            B=B,
            kwargs={"scale_a": A_scale, "scale_b": B_scale, "out_dtype": out_dtype},
            out_dtype=out_dtype,
            reduce_op=reduce_op,
            scatter_dim=scatter_dim,
            group_name=group_name,
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


def _maybe_convert_scalar_types_to_dtypes(
    scalar_types: List[Any],
) -> List[Optional[torch.dtype]]:
    """
    When a list of `torch.dtype`s is passed through the dispatcher as
    `ScalarType[]`, it is converted to a list of scalar type enum values. This
    function converts it back to a list of `torch.dtype`s.
    """
    # Order defined in https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
    _SCALAR_TYPE_TO_DTYPE = {
        0: torch.uint8,
        1: torch.int8,
        2: torch.short,
        3: torch.int,
        4: torch.int64,
        5: torch.half,
        6: torch.float,
        7: torch.double,
        8: torch.complex32,
        9: torch.complex64,
        10: torch.complex128,
        11: torch.bool,
        12: torch.qint8,
        13: torch.quint8,
        14: torch.qint32,
        15: torch.bfloat16,
        16: torch.float8_e5m2,
        17: torch.float8_e4m3fn,
        18: torch.float8_e5m2fnuz,
        19: torch.float8_e4m3fnuz,
    }
    if any(not isinstance(x, (type(None), int)) for x in scalar_types):
        return scalar_types

    dtypes: List[Optional[torch.dtype]] = []
    for scalar_type in scalar_types:
        if scalar_type is None:
            dtypes.append(scalar_type)
        elif scalar_type not in _SCALAR_TYPE_TO_DTYPE:
            raise ValueError("Unrecognized scalar type {scalar_type}")
        else:
            dtypes.append(_SCALAR_TYPE_TO_DTYPE[scalar_type])
    return dtypes


@torch.library.impl(lib, "fused_all_gather_scaled_matmul", "Meta")
def _fused_all_gather_scaled_matmul_fallback(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    A_scale: torch.Tensor,
    B_scales: List[torch.Tensor],
    out_dtypes: List[Optional[torch.dtype]],
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    out_dtypes = _maybe_convert_scalar_types_to_dtypes(out_dtypes)

    group_size = c10d._get_group_size_by_name(group_name)
    A = torch.ops._c10d_functional.all_gather_into_tensor(
        A_shard.contiguous(), group_size, group_name
    )
    A = torch.ops._c10d_functional.wait_tensor(A)
    A = A.view(group_size, *A_shard.shape).movedim(gather_dim + 1, 1).flatten(0, 1)

    def scaled_matmul(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        leading_dims = A.shape[:-1]
        res = torch.ops.aten._scaled_mm(
            A.flatten(0, -2), B, A_scale, B_scale, out_dtype=out_dtype
        )
        return res.unflatten(0, leading_dims)

    return A.movedim(0, gather_dim), [
        scaled_matmul(A, B, A_scale, B_scale, out_dtype).movedim(0, gather_dim)
        for B, B_scale, out_dtype in zip(Bs, B_scales, out_dtypes)
    ]


@torch.library.impl(lib, "fused_all_gather_scaled_matmul", "CUDA")
def _fused_all_gather_scaled_matmul(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    A_scale: torch.Tensor,
    B_scales: List[torch.Tensor],
    out_dtypes: List[Optional[torch.dtype]],
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        A = all_gather_tensor(A_shard, gather_dim, group_name)
        leading_dims = A.shape[:-1]
        res = torch.ops.aten._scaled_mm(A.flatten(0, -2), B, A_scale, B_scale)
        res = res.unflatten(0, leading_dims)

    Optimal stride order for A_shard - if A_shard.movedim(gather_dim, 0) is
    contiguous, no extra copy is required for input layout transformation.
    Otherwise A_shard needs to be copied once.
    """
    out_dtypes = _maybe_convert_scalar_types_to_dtypes(out_dtypes)

    if _is_test_mode:
        return _fused_all_gather_scaled_matmul_fallback(
            A_shard, Bs, A_scale, B_scales, out_dtypes, gather_dim, group_name
        )
    if A_shard.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    for B in Bs:
        if B.dim() != 2:
            raise ValueError("B must be a matrix")
    if gather_dim < 0 or gather_dim >= A_shard.dim():
        raise ValueError("Invalid gather_dim")

    group = c10d._resolve_process_group(group_name)

    with torch.profiler.record_function("fused_all_gather_scaled_matmul"):
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
                dtype=out_dtype if out_dtype is not None else A_shard.dtype,
            )
            for B, out_dtype in zip(Bs, out_dtypes)
        ]
        output_shards = [output.chunk(group.size()) for output in outputs]

        # Computing block-wise matmul along the first dim of A
        def shard_consumer(shard: torch.Tensor, rank: int) -> None:
            for idx, (B, B_scale, out_dtype) in enumerate(
                zip(Bs, B_scales, out_dtypes)
            ):
                torch.ops.aten._scaled_mm(
                    shard,
                    B,
                    A_scale,
                    B_scale,
                    out_dtype=out_dtype,
                    out=output_shards[idx][rank],
                )

        _pipelined_all_gather_and_consume(
            x,
            shard_consumer,
            ag_out,
            group_name,
        )
        return unflatten(ag_out), [unflatten(output) for output in outputs]


class Work(_Work):
    def __init__(self) -> None:
        super().__init__()
        self.event = torch.cuda.Event()
        self.event.record()

    def wait(self, timeout: timedelta = timedelta(seconds=0)) -> bool:
        self.event.wait()
        return True


"""
NOTE [low-contention collectives]
When a collective is overlapped with abundant compute, it makes sense to
prioritize reducing the contention between the collective and the overlapped
compute, even at the cost of a slightly slower collective.

Common collective implementations (e.g., NCCL without user buffer
registration) optimize for throughput with no ambient compute. However, such
implementations may not be optimal when they are overlapped with compute:
- These implementations typically fuse the entire collective into a single
kernel and reserve SM resources based on the most demanding portion of the
collective, even when a large portion of the collective does not require this
much resource.
- These implementations often use SM-based P2P copy as opposed to copy
engine-based P2P copy. Copy engine-based P2P copy may not have a significant
advantage when there's no ambient compute. However, it may significantly
improve overall resource utilization in the presence of ambient compute.

When overlapped with intensive compute (e.g., persistent matmul kernels), the
SM-usage of a collective can lead to inefficient overlapping.

Low-contention collectives achieve their goals with the following strategies:
- Use copy engine-based copy whenever possible.
- Break down portions of a collective with different resource requirements
into multiple kernels. This improves the overlapping efficiency at the cost
of additional launching overhead.
"""


@torch.library.impl(lib, "_low_contention_all_gather", "Meta")
def _low_contention_all_gather_meta(
    tensor: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    group_size = c10d._get_group_size_by_name(group_name)
    return tensor.new_empty(tensor.shape[0] * group_size, *tensor.shape[1:])


@torch.library.impl(lib, "_low_contention_all_gather", "CUDA")
def _low_contention_all_gather(
    tensor: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    """
    Performs all-gather with symmetric memory in a low-contention fashion.

    When `tensor` is already in symmetric memory:
        - The collective is carried out without using SMs.
        - No symmetric memory workspace is required.

    When `tensor` is not in symmetric memory:
        - An extra SM-based copy is performed to copy the input data into the
          symmetric memory workspace.
        - Symmetric memory workspace size requirement: the size of `tensor`.
    """
    symm_mem = _SymmetricMemory.rendezvous(tensor)
    if symm_mem is not None:
        input_is_symm_mem = True
    else:
        symm_mem = get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size()
        )
        input_is_symm_mem = False

    rank = symm_mem.rank
    world_size = symm_mem.world_size

    output = tensor.new_empty(tensor.shape[0] * world_size, *tensor.shape[1:])
    chunks = output.chunk(world_size)

    _get_backend_stream().wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_get_backend_stream()):
        if not input_is_symm_mem:
            local_buf = symm_mem.get_buffer(rank, tensor.shape, tensor.dtype)
            local_buf.copy_(tensor)
        # pull
        symm_mem.barrier()
        for step in range(0, world_size):
            remote_rank = (rank - step) % world_size
            src_buf = symm_mem.get_buffer(remote_rank, tensor.shape, tensor.dtype)
            chunks[remote_rank].copy_(src_buf)
        symm_mem.barrier()
        torch._C._distributed_c10d._register_work(output, Work())
        return output


@torch.library.impl(lib, "_low_contention_reduce_scatter", "Meta")
def _low_contention_reduce_scatter_meta(
    tensor: torch.Tensor,
    reduce_op: str,
    group_name: str,
) -> torch.Tensor:
    group_size = c10d._get_group_size_by_name(group_name)
    return tensor.unflatten(0, (group_size, -1)).mean(dim=0)


def _low_contention_reduce_scatter_with_symm_mem_input(
    tensor: torch.Tensor,
    reduce_op: str,
    symm_mem: _SymmetricMemory,
) -> torch.Tensor:
    rank = symm_mem.rank
    world_size = symm_mem.world_size

    assert tensor.shape[0] % world_size == 0
    a2a_res = torch.empty_like(tensor)
    chunks = a2a_res.chunk(world_size)

    _get_backend_stream().wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_get_backend_stream()):
        # pull + offline reduction
        symm_mem.barrier()
        for step in range(0, world_size):
            remote_rank = (rank - step) % world_size
            src_buf = symm_mem.get_buffer(
                remote_rank,
                chunks[0].shape,
                chunks[0].dtype,
                chunks[0].numel() * rank,
            )
            chunks[remote_rank].copy_(src_buf)
        symm_mem.barrier()

        ret = a2a_res.unflatten(0, (world_size, -1))
        if reduce_op == "sum":
            ret = ret.sum(dim=0)
        elif reduce_op == "avg":
            ret = ret.mean(dim=0)
        else:
            raise ValueError(f"reduce_op ({reduce_op}) is not supported")
        torch._C._distributed_c10d._register_work(ret, Work())
        return ret


def _low_contention_reduce_scatter_with_workspace(
    tensor: torch.Tensor,
    reduce_op: str,
    workspace: _SymmetricMemory,
) -> torch.Tensor:
    rank = workspace.rank
    world_size = workspace.world_size

    assert tensor.shape[0] % world_size == 0
    chunks = tensor.chunk(world_size)

    _get_backend_stream().wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_get_backend_stream()):
        # push + offline reduction
        workspace.barrier()
        for step in range(0, world_size):
            remote_rank = (rank - step) % world_size
            dst_buf = workspace.get_buffer(
                remote_rank, chunks[0].shape, chunks[0].dtype, chunks[0].numel() * rank
            )
            dst_buf.copy_(chunks[remote_rank])
        workspace.barrier()

        buf = workspace.get_buffer(rank, tensor.shape, tensor.dtype)
        ret = buf.unflatten(0, (world_size, -1))
        if reduce_op == "sum":
            ret = ret.sum(dim=0)
        elif reduce_op == "avg":
            ret = ret.mean(dim=0)
        else:
            raise ValueError(f"reduce_op ({reduce_op}) is not supported")
        torch._C._distributed_c10d._register_work(ret, Work())
        return ret


@torch.library.impl(lib, "_low_contention_reduce_scatter", "CUDA")
def _low_contention_reduce_scatter(
    tensor: torch.Tensor,
    reduce_op: str,
    group_name: str,
) -> torch.Tensor:
    """
    Performs reduce-scatter with symmetric memory in a low-contention fashion.

    This implementation performs a P2P-based all-to-all followed by an offline
    reduction.

    When `tensor` is already in symmetric memory:
        - Pull-based all-to-all is used.
        - No symmetric memory workspace is required.

    When `tensor` is not in symmetric memory:
        - Push-based all-to-all is used.
        - Symmetric memory workspace size requirement: the size of `tensor`.

    SM-usage:
        - SM-based copy of the rank's own chunk for the all-to-all.
        - Reduction on the all-to-all result.

    TODO(yifu): the SM-based copy can be avoided with a list-based reduction
    kernel.
    """
    symm_mem = _SymmetricMemory.rendezvous(tensor)
    if symm_mem is not None:
        return _low_contention_reduce_scatter_with_symm_mem_input(
            tensor, reduce_op, symm_mem
        )
    else:
        workspace = get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size()
        )
        return _low_contention_reduce_scatter_with_workspace(
            tensor, reduce_op, workspace
        )
