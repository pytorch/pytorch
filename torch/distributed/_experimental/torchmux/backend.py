"""TorchMux distributed backend.

A custom ``torch.distributed`` backend that routes collectives through
the checkpoint coordinator instead of a dedicated interconnect like NCCL.

Usage::

    import torch.distributed._experimental.torchmux  # registers "torchmux"

    dist.init_process_group(backend="torchmux", ...)

During ``init_process_group``:

1. Rank 0 starts a coordinator server in a background thread and publishes
   its address to the distributed store.
2. All ranks connect and register.  Rank 0 is unblocked immediately; later
   ranks remain blocked in register until their rank-ordered turn and the
   active rank yields.

Set ``COORD_ADDR`` in the environment to skip auto-start and connect
all ranks to an external coordinator instead.
"""

import asyncio
import os
import threading
from datetime import timedelta
from typing import Any, NoReturn, overload

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    _create_work_from_future,
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
)
from torch.futures import Future


def _not_implemented(op_name: str) -> NoReturn:
    raise NotImplementedError(f"torchmux: {op_name} not yet implemented")


def _completed_work(result: object) -> dist.Work:
    future: Future[object] = Future()
    future.set_result(result)
    return _create_work_from_future(future)


def _contains_cuda_tensor(obj: object) -> bool:
    if isinstance(obj, dict):
        return any(_contains_cuda_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_cuda_tensor(v) for v in obj)
    return bool(getattr(obj, "is_cuda", False))


def _reduce_tensor(dst: torch.Tensor, src: torch.Tensor, op: ReduceOp) -> None:
    if op == dist.ReduceOp.SUM or op == dist.ReduceOp.AVG:
        dst.add_(src)
    elif op == dist.ReduceOp.PRODUCT:
        dst.mul_(src)
    elif op == dist.ReduceOp.MIN:
        torch.minimum(dst, src, out=dst)
    elif op == dist.ReduceOp.MAX:
        torch.maximum(dst, src, out=dst)
    elif op == dist.ReduceOp.BAND:
        dst.bitwise_and_(src)
    elif op == dist.ReduceOp.BOR:
        dst.bitwise_or_(src)
    elif op == dist.ReduceOp.BXOR:
        dst.bitwise_xor_(src)
    else:
        _not_implemented(f"allreduce op {op}")


def _finish_reduce(dst: torch.Tensor, op: ReduceOp, world_size: int) -> None:
    if op == dist.ReduceOp.AVG:
        dst.div_(world_size)


def _run_coordinator_in_thread() -> str:
    """Start the coordinator server in a daemon thread. Returns the TCP address."""
    from .coordinator import Coordinator

    ready = threading.Event()
    addr_box: list[str] = []

    async def _serve() -> None:
        coord = Coordinator()
        server = await asyncio.start_server(
            coord.handle_connection, host="127.0.0.1", port=0
        )
        bound = server.sockets[0].getsockname()
        addr_box.append(f"tcp:{bound[0]}:{bound[1]}")
        ready.set()
        await server.serve_forever()

    def _thread_main() -> None:
        asyncio.run(_serve())

    t = threading.Thread(target=_thread_main, daemon=True, name="torchmux-coord")
    t.start()
    ready.wait()
    return addr_box[0]


# -- ProcessGroup ------------------------------------------------------------


class ProcessGroupTorchmux(dist.ProcessGroup):
    """ProcessGroup whose collectives go through the checkpoint coordinator."""

    def __init__(
        self, store: dist.Store, rank: int, size: int, timeout: timedelta
    ) -> None:
        super().__init__(rank, size)
        self._store = store
        self._timeout = timeout
        self._rank = rank
        self._size = size
        self._others = tuple(r for r in range(size) if r != rank)

        coord_addr = os.environ.get("COORD_ADDR")
        if coord_addr is None:
            if rank == 0:
                coord_addr = _run_coordinator_in_thread()
                store.set("coord_addr", coord_addr)
            coord_addr = store.get("coord_addr").decode()

        from .coord_client import CoordClient

        self._client = CoordClient(addr=coord_addr)
        self._client.register(rank)

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.done()

    def getBackendName(self) -> str:
        return "torchmux"

    def _exchange(
        self,
        send: dict[tuple[int, ...], torch.Tensor | None],
        recv: tuple[int, ...],
        tensors: object = (),
        force_cuda: bool = False,
    ) -> dict[int, torch.Tensor | None]:
        """Run prepare + wait_for_recv if needed. Returns the recv dict."""
        result = self._client.prepare(send, recv)
        if result is not None:
            return result
        if force_cuda or _contains_cuda_tensor(send) or _contains_cuda_tensor(tensors):
            return self._client.release_gpu()
        return self._client.wait_for_recv()

    # -- collectives -----------------------------------------------------------
    #
    # The parent ProcessGroup has pybind11 overloads for each collective (e.g.
    # allreduce(tensors, opts), allreduce(tensors, op, timeout), allreduce(
    # tensor, op, timeout)).  The convenience overloads construct an options
    # struct in C++ and dispatch to the virtual (tensors, opts) form — so the
    # Python override only ever receives the first signature.  We declare
    # @overload stubs to match the parent's full signature set.

    @overload
    def allreduce(
        self, tensors: list[torch.Tensor], opts: AllreduceOptions = ...
    ) -> dist.Work: ...

    @overload
    def allreduce(
        self,
        tensors: list[torch.Tensor],
        op: ReduceOp = ...,
        timeout: timedelta | None = None,
    ) -> dist.Work: ...

    @overload
    def allreduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ...,
        timeout: timedelta | None = None,
    ) -> dist.Work: ...

    def allreduce(self, *args: Any, **kwargs: Any) -> dist.Work:
        tensors, opts = args
        t = tensors[0]
        others = self._others
        received = self._exchange(
            send={others: t},
            recv=others,
            tensors=(t,),
        )
        op = opts.reduceOp
        for peer_t in received.values():
            assert peer_t is not None
            _reduce_tensor(t, peer_t, op)
        _finish_reduce(t, op, self._size)
        return _completed_work(tensors)

    @overload
    def broadcast(
        self, tensors: list[torch.Tensor], opts: BroadcastOptions = ...
    ) -> dist.Work: ...

    @overload
    def broadcast(
        self,
        tensor: torch.Tensor,
        root: int,
        timeout: timedelta | None = None,
    ) -> dist.Work: ...

    def broadcast(self, *args: Any, **kwargs: Any) -> dist.Work:
        tensors, opts = args
        t = tensors[0]
        src = opts.rootRank
        others = self._others
        if self._rank == src:
            self._exchange(send={others: t}, recv=(), tensors=(t,))
        else:
            received = self._exchange(send={}, recv=(src,), tensors=(t,))
            src_tensor = received[src]
            assert src_tensor is not None
            t.copy_(src_tensor)
        return _completed_work(tensors)

    @overload
    def allgather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions = ...,
    ) -> dist.Work: ...

    @overload
    def allgather(
        self,
        output_tensors: list[torch.Tensor],
        input_tensor: torch.Tensor,
        timeout: timedelta | None = None,
    ) -> dist.Work: ...

    def allgather(self, *args: Any, **kwargs: Any) -> dist.Work:
        output_tensors, input_tensors, opts = args
        t = input_tensors[0]
        out_list = output_tensors[0]
        others = self._others
        received = self._exchange(
            send={others: t},
            recv=others,
            tensors=(t, out_list),
        )
        for r in range(self._size):
            if r == self._rank:
                out_list[r].copy_(t)
            else:
                peer_t = received[r]
                assert peer_t is not None
                out_list[r].copy_(peer_t)
        return _completed_work(output_tensors)

    @overload
    def barrier(self, opts: BarrierOptions = ...) -> dist.Work: ...

    @overload
    def barrier(self, timeout: timedelta | None = None) -> dist.Work: ...

    def barrier(self, *args: Any, **kwargs: Any) -> dist.Work:
        opts = args[0] if args else None
        others = self._others
        force_cuda = getattr(getattr(opts, "device", None), "type", None) == "cuda"
        self._exchange(send={others: None}, recv=others, force_cuda=force_cuda)
        return _completed_work(None)

    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> dist.Work:
        self._exchange(send={(dst_rank,): tensors[0]}, recv=(), tensors=tensors)
        return _completed_work(tensors)

    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> dist.Work:
        received = self._exchange(send={}, recv=(src_rank,), tensors=tensors)
        src_tensor = received[src_rank]
        assert src_tensor is not None
        tensors[0].copy_(src_tensor)
        return _completed_work(tensors)

    # -- not yet implemented ---------------------------------------------------

    def allreduce_coalesced(self, *args: Any, **kwargs: Any) -> dist.Work:
        _not_implemented("allreduce_coalesced")

    def allgather_into_tensor_coalesced(self, *args: Any, **kwargs: Any) -> dist.Work:
        _not_implemented("allgather_into_tensor_coalesced")

    @overload
    def reduce_scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ReduceScatterOptions = ...,
    ) -> dist.Work: ...

    @overload
    def reduce_scatter(
        self,
        output_tensors: torch.Tensor,
        input_tensor: list[torch.Tensor],
        op: ReduceOp = ...,
        timeout: timedelta | None = None,
    ) -> dist.Work: ...

    def reduce_scatter(self, *args: Any, **kwargs: Any) -> dist.Work:
        _not_implemented("reduce_scatter")

    def reduce_scatter_tensor_coalesced(self, *args: Any, **kwargs: Any) -> dist.Work:
        _not_implemented("reduce_scatter_tensor_coalesced")

    @overload
    def alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: AllToAllOptions = ...,
    ) -> dist.Work: ...

    @overload
    def alltoall_base(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        timeout: timedelta | None = None,
    ) -> dist.Work: ...

    def alltoall_base(self, *args: Any, **kwargs: Any) -> dist.Work:
        _not_implemented("alltoall_base")


# -- Registration ------------------------------------------------------------


def _create_torchmux(
    store: dist.Store, rank: int, size: int, timeout: timedelta
) -> ProcessGroupTorchmux:
    return ProcessGroupTorchmux(store, rank, size, timeout)


dist.Backend.register_backend(
    "torchmux",
    _create_torchmux,
    extended_api=False,
    devices=["cpu", "cuda"],
)
