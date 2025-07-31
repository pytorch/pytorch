"""
This is an experimental new API for PyTorch Distributed. This is actively in development and subject to change or deletion entirely.

This is intended as a proving ground for more flexible and object oriented distributed APIs.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from typing import Protocol, Union

import torch
from torch._C._distributed_c10d import (
    _current_process_group,
    _set_process_group,
    Backend,
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    ProcessGroup,
    ReduceOp,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
)
from torch.distributed.rendezvous import rendezvous


_BACKENDS: dict[str, "CommunicatorFactory"] = {}

__all__ = [
    "Communicator",
    "ReduceOp",
    "CommunicatorFactory",
    "register_backend",
    "new_comm",
    "current_comm",
    "comm",
]


class Communicator:
    """
    A communicator allows for communication between processes. This maps 1:1
    with a single device and underlying Backend communicator.
    """

    def __init__(self, *, _pg: ProcessGroup) -> None:
        """
        Initialize a new communicator.

        This is an internal only API and should not be used directly.

        Args:
            pg: The process group to use.
        """
        self._pg = _pg

    @property
    def rank(self) -> int:
        """Get the rank of the current process."""
        return self._pg.rank()

    @property
    def size(self) -> int:
        """Get the size of the communicator."""
        return self._pg.size()

    @property
    def name(self) -> str:
        """Get the name of the communicator."""
        return self._pg.group_name

    @property
    def unsafe_backend(self) -> Backend:
        """
        This returns the raw backend implementation for experimentation and debugging purposes.

        WARNING: This provides no backwards compatibility guarantees nor
        compatibility across multiple backends.
        """
        return self._pg._get_default_backend()

    def shutdown(self) -> None:
        self._pg.shutdown()

    def abort(self) -> None:
        self._pg.abort()

    def broadcast(
        self, tensor: torch.Tensor, root: int, timeout: timedelta | None = None
    ) -> None:
        """
        Broadcasts the tensor to all processes in the group.

        Args:
            tensor: The tensor to broadcast.
            root: The root process to broadcast from.
            timeout: Optional timeout for the operation.
        """
        opts = BroadcastOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.rootRank = root
        opts.rootTensor = 0
        opts.asyncOp = False
        work = self._pg.broadcast([tensor], opts)
        return work

    def allreduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        timeout: timedelta | None = None,
    ) -> None:
        """
        Reduces the tensor data across all processes in a group in-place.

        Args:
            tensor: Input and output of the collective. The function operates
                in-place.
            op: One of the values from
                ``torch.distributed.ReduceOp``
                enum.  Specifies an operation used for element-wise
                reductions.
            timeout: Optional timeout for the operation.
        """
        opts = AllreduceOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.reduceOp = op
        opts.asyncOp = False
        work = self._pg.allreduce([tensor], opts)
        return work

    def allgather(
        self,
        output_tensors: list[torch.Tensor],
        input_tensor: torch.Tensor,
        timeout: timedelta | None = None,
    ) -> None:
        """
        Gathers tensors from all processes and distributes them to all processes.

        Args:
            output_tensors: List of tensors to be filled with the gathered tensors.
            input_tensor: Tensor to be gathered from all processes.
            timeout: Optional timeout for the operation.
        """
        opts = AllgatherOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.asyncOp = False
        work = self._pg.allgather([output_tensors], [input_tensor], opts)
        return work

    def reduce(
        self,
        tensor: torch.Tensor,
        root: int,
        op: ReduceOp = ReduceOp.SUM,
        timeout: timedelta | None = None,
    ) -> None:
        """
        Reduces the tensor data across all processes in a group to the root process.

        Args:
            tensor: Input and output of the collective. The function operates
                in-place.
            root: The root process to reduce to.
            op: One of the values from
                ``torch.distributed.ReduceOp``
                enum.  Specifies an operation used for element-wise
                reductions.
            timeout: Optional timeout for the operation.
        """
        opts = ReduceOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.reduceOp = op
        opts.rootRank = root
        opts.rootTensor = 0
        opts.asyncOp = False
        work = self._pg.reduce([tensor], opts)
        return work

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: list[torch.Tensor],
        op: ReduceOp = ReduceOp.SUM,
        timeout: timedelta | None = None,
    ) -> None:
        """
        Reduces, then scatters a list of tensors to all processes in a group.

        Args:
            output: Output tensor.
            input_list: List of tensors to reduce and scatter.
            op: One of the values from ``torch.distributed.ReduceOp`` enum.
                Specifies an operation used for element-wise reductions.
            timeout: Optional timeout for the operation.
        """
        opts = ReduceScatterOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.reduceOp = op
        opts.asyncOp = False
        work = self._pg.reduce_scatter([output], [input_list], opts)
        return work

    def barrier(self, timeout: timedelta | None = None) -> None:
        """
        Synchronizes all processes.

        This collective blocks processes until the whole group enters this function.

        Args:
            timeout: Optional timeout for the operation.
        """
        opts = BarrierOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.asyncOp = False
        work = self._pg.barrier(opts)
        return work

    def gather(
        self,
        output_tensors: list[torch.Tensor],
        input_tensor: torch.Tensor,
        root: int,
        timeout: timedelta | None = None,
    ) -> None:
        """
        Gathers a list of tensors from a single source process.

        Args:
            output_tensors: List of tensors to be filled with the gathered tensors.
                Only meaningful on the root process.
            input_tensor: Tensor to be gathered from all processes.
            root: The root process to gather to.
            timeout: Optional timeout for the operation.
        """
        opts = GatherOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.rootRank = root
        opts.asyncOp = False
        work = self._pg.gather(
            [output_tensors] if self.rank == root else [], [input_tensor], opts
        )
        return work

    def scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensors: list[torch.Tensor],
        root: int,
        timeout: timedelta | None = None,
    ) -> None:
        """
        Scatters a list of tensors to all processes in a group.

        Args:
            output_tensor: Output tensor.
            input_tensors: List of tensors to scatter.
                Only meaningful on the root process.
            root: The root process to scatter from.
            timeout: Optional timeout for the operation.
        """
        opts = ScatterOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.rootRank = root
        opts.asyncOp = False
        work = self._pg.scatter(
            [output_tensor], [input_tensors] if self.rank == root else [], opts
        )
        return work

    def all_to_all_single(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        timeout: timedelta | None = None,
    ) -> None:
        """
        Each process splits input tensor and then scatters the split list
        to all processes in the group. Then concatenates the received
        tensors from all processes in the group and returns a single tensor.

        Args:
            output_tensor: Output tensor.
            input_tensor: Input tensor to scatter.
            output_split_sizes: Split sizes for the output tensor.
            input_split_sizes: Split sizes for the input tensor.
            timeout: Optional timeout for the operation.
        """
        opts = AllToAllOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.asyncOp = False
        work = self._pg.alltoall_base(
            output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts
        )
        return work

    def all_to_all(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        timeout: timedelta | None = None,
    ) -> None:
        """
        Each process scatters a list of input tensors to all processes in the group,
        then gathers the received tensors from all processes into a list of output tensors.

        Args:
            output_tensors: List of output tensors.
            input_tensors: List of input tensors to scatter.
            timeout: Optional timeout for the operation.
        """
        opts = AllToAllOptions()
        if timeout is not None:
            opts.timeout = timeout
        opts.asyncOp = False
        work = self._pg.alltoall(output_tensors, input_tensors, opts)
        return work
    

    def split_comm(
        self,
        ranks: list[int],
        timeout: timedelta | None = None,
        opts: object | None = None,
        name: str | None = None,
    ) -> "Communicator | None":
        """
        Creates a new communicator from a subset of ranks.

        Args:
            ranks: List of ranks to include in the new communicator.
            timeout: Optional timeout for operations on the new communicator.
            group_name: Optional name for the new communicator.

        Returns:
            A new communicator containing the specified ranks, or None if this rank is not part of the group.
        """

        # Convert list of lists to a list of vectors of ints
        group = self._pg.split_group(
            ranks=ranks, 
            timeout=timeout,
            opts=opts,
            group_name=name,
        )
        if group is None:
            return None
        return Communicator(_pg=group)

    def merge_remote_comm(
        self,
        store: Store,
        size: int,
        timeout: timedelta,
        name: str,
    ) -> "Communicator":
        """
        Creates a new communicator by bootstrapping using the store.

        Args:
            store: The store to use for the new communicator.
            size: The world size of the new communicator.
            timeout: The timeout for operations on the new communicator.
            name: The name of the new communicator.
        """

        group = self._pg.merge_remote_group(
            store=store,
            size=size,
            timeout=timeout,
            group_name=name,
        )
        return Communicator(_pg=group)


class CommunicatorFactory(Protocol):
    """Protocol for process group factories."""

    def __call__(
        self,
        store: Store,
        rank: int,
        world_size: int,
        timeout: timedelta,
        device: torch.device,
        **kwargs: object,
    ) -> Communicator: ...


def register_backend(name: str, func: CommunicatorFactory) -> None:
    """
    Register a new process group backend.

    Args:
        name: The name of the backend.
        func: The function to create the process group.
    """
    if name in _BACKENDS:
        raise ValueError(f"Backend {name} already registered")

    _BACKENDS[name] = func


def _gloo_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device,
    **kwargs: object,
) -> Communicator:
    from torch.distributed import ProcessGroupGloo

    assert len(kwargs) == 0, "Gloo backend received unexpected kwargs"

    backend_class = ProcessGroupGloo(store, rank, world_size, timeout)
    backend_class._set_sequence_number_for_group()

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    # register devices
    pg._register_backend(device, ProcessGroup.BackendType.GLOO, backend_class)
    pg._register_backend(
        torch.device("cpu"), ProcessGroup.BackendType.GLOO, backend_class
    )
    if torch.cuda.is_available():
        pg._register_backend(
            torch.device("cuda"), ProcessGroup.BackendType.GLOO, backend_class
        )
    return Communicator(_pg=pg)


def _nccl_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device,
    **kwargs: object,
) -> Communicator:
    from torch.distributed import ProcessGroupNCCL

    opts = ProcessGroupNCCL.Options()
    opts._timeout = timeout
    for k, v in kwargs.items():
        if not hasattr(opts, k):
            raise KeyError(f"Unknown option {k}")
        setattr(opts, k, v)

    backend_class = ProcessGroupNCCL(store, rank, world_size, opts)
    backend_class._set_sequence_number_for_group()
    backend_class.eager_connect_single_device(device)

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.NCCL)
    pg._register_backend(device, ProcessGroup.BackendType.NCCL, backend_class)

    return Communicator(_pg=pg)


register_backend("gloo", _gloo_factory)
register_backend("nccl", _nccl_factory)


def new_comm(
    backend: str,
    timeout: timedelta,
    device: Union[str, torch.device],
    **kwargs: object,
) -> Communicator:
    """
    Create a new communicator with the given backend and options. This group is
    independent and will not be globally registered and thus not usable via the
    standard torch.distributed.* APIs.

    Args:
        backend: The backend to use for the process group.
        timeout: The timeout for collective operations.
        device: The device to use for the process group.
        **kwargs: All remaining arguments are passed to the backend constructor.
                  See the backend specific documentation for details.

    Returns:
        A new communicator.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Backend {backend} not registered")

    device = torch.device(device)

    store, rank, world_size = next(iter(rendezvous("env://")))
    store.set_timeout(timeout)

    return _BACKENDS[backend](store, rank, world_size, timeout, device, **kwargs)


_CURRENT_COMMUNICATOR: Communicator | None = None


def current_comm() -> Communicator:
    """
    Get the current process group. Thread local method.

    Returns:
        The current process group.
    """
    return _CURRENT_COMMUNICATOR


@contextmanager
def comm(comm: Communicator) -> Generator[None, None, None]:
    """
    Context manager for communicators. Thread local method.

    When entered, current_comm() will return the given communicator. When the
    context manager exits, the previous communicator will be restored.

    Args:
        pg: The process group to use.
    """
    global _CURRENT_COMMUNICATOR

    prev_pg = _current_process_group()
    prev_comm = _CURRENT_COMMUNICATOR

    pg = comm._pg
    _set_process_group(pg)
    _CURRENT_COMMUNICATOR = comm
    try:
        yield
    finally:
        _set_process_group(prev_pg)
        _CURRENT_COMMUNICATOR = prev_comm
