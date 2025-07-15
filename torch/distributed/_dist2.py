"""
Experimental Object Oriented Distributed API - torch.distributed._dist2
=======================================================================

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
    ProcessGroup,
    ReduceOp,
    Store,
)
from torch.distributed.rendezvous import rendezvous


_BACKENDS: dict[str, "ProcessGroupFactory"] = {}

__all__ = [
    "ProcessGroup",
    "ReduceOp",
    "ProcessGroupFactory",
    "register_backend",
    "new_group",
    "current_process_group",
    "process_group",
]


class ProcessGroupFactory(Protocol):
    """Protocol for process group factories."""

    def __call__(
        self,
        store: Store,
        rank: int,
        world_size: int,
        timeout: timedelta,
        device: torch.device,
        pg_options: Backend.Options,
    ) -> ProcessGroup: ...


def register_backend(name: str, func: ProcessGroupFactory) -> None:
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
    pg_options: Backend.Options,
) -> ProcessGroup:
    from torch.distributed import ProcessGroupGloo

    assert pg_options is None, "Gloo backend does not support options"

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
    return pg


def _nccl_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device,
    pg_options: Backend.Options,
) -> ProcessGroup:
    from torch.distributed import ProcessGroupNCCL

    assert isinstance(pg_options, ProcessGroupNCCL.Options)

    pg_options._timeout = timeout

    backend_class = ProcessGroupNCCL(store, rank, world_size, pg_options)
    backend_class._set_sequence_number_for_group()
    backend_class.eager_connect_single_device(device)

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.NCCL)
    pg._register_backend(device, ProcessGroup.BackendType.NCCL, backend_class)

    return pg


register_backend("gloo", _gloo_factory)
register_backend("nccl", _nccl_factory)


def new_group(
    backend: str,
    timeout: timedelta,
    device: Union[str, torch.device],
    pg_options: Backend.Options,
) -> ProcessGroup:
    """
    Create a new process group with the given backend and options. This group is
    independent and will not be globally registered and thus not usable via the
    standard torch.distributed.* APIs.

    Args:
        backend: The backend to use for the process group.
        timeout: The timeout for collective operations.
        device: The device to use for the process group.
        pg_options: The options to use for the process group.

    Returns:
        A new process group.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Backend {backend} not registered")

    device = torch.device(device)

    store, rank, world_size = next(iter(rendezvous("env://")))
    store.set_timeout(timeout)

    return _BACKENDS[backend](store, rank, world_size, timeout, device, pg_options)


def current_process_group() -> ProcessGroup:
    """
    Get the current process group. Thread local method.

    Returns:
        The current process group.
    """
    return _current_process_group()


@contextmanager
def process_group(pg: ProcessGroup) -> Generator[None, None, None]:
    """
    Context manager for process groups. Thread local method.

    Args:
        pg: The process group to use.
    """
    prev_pg = current_process_group()

    _set_process_group(pg)
    try:
        yield
    finally:
        _set_process_group(prev_pg)
