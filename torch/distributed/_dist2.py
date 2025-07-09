"""
Experimental Object Oriented Distributed API - torch.distributed._dist2
=======================================================================

This is an experimental new API for PyTorch Distributed. This is actively in development and subject to change or deletion entirely.

This is intended as a proving ground for more flexible and object oriented distributed APIs.
"""

from datetime import timedelta
from typing import Optional, Protocol, Union

import torch
from torch._C._distributed_c10d import Backend, ProcessGroup, Store
from torch.distributed.distributed_c10d import _check_valid_timeout
from torch.distributed.rendezvous import rendezvous


_BACKENDS: dict[str, "ProcessGroupFactory"] = {}


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


def split_group(
    new_ranks: list[int],
    parent_pg: ProcessGroup,
    timeout: Optional[timedelta] = None,
    pg_options: Optional[Backend.Options] = None,
    group_desc: Optional[str] = None,
) -> Optional[ProcessGroup]:
    """
    This creates a new subgroup using the specified ranks. The current rank must be included in the list of new_ranks.

    TODO: add more documentation to the args/kwargs
    """
    if len(new_ranks) == 0:
        raise ValueError("the split group cannot be empty")
    if len(new_ranks) > parent_pg.size():
        raise ValueError(
            "the split group's size should be less or equal to the world_size set by init_process_group"
        )
    if len(new_ranks) != len(set(new_ranks)):
        raise ValueError("the split group cannot have duplicate ranks")
    new_ranks = sorted(new_ranks)

    parent_backend = parent_pg._get_backend(torch.device("cuda"))
    # set the group_desc before the color or no_cloor split
    group_desc = (
        f"{parent_pg.group_desc}:split:{parent_backend.comm_split_count()}"  # type: ignore[attr-defined]
        if group_desc is None
        else group_desc
    )
    # TODO: Need a better way to get the split group name
    group_name = f"{parent_pg.group_name}:split:{list(new_ranks)}"

    if pg_options is None:
        # default pg_options same as the parent process group
        pg_options = parent_backend.options

    # If not set we can reuse the timeout from parent process group.
    if timeout is None:
        timeout = pg_options._timeout
    _check_valid_timeout(timeout)
    pg_options._timeout = timeout
    split_backend = parent_backend.split_backend(new_ranks, pg_options, group_desc)

    if not split_backend:
        return None

    # We register the backend after initializing and timeout is set in pg_options.
    pg: ProcessGroup = ProcessGroup(
        split_backend.store(),
        split_backend.rank(),
        split_backend.size(),
    )
    backend_type = parent_pg.default_backend_type
    pg._set_default_backend(backend_type)
    pg._register_backend(torch.device("cuda"), backend_type, split_backend)
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    return pg
