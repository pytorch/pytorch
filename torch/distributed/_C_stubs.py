# mypy: allow-untyped-defs
"""
Python stubs for distributed components when distributed is not available.

This module provides fallback implementations for all distributed components
that would normally be provided by the C++ extension.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import Optional


# Constants
_DEFAULT_FIRST_BUCKET_BYTES = 1024 * 1024
_DEFAULT_PG_TIMEOUT: timedelta = timedelta(seconds=30 * 60)
_DEFAULT_PG_NCCL_TIMEOUT: Optional[timedelta] = None


class DebugLevel(Enum):
    OFF = "off"
    INFO = "info"
    DETAIL = "detail"


def get_debug_level():
    return DebugLevel.OFF


def set_debug_level():
    pass


def set_debug_level_from_env():
    pass


class BuiltinCommHookType(Enum):
    ALLREDUCE = "allreduce"
    FP16_COMPRESS = "fp16_compress"


class ReduceOp:
    class RedOpType(Enum):
        SUM = "sum"
        AVG = "avg"
        PRODUCT = "product"
        MIN = "min"
        MAX = "max"
        BAND = "band"
        BOR = "bor"
        BXOR = "bxor"
        PREMUL_SUM = "premul_sum"
        UNUSED = "unused"

    def __init__(self, op):
        self.op = op

    SUM = RedOpType.SUM
    AVG = RedOpType.AVG
    PRODUCT = RedOpType.PRODUCT
    MIN = RedOpType.MIN
    MAX = RedOpType.MAX
    BAND = RedOpType.BAND
    BOR = RedOpType.BOR
    BXOR = RedOpType.BXOR
    PREMUL_SUM = RedOpType.PREMUL_SUM
    UNUSED = RedOpType.UNUSED


class Store:
    def __init__(self):
        self._data = {}

    def set(self, key: str, value: str):
        self._data[key] = value

    def get(self, key: str) -> bytes:
        return self._data.get(key, "").encode()

    def add(self, key: str, value: int) -> int:
        current = int(self._data.get(key, "0"))
        current += value
        self._data[key] = str(current)
        return current

    def check(self, keys: list[str]) -> bool:
        return all(key in self._data for key in keys)

    def compare_set(self, key: str, expected_value: str, desired_value: str) -> bytes:
        current = self._data.get(key, "")
        if current == expected_value:
            self._data[key] = desired_value
            return desired_value.encode()
        return current.encode()

    def delete_key(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def num_keys(self) -> int:
        return len(self._data)

    def set_timeout(self, timeout: timedelta):
        pass

    def wait(self, keys: list[str], timeout=None):
        pass


class FileStore(Store):
    def __init__(self, path: str, numWorkers: int = 1):
        super().__init__()
        self.path = path
        self.numWorkers = numWorkers


class TCPStore(Store):
    def __init__(
        self,
        host_name: str,
        port: int,
        world_size: Optional[int] = None,
        is_master: bool = False,
        timeout: Optional[timedelta] = None,
        wait_for_workers: bool = False,
        multi_tenant: bool = False,
        master_listen_fd: Optional[int] = None,
        use_libuv: Optional[bool] = None,
    ):
        super().__init__()
        self.host_name = host_name
        self.port = port


class HashStore(Store):
    def __init__(self, *args, **kwargs):
        super().__init__()


class PrefixStore(Store):
    def __init__(self, prefix: str, store: Store):
        super().__init__()
        self.prefix = prefix
        self.store = store

    @property
    def underlying_store(self):
        return self.store


class Work:
    def __init__(self):
        self._completed = True
        self._success = True
        self._exception = None
        self._result = []

    def is_completed(self):
        return self._completed

    def is_success(self):
        return self._success

    def exception(self):
        return self._exception

    def wait(self, timeout=None):
        return True

    def get_future(self):
        future: Future = Future()
        future.set_result(self._result)
        return future

    def source_rank(self):
        return 0

    def _source_rank(self):
        return 0

    def result(self):
        return self._result

    def synchronize(self):
        pass


class FakeWork(Work):
    def __init__(self):
        super().__init__()
        self.seq_id = 0


class FakeProcessGroup:
    def __init__(self, rank: int = 0, world_size: int = 1, backend_opts=None):
        self._rank = rank
        self._world_size = world_size

    def rank(self):
        return self._rank

    def size(self):
        return self._world_size


class ProcessGroup:
    class BackendType(Enum):
        UNDEFINED = "undefined"
        GLOO = "gloo"
        NCCL = "nccl"
        UCC = "ucc"
        MPI = "mpi"
        XCCL = "xccl"
        CUSTOM = "custom"

    def __init__(self, store: Optional[Store] = None, rank: int = 0, size: int = 1):
        self._store = store or Store()
        self._rank = rank
        self._size = size
        self.group_name = f"stub_group_{id(self)}"
        self.bound_device_id = None

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def get_group_store(self):
        return self._store

    def abort(self):
        pass

    def set_timeout(self, timeout: timedelta):
        pass

    def shutdown(self):
        pass

    def broadcast(self, tensor_or_tensors, root=0, timeout=None, **kwargs):
        work = Work()
        if isinstance(tensor_or_tensors, torch.Tensor):
            work._result = [tensor_or_tensors]  # type: ignore[attr-defined]
        else:
            work._result = tensor_or_tensors  # type: ignore[attr-defined]
        return work

    def allreduce(self, tensor_or_tensors, op=None, timeout=None, **kwargs):
        work = Work()
        if isinstance(tensor_or_tensors, torch.Tensor):
            work._result = [tensor_or_tensors]  # type: ignore[attr-defined]
        else:
            work._result = tensor_or_tensors  # type: ignore[attr-defined]
        return work

    def barrier(self, timeout=None, **kwargs):
        return Work()

    def _set_default_backend(self, backend_type):
        """Mock _set_default_backend method that does nothing in stub implementation."""

    def _register_backend(self, device, backend_type, backend_class):
        """Mock _register_backend method that does nothing in stub implementation."""

    def _set_group_name(self, group_name):
        """Mock _set_group_name method that does nothing in stub implementation."""
        self.group_name = group_name

    def _set_group_desc(self, group_desc):
        """Mock _set_group_desc method that does nothing in stub implementation."""
        self.group_desc = group_desc


# Specific process group implementations
class ProcessGroupMPI(ProcessGroup):
    """Mock ProcessGroupMPI for non-distributed builds."""


class ProcessGroupNCCL(ProcessGroup):
    """Mock ProcessGroupNCCL for non-distributed builds."""


class ProcessGroupGloo(ProcessGroup):
    """Mock ProcessGroupGloo for non-distributed builds."""


class ProcessGroupUCC(ProcessGroup):
    """Mock ProcessGroupUCC for non-distributed builds."""


class ProcessGroupXCCL(ProcessGroup):
    """Mock ProcessGroupXCCL for non-distributed builds."""


class _ProcessGroupWrapper:
    """Mock _ProcessGroupWrapper for non-distributed builds."""

    def __init__(self, process_group, *args, **kwargs):
        self._process_group = process_group

    def __getattr__(self, name):
        return getattr(self._process_group, name)


class Backend(str):
    """Mock Backend class for non-distributed builds."""

    __slots__ = ()

    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    UCC = "ucc"
    MPI = "mpi"
    XCCL = "xccl"

    class Options:
        pass

    @classmethod
    def register_backend(cls, name, func, extended_api=False, devices=None):
        """Mock register_backend method."""


class GradBucket:
    def index(self):
        return 0

    def buffer(self):
        return torch.tensor([])

    def gradients(self):
        return []

    def is_last(self):
        return True


class Reducer:
    def __init__(self, *args, **kwargs):
        pass


class Logger:
    def __init__(self, *args, **kwargs):
        pass


class _ControlCollectives:
    pass


class _StoreCollectives:
    def __init__(self, store: Store, rank: int, world_size: int):
        pass


# Function stubs
def _broadcast_coalesced(
    process_group: ProcessGroup,
    tensors: list[torch.Tensor],
    buffer_size: int,
    src: int,
):
    pass


def _compute_bucket_assignment_by_size(
    tensors: list[torch.Tensor],
    bucket_size_limits: list[int],
    expect_sparse_gradient: Optional[list[bool]] = None,
    tensor_indices: Optional[list[int]] = None,
) -> tuple[list[list[int]], list[int]]:
    if tensor_indices is None:
        tensor_indices = list(range(len(tensors)))
    return [tensor_indices], [sum(t.numel() * t.element_size() for t in tensors)]


def _make_nccl_premul_sum(factor) -> ReduceOp:
    # Return a dummy ReduceOp instance
    return ReduceOp(ReduceOp.RedOpType.SUM)  # type: ignore[attr-defined]


def _register_builtin_comm_hook(
    reducer: Reducer,
    comm_hook_type: BuiltinCommHookType,
):
    pass


def _register_comm_hook(reducer: Reducer, state, comm_hook) -> None:
    pass


def _test_python_store(store: Store) -> None:
    pass


def _allow_inflight_collective_as_graph_input() -> bool:
    """Mock function that returns False to indicate no inflight collectives are allowed."""
    return False


def _set_allow_inflight_collective_as_graph_input(value: bool) -> None:
    """Mock function that does nothing in non-distributed builds."""


def _register_work(tensor: torch.Tensor, work: Work) -> ProcessGroup:
    """Mock function to register work with tensor - does nothing in non-distributed builds."""
    return ProcessGroup(store=Store(), rank=0, size=1)


def _set_global_rank(rank):
    """Mock function to set global rank - does nothing in non-distributed builds."""


def _hash_tensors(tensors):
    """Mock function to hash tensors - returns dummy hash in non-distributed builds."""
    return 0


def _is_nvshmem_available() -> bool:
    """Mock function that returns False indicating NVSHMEM is not available."""
    return False


def _nvshmemx_cumodule_init(module: int) -> None:
    """Mock function for NVSHMEM CU module initialization - does nothing in non-distributed builds."""


class _SymmetricMemory:
    """Mock _SymmetricMemory class for non-distributed builds."""

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def empty_strided_p2p(cls, size, stride, dtype, device, group_name=None):
        """Mock empty_strided_p2p that returns a regular tensor."""
        return torch.empty(size, dtype=dtype, device=device)

    @classmethod
    def rendezvous(cls, tensor, group_name=None):
        """Mock rendezvous that returns None."""
        return None

    @classmethod
    def set_group_info(cls, *args, **kwargs):
        """Mock set_group_info that does nothing."""

    @classmethod
    def set_backend(cls, name):
        """Mock set_backend that does nothing."""

    @classmethod
    def get_backend(cls, device):
        """Mock get_backend that returns None."""
        return None

    @classmethod
    def has_multicast_support(cls, device_type, device_index):
        """Mock has_multicast_support that returns False."""
        return False


def _verify_params_across_processes(
    process_group: ProcessGroup,
    params: list[torch.Tensor],
    logger: Logger | None,
):
    pass


# Additional distributed_c10d function stubs
def _register_process_group(group_name: str, process_group: ProcessGroup) -> None:
    pass


def _resolve_process_group(group_name: str) -> ProcessGroup:
    return ProcessGroup(store=Store(), rank=0, size=1)


def _unregister_all_process_groups() -> None:
    pass


def _unregister_process_group(group_name: str) -> None:
    pass


def _current_process_group() -> ProcessGroup:
    """Mock function that returns the default process group."""
    return ProcessGroup(store=Store(), rank=0, size=1)


def _set_process_group(pg: ProcessGroup) -> None:
    """Mock function that does nothing in non-distributed builds."""


class _WorkerServer:
    """Mock _WorkerServer for non-distributed builds."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path

    def shutdown(self):
        """Mock shutdown method that does nothing."""


# Option classes stubs
class _DistributedBackendOptions:
    def __init__(self):
        pass


class AllgatherOptions:
    def __init__(self):
        self.asyncOp = True
        self.timeout = _DEFAULT_PG_TIMEOUT


class AllreduceCoalescedOptions:
    def __init__(self):
        self.timeout = _DEFAULT_PG_TIMEOUT


class AllreduceOptions:
    def __init__(self):
        self.reduceOp = ReduceOp.SUM
        self.timeout = _DEFAULT_PG_TIMEOUT


class AllToAllOptions:
    def __init__(self):
        self.timeout = _DEFAULT_PG_TIMEOUT


class BarrierOptions:
    def __init__(self):
        self.timeout = _DEFAULT_PG_TIMEOUT


class BroadcastOptions:
    def __init__(self):
        self.rootRank = 0
        self.rootTensor = 0
        self.timeout = _DEFAULT_PG_TIMEOUT


class GatherOptions:
    def __init__(self):
        self.rootRank = 0
        self.timeout = _DEFAULT_PG_TIMEOUT


class ReduceOptions:
    def __init__(self):
        self.reduceOp = ReduceOp.SUM
        self.rootRank = 0
        self.rootTensor = 0
        self.timeout = _DEFAULT_PG_TIMEOUT


class ReduceScatterOptions:
    def __init__(self):
        self.reduceOp = ReduceOp.SUM
        self.timeout = _DEFAULT_PG_TIMEOUT


class ScatterOptions:
    def __init__(self):
        self.rootRank = 0
        self.timeout = _DEFAULT_PG_TIMEOUT


def _dump_nccl_trace_json(
    includeCollectives: Optional[bool] = None, onlyActive: Optional[bool] = None
) -> bytes:
    """Mock function that returns empty JSON trace in non-distributed builds.

    Arguments:
        includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
        onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
    Returns:
        Stringified json work traces.
        Default settings return everything - i.e. contains NCCL comm dumps and collective traces.
    """
    return b"{}"


def _dump_nccl_trace(
    includeCollectives: Optional[bool] = None,
    includeStackTraces: Optional[bool] = None,
    onlyActive: Optional[bool] = None,
) -> bytes:
    """Mock function that returns empty pickle trace in non-distributed builds.

    Arguments:
        includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
        includeStackTraces(bool, optional): Whether to include stacktraces in the collective work traces. Default is True.
        onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
    Returns:
        Stringified pickle work traces.
        Default settings return everything - i.e. contains NCCL comm dumps and collective traces.
    """
    return b""


import torch
from torch.futures import Future
