# mypy: allow-untyped-defs
"""
Centralized module for importing and re-exporting torch._C._distributed_c10d components.
This module provides fallback stubs when distributed components are not available.
"""

import sys
from datetime import timedelta
from enum import Enum
from typing import List

import torch
from torch.futures import Future


# Try to import from torch._C._distributed_c10d, providing fallbacks when not available
try:
    from torch._C._distributed_c10d import (  # Basic components; Additional distributed_c10d components
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _ControlCollectives,
        _current_process_group,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _DistributedBackendOptions,
        _make_nccl_premul_sum,
        _register_builtin_comm_hook,
        _register_comm_hook,
        _register_process_group,
        _resolve_process_group,
        _set_process_group,
        _StoreCollectives,
        _test_python_store,
        _unregister_all_process_groups,
        _unregister_process_group,
        _verify_params_across_processes,
        _WorkerServer,
        AllgatherOptions,
        AllreduceCoalescedOptions,
        AllreduceOptions,
        AllToAllOptions,
        Backend as _Backend,
        BarrierOptions,
        BroadcastOptions,
        BuiltinCommHookType,
        DebugLevel,
        FileStore,
        GatherOptions,
        get_debug_level,
        GradBucket,
        Logger,
        PrefixStore,
        ProcessGroup,
        ReduceOp,
        ReduceOptions,
        Reducer,
        ReduceScatterOptions,
        ScatterOptions,
        set_debug_level,
        set_debug_level_from_env,
        Store,
        TCPStore,
        Work as _Work,
    )

    # Try to import NCCL-specific components
    try:
        from torch._C._distributed_c10d import _DEFAULT_PG_NCCL_TIMEOUT
    except ImportError:
        _DEFAULT_PG_NCCL_TIMEOUT = None

    # Try to import optional components that may not be available in all builds
    try:
        from torch._C._distributed_c10d import FakeProcessGroup
    except ImportError:
        FakeProcessGroup = None

    # Import platform-specific components
    if sys.platform != "win32":
        from torch._C._distributed_c10d import HashStore
    else:
        # Provide HashStore stub for Windows
        class HashStore(Store):
            def __init__(self, *args, **kwargs):
                super().__init__()

    # Import graph input collective functions and symmetric memory components
    from torch._C._distributed_c10d import (
        _allow_inflight_collective_as_graph_input,
        _register_work,
        _set_allow_inflight_collective_as_graph_input,
    )

    # Try to import NVSHMEM and symmetric memory components
    try:
        from torch._C._distributed_c10d import (
            _is_nvshmem_available,
            _nvshmemx_cumodule_init,
            _SymmetricMemory,
        )
    except ImportError:
        # Provide fallback stubs if not available
        def _is_nvshmem_available():
            return False

        class _SymmetricMemory:
            def __init__(self, *args, **kwargs):
                pass

            @classmethod
            def empty_strided_p2p(cls, size, stride, dtype, device, group_name=None):
                return torch.empty(size, dtype=dtype, device=device)

            @classmethod
            def rendezvous(cls, tensor, group_name=None):
                return None

            @classmethod
            def set_group_info(cls, *args, **kwargs):
                pass

            @classmethod
            def set_backend(cls, name):
                pass

            @classmethod
            def get_backend(cls, device):
                return None

            @classmethod
            def has_multicast_support(cls, device_type, device_index):
                return False

    # All imports successful, distributed components available
    HAS_DISTRIBUTED = True

except (ImportError, AttributeError):
    # Fallback mode: provide Python stubs for missing C++ components
    HAS_DISTRIBUTED = False

    # Constants
    _DEFAULT_FIRST_BUCKET_BYTES = 1024 * 1024
    _DEFAULT_PG_TIMEOUT = timedelta(seconds=30 * 60)
    _DEFAULT_PG_NCCL_TIMEOUT = None

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

        def check(self, keys: List[str]) -> bool:
            return all(key in self._data for key in keys)

        def compare_set(
            self, key: str, expected_value: str, desired_value: str
        ) -> bytes:
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

        def wait(self, keys: List[str], timeout=None):
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
            world_size: int = None,
            is_master: bool = False,
            timeout: timedelta = None,
            wait_for_workers: bool = False,
            multi_tenant: bool = False,
            master_listen_fd: int = None,
            use_libuv: bool = None,
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

    class _Work:
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
            future = Future()
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

    class FakeProcessGroup:
        def __init__(self, rank: int = 0, world_size: int = 1):
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

        def __init__(self, store: Store = None, rank: int = 0, size: int = 1):
            self._store = store or Store()
            self._rank = rank
            self._size = size
            self.group_name = f"stub_group_{id(self)}"

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
            work = _Work()
            if isinstance(tensor_or_tensors, torch.Tensor):
                work._result = [tensor_or_tensors]
            else:
                work._result = tensor_or_tensors
            return work

        def allreduce(self, tensor_or_tensors, op=None, timeout=None, **kwargs):
            work = _Work()
            if isinstance(tensor_or_tensors, torch.Tensor):
                work._result = [tensor_or_tensors]
            else:
                work._result = tensor_or_tensors
            return work

        def barrier(self, timeout=None, **kwargs):
            return _Work()

    class _Backend:
        def __init__(self, rank: int, size: int):
            self._rank = rank
            self._size = size

    class Backend(str):
        """Mock Backend class for non-distributed builds."""

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
    def _broadcast_coalesced(*args, **kwargs):
        pass

    def _compute_bucket_assignment_by_size(
        tensors, bucket_size_limits, expect_sparse_gradient=None, tensor_indices=None
    ):
        if tensor_indices is None:
            tensor_indices = list(range(len(tensors)))
        return [tensor_indices], [sum(t.numel() * t.element_size() for t in tensors)]

    def _make_nccl_premul_sum(factor):
        pass

    def _register_builtin_comm_hook(*args, **kwargs):
        pass

    def _register_comm_hook(*args, **kwargs):
        pass

    def _test_python_store(store):
        pass

    def _allow_inflight_collective_as_graph_input():
        """Mock function that returns False to indicate no inflight collectives are allowed."""
        return False

    def _set_allow_inflight_collective_as_graph_input(value: bool):
        """Mock function that does nothing in non-distributed builds."""

    def _register_work(tensor, work):
        """Mock function to register work with tensor - does nothing in non-distributed builds."""

    def _is_nvshmem_available():
        """Mock function that returns False indicating NVSHMEM is not available."""
        return False

    def _nvshmemx_cumodule_init(module):
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

    def _verify_params_across_processes(*args, **kwargs):
        pass

    # Additional distributed_c10d function stubs
    def _register_process_group(name, group):
        pass

    def _resolve_process_group(name):
        return ProcessGroup()

    def _unregister_all_process_groups():
        pass

    def _unregister_process_group(name):
        pass

    def _current_process_group():
        """Mock function that returns the default process group."""
        return ProcessGroup()

    def _set_process_group(group):
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


# Provide backwards compatibility by making all symbols available at module level
__all__ = [
    # Basic components
    "_broadcast_coalesced",
    "_compute_bucket_assignment_by_size",
    "_ControlCollectives",
    "_DEFAULT_FIRST_BUCKET_BYTES",
    "_DEFAULT_PG_TIMEOUT",
    "_DEFAULT_PG_NCCL_TIMEOUT",
    "_make_nccl_premul_sum",
    "_register_builtin_comm_hook",
    "_register_comm_hook",
    "_StoreCollectives",
    "_test_python_store",
    "_verify_params_across_processes",
    "_allow_inflight_collective_as_graph_input",
    "_register_work",
    "_set_allow_inflight_collective_as_graph_input",
    "_is_nvshmem_available",
    "_nvshmemx_cumodule_init",
    "_SymmetricMemory",
    "_Backend",
    "Backend",
    "BuiltinCommHookType",
    "DebugLevel",
    "FakeProcessGroup",
    "FileStore",
    "get_debug_level",
    "GradBucket",
    "HashStore",
    "Logger",
    "PrefixStore",
    "ProcessGroup",
    "Reducer",
    "ReduceOp",
    "set_debug_level",
    "set_debug_level_from_env",
    "Store",
    "TCPStore",
    "_Work",
    # Additional distributed_c10d components
    "_DistributedBackendOptions",
    "_register_process_group",
    "_resolve_process_group",
    "_unregister_all_process_groups",
    "_unregister_process_group",
    "_current_process_group",
    "_set_process_group",
    "_WorkerServer",
    "AllgatherOptions",
    "AllreduceCoalescedOptions",
    "AllreduceOptions",
    "AllToAllOptions",
    "BarrierOptions",
    "BroadcastOptions",
    "GatherOptions",
    "ReduceOptions",
    "ReduceScatterOptions",
    "ScatterOptions",
    # Control flag
    "HAS_DISTRIBUTED",
]
