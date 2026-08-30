from __future__ import annotations

import contextlib
import enum
import threading
from typing import Any, cast, NewType, Protocol, runtime_checkable, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Generator

    from torch.cuda import _POOL_HANDLE


# Address of an allocation owned by the graph-tree memory pool.
DataPtr = NewType("DataPtr", int)
# Non-owning address of a StorageImpl. It must never be treated as a DataPtr.
StorageImplHandle = NewType("StorageImplHandle", int)
# Opaque allocator state returned by get_checkpoint_state.
AllocatorState = NewType("AllocatorState", object)


class GraphTreeCaptureMode(enum.Enum):
    POOL_INITIALIZATION = enum.auto()
    MODEL = enum.auto()


@runtime_checkable
class GraphTreeGraph(Protocol):
    def replay(self) -> None: ...


@runtime_checkable
class GraphTreeGraphInterface(Protocol):
    def create_graph(self) -> GraphTreeGraph: ...

    def create_pool(self) -> tuple[int, int]: ...

    def capture(
        self,
        graph: GraphTreeGraph,
        *,
        stream: torch.Stream,
        pool: tuple[int, int],
        mode: GraphTreeCaptureMode,
    ) -> contextlib.AbstractContextManager[None]: ...

    def warmup_setup_context(
        self, *, kernel_free: bool
    ) -> contextlib.AbstractContextManager[None]: ...

    def warmup_execution_context(
        self,
    ) -> contextlib.AbstractContextManager[None]: ...

    def capture_setup_context(self) -> contextlib.AbstractContextManager[None]: ...

    def capture_execution_context(
        self,
    ) -> contextlib.AbstractContextManager[None]: ...


@runtime_checkable
class GraphTreeAllocatorInterface(Protocol):
    def begin_allocate_to_pool(self, device: int, pool: tuple[int, int]) -> None: ...

    def end_allocate_to_pool(self, device: int, pool: tuple[int, int]) -> None: ...

    def release_pool(self, device: int, pool: tuple[int, int]) -> None: ...

    def get_checkpoint_state(
        self, device: int, pool: tuple[int, int]
    ) -> AllocatorState: ...

    def set_checkpoint_pool_state(
        self,
        device: int,
        state: AllocatorState,
        stale_storages: list[StorageImplHandle],
        storages_to_add_deleters: list[StorageImplHandle],
    ) -> None:
        """Restore checkpointed allocator state and update storage deleters.

        ``state`` must come from ``get_checkpoint_state`` for the same device and
        private pool.
        ``stale_storages`` are live StorageImpl objects whose allocations are
        freed by the restore, so their allocator deleters must be removed.
        ``storages_to_add_deleters`` are live StorageImpl objects for allocations
        recreated by the restore, so allocator deleters must be installed.
        All handles are non-owning and must remain valid for the duration of the call.
        """
        ...

    def raw_delete(self, ptr: DataPtr) -> None:
        """Free a live allocation by address without consulting a StorageImpl.

        No live StorageImpl may retain an allocator deleter for this allocation.
        The address is invalid after this call.
        """
        ...

    def construct_tensor_from_storage_and_metadata(
        self, metadata: dict[str, Any], storage: torch.types.Storage
    ) -> torch.Tensor: ...

    def has_standard_deleter(self, storage: StorageImplHandle) -> bool:
        """Return whether ``storage`` owns its allocation via the allocator deleter."""
        ...

    def free_and_remove_deleter(self, storage: StorageImplHandle) -> None:
        """Free ``storage``'s allocation and replace its allocator deleter with a no-op.

        ``storage`` must be live and ``has_standard_deleter(storage)`` must be true.
        The handle is non-owning and remains owned by the caller.
        """
        ...

    def check_pool_live_allocations(
        self,
        device: int,
        pool: tuple[int, int],
        allocations: set[DataPtr],  # noqa: set_linter
    ) -> bool:
        """Check that live allocations exactly match the supplied addresses."""
        ...

    def memory_snapshot(self) -> list[dict[str, Any]]:
        """Return allocator segments used by Graph Trees diagnostics.

        Each segment must provide ``segment_pool_id``, ``address``, and
        ``blocks``. Each block must provide ``size`` and ``state``, and may
        provide allocation ``frames``. Blocks must be ordered and contiguous
        from the segment's ``address``.
        """
        ...

    def is_history_enabled(self) -> bool: ...

    def record_memory_history(self, enabled: bool) -> None: ...


# Keep CUDA-only dependencies below lazy or call-time for third-party backends.
def clear_cublas_cache() -> None:
    torch._C._cuda_clearCublasWorkspaces()


@contextlib.contextmanager
def clear_cublas_manager() -> Generator[None, None, None]:
    clear_cublas_cache()
    try:
        yield
    finally:
        clear_cublas_cache()


@contextlib.contextmanager
def disable_conv_cache_emptying() -> Generator[None, None, None]:
    prev = torch._C._cuda_get_conv_benchmark_empty_cache()
    torch._C._cudnn_set_conv_benchmark_empty_cache(False)
    try:
        yield
    finally:
        torch._C._cudnn_set_conv_benchmark_empty_cache(prev)


class CUDAGraphTreeGraphInterface:
    def create_graph(self) -> GraphTreeGraph:
        return torch.cuda.CUDAGraph()

    def create_pool(self) -> tuple[int, int]:
        return torch.cuda.graph_pool_handle()

    @contextlib.contextmanager
    def capture(
        self,
        graph: GraphTreeGraph,
        *,
        stream: torch.Stream,
        pool: tuple[int, int],
        mode: GraphTreeCaptureMode,
    ) -> Generator[None, None, None]:
        with torch.cuda.graph(
            graph,  # type: ignore[arg-type]
            stream=stream,  # type: ignore[arg-type]
            pool=cast("_POOL_HANDLE", pool),
            capture_error_mode="thread_local",
        ):
            yield

    @contextlib.contextmanager
    def warmup_setup_context(self, *, kernel_free: bool) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if not kernel_free:
                stack.enter_context(disable_conv_cache_emptying())
            stack.enter_context(clear_cublas_manager())
            yield

    def warmup_execution_context(self) -> contextlib.AbstractContextManager[None]:
        from torch._higher_order_ops.cudagraph_conditional_nodes import (
            ControlFlowOpWarmupDispatchMode,
        )

        return ControlFlowOpWarmupDispatchMode()

    def capture_setup_context(self) -> contextlib.AbstractContextManager[None]:
        return clear_cublas_manager()

    def capture_execution_context(self) -> contextlib.AbstractContextManager[None]:
        from torch._higher_order_ops.cudagraph_conditional_nodes import (
            CUDAGraphCaptureControlFlowOpDispatchMode,
        )

        return CUDAGraphCaptureControlFlowOpDispatchMode()


class CUDAGraphTreeAllocatorInterface:
    def begin_allocate_to_pool(self, device: int, pool: tuple[int, int]) -> None:
        torch._C._cuda_beginAllocateCurrentThreadToPool(device, pool)

    def end_allocate_to_pool(self, device: int, pool: tuple[int, int]) -> None:
        torch._C._cuda_endAllocateToPool(device, pool)

    def release_pool(self, device: int, pool: tuple[int, int]) -> None:
        torch._C._cuda_releasePool(device, pool)

    def get_checkpoint_state(
        self, device: int, pool: tuple[int, int]
    ) -> AllocatorState:
        return cast(AllocatorState, torch._C._cuda_getCheckpointState(device, pool))

    def set_checkpoint_pool_state(
        self,
        device: int,
        state: AllocatorState,
        stale_storages: list[StorageImplHandle],
        storages_to_add_deleters: list[StorageImplHandle],
    ) -> None:
        torch._C._cuda_setCheckpointPoolState(
            device,
            state,
            cast(list[int], stale_storages),
            cast(list[int], storages_to_add_deleters),
        )

    def raw_delete(self, ptr: DataPtr) -> None:
        torch._C._cuda_cudaCachingAllocator_raw_delete(ptr)

    def construct_tensor_from_storage_and_metadata(
        self, metadata: dict[str, Any], storage: torch.types.Storage
    ) -> torch.Tensor:
        return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
            metadata, storage
        )

    def has_standard_deleter(self, storage: StorageImplHandle) -> bool:
        return torch._C._has_Standard_Deleter(storage)

    def free_and_remove_deleter(self, storage: StorageImplHandle) -> None:
        torch._C._free_And_Remove_DeleterFn(storage)

    def check_pool_live_allocations(
        self,
        device: int,
        pool: tuple[int, int],
        allocations: set[DataPtr],  # noqa: set_linter
    ) -> bool:
        return torch._C._cuda_checkPoolLiveAllocations(device, pool, allocations)

    def memory_snapshot(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], torch.cuda.memory_snapshot())

    def is_history_enabled(self) -> bool:
        return torch._C._cuda_isHistoryEnabled()

    def record_memory_history(self, enabled: bool) -> None:
        if enabled:
            torch.cuda.memory._record_memory_history()
        else:
            torch.cuda.memory._record_memory_history(None)


_registry_lock = threading.Lock()
_registered_device_type: str | None = None
_graph_interface: GraphTreeGraphInterface | None = None
_allocator_interface: GraphTreeAllocatorInterface | None = None
_initialized = False


def register_graph_tree_backend(
    device_type: str,
    graph_interface: GraphTreeGraphInterface,
    allocator_interface: GraphTreeAllocatorInterface,
) -> None:
    if not device_type:
        raise ValueError("Graph Trees backend device_type must not be empty")
    if not isinstance(graph_interface, GraphTreeGraphInterface):
        raise TypeError("Invalid GraphTreeGraphInterface implementation")
    if not isinstance(allocator_interface, GraphTreeAllocatorInterface):
        raise TypeError("Invalid GraphTreeAllocatorInterface implementation")

    global _registered_device_type, _graph_interface, _allocator_interface
    with _registry_lock:
        if _initialized:
            raise RuntimeError(
                f"Cannot register Graph Trees backend {device_type!r}: backend "
                f"{_registered_device_type!r} has already been initialized. "
                "Graph Trees supports one active accelerator backend per process."
            )
        if _graph_interface is not None:
            raise RuntimeError(
                f"Cannot register Graph Trees backend {device_type!r}: backend "
                f"{_registered_device_type!r} is already registered"
            )
        _registered_device_type = device_type
        _graph_interface = graph_interface
        _allocator_interface = allocator_interface


def is_graph_tree_backend_available() -> bool:
    """Return whether Graph Trees supports the current accelerator.

    CUDA uses the built-in lazy default. Other accelerators must explicitly
    register a backend before Graph Trees is selected.
    """
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        return False

    with _registry_lock:
        if _graph_interface is not None:
            return _registered_device_type == accelerator.type
        return accelerator.type == "cuda"


def _initialize_graph_tree_backend() -> None:
    global _registered_device_type, _graph_interface, _allocator_interface, _initialized
    if _initialized:
        return

    with _registry_lock:
        if _initialized:
            return
        if _graph_interface is None:
            device_type = "cuda"
            graph_interface = CUDAGraphTreeGraphInterface()
            allocator_interface = CUDAGraphTreeAllocatorInterface()
        else:
            device_type = _registered_device_type
            graph_interface = _graph_interface
            allocator_interface = _allocator_interface

        if device_type is None or allocator_interface is None:
            raise AssertionError("Graph Trees backend registration is incomplete")

        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None or accelerator.type != device_type:
            actual_device_type = None if accelerator is None else accelerator.type
            raise RuntimeError(
                f"Graph Trees backend {device_type!r} must match the "
                f"current PyTorch accelerator, got {actual_device_type!r}"
            )
        _registered_device_type = device_type
        _graph_interface = graph_interface
        _allocator_interface = allocator_interface
        _initialized = True


def get_graph_interface() -> GraphTreeGraphInterface:
    _initialize_graph_tree_backend()
    if _graph_interface is None:
        raise AssertionError("Graph Trees graph interface is not initialized")
    return _graph_interface


def get_allocator_interface() -> GraphTreeAllocatorInterface:
    _initialize_graph_tree_backend()
    if _allocator_interface is None:
        raise AssertionError("Graph Trees allocator interface is not initialized")
    return _allocator_interface


def get_device_type() -> str:
    _initialize_graph_tree_backend()
    if _registered_device_type is None:
        raise AssertionError("Graph Trees device type is not initialized")
    return _registered_device_type


__all__ = [
    "AllocatorState",
    "DataPtr",
    "GraphTreeAllocatorInterface",
    "GraphTreeCaptureMode",
    "GraphTreeGraph",
    "GraphTreeGraphInterface",
    "StorageImplHandle",
    "get_allocator_interface",
    "get_device_type",
    "get_graph_interface",
    "is_graph_tree_backend_available",
    "register_graph_tree_backend",
]
