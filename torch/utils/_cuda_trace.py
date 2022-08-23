import logging
from typing import Callable, Generic, List

from typing_extensions import ParamSpec

_CUDA_HOOK_TRACE_OFFSET = -2

logger = logging.getLogger(__name__)
P = ParamSpec("P")


class CallbackRegistry(Generic[P]):
    def __init__(self, name: str):
        self.name = name
        self.callback_list: List[Callable[P, None]] = []

    def add_callback(self, cb: Callable[P, None]) -> None:
        self.callback_list.append(cb)

    def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None:
        for cb in self.callback_list:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Exception in callback for {self.name} registered with CUDA trace"
                )


CUDAEventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event creation"
)
CUDAEventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event deletion"
)
CUDAEventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "CUDA event record"
)
CUDAEventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "CUDA event wait"
)
CUDAMemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory allocation"
)
CUDAMemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory deallocation"
)
CUDAStreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream creation"
)


def register_callback_for_cuda_event_creation(cb: Callable[[int], None]) -> None:
    CUDAEventCreationCallbacks.add_callback(cb)


def register_callback_for_cuda_event_deletion(cb: Callable[[int], None]) -> None:
    CUDAEventDeletionCallbacks.add_callback(cb)


def register_callback_for_cuda_event_record(cb: Callable[[int, int], None]) -> None:
    CUDAEventRecordCallbacks.add_callback(cb)


def register_callback_for_cuda_event_wait(cb: Callable[[int, int], None]) -> None:
    CUDAEventWaitCallbacks.add_callback(cb)


def register_callback_for_cuda_memory_allocation(cb: Callable[[int], None]) -> None:
    CUDAMemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_cuda_memory_deallocation(cb: Callable[[int], None]) -> None:
    CUDAMemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_cuda_stream_creation(cb: Callable[[int], None]) -> None:
    CUDAStreamCreationCallbacks.add_callback(cb)
