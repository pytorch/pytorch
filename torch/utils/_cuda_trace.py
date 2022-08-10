import traceback
import sys
from typing import Callable, List, Tuple, Any


class CallbackRegistry:
    def __init__(self):
        self.callback_list : List[Callable[[Tuple[Any, ...]], None]] = []

    def add_callback(self, cb : Callable[[Tuple[Any, ...]], None]) -> None:
        self.callback_list.append(cb)

    def fire_callbacks(self, *args: int) -> None:
        for cb in self.callback_list:
            try:
                cb(*args)
            except Exception:
                print(
                    f"Callback registered with CUDA trace threw an exception:\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr
                )


CUDAEventCreationCallbacks = CallbackRegistry()
CUDAEventDeletionCallbacks = CallbackRegistry()
CUDAEventRecordCallbacks = CallbackRegistry()
CUDAEventWaitCallbacks = CallbackRegistry()
CUDAMemoryAllocationCallbacks = CallbackRegistry()
CUDAMemoryDeallocationCallbacks = CallbackRegistry()
CUDAStreamAllocationCallbacks = CallbackRegistry()

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

def register_callback_for_cuda_stream_allocation(cb: Callable[[int], None]) -> None:
    CUDAStreamAllocationCallbacks.add_callback(cb)
