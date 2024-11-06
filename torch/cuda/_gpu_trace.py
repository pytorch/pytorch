from typing import Callable

from torch._utils import CallbackRegistry


EventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event creation"
)
EventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event deletion"
)
EventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "CUDA event record"
)
EventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "CUDA event wait"
)
MemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory allocation"
)
MemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory deallocation"
)
StreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream creation"
)
DeviceSynchronizationCallbacks: "CallbackRegistry[[]]" = CallbackRegistry(
    "CUDA device synchronization"
)
StreamSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream synchronization"
)
EventSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event synchronization"
)


def register_callback_for_event_creation(cb: Callable[[int], None]) -> None:
    EventCreationCallbacks.add_callback(cb)


def register_callback_for_event_deletion(cb: Callable[[int], None]) -> None:
    EventDeletionCallbacks.add_callback(cb)


def register_callback_for_event_record(cb: Callable[[int, int], None]) -> None:
    EventRecordCallbacks.add_callback(cb)


def register_callback_for_event_wait(cb: Callable[[int, int], None]) -> None:
    EventWaitCallbacks.add_callback(cb)


def register_callback_for_memory_allocation(cb: Callable[[int], None]) -> None:
    MemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_memory_deallocation(cb: Callable[[int], None]) -> None:
    MemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_stream_creation(cb: Callable[[int], None]) -> None:
    StreamCreationCallbacks.add_callback(cb)


def register_callback_for_device_synchronization(cb: Callable[[], None]) -> None:
    DeviceSynchronizationCallbacks.add_callback(cb)


def register_callback_for_stream_synchronization(cb: Callable[[int], None]) -> None:
    StreamSynchronizationCallbacks.add_callback(cb)


def register_callback_for_event_synchronization(cb: Callable[[int], None]) -> None:
    EventSynchronizationCallbacks.add_callback(cb)
