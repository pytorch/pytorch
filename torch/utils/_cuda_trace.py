import traceback
import sys
from typing import Callable, List


cuda_event_creation_callbacks: List[Callable[[int], None]] = []

def register_callback_for_cuda_event_creation(cb: Callable[[int], None]) -> None:
    cuda_event_creation_callbacks.append(cb)


def _fire_callbacks_for_cuda_event_creation(event_id: int) -> None:
    for cb in cuda_event_creation_callbacks:
        try:
            cb(event_id)
        except Exception:
            print(f"Callback registered with CUDA trace for CUDA event creation threw an exception:\n{traceback.format_exc()}", file=sys.stderr)
