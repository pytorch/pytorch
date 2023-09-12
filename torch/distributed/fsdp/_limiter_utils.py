from collections import OrderedDict
from typing import Optional, Tuple

import torch


class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(
        self,
        max_num_inflight_all_gathers: int,
    ) -> None:
        self._queue: OrderedDict[torch.Tensor, torch.cuda.Event] = OrderedDict()
        self._max_num_inflight_all_gathers = max_num_inflight_all_gathers

    def enqueue(
        self,
        tensor: torch.Tensor,
        event: torch.cuda.Event,
    ) -> None:
        """Enqueues a free event."""
        # Delete and re-insert to maintain event order
        if tensor in self._queue:
            self._queue.pop(tensor)

        self._queue[tensor] = event

    def dequeue_if_needed(self) -> Optional[Tuple[torch.Tensor, torch.cuda.Event]]:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self.dequeue()
        return None

    def dequeue(self) -> Optional[Tuple[torch.Tensor, torch.cuda.Event]]:
        """Dequeues a free event if possible."""
        if self._queue:
            # Set `last` to False to pop item in FIFO style
            return self._queue.popitem(last=False)
        return None

    def pop(self, tensor: torch.Tensor) -> Optional[torch.cuda.Event]:
        """Erase tensor-event pair from queue"""
        return self._queue.pop(tensor, None)
