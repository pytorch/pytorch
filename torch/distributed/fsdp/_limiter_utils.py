import collections
from typing import Deque, Optional

import torch


class EventWithTensor:
    def __init__(
        self,
        event: torch.cuda.Event,
        tensor: torch.Tensor,
    ) -> None:
        self.event = event
        self.tensor = tensor


class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        self._queue: Deque[EventWithTensor] = collections.deque()
        self._max_num_inflight_all_gathers = 2  # empirically chosen

    def enqueue(self, free_event: EventWithTensor) -> None:
        """Enqueues a free event."""
        self._queue.append(free_event)

    def dequeue_if_needed(self) -> Optional[EventWithTensor]:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self.dequeue()
        return None

    def dequeue(self) -> Optional[EventWithTensor]:
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()
            return event
        return None

    def clear(self) -> None:
        """Clear queue."""
        self._queue.clear()
