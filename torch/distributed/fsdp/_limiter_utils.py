import collections
from enum import Enum, auto
from typing import Deque, Optional

import torch


class EventWithTensor:
    """
    This data structure pairs the resharding event (i.e. forward done) with the
    all-gather buffer being resharded. The purpose is to free the all-gather
    buffer when we ask a new all-gather to wait for a resharding event (hence
    avoiding OOM).
    """

    def __init__(
        self,
        event: torch.cuda.Event,
        tensor: torch.Tensor,
    ) -> None:
        self.event = event
        self.tensor = tensor


class _QueueDirection(Enum):
    POP_LEFT = auto()
    POP_RIGHT = auto()


class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        self._queue: Deque[EventWithTensor] = collections.deque()
        self._max_num_inflight_all_gathers = 2  # empirically chosen
        self._direction: _QueueDirection = _QueueDirection.POP_LEFT

    def enqueue(self, free_event: EventWithTensor) -> None:
        """Enqueues a free event."""
        if self._direction == _QueueDirection.POP_LEFT:
            self._queue.append(free_event)
        else:
            self._queue.appendleft(free_event)

    def dequeue_if_needed(self) -> Optional[EventWithTensor]:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self.dequeue()
        return None

    def dequeue(self) -> Optional[EventWithTensor]:
        """Dequeues a free event if possible."""
        if self._queue:
            if self._direction == _QueueDirection.POP_LEFT:
                event = self._queue.popleft()
            else:
                event = self._queue.pop()
            return event
        return None

    def use_backward_direction(self) -> None:
        self._direction = _QueueDirection.POP_RIGHT

    def use_forward_direction(self) -> None:
        self._direction = _QueueDirection.POP_LEFT
