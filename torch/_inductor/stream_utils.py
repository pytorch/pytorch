from __future__ import annotations

import functools
import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

import torch

__all__ = [
    "DEFAULT_STREAM",
    "DEFAULT_STREAM_IDX",
    "ENTRANCE_EVENT",
    "EVENT_NAME_TEMPLATE",
    "STREAM_NAME_TEMPLATE",
    "CUDAStreamPool",
    "get_cuda_stream_pool",
]

DEFAULT_STREAM: str = "default_stream"
DEFAULT_STREAM_IDX: int = 0
ENTRANCE_EVENT: str = "event0"
EVENT_NAME_TEMPLATE: str = "event{event_idx:d}"
STREAM_NAME_TEMPLATE: str = "stream{stream_idx:d}"


@functools.lru_cache
def get_stream_name(stream_idx: int) -> str:
    """Generate CUDA Stream name from stream index number.

    Args:
        stream_idx: Non-negative index number. 0 refers to the default stream, others refer to side
            streams.
    """
    if stream_idx == 0:
        return DEFAULT_STREAM
    else:
        return STREAM_NAME_TEMPLATE.format(stream_idx=stream_idx)


class CUDAStreamPool:
    """A pool managing reusable CUDA streams to optimize GPU operations.

    Attributes:
        pool_size (int): The maximum number of CUDA streams managed by the pool.
        stream_queue (queue.Queue): Queue holding the available CUDA streams.
    """

    def __init__(self, device: int | None = None, pool_size: int = 8) -> None:
        """Initializesthe CUDAStreamPool instance.

        Args:
            device (Optional[int], optional): The CUDA device ID.
                Defaults to None (current device).
            pool_size (int, optional): The maximum number of CUDA streams in the pool.
                Defaults to 8.
        """
        self.pool_size: int = pool_size
        self.stream_queue: queue.Queue[torch.cuda.Stream] = queue.Queue(maxsize=pool_size)

        for _ in range(pool_size):
            stream = torch.cuda.Stream(device=device)
            self.stream_queue.put(stream)

    def acquire(self) -> torch.cuda.Stream:
        """Acquire a CUDA stream from the pool.

        Returns:
            torch.cuda.Stream: A CUDA stream object from the pool.
        """
        return self.stream_queue.get()

    def release(self, stream: torch.cuda.Stream | None) -> None:
        """Return a CUDA stream back to the pool.

        Args:
            stream (Optional[torch.cuda.Stream]): The CUDA stream to return to the pool.
        """
        if stream is not None:
            self.stream_queue.put(stream)

    def __enter__(self) -> torch.cuda.Stream:
        """Enters the runtime context and acquires a CUDA stream.

        Returns:
            torch.cuda.Stream: The acquired CUDA stream.
        """
        self.stream = self.acquire()
        self.stream.__enter__()
        return self.stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context and releases the acquired CUDA stream.

        Args:
            exc_type (type[BaseException] | None): Exception type, if raised.
            exc_val (BaseException | None): Exception instance, if raised.
            exc_tb (TracebackType | None): Traceback object, if raised.
        """
        self.stream.__exit__(exc_type, exc_val, exc_tb)
        self.release(self.stream)


_cuda_stream_pool: CUDAStreamPool | None = None


def get_cuda_stream_pool(device: int | None = None, pool_size: int = 32) -> CUDAStreamPool:
    """Retrieve a global CUDA stream pool, creating it if necessary.

    This function ensures that only one CUDAStreamPool instance exists globally.

    Args:
        device (Optional[int], optional): The CUDA device ID to initialize the pool on.
            Defaults to None (current device).
        pool_size (int, optional): The number of streams in the pool. Defaults to 32.

    Returns:
        CUDAStreamPool: The global CUDA stream pool instance.
    """
    global _cuda_stream_pool
    if _cuda_stream_pool is None:
        _cuda_stream_pool = CUDAStreamPool(device=device, pool_size=pool_size)
    return _cuda_stream_pool