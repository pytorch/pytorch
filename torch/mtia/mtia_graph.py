# pylint: disable=useless-parent-delegation
from __future__ import annotations

from typing_extensions import Self

import torch


_POOL_HANDLE = tuple[int, int]


def graph_pool_handle() -> _POOL_HANDLE:
    """
    Return an opaque token representing the id of a graph memory pool.
    """
    # pyrefly: ignore [missing-attribute]
    return torch._C._mtia_graphPoolHandle()


class MTIAGraph(torch._C._MTIAGraph):
    """
    Wrapper around a MTIA graph.
    """

    def __new__(cls, keep_graph: bool = False) -> Self:
        return super().__new__(cls, keep_graph)

    def capture_begin(self, pool: _POOL_HANDLE) -> None:
        """
        Begin capturing a MTIA graph.
        """
        super().capture_begin(pool)

    def capture_end(self) -> None:
        """
        End the capture of a MTIA graph.
        """
        super().capture_end()

    def instantiate(self) -> None:
        """
        Instantiate the captured MTIA graph.
        """
        super().instantiate()

    def replay(self) -> None:
        """
        Replay the captured MTIA graph.
        """
        super().replay()

    def reset(self) -> None:
        """
        Destroy the captured graph and reset the states.
        """
        super().reset()

    def pool(self) -> _POOL_HANDLE:
        """
        Return an opaque token representing the id of this graph's memory pool
        """
        return super().pool()


class graph:
    default_capture_stream: torch.mtia.Stream | None = None

    def __init__(
        self,
        mtia_graph: MTIAGraph,
        pool: _POOL_HANDLE | None = None,
        stream: torch.mtia.Stream | None = None,
    ):
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.mtia.current_stream()

        self.pool: tuple[()] | tuple[_POOL_HANDLE] = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        if self.capture_stream is None:
            raise AssertionError("capture_stream must not be None")
        self.stream_ctx = torch.mtia.stream(self.capture_stream)
        self.mtia_graph = mtia_graph

    def __enter__(self) -> None:
        torch.mtia.synchronize()
        torch.mtia.empty_cache()

        self.stream_ctx.__enter__()

        pool_arg = self.pool[0] if self.pool else (0, 0)
        self.mtia_graph.capture_begin(pool_arg)

    def __exit__(self, *args: object) -> None:
        self.mtia_graph.capture_end()
        self.stream_ctx.__exit__(*args)


__all__ = [
    "MTIAGraph",
    "graph",
    "graph_pool_handle",
]
