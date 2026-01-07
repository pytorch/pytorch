import gc
from typing import Literal
from typing_extensions import Self

import torch
from torch._C import _acceleratorGraph


class Graph(_acceleratorGraph):
    r"""
    Wrapper around an :ref:`accelerator<accelerators>` graph that supports capture and replay.

    A graph captures a sequence of operations and their dependencies, allowing them to be
    replayed efficiently with reduced overhead. This class can be used as a context manager
    to automatically capture operations on the current stream.

    Arguments:
        keep_graph (bool, optional): If ``False``, the underlying graph is destroyed and the
            executable graph is instantiated on the GPU at the end of ``capture_end``.
            If ``True``, the underlying graph is preserved after ``capture_end``. In this case,
            the executable graph is not instantiated automatically; it must be explicitly created
            by calling ``instantiate``, or it will be instantiated on the first call to ``replay``.
            Defaults to ``False``.
        pool (tuple[int, int], optional): Memory pool identifier for this graph. Multiple graphs
            can share the same pool by passing the same identifier, which can reduce memory overhead.
            Defaults to ``None``.

    Example::

        >>> # xdoctest: +SKIP
        >>> x = torch.zeros([2000], device=0)

        >>> stream = torch.Stream()
        >>> graph = torch.accelerator.Graph()
        >>> with stream, graph:
        ...     x += 1

        >>> graph.replay()
    """

    def __new__(
        cls, keep_graph: bool = False, *, pool: tuple[int, int] | None = None
    ) -> Self:
        return super().__new__(cls, keep_graph)

    def __init__(
        self, keep_graph: bool = False, *, pool: tuple[int, int] | None = None
    ) -> None:
        super().__init__(keep_graph)
        self.graph_pool = pool

    def capture_begin(
        self,
        capture_error_mode: Literal[
            "default", "global", "thread_local", "relaxed"
        ] = "default",
    ) -> None:
        r"""
        Begin graph capture on the current stream.

        All operations executed on the current stream of the current device after this call
        will be recorded into the graph until ``capture_end`` is called. By default, capture
        uses the memory pool provided at construction time.

        Arguments:
            capture_error_mode (Literal["default", "global", "thread_local", "relaxed"], optional):
                Specifies the behavior of graph capture. The exact semantics are backend-specific.
                Defaults to `"default"`.
                `default`, backend-defined default capture behavior.
                `global`, potentially unsafe API calls are prohibited. Errors may occur if capture
                in the current thread affects other threads.
                `thread_local`, potentially unsafe API calls are prohibited. Errors occur only if capture
                in the current thread affects itself.
                `relaxed`, the current thread is allowed to make potentially unsafe API calls, except for
                calls that inherently conflict with stream capture.
        """
        super().capture_begin(pool=self.graph_pool, capture_error_mode=capture_error_mode)

    def capture_end(self) -> None:
        r"""
        End graph capture on the current stream of the current device.

        After this call, the graph can be replayed via ``replay``.
        """
        super().capture_end()

    def instantiate(self) -> None:
        r"""
        Instantiate the underlying graph. Will be called by ``capture_end``
        if ``keep_graph=False``, or by ``replay`` if ``keep_graph=True`` and
        ``instantiate`` has not already been explicitly called.
        """
        super().instantiate()

    def replay(self) -> None:
        r"""Replay the work captured by this graph."""
        super().replay()

    def reset(self) -> None:
        r"""Delete the graph currently held by this instance."""
        super().reset()

    def pool(self) -> tuple[int, int]:
        r"""
        Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.

        Example::
            >>> # xdoctest: +SKIP
            >>> g1 = torch.accelerator.Graph()
            >>> g1.capture_begin()
            >>> # ... operations ...
            >>> g1.capture_end()

            >>> # Share g1's memory pool with a new graph
            >>> pool_id = g1.pool()
            >>> g2 = torch.accelerator.Graph(pool=pool_id)
        """
        return super().pool()

    def enable_debug_mode(self) -> None:
        r"""Enable debugging mode for ``debug_dump``."""
        return super().enable_debug_mode()

    def debug_dump(self, path: str) -> None:
        r"""
        Dump the captured graph to a file for debugging purposes if the debugging is
        enabled via ``enable_debug_mode``.

        Arguments:
            path (str): Path to dump the graph to.
        """
        return super().debug_dump(path)

    def __enter__(self) -> None:
        torch.accelerator.synchronize()
        if torch.compiler.config.force_cudagraph_gc:
            # We previously always ran garbage collection here. While this can help
            # reclaim accelerator device memory held by dead Python cycles, it is
            # very expensive, especially when performing multiple graph captures in sequence.
            gc.collect()
        torch.accelerator.empty_cache()
        self.capture_begin()

    def __exit__(self, *exc_info: object) -> None:
        self.capture_end()


__all__ = ["Graph"]
