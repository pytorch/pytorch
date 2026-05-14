from __future__ import annotations  # FIX 5: enables | union syntax on Python 3.9+

import gc
from typing import Literal, Optional, Tuple
from typing_extensions import Self, TypeAlias

import torch
from torch._C import _acceleratorGraph


# FIX 4 (DRY): Extract Literal type alias so it isn't duplicated in __new__ and __init__
CaptureErrorMode: TypeAlias = Literal["default", "global", "thread_local", "relaxed"]


class Graph(_acceleratorGraph):
    r"""
    Wrapper around an :ref:`accelerator<accelerators>` graph that supports
    capture and replay.

    A graph captures a sequence of operations and their dependencies, allowing
    them to be replayed efficiently with reduced overhead.  This class can be
    used as a context manager to automatically capture operations on the
    current stream.

    Arguments:
        keep_graph (bool, optional): If ``False``, the underlying graph is
            destroyed and the executable graph is instantiated on the device at
            the end of ``capture_end``.  If ``True``, the underlying graph is
            preserved after ``capture_end``; the executable graph must then be
            created explicitly via ``instantiate``, or it will be created on
            the first call to ``replay``.  Default: ``False``.
        pool (tuple[int, int], optional): Memory-pool identifier for this
            graph.  Multiple graphs can share the same pool by passing the
            same identifier, which reduces memory overhead.
            Default: ``None``.
        capture_error_mode (CaptureErrorMode, optional): Controls graph-capture
            error semantics.  The exact behaviour is backend-specific.

            - ``"default"``      – backend-defined default.
            - ``"global"``       – unsafe API calls are prohibited; errors may
              arise when capture in the current thread affects other threads.
            - ``"thread_local"`` – unsafe API calls are prohibited; errors arise
              only when capture affects the current thread itself.
            - ``"relaxed"``      – the current thread may make potentially unsafe
              calls, except those that inherently conflict with stream capture.

            Default: ``"default"``.

    Example::

        >>> # xdoctest: +SKIP
        >>> x = torch.zeros([2000], device=0)
        >>> stream = torch.Stream()
        >>> graph = torch.accelerator.Graph()
        >>> with stream, graph:
        ...     x += 1
        >>> graph.replay()

    Using ``as`` to keep a reference inside the ``with`` block::

        >>> # xdoctest: +SKIP
        >>> with torch.accelerator.Graph() as g:
        ...     x += 1
        >>> g.replay()
    """

    def __new__(
        cls,
        keep_graph: bool = False,
        *,
        pool: Optional[Tuple[int, int]] = None,
        capture_error_mode: CaptureErrorMode = "default",  # FIX 4: reuse alias
    ) -> Self:
        return super().__new__(cls, keep_graph)

    def __init__(
        self,
        keep_graph: bool = False,
        *,
        pool: Optional[Tuple[int, int]] = None,
        capture_error_mode: CaptureErrorMode = "default",  # FIX 4: reuse alias
    ) -> None:
        super().__init__(keep_graph)
        self.graph_pool = pool
        self.capture_error_mode: CaptureErrorMode = capture_error_mode

    # pyrefly: ignore [bad-override]
    def capture_begin(self) -> None:
        r"""
        Begin graph capture on the current stream.

        All operations on the current stream after this call will be recorded
        until ``capture_end`` is called, using the memory pool and capture
        error mode provided at construction time.
        """
        super().capture_begin(
            pool=self.graph_pool,
            capture_error_mode=self.capture_error_mode,
        )

    def capture_end(self) -> None:
        r"""
        End graph capture on the current stream of the current device.

        After this call, the graph can be replayed via ``replay``.
        """
        super().capture_end()

    def instantiate(self) -> None:
        r"""
        Instantiate the underlying graph.

        Called automatically by ``capture_end`` when ``keep_graph=False``, or
        by the first ``replay`` call when ``keep_graph=True`` and
        ``instantiate`` has not been called explicitly.
        """
        super().instantiate()

    def replay(self) -> None:
        r"""Replay the work captured by this graph."""
        super().replay()

    def reset(self) -> None:
        r"""
        Delete the graph currently held by this instance.

        After this call, the graph can be recaptured.  Set
        :attr:`graph_pool` or :attr:`capture_error_mode` beforehand to
        use different settings on the next capture.
        """
        super().reset()

    def pool(self) -> Tuple[int, int]:
        r"""
        Return an opaque token representing the id of this graph's memory pool.

        The id can optionally be passed to another graph's ``capture_begin``
        to hint that the two graphs may share the same memory pool.

        Example::

            >>> # xdoctest: +SKIP
            >>> g1 = torch.accelerator.Graph()
            >>> g1.capture_begin()
            >>> # ... operations ...
            >>> g1.capture_end()

            >>> # Share g1's memory pool with a new graph
            >>> g2 = torch.accelerator.Graph(pool=g1.pool())
        """
        return super().pool()

    def enable_debug_mode(self) -> None:
        r"""Enable debugging mode for ``debug_dump``."""
        return super().enable_debug_mode()

    def debug_dump(self, path: str) -> None:
        r"""
        Dump the captured graph to a file for debugging.

        Debugging must first be enabled via ``enable_debug_mode``.

        Arguments:
            path (str): Filesystem path to write the dump to.

        Example::

            >>> # xdoctest: +SKIP
            >>> s = torch.Stream()
            >>> g = torch.accelerator.Graph()
            >>> g.enable_debug_mode()
            >>> with s, g:
            ...     # ... operations ...
            ...     pass
            >>> g.debug_dump("graph_dump.dot")
        """
        return super().debug_dump(path)

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:  # FIX 1: was `-> None`; must return Self so
        """                        # `with graph as g:` assigns the Graph object.
        Prepare the device and begin graph capture on the current stream.
        """
        torch.accelerator.synchronize()
        if torch.compiler.config.force_cudagraph_gc:
            # Garbage collection can reclaim device memory held by dead Python
            # cycles, but is expensive when doing many sequential captures.
            gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.empty_host_cache()
        self.capture_begin()
        return self  # FIX 1: return self so `with Graph() as g:` works

    def __exit__(  # FIX 2 & 3: was missing exception params and return type
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool | None:
        """
        End graph capture, unless an exception was raised during capture.

        If an exception occurred inside the ``with`` block the capture is
        aborted via ``reset`` to avoid leaving the graph in a corrupt state,
        and the exception is re-raised (return value ``False``/``None``).
        """
        if exc_type is not None:
            # FIX 2: An exception occurred mid-capture.  Calling capture_end()
            # here would produce an incomplete / corrupt graph.  Reset instead
            # so the object can be safely recaptured later.
            try:
                self.reset()
            except Exception:
                pass  # Best-effort cleanup; original exception takes priority.
            return False  # Do not suppress the original exception.

        self.capture_end()
        return None  # FIX 3: explicit None (do not suppress exceptions)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # UPGRADE: useful for debugging / logging
        return (
            f"{self.__class__.__name__}("
            f"graph_pool={self.graph_pool!r}, "
            f"capture_error_mode={self.capture_error_mode!r})"
        )


__all__ = ["Graph", "CaptureErrorMode"]  # UPGRADE: export the TypeAlias too
