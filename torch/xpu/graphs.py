from __future__ import annotations

from typing import Optional, overload, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import ParamSpec, Self, TypeVar

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch.xpu import _POOL_HANDLE

from .._utils import _dummy_type

__all__ = [
    "graph_pool_handle",
    "XPUGraph",
    "graph",
]


if not hasattr(torch._C, "_XpuStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_XPUGraph"] = _dummy_type("_XPUGraph")
    torch._C.__dict__["_graph_pool_handle"] = _dummy_type("_graph_pool_handle")

from torch._C import (  # noqa: F401
    _XPUGraph,
    _graph_pool_handle,
)

def graph_pool_handle() -> _POOL_HANDLE:
    r"""Return an opaque token representing the id of a graph memory pool.

    """
    return torch.xpu._POOL_HANDLE(_graph_pool_handle())

class XPUGraph(torch._C._XPUGraph):
    r"""Wrapper around a XPU graph.

    Arguments:
        keep_graph (bool, optional): If ``keep_graph=False``, the
            executable command graph will be instantiated on GPU at the end of
            ``capture_end`` and the underlying modifiable command graph will be
            destroyed. Note that the executable command graph will not be
            instantiated at the end of ``capture_end`` in this
            case. Instead, it will be instantiated via an explicit called
            to ``instantiate`` or automatically on the first call to
            ``replay`` if ``instantiate`` was not already called. Calling
            ``instantiate`` manually before ``replay`` is recommended to
            prevent increased latency on the first call to ``replay``.

    """

    def __new__(cls, keep_graph: bool = False) -> Self:
        return super().__new__(cls, keep_graph)

    def capture_begin(
        self, pool: Optional[_POOL_HANDLE] = None) -> None:
        r"""Begin capturing XPU work on the current sycl queue.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch.xpu.graph`, which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch.xpu.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch.xpu.XPUGraph.pool>`) that hints this graph may share memory
                with the indicated pool.
        """
        super().capture_begin(pool=pool)

    def capture_end(self) -> None:
        r"""End XPU graph capture on the current stream.

        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch.xpu.graph`, which call ``capture_end`` internally.
        """
        super().capture_end()

    def instantiate(self) -> None:
        r"""Instantiate the XPU graph. Will be called by
        ``capture_end`` if ``keep_graph=False``, or by ``replay`` if
        ``keep_graph=True`` and ``instantiate`` has not already been
        explicitly called. Does not destroy the xpu modify command graph returned
        by ``raw_xpu_graph``.
        """
        super().instantiate()

    def replay(self) -> None:
        r"""Replay the XPU work captured by this graph."""
        super().replay()

    def reset(self) -> None:
        r"""Delete the graph currently held by this instance."""
        super().reset()

class graph:
    r"""Context-manager that captures XPU work into a :class:`torch.xpu.XPUGraph` object for later replay.

    Arguments:
        xpu_graph (torch.xpu.XPUGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.xpu.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.xpu.XPUGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool.
        stream (torch.xpu.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    """  # noqa: B950

    default_capture_stream: Optional[torch.xpu.Stream] = None

    def __init__(
        self,
        xpu_graph: XPUGraph,
        pool: Optional[_POOL_HANDLE] = None,
        stream: Optional[torch.xpu.Stream] = None,
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.xpu.Stream()

        self.pool: Union[tuple[()], tuple[_POOL_HANDLE]] = (
            () if pool is None else (pool,)
        )
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        assert self.capture_stream is not None
        self.stream_ctx = torch.xpu.stream(self.capture_stream)
        self.xpu_graph = xpu_graph

    def __enter__(self) -> None:
        # Free as much memory as we can for the graph
        torch.xpu.synchronize()

        torch.xpu.empty_cache()
        self.stream_ctx.__enter__()

        self.xpu_graph.capture_begin(*self.pool)

    def __exit__(self, *args: object) -> None:
        self.xpu_graph.capture_end()
        self.stream_ctx.__exit__(*args)
