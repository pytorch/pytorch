# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""Breakable CUDA graphs for PyTorch."""

from __future__ import annotations

import contextvars
import functools
import logging
import os
import threading
from typing import Any, TYPE_CHECKING, TypeGuard
from typing_extensions import Self

import torch
import torch.cuda._gpu_trace as _gpu_trace


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from .graphs import _GraphPool


__all__ = [
    "BreakableCUDAGraph",
    "breakable_graph",
    "no_graph",
    "force_no_graph",
    "is_in_breakable_graph",
]

logger: logging.Logger = logging.getLogger(__name__)

_current_breakable_graph_ctx: contextvars.ContextVar[breakable_graph | None] = (
    contextvars.ContextVar("current_breakable_graph_ctx", default=None)
)


def is_in_breakable_graph() -> bool:
    r"""is_in_breakable_graph() -> bool

    Return ``True`` inside :class:`~torch.cuda.breakable_graph`, including its
    eager breaks. Return ``False`` during replay or outside capture.
    """
    return _current_breakable_graph_ctx.get() is not None


# Opt-in debug mode (read once at import). When on, the GPU-trace callbacks
# below track side-stream fork/join so an unjoined-stream error can name the
# offending stream id(s); it never changes capture behavior.
_DEBUG: bool = os.environ.get("TORCH_BREAKABLE_CUDA_GRAPHS_DEBUG", "0") == "1"


def _on_event_record(event_id: int, stream_id: int) -> None:
    ctx = _current_breakable_graph_ctx.get()
    if ctx is not None and _DEBUG:
        ctx._event_to_stream_id[event_id] = stream_id


def _on_event_wait(event_id: int, stream_id: int) -> None:
    ctx = _current_breakable_graph_ctx.get()
    if ctx is None or not _DEBUG:
        return

    recording_stream_id = ctx._event_to_stream_id.get(event_id)
    if recording_stream_id is None:
        return

    recording_is_capturing = ctx._is_capturing_stream_id(recording_stream_id)
    waiting_is_capturing = ctx._is_capturing_stream_id(stream_id)

    if recording_is_capturing == waiting_is_capturing:
        return

    if recording_is_capturing:
        ctx._forked_stream_ids.add(stream_id)
    else:
        ctx._forked_stream_ids.discard(recording_stream_id)


_gpu_trace_initialized: bool = False
_gpu_trace_lock: threading.Lock = threading.Lock()


def _ensure_gpu_trace() -> None:
    global _gpu_trace_initialized
    if _gpu_trace_initialized:
        return
    with _gpu_trace_lock:
        if _gpu_trace_initialized:
            return
        torch._C._activate_gpu_trace()
        _gpu_trace.register_callback_for_event_record(_on_event_record)
        _gpu_trace.register_callback_for_event_wait(_on_event_wait)
        _gpu_trace_initialized = True


def _is_cuda_tensor(x: object) -> TypeGuard[torch.Tensor]:
    return isinstance(x, torch.Tensor) and x.is_cuda


def _make_replay_tensor_alias(x: object) -> object:
    if not _is_cuda_tensor(x):
        return x
    # TODO: Replace with torch._from_blob once pytorch/pytorch#185850 lands.
    us = x.untyped_storage()
    data_ptr = us.data_ptr()
    nbytes = us.nbytes()
    storage = torch._C._construct_storage_from_data_pointer(data_ptr, x.device, nbytes)
    metadata = {
        "data_ptr": data_ptr,
        "nbytes": nbytes,
        "device": x.device,
        "size": list(x.shape),
        "stride": list(x.stride()),
        "storage_offset": x.storage_offset(),
        "dtype": x.dtype,
    }
    alias = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, storage)
    torch._C._set_conj(alias, x.is_conj())
    torch._C._set_neg(alias, x.is_neg())
    return alias


def no_graph(
    fn: Callable[..., Any] | None = None,
    *,
    enable: bool = True,
    capture_stub: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    r"""no_graph(fn=None, *, enable=True, capture_stub=None) -> Callable

    Run a function eagerly inside a :class:`~torch.cuda.breakable_graph` capture.

    Calls to a decorated function end the current CUDA graph segment, execute the
    function normally, record it as an eager replay step, and then begin a new
    graph segment. Outside ``breakable_graph`` capture, the wrapper calls the
    original function directly.

    Decorated functions must not return CUDA tensors. A CUDA tensor returned from
    an eager function cannot be safely reused across replays because its storage
    may move or be reused. Write CUDA outputs into pre-allocated buffers passed as
    arguments instead. Returning a CUDA tensor, including one nested in a tuple,
    list, or dict, raises :class:`RuntimeError` during capture. Scalars and CPU
    tensors are allowed.

    CUDA argument buffers must remain valid until replay finishes. Python values
    and CPU tensors returned during capture are not updated by replay; control
    flow in subsequent graph segments is fixed during capture.

    Can be used as ``@no_graph`` or ``@no_graph(enable=True)``. Passing
    ``enable=False`` leaves the function unchanged. ``capture_stub`` can replace
    the function body during capture; replay still calls the original function.
    The stub receives the same arguments and must follow the same return-value
    restrictions as the decorated function. Its return value is used by the
    remainder of the capture pass, so it must also be compatible with how the
    caller uses the real function's return value.

    If the enclosing :class:`breakable_graph` was given a ``barrier_fn``, it runs
    at capture between ending the preceding segment and calling the decorated
    function.

    Args:
        fn: Function to decorate. Leave as ``None`` when using the configured
            form, e.g. ``@no_graph(enable=...)``.
        enable: Whether to apply the eager-break wrapper. When ``False``,
            ``fn`` is returned unchanged.
        capture_stub: Optional callable used instead of ``fn`` during capture
            only. Useful when the real eager body is expensive or contains
            rank-coupled work whose results are not needed while constructing
            the surrounding graph segments.
    """

    if capture_stub is not None and not callable(capture_stub):
        raise TypeError(f"`capture_stub` must be callable, got {type(capture_stub)}")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not enable:
            return fn

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _current_breakable_graph_ctx.get()
            if ctx is None or ctx._graph_ctx is None:
                return fn(*args, **kwargs)

            ctx._end_segment()

            # Segment finalization can take different amounts of host time across
            # distributed ranks. Re-align them before rank-coupled eager work.
            if ctx._barrier_fn is not None:
                ctx._barrier_fn()

            capture_fn = capture_stub if capture_stub is not None else fn
            captured_result = capture_fn(*args, **kwargs)

            result_leaves, _ = torch.utils._pytree.tree_flatten(captured_result)
            if any(_is_cuda_tensor(leaf) for leaf in result_leaves):
                raise RuntimeError(
                    f"`{fn.__qualname__}` is decorated with `@no_graph` but "
                    f"returns one or more CUDA tensors. A CUDA tensor returned "
                    f"from an eager function cannot be safely reused across "
                    f"replays; write results into a pre-allocated buffer passed "
                    f"in as an argument instead."
                )

            # Store non-owning views of CUDA argument buffers. Eager segments are
            # kept for replay, so retaining the original tensors would extend
            # their lifetimes and can prevent the allocator from reusing those
            # buffers in later graph segments.
            replay_args = torch.utils._pytree.tree_map(_make_replay_tensor_alias, args)
            replay_kwargs = torch.utils._pytree.tree_map(
                _make_replay_tensor_alias, kwargs
            )

            ctx._insert_eager(fn, replay_args, replay_kwargs)
            ctx._begin_segment()

            return captured_result

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


@no_graph
def force_no_graph() -> None:
    r"""force_no_graph() -> None

    Split the current :class:`~torch.cuda.breakable_graph` segment without eager work.

    Outside breakable capture, this function does nothing.
    """


class _EagerSegment:
    r"""Eager replay step stored alongside captured CUDA graph segments.

    Holds a no-graph function plus non-owning argument-buffer views, and exposes
    the same ``replay()`` / ``reset()`` methods as :class:`torch.cuda.CUDAGraph`
    so :class:`BreakableCUDAGraph` can replay all segments uniformly.
    """

    __slots__ = ("fn", "args", "kwargs")

    def __init__(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def replay(self) -> None:
        self.fn(*self.args, **self.kwargs)

    def reset(self) -> None:
        # No captured graph state to free; the args are non-owning pins.
        pass


class BreakableCUDAGraph:
    r"""BreakableCUDAGraph(pool=None)

    Capture output from :class:`~torch.cuda.breakable_graph` that can be replayed.

    A sequence starts empty. Passing it to :class:`breakable_graph` appends CUDA
    graph segments and eager ``@no_graph`` segments as capture progresses. Call
    :meth:`replay` to re-execute the captured sequence, or :meth:`reset` to drop
    all captured segments and reuse the object.

    Args:
        pool (optional): CUDA graph memory pool handle or :class:`~torch.cuda.MemPool`.
            Pass ``other.pool()`` to share a pool with another sequence;
            otherwise a pool is created lazily on first use.

    .. warning::
        This API is in prototype and may change in future releases.
    """

    def __init__(self, pool: _GraphPool | None = None) -> None:
        self._pool = pool
        self._segments: list[torch.cuda.CUDAGraph | _EagerSegment] = []

    def _append_graph(self) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        self._segments.append(g)
        return g

    def pool(self) -> _GraphPool:
        r"""Return the memory pool shared by all captured graph segments.

        See :ref:`Graph memory management<graph-memory-management>` for pool
        sharing constraints. Calling :meth:`reset` preserves this pool.
        """
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        return self._pool

    def replay(self) -> None:
        r"""Replay captured CUDA graph segments and eager functions in order.

        External CUDA input and output buffers must remain alive, at their
        captured addresses, until replay completes.
        """
        for segment in self._segments:
            segment.replay()

    def reset(self) -> None:
        r"""Release captured segments and eager functions, preserving the memory pool."""
        for segment in self._segments:
            segment.reset()
        self._segments.clear()


class breakable_graph:
    r"""breakable_graph(cuda_graph, stream=None, capture_error_mode="global", barrier_fn=None)

    Capture CUDA work as multiple graph segments with eager breaks.

    This context manager behaves like :class:`torch.cuda.graph`, except calls to
    functions decorated with :func:`no_graph` run eagerly and split capture into
    separate CUDA graph segments. The resulting segments are appended to the
    provided :class:`BreakableCUDAGraph`, which can then replay the full sequence.

    Usual :ref:`CUDA graph constraints<cuda-graph-semantics>` apply within each
    segment. Side streams must join the capture stream before an eager break
    or the end of the context. Set ``TORCH_BREAKABLE_CUDA_GRAPHS_DEBUG=1`` before
    importing PyTorch to include unjoined stream IDs in errors.

    Nested breakable captures are not supported. Reset a graph before capturing
    a replacement sequence; captures otherwise append to the existing sequence.

    For concurrent captures from multiple threads with no-graph regions, use
    ``thread_local`` capture error mode and separate streams per thread.

    Args:
        cuda_graph: Breakable CUDA graph that receives captured graph segments
            and eager no-graph segments.
        stream: Optional side stream to use for capture. Matches the ``stream``
            argument of :class:`torch.cuda.graph`.
        capture_error_mode: CUDA stream-capture error mode forwarded to
            :class:`torch.cuda.graph`.
        barrier_fn: Optional zero-argument callable run before each eager break,
            after the preceding graph segment has ended. Runs during capture
            only, not on replay. This can re-align distributed ranks before an
            eager break containing rank-coupled work. Its return value is
            discarded.

    Examples::
        >>> @torch.cuda.no_graph
        ... def copy_to_buffer(dst: torch.Tensor, src: torch.Tensor) -> None:
        ...     dst.copy_(src)
        ...
        >>> graph = torch.cuda.BreakableCUDAGraph()
        >>> static_input = torch.ones(5, device="cuda")
        >>> static_output = torch.empty_like(static_input)
        >>> s = torch.cuda.Stream()
        >>> s.wait_stream(torch.cuda.current_stream())
        >>> with torch.cuda.stream(s):
        ...     copy_to_buffer(static_output, static_input)
        >>> torch.cuda.current_stream().wait_stream(s)
        >>> with torch.cuda.breakable_graph(graph):
        ...     static_input.mul_(2)
        ...     copy_to_buffer(static_output, static_input)
        ...     static_output.add_(1)
        >>> graph.replay()
    """

    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        stream: torch.cuda.Stream | None = None,
        capture_error_mode: str = "global",
        barrier_fn: Callable[[], Any] | None = None,
    ) -> None:
        if barrier_fn is not None and not callable(barrier_fn):
            raise TypeError(f"`barrier_fn` must be callable, got {type(barrier_fn)}")
        self._cuda_graph = cuda_graph
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._barrier_fn = barrier_fn
        self._graph_ctx: torch.cuda.graph | None = None
        self._token: contextvars.Token[breakable_graph | None] | None = None
        self._capturing_stream: torch.cuda.Stream | None = None
        self._capturing_stream_id: int | None = None
        self._forked_stream_ids: set[int] = set()
        self._event_to_stream_id: dict[int, int] = {}
        if _DEBUG:
            _ensure_gpu_trace()

    def _is_capturing_stream_id(self, stream_id: int) -> bool:
        return stream_id == self._capturing_stream_id

    def _insert_eager(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self._graph_ctx is not None:
            raise AssertionError("cannot insert eager work during a graph segment")
        self._cuda_graph._segments.append(_EagerSegment(fn, args, kwargs))

    def _begin_segment(self) -> None:
        g = self._cuda_graph._append_graph()
        graph_ctx = torch.cuda.graph(
            g,
            pool=self._cuda_graph.pool(),
            stream=self._stream,
            capture_error_mode=self._capture_error_mode,
        )
        graph_ctx.__enter__()
        self._graph_ctx = graph_ctx

    def _end_segment(self) -> None:
        try:
            graph_ctx = self._graph_ctx
            if graph_ctx is None:
                raise AssertionError("expected an active CUDA graph segment")
            graph_ctx.__exit__(None, None, None)
        except RuntimeError as e:
            # cudaErrorStreamCaptureUnjoined has no typed code; match its message
            # and re-raise anything else unchanged.
            text = str(e).lower()
            if "unjoined" not in text and "not joined" not in text:
                raise
            msg = (
                "CUDA graph capture failed because a "
                "side stream was not joined back to the capturing stream before "
                "entering an @no_graph function or leaving breakable_graph. "
                "Join the side stream before ending the graph segment."
            )
            if _DEBUG and self._forked_stream_ids:
                msg += (
                    f" Unjoined side-stream id(s): {sorted(self._forked_stream_ids)}."
                )
            raise RuntimeError(msg) from e
        finally:
            self._graph_ctx = None
            if _DEBUG:
                self._event_to_stream_id.clear()

    def __enter__(self) -> Self:
        if _current_breakable_graph_ctx.get() is not None:
            raise RuntimeError("nested breakable_graph captures are not supported")
        self._begin_segment()
        if _DEBUG:
            self._capturing_stream = torch.cuda.current_stream()
            self._capturing_stream_id = self._capturing_stream.cuda_stream
        self._token = _current_breakable_graph_ctx.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        try:
            # _graph_ctx is None if an exception occurred during a no-graph
            # region (after _end_segment but before _begin_segment).
            graph_ctx = self._graph_ctx
            if graph_ctx is not None:
                if exc_type is not None:
                    # An exception is already unwinding; close best-effort and
                    # don't mask it with a cleanup error.
                    try:
                        graph_ctx.__exit__(exc_type, exc_val, exc_tb)
                    except RuntimeError as e:
                        logger.warning("CUDA graph cleanup failed: %s", e)
                else:
                    self._end_segment()
        finally:
            self._graph_ctx = None
            if _DEBUG:
                self._event_to_stream_id.clear()
            if self._token is not None:
                _current_breakable_graph_ctx.reset(self._token)
                self._token = None
        return False
