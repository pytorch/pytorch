"""Cross-stream lifetime utilities for FSDP2.

See the class docstring of :class:`StreamHandoff` for motivation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["PipelinedBuffer", "StreamHandoff"]


class StreamHandoff:
    """Hold a tensor alive across a cross-stream producer->consumer handoff.

    Wraps a triple of ``(tensor, ready_event, release_stream)``. The caching
    allocator only tracks the *allocation* stream of a block; it has no
    knowledge of which other streams are reading the block. When a buffer is
    produced on stream A and consumed on stream B, dropping the Python ref
    too early lets a later stream-A allocation reclaim the block while B is
    still reading - a use-after-free.

    ``StreamHandoff`` encapsulates the three pieces needed to drop the ref
    safely:

    1. ``ready_event`` must be recorded on the producer stream after the
       producer is done writing (or, for keep-alive patterns, after the
       last consumer is done reading).
    2. ``release_stream`` must have ``wait_event(ready_event)`` before the
       ref is dropped. :meth:`release` performs this wait itself as a
       safety net; callers that have already issued the wait pay only the
       cost of a redundant ``cudaStreamWaitEvent`` (O(1), no GPU stall).
    3. The ``del`` of the underlying tensor happens inside a
       ``stream(release_stream)`` context, so the caching allocator routes
       the free to ``release_stream``'s pool - whose FIFO already includes
       the ``wait_event``. Routing the free to any other pool (the bug
       fixed by PR #179443) would allow a different stream's allocation to
       reclaim the block before ``release_stream`` finishes reading.

    Args:
        tensor: Buffer to hold alive. Must not be ``None``.
        ready_event: Event recorded on the producer stream, marking the
            point after which consumers are done. May be ``None`` when
            the producer/consumer path is synchronous (e.g. CPU backends
            where collectives block); in that case ``release()`` skips
            the ``wait_event`` step.
        release_stream: Stream whose caching-allocator pool should receive
            the freed block. Typically the last stream to read ``tensor``,
            or a stream that has a barrier waiting on ``ready_event``.
        device_handle: Device module (e.g. ``torch.cuda``, ``torch.xpu``)
            used to open the stream context at release time. If ``None``,
            inferred from ``tensor.device.type``.

    Example - AR keep-alive in FSDP2 HSDP backward, via
    :class:`PipelinedBuffer`::

        # Layer N+1, top of post_backward: pop prior, use event for ordering.
        prev_event = comm_ctx.all_reduce_buffer.pop_event()
        (..., new_input, new_event, ...) = foreach_reduce(..., prev_event)
        # Layer N, end of post_backward: stash new handoff.
        if new_input is not None:
            comm_ctx.all_reduce_buffer.push(
                StreamHandoff(new_input, new_event, all_reduce_stream)
            )
    """

    __slots__ = (
        "_tensor",
        "_event",
        "_release_stream",
        "_device_handle",
        "_released",
    )

    def __init__(
        self,
        tensor: torch.Tensor,
        ready_event: torch.Event | None,
        release_stream: torch.Stream,
        device_handle: ModuleType | None = None,
    ) -> None:
        if tensor is None:
            raise ValueError("StreamHandoff tensor must not be None")
        if release_stream is None:
            raise ValueError("StreamHandoff release_stream must not be None")
        self._tensor: torch.Tensor | None = tensor
        self._event = ready_event
        self._release_stream = release_stream
        if device_handle is None:
            device_handle = getattr(torch, tensor.device.type, None)
            if device_handle is None:
                raise ValueError(
                    f"Cannot infer device_handle for tensor on device "
                    f"{tensor.device}; pass device_handle explicitly"
                )
        self._device_handle = device_handle
        self._released = False

    @property
    def tensor(self) -> torch.Tensor:
        """The held tensor. Raises ``RuntimeError`` if already released."""
        if self._released or self._tensor is None:
            raise RuntimeError("StreamHandoff has been released")
        return self._tensor

    @property
    def event(self) -> torch.Event | None:
        return self._event

    @property
    def release_stream(self) -> torch.Stream:
        return self._release_stream

    @property
    def released(self) -> bool:
        return self._released

    def wait(self, stream: torch.Stream) -> None:
        """Make ``stream`` wait on ``ready_event``.

        Use for multi-consumer cases where more than one stream reads the
        tensor before release. Callers should invoke :meth:`wait` once per
        extra consumer stream, then call :meth:`release` to drop the ref
        (``release`` also waits on ``release_stream`` internally).

        No-op if ``ready_event`` is ``None`` or the handoff is released.
        """
        if self._released or self._event is None:
            return
        stream.wait_event(self._event)

    def release(self) -> None:
        """Drop the tensor ref, routing the free to ``release_stream``'s pool.

        Idempotent: subsequent calls are no-ops.

        Steps, in order:

        1. ``release_stream.wait_event(ready_event)`` - ensures the release
           stream's FIFO is ordered after the consumer-done point.
           Redundant if the caller has already issued this wait; the second
           wait is a cheap no-op on the GPU side.
        2. ``del tensor`` inside ``stream(release_stream)`` - the caching
           allocator consults the *current* stream at free time to choose
           the free pool. Routing to ``release_stream`` ensures the next
           allocation on ``release_stream`` is ordered after step 1's wait.
        """
        if self._released:
            return
        self._released = True
        if self._tensor is None:
            return
        if self._event is not None:
            self._release_stream.wait_event(self._event)
        with self._device_handle.stream(self._release_stream):
            self._tensor = None

    def __del__(self) -> None:
        # Safety-net cleanup. Callers in hot paths should use explicit
        # release() for determinism and to keep behavior predictable under
        # @_dynamo_disable / torch.compile / refcount-sensitive scenarios.
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._released or self._tensor is None:
            return "<StreamHandoff released>"
        return (
            f"<StreamHandoff tensor={tuple(self._tensor.shape)}"
            f" dtype={self._tensor.dtype}"
            f" release_stream={self._release_stream}>"
        )


class PipelinedBuffer:
    """Sequence of :class:`StreamHandoff` with drain-all and single-slot-pop idioms.

    FSDP2's cross-layer keep-alive buffers come in two shapes:

    - **Single-slot pipeline** (AR, AG): layer N stashes one handoff on the
      context; layer N+1 needs the prior event for causal ordering and then
      releases the prior. Exactly one entry at a time.
    - **Multi-slot queue** (RS): per-group handoffs accumulate across HSDP
      groups and drain together at the next layer / end-of-backward.

    ``PipelinedBuffer`` serves both. Callers pick between :meth:`pop_event`
    (single-slot idiom: returns the sole entry's event, releases it) and
    :meth:`flush` (multi-slot idiom: drain everything, with optional extra
    consumer-stream waits).

    :meth:`flush` accepts additional ``extra_waiters`` streams; each entry
    calls ``handoff.wait(stream)`` on them before ``release()``. This covers
    two cases today:

    - AG prefetch drain: both AG streams (copy-in and AG) need the wait;
      ``release_stream`` is the AG stream, so the copy-in stream is an
      extra waiter.
    - End-of-backward AR flush: the default stream must be ordered after
      the AR event so that subsequent default-stream allocations don't
      alias the AR buffer's storage.

    Example (single-slot, AR)::

        buf = comm_ctx.all_reduce_buffer
        prev_event = buf.pop_event()           # releases prior; may be None
        (..., new_input, new_event, ...) = foreach_reduce(..., prev_event)
        if new_input is not None:
            buf.push(StreamHandoff(new_input, new_event, all_reduce_stream))

    Example (multi-slot, RS)::

        buf = comm_ctx.reduce_scatter_buffer
        if last_group_condition:
            buf.flush()  # drain accumulated entries
        buf.push(StreamHandoff(rs_input, rs_event, default_stream))
    """

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[StreamHandoff] = []

    def push(self, handoff: StreamHandoff) -> None:
        """Append ``handoff`` to the buffer."""
        self._items.append(handoff)

    def pop_event(self) -> torch.Event | None:
        """Pop the sole entry; release it; return its event.

        Returns ``None`` if the buffer is empty. Raises ``RuntimeError`` if
        the buffer holds more than one entry - that would indicate a misuse
        of the single-slot pipeline idiom.
        """
        if not self._items:
            return None
        if len(self._items) > 1:
            raise RuntimeError(
                f"pop_event() requires at most one entry; have {len(self._items)}"
            )
        handoff = self._items.pop()
        event = handoff.event
        handoff.release()
        return event

    def flush(self, *extra_waiters: torch.Stream) -> None:
        """Release every entry. Idempotent.

        For each entry, ``extra_waiters`` are made to wait on the entry's
        event before the entry's ``release()`` runs (which itself waits
        from ``release_stream``). No-op on an empty buffer.
        """
        for handoff in self._items:
            for stream in extra_waiters:
                handoff.wait(stream)
            handoff.release()
        self._items.clear()

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<PipelinedBuffer entries={len(self._items)}>"
