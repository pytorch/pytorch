"""CUDA Event abstractions for Inductor stream support.

This module provides event types for synchronizing between CUDA streams when
nodes have user-annotated stream assignments.

Attributes:
    ENTRANCE_EVENT: Name of the first event on the default CUDA Stream that got recorded before all
        kernels.
    EVENT_NAME_TEMPLATE: Python string template to generate event names. Can be used as:

            idx: int = ...
            event = EVENT_NAME_TEMPLATE.format(event_idx=idx)
"""

from __future__ import annotations

import dataclasses
import functools
import itertools

from torch._inductor import config
from torch._inductor.stream_utils import (
    DEFAULT_STREAM_IDX,
    ENTRANCE_EVENT,
    EVENT_NAME_TEMPLATE,
    get_stream_name,
)

from torch._inductor.codegen.wrapper import IndentedBuffer, WrapperLine
from torch.utils._ordered_set import OrderedSet


@functools.total_ordering
@dataclasses.dataclass
class CudaEventSym:
    """Symbolic representation of CUDA Events in the Inductor scheduling phase.

    Args:
        factory: The CUDAEventFactory that generate this event.
        idx: Indexing number assigned in chronological order during scheduling.
        originate_stream_idx: The index of the CUDA stream that this event originated from.
        ref_count: Reference count of this event instance.
        materialized_event: The actual CUDA Event name that will be used in the final PyTorch
            program. Only symbolic event with reference count larger than one will be materialized.

    Note:
        In most cases this class should not be used standalone. Use
        `CUDAEventFactory.get_sym_event()` to instantiate one.
    """

    factory: CudaEventFactory
    idx: int
    originate_stream_idx: int
    ref_count: int = 0
    materialized_event: str | None = None

    def __lt__(self, rhs: CudaEventSym) -> bool:
        """Whether the current event is generated before the rhs event."""
        if self.factory is not rhs.factory:
            return NotImplemented
        return (self.idx, self.originate_stream_idx) < (
            rhs.idx,
            rhs.originate_stream_idx,
        )

    def __eq__(self, rhs: object) -> bool:
        """Whether the current event is identical to the rhs event."""
        if not isinstance(rhs, CudaEventSym):
            return NotImplemented
        return (
            self.idx == rhs.idx
            and self.originate_stream_idx == rhs.originate_stream_idx
            and self.factory is rhs.factory
        )

    def __str__(self) -> str:
        """Represent this symbolic event in string."""
        ret = f"{self.__class__.__name__} (idx={self.idx}"
        ret += f", originate_stream_idx={self.originate_stream_idx}"
        if self.ref_count:
            ret += f", ref_count={self.ref_count}"
        if self.materialized_event:
            ret += f", materialized to `{self.materialized_event}`"
        ret += ")"
        return ret

    def __hash__(self) -> int:
        """Hash this symbolic event."""
        return hash((id(self.factory), self.idx, self.originate_stream_idx))

    def record(self, stream_idx: int) -> _CudaEventRecordLine:
        """Record this event on a given stream.

        Args:
            stream_idx: The index of the stream that this event will record on.

        Returns:
            An internal data structure that depicts stream <-> event dependency.

        Note:
            This method doesn't necessarily generate a event recording in the final program.
            Instead it records the dependence between the stream and the current event. Whether
            or not this event recording show up in the final program depends on the reference
            count of the current event. I.e., if this event is never waited for by the later
            code, this event recording will not be code-generated.
        """
        stream = get_stream_name(stream_idx)
        return _CudaEventRecordLine(self, stream)

    def wait(self, stream_idx: int) -> _CudaEventWaitLine:
        """Wait for this event to complete by a given stream.

        Args:
            stream_idx: The index of the stream that will be waiting for this event to complete.

        Returns:
            An internal data structure that depicts stream <-> event dependency.

        Note:
            This method doesn't necessarily generate a event waiting in the final program. Instead
            it records the dependence between the stream and the current event and also increase
            the reference count of this event. If an event object has called this method, it is
            guaranteed to be generated in the final program.
        """
        assert stream_idx != self.originate_stream_idx
        self.ref_count += 1
        stream = get_stream_name(stream_idx)
        return _CudaEventWaitLine(self, stream)


@dataclasses.dataclass
class _CudaEventRecordLine(WrapperLine):
    event: CudaEventSym
    stream: str
    _reuse_cuda_event: bool = dataclasses.field(default_factory=lambda: config.reuse_cuda_event)

    def codegen(self, code: IndentedBuffer) -> None:
        assert 0 <= self.event.ref_count
        assert self.event.materialized_event is None
        if self.event.ref_count or not self._reuse_cuda_event:
            self.event.materialized_event = self.event.factory.get_materialized_event(
                code
            )
            code.writeline(f"{self.event.materialized_event}.record({self.stream})")


@dataclasses.dataclass
class _CudaEventWaitLine(WrapperLine):
    event: CudaEventSym
    stream: str

    def codegen(self, code: IndentedBuffer) -> None:
        assert 0 < self.event.ref_count
        assert self.event.materialized_event is not None
        code_line = f"{self.event.materialized_event}.wait({self.stream})"
        self.event.ref_count -= 1
        if self.event.ref_count == 0:
            self.event.factory.deposit_materialized_event(self.event.materialized_event)
            self.event.materialized_event = None
            code_line += f"  # End lifecycle of {self.event}"
        code.writeline(code_line)


class CudaEventFactory:
    """A factory that managements CUDA event creations and materializations.

    This factory maintains internal states to ensure that created cuda events get monotonically
    increasing indices as compilation goes along. It also maintains a pool of materialized cuda
    events that symbolic events can reuse.
    """

    def __init__(self) -> None:
        """Initialize a event factory."""
        self.symbolic_event_idx: itertools.count = itertools.count(start=1)
        self.materialized_event_idx: itertools.count = itertools.count(start=1)
        self.available_materialized_events: OrderedSet[str] = OrderedSet()
        self._entrance_event: CudaEventSym | None = None
        self._reuse_cuda_event: bool = config.reuse_cuda_event

    def get_entrance_event(self) -> CudaEventSym:
        """Return the cuda event that corresponding to compute graph entering."""
        if self._entrance_event is None:
            self._entrance_event = CudaEventSym(
                factory=self,
                idx=0,
                originate_stream_idx=DEFAULT_STREAM_IDX,
            )
            # Code-gen for entrance event is almost hard-coded in device guard enter so the
            # materialization is slightly different here.
            self._entrance_event.materialized_event = ENTRANCE_EVENT
        return self._entrance_event

    def get_sym_event(self, originate_stream_idx: int) -> CudaEventSym:
        """Allocate a symbolic cuda event."""
        return CudaEventSym(
            factory=self,
            idx=next(self.symbolic_event_idx),
            originate_stream_idx=originate_stream_idx,
        )

    def get_materialized_event(self, code: IndentedBuffer) -> str:
        """Allocate or reuse a materialized cuda event."""
        if self._reuse_cuda_event and self.available_materialized_events:
            return self.available_materialized_events.pop()
        else:
            event = EVENT_NAME_TEMPLATE.format(
                event_idx=next(self.materialized_event_idx)
            )
            code.writeline(f"{event} = torch.cuda.Event()")
            return event

    def deposit_materialized_event(self, event: str) -> None:
        """Give back a materialized cuda event when the corresponding sym event ends lifecycle."""
        assert event not in self.available_materialized_events
        self.available_materialized_events.add(event)
