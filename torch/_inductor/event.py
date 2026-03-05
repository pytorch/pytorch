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
import itertools

from torch._inductor.codegen.wrapper import IndentedBuffer, WrapperLine
from torch._inductor.stream_constants import (
    DEFAULT_STREAM_IDX,
    ENTRANCE_EVENT,
    EVENT_NAME_TEMPLATE,
)
from torch._inductor.stream_utils import get_stream_name


@dataclasses.dataclass(eq=False)
class CudaEventSym:
    """Symbolic representation of CUDA Events in the Inductor scheduling phase.

    Args:
        idx: Indexing number assigned in chronological order during scheduling.
        originate_stream_idx: The index of the CUDA stream that this event originated from.
        materialized_event: The actual CUDA Event name that will be used in the final PyTorch
            program.

    Note:
        In most cases this class should not be used standalone. Use
        `CudaEventFactory.get_sym_event()` to instantiate one.
    """

    idx: int
    originate_stream_idx: int
    materialized_event: str | None = None

    def record(
        self, factory: CudaEventFactory, stream_idx: int
    ) -> _CudaEventRecordLine:
        """Record this event on a given stream."""
        stream = get_stream_name(stream_idx)
        return _CudaEventRecordLine(self, factory, stream)

    def wait(self, stream_idx: int) -> _CudaEventWaitLine:
        """Wait for this event to complete on a given stream."""
        stream = get_stream_name(stream_idx)
        return _CudaEventWaitLine(self, stream)


@dataclasses.dataclass
class _CudaEventRecordLine(WrapperLine):
    event: CudaEventSym
    factory: CudaEventFactory
    stream: str

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.event.materialized_event is None
        self.event.materialized_event = self.factory.get_materialized_event(code)
        code.writeline(f"{self.event.materialized_event}.record({self.stream})")


@dataclasses.dataclass
class _CudaEventWaitLine(WrapperLine):
    event: CudaEventSym
    stream: str

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.event.materialized_event is not None
        code.writeline(f"{self.event.materialized_event}.wait({self.stream})")


class CudaEventFactory:
    """A factory that manages CUDA event creations and materializations.

    This factory maintains internal states to ensure that created cuda events get monotonically
    increasing indices as compilation goes along.
    """

    def __init__(self) -> None:
        self.symbolic_event_idx: itertools.count = itertools.count(start=1)
        self.materialized_event_idx: itertools.count = itertools.count(start=1)
        self._entrance_event: CudaEventSym | None = None

    def get_entrance_event(self) -> CudaEventSym:
        """Return the cuda event that corresponding to compute graph entering."""
        if self._entrance_event is None:
            self._entrance_event = CudaEventSym(
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
            idx=next(self.symbolic_event_idx),
            originate_stream_idx=originate_stream_idx,
        )

    def get_materialized_event(self, code: IndentedBuffer) -> str:
        """Allocate a materialized cuda event."""
        event = EVENT_NAME_TEMPLATE.format(event_idx=next(self.materialized_event_idx))
        code.writeline(f"{event} = torch.cuda.Event()")
        return event
