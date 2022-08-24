import enum
import io
import logging
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils._cuda_trace import _CUDA_HOOK_TRACE_OFFSET
from torch.utils._python_dispatch import _PYTHON_DISPATCH_TRACE_OFFSET


DataPtr = int
StreamId = int
EventId = int
SeqNum = int


logger = logging.getLogger(__name__)


class AccessType(enum.Enum):
    READ = enum.auto()
    WRITE = enum.auto()

    def __str__(self):
        return "reading from" if self is AccessType.READ else "writing to"


@dataclass
class Access:
    type: AccessType
    seq_num: SeqNum
    stream: StreamId
    operator: str
    names: List[str]
    stack_trace: List[str]


class SynchronizationError(Exception):
    """Base class for errors detected by CSAN."""

    pass


class UnsynchronizedAccessError(SynchronizationError):
    def __init__(
        self,
        data_ptr: DataPtr,
        allocation_trace: Optional[List[str]],
        current_access: Access,
        previous_access: Access,
    ):
        with io.StringIO() as message:
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race on tensor with data pointer {data_ptr}
                    Access by stream {current_access.stream} during kernel:
                    {current_access.operator}
                    {current_access.type} argument: {', '.join(current_access.names)}
                    With stack trace:
                    """
                )
            )
            message.write(f"{''.join(current_access.stack_trace)}\n")
            message.write(
                textwrap.dedent(
                    f"""\
                    Previous access by stream {previous_access.stream} during kernel:
                    {previous_access.operator}
                    {previous_access.type} argument: {', '.join(previous_access.names)}
                    With stack trace:
                    """
                )
            )
            message.write(f"{''.join(previous_access.stack_trace)}\n")
            if allocation_trace:
                message.write(
                    "Tensor was allocated with stack trace:\n"
                    f"{''.join(allocation_trace)}"
                )
            else:
                message.write("Trace for tensor allocation not found.")
            super().__init__(message.getvalue())


def format_log_message(message: str) -> str:
    return " ".join(line.strip() for line in message.strip().splitlines())


@dataclass
class TensorInfo:
    allocation_trace: Optional[List[str]]
    reads: List[Access] = field(default_factory=list)
    write: Optional[Access] = None


class _TensorsAccessed:
    def __init__(self):
        self.accesses: Dict[DataPtr, TensorInfo] = {}

    def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
        if data_ptr not in self.accesses:
            logger.info(
                format_log_message(
                    f"""
                    Found tensor with pointer: {data_ptr}, but no matching tensor
                    allocation in the trace. Backfilling the trace now.
                    No action is necessary.
                    """
                )
            )
            self.create_tensor(data_ptr, None)

    def ensure_tensor_does_not_exist(self, data_ptr: DataPtr) -> None:
        if data_ptr in self.accesses:
            logger.info(
                format_log_message(
                    f"""
                    Found duplicate tensor allocation in the trace for tensor with
                    pointer: {data_ptr}. Assuming the trace for tensor deallocation
                    wasn't caught and backfilling it now.
                    No action is necessary.
                    """
                )
            )
            self.delete_tensor(data_ptr)

    def create_tensor(
        self, data_ptr: DataPtr, stack_trace: Optional[List[str]]
    ) -> None:
        self.accesses[data_ptr] = TensorInfo(stack_trace)

    def delete_tensor(self, data_ptr: DataPtr) -> None:
        del self.accesses[data_ptr]

    def were_there_reads_since_last_write(self, data_ptr: DataPtr) -> bool:
        return True if self.accesses[data_ptr].reads else False

    def get_allocation_trace(self, data_ptr: DataPtr) -> Optional[List[str]]:
        return self.accesses[data_ptr].allocation_trace

    def get_write(self, data_ptr: DataPtr) -> Optional[Access]:
        return self.accesses[data_ptr].write

    def get_reads(self, data_ptr: DataPtr) -> List[Access]:
        return self.accesses[data_ptr].reads

    def add_read(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].reads.append(access)

    def set_write(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].write = access
        self.accesses[data_ptr].reads = []


class StreamSynchronizations:
    def __init__(self):
        self.current_sync_states: Dict[StreamId, Dict[StreamId, SeqNum]] = {}
        self.recorded_sync_states: Dict[EventId, Dict[StreamId, SeqNum]] = {}

    def _ensure_stream_exists(self, stream: StreamId) -> None:
        if stream not in self.current_sync_states:
            logger.info(
                format_log_message(
                    f"""
                    Found Stream with id: {stream}, but no matching stream
                    creation in the trace. Backfilling the trace now.
                    No action is necessary.
                    """
                )
            )
            self.create_stream(stream)

    def _ensure_event_exists(self, event: EventId) -> None:
        if event not in self.recorded_sync_states:
            logger.info(
                format_log_message(
                    f"""
                    Found Event with id: {event}, but no matching event
                    creation in the trace. Backfilling the trace now.
                    No action is necessary.
                    """
                )
            )
            self.create_event(event)

    def _ensure_event_does_not_exist(self, event: EventId) -> None:
        if event in self.recorded_sync_states:
            logger.info(
                format_log_message(
                    f"""
                    Found duplicate event creation in the trace for event with
                    id: {event}. Assuming the trace for event deletion wasn't caught
                    and backfilling it now. No action is necessary.
                    """
                )
            )
            self.delete_event(event)

    def create_stream(self, stream: StreamId) -> None:
        if stream in self.current_sync_states:
            logger.info(
                format_log_message(
                    f"""
                    Found duplicate Stream creation in the trace for Stream with
                    id: {stream}. PyTorch Streams are only created once, so this
                    trace entry is ignored.
                    """
                )
            )
        else:
            self.current_sync_states[stream] = {}

    def create_event(self, event: EventId) -> None:
        self._ensure_event_does_not_exist(event)
        self.recorded_sync_states[event] = {}

    def delete_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        del self.recorded_sync_states[event]

    def update_seq_num(self, stream: StreamId, seq_num: SeqNum) -> None:
        self._ensure_stream_exists(stream)
        self.current_sync_states[stream][stream] = seq_num

    def record_state(self, event: EventId, stream: StreamId) -> None:
        self._ensure_event_exists(event)
        self._ensure_stream_exists(stream)
        self.recorded_sync_states[event] = self.current_sync_states[stream].copy()

    def state_wait_for_event(self, stream: StreamId, event: EventId) -> None:
        self._ensure_event_exists(event)
        self._ensure_stream_exists(stream)
        for other_stream, seq_num in self.recorded_sync_states[event].items():
            self.current_sync_states[stream][other_stream] = max(
                self.current_sync_states[stream].get(other_stream, -1), seq_num
            )

    def is_ordered_after(
        self, current_stream: StreamId, seq_num: SeqNum, other_stream: StreamId
    ) -> bool:
        self._ensure_stream_exists(current_stream)
        self._ensure_stream_exists(other_stream)
        return seq_num <= self.current_sync_states[current_stream].get(other_stream, -1)


class TraceChecker:
    """Analyzes CSAN trace for synchronization errors.

    Stores information on each stream's synchronizations with other streams as well
    as tensor accesses to determine whether a given kernel launch might cause a
    data race.
    """

    def __init__(self):
        self.tensors_accessed = _TensorsAccessed()
        self.syncs = StreamSynchronizations()
        self.seq_num: SeqNum = 0

        torch._C._activate_cuda_trace()
        cuda_trace.register_callback_for_cuda_event_creation(
            self._handle_event_creation
        )
        cuda_trace.register_callback_for_cuda_event_deletion(
            self._handle_event_deletion
        )
        cuda_trace.register_callback_for_cuda_event_record(self._handle_event_record)
        cuda_trace.register_callback_for_cuda_event_wait(self._handle_event_wait)
        cuda_trace.register_callback_for_cuda_memory_allocation(
            self._handle_memory_allocation
        )
        cuda_trace.register_callback_for_cuda_memory_deallocation(
            self._handle_memory_deallocation
        )
        cuda_trace.register_callback_for_cuda_stream_creation(
            self._handle_stream_creation
        )

    def _handle_kernel_launch(
        self,
        stream: StreamId,
        read_only: List[DataPtr],
        read_write: List[DataPtr],
        operator: str,
        tensor_names: Dict[int, List[str]],
    ) -> List[SynchronizationError]:
        def check_conflict(
            data_ptr: DataPtr, current_access: Access, previous_access: Optional[Access]
        ) -> None:
            if previous_access is None:
                return
            if not self.syncs.is_ordered_after(
                current_access.stream, previous_access.seq_num, previous_access.stream
            ):
                error_list.append(
                    UnsynchronizedAccessError(
                        data_ptr,
                        self.tensors_accessed.get_allocation_trace(data_ptr),
                        current_access,
                        previous_access,
                    )
                )

        error_list: List[SynchronizationError] = []
        self.seq_num += 1
        self.syncs.update_seq_num(stream, self.seq_num)
        stack_trace = traceback.format_stack()[: _PYTHON_DISPATCH_TRACE_OFFSET - 1]

        for data_ptr in read_only:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.READ,
                self.seq_num,
                stream,
                operator,
                tensor_names[data_ptr],
                stack_trace,
            )
            check_conflict(
                data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
            )
            self.tensors_accessed.add_read(data_ptr, current_access)

        for data_ptr in read_write:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.WRITE,
                self.seq_num,
                stream,
                operator,
                tensor_names[data_ptr],
                stack_trace,
            )
            if self.tensors_accessed.were_there_reads_since_last_write(data_ptr):
                for previous_access in self.tensors_accessed.get_reads(data_ptr):
                    check_conflict(data_ptr, current_access, previous_access)
            else:
                check_conflict(
                    data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
                )
            self.tensors_accessed.set_write(data_ptr, current_access)

        return error_list

    def _handle_event_creation(self, event: EventId) -> None:
        self.syncs.create_event(event)

    def _handle_event_deletion(self, event: EventId) -> None:
        self.syncs.delete_event(event)

    def _handle_event_record(self, event: EventId, stream: StreamId) -> None:
        self.syncs.record_state(event, stream)

    def _handle_event_wait(self, event: EventId, stream: StreamId) -> None:
        self.syncs.state_wait_for_event(stream, event)

    def _handle_memory_allocation(self, data_ptr: DataPtr) -> None:
        self.tensors_accessed.ensure_tensor_does_not_exist(data_ptr)
        trace_offset = _PYTHON_DISPATCH_TRACE_OFFSET + _CUDA_HOOK_TRACE_OFFSET - 1
        self.tensors_accessed.create_tensor(
            data_ptr, traceback.format_stack()[:trace_offset]
        )

    def _handle_memory_deallocation(self, data_ptr: DataPtr) -> None:
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        self.tensors_accessed.delete_tensor(data_ptr)

    def _handle_stream_creation(self, stream: StreamId) -> None:
        self.syncs.create_stream(stream)
