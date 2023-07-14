r"""
This module introduces CUDA Sanitizer, a tool for detecting synchronization errors
between kernels ran on different streams. It stores information on accesses to tensors
to determine if they are synchronized or not. When enabled in a python program and a
possible data race is detected, a detailed warning will be printed and the program
will exit.

It can be enabled either by importing this module and calling
:func:`enable_cuda_sanitizer()` or by exporting the ``TORCH_CUDA_SANITIZER``
environment variable.
"""

import enum
import functools
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar

import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


DEFAULT_STREAM_ID = 0

TK = TypeVar("TK")
TVa = TypeVar("TVa")
TVb = TypeVar("TVb")

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
    r"""Stores information about a single access to a tensor by a kernel.

    Args:
        type: either AccessType.READ or AccessType.Write.
        seq_num: the sequential number of the kernel performing the access.
        stream: the stream id of the stream executing the kernel.
        operator: the schema of the launched kernel, which lists the
            arguments and return type.
        aliases: the arguments in the schema this access corresponds to.
        is_output: Whether the tensor was an output of the kernel.
        stack_trace: the stack summary object captured during access.
    """
    type: AccessType
    seq_num: SeqNum
    stream: StreamId
    operator: str
    aliases: List[str]
    is_output: bool
    stack_trace: traceback.StackSummary


class SynchronizationError(Exception):
    """Base class for errors detected by CUDA Sanitizer."""

    pass


class UnsynchronizedAccessError(SynchronizationError):
    """Stores information about two unsynchronized accesses to one data pointer."""

    def __init__(
        self,
        data_ptr: DataPtr,
        allocation_stack_trace: Optional[traceback.StackSummary],
        current_access: Access,
        previous_access: Access,
    ):
        self.data_ptr = data_ptr
        self.allocation_stack_trace = allocation_stack_trace
        self.current_access = current_access
        self.previous_access = previous_access

    def __str__(self):
        def format_access(access: Access):
            message.write(f"{access.operator}\n{access.type}")
            if access.aliases:
                message.write(" argument(s) " + ", ".join(access.aliases))
                if access.is_output:
                    message.write(", and to")
            if access.is_output:
                message.write(" the output")
            message.write(
                f"\nWith stack trace:\n{''.join(access.stack_trace.format())}\n"
            )

        with io.StringIO() as message:
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race on tensor with data pointer {self.data_ptr}
                    Access by stream {self.current_access.stream} during kernel:
                    """
                )
            )
            format_access(self.current_access)

            message.write(
                f"Previous access by stream {self.previous_access.stream} during kernel:\n"
            )
            format_access(self.previous_access)

            if self.allocation_stack_trace:
                message.write(
                    "Tensor was allocated with stack trace:\n"
                    f"{''.join(self.allocation_stack_trace.format())}"
                )
            else:
                message.write("Trace for tensor allocation not found.")
            return message.getvalue()


class CUDASanitizerErrors(Exception):
    """Wrapper class for errors reported by CUDA Sanitizer."""

    def __init__(self, errors: List[SynchronizationError]):
        self.errors = errors

    def __str__(self):
        return f"detected {len(self.errors)} errors"


@dataclass
class TensorInfo:
    r"""Stores information about a single tensor and recent accesses to it.

    Args:
        allocation_stack_trace: the stack summary object captured during tensor
            allocation. Can be ``None`` if the allocation wasn't caught by CSAN.
        reads: list of read accesses to the tensor that were performed since
            the last write.
        write: the last write access to the tensor.
    """
    allocation_stack_trace: Optional[traceback.StackSummary]
    reads: List[Access] = field(default_factory=list)
    write: Optional[Access] = None


class _TensorsAccessed:
    def __init__(self):
        self.accesses: Dict[DataPtr, TensorInfo] = {}

    def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
        if data_ptr not in self.accesses:
            logger.info(
                "Found tensor with pointer: %s, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr
            )
            self.create_tensor(data_ptr, None)

    def ensure_tensor_does_not_exist(self, data_ptr: DataPtr) -> None:
        if data_ptr in self.accesses:
            logger.info(
                "Found duplicate tensor allocation in the trace for tensor with "
                "pointer: %s. Assuming the trace for tensor deallocation "
                "wasn't caught and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr
            )
            self.delete_tensor(data_ptr)

    def create_tensor(
        self, data_ptr: DataPtr, stack_trace: Optional[traceback.StackSummary]
    ) -> None:
        self.accesses[data_ptr] = TensorInfo(stack_trace)

    def delete_tensor(self, data_ptr: DataPtr) -> None:
        del self.accesses[data_ptr]

    def were_there_reads_since_last_write(self, data_ptr: DataPtr) -> bool:
        return True if self.accesses[data_ptr].reads else False

    def get_allocation_stack_trace(
        self, data_ptr: DataPtr
    ) -> Optional[traceback.StackSummary]:
        return self.accesses[data_ptr].allocation_stack_trace

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
        self.host_sync_state: Dict[StreamId, SeqNum] = {}
        self.create_stream(DEFAULT_STREAM_ID)

    def _ensure_stream_exists(self, stream: StreamId) -> None:
        if stream not in self.current_sync_states:
            logger.info(
                "Found Stream with id: %s, but no matching stream "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                stream
            )
            self.create_stream(stream)

    def _ensure_event_exists(self, event: EventId) -> None:
        if event not in self.recorded_sync_states:
            logger.info(
                "Found Event with id: %s, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event
            )
            self.create_event(event)

    def _ensure_event_does_not_exist(self, event: EventId) -> None:
        if event in self.recorded_sync_states:
            logger.info(
                "Found duplicate event creation in the trace for event with "
                "id: %s. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event
            )
            self.delete_event(event)

    def create_stream(self, stream: StreamId) -> None:
        if stream in self.current_sync_states:
            logger.info(
                "Found duplicate Stream creation in the trace for Stream with "
                "id: %s. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
                stream
            )
        else:
            self.host_sync_state[stream] = 0
            self.current_sync_states[stream] = self.host_sync_state.copy()

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

    def _state_wait_for_other(
        self, state: Dict[StreamId, SeqNum], other: Dict[StreamId, SeqNum]
    ) -> None:
        for stream, seq_num in other.items():
            state[stream] = max(state.get(stream, -1), seq_num)

    def stream_wait_for_event(self, stream: StreamId, event: EventId) -> None:
        self._ensure_stream_exists(stream)
        self._ensure_event_exists(event)
        self._state_wait_for_other(
            self.current_sync_states[stream], self.recorded_sync_states[event]
        )

    def all_streams_wait_for_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        for stream in self.current_sync_states.keys():
            self.stream_wait_for_event(stream, event)

        self._state_wait_for_other(
            self.host_sync_state, self.recorded_sync_states[event]
        )

    def all_streams_wait_for_stream(self, stream: StreamId) -> None:
        self._ensure_stream_exists(stream)
        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.current_sync_states[stream])

        self._state_wait_for_other(
            self.host_sync_state, self.current_sync_states[stream]
        )

    def sync_all_streams(self) -> None:
        for stream, state in self.current_sync_states.items():
            self.host_sync_state[stream] = state[stream]

        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.host_sync_state)

    def is_ordered_after(
        self, current_stream: StreamId, seq_num: SeqNum, other_stream: StreamId
    ) -> bool:
        self._ensure_stream_exists(current_stream)
        self._ensure_stream_exists(other_stream)
        return seq_num <= self.current_sync_states[current_stream].get(other_stream, -1)


class EventHandler:
    """Analyzes CSAN trace for synchronization errors.

    Stores information on each stream's synchronizations with other streams as well
    as tensor accesses to determine whether a given kernel launch might cause a
    data race.
    """

    def __init__(self):
        self.tensors_accessed = _TensorsAccessed()
        self.syncs = StreamSynchronizations()
        self.seq_num: SeqNum = 0

    def _handle_kernel_launch(
        self,
        stream: StreamId,
        read_only: Set[DataPtr],
        read_write: Set[DataPtr],
        outputs: Set[DataPtr],
        operator: str,
        tensor_aliases: Dict[int, List[str]],
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
                        self.tensors_accessed.get_allocation_stack_trace(data_ptr),
                        current_access,
                        previous_access,
                    )
                )

        error_list: List[SynchronizationError] = []
        self.seq_num += 1
        self.syncs.update_seq_num(stream, self.seq_num)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(None), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace.reverse()

        for data_ptr in read_only:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.READ,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
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
                tensor_aliases[data_ptr],
                data_ptr in outputs,
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
        self.syncs.stream_wait_for_event(stream, event)

    def _handle_memory_allocation(self, data_ptr: DataPtr) -> None:
        self.tensors_accessed.ensure_tensor_does_not_exist(data_ptr)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(None), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace.reverse()
        self.tensors_accessed.create_tensor(
            data_ptr,
            stack_trace,
        )

    def _handle_memory_deallocation(self, data_ptr: DataPtr) -> None:
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        self.tensors_accessed.delete_tensor(data_ptr)

    def _handle_stream_creation(self, stream: StreamId) -> None:
        self.syncs.create_stream(stream)

    def _handle_device_synchronization(self) -> None:
        self.syncs.sync_all_streams()

    def _handle_stream_synchronization(self, stream: StreamId) -> None:
        self.syncs.all_streams_wait_for_stream(stream)

    def _handle_event_synchronization(self, event: EventId) -> None:
        self.syncs.all_streams_wait_for_event(event)


def zip_by_key(a: Dict[TK, TVa], b: Dict[TK, TVb]) -> Iterator[Tuple[TK, TVa, TVb]]:
    for arg, value in a.items():
        if arg in b:
            yield arg, value, b[arg]


def zip_arguments(
    schema: torch.FunctionSchema, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Iterator[Tuple[torch.Argument, Any]]:
    schema_args = schema.arguments[: len(args)]
    schema_kwargs = {arg.name: arg for arg in schema.arguments[len(args) :]}

    yield from zip(schema_args, args)

    for _, argument, value in zip_by_key(schema_kwargs, kwargs):
        yield (argument, value)


class ArgumentHandler:
    def __init__(self):
        self.dataptrs_read: Set[DataPtr] = set()
        self.dataptrs_written: Set[DataPtr] = set()
        self.tensor_aliases: Dict[DataPtr, List[str]] = dict()
        self.outputs: Set[DataPtr] = set()

    def _handle_argument(
        self,
        value: Any,
        is_write: bool,
        name: Optional[str] = None,
        is_output: bool = False,
    ) -> None:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            data_ptr = value.data_ptr()
            if is_write:
                self.dataptrs_written.add(data_ptr)
            else:
                self.dataptrs_read.add(data_ptr)

            self.tensor_aliases.setdefault(data_ptr, [])
            if name is not None:
                self.tensor_aliases[data_ptr].append(name)
            if is_output:
                self.outputs.add(data_ptr)

    def parse_inputs(
        self,
        schema: torch.FunctionSchema,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        for argument, value in zip_arguments(schema, args, kwargs):
            is_write = argument.alias_info is not None and argument.alias_info.is_write
            tree_map(
                functools.partial(
                    self._handle_argument, is_write=is_write, name=argument.name
                ),
                value,
            )

    def parse_outputs(self, outputs: Any) -> None:
        tree_map(
            functools.partial(self._handle_argument, is_write=True, is_output=True),
            outputs,
        )


class CUDASanitizerDispatchMode(TorchDispatchMode):
    def __init__(self):
        self.event_handler = EventHandler()
        torch._C._activate_cuda_trace()
        cuda_trace.register_callback_for_cuda_event_creation(
            self.event_handler._handle_event_creation
        )
        cuda_trace.register_callback_for_cuda_event_deletion(
            self.event_handler._handle_event_deletion
        )
        cuda_trace.register_callback_for_cuda_event_record(
            self.event_handler._handle_event_record
        )
        cuda_trace.register_callback_for_cuda_event_wait(
            self.event_handler._handle_event_wait
        )
        cuda_trace.register_callback_for_cuda_memory_allocation(
            self.event_handler._handle_memory_allocation
        )
        cuda_trace.register_callback_for_cuda_memory_deallocation(
            self.event_handler._handle_memory_deallocation
        )
        cuda_trace.register_callback_for_cuda_stream_creation(
            self.event_handler._handle_stream_creation
        )
        cuda_trace.register_callback_for_cuda_device_synchronization(
            self.event_handler._handle_device_synchronization
        )
        cuda_trace.register_callback_for_cuda_stream_synchronization(
            self.event_handler._handle_stream_synchronization
        )
        cuda_trace.register_callback_for_cuda_event_synchronization(
            self.event_handler._handle_event_synchronization
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        argument_handler = ArgumentHandler()
        argument_handler.parse_inputs(func._schema, args, kwargs)

        outputs = func(*args, **kwargs)

        argument_handler.parse_outputs(outputs)
        errors = self.event_handler._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            argument_handler.dataptrs_read - argument_handler.dataptrs_written,
            argument_handler.dataptrs_written,
            argument_handler.outputs,
            func._schema,
            argument_handler.tensor_aliases,
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise CUDASanitizerErrors(errors)

        return outputs


class CUDASanitizer:
    """Manages the lifetime of a CUDASanitizer dispatch mode object.

    The CUDASanitizer class wraps the entering/exiting functions of the dispatch mode
    context manager in the enable function/destructor, respectively. This is to
    explicitly set the lifetime of the dispatch mode object to that of the application.
    This approach was deemed more elegant than using the atexit module.
    """

    def __init__(self):
        self.dispatch = CUDASanitizerDispatchMode()
        self.enabled = False

    def enable(self):
        self.dispatch.__enter__()
        self.enabled = True

    def __del__(self):
        if self.enabled:
            self.dispatch.__exit__(None, None, None)


def enable_cuda_sanitizer():
    """Enables CUDA Sanitizer.

    The sanitizer will begin to analyze low-level CUDA calls invoked by torch functions
    for synchronization errors. All data races found will be printed to the standard
    error output along with stack traces of suspected causes. For best results, the
    sanitizer should be enabled at the very beginning of the program.
    """
    cuda_sanitizer.enable()


cuda_sanitizer = CUDASanitizer()
