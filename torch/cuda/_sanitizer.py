# mypy: allow-untyped-defs
r"""
This module introduces CUDA Sanitizer, a tool for detecting synchronization errors between kernels ran on different streams.

It stores information on accesses to tensors to determine if they are synchronized
or not. When enabled in a python program and a possible data race is detected, a
detailed warning will be printed and the program will exit.

It can be enabled either by importing this module and calling
:func:`enable_cuda_sanitizer()` or by exporting the ``TORCH_CUDA_SANITIZER``
environment variable.
"""

import enum
import functools
import inspect
import io
import logging
import re
import sys
import textwrap
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
import torch.cuda._gpu_trace as gpu_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

DEFAULT_STREAM_ID = 0

TK = TypeVar("TK")
TVa = TypeVar("TVa")
TVb = TypeVar("TVb")

DataPtr = int
StreamId = int
EventId = int
SeqNum = int

logger = logging.getLogger(__name__)

# Note that this is only factories that take Tensor as input as they are
# the ones we care about.
FACTORY_FUNCTION_REGEX = re.compile("(new_.*|.*_like)")


class AccessType(enum.Enum):
    READ = enum.auto()
    WRITE = enum.auto()

    def __str__(self):
        return "reading from" if self is AccessType.READ else "writing to"


@dataclass(slots=True)
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
    aliases: list[str]
    is_output: bool
    stack_trace: traceback.StackSummary


class SynchronizationError(Exception):
    """Base class for errors detected by CUDA Sanitizer."""


class UnsynchronizedAccessError(SynchronizationError):
    """Stores information about two unsynchronized accesses to one data pointer."""

    def __init__(
        self,
        data_ptr: DataPtr,
        allocation_stack_trace: traceback.StackSummary | None,
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


class AllocatorReuseError(SynchronizationError):
    """Detected when the caching allocator reuses memory without proper stream sync.

    This occurs when a tensor is freed (refcount drops to zero, returning memory
    to the caching allocator's free pool) while a stream may still be accessing
    that memory, and the allocator then hands the same memory block to a new
    allocation on a different stream without synchronization.
    """

    def __init__(
        self,
        data_ptr: DataPtr,
        alloc_stream: StreamId,
        previous_accesses: list[tuple[StreamId, SeqNum]],
        dealloc_stack_trace: traceback.StackSummary | None,
        alloc_stack_trace: traceback.StackSummary | None,
    ):
        self.data_ptr = data_ptr
        self.alloc_stream = alloc_stream
        self.previous_accesses = previous_accesses
        self.dealloc_stack_trace = dealloc_stack_trace
        self.alloc_stack_trace = alloc_stack_trace

    def __str__(self):
        with io.StringIO() as message:
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race from caching allocator memory reuse
                    Memory at data pointer {self.data_ptr}
                    New allocation on stream {self.alloc_stream}, but previous access(es) on
                    unsynchronized stream(s) may still be using this memory:
                    """
                )
            )
            for stream, seq_num in self.previous_accesses:
                message.write(f"  stream {stream} (seq_num {seq_num})\n")

            if self.dealloc_stack_trace:
                message.write(
                    "\nMemory was freed with stack trace:\n"
                    f"{''.join(self.dealloc_stack_trace.format())}\n"
                )

            if self.alloc_stack_trace:
                message.write(
                    "Memory was re-allocated with stack trace:\n"
                    f"{''.join(self.alloc_stack_trace.format())}"
                )
            return message.getvalue()


class CUDASanitizerErrors(Exception):
    """Wrapper class for errors reported by CUDA Sanitizer."""

    def __init__(self, errors: list[SynchronizationError]):
        self.errors = errors

    def __str__(self):
        return f"detected {len(self.errors)} errors"


@dataclass(slots=True)
class TensorInfo:
    r"""Stores information about a single tensor and recent accesses to it.

    Args:
        allocation_stack_trace: the stack summary object captured during tensor
            allocation. Can be ``None`` if the allocation wasn't caught by CSAN.
        reads: list of read accesses to the tensor that were performed since
            the last write.
        write: the last write access to the tensor.
    """

    allocation_stack_trace: traceback.StackSummary | None
    reads: list[Access] = field(default_factory=list)
    write: Access | None = None


@dataclass(slots=True)
class PendingReuse:
    r"""Tracks a freed memory block awaiting potential reuse by the caching allocator.

    When a tensor is deallocated, its access history is condensed into this
    record.  If the caching allocator later hands the same data pointer to a
    new allocation, CSAN checks whether the new stream has a happens-before
    relationship with every stream recorded here.

    Args:
        stream_seq_nums: mapping from stream id to the latest seq_num that
            accessed this memory before it was freed.
        dealloc_stream: the stream on which the deallocation occurred.
        dealloc_stack_trace: stack trace captured at deallocation time.
    """

    stream_seq_nums: dict[StreamId, SeqNum]
    dealloc_stream: StreamId
    dealloc_stack_trace: traceback.StackSummary | None


class _TensorsAccessed:
    def __init__(self) -> None:
        self.accesses: dict[DataPtr, TensorInfo] = {}

    def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
        if data_ptr not in self.accesses:
            logger.info(
                "Found tensor with pointer: %s, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.create_tensor(data_ptr, None)

    def ensure_tensor_does_not_exist(self, data_ptr: DataPtr) -> None:
        if data_ptr in self.accesses:
            logger.info(
                "Found duplicate tensor allocation in the trace for tensor with "
                "pointer: %s. Assuming the trace for tensor deallocation "
                "wasn't caught and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.delete_tensor(data_ptr)

    def create_tensor(
        self, data_ptr: DataPtr, stack_trace: traceback.StackSummary | None
    ) -> None:
        self.accesses[data_ptr] = TensorInfo(stack_trace)

    def delete_tensor(self, data_ptr: DataPtr) -> None:
        del self.accesses[data_ptr]

    def were_there_reads_since_last_write(self, data_ptr: DataPtr) -> bool:
        return bool(self.accesses[data_ptr].reads)

    def get_allocation_stack_trace(
        self, data_ptr: DataPtr
    ) -> traceback.StackSummary | None:
        return self.accesses[data_ptr].allocation_stack_trace

    def get_write(self, data_ptr: DataPtr) -> Access | None:
        return self.accesses[data_ptr].write

    def get_reads(self, data_ptr: DataPtr) -> list[Access]:
        return self.accesses[data_ptr].reads

    def add_read(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].reads.append(access)

    def set_write(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].write = access
        self.accesses[data_ptr].reads = []


class StreamSynchronizations:
    def __init__(self) -> None:
        self.current_sync_states: dict[StreamId, dict[StreamId, SeqNum]] = {}
        self.recorded_sync_states: dict[EventId, dict[StreamId, SeqNum]] = {}
        self.host_sync_state: dict[StreamId, SeqNum] = {}
        self.create_stream(DEFAULT_STREAM_ID)

    def _ensure_stream_exists(self, stream: StreamId) -> None:
        if stream not in self.current_sync_states:
            logger.info(
                "Found Stream with id: %s, but no matching stream "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                stream,
            )
            self.create_stream(stream)

    def _ensure_event_exists(self, event: EventId) -> None:
        if event not in self.recorded_sync_states:
            logger.info(
                "Found Event with id: %s, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.create_event(event)

    def _ensure_event_does_not_exist(self, event: EventId) -> None:
        if event in self.recorded_sync_states:
            logger.info(
                "Found duplicate event creation in the trace for event with "
                "id: %s. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.delete_event(event)

    def create_stream(self, stream: StreamId) -> None:
        if stream in self.current_sync_states:
            logger.info(
                "Found duplicate Stream creation in the trace for Stream with "
                "id: %s. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
                stream,
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
        self, state: dict[StreamId, SeqNum], other: dict[StreamId, SeqNum]
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
        for stream in self.current_sync_states:
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

    def __init__(self) -> None:
        self.tensors_accessed = _TensorsAccessed()
        self.syncs = StreamSynchronizations()
        self.seq_num: SeqNum = 0
        self.pending_reuse: dict[DataPtr, PendingReuse] = {}
        # Errors detected in memory callbacks are deferred here because
        # CallbackRegistry.fire_callbacks swallows exceptions.  They are
        # drained and raised in the next __torch_dispatch__ call.
        self.deferred_errors: list[SynchronizationError] = []
        # Streams registered via record_stream() per data pointer.  The
        # caching allocator guarantees it will not reuse memory until all
        # recorded streams have completed, so these streams are safe and
        # should be excluded from PendingReuse checks.
        self.recorded_streams: dict[DataPtr, set[StreamId]] = {}

    def _handle_kernel_launch(
        self,
        stream: StreamId,
        read_only: set[DataPtr],
        read_write: set[DataPtr],
        outputs: set[DataPtr],
        operator: str,
        tensor_aliases: dict[int, list[str]],
    ) -> list[SynchronizationError]:
        def check_conflict(
            data_ptr: DataPtr, current_access: Access, previous_access: Access | None
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

        error_list: list[SynchronizationError] = []
        self.seq_num += 1
        self.syncs.update_seq_num(stream, self.seq_num)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
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

    def _handle_memory_allocation(self, data_ptr: DataPtr, stream: StreamId) -> None:
        # Check for caching allocator reuse races: if this data_ptr was
        # recently freed, verify that the new allocation stream has a
        # happens-before relationship with all streams that previously
        # accessed the memory.
        if data_ptr in self.pending_reuse:
            pending = self.pending_reuse.pop(data_ptr)
            self.syncs._ensure_stream_exists(stream)
            unsynchronized = []
            for old_stream, seq_num in pending.stream_seq_nums.items():
                if not self.syncs.is_ordered_after(stream, seq_num, old_stream):
                    unsynchronized.append((old_stream, seq_num))
            if unsynchronized:
                alloc_stack_trace = traceback.StackSummary.extract(
                    traceback.walk_stack(inspect.currentframe()),
                    lookup_lines=False,
                )
                alloc_stack_trace.reverse()
                error = AllocatorReuseError(
                    data_ptr,
                    stream,
                    unsynchronized,
                    pending.dealloc_stack_trace,
                    alloc_stack_trace,
                )
                # Cannot raise here — this runs inside
                # CallbackRegistry.fire_callbacks which swallows exceptions.
                # Defer the error to be raised on the next __torch_dispatch__.
                self.deferred_errors.append(error)

        self.tensors_accessed.ensure_tensor_does_not_exist(data_ptr)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it
        # must be reversed.
        stack_trace.reverse()
        self.tensors_accessed.create_tensor(
            data_ptr,
            stack_trace,
        )

    def _handle_record_stream(self, data_ptr: DataPtr, stream: StreamId) -> None:
        """Handle tensor.record_stream(stream).

        The caching allocator guarantees it will not reuse a block's memory
        until every stream passed to record_stream has completed.  We track
        these streams so that _handle_memory_deallocation can exclude them
        from PendingReuse — the allocator itself ensures safety for recorded
        streams.
        """
        self.recorded_streams.setdefault(data_ptr, set()).add(stream)

    def _handle_memory_deallocation(self, data_ptr: DataPtr, stream: StreamId) -> None:
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        # Condense the tensor's access history into a PendingReuse record so
        # we can detect races if the caching allocator hands this memory block
        # to a new allocation before all prior streams have finished.
        info = self.tensors_accessed.accesses[data_ptr]
        stream_seq_nums: dict[StreamId, SeqNum] = {}
        if info.write is not None:
            stream_seq_nums[info.write.stream] = info.write.seq_num
        for read in info.reads:
            prev = stream_seq_nums.get(read.stream, -1)
            stream_seq_nums[read.stream] = max(prev, read.seq_num)

        # Exclude streams that were registered via record_stream().  The
        # caching allocator guarantees it will not reuse this memory until
        # those streams have completed, so they cannot race.
        safe_streams = self.recorded_streams.pop(data_ptr, set())
        for s in safe_streams:
            stream_seq_nums.pop(s, None)

        if stream_seq_nums:
            dealloc_stack_trace = traceback.StackSummary.extract(
                traceback.walk_stack(inspect.currentframe()),
                lookup_lines=False,
            )
            dealloc_stack_trace.reverse()
            self.pending_reuse[data_ptr] = PendingReuse(
                stream_seq_nums=stream_seq_nums,
                dealloc_stream=stream,
                dealloc_stack_trace=dealloc_stack_trace,
            )

        self.tensors_accessed.delete_tensor(data_ptr)

    def _handle_stream_creation(self, stream: StreamId) -> None:
        self.syncs.create_stream(stream)

    def _handle_device_synchronization(self) -> None:
        self.syncs.sync_all_streams()
        # After a full device sync, all streams are synchronized with each
        # other, so every pending reuse is safe — prune the entire map.
        self.pending_reuse.clear()

    def _handle_stream_synchronization(self, stream: StreamId) -> None:
        self.syncs.all_streams_wait_for_stream(stream)
        # After synchronizing with a specific stream, prune pending_reuse
        # entries whose only unsynchronized accesses were on that stream.
        to_delete = []
        for data_ptr, pending in self.pending_reuse.items():
            all_synced = True
            for old_stream in pending.stream_seq_nums:
                # Check if all known streams now see old_stream's seq_num.
                # After all_streams_wait_for_stream(stream), every stream's
                # view of `stream` is up-to-date. If old_stream == stream,
                # those accesses are now visible to everyone.
                if old_stream != stream:
                    all_synced = False
                    break
            if all_synced:
                to_delete.append(data_ptr)
        for data_ptr in to_delete:
            del self.pending_reuse[data_ptr]

    def _handle_event_synchronization(self, event: EventId) -> None:
        self.syncs.all_streams_wait_for_event(event)


def zip_by_key(a: dict[TK, TVa], b: dict[TK, TVb]) -> Iterator[tuple[TK, TVa, TVb]]:
    for arg, value in a.items():
        if arg in b:
            yield arg, value, b[arg]


def zip_arguments(
    schema: torch.FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterator[tuple[torch.Argument, Any]]:
    schema_args = schema.arguments[: len(args)]
    schema_kwargs = {arg.name: arg for arg in schema.arguments[len(args) :]}

    yield from zip(schema_args, args)

    for _, argument, value in zip_by_key(schema_kwargs, kwargs):
        yield (argument, value)


class ArgumentHandler:
    def __init__(self) -> None:
        self.dataptrs_read: set[DataPtr] = set()
        self.dataptrs_written: set[DataPtr] = set()
        self.tensor_aliases: dict[DataPtr, list[str]] = {}
        self.outputs: set[DataPtr] = set()

    def _handle_argument(
        self,
        value: Any,
        is_write: bool,
        metadata_only: bool,
        name: str | None = None,
        is_output: bool = False,
    ) -> None:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            # data_ptr() is preferred, but distinguish Tensors with null data_ptr()
            # otherwise two empty Tensors could incorrectly match as a conflict
            data_ptr = value.data_ptr() if value.data_ptr() else id(value)
            if is_write:
                self.dataptrs_written.add(data_ptr)
            elif not metadata_only:
                self.dataptrs_read.add(data_ptr)

            self.tensor_aliases.setdefault(data_ptr, [])
            if name is not None:
                self.tensor_aliases[data_ptr].append(name)
            if is_output:
                self.outputs.add(data_ptr)

    def parse_inputs(
        self,
        schema: torch.FunctionSchema,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        is_factory: bool,
    ) -> None:
        for argument, value in zip_arguments(schema, args, kwargs):
            is_write = argument.alias_info is not None and argument.alias_info.is_write
            # A change is metadata only if it is a view or a factory function that
            # reads only metadata
            metadata_only = is_factory or (
                argument.alias_info is not None and not argument.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=is_write,
                    name=argument.name,
                    metadata_only=metadata_only,
                ),
                value,
            )

    def parse_outputs(
        self, schema: torch.FunctionSchema, outputs: Any, *, is_factory: bool
    ) -> None:
        for res, value in zip(schema.returns, (outputs,)):
            metadata_only = is_factory or (
                res.alias_info is not None and not res.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=not metadata_only,
                    is_output=True,
                    metadata_only=metadata_only,
                ),
                value,
            )


class CUDASanitizerDispatchMode(TorchDispatchMode):
    def __init__(self) -> None:
        self.event_handler = EventHandler()
        torch._C._activate_gpu_trace()
        gpu_trace.register_callback_for_event_creation(
            self.event_handler._handle_event_creation
        )
        gpu_trace.register_callback_for_event_deletion(
            self.event_handler._handle_event_deletion
        )
        gpu_trace.register_callback_for_event_record(
            self.event_handler._handle_event_record
        )
        gpu_trace.register_callback_for_event_wait(
            self.event_handler._handle_event_wait
        )
        gpu_trace.register_callback_for_memory_allocation(
            self.event_handler._handle_memory_allocation
        )
        gpu_trace.register_callback_for_memory_deallocation(
            self.event_handler._handle_memory_deallocation
        )
        gpu_trace.register_callback_for_stream_creation(
            self.event_handler._handle_stream_creation
        )
        gpu_trace.register_callback_for_device_synchronization(
            self.event_handler._handle_device_synchronization
        )
        gpu_trace.register_callback_for_stream_synchronization(
            self.event_handler._handle_stream_synchronization
        )
        gpu_trace.register_callback_for_event_synchronization(
            self.event_handler._handle_event_synchronization
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # record_stream tells the caching allocator that a tensor is used on
        # an additional stream.  Forward this to EventHandler so it can
        # exclude the recorded stream from allocator-reuse race checks.
        if func is aten.record_stream.default:
            tensor, stream_arg = args[0], args[1]  # type: ignore[index]
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                data_ptr = tensor.data_ptr() if tensor.data_ptr() else id(tensor)
                # stream_arg is a torch.Stream with (device_type, device_index,
                # stream_id).  Convert to the raw cuda_stream handle that CSAN
                # uses everywhere else for stream identity.
                cuda_stream = torch.cuda.Stream(
                    stream_id=stream_arg.stream_id,
                    device_index=stream_arg.device_index,
                    device_type=stream_arg.device_type,
                ).cuda_stream
                self.event_handler._handle_record_stream(data_ptr, cuda_stream)
            return func(*args, **kwargs)

        is_factory = bool(FACTORY_FUNCTION_REGEX.match(func._schema.name))

        argument_handler = ArgumentHandler()
        argument_handler.parse_inputs(func._schema, args, kwargs, is_factory=is_factory)

        outputs = func(*args, **kwargs)

        argument_handler.parse_outputs(func._schema, outputs, is_factory=is_factory)
        errors = self.event_handler._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            argument_handler.dataptrs_read - argument_handler.dataptrs_written,
            argument_handler.dataptrs_written,
            argument_handler.outputs,
            func._schema,
            argument_handler.tensor_aliases,
        )
        # Drain any errors deferred from memory allocation/deallocation
        # callbacks (which run inside CallbackRegistry.fire_callbacks and
        # cannot propagate exceptions directly).
        if self.event_handler.deferred_errors:
            errors = list(errors) + self.event_handler.deferred_errors
            self.event_handler.deferred_errors.clear()
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

    def __init__(self) -> None:
        self.dispatch = CUDASanitizerDispatchMode()
        self.enabled = False

    def enable(self):
        self.dispatch.__enter__()
        self.enabled = True

    def disable(self):
        self.dispatch.__exit__(None, None, None)
        self.enabled = False

    def __del__(self):
        # Since this object lifetime is linked to the `torch.cuda._sanitizer` python
        # module, it often gets deleted as part of the overall `torch` module cleanup
        # At that time, depending on CPython version, the torch.* module might be in
        # different states of being already cleaned up.
        # Similarly other imports might already have been cleaned up so `sys` might
        # be already gone as well.
        # Skip exiting the mode if it outlived the runtime.
        if (sys is not None) and (not sys.is_finalizing()) and self.enabled:
            self.disable()


def enable_cuda_sanitizer():
    """Enable CUDA Sanitizer.

    The sanitizer will begin to analyze low-level CUDA calls invoked by torch functions
    for synchronization errors. All data races found will be printed to the standard
    error output along with stack traces of suspected causes. For best results, the
    sanitizer should be enabled at the very beginning of the program.
    """
    cuda_sanitizer.enable()


cuda_sanitizer = CUDASanitizer()
