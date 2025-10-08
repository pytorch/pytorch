"""
Lookup table recorder system for capturing autotuning results.

This module provides a system to record and emit autotuning results from kernel selection.
It supports both immediate emission (logging) and table recording (building lookup tables).
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from torch._inductor import config
from torch._inductor.ir import ChoiceCaller
from torch._inductor.kernel_inputs import KernelInputs
from torch._inductor.kernel_template_choice import KernelTemplateChoice
from torch.utils._ordered_set import OrderedSet

from .choices import LookupTableChoices


log = logging.getLogger(__name__)


@dataclass
class LookupTableEntry:
    """Single entry representing one autotuning result"""

    key: str  # device_key+op_name+input_key
    value: dict[str, Any]  # Contains template_id and all kwargs
    metadata: dict[str, Any]  # Contains timing, rank, and other recording metadata
    runtime: (
        float  # Contains the unique timing information. This is used to record topk
    )

    @classmethod
    def from_ktc_and_timing(
        cls,
        ktc: KernelTemplateChoice,
        timing: float,
        rank: int,
        op_name: str,
    ) -> Optional["LookupTableEntry"]:
        """Create entry from a KTC and its timing"""
        # KTC must have a template - this is a requirement
        assert ktc.template is not None, "KernelTemplateChoice must have a template"

        # Use V.choices_handler to make lookup key if it's a LookupTableChoices instance
        key = _make_lookup_key(
            ktc.inputs, op_name, config.lookup_table.record_with_device_key
        )
        if key is None:
            return None

        # Build value dict from KTC kwargs
        value = dict(template_id=ktc.template.uid, **ktc.params.to_serializeable_dict())

        # Add template hash if available and configured
        if config.lookup_table.record_template_hash:
            # Use src_hash directly from the template
            template_hash = getattr(ktc.template, "src_hash", None)
            if template_hash is not None:
                value["template_hash"] = template_hash

        # Create metadata dict with timing and rank info
        metadata = {
            "timing": timing,
            "rank": rank,
        }

        return cls(key=key, value=value, metadata=metadata, runtime=timing)


def _make_lookup_key(
    kernel_inputs: KernelInputs, op_name: str, include_device: bool = False
) -> Optional[str]:
    """Make lookup key using V.choices_handler if available, otherwise use LookupTableChoices static methods"""
    from torch._inductor.virtualized import V

    if hasattr(V, "choices_handler") and isinstance(
        V.choices_handler, LookupTableChoices
    ):
        return V.choices_handler.make_lookup_key(kernel_inputs, op_name, include_device)
    else:
        # Fallback: create a temporary LookupTableChoices instance to use its methods
        choices_handler = LookupTableChoices()
        return choices_handler.make_lookup_key(kernel_inputs, op_name, include_device)


class Backend:
    """Base class for backends"""

    def clear(self) -> None:
        pass


class EmitBackend(ABC, Backend):
    """Backend for immediate emission of single entries"""

    @abstractmethod
    def emit(self, entry: LookupTableEntry) -> None:
        pass


class RecordBackend(ABC, Backend):
    """Backend for dumping recorded table"""

    @abstractmethod
    def dump(self, data: dict[str, list[LookupTableEntry]]) -> None:
        pass


# Track registered backends to avoid double registration
_registered_backends: OrderedSet[Any] = OrderedSet()


def _backend_key(
    backend_class: type, kwargs: dict[str, Any]
) -> tuple[type, frozenset[Any]]:
    """Create a unique key for backend class + kwargs"""
    return (backend_class, frozenset(kwargs.items()) if kwargs else frozenset())


class LogEmitBackend(EmitBackend):
    """Default emit backend that logs entries"""

    def emit(self, entry: LookupTableEntry) -> None:
        log.debug("LookupTable: %r -> %r", entry.key, entry.value)


class DirectoryRecordBackend(RecordBackend):
    """Default record backend that saves to timestamped files in a directory"""

    def __init__(self, directory: str):
        self.directory = directory
        # Generate timestamped filename with 3-digit millisecond precision
        self.setup()

    def setup(self) -> None:
        now = datetime.now(tz=timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
        filename = f"inductor_lut_{timestamp}.json"
        self.filepath = os.path.join(self.directory, filename)

    def dump(self, data: dict[str, list[LookupTableEntry]]) -> None:
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)

        # extract only the value from the entries and dump those
        data_values = {}
        for k, entries in data.items():
            data_values[k] = [e.value for e in entries]
        # just override it again
        with open(self.filepath, "w") as f:
            json.dump(data_values, f, indent=2)

    def clear(self) -> None:
        # generate a new path
        self.setup()


class LookupTableRecorder:
    """Main recorder that manages both emit and record backends"""

    def __init__(self, topk: Optional[int] = None) -> None:
        self.data: dict[str, list[LookupTableEntry]] = {}
        self.emit_backends: list[EmitBackend] = []
        self.record_backends: list[RecordBackend] = []
        # Use provided topk or fall back to config
        self.topk = topk if topk is not None else config.lookup_table.recorder_topk

    @property
    def input_entries(self) -> int:
        """how many unique input entries have been recorded"""
        return len(self.data)

    def add_backend(self, backend: Backend) -> None:
        """Add a backend to the appropriate list based on its type"""
        if isinstance(backend, EmitBackend):
            self.emit_backends.append(backend)
        elif isinstance(backend, RecordBackend):
            self.record_backends.append(backend)
        else:
            raise ValueError(
                f"Backend must be an instance of EmitBackend or RecordBackend, "
                f"got {type(backend).__name__}"
            )

    def emit(self, entry: LookupTableEntry) -> None:
        """Emit a single entry immediately"""
        for backend in self.emit_backends:
            backend.emit(entry)

    def record(self, entry: LookupTableEntry) -> None:
        """Record entry to table and emit it"""
        # Always emit when recording
        self.emit(entry)
        # Initialize key if not exists
        if entry.key not in self.data:
            self.data[entry.key] = []

        # just insert and sort, it's a small topk usually
        # not worth doing bisection
        entries_for_key = self.data[entry.key]
        entries_for_key.append(entry)
        entries_for_key.sort(
            key=lambda x: x.runtime if x.runtime is not None else float("inf")
        )

        # Trim to topk if necessary (only if topk is positive)
        if self.topk is not None and self.topk > 0 and len(entries_for_key) > self.topk:
            topk: int = self.topk
            # Log which entries we're replacing
            replaced_entries = entries_for_key[topk:]
            log.info(
                "Replacing %d entries with new entry (value: %r, runtime: %f) due to topk=%d. "
                "Replaced entries: %s",
                len(replaced_entries),
                entry.value,
                entry.runtime,
                self.topk,
                [{"value": e.value, "runtime": e.runtime} for e in replaced_entries],
            )
            del entries_for_key[topk:]

    def dump(self) -> None:
        """Dump via all record backends"""
        for backend in self.record_backends:
            backend.dump(self.data)

    def clear(self) -> None:
        self.data.clear()
        for backend in self.emit_backends + self.record_backends:
            backend.clear()


# Module-wide instance
_lookup_table_recorder: Optional[LookupTableRecorder] = None


def get_lookup_table_recorder() -> LookupTableRecorder:
    """Get the global lookup table recorder"""
    global _lookup_table_recorder
    if _lookup_table_recorder is None:
        _lookup_table_recorder = LookupTableRecorder()
    # Always register any pending backends
    _register_pending_backends(_lookup_table_recorder)
    assert _lookup_table_recorder is not None
    return _lookup_table_recorder


def add_backend(backend: Backend) -> None:
    """Add a backend to the global lookup table recorder"""
    recorder = get_lookup_table_recorder()
    if recorder is not None:
        recorder.add_backend(backend)


def _register_pending_backends(recorder: LookupTableRecorder) -> None:
    """Register built-in backends based on current config"""
    global _registered_backends

    # Add built-in LogEmitBackend if enabled and not already registered
    if config.lookup_table.recorder_emit:
        emit_key = _backend_key(LogEmitBackend, {})
        if emit_key not in _registered_backends:
            try:
                recorder.add_backend(LogEmitBackend())
                _registered_backends.add(emit_key)
                log.debug("Registered LogEmitBackend")
            except Exception as e:
                log.warning("Failed to register LogEmitBackend: %r", e)

    # Add built-in DirectoryRecordBackend if enabled and not already registered
    record_dir = config.lookup_table.recorder_record_dir
    if record_dir:
        record_key = _backend_key(DirectoryRecordBackend, {"directory": record_dir})
        if record_key not in _registered_backends:
            try:
                recorder.add_backend(DirectoryRecordBackend(record_dir))
                _registered_backends.add(record_key)
                log.debug("Registered DirectoryRecordBackend: %s", record_dir)
            except Exception as e:
                log.warning(
                    "Failed to register DirectoryRecordBackend %s: %r", record_dir, e
                )


def record_topk_choices(
    timings: dict[ChoiceCaller, float],
    op_name: str,
    input_nodes: list[Any],
    choices: list[ChoiceCaller],
    profiled_time_fn: Callable[[], dict[Any, Any]],
) -> None:
    """
    Feedback function to record topk choices based on timing results.

    Args:
        timings: Mapping from choices to benchmark times
        op_name: Name of the operation (e.g. "mm", "addmm")
        input_nodes: List of input ir.Nodes
        choices: List of ChoiceCaller objects
        profiled_time_fn: Function to get profiled times (unused in this implementation)
    """
    # Fast bail if recording not active
    if not config.lookup_table.recording_active:
        log.debug(
            "Recording disabled (recording_active=False) for operation %s", op_name
        )
        return

    # If topk is 0, don't record anything
    if config.lookup_table.recorder_topk == 0:
        log.debug("Recording disabled (topk=0) for operation %s", op_name)
        return

    # Get recorder
    recorder = get_lookup_table_recorder()
    if recorder is None:
        log.warning("Failed to get lookup table recorder for operation %s", op_name)
        return
    # adjust the recorder topk if necessary
    recorder.topk = config.lookup_table.recorder_topk

    # Filter choices that have valid timings and KTC references
    valid_choices = []
    filtered_count = 0

    for choice in choices:
        if (
            choice not in timings
            or not hasattr(choice, "annotations")
            or "ktc" not in choice.annotations
            or timings[choice] == float("inf")
        ):
            filtered_count += 1
        else:
            valid_choices.append(choice)

    if filtered_count > 0:
        log.debug(
            "Recording %s: filtered %d/%d invalid choices",
            op_name,
            filtered_count,
            len(choices),
        )

    if not valid_choices:
        log.debug("Recording %s: no valid choices", op_name)
        return

    # Sort and trim to topk
    sorted_choices = sorted(valid_choices, key=lambda c: timings[c])
    if recorder.topk and recorder.topk > 0 and len(sorted_choices) > recorder.topk:
        sorted_choices = sorted_choices[: recorder.topk]

    # Record each choice
    recorded_count = 0
    for rank, choice in enumerate(sorted_choices):
        ktc = choice.annotations["ktc"]
        if ktc is None:
            log.warning(
                "Recording %s: KTC is None for choice %s",
                op_name,
                getattr(choice, "name", choice),
            )
            continue

        entry = LookupTableEntry.from_ktc_and_timing(
            ktc=ktc, timing=timings[choice], rank=rank, op_name=op_name
        )
        if entry is not None:
            recorder.record(entry)
            recorded_count += 1

    log.info(
        "Recording %s: saved %d/%d entries",
        op_name,
        recorded_count,
        len(sorted_choices),
    )
    # Any time we record the table, we continue to dump
    # The backends need to be implemented so that they can
    # accommodate progressive dumping
    recorder.dump()


def dump() -> None:
    """Dump the global lookup table recorder"""
    recorder = get_lookup_table_recorder()
    if recorder is not None:
        recorder.dump()


def clear() -> None:
    """Clear the global lookup table recorder"""
    recorder = get_lookup_table_recorder()
    if recorder is not None:
        recorder.clear()


# Auto-register the feedback function when the module is imported
try:
    from torch._inductor.select_algorithm import add_feedback_saver

    add_feedback_saver(record_topk_choices)
    log.debug("Registered lookup table recorder feedback function")
except ImportError:
    log.warning(
        "Failed to register lookup table recorder feedback function - select_algorithm not available"
    )
