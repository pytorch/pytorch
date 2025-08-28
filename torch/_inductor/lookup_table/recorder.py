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
from typing import Any, Callable, Optional, Union

from torch._inductor.ir import ChoiceCaller

from .. import config as inductor_config
from ..kernel_template_choice import KernelTemplateChoice
from .core import make_lookup_key


log = logging.getLogger(__name__)


@dataclass
class LookupTableEntry:
    """Single entry representing one autotuning result"""

    key: str  # device_key+op_name+input_key
    value: dict[str, Any]  # Contains template_id and all kwargs
    metadata: dict[str, Any]  # Contains timing, rank, and other recording metadata

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

        # Use existing lookup key creation logic
        key = make_lookup_key(ktc.inputs, op_name)
        if key is None:
            return None

        # Extract template info
        template_id = (
            ktc.template.uid if hasattr(ktc.template, "uid") else ktc.template.name
        )

        # Build value dict from KTC kwargs
        value = dict(template_id=template_id, **ktc.kwargs)

        # Add template hash if available and configured
        if inductor_config.template_config_lookup_table.record_template_hash:
            # Use src_hash directly from the template
            template_hash = getattr(ktc.template, "src_hash", None)
            if template_hash is not None:
                value["template_hash"] = template_hash

        # Create metadata dict with timing and rank info
        metadata = {
            "timing": timing,
            "rank": rank,
        }

        return cls(key=key, value=value, metadata=metadata)


class EmitBackend(ABC):
    """Backend for immediate emission of single entries"""

    @abstractmethod
    def emit(self, entry: LookupTableEntry) -> None:
        pass


class RecordBackend(ABC):
    """Backend for dumping recorded table"""

    @abstractmethod
    def dump(self, data: dict[str, list[dict[str, Any]]]) -> None:
        pass


# Registry for decorated backends with their initialization arguments
_emit_backend_registry: list[tuple[type[EmitBackend], dict[str, Any]]] = []
_record_backend_registry: list[tuple[type[RecordBackend], dict[str, Any]]] = []


def emit_backend(
    should_register: bool = True, **kwargs: Any
) -> Callable[[type[EmitBackend]], type[EmitBackend]]:
    """Decorator to register an emit backend class with optional arguments"""

    def decorator(cls: type[EmitBackend]) -> type[EmitBackend]:
        if should_register:
            _emit_backend_registry.append((cls, kwargs))
        return cls

    return decorator


def record_backend(
    should_register: bool = True, **kwargs: Any
) -> Callable[[type[RecordBackend]], type[RecordBackend]]:
    """Decorator to register a record backend class with optional arguments"""

    def decorator(cls: type[RecordBackend]) -> type[RecordBackend]:
        if should_register:
            _record_backend_registry.append((cls, kwargs))
        return cls

    return decorator


@emit_backend(
    should_register=inductor_config.template_config_lookup_table.recorder_emit
)
class LogEmitBackend(EmitBackend):
    """Default emit backend that logs entries"""

    def emit(self, entry: LookupTableEntry) -> None:
        log.debug("LookupTable: %r -> %r", entry.key, entry.value)


@record_backend(
    should_register=bool(
        inductor_config.template_config_lookup_table.recorder_record_dir
    ),
    directory=inductor_config.template_config_lookup_table.recorder_record_dir,
)
class DirectoryRecordBackend(RecordBackend):
    """Default record backend that saves to timestamped files in a directory"""

    def __init__(self, directory: str):
        self.directory = directory

    def dump(self, data: dict[str, list[dict[str, Any]]]) -> None:
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)

        # Generate timestamped filename with 3-digit millisecond precision
        now = datetime.now(tz=timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
        filename = f"inductor_lut_{timestamp}.json"
        filepath = os.path.join(self.directory, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class LookupTableRecorder:
    """Main recorder that manages both emit and record backends"""

    def __init__(self) -> None:
        self.data: dict[str, list[dict[str, Any]]] = {}
        self.emit_backends: list[EmitBackend] = []
        self.record_backends: list[RecordBackend] = []

    def add_backend(self, backend: Union[EmitBackend, RecordBackend]) -> None:
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

        # Add to internal data (store clean value without metadata)
        if entry.key not in self.data:
            self.data[entry.key] = []
        self.data[entry.key].append(entry.value)

    def dump(self) -> None:
        """Dump via all record backends"""
        for backend in self.record_backends:
            backend.dump(self.data)

    def clear(self) -> None:
        self.data.clear()


# Module-wide instance
_lookup_table_recorder: Optional[LookupTableRecorder] = None


def get_lookup_table_recorder() -> LookupTableRecorder:
    """Get the global lookup table recorder"""
    global _lookup_table_recorder
    if _lookup_table_recorder is None:
        _lookup_table_recorder = LookupTableRecorder()
        _register_backends(_lookup_table_recorder)
    return _lookup_table_recorder


def add_backend(backend: Union[EmitBackend, RecordBackend]) -> None:
    """Add a backend to the global lookup table recorder"""
    recorder = get_lookup_table_recorder()
    if recorder is not None:
        recorder.add_backend(backend)


def _register_backends(recorder: LookupTableRecorder) -> None:
    """Register all decorated backends"""
    # Register all decorated emit backends
    for backend_cls, init_kwargs in _emit_backend_registry + _record_backend_registry:
        instance = backend_cls(**init_kwargs)
        try:
            recorder.add_backend(instance)
        except Exception as e:
            log.warning("Skipping backend %r - error: %r", backend_cls.__name__, e)


def record_topk_choices(
    timings: dict[ChoiceCaller, float],
    op_name: str,
    input_nodes: list[Any],
    choices: list[ChoiceCaller],
    profiled_time_fn: Callable[[], dict[Any, Any]],
    topk: Optional[int] = None,
) -> None:
    """
    Feedback function to record topk choices based on timing results.

    Args:
        timings: Mapping from choices to benchmark times
        op_name: Name of the operation (e.g. "mm", "addmm")
        input_nodes: List of input ir.Nodes
        choices: List of ChoiceCaller objects
        profiled_time_fn: Function to get profiled times (unused in this implementation)
        topk: Number of top choices to record. If None or negative, record all.
                If 0, record nothing.
    """
    # Get topk config
    if topk is None:
        topk = getattr(
            inductor_config.template_config_lookup_table, "recorder_topk", None
        )

    # If topk is 0, don't record anything
    if topk == 0:
        return

    # If topk is negative or None, record everything
    record_all = topk is None or topk < 0

    # Get recorder
    recorder = get_lookup_table_recorder()
    if recorder is None:
        return

    # Filter choices that have valid timings and KTC references
    valid_choices = []
    for choice in choices:
        if (
            choice in timings
            and choice.get_ktc() is not None
            and timings[choice] != float("inf")
        ):
            valid_choices.append(choice)
        else:
            log.debug(
                "Skipping choice %r - no timing or KTC reference: %r",
                choice,
                timings.get(choice, None),
            )
    if not valid_choices:
        return

    # Sort by timing (best first)
    sorted_choices = sorted(valid_choices, key=lambda c: timings[c])

    # Apply topk limit
    if not record_all:
        sorted_choices = sorted_choices[:topk]

    # Record each choice
    for rank, choice in enumerate(sorted_choices):
        ktc = choice.get_ktc()
        if ktc is None:
            # this should never happen as they have been filtered out above
            continue
        timing = timings[choice]

        entry = LookupTableEntry.from_ktc_and_timing(
            ktc=ktc,
            timing=timing,
            rank=rank,
            op_name=op_name,
        )

        if entry is not None:
            recorder.record(entry)


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
    from ..select_algorithm import add_feedback_saver

    add_feedback_saver(record_topk_choices)
    log.debug("Registered lookup table recorder feedback function")
except ImportError:
    log.warning(
        "Failed to register lookup table recorder feedback function - select_algorithm not available"
    )
