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

from . import config as inductor_config
from .kernel_inputs import KernelInputs
from .lookup_table import make_lookup_key


log = logging.getLogger(__name__)


@dataclass
class LookupTableEntry:
    """Single entry representing one autotuning result"""

    key: str  # device_key+op_name+input_key
    value: dict[str, Any]  # Contains template_id, timing, rank, and all kwargs

    @classmethod
    def from_template_kwargs(
        cls,
        kernel_inputs: KernelInputs,
        op_name: str,
        template_id: str,
        kwargs: dict[str, Any],
        template_hash: Optional[str] = None,
    ) -> Optional["LookupTableEntry"]:
        """Create entry from a choice and its timing"""
        value = dict(template_id=template_id, **kwargs)
        if (
            template_hash is not None
            and inductor_config.template_lookup_table_config.record_template_hash
        ):
            value["template_hash"] = template_hash
        key = make_lookup_key(kernel_inputs.nodes(), op_name)
        if key is not None:
            return cls(key=key, value=value)
        return None


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
    should_register=inductor_config.template_lookup_table_config.recorder_emit
)
class LogEmitBackend(EmitBackend):
    """Default emit backend that logs entries"""

    def emit(self, entry: LookupTableEntry) -> None:
        log.debug("LookupTable: %r -> %r", entry.key, entry.value)


@record_backend(
    should_register=bool(
        inductor_config.template_lookup_table_config.recorder_record_dir
    ),
    directory=inductor_config.template_lookup_table_config.recorder_record_dir,
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
    """Single table that manages both emit and record backends"""

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

        # Add to internal data
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


def record(
    kernel_inputs: KernelInputs,
    op_name: str,
    template_id: str,
    kwargs: dict[str, Any],
    template_hash: Optional[str] = None,
) -> None:
    """Record a single entry to the global lookup table recorder"""
    recorder = get_lookup_table_recorder()
    if recorder is not None:
        entry = LookupTableEntry.from_template_kwargs(
            kernel_inputs, op_name, template_id, kwargs, template_hash
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
