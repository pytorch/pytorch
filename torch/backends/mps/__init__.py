import sys
from contextlib import contextmanager
from functools import lru_cache as _lru_cache
from types import TracebackType
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule
from torch.library import Library as _Library


__all__ = [
    "AutotuneTrace",
    "autotune_trace",
    "benchmark",
    "flags",
    "get_core_count",
    "get_name",
    "is_available",
    "is_built",
    "is_macos13_or_newer",
    "is_macos_or_newer",
    "set_flags",
]


class AutotuneTrace:
    r"""Captured MPS autotuning launches and selections."""

    def __init__(self, max_entries: int, wait_until_completed: bool) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an int")
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        if not isinstance(wait_until_completed, bool):
            raise TypeError("wait_until_completed must be a bool")
        self.max_entries = max_entries
        self.wait_until_completed = wait_until_completed
        self.records: list[dict[str, Any]] = []
        self.dropped = 0
        self.schema_version = 1
        self._active = False

    def __enter__(self) -> "AutotuneTrace":
        if not is_built():
            raise RuntimeError("MPS is not available")
        torch._C._mps_start_autotune_trace(self.max_entries)
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self.wait_until_completed and exc_type is None:
                torch.mps.synchronize()
        finally:
            snapshot = torch._C._mps_stop_autotune_trace(self.wait_until_completed)
            self.records = snapshot["records"]
            self.dropped = snapshot["dropped"]
            self.schema_version = snapshot["schema_version"]
            self._active = False
            from torch._logging import trace_structured

            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "mps_autotune_trace",
                    "encoding": "json",
                },
                payload_fn=lambda: snapshot,
                expect_trace_id=False,
                record_logging_overhead=False,
            )


def autotune_trace(
    max_entries: int = 1024, wait_until_completed: bool = True
) -> AutotuneTrace:
    r"""Capture the MPS kernels and tile configurations used by autotuning.

    The trace records heuristic, exploratory, cached, and forced launches. It
    contains tensor metadata but never tensor contents or memory addresses.
    After the context exits, ``records`` is a JSON-serializable list of launch
    and selection records. ``dropped`` reports records evicted by the bound.
    Only one MPS autotune trace can be active at a time.
    When PyTorch structured tracing is enabled, the snapshot is also emitted as
    the ``mps_autotune_trace`` artifact for extraction with ``tlparse``.

    Args:
        max_entries: Maximum number of records retained. The oldest record is
            discarded when the trace exceeds this bound.
        wait_until_completed: Synchronize MPS and wait for pending autotuning
            measurements so their selection records are included. When False,
            late selection records can be omitted.
    """
    return AutotuneTrace(max_entries, wait_until_completed)


@contextmanager
def _autotune_override(operation: str, config: str):
    if not is_built():
        raise RuntimeError("MPS is not available")
    previous = torch._C._mps_get_autotune_override(operation)
    torch._C._mps_set_autotune_override(operation, config)
    try:
        yield
    finally:
        torch._C._mps_set_autotune_override(operation, previous)


def _clear_autotune_cache() -> None:
    if not is_built():
        raise RuntimeError("MPS is not available")
    torch._C._mps_clear_autotune_cache()


def set_flags(_benchmark=None):
    r"""Set whether MPS kernel tile autotuning is enabled globally."""
    original = (torch._C._get_mps_benchmark(),)
    if _benchmark is not None:
        torch._C._set_mps_benchmark(_benchmark)
    return original


@contextmanager
def flags(benchmark=False):
    r"""Context manager for setting MPS kernel tile autotuning globally."""
    with __allow_nonbracketed_mutation():
        original = set_flags(benchmark)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*original)


def is_built() -> bool:
    r"""Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run on a machine with working MPS drivers
    and devices, we would be able to use it.
    """
    return torch._C._has_mps


@_lru_cache
def is_available() -> bool:
    r"""Return a bool indicating if MPS is currently available."""
    return torch._C._mps_is_available()


@_lru_cache
def is_macos_or_newer(major: int, minor: int) -> bool:
    r"""Return a bool indicating whether MPS is running on given MacOS or newer."""
    return torch._C._mps_is_on_macos_or_newer(major, minor)


@_lru_cache
def is_macos13_or_newer(minor: int = 0) -> bool:
    r"""Return a bool indicating whether MPS is running on MacOS 13 or newer."""
    return torch._C._mps_is_on_macos_or_newer(13, minor)


@_lru_cache
def get_name() -> str:
    r"""Return Metal device name"""
    return torch._C._mps_get_name()


@_lru_cache
def get_core_count() -> int:
    r"""Return GPU core count.

    According to the documentation, one core is comprised of 16 Execution Units.
    One execution Unit has 8 ALUs.
    And one ALU can run 24 threads, i.e. one core is capable of executing 3072 threads concurrently.
    """
    return torch._C._mps_get_core_count()


class MPSModule(PropModule):
    benchmark = ContextProp(torch._C._get_mps_benchmark, torch._C._set_mps_benchmark)


sys.modules[__name__] = MPSModule(sys.modules[__name__], __name__)

benchmark: bool
