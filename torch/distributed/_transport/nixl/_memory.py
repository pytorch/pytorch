from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    import torch

    from ._transport import NIXLTransport


@dataclass(frozen=True)
class NIXLRemoteBuffer:
    """A serializable descriptor for memory registered by a NIXL peer."""

    agent_name: str
    address: int
    length: int
    device_id: int
    memory_type: str
    metadata: bytes


@dataclass
class _Registration:
    tensor: torch.Tensor
    storage: Any
    descs: Any
    address: int
    length: int
    device_id: int
    memory_type: str
    active: bool = True

    def ensure_active(self) -> None:
        if not self.active:
            raise RuntimeError("memory has been unregistered")


class NIXLMemoryView:
    """A byte range in NIXL-registered memory."""

    def __init__(self, memory: NIXLMemory, offset: int, length: int) -> None:
        self._memory = memory
        self._offset = offset
        self._length = length

    def size(self) -> int:
        return self._length


class NIXLMutableMemoryView(NIXLMemoryView):
    """A writable byte range in NIXL-registered memory."""

    writable: Literal[True] = True


class NIXLMemory:
    """A tensor registered with a NIXL transport."""

    def __init__(
        self, transport: NIXLTransport, registration: _Registration, reused: bool
    ) -> None:
        self._transport = transport
        self._registration = registration
        # The Python handle is new; reused records whether its native
        # registration already existed when register_memory() was called.
        # It cannot be inferred from the shared registration after construction.
        self._reused = reused

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        self._registration.ensure_active()
        offset = 0 if offset is None else operator.index(offset)
        if offset < 0 or offset > self._registration.length:
            raise ValueError("offset is outside the registered memory")
        length = (
            self._registration.length - offset
            if length is None
            else operator.index(length)
        )
        if length < 0 or length > self._registration.length - offset:
            raise ValueError("view exceeds the registered memory")
        return offset, length

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> NIXLMemoryView:
        return NIXLMemoryView(self, *self._range(offset, length))

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> NIXLMutableMemoryView:
        return NIXLMutableMemoryView(self, *self._range(offset, length))

    def to_remote_buffer(self, *, timeout: float | None = None) -> NIXLRemoteBuffer:
        return self._transport._call(
            lambda: self._transport._remote_buffer(self._registration), timeout
        )

    def reused_registration(self) -> bool:
        return self._reused
