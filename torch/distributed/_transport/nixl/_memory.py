from __future__ import annotations

import builtins
import operator
from dataclasses import dataclass, replace
from typing import Any, Literal, TYPE_CHECKING
from typing_extensions import Self

from .._serialization import _WireDescriptor


if TYPE_CHECKING:
    import torch

    from ._transport import NIXLTransport


def _subrange(
    size: int, offset: int | None, length: int | None, name: str = "registered memory"
) -> tuple[int, int]:
    offset = 0 if offset is None else operator.index(offset)
    if offset < 0 or offset > size:
        raise ValueError(f"offset is outside the {name}")
    length = size - offset if length is None else operator.index(length)
    if length < 0 or length > size - offset:
        raise ValueError(f"range exceeds the {name}")
    return offset, length


def _slice_range(size: int, key: builtins.slice) -> tuple[int, int]:
    if not isinstance(key, builtins.slice):
        raise TypeError("expected a slice such as [a:b]")
    start, stop, step = key.indices(size)
    if step != 1:
        raise ValueError("slice step must be 1")
    return start, max(stop - start, 0)


@dataclass(frozen=True)
class NIXLRemoteBuffer(_WireDescriptor):
    """A serializable descriptor for memory registered by a NIXL peer."""

    _backend = "nixl"
    _fields = {
        "agent_name": str,
        "address": int,
        "length": int,
        "device_id": int,
        "memory_type": str,
        "metadata": bytes,
    }

    agent_name: str
    address: int
    length: int
    device_id: int
    memory_type: str
    metadata: bytes

    def slice(self, offset: int | None = None, length: int | None = None) -> Self:
        offset, length = _subrange(self.length, offset, length, "remote buffer")
        return replace(self, address=self.address + offset, length=length)

    def __getitem__(self, key: builtins.slice) -> Self:
        return self.slice(*_slice_range(self.length, key))


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

    def slice(self, offset: int | None = None, length: int | None = None) -> Self:
        self._memory._registration.ensure_active()
        offset, length = _subrange(self._length, offset, length, "view")
        return type(self)(self._memory, self._offset + offset, length)

    def __getitem__(self, key: builtins.slice) -> Self:
        return self.slice(*_slice_range(self._length, key))


class NIXLMutableMemoryView(NIXLMemoryView):
    """A writable byte range in NIXL-registered memory."""

    writable: Literal[True] = True


class NIXLMemory:
    """A tensor registered with a NIXL transport."""

    def __init__(self, transport: NIXLTransport, registration: _Registration) -> None:
        self._transport = transport
        self._registration = registration

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        self._registration.ensure_active()
        return _subrange(self._registration.length, offset, length)

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
