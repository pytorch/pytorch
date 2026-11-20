from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Self, TYPE_CHECKING

from .._api import _slice_range, _subrange
from .._serialization import _WireDescriptor


if TYPE_CHECKING:
    import builtins

    import torch

    from ._transport import MooncakeTransport


@dataclass(frozen=True)
class MooncakeRemoteBuffer(_WireDescriptor):
    """A serializable descriptor for memory registered by a Mooncake peer.

    Mooncake addresses peers by endpoint and reaches their registered segments
    directly, so the descriptor carries no opaque per-buffer metadata blob. A
    slice narrows ``address``/``length``; Mooncake resolves the sub-range to the
    covering registered buffer.
    """

    _backend = "mooncake"
    _fields = {
        "endpoint": str,
        "address": int,
        "length": int,
        "device_id": int,
        "memory_type": str,
    }

    endpoint: str
    address: int
    length: int
    device_id: int
    memory_type: str

    def slice(self, offset: int | None = None, length: int | None = None) -> Self:
        offset, length = _subrange(self.length, offset, length, "remote buffer")
        return replace(self, address=self.address + offset, length=length)

    def __getitem__(self, key: builtins.slice) -> Self:
        return self.slice(*_slice_range(self.length, key))


@dataclass
class _Registration:
    tensor: torch.Tensor
    storage: Any
    address: int
    length: int
    device_id: int
    memory_type: str
    active: bool = True
    # Set when a transfer times out: Mooncake may still be accessing the memory.
    pinned: bool = False

    def ensure_active(self) -> None:
        if not self.active:
            raise RuntimeError("memory has been unregistered")


class MooncakeMemoryView:
    """A byte range in Mooncake-registered memory."""

    def __init__(self, memory: MooncakeMemory, offset: int, length: int) -> None:
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


class MooncakeMutableMemoryView(MooncakeMemoryView):
    """A writable byte range in Mooncake-registered memory."""

    writable: Literal[True] = True


class MooncakeMemory:
    """A tensor registered with a Mooncake transport."""

    def __init__(
        self, transport: MooncakeTransport, registration: _Registration
    ) -> None:
        self._transport = transport
        self._registration = registration

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        self._registration.ensure_active()
        return _subrange(self._registration.length, offset, length)

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MooncakeMemoryView:
        return MooncakeMemoryView(self, *self._range(offset, length))

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MooncakeMutableMemoryView:
        return MooncakeMutableMemoryView(self, *self._range(offset, length))

    def to_remote_buffer(self, *, timeout: float | None = None) -> MooncakeRemoteBuffer:
        return self._transport._call(
            lambda: self._transport._remote_buffer(self._registration), timeout
        )
