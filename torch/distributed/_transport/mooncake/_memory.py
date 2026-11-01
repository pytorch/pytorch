from __future__ import annotations

import builtins
import operator
from dataclasses import dataclass, replace
from typing import Any, Literal, TYPE_CHECKING
from typing_extensions import Self

from torch.distributed._transport._serialization import _WireDescriptor


if TYPE_CHECKING:
    import torch

    from ._transport import MooncakeTransport


def _subrange(size: int, offset: int | None, length: int | None) -> tuple[int, int]:
    offset = 0 if offset is None else operator.index(offset)
    length = size - offset if length is None else operator.index(length)
    if not 0 <= offset <= size or not 0 <= length <= size - offset:
        raise ValueError("view exceeds the registered memory")
    return offset, length


def _slice_range(size: int, key: builtins.slice) -> tuple[int, int]:
    if not isinstance(key, builtins.slice):
        raise TypeError("expected a slice such as [a:b]")
    start, stop, step = key.indices(size)
    if step != 1:
        raise ValueError("slice step must be 1")
    return start, max(stop - start, 0)


@dataclass(frozen=True)
class _Endpoint(_WireDescriptor):
    _backend = "mooncake.endpoint"
    _fields = {"session": str, "segment": str}

    session: str
    segment: str


@dataclass(frozen=True)
class MooncakeRemoteBuffer(_WireDescriptor):
    """Byte-range descriptor; exchange only through a trusted control plane."""

    _backend = "mooncake"
    _fields = {
        "session": str,
        "segment": str,
        "address": int,
        "length": int,
        "generation": int,
    }

    session: str
    segment: str
    address: int
    length: int
    generation: int = 0

    def slice(self, offset: int | None = None, length: int | None = None) -> Self:
        offset, length = _subrange(self.length, offset, length)
        return replace(self, address=self.address + offset, length=length)

    def __getitem__(self, key: builtins.slice) -> Self:
        return self.slice(*_slice_range(self.length, key))


@dataclass
class _Registration:
    tensor: torch.Tensor
    storage: Any
    address: int
    length: int
    active: bool = True

    def check(self) -> None:
        if not self.active:
            raise RuntimeError("memory has been unregistered")
        if self.tensor.data_ptr() != self.address or (
            self.tensor.numel() * self.tensor.element_size() != self.length
        ):
            raise RuntimeError("registered tensor was resized or its storage replaced")


class MooncakeMemoryView:
    def __init__(self, memory: MooncakeMemory, offset: int, length: int) -> None:
        self._memory = memory
        self._offset = offset
        self._length = length

    def size(self) -> int:
        return self._length

    def slice(self, offset: int | None = None, length: int | None = None) -> Self:
        self._memory._registration.check()
        offset, length = _subrange(self._length, offset, length)
        return type(self)(self._memory, self._offset + offset, length)

    def __getitem__(self, key: builtins.slice) -> Self:
        return self.slice(*_slice_range(self._length, key))


class MooncakeMutableMemoryView(MooncakeMemoryView):
    writable: Literal[True] = True


class MooncakeMemory:
    def __init__(
        self, transport: MooncakeTransport, registration: _Registration
    ) -> None:
        self._transport = transport
        self._registration = registration

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        self._registration.check()
        capacity = self._registration.length
        offset, length = _subrange(capacity, offset, length)
        return offset, length

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MooncakeMemoryView:
        return MooncakeMemoryView(self, *self._range(offset, length))

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MooncakeMutableMemoryView:
        return MooncakeMutableMemoryView(self, *self._range(offset, length))

    def to_remote_buffer(self, *, timeout: float | None = None) -> MooncakeRemoteBuffer:
        with self._transport._locked(timeout):
            self._transport._ensure_open()
            self._registration.check()
            endpoint = self._transport._endpoint
            return MooncakeRemoteBuffer(
                endpoint.session,
                endpoint.segment,
                self._registration.address,
                self._registration.length,
                self._transport._generation,
            )
