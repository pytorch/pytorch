from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable
from typing_extensions import Self

import torch


@runtime_checkable
class MemoryView(Protocol):
    """A read-only view of registered memory."""

    def size(self) -> int: ...


@runtime_checkable
class MutableMemoryView(MemoryView, Protocol):
    """A writable view of registered memory."""


@runtime_checkable
class RemoteBuffer(Protocol):
    """A serializable descriptor for registered memory on a peer."""


@runtime_checkable
class Memory(Protocol):
    """Memory registered with a transport."""

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MemoryView: ...

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MutableMemoryView: ...

    def to_remote_buffer(self) -> RemoteBuffer: ...

    def reused_registration(self) -> bool: ...


class Transport(ABC):
    """Base class for one-sided tensor transports."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)

    @staticmethod
    @abstractmethod
    def supported() -> bool:
        """Return whether the transport can be used in this process."""

    @abstractmethod
    def bind(self) -> bytes:
        """Bind the endpoint and return its opaque connection URL."""

    @abstractmethod
    def connect(self, peer_url: bytes) -> int:
        """Connect to a bound peer and return zero on success."""

    @abstractmethod
    def connected(self) -> bool:
        """Return whether the endpoint is connected to its peer."""

    @abstractmethod
    def register_memory(self, tensor: torch.Tensor) -> Memory:
        """Register a contiguous tensor for transport operations."""

    @abstractmethod
    def write(self, local_buffer: MemoryView, remote_buffer: RemoteBuffer) -> int:
        """Write a local view to remote memory and return zero on success."""

    @abstractmethod
    def read(self, local_buffer: MutableMemoryView, remote_buffer: RemoteBuffer) -> int:
        """Read remote memory into a local view and return zero on success."""

    @abstractmethod
    def close(self) -> None:
        """Release transport resources."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
