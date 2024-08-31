# mypy: allow-untyped-defs
from abc import ABC, abstractmethod


class _StreamBase(ABC):
    r"""Base stream class abstraction for multi backends Stream to herit from"""

    @abstractmethod
    def wait_event(self, event) -> None:
        raise NotImplementedError

    @abstractmethod
    def wait_stream(self, stream) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_event(self, event=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, stream) -> bool:
        raise NotImplementedError


class _EventBase(ABC):
    r"""Base Event class abstraction for multi backends Event to herit from"""

    @abstractmethod
    def wait(self, stream=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError
