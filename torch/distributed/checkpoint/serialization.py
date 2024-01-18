import abc
import io
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from .metadata import Metadata


class Serializer(abc.ABC):
    @property
    @abc.abstractmethod
    def signature(self) -> str:
        ...

    @abc.abstractmethod
    def serialize_metadata(
        self, metadata: Metadata, stream: Optional[io.IOBase] = None
    ) -> Optional[io.BytesIO]:
        ...

    @abc.abstractmethod
    def serialize_tensor(
        self, tensor: torch.Tensor, stream: Optional[io.IOBase] = None
    ) -> Optional[io.BytesIO]:
        ...

    @abc.abstractmethod
    def serialize_object(
        self, obj: object, stream: Optional[io.IOBase] = None
    ) -> Optional[io.BytesIO]:
        ...


class Deserializer(abc.ABC):
    @property
    @abc.abstractmethod
    def signature(self) -> str:
        ...

    @abc.abstractmethod
    def deserialize_metadata(self, data: io.BytesIO) -> Metadata:
        ...

    @abc.abstractmethod
    def deserialize_tensor(self, data: io.BytesIO) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def deserialize_object(self, data: io.BytesIO) -> object:
        ...
