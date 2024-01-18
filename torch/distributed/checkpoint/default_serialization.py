import abc
import io
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from .metadata import Metadata

from .serialization import Deserializer, Serializer


__all__ = ["save_state_dict", "save"]


class _DefaultSerializer(Serializer):
    @property
    def signature(self) -> str:
        return "default_serialization_v1"

    @staticmethod
    def _serialize(obj: object, stream: Optional[io.IOBase]) -> Optional[io.BytesIO]:
        if stream is not None:
            torch.save(obj, stream)
            return None
        else:
            bytes_io = io.BytesIO()
            torch.save(obj, bytes_io)
            return bytes_io

    def serialize_metadata(
        self, metadata: Metadata, stream: Optional[io.IOBase] = None
    ) -> Optional[io.BytesIO]:
        return _DefaultSerializer._serialize(metadata, stream)

    def serialize_tensor(
        self, tensor: torch.Tensor, stream: Optional[io.IOBase] = None
    ) -> Optional[io.BytesIO]:
        return _DefaultSerializer._serialize(tensor, stream)

    def serialize_object(
        self, obj: object, stream: Optional[io.IOBase] = None
    ) -> Optional[io.BytesIO]:
        return _DefaultSerializer._serialize(obj, stream)


class _DefaultDeserializer(Deserializer):
    @property
    def signature(self) -> str:
        return "default_serialization_v1"

    @staticmethod
    def _deserialize(data: io.BytesIO) -> object:
        return torch.load(data)

    def deserialize_metadata(self, data: io.BytesIO) -> Metadata:
        return _DefaultDeserializer._deserialize(data)

    def deserialize_tensor(self, data: io.BytesIO) -> torch.Tensor:
        return _DefaultDeserializer._deserialize(data)

    def deserialize_object(self, data: io.BytesIO) -> object:
        return _DefaultDeserializer._deserialize(data)
