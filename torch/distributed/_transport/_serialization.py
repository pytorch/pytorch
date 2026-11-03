from __future__ import annotations

import base64
import binascii
import json
from typing import Any, ClassVar
from typing_extensions import Self


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate descriptor field {key!r}")
        result[key] = value
    return result


class _WireDescriptor:
    """JSON v1 framing; each backend explicitly declares its fields and types."""

    _backend: ClassVar[str]
    _fields: ClassVar[dict[str, type]]

    def serialize(self) -> bytes:
        values = {name: getattr(self, name) for name in self._fields}
        self._validate(values)
        fields = {
            name: base64.b64encode(value).decode("ascii")
            if type(value) is bytes
            else value
            for name, value in values.items()
        }
        return json.dumps(
            {"version": 1, "backend": self._backend, "fields": fields},
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    @classmethod
    def deserialize(cls, data: bytes) -> Self:
        if type(data) is not bytes:
            raise TypeError("serialized descriptor must be bytes")
        try:
            value = json.loads(data.decode("utf-8"), object_pairs_hook=_object)
            if type(value) is not dict or set(value) != {
                "version",
                "backend",
                "fields",
            }:
                raise ValueError("invalid descriptor envelope")
            if type(value["version"]) is not int or value["version"] != 1:
                raise ValueError("unsupported descriptor version")
            if value["backend"] != cls._backend:
                raise ValueError("descriptor belongs to a different backend")
            fields = value["fields"]
            if type(fields) is not dict or set(fields) != set(cls._fields):
                raise ValueError("invalid descriptor fields")
            for name, kind in cls._fields.items():
                if kind is bytes:
                    if type(fields[name]) is not str:
                        raise ValueError(f"invalid descriptor field {name!r}")
                    fields[name] = base64.b64decode(fields[name], validate=True)
            cls._validate(fields)
            return cls(**fields)
        except (UnicodeError, json.JSONDecodeError, binascii.Error) as error:
            raise ValueError("malformed descriptor") from error

    @classmethod
    def _validate(cls, fields: dict[str, Any]) -> None:
        for name, kind in cls._fields.items():
            value = fields[name]
            if type(value) is not kind or (kind is int and value < 0):
                raise ValueError(f"invalid descriptor field {name!r}")
