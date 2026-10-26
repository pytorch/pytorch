from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"tuple": [_canonical_value(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_canonical_value(item) for item in value]}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {
            "dict": {key: _canonical_value(item) for key, item in sorted(value.items())}
        }
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value).removeprefix("torch.")}
    if isinstance(value, type):
        return {
            "type": {
                "module": value.__module__,
                "qualname": value.__qualname__,
            }
        }
    raise TypeError(
        "portable FlyDSL compile-time values must be primitives, containers, "
        f"or importable types; got {type(value).__qualname__}"
    )


def _decode_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "tuple" in value:
        return tuple(_decode_value(item) for item in value["tuple"])
    if "list" in value:
        return [_decode_value(item) for item in value["list"]]
    if "dict" in value:
        return {key: _decode_value(item) for key, item in value["dict"].items()}
    if "type" in value:
        spec = value["type"]
        resolved = importlib.import_module(spec["module"])
        for component in spec["qualname"].split("."):
            resolved = getattr(resolved, component)
        return resolved
    if "torch_dtype" in value:
        return getattr(torch, value["torch_dtype"])
    raise ValueError(f"invalid portable FlyDSL value: {value!r}")


def _dependency_digest(value: Any) -> str:
    if inspect.isfunction(value) or inspect.isclass(value):
        payload = {
            "module": value.__module__,
            "qualname": value.__qualname__,
            "source": inspect.getsource(value),
        }
    elif inspect.ismodule(value):
        try:
            version = importlib.metadata.version(value.__name__.split(".", 1)[0])
        except importlib.metadata.PackageNotFoundError:
            version = None
        payload = {"module": value.__name__, "version": version}
    else:
        payload = {"value": _canonical_value(value)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dependency_fingerprints(func: Any) -> tuple[tuple[str, str], ...]:
    closure = inspect.getclosurevars(func)
    values = {**closure.globals, **closure.nonlocals}
    return tuple(
        (name, _dependency_digest(value))
        for name, value in sorted(values.items())
        if name not in closure.builtins
    )


@dataclass(frozen=True)
class FlyDSLProgramSpec:
    module: str
    qualname: str
    source: str
    source_sha256: str
    signature: str
    parameter_names: tuple[str, ...]
    mutated_arg_names: tuple[str, ...]
    dependencies: tuple[tuple[str, str], ...]
    compatibility: str = ">=0.4.1,<0.5"

    @classmethod
    def from_launcher(
        cls,
        launcher: Any,
        *,
        mutated_arg_names: tuple[str, ...],
    ) -> FlyDSLProgramSpec:
        func = launcher.func
        if func.__module__ == "__main__":
            raise TypeError(
                "portable FlyDSL launchers must live in an importable module"
            )
        if "<locals>" in func.__qualname__:
            raise TypeError("portable FlyDSL launchers must be defined at module scope")
        source = inspect.getsource(func)
        return cls(
            module=func.__module__,
            qualname=func.__qualname__,
            source=source,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            signature=str(inspect.signature(func)),
            parameter_names=tuple(inspect.signature(func).parameters),
            mutated_arg_names=tuple(sorted(mutated_arg_names)),
            dependencies=_dependency_fingerprints(func),
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> FlyDSLProgramSpec:
        payload = json.loads(value)
        payload["parameter_names"] = tuple(payload["parameter_names"])
        payload["mutated_arg_names"] = tuple(payload["mutated_arg_names"])
        payload["dependencies"] = tuple(
            (str(name), str(digest)) for name, digest in payload["dependencies"]
        )
        spec = cls(**payload)
        if hashlib.sha256(spec.source.encode("utf-8")).hexdigest() != (
            spec.source_sha256
        ):
            raise ValueError("FlyDSL program source hash does not match bundle")
        return spec

    def resolve(self) -> Any:
        resolved = importlib.import_module(self.module)
        for component in self.qualname.split("."):
            resolved = getattr(resolved, component)
        source = inspect.getsource(getattr(resolved, "func", resolved))
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != self.source_sha256:
            raise RuntimeError(
                f"installed FlyDSL launcher {self.module}:{self.qualname} "
                "does not match the exported program bundle"
            )
        dependencies = _dependency_fingerprints(getattr(resolved, "func", resolved))
        if dependencies != self.dependencies:
            raise RuntimeError(
                f"installed FlyDSL launcher {self.module}:{self.qualname} "
                "has different helper/global dependencies"
            )
        return resolved


@dataclass(frozen=True)
class FlyDSLCallSpec:
    values: tuple[tuple[int, Any], ...]

    @classmethod
    def from_values(cls, values: dict[int, Any]) -> FlyDSLCallSpec:
        return cls(
            tuple(
                (index, _canonical_value(value))
                for index, value in sorted(values.items())
            )
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(
            {"values": self.values},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> FlyDSLCallSpec:
        payload = json.loads(value)
        return cls(tuple((int(index), encoded) for index, encoded in payload["values"]))

    def decode(self) -> dict[int, Any]:
        return {index: _decode_value(value) for index, value in self.values}
