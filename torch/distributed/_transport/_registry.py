from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from importlib.metadata import entry_points, EntryPoint
from typing import Any, cast, TYPE_CHECKING

from ._api import Transport


if TYPE_CHECKING:
    import torch


_ENTRY_POINT_GROUP = "torch.distributed.transports"
_BUILTIN_ENTRY_POINTS = {
    "tcp": "torch.distributed._transport._tcp:TCPTransport",
    "torchcomms": "torch.distributed._transport._torchcomms:TorchCommsTransport",
    "ucxx": "torch.distributed._transport._ucxx:UCXXTransport",
}

TransportFactory = Callable[..., Transport]
_registered_transports: dict[str, TransportFactory] = {}


def register_transport(
    name: str, factory: TransportFactory, *, replace: bool = False
) -> None:
    """Register a transport factory for this process."""
    name = name.lower()
    if not name:
        raise ValueError("transport name cannot be empty")
    if not callable(factory):
        raise TypeError("transport factory must be callable")
    exists = name in _registered_transports or name in _BUILTIN_ENTRY_POINTS
    if exists and not replace:
        raise ValueError(f"transport {name!r} is already registered")
    _registered_transports[name] = factory


def _load_object(spec: str) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError(f"invalid transport entry point {spec!r}")
    return getattr(importlib.import_module(module_name), attribute)


def _iter_entry_points() -> Iterator[EntryPoint]:
    yield from entry_points(group=_ENTRY_POINT_GROUP)


def _find_factory(name: str) -> TransportFactory:
    if factory := _registered_transports.get(name):
        return factory
    matches = [
        entry_point
        for entry_point in _iter_entry_points()
        if entry_point.name.lower() == name
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple entry points registered transport {name!r}")
    if matches:
        try:
            factory = matches[0].load()
        except Exception as error:
            raise RuntimeError(
                f"failed to load transport entry point {name!r}"
            ) from error
    else:
        spec = _BUILTIN_ENTRY_POINTS.get(name)
        if spec is None:
            available = ", ".join(available_transports()) or "none"
            raise ValueError(f"unknown transport {name!r}; available: {available}")
        try:
            factory = _load_object(spec)
        except Exception as error:
            raise RuntimeError(f"failed to load transport {name!r}") from error
    if not callable(factory):
        raise TypeError(f"transport entry point {name!r} must load a callable")
    factory = cast(TransportFactory, factory)
    _registered_transports[name] = factory
    return factory


def available_transports() -> tuple[str, ...]:
    """Return registered and discoverable transport names."""
    names = set(_BUILTIN_ENTRY_POINTS) | set(_registered_transports)
    names.update(entry_point.name.lower() for entry_point in _iter_entry_points())
    return tuple(sorted(names))


def new_transport(
    backend: str,
    device: torch.device | str,
    **kwargs: Any,
) -> Transport:
    """Construct a transport registered under ``torch.distributed.transports``."""
    name = backend.lower()
    factory = _find_factory(name)
    if (
        isinstance(factory, type)
        and issubclass(factory, Transport)
        and not factory.supported()
    ):
        raise RuntimeError(f"transport {name!r} is not supported")
    try:
        transport = factory(device=device, **kwargs)
    except Exception as error:
        raise RuntimeError(f"failed to create transport {name!r}") from error
    if not isinstance(transport, Transport):
        raise TypeError(
            f"transport entry point {name!r} returned {type(transport).__name__}, "
            "expected a Transport"
        )
    if not transport.supported():
        transport.close()
        raise RuntimeError(f"transport {name!r} is not supported")
    return transport
