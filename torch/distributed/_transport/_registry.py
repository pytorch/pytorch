from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib.metadata import entry_points, EntryPoint
from typing import Any, cast, TYPE_CHECKING

from ._api import Transport


if TYPE_CHECKING:
    import torch
    import torch.distributed as dist


_ENTRY_POINT_GROUP = "torch.distributed.transports"

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
    exists = name in _registered_transports or any(
        entry_point.name.lower() == name for entry_point in _iter_entry_points()
    )
    if exists and not replace:
        raise ValueError(f"transport {name!r} is already registered")
    _registered_transports[name] = factory


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
    if not matches:
        available = ", ".join(available_transports()) or "none"
        raise ValueError(f"unknown transport {name!r}; available: {available}")
    try:
        factory = matches[0].load()
    except Exception as error:
        raise RuntimeError(f"failed to load transport entry point {name!r}") from error
    if not callable(factory):
        raise TypeError(f"transport entry point {name!r} must load a callable")
    factory = cast(TransportFactory, factory)
    _registered_transports[name] = factory
    return factory


def available_transports() -> tuple[str, ...]:
    """Return registered and discoverable transport names."""
    names = set(_registered_transports)
    names.update(entry_point.name.lower() for entry_point in _iter_entry_points())
    return tuple(sorted(names))


def new_transport(
    backend: str,
    device: torch.device | str | None = None,
    *,
    peer_rank: int | None = None,
    group: dist.ProcessGroup | None = None,
    bootstrap_tag: str | None = None,
    bootstrap_timeout: float = 30.0,
    **kwargs: Any,
) -> Transport:
    """Construct a one-sided transport, optionally restricting tensor devices.

    Args:
        backend (str): Name registered with ``register_transport`` or an installed
            ``torch.distributed.transports`` entry point.
        device: Optional CPU/CUDA device restriction; otherwise infer each tensor's
            device at registration.
        peer_rank: Optional peer rank relative to ``group``. Both peers must call
            concurrently with reciprocal ranks and the same ``bootstrap_tag``.
            Uses the process group's Store, not a collective; other ranks need
            not participate. Requires an initialized process group.
        group: Bootstrap process group; defaults to the default group.
        bootstrap_tag: Required with ``peer_rank``. Unique per rank pair for the
            lifetime of the group Store, including failed attempts. Reusing a tag
            raises instead of connecting to stale endpoints.
        bootstrap_timeout: Bind, endpoint exchange, and connect wait budget in
            seconds. Native backend calls and Store network operations may exceed
            this budget. Does not change the group's Store timeout.
        **kwargs: Options forwarded to the selected backend's constructor.
            Backend modules are imported only when selected.

    Rank bootstrap (run concurrently on ranks 0 and 1)::

        import torch.distributed as dist
        from torch.distributed._transport import new_transport

        # An initialized process group supplies only the control-plane Store.
        transport = new_transport(
            "nixl",
            "cpu",
            peer_rank=1 - dist.get_rank(),
            bootstrap_tag="checkpoint-channel-0",
            bootstrap_timeout=30.0,
        )
        # Already bound and connected. Exchange registered-memory descriptors
        # separately, and coordinate with the peer before unregistering/closing.

    Manual bootstrap::

        import torch
        from torch.distributed._transport import new_transport

        backend = "my_backend"  # An installed or process-registered backend.

        # Both endpoints are shown locally. Across processes, exchange bind()
        # results and remote descriptors through your application's control plane.
        with (
            new_transport(backend, "cpu") as trainer,
            new_transport(backend, "cpu") as replica,
        ):
            trainer_url, replica_url = trainer.bind(), replica.bind()
            trainer.connect(replica_url)
            replica.connect(trainer_url)
            weights = torch.arange(8, dtype=torch.float32)
            received = torch.empty_like(weights)
            source = trainer.register_memory(weights)
            target = replica.register_memory(received)
            descriptor = target.to_remote_buffer()
            # Exchange these bytes through the application control plane.
            remote = type(descriptor).deserialize(descriptor.serialize())
            work = trainer.write(source.to_view(), remote, async_op=True)
            work.wait()
            torch.testing.assert_close(received, weights)
            # Coordinate with peers before closing either endpoint.


        # Inside an asyncio application, after registration and metadata exchange:
        async def push_weights(trainer, source, remote):
            await trainer.write_async(source.to_view(), remote, timeout=30.0)
            # Coordinate with peers before cleanup; cancellation does not cancel DMA.
            await trainer.close_async(timeout=30.0)
    """
    name = backend.lower()
    bootstrap = None
    if peer_rank is not None:
        from ._bootstrap import _RankBootstrap

        bootstrap = _RankBootstrap.create(
            name, peer_rank, group, bootstrap_tag, bootstrap_timeout
        )
    elif group is not None or bootstrap_tag is not None:
        raise ValueError("group and bootstrap_tag require peer_rank")
    factory = _find_factory(name)
    if (
        isinstance(factory, type)
        and issubclass(factory, Transport)
        and not factory.supported()
    ):
        raise RuntimeError(f"transport {name!r} is not supported")
    try:
        if device is not None:
            kwargs["device"] = device
        transport = factory(**kwargs)
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
    if bootstrap is not None:
        bootstrap.connect(transport)
    return transport
