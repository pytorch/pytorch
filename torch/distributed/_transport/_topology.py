"""Best-effort selection of the RDMA NICs nearest to an accelerator.

Linux only: reads ``/sys/class/infiniband`` and ``/sys/bus/pci/devices``.
Anything missing or unexpected disables selection, leaving the backend's own
device discovery in charge.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)

_IB_ROOT = Path("/sys/class/infiniband")
_PCI_ROOT = Path("/sys/bus/pci/devices")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _accelerator(device: torch.device | None) -> torch.device | None:
    """``device`` if it is an accelerator, else the current accelerator device.

    Falls back only when the accelerator runtime is already initialized (e.g.
    by ``set_device``), so CPU-only processes neither initialize it (breaking
    ``fork``) nor all select device 0's NICs.
    """
    if device is not None and device.type != "cpu":
        return device
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        return None
    is_initialized = getattr(
        torch.get_device_module(accelerator), "is_initialized", None
    )
    if is_initialized is None or not is_initialized():
        return None
    return torch.device(accelerator.type, torch.accelerator.current_device_index())


def _pci_address(device: torch.device) -> str | None:
    """Sysfs PCI address, if the device properties expose it (CUDA, ROCm)."""
    try:
        properties = torch.get_device_module(device).get_device_properties(device)
        domain = int(properties.pci_domain_id)
        bus = int(properties.pci_bus_id)
        slot = int(properties.pci_device_id)
    except Exception:
        return None
    return f"{domain:04x}:{bus:02x}:{slot:02x}.0"


def _nics() -> dict[str, Path]:
    """Map active, fastest RDMA devices to their PCI sysfs paths."""
    nics: dict[str, tuple[Path, int]] = {}
    try:
        entries = sorted(_IB_ROOT.iterdir())
    except OSError:
        return {}
    for entry in entries:
        rate = 0
        active = False
        try:
            ports = list((entry / "ports").iterdir())
        except OSError:
            continue
        for port in ports:
            if "ACTIVE" not in _read(port / "state"):
                continue
            active = True
            try:
                rate = max(rate, int(float(_read(port / "rate").split()[0])))
            except (IndexError, ValueError):
                pass
        if not active:
            continue
        try:
            nics[entry.name] = ((entry / "device").resolve(strict=True), rate)
        except OSError:
            continue
    # Prefer the data-plane NICs: drop slower frontend devices.
    fastest = max((rate for _, rate in nics.values()), default=0)
    return {name: path for name, (path, rate) in nics.items() if rate == fastest}


def _numa_node(path: Path) -> str:
    return _read(path / "numa_node")


def _nearest(path: Path, nics: dict[str, Path]) -> tuple[str, ...]:
    """NICs sharing the longest PCIe path prefix with ``path``, then NUMA node."""

    def score(nic: Path) -> tuple[int, bool]:
        shared = 0
        for a, b in zip(path.parts, nic.parts):
            if a != b:
                break
            shared += 1
        return shared, _numa_node(nic) == _numa_node(path)

    scores = {name: score(nic) for name, nic in nics.items()}
    best = max(scores.values())
    return tuple(name for name, value in scores.items() if value == best)


def _peers(path: Path) -> list[Path]:
    """Physical PCI functions with the accelerator's vendor and class.

    Includes devices hidden by ``*_VISIBLE_DEVICES`` so every local process
    sees the same peers; excludes SR-IOV virtual functions.
    """
    vendor, cls = _read(path / "vendor"), _read(path / "class")
    peers = []
    try:
        entries = sorted(_PCI_ROOT.iterdir())
    except OSError:
        return [path]
    for entry in entries:
        if _read(entry / "vendor") != vendor or _read(entry / "class") != cls:
            continue
        if (entry / "physfn").exists():
            continue
        try:
            peers.append(entry.resolve(strict=True))
        except OSError:
            continue
    return peers if path in peers else [path]


def select_nics(device: torch.device | None) -> list[str] | None:
    """Return the RDMA device names a transport should use, or ``None`` for all.

    Picks the NICs nearest the transport's accelerator (see :func:`_accelerator`)
    by PCIe path, then NUMA node. Accelerators tied on the same NICs split
    them in PCI address order: each gets an equal disjoint share (leftover
    NICs are unused), or NICs are shared round-robin when there are more
    accelerators than NICs.
    """
    try:
        accelerator = _accelerator(device)
        address = None if accelerator is None else _pci_address(accelerator)
        if address is None:
            return None
        nics = _nics()
        if not nics:
            return None
        path = (_PCI_ROOT / address).resolve(strict=True)
        nearest = _nearest(path, nics)
        tied = [peer for peer in _peers(path) if _nearest(peer, nics) == nearest]
        rank = tied.index(path)
        share = len(nearest) // len(tied)
        if share:
            selected = list(nearest[rank * share : (rank + 1) * share])
        else:
            selected = [nearest[rank % len(nearest)]]
    except Exception:
        logger.debug("RDMA NIC selection failed", exc_info=True)
        return None
    logger.info(
        "transport on %s (%s) uses RDMA NICs %s", accelerator, address, selected
    )
    return selected
