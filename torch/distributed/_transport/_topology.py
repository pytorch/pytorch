"""Best-effort selection of the RDMA NICs nearest to an accelerator.

Linux only: reads ``/sys/class/infiniband`` and ``/sys/bus/pci/devices``.
Anything missing or unexpected disables selection, leaving the backend's own
device discovery in charge.
"""

from __future__ import annotations

import logging
import os
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
    """``device`` if it is an indexed accelerator, else the current device.

    For a CPU or ``None`` device, falls back only when the accelerator runtime
    is already initialized (e.g. by ``set_device``), so CPU-only processes
    neither initialize it (breaking ``fork``) nor all select device 0's NICs.
    Initialization alone leaves the current device at 0, so device 0 with a
    nonzero ``LOCAL_RANK`` and more than one visible device is treated as
    unknown.
    """
    if device is not None and device.type != "cpu":
        if device.index is not None:
            return device
        accelerator = device
    else:
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None:
            return None
        module = torch.get_device_module(accelerator)
        is_initialized = getattr(module, "is_initialized", None)
        if is_initialized is None or not is_initialized():
            return None
    index = torch.accelerator.current_device_index()
    if (
        index == 0
        and os.environ.get("LOCAL_RANK", "0") != "0"
        and torch.accelerator.device_count() > 1
    ):
        logger.warning(
            "transport has no device index and the current %s device is 0 with "
            "LOCAL_RANK=%s; using the backend's default RDMA NICs. Call "
            "set_device, pass an indexed device, or list the NICs explicitly "
            "(Mooncake device_name)",
            accelerator.type,
            os.environ["LOCAL_RANK"],
        )
        return None
    return torch.device(accelerator.type, index)


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
        port = entry / "ports" / "1"
        if "ACTIVE" not in _read(port / "state"):
            continue
        # SR-IOV virtual functions share their physical function's port.
        if (entry / "device" / "physfn").exists():
            continue
        try:
            rate = int(float(_read(port / "rate").split()[0]))
        except (IndexError, ValueError):
            rate = 0
        try:
            nics[entry.name] = ((entry / "device").resolve(strict=True), rate)
        except OSError:
            continue
    # Prefer the data-plane NICs: drop slower frontend devices.
    fastest = max((rate for _, rate in nics.values()), default=0)
    return {name: path for name, (path, rate) in nics.items() if rate == fastest}


def _numa_node(path: Path) -> str:
    return _read(path / "numa_node")


def _nearest(
    path: Path, nics: dict[str, Path]
) -> tuple[tuple[int, bool], tuple[str, ...]]:
    """NICs sharing the longest PCIe path prefix with ``path``, then NUMA node.

    Returns the best score and the NICs that achieve it.
    """

    def score(nic: Path) -> tuple[int, bool]:
        shared = 0
        for a, b in zip(path.parts, nic.parts):
            if a != b:
                break
            shared += 1
        return shared, _numa_node(nic) == _numa_node(path)

    scores = {name: score(nic) for name, nic in nics.items()}
    best = max(scores.values())
    return best, tuple(name for name, value in scores.items() if value == best)


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


def _split(path: Path, nics: dict[str, Path]) -> list[str]:
    """Assign NICs to ``path`` and every peer accelerator, return ``path``'s.

    Peers with the same nearest NICs form a group. Groups claim NICs closest
    first, so a group whose nearest NICs overlap a closer group's uses only
    the unclaimed ones, or shares them all if none are left. Each group
    splits its NICs in name order: an
    equal disjoint share each (leftovers stay free for other groups), or
    round-robin when there are more accelerators than NICs.
    """
    groups: dict[tuple[str, ...], tuple[tuple[int, bool], list[Path]]] = {}
    for peer in _peers(path):
        best, nearest = _nearest(peer, nics)
        score, members = groups.get(nearest, (best, []))
        groups[nearest] = (max(score, best), members + [peer])
    claimed: set[str] = set()
    for nearest, (_, members) in sorted(
        groups.items(), key=lambda item: item[1][0], reverse=True
    ):
        free = [name for name in nearest if name not in claimed] or list(nearest)
        share = len(free) // len(members)
        claimed.update(free[: share * len(members)] if share else free)
        if path not in members:
            continue
        rank = members.index(path)
        if share:
            return free[rank * share : (rank + 1) * share]
        return [free[rank % len(free)]]
    raise AssertionError("accelerator missing from its peers")


def select_nics(device: torch.device | None) -> list[str] | None:
    """Return the RDMA device names a transport should use, or ``None`` for all.

    Picks the NICs nearest the transport's accelerator (see :func:`_accelerator`)
    by PCIe path, then NUMA node, split among accelerators as in :func:`_split`.
    """
    try:
        nics = _nics()
        if not nics:
            return None
        accelerator = _accelerator(device)
        if accelerator is None:
            return None
        address = _pci_address(accelerator)
        if address is None:
            return None
        path = (_PCI_ROOT / address).resolve(strict=True)
        selected = _split(path, nics)
    except Exception:
        logger.debug("RDMA NIC selection failed", exc_info=True)
        return None
    logger.info(
        "transport on %s (%s) uses RDMA NICs %s", accelerator, address, selected
    )
    return selected
