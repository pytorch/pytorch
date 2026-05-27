"""Custom ops for async activation offloading between GPU and CPU.

These ops encapsulate stream management internally, producing a clean 2-node
IR pattern (offload/reload + wait_tensor) similar to c10d functional collectives.

A single dedicated transfer stream handles all D2H/H2D copies.
Completion events are keyed by output tensor data_ptr() and stored in a
module-level registry, so ``ao.wait_tensor`` takes only the tensor itself
(plus an optional keepalive).

Offload pattern:
    cpu_tensor = ao.offload(gpu_tensor)
    cpu_tensor = ao.wait_tensor(cpu_tensor, keepalive=gpu_tensor)
        keepalive frees the GPU tensor's storage after the D2H copy completes.

Reload pattern:
    gpu_tensor = ao.reload(cpu_tensor, device)
    gpu_tensor = ao.wait_tensor(gpu_tensor, keepalive=cpu_tensor)
        keepalive frees the CPU tensor's storage after the H2D copy completes.
"""

import ctypes
import ctypes.util
import functools
import os
import platform

import torch
from torch._library.custom_ops import custom_op
from torch.fx import has_side_effect


# --- Global transfer stream (one per device, lazily created) ---
_transfer_streams: dict[torch.device, torch.Stream] = {}


# --- NUMA-aware pinned memory allocation ---
#
# On systems with NVLink-C2C (e.g. GB200), allocating pinned CPU memory on the
# NUMA node closest to the GPU is critical for bandwidth: ~350 GB/s NUMA-local
# vs ~120 GB/s cross-NUMA.
#
# We use the set_mempolicy syscall with MPOL_BIND to strictly bind allocations
# to the GPU's NUMA node during pinned memory allocation. MPOL_PREFERRED is too
# weak - the kernel can still allocate on a remote node when the thread is
# running on a different NUMA node's CPUs.

_MPOL_DEFAULT = 0
_MPOL_BIND = 2
_SYS_SET_MEMPOLICY: int | None = None
_SYS_GET_MEMPOLICY: int | None = None
_libc: ctypes.CDLL | None = None


def _init_mempolicy() -> bool:
    """Initialize syscall numbers and libc for set/get_mempolicy. Linux-only."""
    global _SYS_SET_MEMPOLICY, _SYS_GET_MEMPOLICY, _libc
    if _libc is not None:
        return _SYS_SET_MEMPOLICY is not None
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except OSError:
        _libc = False  # type: ignore[assignment]
        return False
    _libc = lib
    machine = platform.machine()
    if machine == "x86_64":
        _SYS_SET_MEMPOLICY = 238
        _SYS_GET_MEMPOLICY = 239
    elif machine == "aarch64":
        _SYS_SET_MEMPOLICY = 237
        _SYS_GET_MEMPOLICY = 236
    else:
        return False
    return True


@functools.cache
def _gpu_numa_node(device_index: int) -> int | None:
    """Map a CUDA device index to its closest CPU NUMA node via sysfs.

    Uses torch.cuda.get_device_properties() for the PCI address, matching
    the approach in torch/numa/binding.py.
    """
    try:
        props = torch.cuda.get_device_properties(device_index)
        domain = props.pci_domain_id  # type: ignore[attr-defined]
        bus = props.pci_bus_id  # type: ignore[attr-defined]
        device = props.pci_device_id  # type: ignore[attr-defined]
        pci_addr = f"{domain:04x}:{bus:02x}:{device:02x}.0"
        numa_path = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
        if os.path.exists(numa_path):
            with open(numa_path) as f:
                node = int(f.read().strip())
                return node if node >= 0 else None
    except Exception:
        pass
    return None


_BITS_PER_ULONG = ctypes.sizeof(ctypes.c_ulong) * 8


def _nodemask_for(node: int) -> tuple[ctypes.Array[ctypes.c_ulong], int]:
    """Build a nodemask with a single bit set for the given NUMA node.

    Returns (nodemask_array, maxnode) sized to hold at least node+1 bits.
    """
    n_ulongs = node // _BITS_PER_ULONG + 1
    maxnode = n_ulongs * _BITS_PER_ULONG
    mask = (ctypes.c_ulong * n_ulongs)()
    mask[node // _BITS_PER_ULONG] = 1 << (node % _BITS_PER_ULONG)
    return mask, maxnode


# Nodemask large enough for get_mempolicy to return any node the kernel knows.
_GET_POLICY_MAXNODE = 1024
_GET_POLICY_N_ULONGS = _GET_POLICY_MAXNODE // _BITS_PER_ULONG


def _get_mempolicy() -> tuple[int, ctypes.Array[ctypes.c_ulong], int] | None:
    """Capture the thread's current NUMA memory policy via get_mempolicy.

    Returns (mode, nodemask, maxnode) or None on failure.
    """
    if _libc is None or not _libc or _SYS_GET_MEMPOLICY is None:
        return None
    mode = ctypes.c_int(0)
    nodemask = (ctypes.c_ulong * _GET_POLICY_N_ULONGS)()
    ret = _libc.syscall(
        ctypes.c_long(_SYS_GET_MEMPOLICY),
        ctypes.byref(mode),
        ctypes.cast(nodemask, ctypes.POINTER(ctypes.c_ulong)),
        ctypes.c_ulong(_GET_POLICY_MAXNODE),
        ctypes.c_void_p(0),
        ctypes.c_ulong(0),
    )
    if ret != 0:
        return None
    return (mode.value, nodemask, _GET_POLICY_MAXNODE)


def _set_mempolicy(
    mode: int, nodemask: ctypes.Array[ctypes.c_ulong] | None, maxnode: int = 0
) -> bool:
    """Set the thread's NUMA memory policy via set_mempolicy."""
    if _libc is None or not _libc or _SYS_SET_MEMPOLICY is None:
        return False
    if mode == _MPOL_DEFAULT or nodemask is None:
        ret = _libc.syscall(
            ctypes.c_long(_SYS_SET_MEMPOLICY),
            ctypes.c_int(_MPOL_DEFAULT),
            ctypes.c_void_p(0),
            ctypes.c_ulong(0),
        )
    else:
        ret = _libc.syscall(
            ctypes.c_long(_SYS_SET_MEMPOLICY),
            ctypes.c_int(mode),
            ctypes.cast(nodemask, ctypes.POINTER(ctypes.c_ulong)),
            ctypes.c_ulong(maxnode),
        )
    return ret == 0


def _pinned_empty_like_numa(tensor: torch.Tensor) -> torch.Tensor:
    """Allocate a pinned CPU tensor on the NUMA node closest to tensor's GPU.

    Saves and restores the thread's current memory policy around the
    allocation so that any existing NUMA binding (e.g. from torchrun or
    c10 NUMA APIs) is preserved.
    """
    if not _init_mempolicy():
        return torch.empty_like(tensor, device="cpu", pin_memory=True)

    numa_node = _gpu_numa_node(tensor.device.index or 0)
    if numa_node is None:
        return torch.empty_like(tensor, device="cpu", pin_memory=True)

    saved = _get_mempolicy()

    nodemask, maxnode = _nodemask_for(numa_node)
    bound = _set_mempolicy(_MPOL_BIND, nodemask, maxnode)
    try:
        result = torch.empty_like(tensor, device="cpu", pin_memory=True)
    finally:
        if bound:
            if saved is not None:
                _set_mempolicy(saved[0], saved[1], saved[2])
            else:
                _set_mempolicy(_MPOL_DEFAULT, None)
    return result


def _get_or_create_transfer_stream(device: torch.device) -> torch.Stream:
    if device not in _transfer_streams:
        _transfer_streams[device] = torch.Stream(device=device)
    return _transfer_streams[device]


# --- Wait registry: maps data_ptr() -> (completion_event, device) ---
# Created by ao.offload / ao.reload, consumed (popped) by ao.wait_tensor.
# Not thread-safe — graph execution is single-threaded Python.
_wait_registry: dict[int, tuple[torch.Event, torch.device]] = {}


def _register_wait(tensor: torch.Tensor, device: torch.device) -> torch.Event:
    """Create an event for an async transfer and register it for wait_tensor."""
    event = torch.Event()
    _wait_registry[tensor.data_ptr()] = (event, device)
    return event


def _pop_wait(tensor: torch.Tensor) -> tuple[torch.Event, torch.device]:
    key = tensor.data_ptr()
    try:
        return _wait_registry.pop(key)
    except KeyError:
        raise RuntimeError(
            f"ao.wait_tensor: no pending transfer for tensor with data_ptr={key}. "
            "Every ao.wait_tensor must be paired with a preceding ao.offload or ao.reload."
        ) from None


def _clear_wait_registry() -> None:
    _wait_registry.clear()


@custom_op("ao::offload", mutates_args=())
def offload(tensor: torch.Tensor) -> torch.Tensor:
    """Async offload a GPU tensor to CPU on the dedicated transfer stream.

    Callers MUST pair this with an ``ao.wait_tensor`` that passes the source GPU
    tensor as ``keepalive`` to extend its lifetime past the async D2H copy.
    Do NOT use ``record_stream`` — it causes memory fragmentation and
    unbounded memory growth.

    Uses pinned-memory allocation + copy_ so the transfer is compatible
    with CUDA graph capture.
    """
    device = tensor.device
    transfer_stream = _get_or_create_transfer_stream(device)
    current_stream = torch.accelerator.current_stream(device)

    transfer_stream.wait_stream(current_stream)

    torch.accelerator.set_stream(transfer_stream)
    result = _pinned_empty_like_numa(tensor)
    completion_event = _register_wait(result, device)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    return result


@offload.register_fake
def _(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor, device="cpu")


@custom_op("ao::reload", mutates_args=())
def reload(
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Async reload a CPU tensor to GPU on the dedicated transfer stream.

    The GPU tensor is allocated on the compute stream to avoid cross-stream
    allocator ownership issues. The H2D copy runs on the transfer stream.
    The completion event is keyed by the output tensor's data_ptr.
    """
    transfer_stream = _get_or_create_transfer_stream(device)
    current_stream = torch.accelerator.current_stream(device)

    # Allocate on compute stream so the allocator tracks ownership correctly
    result = torch.empty_like(tensor, device=device)
    completion_event = _register_wait(result, device)

    transfer_stream.wait_stream(current_stream)

    torch.accelerator.set_stream(transfer_stream)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    return result


@reload.register_fake
def _(
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty_like(tensor, device=device)


# ao::wait_tensor is defined via torch.library with an aliasing schema so the
# output can alias the input (custom_op forbids this).
#
# Uses CompositeExplicitAutograd (single impl for all devices) because the
# offload case has mixed-device args: ``tensor`` is CPU (the offload result)
# while ``keepalive`` is CUDA (the source GPU tensor). A single impl avoids
# relying on device-priority dispatch ordering.
#
# Synchronization details (completion event, device) are looked up from
# ``_wait_registry`` keyed on ``tensor.data_ptr()``.
#
# ``keepalive`` is the source tensor of the async transfer. It creates a
# graph dependency that extends the source tensor's lifetime until the
# compute stream has waited on the transfer completion event. After the
# wait, the op frees the source tensor's storage via ``resize_(0)`` since
# it is no longer needed:
#   - Offload (D2H): keepalive is the GPU tensor; freed after the D2H copy.
#   - Reload (H2D): keepalive is the CPU tensor; freed after the H2D copy.
_lib = torch.library.Library("ao", "DEF")
_lib.define(
    "wait_tensor(Tensor(a) tensor, Tensor? keepalive=None, Tensor? dep=None) -> Tensor(a)"
)


@torch.library.impl("ao::wait_tensor", "CompositeExplicitAutograd")
def _ao_wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    dep: torch.Tensor | None = None,
) -> torch.Tensor:
    completion_event, device = _pop_wait(tensor)
    current_stream = torch.accelerator.current_stream(device)

    current_stream.wait_event(completion_event)
    if keepalive is not None:
        storage = keepalive.untyped_storage()
        if storage.size() > 0:
            storage.resize_(0)
    return tensor


@torch.library.register_fake("ao::wait_tensor")
def _ao_wait_tensor_fake(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    dep: torch.Tensor | None = None,
) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.ao.wait_tensor.default)


def wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    dep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Callable wrapper so ``wait_tensor`` can be imported by name for op registration."""
    return torch.ops.ao.wait_tensor.default(tensor, keepalive, dep)
