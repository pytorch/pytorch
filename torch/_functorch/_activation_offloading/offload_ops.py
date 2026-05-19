"""Custom ops for async activation offloading between GPU and CPU.

These ops encapsulate stream management internally, producing a clean 2-node
IR pattern (offload/reload + wait_tensor) similar to c10d functional collectives.

A single dedicated transfer stream handles all D2H/H2D copies.
Completion events are keyed by output tensor data_ptr() and stored in a
module-level registry, so ``ao.wait_tensor`` takes only the tensor itself
(plus optional ``keepalive`` and ``last_use_of_storage`` args).

Offload pattern:
    cpu_tensor = ao.offload(gpu_tensor)
    cpu_tensor = ao.wait_tensor(cpu_tensor, keepalive=gpu_tensor)
        keepalive keeps the GPU tensor live until the D2H copy completes.

Reload pattern:
    gpu_tensor = ao.reload(cpu_tensor, device)
    gpu_tensor = ao.wait_tensor(gpu_tensor, keepalive=cpu_tensor)
        keepalive keeps the CPU tensor live until the H2D copy completes.

The optional ``last_use_of_storage`` arg creates a dependency on a tensor whose
storage must outlive the async transfer but is not otherwise connected by a
data-flow edge in the graph.
"""

import torch
from torch._library.custom_ops import custom_op
from torch.fx import has_side_effect


# --- Global transfer stream (one per device, lazily created) ---
_transfer_streams: dict[torch.device, torch.Stream] = {}


def _get_or_create_transfer_stream(device: torch.device) -> torch.Stream:
    if device not in _transfer_streams:
        _transfer_streams[device] = torch.Stream(device=device)
    return _transfer_streams[device]


# --- Wait registry: maps data_ptr() -> (completion_event, device) ---
# Created by ao.offload / ao.reload, consumed (popped) by ao.wait_tensor.
# Not thread-safe — graph execution is single-threaded Python.
_wait_registry: dict[int, tuple[torch.Event, torch.device]] = {}
_stream_event_registry: dict[int, tuple[torch.Event, torch.device]] = {}


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
    _stream_event_registry.clear()


def _register_stream_event(token: torch.Tensor, device: torch.device) -> torch.Event:
    event = torch.Event()
    _stream_event_registry[token.data_ptr()] = (event, device)
    return event


def _pop_stream_event(token: torch.Tensor) -> tuple[torch.Event, torch.device]:
    key = token.data_ptr()
    try:
        return _stream_event_registry.pop(key)
    except KeyError:
        raise RuntimeError(
            f"ao.reload_after: no pending stream event for token with data_ptr={key}."
        ) from None


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
    result = torch.empty_like(tensor, device="cpu", pin_memory=True)
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

    The GPU tensor is allocated and populated on the transfer stream. The
    compute stream waits on the recorded event before first use.
    """
    transfer_stream = _get_or_create_transfer_stream(device)
    current_stream = torch.accelerator.current_stream(device)

    torch.accelerator.set_stream(transfer_stream)
    result = torch.empty_like(tensor, device=device)
    completion_event = _register_wait(result, device)
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


@custom_op("ao::record_stream_event", mutates_args=())
def record_stream_event(tensor: torch.Tensor) -> torch.Tensor:
    """Record a compute-stream event and return a CPU token for a later reload."""
    device = tensor.device
    current_stream = torch.accelerator.current_stream(device)
    token = torch.empty((), device="cpu")
    event = _register_stream_event(token, device)
    current_stream.record_event(event)
    return token


@record_stream_event.register_fake
def _(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty((), device="cpu")


@custom_op("ao::reload_after", mutates_args=())
def reload_after(
    tensor: torch.Tensor,
    device: torch.device,
    wait_token: torch.Tensor,
) -> torch.Tensor:
    """Reload CPU tensor after the compute-stream event represented by wait_token."""
    event, event_device = _pop_stream_event(wait_token)
    if event_device != device:
        raise RuntimeError(
            f"ao.reload_after: token device {event_device} does not match reload device {device}"
        )
    event.synchronize()
    return reload(tensor, device)


@reload_after.register_fake
def _(
    tensor: torch.Tensor,
    device: torch.device,
    wait_token: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(tensor, device=device)


@custom_op("ao::reload_inplace", mutates_args={"dst"})
def reload_inplace(
    tensor: torch.Tensor,
    dst: torch.Tensor,
    wait_token: torch.Tensor,
) -> torch.Tensor:
    """Reload CPU tensor into an existing GPU buffer after wait_token."""
    event, device = _pop_stream_event(wait_token)
    if dst.device != device:
        raise RuntimeError(
            f"ao.reload_inplace: token device {device} does not match dst device {dst.device}"
        )
    transfer_stream = _get_or_create_transfer_stream(device)
    current_stream = torch.accelerator.current_stream(device)

    torch.accelerator.set_stream(transfer_stream)
    transfer_stream.wait_event(event)
    completion_event = _register_wait(dst, device)
    dst.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)
    return torch.empty((), device="cpu")


@reload_inplace.register_fake
def _(
    tensor: torch.Tensor,
    dst: torch.Tensor,
    wait_token: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((), device="cpu")


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
# compute stream has waited on the transfer completion event:
#   - Offload (D2H): keepalive is the GPU tensor.
#   - Reload (H2D): keepalive is the CPU tensor.
#
# ``last_use_of_storage`` is an optional tensor whose storage is shared with
# other live tensors (views/aliases). Passing it here tells the scheduler
# that this wait_tensor call is the last consumer of that storage, creating
# a data dependency edge that prevents the storage from being freed or
# reused before the async transfer completes. This is needed when the
# storage is not otherwise kept alive by a direct data-flow edge in the
# graph -- without it, a compiler pass could schedule a storage-freeing op
# before the transfer finishes.
_lib = torch.library.Library("ao", "DEF")
_lib.define(
    "wait_tensor(Tensor(a) tensor, Tensor? keepalive=None, Tensor? last_use_of_storage=None, Tensor? prefetch_dependency=None) -> Tensor(a)"
)


@torch.library.impl("ao::wait_tensor", "CompositeExplicitAutograd")
def _ao_wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
    prefetch_dependency: torch.Tensor | None = None,
) -> torch.Tensor:
    completion_event, device = _pop_wait(tensor)
    current_stream = torch.accelerator.current_stream(device)

    current_stream.wait_event(completion_event)
    if tensor.device == device:
        tensor.record_stream(current_stream)
    return tensor


@torch.library.register_fake("ao::wait_tensor")
def _ao_wait_tensor_fake(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
    prefetch_dependency: torch.Tensor | None = None,
) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.ao.offload.default)
has_side_effect(torch.ops.ao.reload.default)
has_side_effect(torch.ops.ao.record_stream_event.default)
has_side_effect(torch.ops.ao.reload_after.default)
has_side_effect(torch.ops.ao.reload_inplace.default)
has_side_effect(torch.ops.ao.wait_tensor.default)


def wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
    prefetch_dependency: torch.Tensor | None = None,
) -> torch.Tensor:
    """Callable wrapper so ``wait_tensor`` can be imported by name for op registration."""
    return torch.ops.ao.wait_tensor.default(
        tensor, keepalive, last_use_of_storage, prefetch_dependency
    )
