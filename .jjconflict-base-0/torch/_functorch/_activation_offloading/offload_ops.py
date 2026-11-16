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
        keepalive frees the GPU tensor's storage after the D2H copy completes.

Reload pattern:
    gpu_tensor = ao.reload(cpu_tensor, device)
    gpu_tensor = ao.wait_tensor(gpu_tensor, keepalive=cpu_tensor)
        keepalive frees the CPU tensor's storage after the H2D copy completes.

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


# --- Pinned memory pool (avoids per-offload cudaHostAlloc overhead) ---
# Keyed by (numel, dtype) to prevent cross-dtype reuse.
# Not keyed by device: pinned CPU memory is host-side and accessible from
# any GPU. Cross-device reuse is safe because wait_tensor synchronizes the
# transfer (via wait_event) before _pool_free returns the buffer — the
# buffer is never in-flight when reused.
_pinned_pool: dict[tuple[int, torch.dtype], list[torch.Tensor]] = {}
_pool_enabled: bool = False
_pool_managed_ptrs: set[int] = set()


def _maybe_pool_alloc(numel: int, dtype: torch.dtype) -> torch.Tensor:
    """Get a pinned CPU buffer from the pool (if enabled), or allocate fresh."""
    if _pool_enabled:
        key = (numel, dtype)
        bucket = _pinned_pool.get(key)
        if bucket:
            buf = bucket.pop()
            _pool_managed_ptrs.add(buf.data_ptr())
            return buf
        buf = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
        _pool_managed_ptrs.add(buf.data_ptr())
        return buf
    return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)


def _pool_free(buf: torch.Tensor) -> None:
    """Return a pinned buffer to the pool, or free its storage if pool disabled."""
    ptr = buf.data_ptr()
    if ptr in _pool_managed_ptrs:
        key = (buf.nelement(), buf.dtype)
        _pinned_pool.setdefault(key, []).append(buf)
    else:
        storage = buf.untyped_storage()
        if storage.size() > 0:
            storage.resize_(0)


def _pool_clear() -> None:
    """Free all cached pinned buffers."""
    _pinned_pool.clear()
    _pool_managed_ptrs.clear()


import contextlib


@contextlib.contextmanager
def pinned_memory_pool(enabled: bool = True):
    """Context manager that controls pinned memory pooling for offload ops.

    Without this context manager, ``ao.offload`` allocates a fresh pinned
    buffer every call and ``ao.wait_tensor`` does not cache freed buffers.
    Inside the context with ``enabled=True``, buffers are reused across
    calls, avoiding the ~3 ms per-tensor ``cudaHostAlloc`` overhead::

        with pinned_memory_pool():
            for step in range(num_steps):
                train_step()  # ao.offload/reload reuse pooled buffers
        # pinned buffers freed here

    Nestable: inner contexts save and restore the outer state.
    Pass ``enabled=False`` to temporarily disable pooling.
    """
    global _pool_enabled
    saved = _pool_enabled
    _pool_enabled = enabled
    try:
        yield
    finally:
        if not saved:
            _pool_clear()
        _pool_enabled = saved


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
    result = _maybe_pool_alloc(tensor.nelement(), tensor.dtype).view(tensor.shape)
    completion_event = _register_wait(result, device)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    return result


@offload.register_fake
def _(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")


@custom_op("ao::reload", mutates_args=())
def reload(
    tensor: torch.Tensor,
    device: torch.device,
    original_size: list[int] | None = None,
    original_stride: list[int] | None = None,
) -> torch.Tensor:
    """Async reload a CPU tensor to GPU on the dedicated transfer stream.

    The GPU tensor is allocated on the compute stream to avoid cross-stream
    allocator ownership issues. The H2D copy runs on the transfer stream.
    The completion event is keyed by the output tensor's data_ptr.

    ``original_size`` and ``original_stride`` restore the GPU tensor's
    original layout (which may be non-contiguous, e.g. from transpose).
    When None, the tensor's own size/stride are used.
    """
    size = original_size if original_size is not None else list(tensor.shape)
    stride = original_stride if original_stride is not None else list(tensor.stride())
    transfer_stream = _get_or_create_transfer_stream(device)
    current_stream = torch.accelerator.current_stream(device)

    result = torch.empty_strided(size, stride, dtype=tensor.dtype, device=device)
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
    original_size: list[int] | None = None,
    original_stride: list[int] | None = None,
) -> torch.Tensor:
    size = original_size if original_size is not None else list(tensor.shape)
    stride = original_stride if original_stride is not None else list(tensor.stride())
    return torch.empty_strided(size, stride, dtype=tensor.dtype, device=device)


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
    "wait_tensor(Tensor(a) tensor, Tensor? keepalive=None, Tensor? last_use_of_storage=None) -> Tensor(a)"
)


@torch.library.impl("ao::wait_tensor", "CompositeExplicitAutograd")
def _ao_wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
) -> torch.Tensor:
    completion_event, device = _pop_wait(tensor)
    current_stream = torch.accelerator.current_stream(device)

    current_stream.wait_event(completion_event)
    if keepalive is not None:
        if keepalive.is_pinned():
            # Return CPU pinned buffer to pool for reuse
            _pool_free(keepalive.view(-1))
        else:
            storage = keepalive.untyped_storage()
            if storage.size() > 0:
                storage.resize_(0)
    return tensor


@torch.library.register_fake("ao::wait_tensor")
def _ao_wait_tensor_fake(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.ao.wait_tensor.default)


def wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Callable wrapper so ``wait_tensor`` can be imported by name for op registration."""
    return torch.ops.ao.wait_tensor.default(tensor, keepalive, last_use_of_storage)
