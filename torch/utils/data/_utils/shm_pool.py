# mypy: allow-untyped-defs
r"""SharedMemory slot pool for DataLoader pin_memory.

When ``DataLoader(pin_memory=True, num_workers>0)`` uses the default
``file_descriptor`` sharing strategy, every tensor in each batch is transferred
worker → main via a fresh ``/dev/shm`` file descriptor.  The pin_memory thread
consumes the batch, and when the previous batch's tensors are deallocated each
fd is ``close()``\ d + ``munmap()``\ d.  For workloads with large or
many-tensor batches this per-step fd churn + munmap dominates the pin thread
and shows up as a large wall-clock gap in the training step.

This module provides a reusable **SharedMemory slot pool** that retains
persistent ``/dev/shm`` mappings across batches instead of alloc/munmap per
batch.  **Every** CPU tensor in the batch is written into a pre-allocated slot
(a single ``numpy.copyto`` per tensor — one memcpy, zero syscalls, far cheaper
than the ``open``/``ftruncate``/``mmap``/``close``/``munmap`` + resource_sharer
IPC it replaces); a lightweight stub crosses the Queue without triggering
tensor reduction.  The pin thread rebuilds zero-copy views from the slot, pins
as normal, and recycles the previous slot.  ``del view`` does not munmap the
underlying mapping (the pool holds the ``SharedMemory`` object permanently).

The cost being pooled is **per-tensor fd churn**, which scales with tensor
*count* — not size.  Small tensors are just as costly in fd/munmap count and
nearly free to copy, so they are pooled too.

Backed 100% by Python stdlib ``multiprocessing.shared_memory`` — no custom
allocator.

Usage (single opt-in environment variable)::

    SHM_POOL=1                     enable

Everything else is self-tuning:
  * slot count  = ``max(prefetch_factor * num_workers, 4)``
  * slot bytes  = auto-measured from the first batch (× 2 headroom for
                  variable-length batches); tmpfs lazy-allocates so oversized
                  slots cost only virtual address space, not physical RAM
  * all CPU tensors are pooled (no size threshold — the overhead is per-tensor
    fd count, and ``copyto`` of a small tensor is cheaper than the fd it
    replaces); non-tensor fields cross the Queue via normal pickle

Lifecycle (fork start method):
  1. main process, before fork: ``init_coordination(n_slots)`` creates an
     ``mp.Event`` + ``mp.Value`` (slot size) + a free-list ``mp.Queue``.
     No SharedMemory yet — size is unknown.
  2. first batch: worker sees the pool not ready → sends tensors normally
     (file_descriptor).  pin thread measures the batch, creates the slots
     (``init_pool_from_batch``), fills its own ``_SLOTS``, sets the Event.
  3. subsequent batches: worker ``wait()`` on the Event (back-pressure until
     the pool is ready), ``attach`` to the slots once (cached), allocates a
     free slot, copies large tensors in, sends a ``SlotStub``.
  4. pin thread rebuilds views from the stub, pins, recycles the previous slot.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Any, List, Optional, Tuple

import numpy
import torch


# ---------------------------------------------------------------------------
# module-level state
# ---------------------------------------------------------------------------
# fork-inherited coordination (created by init_coordination before fork)
_POOL_READY: Optional[mp.synchronize.Event] = None
_SLOT_BYTES_VAL: Optional[mp.sharedctypes.Synchronized] = None
_N_SLOTS: int = 0
_FREE: Optional[mp.Queue] = None

# per-process slot handles
#   main/pin process: populated by init_pool_from_batch (create=True)
#   worker process:  populated lazily by _ensure_attached (create=False / attach)
_SLOTS: List[shared_memory.SharedMemory] = []
_SLOT_BYTES: int = 0

_ENABLED: Optional[bool] = None
_RANK: str = mp.current_process().name if False else "0"  # placeholder, set below

import os as _os

_RANK = _os.environ.get("RANK", "0")

_TORCH_TO_NP = {
    torch.int64: numpy.int64,
    torch.int32: numpy.int32,
    torch.int16: numpy.int16,
    torch.int8: numpy.int8,
    torch.uint8: numpy.uint8,
    torch.float64: numpy.float64,
    torch.float32: numpy.float32,
    torch.float16: numpy.float16,
    torch.bool: numpy.bool_,
}


def is_enabled() -> bool:
    """Whether the slot pool is enabled (cached from ``SHM_POOL`` env)."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = _os.environ.get("SHM_POOL", "0") == "1"
    return _ENABLED


# ---------------------------------------------------------------------------
# phase 1: main process, before fork — coordination primitives only
# ---------------------------------------------------------------------------
def init_coordination(n_slots: int) -> None:
    """Create the fork-inherited coordination primitives (no slots yet).

    Builds a free-list ``mp.Queue`` pre-seeded with ``0..n_slots-1`` plus an
    ``mp.Event`` / ``mp.Value`` the pin thread will use to tell workers the
    pool is sized and ready.  Idempotent.
    """
    global _POOL_READY, _SLOT_BYTES_VAL, _N_SLOTS, _FREE
    if _POOL_READY is not None:
        return
    _N_SLOTS = n_slots
    _POOL_READY = mp.Event()
    _SLOT_BYTES_VAL = mp.Value("Q", 0)  # unsigned long long
    _FREE = mp.Queue()
    for s in range(n_slots):
        _FREE.put(s)


# ---------------------------------------------------------------------------
# phase 2: pin thread, on first batch — measure + create slots + signal
# ---------------------------------------------------------------------------
def _get_or_create_slots(n_slots: int, slot_bytes: int) -> List[shared_memory.SharedMemory]:
    """Create N SharedMemory regions (rank-scoped names)."""
    slots: List[shared_memory.SharedMemory] = []
    for s in range(n_slots):
        name = f"__dataloader_shmpool_{_RANK}_{s}"
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=slot_bytes)
        except FileExistsError:
            shared_memory.SharedMemory(name=name).close()
            try:
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass
            shm = shared_memory.SharedMemory(name=name, create=True, size=slot_bytes)
        slots.append(shm)
    return slots


def estimate_slot_bytes(batch: Any, factor: float = 2.0) -> int:
    """Estimate slot size from a sample batch (sum of CPU-tensor bytes × factor).

    ``factor`` defaults to 2.0 to leave headroom for variable-length batches
    (e.g. recommendation-system KJT whose ``_values`` size varies per step).
    """
    total = 0

    def walk(obj: Any) -> None:
        nonlocal total
        if isinstance(obj, torch.Tensor) and not obj.is_cuda:
            total += obj.element_size() * obj.numel()
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif hasattr(obj, "__dict__") and not isinstance(
            obj, (str, bytes, int, float, bool, type(None))
        ):
            # Arbitrary object (e.g. KJT): recurse into __dict__ like pickle.
            for v in vars(obj).values():
                walk(v)

    walk(batch)
    # 16-byte alignment slack per tensor + headroom
    return int(total * factor) or 1


def init_pool_from_batch(batch: Any) -> None:
    """Pin-thread entry: size the pool from *batch*, create slots, signal workers.

    Called once on the first batch the pin thread sees.  Measures the large-
    tensor footprint, creates ``_N_SLOTS`` persistent SharedMemory regions,
    publishes the slot size, and sets the ready Event so workers attach and
    begin pooling on subsequent batches.
    """
    global _SLOTS, _SLOT_BYTES
    if _SLOTS:
        return
    assert _N_SLOTS > 0 and _POOL_READY is not None and _SLOT_BYTES_VAL is not None
    slot_bytes = estimate_slot_bytes(batch)
    _SLOTS = _get_or_create_slots(_N_SLOTS, slot_bytes)
    _SLOT_BYTES = slot_bytes
    _SLOT_BYTES_VAL.value = slot_bytes
    _POOL_READY.set()


# ---------------------------------------------------------------------------
# worker side: attach once, write data into a slot, return a lightweight stub
# ---------------------------------------------------------------------------
def _ensure_attached() -> None:
    """Worker: wait for the pool to be ready, then attach to the slots once."""
    global _SLOTS, _SLOT_BYTES
    if _SLOTS:
        return
    assert _POOL_READY is not None and _SLOT_BYTES_VAL is not None
    _POOL_READY.wait()  # back-pressure until the pin thread has sized the pool
    _SLOT_BYTES = _SLOT_BYTES_VAL.value
    for s in range(_N_SLOTS):
        shm = shared_memory.SharedMemory(
            name=f"__dataloader_shmpool_{_RANK}_{s}", create=False
        )
        _SLOTS.append(shm)


def _has_slots(obj: Any) -> bool:
    """Whether *obj* (or any class in its MRO) declares ``__slots__``."""
    for cls in type(obj).__mro__:
        if getattr(cls, "__slots__", None):
            return True
    return False


def _instance_attrs(obj: Any) -> dict:
    """Collect all per-instance attributes of *obj* (``__dict__`` + ``__slots__``).

    Objects may carry state in ``__slots__`` (e.g. ZeusKeyedJaggedTensor stores
    ``_zeus_show_click_stats`` there) which ``vars()`` misses because slot
    attributes are not stored in ``__dict__``.  Walk every class in the MRO,
    union its ``__slots__`` entries with ``vars(obj)``, and return the merged
    dict so pooled views are restored for *all* instance attributes.
    """
    attrs = {}
    d = getattr(obj, "__dict__", None)
    if d:
        attrs.update(d)
    for cls in type(obj).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for s in slots:
            if hasattr(obj, s):
                attrs[s] = getattr(obj, s)
    return attrs


class _BatchTooLarge(Exception):
    """Internal sentinel: the batch overflowed the slot mid-walk."""


class SlotStub:
    """Lightweight stub that crosses the Queue without triggering reduction.

    Contains only the slot index + per-tensor metadata (offset, nbytes, dtype,
    shape) + any small (unpooled) fields.  No large tensor → no
    ``reduce_storage`` → no new ``/dev/shm`` fd.
    """

    __slots__ = ("slot_idx", "meta", "small")

    def __init__(self, slot_idx: int, meta: list, small: Any):
        self.slot_idx = slot_idx
        self.meta = meta  # list of (key_path, offset, nbytes, np_dtype, shape)
        self.small = small  # unpooled small data (pickled normally)


def _alloc_slot() -> int:
    """Get a free slot index from the free-list queue."""
    assert _FREE is not None
    return _FREE.get()


def ret_slot(slot_idx: int) -> None:
    """Return a slot to the free-list queue."""
    assert _FREE is not None
    _FREE.put(slot_idx)


def _write_tensor_into_slot(
    shm: shared_memory.SharedMemory, offset: int, tensor: torch.Tensor
) -> Tuple[int, int, Any, tuple]:
    """Write a tensor into the slot at *offset* (single ``torch.copy_``).

    ``torch.copy_`` is ~8-10x faster than ``numpy.copyto`` here (1.9 ms vs
    19 ms per 160 MB on H20) — pure-C, SIMD-aware memcpy, no numpy-ufunc
    overhead.

    Returns (offset, nbytes, np_dtype, shape).
    """
    t = tensor.contiguous()
    npt = _TORCH_TO_NP.get(t.dtype, t.numpy().dtype)
    nbytes = t.element_size() * t.numel()
    dst = torch.from_numpy(
        numpy.ndarray(t.shape, dtype=npt, buffer=shm.buf, offset=offset)
    )
    dst.copy_(t)  # single C-layer copy
    return offset, nbytes, npt, tuple(t.shape)


def copy_batch_into_slot(batch: Any) -> Any:
    """Walk *batch*, write every CPU tensor into a free slot, return a SlotStub.

    If the pool is not yet ready (first batch) or the batch is larger than the
    slot, the batch is returned unchanged so it crosses the Queue via the
    normal ``file_descriptor`` path (graceful degradation).
    """
    if _POOL_READY is None or not _POOL_READY.is_set():
        return batch  # pool not sized yet — first batch, send normally
    _ensure_attached()

    slot_idx = _alloc_slot()
    shm = _SLOTS[slot_idx]
    offset = 0
    meta: list = []

    def walk(obj: Any, path: tuple) -> Any:
        nonlocal offset
        if isinstance(obj, torch.Tensor) and not obj.is_cuda:
            nb = obj.element_size() * obj.numel()
            # mid-walk capacity bail-out: a variable-length spike (e.g. KJT
            # _values) would overflow the slot.  Drop everything written so far
            # and return the original batch so it crosses the Queue via the
            # normal file_descriptor path (graceful degradation, one pass).
            if offset + nb > _SLOT_BYTES:
                raise _BatchTooLarge
            off, _, npt, shape = _write_tensor_into_slot(shm, offset, obj)
            meta.append((path, off, nb, npt, shape))
            offset = (offset + nb + 15) & ~15  # 16-byte align
            return None  # placeholder; rebuilt on pin side
        elif isinstance(obj, dict):
            return {k: walk(v, path + (k,)) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [walk(v, path + (i,)) for i, v in enumerate(obj)]
        elif isinstance(obj, tuple):
            # tuple is immutable — can't setitem views on rebuild.  Carry it as
            # a marked mutable list so _set_leaf/_descend work, then convert
            # back to tuple at the end of rebuild_batch_from_slot.
            return ("__tuple__", [walk(v, path + (i,)) for i, v in enumerate(obj)])
        elif not isinstance(
            obj, (str, bytes, int, float, bool, type(None))
        ) and (hasattr(obj, "__dict__") or _has_slots(obj)):
            # Arbitrary object (e.g. KJT/SparseBatch): recurse into all instance
            # attributes (``__dict__`` + ``__slots__``) like pickle.  Tensors
            # found inside are pooled; the object is preserved and tensors are
            # restored as zero-copy views on the pin side via setattr.  This
            # mirrors how pickle/reduce_storage discovers tensors inside
            # arbitrary containers.
            walked = {k: walk(v, path + (k,)) for k, v in _instance_attrs(obj).items()}
            return ("__obj__", type(obj), walked)
        return obj

    try:
        small = walk(batch, ())
    except _BatchTooLarge:
        ret_slot(slot_idx)  # release the slot we grabbed but won't use
        return batch  # degrade this batch to normal file_descriptor path
    return SlotStub(slot_idx, meta, small)


# ---------------------------------------------------------------------------
# pin-thread side: rebuild zero-copy views from the slot
# ---------------------------------------------------------------------------
def rebuild_batch_from_slot(stub: SlotStub) -> Any:
    """Rebuild the batch from a SlotStub as zero-copy views into the slot.

    ``del`` of the returned tensors does not munmap the underlying mapping
    (the pool holds the SharedMemory permanently).  ``__obj__`` placeholders
    left by walk are reconstructed into real objects first, then their tensor
    attributes are restored as views.
    """
    assert _SLOTS, "init_pool_from_batch() must have run first"
    shm = _SLOTS[stub.slot_idx]
    data = _materialize(stub.small)

    for path, off, nb, npt, shape in stub.meta:
        arr = numpy.ndarray(shape, dtype=npt, buffer=shm.buf, offset=off)
        view = torch.from_numpy(arr)
        cur = data
        for key in path[:-1]:
            cur = _descend(cur, key)
        _set_leaf(cur, path[-1], view)

    return _finalize_tuples(data)


def _materialize(obj: Any) -> Any:
    """Reconstruct real objects from ``__obj__`` placeholders left by walk.

    Recursively turns ``("__obj__", cls, walked_dict)`` back into ``cls``
    instances with their (already-materialized) attributes set via setattr.
    ``("__tuple__", walked_list)`` is kept as-is (its list is materialized)
    so views can be setitem'd; ``_finalize_tuples`` converts them back at the
    end.
    """
    if isinstance(obj, tuple) and len(obj) == 3 and obj[0] == "__obj__":
        _, cls, walked = obj
        try:
            instance = cls.__new__(cls)
        except TypeError:
            # Fallback: construct via __init__ with no args if possible.
            instance = cls()
        for k, v in walked.items():
            setattr(instance, k, _materialize(v))
        return instance
    elif isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__tuple__":
        return ("__tuple__", [_materialize(v) for v in obj[1]])
    elif isinstance(obj, dict):
        return {k: _materialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_materialize(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_materialize(v) for v in obj)
    return obj


def _finalize_tuples(obj: Any) -> Any:
    """Convert ``("__tuple__", list)`` placeholders back to real tuples.

    Called after all views have been setitem'd into the mutable lists.
    """
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__tuple__":
        return tuple(_finalize_tuples(v) for v in obj[1])
    if isinstance(obj, dict):
        return {k: _finalize_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finalize_tuples(v) for v in obj]
    # real objects (incl. SparseBatch/KJT): recurse into __dict__ so nested
    # tuples inside attributes are finalized too.
    if hasattr(obj, "__dict__") and not isinstance(
        obj, (str, bytes, int, float, bool, type(None))
    ):
        for k, v in list(vars(obj).items()):
            obj.__dict__[k] = _finalize_tuples(v)
    return obj


def _descend(container: Any, key: Any) -> Any:
    """Descend into *container* at *key*.

    After ``_materialize`` the ``__obj__`` placeholders are already real
    objects, so descend via ``getattr`` for arbitrary objects and ``[]`` for
    dict/list (the only container types ``walk`` produces for non-leaf paths).
    ``__tuple__`` placeholders carry a mutable list (tuples were converted so
    views can be setitem'd); descend into that list.
    """
    if isinstance(container, tuple) and len(container) == 2 and container[0] == "__tuple__":
        return container[1][key]
    if isinstance(container, dict):
        return container[key]
    if isinstance(container, list):
        return container[key]
    return getattr(container, key)


def _set_leaf(container: Any, key: Any, value: Any) -> None:
    """Set *value* at *key* in *container*."""
    if isinstance(container, tuple) and len(container) == 2 and container[0] == "__tuple__":
        container[1][key] = value
    elif isinstance(container, (list, dict)):
        container[key] = value
    else:
        setattr(container, key, value)


def is_slot_stub(obj: Any) -> bool:
    """Check whether *obj* is a SlotStub (used by the pin thread)."""
    return isinstance(obj, SlotStub)
