# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **need** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import collections
import copy
import queue

import torch
from torch._utils import ExceptionWrapper

from . import MP_STATUS_CHECK_INTERVAL


def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device) -> None:
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.multiprocessing._set_thread_name("pt_data_pin")
    torch.accelerator.set_device_index(device_id)

    # Slot pool: when SHM_POOL=1, data arriving from workers may be a SlotStub
    # (lightweight metadata pointing into a persistent SharedMemory slot)
    # instead of real tensors. We rebuild zero-copy views from the slot, pin
    # them, and recycle the previous slot. ``del view`` does not munmap the
    # underlying mapping (the pool holds it permanently), avoiding the per-batch
    # /dev/shm fd churn + munmap that otherwise dominates the pin thread.
    # On the first batch (no stub yet) we measure it and size the pool.
    _shm_enabled = False
    _prev_slot = None
    try:
        from .shm_pool import is_enabled as _shm_is_enabled
        from .shm_pool import is_slot_stub as _is_slot_stub
        from .shm_pool import rebuild_batch_from_slot as _rebuild_from_slot
        from .shm_pool import ret_slot as _ret_slot
        from .shm_pool import init_pool_from_batch as _init_pool_from_batch
        _shm_enabled = _shm_is_enabled()
    except ImportError:
        pass

    def do_one_step() -> None:
        nonlocal _prev_slot
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                if _shm_enabled:
                    if _is_slot_stub(data):
                        # Recycle the previous slot now that its views are gone
                        # (del happened when `data` was overwritten above — the
                        # previous views went out of scope without munmap).
                        if _prev_slot is not None:
                            _ret_slot(_prev_slot)
                        _prev_slot = data.slot_idx
                        data = _rebuild_from_slot(data)
                    else:
                        # First batch: measure it and size the pool so workers
                        # begin pooling on subsequent batches.
                        _init_pool_from_batch(data)
                data = pin_memory(data, device)
            except Exception:
                data = ExceptionWrapper(
                    where=f"in pin memory thread for device {device_id}"
                )
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()



def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()

    if hasattr(data, "pin_memory"):
        return data.pin_memory()

    if isinstance(data, (str, bytes)):
        return data

    if isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(data)
                clone.update(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
                return clone
            else:
                # pyrefly: ignore [bad-instantiation]
                return type(data)(
                    # pyrefly: ignore [bad-argument-count]
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}

    if isinstance(data, tuple):
        if hasattr(data, "_fields"):  # namedtuple
            return type(data)(*(pin_memory(sample, device) for sample in data))
        return type(data)(pin_memory(sample, device) for sample in data)

    if isinstance(data, collections.abc.Sequence):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, item in enumerate(data):
                    clone[i] = pin_memory(item, device)
                return clone
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [pin_memory(sample, device) for sample in data]

    return data
