# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import collections
import copy
import queue

import torch
from torch._utils import ExceptionWrapper

from . import MP_STATUS_CHECK_INTERVAL


def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.multiprocessing._set_thread_name("pt_data_pin")

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)
    elif device is None:
        torch.accelerator.set_device_idx(device_id)

    def do_one_step():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
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
        return data.pin_memory(device)
    elif isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)
                clone.update(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
                return clone
            else:
                return type(data)({k: pin_memory(sample, device) for k, sample in data.items()})  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(*(pin_memory(sample, device) for sample in data))
    elif isinstance(data, tuple):
        return [
            pin_memory(sample, device) for sample in data
        ]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
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
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data
