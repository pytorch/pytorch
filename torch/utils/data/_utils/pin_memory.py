r"""Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import queue

import torch
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper
from ..._pytree import tree_map


def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)

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
                    where=f"in pin memory thread for device {device_id}")
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
    def pin_mem(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.pin_memory(device)
        elif isinstance(tensor, (str, bytes)):
            return tensor
        elif hasattr(tensor, "pin_memory"):
            return tensor.pin_memory()
        else:
            return tensor
    return tree_map(pin_mem, data)
