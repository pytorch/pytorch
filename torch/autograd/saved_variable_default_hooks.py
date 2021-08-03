import torch
from typing import Any

def set_saved_tensors_default_hooks(pack_hook, unpack_hook):
    torch._C._autograd._register_default_hooks(pack_hook, unpack_hook)

def reset_saved_tensors_default_hooks():
    torch._C._autograd._reset_default_hooks()

class save_on_cpu(object):
    """"Context-manager under which tensors saved by the forward pass will be
    stored on cpu, then retrieved for backward.

    When performing operations within this context manager, intermediary
    results saved in the graph during the forward pass will be moved to CPU,
    then copied back to the original device when needed for the backward pass.
    If the graph was already on CPU, no tensor copy is performed.

    Use this context-manager to tradeoff speed for less GPU memory usage.

    Args:
        pin_memory (bool): If ``True`` tensors will be saved to CPU pinned memory
                           during packing and copied to GPU asynchronously during unpacking.
                           Defaults to ``False``.
                           Also see :ref:`cuda-memory-pinning`.


    Example::

        >>> a = torch.randn(5, requires_grad=True, device="cuda")
        >>> b = torch.randn(5, requires_grad=True, device="cuda")
        >>> c = torch.randn(5, requires_grad=True, device="cuda")
        >>> d = a * b # a and b are saved in the graph (on GPU)
        >>> with torch.autograd.graph.save_on_cpu():
        ...    e = d * c # d and c are saved on CPU
        >>> f = a * e # a and e are saved on GPU
        >>> del a, b, c, d, e
        >>> # the content of a, b, e are still alive on GPU
        >>> # the content of c and d only live on CPU

    """
    def __init__(self, pin_memory=False):
        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu())

            storage = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(torch.cuda.is_available() and not tensor.is_sparse))
            storage.copy_(tensor)
            return (tensor.device, storage)

        def unpack_from_cpu(packed):
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        self.pack_hook = pack_to_cpu
        self.unpack_hook = unpack_from_cpu

    def __enter__(self):
        torch._C._autograd._register_default_hooks(self.pack_hook, self.unpack_hook)

    def __exit__(self, *args: Any):
        torch._C._autograd._reset_default_hooks()
