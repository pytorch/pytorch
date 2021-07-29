import torch

def set_saved_tensors_default_hooks(pack_hook, unpack_hook):
    torch._C._autograd._register_default_hooks(pack_hook, unpack_hook)

def reset_saved_tensors_default_hooks():
    torch._C._autograd._reset_default_hooks()

def set_save_on_cpu_hooks(pin_memory=False):
    """Sets pack_to_cpu / unpack_from_cpu hooks for saved tensors.

    When these hooks are set, intermediary results saved in the graph during
    the forward pass will be moved to CPU, then copied back to the original device
    when needed for the backward pass. If the graph was already on CPU, no tensor copy
    is performed.

    Use this hook to tradeoff speed for less GPU memory usage.
    You can set these hooks once before creating the graph; or you can control
    which part of the graph should be saved on CPU by registering these hooks
    before - and resetting them after - creating the part of the graph to be saved
    on CPU.

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
        >>> torch.autograd.graph.set_save_on_cpu_hooks()
        >>> e = d * c # d and c are saved on CPU
        >>> torch.autograd.graph.reset_saved_tensors_default_hooks()
        >>> f = a * e # a and e are saved on GPU
        >>> del a, b, c, d, e
        >>> # the content of a, b, e are still alive on GPU
        >>> # the content of c and d only live on CPU

    """
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

    torch._C._autograd._register_default_hooks(pack_to_cpu, unpack_from_cpu)
