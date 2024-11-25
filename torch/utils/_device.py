# mypy: allow-untyped-defs
from typing import Optional
import torch
from torch.overrides import TorchFunctionMode, _pop_mode, _push_mode
from torch.utils._contextlib import context_decorator
from torch._C import _len_torch_function_stack
import functools

CURRENT_DEVICE: Optional[torch.device] = None

@functools.lru_cache(1)
def _device_constructors():
    return {
        # standard ones
        torch.empty,
        torch.empty_permuted,
        torch.empty_strided,
        torch.empty_quantized,
        torch.ones,
        torch.arange,
        torch.bartlett_window,
        torch.blackman_window,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.full,
        torch.fill,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.nested.nested_tensor,
        # This function doesn't actually take a device argument
        # torch.normal,
        torch.ones,
        torch.rand,
        torch.randn,
        torch.randint,
        torch.randperm,
        torch.range,
        torch.sparse_coo_tensor,
        torch.sparse_compressed_tensor,
        torch.sparse_csr_tensor,
        torch.sparse_csc_tensor,
        torch.sparse_bsr_tensor,
        torch.sparse_bsc_tensor,
        torch.tril_indices,
        torch.triu_indices,
        torch.vander,
        torch.zeros,
        torch.asarray,
        # weird ones
        torch.tensor,
        torch.as_tensor,
        torch.scalar_tensor,
        torch.asarray,
    }

# NB: This is directly called from C++ in torch/csrc/Device.cpp
class DeviceContext(TorchFunctionMode):
    def __init__(self, device):
        self.device = torch.device(device)

    def __enter__(self):
        global CURRENT_DEVICE
        self.old_device = CURRENT_DEVICE
        CURRENT_DEVICE = self.device
        # We need to put the device at the bottom of the stack
        # If we set default device within a function mode context
        # exiting that context mode will pop the device function mode off
        # of the stack incorrectly
        cur_stack = [_pop_mode() for _ in range(_len_torch_function_stack())]

        _push_mode(self)

        for mode in reversed(cur_stack):
            _push_mode(mode)


    def __exit__(self, exc_type, exc_val, exc_tb):
        global CURRENT_DEVICE
        CURRENT_DEVICE = self.old_device
        cur_stack = []
        # Invariant: there should only be one DeviceContext on the stack at any time
        # (At the bottom), pop all mdoes until we hit the bottom, assert it's a DeviceContext
        # or else someone else has popped it!
        for _ in range(_len_torch_function_stack() - 1):
            mode = _pop_mode()
            assert not isinstance(mode, DeviceContext)
            cur_stack.append(mode)

        if _len_torch_function_stack() > 0:
            mode = _pop_mode()
            assert isinstance(mode, DeviceContext)

        for mode in reversed(cur_stack):
            _push_mode(mode)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _device_constructors() and kwargs.get('device') is None:
            kwargs['device'] = self.device
        return func(*args, **kwargs)

# NB: This is directly called from C++ in torch/csrc/Device.cpp
def device_decorator(device, func):
    return context_decorator(lambda: device, func)

def set_device(device):
    """
    Set the default device inside of the wrapped function by decorating it with this function.

    If you would like to use this as a context manager, use device as a
    context manager directly, e.g., ``with torch.device(device)``.
    """
    return lambda func: device_decorator(torch.device(device), func)
