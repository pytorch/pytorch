import torch
from torch.overrides import TorchFunctionMode

_DEVICE_CONSTRUCTOR = {
    # standard ones
    torch.empty,
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
}

class DeviceMode(TorchFunctionMode):
    def __init__(self, device):
        self.device = torch.device(device)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in _DEVICE_CONSTRUCTOR:
            if kwargs.get('device') is None:
                kwargs['device'] = self.device
            return func(*args, **kwargs)
        return func(*args, **kwargs)

GLOBAL_DEVICE_MODE = None

def set_default_tensor_device(device):
    global GLOBAL_DEVICE_MODE
    if GLOBAL_DEVICE_MODE is not None:
        GLOBAL_DEVICE_MODE.__exit__(None, None, None)
    if device is None:
        GLOBAL_DEVICE_MODE = None
        return
    GLOBAL_DEVICE_MODE = DeviceMode(device)
    GLOBAL_DEVICE_MODE.__enter__()
