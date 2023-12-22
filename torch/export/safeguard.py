
import torch
from torch.overrides import TorchFunctionMode

class GradModeUnsupportedSafeguard(TorchFunctionMode):

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unsupported_grad_mode_ops = [
            torch._C._set_grad_enabled,
        ]
        if func in unsupported_grad_mode_ops:
            raise RuntimeError(f"{func} is not supported for grad mode")
        return func(*args, **kwargs)
