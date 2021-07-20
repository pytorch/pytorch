import torch

from typing import Any

class saved_tensors_default_hooks(object):
    r"""Context-manager that registers default hooks for Saved Tensors.
    """
    def __init__(self, pack_hook, unpack_hook):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        torch._C._autograd._register_default_hooks(self.pack_hook, self.unpack_hook)

    def __exit__(self, *args: Any) -> None:
        torch._C._autograd._reset_default_hooks()

class save_on_cpu(object):
    r"""Context-manager under which tensors saved by the forward pass will be
    stored on cpu, then retrieved for backward
    """
    def __init__(self):
        def pack_hook(tensor):
            return (tensor.device, tensor.cpu())

        def unpack_hook(packed):
            device, tensor = packed
            return tensor.to(device)

        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        torch._C._autograd._register_default_hooks(self.pack_hook, self.unpack_hook)

    def __exit__(self, *args: Any) -> None:
        torch._C._autograd._reset_default_hooks()
