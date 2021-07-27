import torch

def set_saved_tensors_default_hooks(pack_hook, unpack_hook):
    torch._C._autograd._register_default_hooks(pack_hook, unpack_hook)

def reset_saved_tensors_default_hooks():
    torch._C._autograd._reset_default_hooks()
