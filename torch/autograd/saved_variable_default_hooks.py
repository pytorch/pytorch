import torch

def set_saved_tensors_default_hooks(pack_hook, unpack_hook):
    torch._C._autograd._register_default_hooks(pack_hook, unpack_hook)

def reset_saved_tensors_default_hooks():
    torch._C._autograd._reset_default_hooks()

def set_save_on_cpu(save_on_cpu):
    if not save_on_cpu:
        torch._C._autograd._reset_default_hooks()
        return

    def pack_hook(tensor):
        storage = torch.empty(tensor.size(), pin_memory=torch.cuda.is_available())
        storage.copy_(tensor)
        return (tensor.device, storage)

    def unpack_hook(packed):
        device, tensor = packed
        return tensor.to(device, non_blocking=True)

    torch._C._autograd._register_default_hooks(pack_hook, unpack_hook)
