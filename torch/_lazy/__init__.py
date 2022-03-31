import torch._C._lazy


def mark_step(device: str = "lazy:0", wait=False):
    """Triggers a mark step, which amounts to
    - collecting a group of 'live' lazy tensors to index into the compilation cache
      (lowering/compiling their IR graphs if not cached)
    - kicking off execution of the compiled function
    - (optionally, wait=True) waiting for cpu-side execution to complete (does not sync the accelerator)
    """
    # TODO(whc) expand this to include backend hooks and align with XLA backend needs
    torch._C._lazy._mark_step(device, [], wait=wait)

def wait_device_ops(devices=None):
    """Waits for all the async operations on the given devices to complete.
    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
    if devices is None:
        devices = []
    torch._C._lazy._wait_device_ops(devices=devices)

def sync_multi(tensors, devices):
    """
    Sync the list of lazy tensors so there IR get lowered for the activate backend
    and the compiled computation graph get cached.
    """
    torch._C._lazy._sync_multi(tensors, devices)

def get_tensor_id(tensor):
    """Return a unique id of the lazy tensor maintained by LTC"""
    return torch._C._lazy._get_tensor_id(tensor)

def get_tensors_text(tensors):
    """Return a dump of LTC IRs for the tensors"""
    return torch._C._lazy._get_tensors_text(tensors)

def get_tensors_dot(tensors):
    """Return a text dump of the LTC IR graph in dot format for the tensors.
       The text can be processed by tools like dot to be rendered in pdf,png etc."""
    return torch._C._lazy._get_tensors_dot(tensors)

def get_tensors_backend(tensors):
    """Return a dump of the current activate backend's IRs for the tensors"""
    return torch._C._lazy._get_tensors_backend(tensors)
