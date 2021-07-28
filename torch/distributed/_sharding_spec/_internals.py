import torch
from torch.distributed.utils import _parse_remote_device

def is_valid_device(device):
    """
    Checks if this is a valid local/remote device.
    """
    # Check for torch.device
    try:
        torch.device(device)
        return True
    except Exception:
        pass

    # Check for remote device.
    try:
        _parse_remote_device(device)
        return True
    except Exception:
        pass

    return False
