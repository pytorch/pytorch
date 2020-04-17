import torch
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator


def _combined_found_inf_helper(optimizer, grad_scaler, device):

    found_inf_dict = grad_scaler._check_inf_per_device(optimizer)
    # Combines found_inf tensors from all devices. As in GradScaler.update(),
    # tensors are combined on the scale's device, which is an arbitrary but
    # reasonable choice that avoids new context creation.
    found_infs = [f.to(device, non_blocking=True) for f in found_inf_dict.values()]
    assert len(found_infs) > 0, "No inf checks were recorded in _check_inf_per_device."
    found_inf_combined = found_infs[0]
    if len(found_infs) > 1:
        with torch.no_grad():
            for i in range(1, len(found_infs)):
                found_inf_combined += found_infs[i]
    return _MultiDeviceReplicator(found_inf_combined)
