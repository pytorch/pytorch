import torch


def _get_device_index(device, optional=False):
    if isinstance(device, torch.device):
        dev_type = device.type
        if device.type != 'cuda':
            raise ValueError('Execpted a cuda device, but got: {}'.format(device))
        device = device.index
    if device is None:
        if optional:
            # default cuda device
            return current_device()
        else:
            raise ValueError('Execpted a cuda device or a device index, but got: None')
    return device
