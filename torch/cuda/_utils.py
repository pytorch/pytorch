import torch


def _get_device_index(device, optional=False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for CUDA device without sepecified index, i.e.,
    ``torch.devie('cuda')``, this will return the current default CUDA device if
    :attr:`optional` is ``True``.

    If :attr:`device` is a Python interger, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, torch.device):
        dev_type = device.type
        if device.type != 'cuda':
            raise ValueError('Expected a cuda device, but got: {}'.format(device))
        device_idx = device.index
    else:
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return torch.cuda.current_device()
        else:
            raise ValueError('Expected a cuda device with sepecified index or '
                             'an integer, but got: '.format(device))
    return device_idx
