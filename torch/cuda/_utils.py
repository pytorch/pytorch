import torch
import torch._six


def _get_device_index(device, optional=False, allow_cpu=False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, torch._six.string_classes):
        device = torch.device(device)
    if isinstance(device, torch.device):
        dev_type = device.type
        if allow_cpu:
            if device.type not in {'cuda', 'cpu'}:
                raise ValueError('Expected a cuda or cpu device, but got: {}'.format(device))
        elif device.type != 'cuda':
            raise ValueError('Expected a cuda device, but got: {}'.format(device))
        device_idx = -1 if device.type == 'cpu' else device.index
    else:
        if device is not None and not isinstance(device, torch._six.int_classes):
            raise ValueError('Cannot recognize device {}'.format(device))
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return torch.cuda.current_device()
        else:
            raise ValueError('Expected a cuda device with a specified index '
                             'or an integer, but got: {}'.format(device))
    return device_idx
