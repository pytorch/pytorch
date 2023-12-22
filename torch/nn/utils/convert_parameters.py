import torch
from typing import Iterable, Optional


def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Flatten an iterable of parameters into a single vector.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Copy slices of a vector into an iterable of parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f'expected torch.Tensor, but got: {torch.typename(vec)}')
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""Check if the parameters are located on the same device.

    Currently, the conversion between model parameters and single vector form is not supported
    for multiple allocations, e.g. parameters in different GPUs/PrivateUse1s, or mixture of CPU/GPU/PrivateUse1.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """
    # Meet the first parameter
    support_device_types = ["cuda", torch._C._get_privateuse1_backend_name()]
    if old_param_device is None:
        old_param_device = param.get_device() if param.device.type in support_device_types else -1
    else:
        warn = False
        if param.device.type in support_device_types:  # Check if in same GPU/PrivateUse1
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device
