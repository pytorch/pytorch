import torch
from torch.autograd import Variable
from ._functions import Scatter, Gather


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(data):
        return Scatter.apply(target_gpus, None, dim, data)
    if isinstance(inputs, Variable):
        return scatter_map(inputs)
    assert not torch.is_tensor(inputs), "Tensors not supported in scatter."
    if isinstance(inputs, tuple) and len(inputs) > 0:
        return list(zip(*map(scatter_map, inputs)))
    elif isinstance(inputs, list) and len(inputs) > 0:
        return list(map(list, zip(*map(scatter_map, inputs))))
    elif isinstance(inputs, dict) and len(inputs) > 0:
        return list(map(type(inputs), zip(*map(scatter_map, inputs.items()))))
    else:
        return [inputs for targets in target_gpus]

    
def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather(outputs, target_device, dim=0):
    r"""
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Variable):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)
