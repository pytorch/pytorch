import torch
from torch.autograd import Variable
from ._functions import Scatter, Gather


def scatter(inputs, target_gpus, dim=0):
    """
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter(target_gpus, dim=dim)(obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple):
            return tuple(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return tuple(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict):
            return tuple(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return tuple(obj for targets in target_gpus)

    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim)
    if kwargs is None or len(kwargs) == 0:
        kwargs = tuple({} for _ in inputs)
    else:
        kwargs = scatter(kwargs, target_gpus, dim)[:len(inputs)]
    return inputs, kwargs


def gather(outputs, target_device, dim=0):
    """
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Variable):
            return Gather(target_device, dim=dim)(*outputs)
        if out is None:
            return None
        return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)
