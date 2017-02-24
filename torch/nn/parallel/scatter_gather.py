import torch
from torch.autograd import Variable
from ._functions import Scatter, Gather
from torch.cuda.comm import broadcast


def scatter(input, target_gpus, dim=0):
    """
    Slices a given variable into approximately equal chunks and distributes
      them accross given GPUs
    Duplicates references to objects that are not variables.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter(target_gpus, dim=dim)(obj)
        if isinstance(obj, tuple):
            return tuple(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return list(zip(*map(scatter_map, obj)))
        if torch.is_tensor(obj):
            return broadcast(obj, target_gpus)
        return tuple(obj for targets in target_gpus)
    return scatter_map(input)


def gather(outputs, target_device, dim=0):
    """Gathers variables from different GPUs on a specified device
       (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Variable):
            return Gather(target_device, dim=dim)(*outputs)
        if isinstance(out, tuple):
            return type(out)(map(gather_map, zip(*outputs)))
        if isinstance(out, list):
            return type(out)(map(gather_map, zip(*outputs)))

    return gather_map(outputs)
