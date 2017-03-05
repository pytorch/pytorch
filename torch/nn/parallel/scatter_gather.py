import torch
from torch.autograd import Variable
from ._functions import Scatter, Gather
from torch.cuda.comm import broadcast


def scatter(input, target_gpus, dim=0):
    """
    Slices variables into approximately equal chunks and
    distributes them accross given GPUs. Duplicates 
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter(target_gpus, dim=dim)(obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple) or isinstance(obj, list):
            return type(obj)(zip(*map(scatter_map, obj)))
        return tuple(obj for targets in target_gpus)

    return scatter_map(input)


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
