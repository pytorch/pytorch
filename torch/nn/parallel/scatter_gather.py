from torch.autograd import Variable
from ._functions import Scatter, Gather


def scatter(input, target_gpus):
    """Slices a given variable into approximately equal chunks and distributes
       them accross given GPUs
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter(target_gpus)(obj)
        return tuple(zip(*map(scatter_map, obj)))
    return scatter_map(input)


def gather(outputs, target_device):
    """Gathers variables from different GPUs on a specified device
       (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Variable):
            return Gather(target_device)(*outputs)
        return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)
