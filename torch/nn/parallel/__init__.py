from .functions import Gather, Scatter
from .parallel_apply import parallel_apply
from .replicate import replicate


__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel']


def scatter(variable, target_gpus):
    """Slices a given variable into approximately equal chunks and distributes
       them accross given GPUs
    """
    return Scatter(target_gpus)(variable)


def gather(variables, target_device):
    """Gathers variables from different GPUs on a specified device
       (-1 means the CPU).
    """
    return Gather(target_device)(*variables)


def data_parallel(module, input, device_ids, output_device=None):
    """Distributes replicas of module accross gpus given in device_ids,
       slices the input and applies the copies in parallel.

       Outputs are concatenated on the same device as input (or on
       output_device if specified). Device id -1 means the CPU.
    """
    if not device_ids:
        return module(input)
    replicas = replicate(module, device_ids)
    inputs = scatter(input, device_ids)
    outputs = parallel_apply(replicas, inputs)
    if output_device is None:
        output_device = -1 if not input.is_cuda else input.get_device()
    return gather(outputs, output_device)

