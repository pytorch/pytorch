from .parallel_apply import parallel_apply
from .replicate import replicate
from .utils import _ensure_iterable
from torch.autograd import Variable


__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel']


def scatter(input, target_gpus):
    """Slices a given variable into approximately equal chunks and distributes
       them accross given GPUs
    """
    from .functions import Scatter
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return _ensure_iterable(Scatter(target_gpus)(obj))
        return tuple(zip(*map(scatter_map, obj)))
    return scatter_map(input)


def gather(outputs, target_device):
    """Gathers variables from different GPUs on a specified device
       (-1 means the CPU).
    """
    from .functions import Gather
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Variable):
            return Gather(target_device)(*outputs)
        return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)


def data_parallel(module, input, device_ids, output_device=None):
    """Distributes replicas of module accross gpus given in device_ids,
       slices the input and applies the copies in parallel.

       Outputs are concatenated on the same device as input (or on
       output_device if specified). Device id -1 means the CPU.
    """
    if not device_ids:
        return module(input)

    var_input = input
    while not isinstance(var_input, Variable):
        var_input = var_input[0]
    if output_device is None:
        output_device = -1 if not var_input.is_cuda else var_input.get_device()

    replicas = replicate(module, device_ids)
    inputs = scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)

