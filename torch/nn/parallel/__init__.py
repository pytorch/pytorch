from .functions import Gather, Scatter
from .parallel_apply import parallel_apply
from .replicate import replicate


__all__ = ['replicate', 'split', 'parallel_apply', 'join', 'data_parallel']


def scatter(variable, target_gpus):
    return Scatter(target_gpus)(variable)


def gather(variables, target_gpu):
    return Gather(target_gpu)(*variables)


def data_parallel(module, input, device_ids):
    if not device_ids:
        return module(input)
    replicas = replicate(module, device_ids)
    inputs = scatter(input, device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, device_ids[0])

