from copy import copy
from collections import OrderedDict

from ..modules import Module
import torch.cuda.comm as comm


def _replicate_module(module, gpu, param_remap):
    if module is None:
        return module
    replica = copy(module)
    replica._parameters = OrderedDict()
    for key, param in module._parameters.items():
        replica._parameters[key] = param_remap.get(param)
    replica._buffers = {}
    for key, buffer in module._buffers.items():
        replica._buffers[key] = param_remap.get(buffer)
    if replica._modules:
        replica._modules = OrderedDict()
        for name, child in module._modules.items():
            replica._modules[name] = _replicate_module(child, gpu, param_remap)
    return replica


def replicate(module, device_ids):
    from ._functions import Broadcast
    seen_params = set()
    param_remap = [{} for dev_id in device_ids]
    for param in module.parameters():
        if param in seen_params:
            continue
        seen_params.add(param)
        param_copies = Broadcast(device_ids)(param)
        for param_copy, remap in zip(param_copies, param_remap):
            remap[param] = param_copy
    for m in module.modules():
        for buffer in m._buffers.values():
            copies = comm.broadcast(buffer, device_ids)
            for buf_copy, remap in zip(copies, param_remap):
                remap[buffer] = buf_copy
    return [_replicate_module(module, device_id, remap)
            for device_id, remap in zip(device_ids, param_remap)]
