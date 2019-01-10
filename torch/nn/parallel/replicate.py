import torch.cuda.comm as comm
from torch.cuda._utils import _get_device_index


def replicate(network, devices, detach=False):
    from ._functions import Broadcast

    devices = list(map(lambda x: _get_device_index(x, True), devices))
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    param_copies = Broadcast.apply(devices, *params)
    if len(params) > 0:
        param_copies = [param_copies[i:i + len(params)]
                        for i in range(0, len(param_copies), len(params))]

    buffers = list(network.buffers())
    buffer_indices = {buf: idx for idx, buf in enumerate(buffers)}
    buffer_copies = comm.broadcast_coalesced(buffers, devices)

    modules = list(network.modules())
    module_copies = [[] for device in devices]
    module_indices = {}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module.__new__(type(module))
            replica.__dict__ = module.__dict__.copy()
            replica._parameters = replica._parameters.copy()
            replica._buffers = replica._buffers.copy()
            replica._modules = replica._modules.copy()
            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = module_copies[j][module_idx]
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = param_copies[j][param_idx].detach() \
                        if detach else param_copies[j][param_idx]
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                buffer_idx = buffer_indices[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = buffer_copies[j][buffer_idx]

    return [module_copies[j][0] for j in range(num_replicas)]
