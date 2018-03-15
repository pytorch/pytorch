import torch.cuda.comm as comm


def replicate(network, devices):
    from ._functions import Broadcast

    devices = tuple(devices)
    num_replicas = len(devices)

    params = list(network.parameters())
    cuda_params = list(filter(lambda x: x.is_cuda, params))
    cpu_params = list(filter(lambda x: not x.is_cuda, params))
    cuda_param_indices = {param: idx for idx, param in enumerate(cuda_params)}
    cuda_param_copies = Broadcast.apply(devices, *cuda_params)
    if len(params) > 0:
        cuda_param_copies = [cuda_param_copies[i:i + len(cuda_params)]
                        for i in range(0, len(cuda_param_copies), len(cuda_params))]

    buffers = list(network._all_buffers())
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
                if param.is_cuda:
                    param_idx = cuda_param_indices[param]
                    for j in range(num_replicas):
                        replica = module_copies[j][i]
                        replica._parameters[key] = cuda_param_copies[j][param_idx]
                else:
                    for j in range(num_replicas):
                        replica = module_copies[j][i]
                        replica._parameters[key] = param
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
