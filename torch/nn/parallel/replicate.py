from . import comm
from torch._utils import _get_device_index

from collections import OrderedDict


def _is_script_module(module):
    import torch.jit
    return isinstance(module, torch.jit.ScriptModule)


def _is_script_method(module):
    import torch.jit
    return isinstance(module, torch._C.ScriptMethod)


def _init_script_module():
    import torch.jit
    return torch.jit.ScriptModule()


def _is_jit_enabled():
    import torch.jit
    return torch.jit._state._enabled


# Check if we can safely replicate the module.
# there are two types of module:
# 1. python modules
# 2. ScriptModule
#
# currently a module cannot be replicated properly if the descendants of
# any ScriptModule contains python module (type 1 above)
def _replicatable_module(module, memo=None):

    # module.modules() contains module itself as the first element
    def descendant_modules(module):
        gen = module.modules()
        next(gen)
        return gen

    if not _is_jit_enabled():
        return True
    if memo is None:
        memo = set()

    # memorize visited modules
    memo.add(module)
    if _is_script_module(module):
        memo.update(descendant_modules(module))
        return all(_is_script_module(descendant) for
                   descendant in descendant_modules(module))

    for child in module.children():
        # since any unreplicatable module will cause the check to return
        # False early, visited modules here can be safely ignored.
        if child in memo:
            continue
        if not _replicatable_module(child, memo):
            return False

    return True

def _broadcast_coalesced_reshape(tensors, devices, detach=False):
    from ._functions import Broadcast
    if detach:
        return comm.broadcast_coalesced(tensors, devices)
    else:
        # Use the autograd function to broadcast if not detach
        if len(tensors) > 0:
            tensor_copies = Broadcast.apply(devices, *tensors)
            return [tensor_copies[i:i + len(tensors)]
                    for i in range(0, len(tensor_copies), len(tensors))]
        else:
            return []


def replicate(network, devices, detach=False):
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

    devices = list(map(lambda x: _get_device_index(x, True), devices))
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    buffers = list(network.buffers())
    buffers_rg = []
    buffers_not_rg = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(buffers_not_rg, devices, detach=True)

    modules = list(network.modules())
    module_copies = [[] for device in devices]
    module_indices = {}
    scriptmodule_skip_attr = {"_parameters", "_buffers", "_modules", "forward", "_c"}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module._replicate_for_data_parallel()
            # This is a temporary fix for DDP. DDP needs to access the
            # replicated model parameters. It used to do so through
            # `mode.parameters()`. The fix added in #33907 for DP stops the
            # `parameters()` API from exposing the replicated parameters.
            # Hence, we add a `_former_parameters` dict here to support DDP.
            replica._former_parameters = OrderedDict()

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
                    setattr(replica, key, module_copies[j][module_idx])
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param = param_copies[j][param_idx]
                    # parameters in replicas are no longer leaves,
                    # so setattr them as non-parameter attributes
                    setattr(replica, key, param)
                    # expose the parameter for DDP
                    replica._former_parameters[key] = param
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    return [module_copies[j][0] for j in range(num_replicas)]
