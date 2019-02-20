import torch.cuda.comm as comm
from torch.cuda._utils import _get_device_index


def _is_script_module(module):
    import torch.jit
    return isinstance(module, torch.jit.ScriptModule)


def _init_script_module():
    import torch.jit
    return torch.jit.ScriptModule()


def _is_jit_enabled():
    import torch.jit
    return torch.jit._enabled


# Check if we can safely replicate the module.
# there are three types of module:
# 1. python modules
# 2. weak python modules (nn.Module annotated by @weak_module)
# 3. ScriptModule
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


def _build_param_dict(modules, module_copies, module_indices):
    param_dict = {}
    for module in modules:
        if not _is_script_module(module):
            continue
        replica = module_copies[module_indices[module]]
        for name, param in module.named_parameters(recurse=False):
            param_dict[param] = (replica, name)
        for name, buffer in module.named_buffers(recurse=False):
            param_dict[buffer] = (replica, name)
    return param_dict


def _copy_scriptmodule_methods(modules, module_copies, module_indices):
    param_dict = _build_param_dict(modules, module_copies, module_indices)
    for i, module in enumerate(modules):
        if not _is_script_module(module):
            continue
        replica = module_copies[i]
        for method_name in module._method_names():
            method = module._get_method(method_name)
            param_list = []
            for param in method.params():
                param_list.append(param_dict[param])
            replica._copy_method(method_name, param_list, module)


def replicate(network, devices, detach=False):
    from ._functions import Broadcast

    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

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
    scriptmodule_skip_attr = {"_parameters", "_buffers", "_modules"}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            if _is_script_module(module):
                # we have to initialize ScriptModule properly so that
                # it works with pybind11
                replica = _init_script_module()
                keys = set(module.__dict__.keys()) - scriptmodule_skip_attr
                for key in keys:
                    replica.__dict__[key] = module.__dict__[key]
            else:
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

    for j in range(num_replicas):
        _copy_scriptmodule_methods(modules, module_copies[j], module_indices)

    return [module_copies[j][0] for j in range(num_replicas)]
