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


def _to_device_index(devices):
    if not devices:
        raise RuntimeError("Cannot replicate using an empty device list.")

    if isinstance(devices, list) and isinstance(devices[0], list):
        device_ids = []
        seen = set()
        for i, replica_devs in enumerate(devices):
            assert len(replica_devs) == len(devices[0]), (
                "Cannot replicate to unidentical number of devices, but got "
                "device list {} and {} for replica {} and {}."
            ).format(devices[0], devices[i], 0, i)

            assert len(seen.intersection(replica_devs)) == 0, (
                "Devices {} are shared by multiple replicas."
            ).format(seen.intersection(replica_devs))
            seen.update(replica_devs)

            device_ids.append(_to_device_index(replica_devs))
        return device_ids
    else:
        assert len(devices) == len(set(devices)), (
            "Duplicated device ids {}."
        ).format(devices)

        return list(map(lambda x: _get_device_index(x, True), devices))


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
            for param in method.initial_ivalues():
                param_list.append(param_dict[param])
            replica._copy_method(method_name, param_list, module)


# Group tensors on the same device together, which can later be broadcast to
# a list of devices. For example,consider 5 tensors on 2 devices
#   a = torch.Tensor(0).cuda(0)
#   b = torch.Tensor(0).cuda(0)
#   c = torch.Tensor(0).cuda(1)
#   d = torch.Tensor(0).cuda(0)
#   e = torch.Tensor(0).cuda(1).
# Let inputs be
#   tensors = [a, b, c, d, e] and
#   devices = [[0, 1], [2, 3]].
# Then, outputs will be:
#   grouped_tensors = [[a, b, d], [c, e]],
#   grouped_devices = [[0, 2], [1, 3]],
#   original_index = [[0, 1, 3], [2, 4]],
# meaning that grouped_tensors[i] will be broadcast to grouped_devices[i].
def _group_by_device(tensors, devices):
    if isinstance(devices[0], list):
        # all tensor devices must appear in devices[0]
        missing_devs = [t.device.index for t in tensors
                        if t.device.index not in devices[0]]
        assert not missing_devs, (
            "tensor devices {} are missing from devices[0] {}."
        ).format(missing_devs, devices[0])

        # device id to output group index, this is necessary when `tensors` only
        # use a subset of devices in `devices[0]`
        dev_to_group_idx = {}
        for t in tensors:
            if t.device.index not in dev_to_group_idx:
                dev_to_group_idx[t.device.index] = len(dev_to_group_idx)

        # Group tensors by devices and remember each tensor's original index.
        # The original_index helps to recover the original input tensor order
        # from grouped tensors.
        grouped_tensors = [[] for _ in range(len(dev_to_group_idx))]
        original_index = [[] for _ in range(len(dev_to_group_idx))]
        for i, t in enumerate(tensors):
            group_id = dev_to_group_idx[t.device.index]
            original_index[group_id].append(i)
            grouped_tensors[group_id].append(t)

        # group devices together if they should be in the same broadcast call
        grouped_devices = [[] for _ in range(len(dev_to_group_idx))]
        transpose = list(zip(*devices))
        for row in transpose:
            if row[0] in dev_to_group_idx:
                grouped_devices[dev_to_group_idx[row[0]]] = list(row)

        return grouped_tensors, grouped_devices, original_index
    else:
        return [tensors], [devices], [list(range(len(tensors)))]


# Return len(devices) replicas of input tensors. If input tensors reside on
# multiple GPUs, devices must be a 2D list with devices[0] matching input
# tensors' devices. For example,consider 5 tensors on 2 devices
#   a = torch.Tensor(0).cuda(0)
#   b = torch.Tensor(0).cuda(0)
#   c = torch.Tensor(0).cuda(1)
#   d = torch.Tensor(0).cuda(0)
#   e = torch.Tensor(0).cuda(1).
# Let inputs be
#   tensors = [a, b, c, d, e] and
#   devices = [[0, 1], [2, 3]].
#
# The output will be a 2D list of tensors:
#   [[a0, b0, c0, d0, e0],
#    [a1, b1, c1, d1, e1]], where
# a0, b0, d0 are on device 0
# a1, b1, d1 are on device 2
# c0, e0 are on device 1
# c1, e1 are on device 3
#
# This example will be used throughout the implementation of this function.
def _broadcast_coalesced_reshape(tensors, devices, detach=False):
    from ._functions import Broadcast

    # a triply-nested list of 1) broadcast group, 2) tensor list replica,
    # 3) tensors on the same device.
    grouped_replicas = []
    grouped_tensors, grouped_devices, original_index = \
        _group_by_device(tensors, devices)
    # For the example input described above, we have
    # grouped_tensors =[[a, b, d], [c, e]]
    # grouped_devices = [[0, 2], [1, 3]]
    # original_index = [[0, 1, 3], [2, 4]]
    for tensor_group, device_group in zip(grouped_tensors, grouped_devices):
        if detach:
            grouped_replicas.append(
                comm.broadcast_coalesced(tensor_group, device_group))
        else:
            if len(tensor_group) > 0:
                # Use the autograd function to broadcast if not detach
                tensor_copies = Broadcast.apply(device_group, *tensor_group)
                grouped_replicas.append(
                    [tensor_copies[i:i + len(tensor_group)]
                        for i in range(
                            0, len(tensor_copies), len(tensor_group))])
            else:
                grouped_replicas.append([])

    if isinstance(devices[0], list):
        # convert the triply-nested list into a doubly-nested list of 1) replica
        # 2) tensors in the same replica (can be on different devices)
        #
        # For the example input described above, we have
        #   grouped_replicas = [
        #       [[a0, b0, d0],   # on device 0
        #        [a1, b1, d1]],  # on device 2
        #       [[c0, e0],       # on device 1
        #        [c1, e1]]       # on device 3
        #   ]
        #
        # The code below re-organize elements in grouped_replicas to the
        # expected form:
        #   [[a0, b0, c0, d0, e0],
        #    [a1, b1, c1, d1, e1]].
        transpose = [0 for _ in tensors]
        for g_idx in range(len(original_index)):
            for t_idx in range(len(original_index[g_idx])):
                # g_idx is the broadcast group index.
                # t_idx is the tensor's index in a replica within a group.
                # Tensors in grouped_replicas[g_idx, :, t_idx] are replicas of
                # input tensor[original_index[g_idx][t_idx]]. Retrieve the
                # column and add it as the original_index[g_idx][t_idx]'s row in
                # transpose.
                transpose[original_index[g_idx][t_idx]] = \
                    [replica[t_idx] for replica in grouped_replicas[g_idx]]

        # transpose the result to stay consistent with the 1D devices case.
        return list(zip(*transpose))
    else:
        return grouped_replicas[0]


def replicate(network, devices, detach=False):
    r"""Replicate the input :attr:`network` to given :attr:`devices`. If
    :attr:`network` resides on CPU or a single GPU, :attr:`devices` must be a 1D
    list of destination devices. If :attr:`network` resides on multiple GPUs,
    :attr:`devices` must be satisfy the following conditions:

    1. :attr:`devices` must be a 2D list,
    2. ``devices[0]`` must match the :attr:`network`'s devices, in any order.
    3. All ``devices[i]`` must have the same length.

    For example, :attr:`network` is a ``Sequential`` module with two ``Linear``
    layers stored on ``cuda:0`` and ``cuda:1`` respectively. Setting
    :attr:`devices` to ``[[0, 1], [2, 3], [4, 5]]`` will replicate
    :attr:`network` three times with replicas stored on devices
    ``[cuda:0, cuda:1]``, ``[cuda:2, cuda:3]``, and ``[cuda:4, cuda:5]``
    respectively.


    Args:
        network (Module): modules to be replicate
        devices (1D or 2D list of int or torch.device): CUDA devices
        detach (bool, optional): detached replicas from the current graph.
    """
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

    devices = _to_device_index(devices)
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
                    replica._parameters[key] = param_copies[j][param_idx]
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
                    replica._buffers[key] = buffer_copies[j][buffer_idx]

    for j in range(num_replicas):
        _copy_scriptmodule_methods(modules, module_copies[j], module_indices)

    return [module_copies[j][0] for j in range(num_replicas)]
