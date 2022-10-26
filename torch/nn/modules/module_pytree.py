import copy
import functools
from typing import List, Any

import torch
from ..modules.module import Module
from torch.utils._pytree import _register_pytree_node, _register_pytree_node_grad


def _named_parameters(mod, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    gen = mod._named_members(
        lambda module: module._parameters.items(),
        prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    for elem in gen:
        yield elem


def _build_tied_map(name_to_weight_with_repeat):
    tensor_mapping = {}
    for key, tensor in name_to_weight_with_repeat:
        if tensor in tensor_mapping:
            tensor_mapping[tensor].add(key)
        else:
            tensor_mapping[tensor] = {key}
    return tensor_mapping


def _build_dict_from_tied_map_values(tied_names, weights):
    mapping = {}
    for names, weight in zip(tied_names, weights):
        for name in names:
            mapping[name] = weight
    return mapping


def _make_functional_module(module, is_grad):
    params = module.parameters()
    buffers = module.buffers()
    build_maps = {id(t): torch.empty_like(t) for t in params}
    for t in buffers:
        build_maps[id(t)] = t if is_grad else torch.empty_like(t)
    return copy.deepcopy(module, memo=build_maps)


def _module_flatten(module: Module, is_grad):
    functional_module = _make_functional_module(module, is_grad)
    if is_grad:
        param_map = _build_tied_map(_named_parameters(module, remove_duplicate=False))
        return list(param_map.keys()), (functional_module, list(param_map.values()))
    else:
        param_map = _build_tied_map(_named_parameters(module, remove_duplicate=False))
        buffer_map = _build_tied_map(module.named_buffers(remove_duplicate=False))
        return list(param_map.keys()) + list(buffer_map.keys()),\
               (functional_module, list(param_map.values()), list(buffer_map.values()))


def _module_unflatten(parameters_and_buffers: List[torch.Tensor], module_and_names: Any, is_grad) -> Any:
    if is_grad:
        module, param_names = module_and_names
        buffers = None
        params = parameters_and_buffers  # all buffers saved in the context
    else:
        module, param_names, buffer_names = module_and_names
        params = parameters_and_buffers[:len(param_names)]
        buffers = _build_dict_from_tied_map_values(buffer_names, parameters_and_buffers[len(param_names):])

    for param in params:
        param._is_param = True
    params = _build_dict_from_tied_map_values(param_names, params)
    return torch.nn.utils.stateless.replace_parameters(module, params, buffers)

def _module_unflatten_tangent_type(params: List[torch.Tensor], module_and_names: Any) -> Any:
    _, tied_names = module_and_names
    out = {}
    for names, weight in zip(tied_names, params):
        for name in names:
            out[name] = weight
            break
    return out

_register_pytree_node(Module,
                      functools.partial(_module_flatten, is_grad=False),
                      functools.partial(_module_unflatten, is_grad=False))
_register_pytree_node_grad(Module,
                           functools.partial(_module_flatten, is_grad=True),
                           functools.partial(_module_unflatten, is_grad=True),
                           _module_unflatten_tangent_type)