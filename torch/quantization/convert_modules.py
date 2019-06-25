from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch.nn.quantized as nnq
import torch
from .observer import *

def _forward_hook(self, input, output):
    for module in self.children():
        module.forward(output)

def add_observer(module, qconfig_dict, qconfig_parent=None, prefix=''):
    r"""Transform a module to a quantized module according to qconfig_dict

    This function insert observer module to all leaf child module of a given
    module based on qconfig_dict.

    Args:
        module: instance of the module we want to transform
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict
    """
    assert not hasattr(module, 'qconfig'), 'Original module should not have \
        qconfig object already'
    module.qconfig = None
    if prefix in qconfig_dict:
        module.qconfig = qconfig_dict[prefix]
    else:
        module.qconfig = qconfig_parent

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if module.qconfig is not None and len(module._modules) == 0:
        # observer will be when we swap the module
        module.add_module('observer', module.qconfig.activation())
        # when do we remove hooks?
        module.observer_hook_handle = module.register_forward_hook(_forward_hook)

    for name, child in module.named_children():
        if name != 'observer':
            module_prefix = prefix + '.' + name if prefix else name
            add_observer(child, qconfig_dict, module.qconfig, module_prefix)

def add_quant_dequant_call(base_fn):
    r"""Insert calls to quant and dequant around base_fn calls

    Given a `base_fn` function, insert a call to `quant` before the function
    call which is supposed to quantize the input and a call to `dequant` after
    the function call which will dequantize the output of the function.

    Args:
        base_fn: base function we want to call

    Returns:
        new_fn: a modified function which calls quant -> base_bn -> dequant
    """
    def new_fn(x):
        x = base_fn.__self__.quant(x)
        x = base_fn(x)
        x = base_fn.__self__.dequant(x)
        return x
    return new_fn

class QuantWrapper(nn.Module):
    def __init__(self, module, qconfig):
        super(QuantWrapper, self).__init__()
        self.quant = QuantStub(qconfig)
        self.dequant = DeQuantStub()
        self.module = module

    def forward(self, X):
        self.quant(X)
        self.module.forward(X)
        self.dequant(X)

def get_config_key(name, qconfig_dict):
    if name in qconfig_dict:
        return name
    elif name:
        parent = '.'.join(name.split('.')[:-1])
        return get_config_key(parent, qconfig_dict)
    else:
        return '' if '' in qconfig_dict else None

def get_module(mod, name):
    if name == '':
        return mod
    splits = name.split('.')
    for split in splits:
        mod = mod._modules[split]
    return mod

def add_quant_dequant_module(module, qconfig_dict, name):
    qconfig = qconfig_dict[name]
    splits = name.split('.')
    parent_mododule = module
    for split in splits:
        parent_module = module
        module = module._modules[split]
    parent_module._modules[split[-1]] = QuantWrapper(module, qconfig)
    return module

def add_quant_dequant(module, qconfig_dict):
    r"""Add QuantStub and DeQuantStub module and modify forward function
    to call quant dequant accordign to qconfig_dict
    """
    mod_key_list = []
    for name, _ in module.named_modules():
        dict_key = get_config_key(name, qconfig_dict)
        if dict_key is not None:
            mod_key_list.append(dict_key)

    for name in set(mod_key_list):
        if name == '':
            module = QuantWrapper(module, qconfig_dict[name])
        else:
            add_quant_dequant_module(module, qconfig_dict, name)
    return module

def prepare(module, qconfig_dict):
    r"""Transform a module to a quantized module according to qconfig_dict,
    it adds observer and quant dequant stub modules to the module. and
    changes the forward method to invoke them

    Args:
        mod: instance of the module we want to transform
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
    """
    add_observer(module, qconfig_dict)
    return add_quant_dequant(module, qconfig_dict)

class QuantStub(nn.Module):
    r"""Quantize stub module which will be replace to actual Quantize module
    in swap_module function
    """
    def __init__(self, qconfig):
        super(QuantStub, self).__init__()
        self.add_module('observer', qconfig.activation())

    def forward(self, x):
        self.observer(x)
        return x

class DeQuantStub(nn.Module):
    r"""Dequantize stub module which will be replace to actual DeQuantize module
    in swap_module function
    """
    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x):
        return x

def quantize(module, qconfig_dict, eval_fn, *eval_args):
    r"""Given a float module and qconfig_dict, convert it to a quantized module

    First it will prepare the module to add observer and quant
    deqaunt stub modules, then it calls `eval_fn` which will run the calibration
    step and observers will record the stats of the tensors, after that we will
    call `convert` which will convert the module to a quantized module.
    """
    module = prepare(module, qconfig_dict)
    eval_fn(module, *eval_args)
    convert_to_quantized(module)
    return module

def convert_to_quantized(module):
    r"""Convert a module with qparams to a quantized version of the module
    """
    module_swapped = swap_module(module)
    if len(list(module.named_children())) == 0:
        return module_swapped

    reassign = {}
    for name, mod in module.named_children():
        new_mod = convert_to_quantized(mod)
        if new_mod is not mod:
            reassign[name] = new_mod

    for name, mod in reassign.items():
        setattr(module_swapped, name, mod)

    return module_swapped

DEFAULT_MODULE_MAPPING = {
    torch.nn.Linear: nnq.Linear,
    torch.nn.ReLU: nnq.ReLU,
}

STUB_MODULE_MAPPING = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize
}

def swap_module(mod, mapping=DEFAULT_MODULE_MAPPING):
    r""" Check if a module has a quantized counterpart and swap it.
    """
    new_mod = mod
    if hasattr(mod, 'observer'):
        if type(mod) in mapping:
            new_mod = mapping[type(mod)].from_float(mod)

    if type(mod) in STUB_MODULE_MAPPING:
        new_mod = STUB_MODULE_MAPPING[type(mod)].from_float(mod)

    # keep the modification to forward
    new_mod.forward = mod.forward
    return new_mod
