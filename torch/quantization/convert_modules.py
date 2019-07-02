from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch.nn.quantized as nnq
import torch
from .observer import *

class QuantWrapper(nn.Module):
    def __init__(self, module, qconfig):
        super(QuantWrapper, self).__init__()
        self.quant = QuantStub(qconfig)
        self.dequant = DeQuantStub()
        self.module = module

    def forward(self, X):
        X = self.quant(X)
        self.module.forward(X)
        X = self.dequant(X)
        return X

def _observer_forward_hook(self, input, output):
    self.observer(output)

def add_observer(module, qconfig_dict, qconfig_parent=None, prefix=''):
    r"""Transforms a module to a quantized module according to qconfig_dict.

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
        # observer and hook will be gone after we swap the module
        module.add_module('observer', module.qconfig.activation())
        module.register_forward_hook(_observer_forward_hook)
        return QuantWrapper(module, module.qconfig)

    for name, child in module.named_children():
        if name != 'observer':
            module_prefix = prefix + '.' + name if prefix else name
            module._modules[name] = add_observer(child, qconfig_dict, module.qconfig, module_prefix)
    return module

def prepare(module, qconfig_dict):
    r"""Prepares the module for calibration according to qconfig_dict.

    The function adds observer and quant dequant stub modules to the module.
    It also adds the forward hooks to call quant and dequant methods.

    Args:
        mod: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                      configuration
    """
    return add_observer(module, qconfig_dict)

class QuantStub(nn.Module):
    r"""Quantize stub module which replaces the quantized module
    in `swap_module` function.
    """
    def __init__(self, qconfig):
        super(QuantStub, self).__init__()
        self.add_module('observer', qconfig.activation())

    def forward(self, x):
        self.observer(x)
        return x

class DeQuantStub(nn.Module):
    r"""Dequantizes stub module, which replaces the actual DeQuantize module
    in `swap_module` function.
    """
    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x):
        return x

# TODO: add quantizable_modules argument
def quantize(module, qconfig_dict, eval_fn, eval_args):
    r"""Converts a float module to quantized module.

    First it will prepare the module for calibration or training, then it calls
    `eval_fn` which will run the calibration step or training step,
    after that we will call `convert` which will convert the module to a
    quantized module.
    """
    module = prepare(module, qconfig_dict)
    eval_fn(module, *eval_args)
    convert_to_quantized(module)
    return module

def convert_to_quantized(module):
    r"""Converts the float module with qparams to a quantized module.
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

DEFAULT_SWAP_MODULE_MAPPING = {
    torch.nn.Linear: nnq.Linear,
    torch.nn.ReLU: nnq.ReLU,
}

STUB_MODULE_MAPPING = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize
}

def swap_module(mod, mapping=DEFAULT_SWAP_MODULE_MAPPING):
    r"""Swaps the module if it has a quantized counterpart.
    """
    new_mod = mod
    if hasattr(mod, 'observer'):
        if type(mod) in mapping:
            new_mod = mapping[type(mod)].from_float(mod)

    if type(mod) in STUB_MODULE_MAPPING:
        new_mod = STUB_MODULE_MAPPING[type(mod)].from_float(mod)

    return new_mod
