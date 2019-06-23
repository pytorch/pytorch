from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch.nn.quantized as nnq
import torch
from .observer import *

def _forward_hook(self, input, output):
    for module in self.children():
        module.forward(output)

def addObserver(myMod, qConfigDict, qConfigParent=None, prefix=''):
    r"""Transform a module to a quantized module according to qConfigDict

    This function insert observer module to all leaf child module of a given
    module based on qConfigDict.

    Args:
        myMod: instance of the module we want to transform
        qCofnigDict: dictionary that maps from name of submodule to quantization
                     configuration
        qConfigParent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qConfigDict
    """
    myMod.qConfig = None
    if prefix in qConfigDict:
        myMod.qConfig = qConfigDict[prefix]
    else:
        if qConfigParent is not None:
            myMod.qConfig = qConfigParent
    # Insert observers only for leaf nodes
    if myMod.qConfig is not None and len(myMod._modules) == 0:
        myMod.add_module('observer', myMod.qConfig.activation())
        myMod.register_forward_hook(_forward_hook)

    for name, child in myMod.named_children():
        if name is not 'observer':
            if prefix:
                module_prefix = prefix + '.' + name
            else:
                module_prefix = name
            addObserver(child, qConfigDict, myMod.qConfig, module_prefix)

def addQuantDeQuantCall(base_fn):
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

def addQuantDeQuantModule(mod, qConfig):
    r"""Add Quantize and DeQuantize to the input module and modify forward function

    Args:
        mod: module to be changed
        qConfig: quantization configuration

    """
    mod.add_module('quant', AbstractQuant(qConfig))
    mod.add_module('dequant', AbstractDeQuant())
    mod.forward = addQuantDeQuantCall(mod.forward)

def get_config_key(name, qConfigDict):
    if name in qConfigDict:
        return name
    elif name:
        parent = '.'.join(name.split('.')[:-1])
        return get_config_key(parent, qConfigDict)
    else:
        return '' if '' in qConfigDict else None

def getModule(modInstance, name):
    splits = name.split('.')
    mod = modInstance._modules[splits[0]]
    for i in range(1, len(splits)):
        mod = mod._modules[splits[i]]
    return mod

def addQuantDeQuant(modInstance, qConfigDict):
    r"""Add AbstractQuant and AbstractDeQuant module and modify forward function
    to call quant dequant accordign to qConfigDict
    """
    mod_key_list = []
    for name, mod in modInstance.named_modules():
        dict_key = get_config_key(name, qConfigDict)
        if dict_key is not None:
            mod_key_list.append(dict_key)

    for name in set(mod_key_list):
        if name is not '':
            addQuantDeQuantModule(getModule(modInstance, name), qConfigDict[name])
        else:
            addQuantDeQuantModule(modInstance, qConfigDict[name])

def prepare(modInstance, qConfigDict):
    r"""Transform a module to a quantized module according to qConfigDict,
    it adds observer and abstract quant dequant modules to the module. and
    changes the forward method to invoke them

    Args:
        modInstance: instance of the module we want to transform
        qCofnigDict: dictionary that maps from name of submodule to quantization
                     configuration
    """
    addObserver(modInstance, qConfigDict)
    addQuantDeQuant(modInstance, qConfigDict)

class AbstractQuant(nn.Module):
    r"""Abstract quantize module which will be replace to actual Quantize module
    in swapModule function
    """
    def __init__(self, qconfig):
        super(AbstractQuant, self).__init__()
        self.add_module('observer', qconfig.activation())

    def forward(self, x):
        self.observer(x)
        return x

class AbstractDeQuant(nn.Module):
    r"""Abstract dequantize module which will be replace to actual DeQuantize module
    in swapModule function
    """
    def __init__(self):
        super(AbstractDeQuant, self).__init__()

    def forward(self, x):
        return x

def calculateQParams(MyModule):
    r""" Calculates Quantizer parameters for activation tensors based on observer statistics
    Updates the qparams parameter for each module with the quantizer params.

    Args:
        MyModule: Model with observer stats
    """
    if hasattr(MyModule, 'observer'):
        if isinstance(MyModule.observer, Observer):
            # Simple symmetric quantization over entire observed range
            MyModule.qparams = MyModule.observer.calculate_qparams()
    for Module in MyModule.children():
        calculateQParams(Module)

def quantize(module, qConfigDict, eval_fn, *eval_args):
    r"""Given a float module and qConfigDict, convert it to a quantized module

    First it will prepare the module to add observer and abstract quant
    deqaunt modules, then it calls `eval_fn` which will run the calibration
    step and observers will record the stats of the tensors, after that we will
    call `convert` which will convert the module to a quantized module.
    """
    prepare(module, qConfigDict)
    eval_fn(module, *eval_args)
    convert(module)
    return

def convert(module):
    r""" Utility function to traverse module tree and swap modules marked with
         AbstractQuant and AbstractDeQuant with
         quantized versions.
    """
    calculateQParams(module)
    return convertToQuantized(module)

def convertToQuantized(module):
    r"""Convert a module with qparams to a quantized version of the module
    """
    module_swapped = swapModule(module)
    if len(list(module.named_children())) == 0:
        return module_swapped

    reassign = {}
    for name, mod in module.named_children():
        new_mod = convertToQuantized(mod)
        if new_mod is not mod:
            reassign[name] = new_mod

    for name, mod in reassign.items():
        setattr(module_swapped, name, mod)

    return module_swapped

DEFAULT_MODULE_MAPPING = {
    torch.nn.Linear: nnq.Linear,
    torch.nn.ReLU: nnq.ReLU,
}

ABSTRACT_MODULE_MAPPING = {
    AbstractQuant: nnq.Quantize,
    AbstractDeQuant: nnq.DeQuantize
}

def swapModule(mod, mapping=DEFAULT_MODULE_MAPPING):
    r""" Check if a module has a quantized counterpart and swap it.
    """
    new_mod = mod
    if hasattr(mod, 'observer'):
        if type(mod) in mapping:
            new_mod = mapping[type(mod)].from_float(mod)

    if type(mod) in ABSTRACT_MODULE_MAPPING:
        new_mod = ABSTRACT_MODULE_MAPPING[type(mod)].from_float(mod)

    # keep the modification to forward
    new_mod.forward = mod.forward
    return new_mod
