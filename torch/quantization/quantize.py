from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as qat
from .QConfig import default_qconfig


def propagate_qconfig_helper(module, qconfig_dict, qconfig_parent=None, prefix=''):
    r"""This is a helper function for `propagate_qconfig`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict

    Return:
        None, module is modified inplace with qconfig attached
    """
    if not hasattr(module, 'qconfig'):
        module.qconfig = qconfig_parent
        if qconfig_dict:
            if prefix in qconfig_dict:
                module.qconfig = qconfig_dict[prefix]
            elif type(module) in qconfig_dict:
                module.qconfig = qconfig_dict[type(module)]

    for name, child in module.named_children():
        module_prefix = prefix + '.' + name if prefix else name
        propagate_qconfig_helper(child, qconfig_dict, module.qconfig, module_prefix)

def propagate_qconfig(module, qconfig_dict=None):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
            configuration, qconfig applies to all submodules of a given
            module unless qconfig for the submodules are specified(when the
            submodule already has qconfig attribute)

    Return:
        None, module is modified inplace with qconfig attached
    """
    if qconfig_dict is None:
        qconfig_dict = {}
    propagate_qconfig_helper(module, qconfig_dict)

def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output
    """
    return self.observer(output)

DEFAULT_SKIP_LIST = [nn.Identity, nn.MaxPool2d]

def add_observer(module, skip_list=DEFAULT_SKIP_LIST):
    r"""Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize

    Return:
        None, module is modified inplace with added observer modules and
            forward_hooks
    """
    for child in module.children():
        add_observer(child)

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if hasattr(module, 'qconfig') and module.qconfig is not None and \
       len(module._modules) == 0 and type(module) not in skip_list:
        # observer and hook will be gone after we swap the module
        if type(module) == nnq.FloatFunctional:
            module.observer = module.qconfig.activation()
        else:
            module.add_module('observer', module.qconfig.activation())
        module.register_forward_hook(_observer_forward_hook)

class QuantWrapper(nn.Module):
    r"""A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
    def __init__(self, module):
        super(QuantWrapper, self).__init__()
        qconfig = module.qconfig if hasattr(module, 'qconfig') else None
        self.add_module('quant', QuantStub(qconfig))
        self.add_module('dequant', DeQuantStub(qconfig))
        self.add_module('module', module)
        self.train(module.training)

    def forward(self, X):
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)

def add_quant_dequant(module):
    r"""Wrap the leaf child module in QuantWrapper if it has a valid qconfig
    Note that this function will modify the children of module inplace and it
    can return a new module which wraps the input module as well.

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize

    Return:
        Either the inplace modified module with submodules wrapped in
        `QuantWrapper` based on qconfig or a new `QuantWrapper` module which
        wraps the input module, the latter case only happens when the input
        module is a leaf module and we want to quantize it.
    """
    if len(module._modules) == 0 and hasattr(module, 'qconfig') and module.qconfig:
        return QuantWrapper(module)

    for name, child in module.named_children():
        module._modules[name] = add_quant_dequant(child)
    return module

def prepare(model):
    r"""Prepares the model for calibration or training.
    Note that the model will be modified inplace but in case the input model
    is a leaf model, a wrapped model will be returned.

    Args:
        mod: input model
    Return:
        A model with qconfig propogated, observer and quant dequant or fake
        quant modules attached, a model that is ready for calibration or
        training
    """
    propagate_qconfig(model)
    add_observer(model)
    return model

class QuantStub(nn.Module):
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    def __init__(self, qconfig=None):
        super(QuantStub, self).__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return x

class DeQuantStub(nn.Module):
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.
    """
    def __init__(self, qconfig=None):
        super(DeQuantStub, self).__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return x

# Map for swapping float module to quantized ones
DEFAULT_MODULE_MAPPING = {
    nn.Linear: nnq.Linear,
    nn.ReLU: nnq.ReLU,
    nn.Conv2d: nnq.Conv2d,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # Generated modules:
    nn.Add: nnq.Add,
    # QAT modules:
    qat.Linear: nnq.Linear,
    qat.Conv2d: nnq.Conv2d,
}

DEFAULT_DYNAMIC_MODULE_MAPPING = {
    nn.Linear: nnqd.Linear
}

# Map for swapping float module to qat modules
DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: qat.Linear,
    nn.Conv2d: qat.Conv2d,
}

def quantize(model, run_fn, run_args, mapping=DEFAULT_MODULE_MAPPING):
    r"""Converts a float model to quantized model.

    First it will prepare the model for calibration or training, then it calls
    `run_fn` which will run the calibration step or training step,
    after that we will call `convert` which will convert the model to a
    quantized model.

    Args:
        model: input model
        run_fn: a function for evaluating the prepared model, can be a
            function that simply runs the prepared model or a training loop
        run_args: positional arguments for `run_fn`

    Return:
        A quantized model
    """
    model.eval()
    model = prepare(model)
    run_fn(model, run_args)
    convert(model, mapping)
    return model

DEFAULT_QCONFIG_DICT = {
    nn.Linear : default_qconfig
}

def quantize_dynamic(model, qconfig_dict=DEFAULT_QCONFIG_DICT, mapping=DEFAULT_DYNAMIC_MODULE_MAPPING):
    r"""Converts a float model to dynamic quantized model. Do dynamic training and output a quantized model.
    """
    model.eval()
    propagate_qconfig(model, qconfig_dict)
    convert(model, mapping)
    return model

def prepare_qat(model, mapping=DEFAULT_QAT_MODULE_MAPPING):
    model = prepare(model)
    model = convert(model, mapping)
    return model

def quantize_qat(model, run_fn, run_args, mapping=DEFAULT_QAT_MODULE_MAPPING):
    r"""Do quantization aware training and output a quantized model
    """
    model.train()
    model = prepare_qat(model, mapping)
    run_fn(model, run_args)
    convert(model)
    return model

# Map for swapping float module to quantized ones
DEFAULT_MODULE_MAPPING = {
    nn.Linear: nnq.Linear,
    nn.ReLU: nnq.ReLU,
    nn.Conv2d: nnq.Conv2d,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # QAT modules:
    qat.Linear: nnq.Linear,
    qat.Conv2d: nnq.Conv2d,
}

# Map for swapping float module to qat modules
DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: qat.Linear,
    nn.Conv2d: qat.Conv2d,
}

def convert(module, mapping=DEFAULT_MODULE_MAPPING):
    r"""Converts the float module with observers(where we can get quantization
    parameters) to a quantized module.
    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized
           module type, can be overwrritten to allow swapping user defined Modules
    Return:
        A quantized module
    """
    module_swapped = swap_module(module, mapping)

    reassign = {}
    for name, mod in module.named_children():
        new_mod = convert(mod, mapping)
        if new_mod is not mod:
            reassign[name] = new_mod

    for name, mod in reassign.items():
        setattr(module_swapped, name, mod)

    return module_swapped

def swap_module(mod, mapping):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    if hasattr(mod, 'qconfig') and mod.qconfig is not None:
        if type(mod) in mapping:
            new_mod = mapping[type(mod)].from_float(mod)

    return new_mod
