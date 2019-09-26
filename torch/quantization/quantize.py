from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import torch
import torch.nn as nn
import torch.nn._intrinsic as nni
import torch.nn._intrinsic.quantized as nniq
import torch.nn._intrinsic.qat as nniqat
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
from .QConfig import default_dynamic_qconfig, float16_dynamic_qconfig
import torch.nn.qat as nnqat
import warnings

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
    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x):
        return x

DEFAULT_SKIP_LIST = [nn.Dropout, nn.Identity, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, DeQuantStub]

def _propagate_qconfig_helper(module, qconfig_dict, skip_list=DEFAULT_SKIP_LIST, qconfig_parent=None, prefix=''):
    r"""This is a helper function for `propagate_qconfig_`

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
    if type(module) in skip_list:
        module.qconfig = None
    if not hasattr(module, 'qconfig'):
        module.qconfig = qconfig_parent
        if qconfig_dict:
            if prefix in qconfig_dict:
                module.qconfig = qconfig_dict[prefix]
            elif type(module) in qconfig_dict:
                module.qconfig = qconfig_dict[type(module)]

    # Don't quantize empty Sequential, empty Sequential is same as
    # Identity, but we can't put Sequential into skip list because
    # we also have non-empty Sequential and the qconfig needs to
    # be propagated to child in that case
    # TODO: Add test
    if len(module._modules) == 0 and type(module) == nn.Sequential:
        module.qconfig = None

    for name, child in module.named_children():
        module_prefix = prefix + '.' + name if prefix else name
        _propagate_qconfig_helper(child, qconfig_dict, skip_list, module.qconfig, module_prefix)

# TODO(jerryzh): expose skip_list
def propagate_qconfig_(module, qconfig_dict=None):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to quantization
            configuration, qconfig applies to all submodules of a given
            module unless qconfig for the submodules are specified (when the
            submodule already has qconfig attribute)

    Return:
        None, module is modified inplace with qconfig attached
    """
    if qconfig_dict is None:
        qconfig_dict = {}
    _propagate_qconfig_helper(module, qconfig_dict)

def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output
    """
    return self.observer(output)

def add_observer_(module):
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
        if type(child) == nnq.FloatFunctional:
            if hasattr(child, 'qconfig') and child.qconfig is not None:
                child.observer = child.qconfig.activation()
        else:
            add_observer_(child)

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if hasattr(module, 'qconfig') and module.qconfig is not None and \
       len(module._modules) == 0:
        # observer and hook will be gone after we swap the module
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
        self.add_module('dequant', DeQuantStub())
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

def prepare(model, qconfig_dict=None, inplace=False):
    r"""Prepares a copy of the model for quantization calibration or quantization-aware training.

    Quantization configuration can be passed as an `qconfig_dict` or assigned preemptively
    to individual submodules in `.qconfig` attribute.

    The model will be attached with observer or fake quant modules, and qconfig
    will be propagated.

    Args:
        model: input model to be modified in-place
        qconfig_dict: dictionary that maps from name or type of submodule to quantization
            configuration, qconfig applies to all submodules of a given
            module unless qconfig for the submodules are specified (when the
            submodule already has qconfig attribute)
        inplace: carry out model transformations in-place, the original module is mutated
    """
    if not inplace:
        model = copy.deepcopy(model)
    propagate_qconfig_(model)
    # sanity check common API misusage
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in model.modules()):
        warnings.warn("None of the submodule got qconfig applied. Make sure you "
                      "passed correct configuration through `qconfig_dict` or "
                      "by assigning the `.qconfig` attribute directly on submodules")
    add_observer_(model)
    return model

# Map for swapping float module to quantized ones
DEFAULT_MODULE_MAPPING = {
    nn.Linear: nnq.Linear,
    nn.ReLU: nnq.ReLU,
    nn.Conv2d: nnq.Conv2d,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.LinearReLU: nniq.LinearReLU,
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    nniqat.LinearReLU: nniq.LinearReLU,
    nniqat.ConvBn2d: nnq.Conv2d,
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    # QAT modules:
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
}

# Map for swapping float module to qat modules
DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: nnqat.Linear,
    nn.Conv2d: nnqat.Conv2d,
    # Intrinsic modules:
    nni.ConvBn2d: nniqat.ConvBn2d,
    nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.LinearReLU: nniqat.LinearReLU
}

DEFAULT_DYNAMIC_MODULE_MAPPING = {
    nn.Linear: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
}

def quantize(model, run_fn, run_args, mapping=DEFAULT_MODULE_MAPPING, inplace=False):
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
        inplace: carry out model transformations in-place, the original module is mutated
        mapping: correspondence between original module types and quantized counterparts

    Return:
        Quantized model.
    """

    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    prepare(model, inplace=True)
    run_fn(model, run_args)
    convert(model, mapping, inplace=True)
    return model

def quantize_dynamic(model, qconfig_dict=None, dtype=torch.qint8, mapping=DEFAULT_DYNAMIC_MODULE_MAPPING, inplace=False):
    r"""Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization
    by default is performed for layers with large weights size - i.e. Linear and RNN variants.

    Fine grained control is possible with `qconfig_dict` and `mapping` that act similarly to `quantize()`.
    If `qconfig_dict` is provided, the `dtype` argument is ignored.

    Args:
        module: input model
        qconfig_dict: dictionary that maps from name or type of submodule to quantization
            configuration, qconfig applies to all submodules of a given
            module unless qconfig for the submodules are specified (when the
            submodule already has qconfig attribute). Entries in the dictionary
            need to be QConfigDynamic instances.
        inplace: carry out model transformations in-place, the original module is mutated
        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced
    """
    if qconfig_dict is None:
        if dtype == torch.qint8:
            qconfig_dict = {
                nn.Linear : default_dynamic_qconfig,
                nn.LSTM : default_dynamic_qconfig,
            }
        elif dtype == torch.float16:
            qconfig_dict = {
                # TODO: uncomment when float16 Linear support is added
                # nn.Linear : default_dynamic_qconfig,
                nn.LSTM : float16_dynamic_qconfig,
            }
        else:
            raise ValueError(
                "Don't know how to quantize with default settings for {}. Provide full qconfig please".format(dtype))

    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    propagate_qconfig_(model, qconfig_dict)
    convert(model, mapping, inplace=True)
    return model

def prepare_qat(model, mapping=DEFAULT_QAT_MODULE_MAPPING, inplace=False):
    model = prepare(model, inplace=inplace)
    convert(model, mapping, inplace=True)
    return model

def quantize_qat(model, run_fn, run_args, inplace=False):
    r"""Do quantization aware training and output a quantized model

    Args:
        model: input model
        run_fn: a function for evaluating the prepared model, can be a
            function that simply runs the prepared model or a training loop
        run_args: positional arguments for `run_fn`

    Return:
        Quantized model.
    """
    if not inplace:
        model = copy.deepcopy(model)
    model.train()
    prepare_qat(model, inplace=True)
    run_fn(model, run_args)
    convert(model, inplace=True)
    return model

def convert(module, mapping=DEFAULT_MODULE_MAPPING, inplace=False):
    r"""Converts the float module with observers (where we can get quantization
    parameters) to a quantized module.
    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized
           module type, can be overwrritten to allow swapping user defined Modules
        inplace: carry out model transformations in-place, the original module is mutated
    """
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    # TODO(jerryzh): remove after deciding on the impl of intrinsic modules
    # This is required because intrinsic modules right now are implemented as
    # nn.Sequential and we don't want to swap their constituents
    SWAPPABLE_MODULES = (nni.ConvBn2d,
                         nni.ConvBnReLU2d,
                         nni.LinearReLU,
                         nni.ConvReLU2d)

    for name, mod in module.named_children():
        if type(mod) not in SWAPPABLE_MODULES:
            convert(mod, mapping, inplace=True)
        reassign[name] = swap_module(mod, mapping)

    for key, value in reassign.items():
        module._modules[key] = value

    return module

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
    # Always replace dequantstub with dequantize
    if hasattr(mod, 'qconfig') and mod.qconfig is not None or type(mod) == DeQuantStub:
        if type(mod) in mapping:
            new_mod = mapping[type(mod)].from_float(mod)
    return new_mod

def get_observer_dict(mod, target_dict, prefix=""):
    r"""Traverse the modules and save all observers into dict.
    This is mainly used for quantization accuracy debug
    Args:
        mod: the top module we want to save all observers
        prefix: the prefix for the current module
        target_dict: the dictionary used to save all the observers
    """
    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + '.'

    if hasattr(mod, 'observer'):
        target_dict[get_prefix(prefix) + 'observer'] = mod.observer
    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        get_observer_dict(child, target_dict, module_prefix)
