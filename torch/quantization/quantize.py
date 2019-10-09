from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import itertools
import warnings

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq

from .default_mappings import (DEFAULT_DYNAMIC_MODULE_MAPPING,
                               DEFAULT_MODULE_MAPPING,
                               DEFAULT_QAT_MODULE_MAPPING,
                               DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST)
from .stubs import DeQuantStub, QuantWrapper
from .QConfig import default_dynamic_qconfig, float16_dynamic_qconfig

def _propagate_qconfig_helper(module, qconfig_dict, white_list=None,
                              qconfig_parent=None, prefix=''):
    r"""This is a helper function for `propagate_qconfig_`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        white_list: list of quantizable modules
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict

    Return:
        None, module is modified inplace with qconfig attached
    """
    # TODO: Add test
    if white_list is None:
        white_list = DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST

    module_qconfig = qconfig_dict.get(type(module), qconfig_parent)
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)
    module_qconfig = getattr(module, 'qconfig', module_qconfig)

    if type(module) in white_list:
        module.qconfig = module_qconfig
    for name, child in module.named_children():
        module_prefix = prefix + '.' + name if prefix else name
        _propagate_qconfig_helper(child, qconfig_dict, white_list,
                                  module_qconfig, module_prefix)

# TODO(jerryzh): expose white_list
def propagate_qconfig_(module, qconfig_dict=None):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)

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
        module: input module with qconfig attributes for all the leaf modules that we want to quantize

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
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

def quantize(model, run_fn, run_args, mapping=None, inplace=False):
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
    if mapping is None:
        mapping = DEFAULT_MODULE_MAPPING
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    prepare(model, inplace=True)
    run_fn(model, run_args)
    convert(model, mapping, inplace=True)
    return model

def quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8,
                     mapping=None, inplace=False):
    r"""Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization
    by default is performed for layers with large weights size - i.e. Linear and RNN variants.

    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        module: input model
        qconfig_spec: Either:
            * A dictionary that maps from name or type of submodule to quantization
              configuration, qconfig applies to all submodules of a given
              module unless qconfig for the submodules are specified (when the
              submodule already has qconfig attribute). Entries in the dictionary
              need to be QConfigDynamic instances.
            * A set of types and/or submodule names to apply dynamic quantization to,
              in which case the `dtype` argument is used to specifiy the bit-width
        inplace: carry out model transformations in-place, the original module is mutated
        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced
    """
    if qconfig_spec is None:
        if dtype == torch.qint8:
            qconfig_spec = {
                nn.Linear : default_dynamic_qconfig,
                nn.LSTM : default_dynamic_qconfig,
            }
        elif dtype == torch.float16:
            qconfig_spec = {
                # TODO: uncomment when float16 Linear support is added
                # nn.Linear : default_dynamic_qconfig,
                nn.LSTM : float16_dynamic_qconfig,
            }
        else:
            raise ValueError(
                "Don't know how to quantize with default settings for {}. Provide full qconfig please".format(dtype))
    elif isinstance(qconfig_spec, set):
        if dtype is torch.qint8:
            default_qconfig = default_dynamic_qconfig
        elif dtype is torch.float16:
            default_qconfig = float16_dynamic_qconfig
        else:
            raise RuntimeError('Unknown dtype specified for quantize_dynamic: ', str(dtype))
        qconfig_spec = dict(zip(qconfig_spec, itertools.repeat(default_qconfig)))

    if mapping is None:
        mapping = DEFAULT_DYNAMIC_MODULE_MAPPING

    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    propagate_qconfig_(model, qconfig_spec)
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
                function that simply runs the prepared model or a training 
                loop
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

def convert(module, mapping=None, inplace=False):
    r"""Converts the float module with observers (where we can get quantization
    parameters) to a quantized module.

    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized module type, can 
                 be overwrritten to allow swapping user defined Modules
        inplace: carry out model transformations in-place, the original module is mutated
    """
    if mapping is None:
        mapping = DEFAULT_MODULE_MAPPING
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
