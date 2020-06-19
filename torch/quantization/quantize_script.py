from __future__ import absolute_import, division, print_function, unicode_literals

import enum
import torch
from .qconfig import QConfig
from torch.jit._recursive import wrap_cpp_module

# Quantization type (dynamic quantization, static quantization).
# Should match the c++ enum in quantization_type.h
class QuantType(enum.IntEnum):
    DYNAMIC = 0
    STATIC = 1

def _check_is_script_module(model):
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError('input must be a script module, got: ' + str(type(model)))

def _check_forward_method(model):
    if not model._c._has_method('forward'):
        raise ValueError('input script module does not have forward method')

def script_qconfig(qconfig):
    return QConfig(
        activation=torch.jit.script(qconfig.activation())._c,
        weight=torch.jit.script(qconfig.weight())._c)

def script_qconfig_dict(qconfig_dict):
    return {k: script_qconfig(v) if v else None for k, v in qconfig_dict.items()}

def _prepare_script(model, qconfig_dict, inplace=False, quant_type=QuantType.STATIC):
    assert not inplace, "The inplace support is still in development"
    _check_is_script_module(model)
    _check_forward_method(model)
    if not all(isinstance(x, str) for x in qconfig_dict.keys()):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
    return wrap_cpp_module(torch._C._jit_pass_insert_observers(model._c,
                                                               'forward',
                                                               scripted_qconfig_dict,
                                                               inplace,
                                                               quant_type))

def prepare_script(model, qconfig_dict, inplace=False):
    return _prepare_script(model, qconfig_dict, inplace, quant_type=QuantType.STATIC)

def prepare_dynamic_script(model, qconfig_dict, inplace=False):
    return _prepare_script(model, qconfig_dict, inplace, quant_type=QuantType.DYNAMIC)

def _convert_script(model, inplace=False, debug=False, quant_type=QuantType.STATIC):
    assert not inplace, "The inplace support is still in development"
    _check_is_script_module(model)
    model.eval()
    model = wrap_cpp_module(torch._C._jit_pass_insert_quant_dequant(model._c, 'forward', inplace, debug, quant_type))
    if not debug:
        # Moving model parameters to CPU since quantized operators
        # are only supported on CPU right now
        model.cpu()
        model = wrap_cpp_module(torch._C._jit_pass_quant_finalize(model._c, quant_type))
    return model

def convert_script(model, inplace=False, debug=False):
    return _convert_script(model, inplace, debug, quant_type=QuantType.STATIC)

def convert_dynamic_script(model, inplace=False, debug=False):
    return _convert_script(model, inplace, debug, quant_type=QuantType.DYNAMIC)

def _quantize_script(model, qconfig_dict, run_fn=None, run_args=None, inplace=False, debug=False, quant_type=QuantType.STATIC):
    assert not inplace, "We don't support inplace right now"
    # Always do inplace convert because the Tensor is already
    # copied in prepare_script when inplace is False
    if quant_type == QuantType.DYNAMIC:
        model = prepare_dynamic_script(model, qconfig_dict, inplace)
        # TODO: change inplace to True
        model = convert_dynamic_script(model, False, debug)
    else:
        assert run_fn, "Must provide calibration function for post training static quantization"
        assert run_args, "Must provide calibration dataset for post training static quantization"
        model = prepare_script(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        # TODO: change inplace to True
        model = convert_script(model, False, debug)

    return model

def quantize_script(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    return _quantize_script(model, qconfig_dict, run_fn, run_args, inplace, debug, quant_type=QuantType.STATIC)

def quantize_dynamic_script(model, qconfig_dict, inplace=False, debug=False):
    return _quantize_script(model, qconfig_dict, inplace=inplace, debug=debug, quant_type=QuantType.DYNAMIC)
