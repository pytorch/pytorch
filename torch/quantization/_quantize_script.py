from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from .qconfig import QConfig
from torch.jit._recursive import wrap_cpp_module

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

def _prepare_script(model, qconfig_dict, is_dynamic):
    _check_is_script_module(model)
    _check_forward_method(model)
    if not all(isinstance(x, str) for x in qconfig_dict.keys()):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    torch._C._jit_pass_dedup_module_uses(model._c)
    model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
    return wrap_cpp_module(torch._C._jit_pass_insert_observers(model._c,
                                                               'forward',
                                                               scripted_qconfig_dict,
                                                               False,
                                                               is_dynamic))

def prepare_script(model, qconfig_dict, inplace=False):
    if not inplace:
        model = model.copy()
    return _prepare_script(model, qconfig_dict, is_dynamic=False)

def prepare_dynamic_script(model, qconfig_dict):
    return _prepare_script(model, qconfig_dict, is_dynamic=True)

def _convert_script(model, is_dynamic, debug=False):
    _check_is_script_module(model)
    model.eval()
    model = wrap_cpp_module(torch._C._jit_pass_insert_quant_dequant(model._c, 'forward', False, is_dynamic))
    if not debug:
        model = wrap_cpp_module(torch._C._jit_pass_quant_finalize(model._c, is_dynamic))
    return model

def convert_script(model, inplace=False, debug=False):
    if not inplace:
        model = model.copy()
    return _convert_script(model, is_dynamic=False, debug=debug)

def convert_dynamic_script(model, debug=False):
    return _convert_script(model, is_dynamic=True, debug=debug)

def _quantize_script(model, qconfig_dict, run_fn=None, run_args=None, is_dynamic=False, debug=False):
    if is_dynamic:
        model = prepare_dynamic_script(model, qconfig_dict)
        model(*run_args)
        model = convert_dynamic_script(model, debug)
    else:
        model = prepare_script(model, qconfig_dict, True)
        run_fn(model._c._get_method('forward'), *run_args)
        model = convert_script(model, True, debug)

    return model

def quantize_script(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    assert not inplace, "We don't support inplace right now"
    if not inplace:
        model = model.copy()
    return _quantize_script(model, qconfig_dict, run_fn, run_args, is_dynamic=False, debug=debug)

def quantize_dynamic_script(model, qconfig_dict, sample_model_inputs, debug=False):
    return _quantize_script(model, qconfig_dict, run_args=sample_model_inputs, is_dynamic=True, debug=debug)
