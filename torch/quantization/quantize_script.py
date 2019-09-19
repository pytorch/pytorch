from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.jit import ScriptModule

def prepare_script(model, qconfig_dict, inplace=False):
    if not inplace:
        model = model.copy()
    torch._C._jit_pass_insert_observers(model._c,
                                        'forward',
                                        qconfig_dict,
                                        True)
    return model

def convert_script(model, inplace=False):
    if not inplace:
        model = model.copy()
    torch._C._jit_pass_insert_quant_dequant(model._c, 'forward', True)
    return model

def quantize_script(model, qconfig_dict, run_fn, run_args, inplace=False):
    if not model._c._has_method('forward'):
        raise ValueError('input script module does not have forward method')
    if not inplace:
        model = model.copy()
    torch._C._jit_pass_fold_convbn(model._c)
    prepare_script(model, qconfig_dict, True)
    run_fn(model._c._get_method('forward'), *run_args)
    convert_script(model, True)
    return model
