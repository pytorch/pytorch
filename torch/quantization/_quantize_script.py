from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from .qconfig import QConfig

linear_packed_params = None
conv_packed_params = None
if 'fbgemm' in torch.backends.quantized.supported_engines:
    linear_packed_params = torch.jit.script(torch.nn.quantized.modules.linear.LinearPackedParams())._c
    # TODO: conv3d
    conv_packed_params = torch.jit.script(torch.nn.quantized.modules.conv.ConvPackedParams(spatial_dim=2))._c

def _check_is_script_module(model):
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError('input must be a script module, got: ' + str(type(model)))

def prepare_script(model, qconfig_dict, inplace=False):
    _check_is_script_module(model)
    if not inplace:
        model = model.copy()
    torch._C._jit_pass_insert_observers(model._c,
                                        'forward',
                                        qconfig_dict,
                                        True)
    return model

def convert_script(model, inplace=False):
    _check_is_script_module(model)
    if not inplace:
        model = model.copy()
    torch._C._jit_pass_insert_quant_dequant(model._c, 'forward', True)
    if 'fbgemm' in torch.backends.quantized.supported_engines:
        torch._C._jit_pass_insert_prepack_unpack(model._c)
        if linear_packed_params and conv_packed_params:
            torch._C._jit_pass_fold_prepack(model._c,
                                            linear_packed_params,
                                            conv_packed_params)

    return model

# TODO: non-scriptable QConfig will be supported later
def script_qconfig(qconfig):
    return QConfig(
        activation=torch.jit.script(qconfig.activation())._c,
        weight=torch.jit.script(qconfig.weight())._c)

def quantize_script(model, qconfig_dict, run_fn, run_args, inplace=False):
    _check_is_script_module(model)
    if not model._c._has_method('forward'):
        raise ValueError('input script module does not have forward method')
    assert not inplace, "We don't support inplace right now"
    if not inplace:
        model = model.copy()
    scripted_qconfig_dict = {k: script_qconfig(v) for k, v in qconfig_dict.items()}
    # We are not going to run fold_convbn pass right now
    # since it is not able to work correctly, we will
    # revisit after constants is properly handled in
    # JIT
    # torch._C._jit_pass_fold_convbn(model._c)
    prepare_script(model, scripted_qconfig_dict, True)
    run_fn(model._c._get_method('forward'), *run_args)
    # When we mutating graph we didn't create a new ClassType
    # and the graph executor will run an out dated version
    # of the graph if we do inplace graph mutation, therefore
    # we copy the model here
    # [TODO] This will be fixed later when we figure out
    # how to properly mutate types
    model = convert_script(model, False)
    return model
