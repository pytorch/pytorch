from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List, Optional

import torch
from .qconfig import QConfig
from torch.jit._recursive import wrap_cpp_module

class ConvPackedParams(torch.nn.Module):
    def __init__(self):
        super(ConvPackedParams, self).__init__()
        wq = torch._empty_affine_quantized([1, 1, 1, 1], scale=1.0, zero_point=0, dtype=torch.qint8)
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.dilation = [1, 1]
        self.groups = 1
        self.set_weight_bias(wq, None)

    @torch.jit.export
    def set_conv_params(self, stride, padding, dilation, groups):
        # type: (List[int], List[int], List[int], int) -> None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    @torch.jit.export
    def set_weight_bias(self, weight, bias):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params = torch.ops.quantized.conv2d_prepack(weight, bias, self.stride,
                                                                 self.padding, self.dilation, self.groups)

    @torch.jit.export
    def _weight_bias(self):
        return torch.ops.quantized.conv2d_unpack(self._packed_params)

    def forward(self, x):
        return x

    @torch.jit.export
    def __getstate__(self):
        qweight, bias = self._weight_bias()
        return (qweight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.training)

    @torch.jit.export
    def __setstate__(self, state):
        self.stride = state[2]
        self.padding = state[3]
        self.dilation = state[4]
        self.groups = state[5]
        self.set_weight_bias(state[0],
                             state[1])
        self.training = state[6]

linear_packed_params = None
conv_packed_params = None
if 'fbgemm' in torch.backends.quantized.supported_engines:
    linear_packed_params = torch.jit.script(torch.nn.quantized.modules.linear.LinearPackedParams())._c
    conv_packed_params = torch.jit.script(ConvPackedParams())._c

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

def get_scripted_qconfig_dict(qconfig_dict):
    return {k: script_qconfig(v) if v else None for k, v in qconfig_dict.items()}

def _prepare_script(model, qconfig_dict, is_dynamic):
    _check_is_script_module(model)
    if any(map(lambda x : not isinstance(x, str), qconfig_dict.keys())):
        raise ValueError('qconfig_dict should contain names(str) as keys.')
    scripted_qconfig_dict = get_scripted_qconfig_dict(qconfig_dict)
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
    _check_is_script_module(model)
    _check_forward_method(model)
    torch._C._jit_pass_dedup_module_uses(model._c)
    model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
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
