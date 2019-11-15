from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from .qconfig import QConfig

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
        self._packed_params = torch.ops.quantized.conv2d_prepack(weight, bias, self.stride, self.padding, self.dilation, self.groups)

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

class LinearPackedParams(torch.nn.Module):
    def __init__(self):
        super(LinearPackedParams, self).__init__()
        wq = torch._empty_affine_quantized([1, 1], scale=1.0, zero_point=0, dtype=torch.qint8)
        self.set_weight_bias(wq, None)

    @torch.jit.export
    def set_weight_bias(self, weight, bias):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params = torch.ops.quantized.linear_prepack(weight, bias)

    @torch.jit.export
    def _weight_bias(self):
        return torch.ops.quantized.linear_unpack(self._packed_params)

    def forward(self, x):
        return x

    @torch.jit.export
    def __getstate__(self):
        qweight, bias = self._weight_bias()
        return qweight, bias, self.training

    @torch.jit.export
    def __setstate__(self, state):
        # type: (Tuple[Tensor, Optional[Tensor], bool]) -> None
        self.set_weight_bias(state[0], state[1])
        self.training = state[2]


linear_packed_params = None
conv_packed_params = None
if 'fbgemm' in torch.backends.quantized.supported_engines:
    linear_packed_params = torch.jit.script(LinearPackedParams())._c
    conv_packed_params = torch.jit.script(ConvPackedParams())._c

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
