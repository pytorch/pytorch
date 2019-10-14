from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from .QConfig import QConfig

class PackedParams(torch.nn.Module):
    def __init__(self):
        super(PackedParams, self).__init__()
        w = torch.rand((5, 5), dtype=torch.float)
        wq = torch.quantize_per_tensor(w, 2.0, 0, torch.qint8)
        self.set_weight_bias(wq, torch.rand(5))

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
        return self._weight_bias(), self.training

    @torch.jit.export
    def __setstate__(self, state):
        self.set_weight_bias(state[0][0], state[0][1])
        self.training = state[1]

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
        _packed_params_scripted = torch.jit.script(PackedParams())._c
        torch._C._jit_pass_insert_prepack_unpack(model._c)
        torch._C._jit_pass_fold_prepack(model._c, _packed_params_scripted)
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
    torch._C._jit_pass_fold_convbn(model._c)
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
