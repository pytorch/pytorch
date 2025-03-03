
import torch
from torch._inductor.ir import NoneAsConstantBuffer
import torch.nn as nn
import torch.nn.functional as F
import depyf
depyf.install()


def fn(loss):
    gm = None
    args = None

    def noop(_gm):
        nonlocal gm
        gm = _gm
        def _noop(*_args, **_kwargs):
            assert not _kwargs
            nonlocal args
            args = _args
            return []
        return _noop

    with torch._dynamo.compiled_autograd._enable(noop):
        loss.backward()

    return gm, args


result = torch._dynamo.compiled_autograd.Op("FunctionalCompiledAutograd", fn, is_custom_function=False)
setattr(torch._dynamo.compiled_autograd.ops, "FunctionalCompiledAutograd", torch._dynamo.allow_in_graph(result))


x = torch.randn(64, 3)
t = torch.randn(64, 1)

model = nn.Linear(3, 1)

torch._dynamo.config.compiled_autograd = True
torch._dynamo.config.do_not_emit_runtime_asserts = True

@torch.compile(backend="eager")
def train(model, x, t):
    y = model(x)
    loss = F.mse_loss(y, t)
    gm, args = torch._dynamo.compiled_autograd.ops.FunctionalCompiledAutograd(loss)
    gm(*args)
    return ()

# with torch._dynamo.compiled_autograd._enable(noop):
train(model, x, t)

for p in model.parameters():
    assert p.grad is not None


"""
# this kinda works, but not ideal
 ===== __compiled_fn_1 =====
 /home/xmfan/core/a/pytorch/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_model_parameters_weight_: "f32[1, 3][3, 1]cpu", L_model_parameters_bias_: "f32[1][1]cpu", L_x_: "f32[64, 3][3, 1]cpu", L_t_: "f32[64, 1][1, 1]cpu"):
        l_model_parameters_weight_ = L_model_parameters_weight_
        l_model_parameters_bias_ = L_model_parameters_bias_
        l_x_ = L_x_
        l_t_ = L_t_
        
         # File: /home/xmfan/core/a/pytorch/hack.py:44 in train, code: y = model(x)
        y: "f32[64, 1][1, 1]cpu" = torch._C._nn.linear(l_x_, l_model_parameters_weight_, l_model_parameters_bias_);  l_x_ = l_model_parameters_weight_ = l_model_parameters_bias_ = None
        
         # File: /home/xmfan/core/a/pytorch/hack.py:45 in train, code: loss = F.mse_loss(y, t)
        loss: "f32[][]cpu" = torch.nn.functional.mse_loss(y, l_t_);  y = l_t_ = None
        
         # File: /home/xmfan/core/a/pytorch/hack.py:46 in train, code: gm, args = torch._dynamo.compiled_autograd.ops.FunctionalCompiledAutograd(loss)
        functional_compiled_autograd = torch__dynamo_compiled_autograd_ops_FunctionalCompiledAutograd(loss);  loss = None
        getitem = functional_compiled_autograd[1];  functional_compiled_autograd = None
        getitem_1 = getitem[0];  getitem = None
        getitem_8: "f32[][]cpu" = getitem_1[0]
        getitem_9: "f32[64, 1][1, 1]cpu" = getitem_1[1]
        getitem_10: "f32[64, 1][1, 1]cpu" = getitem_1[2]
        getitem_11: "f32[64, 3][3, 1]cpu" = getitem_1[3];  getitem_1 = None
        
         # File: <eval_with_key>.0:11 in forward, code: validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], True)]);  getitem = None
        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_8], [((None, None, device(type='cpu'), 6, 0, None), [], True)]);  getitem_8 = None
        getitem_15: "f32[][]cpu" = validate_outputs[0];  validate_outputs = None
        
         # File: <eval_with_key>.0:13 in forward, code: mse_loss_backward0 = torch__dynamo_compiled_autograd_ops_MseLossBackward0([getitem_6], [True, False], 1, getitem_1, getitem_2);  getitem_6 = getitem_1 = getitem_2 = None
        mse_loss_backward0 = torch__dynamo_compiled_autograd_ops_MseLossBackward0([getitem_15], [True, False], 1, getitem_9, getitem_10);  getitem_15 = getitem_9 = getitem_10 = None
        getitem_17: "f32[64, 1][1, 1]cpu" = mse_loss_backward0[0];  mse_loss_backward0 = None
        
         # File: <eval_with_key>.0:16 in forward, code: validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_7, getitem_8], [((None, None, device(type='cpu'), 6, 0, None), [64, 1], True), None]);  getitem_7 = getitem_8 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_17, None], [((None, None, device(type='cpu'), 6, 0, None), [64, 1], True), None]);  getitem_17 = None
        getitem_19: "f32[64, 1][1, 1]cpu" = validate_outputs_1[0];  validate_outputs_1 = None
        
         # File: <eval_with_key>.0:18 in forward, code: addmm_backward0 = torch__dynamo_compiled_autograd_ops_AddmmBackward0([getitem_9], [True, False, True], 1, 1, getitem_3, 0, [64, 3], [], None, 0, [3, 1], [1, 3]);  getitem_9 = getitem_3 = None
        addmm_backward0 = torch__dynamo_compiled_autograd_ops_AddmmBackward0([getitem_19], [True, False, True], 1, 1, getitem_11, 0, [64, 3], [], None, 0, [3, 1], [1, 3]);  getitem_19 = getitem_11 = None
        getitem_22: "f32[64, 1][1, 1]cpu" = addmm_backward0[0]
        getitem_23: "f32[3, 1][1, 3]cpu" = addmm_backward0[2];  addmm_backward0 = None
        
         # File: <eval_with_key>.0:22 in forward, code: validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_11, getitem_12, getitem_13], [((None, None, device(type='cpu'), 6, 0, None), [1], True), None, ((None, None, device(type='cpu'), 6, 0, None), [3, 1], True)]);  getitem_11 = getitem_12 = getitem_13 = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_22, None, getitem_23], [((None, None, device(type='cpu'), 6, 0, None), [1], True), None, ((None, None, device(type='cpu'), 6, 0, None), [3, 1], True)]);  getitem_22 = getitem_23 = None
        getitem_26: "f32[1][1]cpu" = validate_outputs_2[0]
        getitem_27: "f32[3, 1][1, 3]cpu" = validate_outputs_2[2];  validate_outputs_2 = None
        
         # File: /home/xmfan/core/a/pytorch/torch/_dynamo/polyfills/__init__.py:80 in accumulate_grad, code: new_grad = torch.clone(new_grad)
        new_grad: "f32[1][1]cpu" = torch.clone(getitem_26);  getitem_26 = new_grad = None
        
         # File: <eval_with_key>.0:26 in forward, code: tbackward0 = torch__dynamo_compiled_autograd_ops_TBackward0([getitem_16], [True]);  getitem_16 = None
        tbackward0 = torch__dynamo_compiled_autograd_ops_TBackward0([getitem_27], [True]);  getitem_27 = None
        getitem_29: "f32[1, 3][3, 1]cpu" = tbackward0[0];  tbackward0 = None
        
         # File: <eval_with_key>.0:28 in forward, code: validate_outputs_3 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_17], [((None, None, device(type='cpu'), 6, 0, None), [1, 3], True)]);  getitem_17 = None
        validate_outputs_3 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_29], [((None, None, device(type='cpu'), 6, 0, None), [1, 3], True)]);  getitem_29 = None
        getitem_31: "f32[1, 3][3, 1]cpu" = validate_outputs_3[0];  validate_outputs_3 = None
        
         # File: /home/xmfan/core/a/pytorch/torch/_dynamo/polyfills/__init__.py:80 in accumulate_grad, code: new_grad = torch.clone(new_grad)
        new_grad_1: "f32[1, 3][3, 1]cpu" = torch.clone(getitem_31);  getitem_31 = new_grad_1 = None
        return ()
"""
