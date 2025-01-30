import torch
from torch._dynamo.compiled_autograd import disable_start, disable_end

torch._dynamo.config.compiled_autograd = True


def _fn3(x):
    return x + 10

def fn3(x):
    x = disable_start(x)
    ####################
    # CA disabled region
    out = _fn3(x)
    ####################
    out = disable_end(out)
    return out

def fn(model, x):
    y = model(x)
    return fn3(y)

# @torch.compile(backend="aot_eager")
def fn2(model, x):
    return fn(model, x).sum()

model = torch.nn.Linear(10, 10)
x = torch.randn(10, 10)

loss = fn2(model, x)
with torch._dynamo.compiled_autograd._enable(torch.compile(backend="eager")):
    loss.backward()

""" Example CA graph:
 # File: /data/users/xmfan/core/b/pytorch/torch/_dynamo/compiled_autograd.py:1148 in set_node_origin, code: CaDisableRegionBeginBackward (NodeCall 2)
getitem_7 = hooks[0]
call_backward = torch__dynamo_external_utils_call_backward(getitem_7, (), getitem_6);  getitem_7 = getitem_6 = None
getitem_9 = call_backward[0];  call_backward = None
validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_9], [((None, None, device(type='cpu'), 6, 0, None), [10, 10], False)]);  getitem_9 = None
getitem_10 = validate_outputs_2[0];  validate_outputs_2 = None

# File: /data/users/xmfan/core/b/pytorch/torch/_dynamo/compiled_autograd.py:1148 in set_node_origin, code: AddBackward0 (NodeCall 3)
add_backward0 = torch__dynamo_compiled_autograd_ops_AddBackward0([getitem_10], [True, False], 1, 4, 6);  getitem_10 = None
getitem_11 = add_backward0[0]
getitem_12 = add_backward0[1];  add_backward0 = None
validate_outputs_3 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_11, getitem_12], [((None, None, device(type='cpu'), 6, 0, None), [10, 10], False), None]);  getitem_11 = getitem_12 = None
getitem_13 = validate_outputs_3[0];  validate_outputs_3 = None

# File: /data/users/xmfan/core/b/pytorch/torch/_dynamo/compiled_autograd.py:1148 in set_node_origin, code: CaDisableRegionEndBackward (NodeCall 4)
getitem_15 = hooks[1];  hooks = None
call_backward_1 = torch__dynamo_external_utils_call_backward(getitem_15, (), getitem_13);  getitem_15 = getitem_13 = None
getitem_17 = call_backward_1[0];  call_backward_1 = None
validate_outputs_4 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_17], [((None, None, device(type='cpu'), 6, 0, None), [10, 10], False)]);  getitem_17 = None
getitem_18 = validate_outputs_4[0];  validate_outputs_4 = None
"""
