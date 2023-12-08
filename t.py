import torch

class Test:
        def __init__(self):
                self.L__self____tensor_constant0 = torch.arange(0.0, 10, requires_grad=True)

        def forward(self, L_inputs_0_ : torch.Tensor, L_inputs_1_ : torch.Tensor):
                getitem = L_inputs_0_
                getitem_1 = L_inputs_1_

                # File: <eval_with_key>.0:7, code: expand = torch.ops.aten.expand.default(getitem, [10]);  getitem = None
                expand = torch.ops.aten.expand.default(getitem, [10]);  getitem = None

                # File: <eval_with_key>.0:9, code: _tensor_constant0 = self._tensor_constant0
                x = self.L__self____tensor_constant0

                # File: /data/users/xmfan/core/pytorch/test/inductor/test_compiled_autograd.py:79, code: return gO * torch.cos(x)
                cos = torch.cos(x);  x = None
                call_backward = expand * cos;  expand = cos = None

                breakpoint()

                # File: <eval_with_key>.0:11, code: getitem_3 = call_backward[0];  call_backward = None
                getitem_3 = call_backward;  call_backward = None

                # File: <eval_with_key>.0:12, code: accumulate_grad_ = torch.ops.inductor.accumulate_grad_.default(getitem_1, getitem_3);  getitem_1 = getitem_3 = None
                accumulate_grad__default = torch.ops.inductor.accumulate_grad_.default(getitem_1, getitem_3);

                getitem_1 = getitem_3 = None
                return ()


a = torch.tensor((1.))
b = torch.arange(0.0, 10, requires_grad=True)
mytest=Test()
mytest.forward(a, b)
breakpoint()
