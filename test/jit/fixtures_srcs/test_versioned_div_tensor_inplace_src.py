import torch

class TestVersionedDivTensorInplaceModule(torch.nn.Module):
    def __init__(self):
        super(TestVersionedDivTensorInplaceModule, self).__init__()

    def forward(self, a, b):
        result_0 = a / b
        result_1 = torch.div(a, b)
        result_2 = a.div(b)

        return result_0, result_1, result_2

script_module = torch.jit.script(TestVersionedDivTensorInplaceModule())

a = torch.tensor(4)
b = torch.tensor(2)
c = script_module.forward(a, b)
