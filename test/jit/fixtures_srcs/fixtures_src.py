import torch

class TestVersionedDivTensorExampleV4(torch.nn.Module):
    def __init__(self):
        super(TestVersionedDivTensorExampleV4, self).__init__()

    def forward(self, a, b):
        result_0 = a / b
        result_1 = torch.div(a, b)
        result_2 = a.div(b)
        return result_0, result_1, result_2
