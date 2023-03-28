import unittest
import torch
import torch.onnx

class TestDep(unittest.TestCase):
    def test_linear(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, x):
                return self.linear(x)

        export_output = torch.onnx.flash_export(LinearModel(), torch.randn(1, 3))
