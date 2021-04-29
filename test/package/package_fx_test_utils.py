import torch

from torch.fx import Tracer


class MyModule(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class SimpleTest(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 3.0)


class PackageFXTestTracer(Tracer):
    def __init__(self):
        super().__init__()
        self.is_package_fx_test_tracer = True

    def is_leaf_module(self, m, module_qualified_name,) -> bool:
        return super().is_leaf_module(m, module_qualified_name) or module_qualified_name == "MyModule"
