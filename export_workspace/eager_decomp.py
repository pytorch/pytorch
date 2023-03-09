import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch._decomp import core_aten_decompositions

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        return self.bn(x)

x = torch.rand(2, 3, 244, 244)
model = TestModule()

class TorchEagerDecomposeMode(TorchDispatchMode):
    def __init__(self, decomposition_table):
        super().__init__()
        self.decomposition_table = decomposition_table

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in self.decomposition_table:
            print(f"Decomposing {func}")
            with self:
                return self.decomposition_table[func](*args, **kwargs)
        else:
            print(f"Running {func}")
            return func(*args, **kwargs)

enable_eager_decomp = False
decomposition_table = core_aten_decompositions() if enable_eager_decomp else {}

with TorchEagerDecomposeMode(decomposition_table):
    y = model(x)
