import torch
import torch.nn as nn
from torch._dynamo.test_case import run_tests, TestCase

def set_attrs_from_orig_model(cls_instance, mod, *func_names):
    cls_instance.__dict__.update(mod.__dict__)
    if func_names is not None:
        for func in func_names:
            setattr(cls_instance, func, getattr(mod, func))

class PatchedMyModule(nn.Module):
    def __init__(self, mod):
        super().__init__()
        set_attrs_from_orig_model(self, mod, "resolve_input")

    def forward(self, x):
        x = self.resolve_input(x)
        return x

class MyModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

    def resolve_input(self, x):
        x = torch.nn.Dropout(0.1)(self.linear(x))
        return x

    def forward(self, x):
        x = self.linear(x)
        return x

class TestPatchedMyModule(TestCase):
    def test_forward(self):
        module = MyModule(input_dim=1, output_dim=1)
        patched_module = PatchedMyModule(module)
        compiled_module = torch.compile(patched_module, fullgraph=True)

        input_tensor = torch.tensor([1.], dtype=torch.float)
        res = compiled_module(input_tensor)

        self.assertIsNotNone(res)
        self.assertEqual(res.shape, torch.Size([1]))

if __name__ == "__main__":
    run_tests()
