# Owner(s): ["module: fx"]

import torch
import itertools
import inspect
import functools
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    TestCase, parametrize, instantiate_parametrized_tests, run_tests)

class NestedModuleL2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(3,))
        self.register_buffer("running_var", torch.zeros(3,))
        self.weight = torch.nn.Parameter(torch.zeros(3), requires_grad=True)
        self.li = torch.nn.Linear(3, 3)

    def forward(self, input):
        a = self.li(self.weight)
        return torch.batch_norm(input, a, None, self.running_mean,
                                self.running_var, False, 0.1, 1e-5, False)[0].cos()

class NestedModuleL1(torch.nn.Module):
    def __init__(self, my_mod):
        super().__init__()
        self.mod = my_mod
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, input):
        return self.mod(input) + self.weight + 1


mod = NestedModuleL1(NestedModuleL2())
mod1 = NestedModuleL2()
@functools.wraps(mod)
def mod_wrapped_in_func(*args, **kwargs):
    return mod(*args, **kwargs)

callables = [mod, mod1, mod_wrapped_in_func]
tracing_modes = ["real", "fake", "symbolic"]
inputs = [torch.randn(2, 3, 1, 2), ]


def name_fn(callable, tracing_mode, input):
    """Names parameterized test cases"""
    return f'{type(callable).__name__}_{tracing_mode}_mode'

@instantiate_parametrized_tests
class TestMakefxPreserveModAttrNames(TestCase):

    @parametrize("callable,tracing_mode,input", itertools.product(callables, tracing_modes, inputs), name_fn)
    def test_makefx_preserve_mod_attr_names(self, callable, tracing_mode, input):
        def get_attr(m, fully_qualified_access_path):
            obj = m
            for key in fully_qualified_access_path.split("."):
                obj = getattr(obj, key)
            return obj

        # Get the nn.Module wrapped in function
        wrapped_m = inspect.unwrap(callable)
        self.assertTrue(isinstance(wrapped_m, torch.nn.Module))

        wrapped_m_params_and_buffers = {**dict(wrapped_m.named_parameters()), **dict(wrapped_m.named_buffers())}
        gm = make_fx(callable, None, tracing_mode, _allow_non_fake_inputs=True)(input)

        # Check attrs' access path in wrapped_m is valid for make_fx's output graph module
        for access_path, value in wrapped_m_params_and_buffers.items():
            self.assertTrue(value is get_attr(gm, access_path))

if __name__ == '__main__':
    run_tests()
