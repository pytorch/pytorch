# Owner(s): ["oncall: jit"]

import io
import math

import torch
from torch.nn import init
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import skipIfLegacyJitExecutor


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestGenerator(JitTestCase):
    @skipIfLegacyJitExecutor("legacy JIT executor does not support Generator type")
    def test_trace(self):
        def f():
            generator = torch.Generator()
            generator.seed()
            generator.manual_seed(2023)
            generator.initial_seed()
            tensor = torch.empty(2, 2)
            tensor.uniform_(0, 1, generator=generator)
            return tensor

        torch.manual_seed(1)

        eager_tensor = f()

        # Change the seed of the default generator to
        # check that we're using the generator from the
        # trace
        torch.manual_seed(2)

        traced_f = torch.jit.trace(f, ())
        traced_tensor = traced_f()

        self.assertEqual(eager_tensor, traced_tensor)

    def test_script(self):
        def f():
            generator = torch.Generator()
            generator.seed()
            generator.manual_seed(2023)
            generator.initial_seed()
            tensor = torch.empty(2, 2)
            tensor.normal_(-1.0, 1.0, generator=generator)
            return tensor

        torch.manual_seed(1)

        eager_tensor = f()

        # Change the seed of the default generator to
        # check that we're using the generator from the
        # trace
        torch.manual_seed(2)

        script_f = torch.jit.script(f, ())
        script_tensor = script_f()

        self.assertEqual(eager_tensor, script_tensor)

    def test_default_generator(self):
        def f():
            # check that calling manual seed for the default generator works
            torch.manual_seed(2023)
            tensor = torch.empty(2, 2)
            tensor.normal_(-1.0, 1.0)
            return tensor

        torch.manual_seed(1)

        eager_tensor = f()

        torch.manual_seed(2)

        script_f = torch.jit.script(f, ())
        script_tensor = script_f()

        self.assertEqual(eager_tensor, script_tensor)

    def test_save_load(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)

                self.reset_parameters()

            def reset_parameters(self):
                def reset_linear(module, generator):
                    init.kaiming_uniform_(
                        module.weight, a=math.sqrt(5), generator=generator
                    )
                    if module.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(module.bias, -bound, bound, generator=generator)

                generator = torch.Generator()
                generator.manual_seed(1)
                reset_linear(self.foo, generator)

                generator = torch.Generator()
                generator.manual_seed(2)
                reset_linear(self.bar, generator)

            def forward(self, x):
                x = self.foo(x)
                x = self.bar(x)
                return x

        eager_foo = Foo()

        script_module = torch.jit.script(Foo())
        saved_module = io.BytesIO()
        torch.jit.save(script_module, saved_module)
        saved_module.seek(0)

        loaded_module = torch.jit.load(saved_module)

        self.assertEqual(eager_foo.foo.weight, loaded_module.foo.weight)
        self.assertEqual(eager_foo.foo.bias, loaded_module.foo.bias)
        self.assertEqual(eager_foo.bar.weight, loaded_module.bar.weight)
        self.assertEqual(eager_foo.bar.bias, loaded_module.bar.bias)
