# Owner(s): ["oncall: jit"]

import io
import math
import unittest

import torch
from torch.nn import init
from torch.testing._internal.common_utils import skipIfLegacyJitExecutor
from torch.testing._internal.jit_utils import JitTestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestGenerator(JitTestCase):
    # torch.jit.trace does not properly capture the generator manual seed
    # and thus is non deterministic even if the generator is manually seeded
    @skipIfLegacyJitExecutor("legacy JIT executor does not support Generator type")
    @unittest.expectedFailure
    def test_trace(self):
        def f():
            generator = torch.Generator()
            generator.seed()
            generator.manual_seed(2023)
            generator.initial_seed()
            tensor = torch.empty(2, 2)
            tensor.uniform_(0, 1, generator=generator)
            return tensor

        traced_f = torch.jit.trace(f, ())

        # Run this 3 times to ensure that the generator is being manually seeded
        # each time the traced function is run
        for i in range(3):
            torch.manual_seed(1)

            eager_tensor = f()

            # Change the seed of the default generator to
            # check that we're using the generator from the
            # trace
            torch.manual_seed(2)
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

        script_f = torch.jit.script(f, ())

        # Run this 3 times to ensure that the generator is being manually seeded
        # each time the traced function is run
        for i in range(3):
            torch.manual_seed(1)

            eager_tensor = f()

            # Change the seed of the default generator to
            # check that we're using the generator from the
            # trace
            torch.manual_seed(2)

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

    def test_generator_arg(self):
        def f(generator: torch.Generator):
            tensor = torch.empty(2, 2)
            tensor.normal_(-1.0, 1.0, generator=generator)
            return tensor

        generator = torch.Generator()
        generator.manual_seed(2023)

        script_f = torch.jit.script(f, (generator,))

        for i in range(3):
            generator = torch.Generator()
            generator.manual_seed(2023 + i)

            torch.manual_seed(1 + i)

            eager_tensor = f(generator)

            generator = torch.Generator()
            generator.manual_seed(2023 + i)

            torch.manual_seed(1 + i)

            script_tensor = script_f(generator)

            self.assertEqual(eager_tensor, script_tensor)

    def test_save_load(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2, bias=False)
                self.bar = torch.nn.Linear(2, 2, bias=False)

                self.reset_parameters()

            def reset_linear(self, module, generator):
                init.kaiming_uniform_(
                    module.weight, a=math.sqrt(5), generator=generator
                )

            def reset_parameters(self):
                generator = torch.Generator()
                generator.manual_seed(1)
                self.reset_linear(self.foo, generator)

                generator = torch.Generator()
                generator.manual_seed(2)
                self.reset_linear(self.bar, generator)

            def forward(self, x):
                x = self.foo(x)
                x = self.bar(x)

                generator = torch.Generator()
                generator.manual_seed(3)
                r = torch.empty_like(x)
                r.normal_(0.0, 1.0, generator=generator)

                return x, r

        eager_foo = Foo()

        script_module = torch.jit.script(Foo())
        saved_module = io.BytesIO()
        torch.jit.save(script_module, saved_module)
        saved_module.seek(0)

        loaded_module = torch.jit.load(saved_module)

        self.assertEqual(eager_foo.foo.weight, loaded_module.foo.weight)
        self.assertEqual(eager_foo.bar.weight, loaded_module.bar.weight)

        try:
            # Run this 3 times so make sure that the generator seed is being set
            # every time forward is called
            for i in range(3):
                x = torch.ones(2, 2)
                out1, r1 = eager_foo(x)
                out2, r2 = loaded_module(x)

                try:
                    self.assertEqual(out1, out2)
                except:  # noqa: B001, E722
                    print(f"Iteration {i}:\n{out1=}\n{out2=}")
                    raise

                try:
                    self.assertEqual(r1, r2)
                except:  # noqa: B001, E722
                    print(f"Iteration {i}:\n{r1=}\n{r2=}")
                    raise
        except:  # noqa: B001, E722
            print(loaded_module.forward.code)
            raise
