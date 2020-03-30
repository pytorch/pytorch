import os
import io
import sys

import torch
from torch import Tensor
from typing import NamedTuple

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, clear_class_registry

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestSaveLoad(JitTestCase):
    def test_different_modules(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """
        class Foo(torch.nn.Module):
            def __init__(self):
                super(Foo, self).__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.foo(x)
                x = self.bar(x)
                return x

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        torch.jit.save(first_script_module, first_saved_module)
        first_saved_module.seek(0)

        clear_class_registry()

        class Foo(torch.nn.Module):
            def __init__(self):
                super(Foo, self).__init__()
                self.foo = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.foo(x)
                return x

        second_script_module = torch.jit.script(Foo())
        second_saved_module = io.BytesIO()
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)
        second_saved_module.seek(0)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name, second_script_module._c.qualified_name
        )

        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = io.BytesIO()
        torch.jit.save(sm, contains_both)
        contains_both.seek(0)
        sm = torch.jit.load(contains_both)

    def test_different_functions(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """
        def lol(x):
            return x

        class Foo(torch.nn.Module):
            def forward(self, x):
                return lol(x)

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        torch.jit.save(first_script_module, first_saved_module)
        first_saved_module.seek(0)

        clear_class_registry()

        def lol(x):  # noqa: F811
            return "hello"

        class Foo(torch.nn.Module):
            def forward(self, x):
                return lol(x)

        second_script_module = torch.jit.script(Foo())
        second_saved_module = io.BytesIO()
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)
        second_saved_module.seek(0)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name, second_script_module._c.qualified_name
        )

        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = io.BytesIO()
        torch.jit.save(sm, contains_both)
        torch.jit.save(sm, "/scratch/suo/both.pt")
        contains_both.seek(0)
        sm = torch.jit.load(contains_both)

    def test_different_interfaces(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """
        @torch.jit.interface
        class MyInterface(object):
            def bar(self, x):
                # type: (Tensor) -> Tensor
                pass

        @torch.jit.script
        class ImplementInterface(object):
            def __init__(self):
                pass

            def bar(self, x):
                return x

        class Foo(torch.nn.Module):
            __annotations__ = {"interface": MyInterface}

            def __init__(self):
                super().__init__()
                self.interface = ImplementInterface()

            def forward(self, x):
                return self.interface.bar(x)

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        torch.jit.save(first_script_module, first_saved_module)
        first_saved_module.seek(0)

        clear_class_registry()

        @torch.jit.interface
        class MyInterface(object):
            def not_bar(self, x):
                # type: (Tensor) -> Tensor
                pass

        @torch.jit.script  # noqa: F811
        class ImplementInterface(object):  # noqa: F811
            def __init__(self):
                pass

            def not_bar(self, x):
                return x

        class Foo(torch.nn.Module):
            __annotations__ = {"interface": MyInterface}

            def __init__(self):
                super().__init__()
                self.interface = ImplementInterface()

            def forward(self, x):
                return self.interface.not_bar(x)

        second_script_module = torch.jit.script(Foo())
        second_saved_module = io.BytesIO()
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)
        second_saved_module.seek(0)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name, second_script_module._c.qualified_name
        )

        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = io.BytesIO()
        torch.jit.save(sm, contains_both)
        contains_both.seek(0)
        sm = torch.jit.load(contains_both)
