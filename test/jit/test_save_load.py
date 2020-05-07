import os
import io
import sys
import random
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
    def test_versioned_symbols(self):
        """
        Tests Torchscript symbol versioning. See note [Versioned Symbols].
        This test uses an undocumented, test-only function
        torch._test_serialization_subcmul.

        This function is implemented as (a - alpha * b) with a default value
        of 1 for alpha. In file format version 2, however, it was implemented
        as (b - alpha * a) with a default value of 2 for alpha.
        This test verifies a module seralized with file format version 2
        exhibits the old behavior, and that the same module newly serialized
        exhibits the current behavior.
        #T
        """
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, a, b, alpha: float):
                no_alpha = torch._test_serialization_subcmul(a, b)
                with_alpha = torch._test_serialization_subcmul(a, b, alpha)
                return no_alpha, with_alpha

        def historic_subcmul(a, b, alpha=2):
            return b - alpha * a

        def current_subcmul(a, b, alpha=1):
            return a - alpha * b

        # Loads and verifies the historic behavior of the module
        # that was serialized with version 2
        module_v2 = torch.jit.load(pytorch_test_dir + "/jit/fixtures/_test_serialization_subcmul_v2.pt")
        a = torch.randn((5,))
        b = torch.randn((5,))
        alpha = random.random()
        args = (a, b, alpha)
        no_alpha_v2, with_alpha_v2 = module_v2(*args)
        self.assertEqual(no_alpha_v2, historic_subcmul(a, b))
        self.assertEqual(with_alpha_v2, historic_subcmul(*args))

        # Scripts, saves, loads and verifies the current behavior of the module
        scripted_module = torch.jit.script(MyModule())
        buffer = io.BytesIO()
        torch.jit.save(scripted_module, buffer)
        buffer.seek(0)
        module_current = torch.jit.load(buffer)
        no_alpha_current, with_alpha_current = module_current(*args)
        self.assertEqual(no_alpha_current, current_subcmul(a, b))
        self.assertEqual(with_alpha_current, current_subcmul(*args))

    def test_versioned_symbols_reserialization(self):
        """
        Tests that loading and saving serialized Torchscript with a versioned
        symbol won't persist the original function and will inline the
        versioned builtin.
        """
        module_v2 = torch.jit.load(pytorch_test_dir + "/jit/fixtures/_test_serialization_subcmul_v2.pt")
        buffer = io.BytesIO()
        torch.jit.save(module_v2, buffer)
        buffer.seek(0)
        module_reserialized = torch.jit.load(buffer)

        subcmul_nodes = sum("subcmul" in n.kind() for
                            n in module_reserialized.graph.nodes())
        self.assertEqual(subcmul_nodes, 0)

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

    def test_many_collisions(self):
        class MyCoolNamedTuple(NamedTuple):
            a: int

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

        def lol(x):
            return x

        class Foo(torch.nn.Module):
            interface: MyInterface

            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)
                self.interface = ImplementInterface()

            def forward(self, x):
                x = self.foo(x)
                x = self.bar(x)
                x = lol(x)
                x = self.interface.bar(x)

                return x, MyCoolNamedTuple(a=5)


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

        @torch.jit.script   # noqa F811
        class ImplementInterface(object):  # noqa F811
            def __init__(self):
                pass

            def not_bar(self, x):
                return x

        def lol(x):  # noqa F811
            return "asdofij"

        class MyCoolNamedTuple(NamedTuple):  # noqa F811
            a: str

        class Foo(torch.nn.Module):
            interface: MyInterface

            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.interface = ImplementInterface()

            def forward(self, x):
                x = self.foo(x)
                self.interface.not_bar(x)
                x = lol(x)
                return x, MyCoolNamedTuple(a="hello")

        second_script_module = torch.jit.script(Foo())
        second_saved_module = io.BytesIO()
        torch.jit.save(second_script_module, second_saved_module)
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
                x, named_tuple_1 = self.first(x)
                x, named_tuple_2 = self.second(x)
                return len(x + named_tuple_2.a) + named_tuple_1.a

        sm = torch.jit.script(ContainsBoth())
        contains_both = io.BytesIO()
        torch.jit.save(sm, contains_both)
        contains_both.seek(0)
        sm = torch.jit.load(contains_both)
