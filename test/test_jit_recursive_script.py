from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import Tensor
import torch.nn as nn
import unittest
from test_jit import JitTestCase
from common_utils import run_tests, freeze_rng_state
from torch.testing import FileCheck
from typing import List, Dict, Optional, Tuple

class TestRecursiveScript(JitTestCase):
    def checkModule(self, nn_module, args):
        """
        Check that a nn.Module's results in Script mode match eager and that it
        can be exported
        """
        sm = torch.jit.script(nn_module)

        with freeze_rng_state():
            eager_out = nn_module(*args)

        with freeze_rng_state():
            script_out = sm(*args)

        self.assertEqual(eager_out, script_out)
        self.assertExportImportModule(sm, args)

        return sm

    def test_module_name(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        m = torch.jit.script(MyModule())
        FileCheck().check("ClassType<MyModule>").run(m.graph)

    @unittest.skipIf(True, "Class annotations are a thing in > 3.5, need to fix for < 3.7")
    def test_constants_with_final(self):
        class M(torch.nn.Module):
            # TODO: Use this (see below)
            # x : torch.jit.Final[int]

            def __init__(self):
                super(M, self).__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x


        # TODO: Fix this test so that we can actually define the class like
        #   class M(torch.nn.Module):
        #       x : torch.jit.Final[int]
        M.__annotations__ = {'x': torch.jit.Final[int]}

        m = M()

        self.checkModule(M(), (torch.randn(2, 2),))

    def test_method_call(self):
        class M(nn.Module):
            def test(self, x):
                return x

            def forward(self, z):
                y = self.test(z)
                return z + 20 + y

        self.checkModule(M(), (torch.randn(2, 2),))

    def test_class_compile(self):
        def other_fn(a, b):
            # type: (int, Tensor) -> Tensor
            return a * b

        class B(object):
            def __init__(self, x):
                self.x = 2

            def helper(self, a):
                return self.x + a + other_fn(self.x, a)


        class N(torch.nn.Module):
            def __init__(self):
                super(N, self).__init__()

            def forward(self, x):
                b = B(x)
                return b.helper(x)

        self.checkModule(N(), (torch.randn(2, 2),))

    def test_error_stack(self):
        def d(x):
            # type: (int) -> int
            return x + 10

        def c(x):
            return d("hello") + d(x)

        def b(x):
            return c(x)

        def a(x):
            return b(x)

        try:
            scripted = torch.jit.script(a)
        except RuntimeError as e:
            checker = FileCheck()
            checker.check("Expected a value of type 'int'")
            checker.check("def c(x)")
            checker.check("def b(x)")
            checker.check("def a(x)")
            checker.run(str(e))

    def test_error_stack_module(self):
        def d(x):
            # type: (int) -> int
            return x + 10

        def c(x):
            return d("hello") + d(x)

        def b(x):
            return c(x)

        class Submodule(torch.nn.Module):
            def __init__(self):
                super(Submodule, self).__init__()

            def forward(self, x):
                return b(x)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.submodule = Submodule()

            def some_method(self, y):
                return y + self.submodule(y)

            def forward(self, x):
                return self.some_method(x)

        try:
            scripted = torch.jit.script(M())
        except RuntimeError as e:
            checker = FileCheck()
            checker.check("Expected a value of type 'int'")
            checker.check("return d(\"hello\")")
            checker.check("\'b")
            checker.check("return c(x)")
            checker.check("Submodule.forward\'")
            checker.check("return b(x)")
            checker.check("M.some_method\'")
            checker.check("return y + self.submodule(y)")
            checker.check("M.forward\'")
            checker.check("return self.some_method(x)")
            checker.run(str(e))

    def test_script_basic(self):
        def a_python_fn(a, b, c):
            return a + b + c

        @torch.jit.script
        def a_script_fn(d, e, f):
            return a_python_fn(d, e, f)

        graph = str(a_script_fn.graph)
        FileCheck().check("aten::add").run(graph)
        FileCheck().check_not("a_python_fn").run(graph)
        t = torch.ones(2, 2)
        self.assertEqual(a_script_fn(t, t, t), t + t + t)

    def test_error_stack_class(self):
        class X(object):
            def bad_fn(self):
                import pdb  # noqa

        def fn(x):
            return X(10)

        try:
            torch.jit.script(fn)
        except Exception as e:
            checker = FileCheck()
            checker.check("import statements")
            checker.check("is being compiled since it was called from")
            checker.run(str(e))

    def test_module_basic(self):
        class Other(torch.nn.Module):
            __constants__ = ['x']

            def __init__(self, x):
                super(Other, self).__init__()
                self.x = x
                self.param = torch.nn.Parameter(torch.ones(2, 2))

            def some_unscriptable_method(self):
                a = 2
                a = [2]
                return a

            def forward(self, t):
                return t + self.x + self.param


        class M(torch.nn.Module):
            __constants__ = ['x']

            def __init__(self):
                super(M, self).__init__()
                self.other = Other(200)

            def forward(self, t):
                return self.other(t) * 2

        self.checkModule(M(), (torch.ones(2, 2),))

    def test_module_function_export(self):
        class Other(torch.nn.Module):
            __constants__ = ['x']

            def __init__(self, x):
                super(Other, self).__init__()
                self.x = x
                self.param = torch.nn.Parameter(torch.ones(2, 2))

            @torch.jit.export
            def some_entry_point(self, y):
                return y + 20

            def forward(self, t):
                return t + self.x + self.param


        class M(torch.nn.Module):
            __constants__ = ['x']

            def __init__(self):
                super(M, self).__init__()
                self.other = Other(200)

            def forward(self, t):
                return self.other(t) * 2

        self.checkModule(M(), (torch.ones(2, 2),))

    def test_iterable_modules(self):
        class Inner(torch.nn.Module):
            def forward(self, x):
                return x + 10

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.sequential = nn.Sequential(
                    Inner(),
                    Inner(),
                    nn.Sequential(Inner(), Inner())
                )
                self.module_list = nn.ModuleList([Inner(), Inner()])

            def forward(self, x):
                for mod in self.module_list:
                    x += mod(x)
                x += self.sequential(x)
                return x

        self.checkModule(M(), (torch.randn(5, 5),))

    def test_attributes(self):
        @torch.jit.script
        class Inner(object):
            def __init__(self):
                self.b = "a string"

        @torch.jit.script
        class Foo(object):
            def __init__(self):
                self.a = 4
                self.inner = Inner()

        @torch.jit.script
        class SFoo(object):
            def __init__(self):
                self.a = 4
                self.inner = Inner()

            def __setstate__(self, obj):
                # type: (Tuple[int, Inner]) -> None
                a, inner = obj
                self.a = a
                self.inner = inner

            def __getstate__(self):
                return (self.a, self.inner)


        untyped_values = (
            ('my_dict', {"I": "am", "a test": "test"}),
            ('my_float', 2.3),
            ('my_int', 99),
            ('my_bool', False),
            ('my_tuple', (1, 2, 3, 4)),
            ('my_list', [(1, 2), (3, 4)]),
            # ('my_tensor', torch.randn(2, 2)),
            ('my_int_list', [1, 2, 3, 4]),
            # ('my_tensor_list', [torch.ones(2, 2) + i for i in range(4)]),
            ('my_bool_list', [True, True, False, True]),
            ('my_float_list', [1., 2., 3., 4.]),
            ('my_str_list', ['hello', 'bye']),
        )
        typed_values = (
            ('my_empty_list', []),
            ('my_empty_dict', {}),
            ('my_none', None),
            ('my_object', Foo()),
            ('my_object2', SFoo()),
        )

        class M(torch.nn.Module):
            # TODO: re-enable this once this test is in a Python 3-only syntax
            # file
            # my_empty_list : List[int]
            # my_empty_dict : Dict[str, int]
            # my_none : Optional[int]

            def __init__(self):
                super(M, self).__init__()

            def forward(self, x):
                return (
                    self.my_dict,
                    self.my_float,
                    self.my_int,
                    self.my_bool,
                    # self.my_tensor,
                    self.my_int_list,
                    # self.my_tensor_list,
                    self.my_bool_list,
                    self.my_float_list,
                    self.my_str_list,
                    self.my_empty_list,
                    self.my_empty_dict,
                    self.my_none,
                    self.my_object.a,
                    self.my_object.inner.b,
                    self.my_object.a,
                    self.my_object2.inner.b,
                )

        # TODO: as a followup, fix this test
        # We can't define class attributes like we should be doing:
        #   class M(torch.nn.Module):
        #       my_empty_list : List[int]
        #       my_empty_dict : Dict[str, int]
        #       my_none : Optional[int]
        #       my_out_of_line_attribute: List[int] = [1, 2, 3]
        # since there's no string frontend for Python classes (so the `define`)
        # trick doesn't work.
        M.__annotations__ = {
            'my_empty_list': List[int],
            'my_empty_dict': Dict[str, int],
            'my_none': Optional[int],
            'my_object': Foo,
            'my_object2': SFoo,
        }

        m = M()
        for name, value in untyped_values + typed_values:
            setattr(m, name, value)

        self.checkModule(m, (torch.randn(5, 5),))


if __name__ == '__main__':
    run_tests()
