# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import os
import re
import sys
import types
import typing
import typing_extensions
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.jit.frontend
import torch.nn as nn
from torch import Tensor
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import (
    _tmp_donotuse_dont_inline_everything,
    JitTestCase,
)


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestRecursiveScript(JitTestCase):
    def test_inferred_nonetype(self):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = None

            def forward(self):
                assert self.x is None

        m = torch.jit.script(M())
        self.checkModule(M(), ())

    def test_script_function_attribute(self):
        @torch.jit.script
        def fn1(x):
            return x + x

        @torch.jit.script
        def fn2(x):
            return x - x

        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        self.checkModule(fn1_mod, (torch.randn(2, 2),))
        self.checkModule(fn2_mod, (torch.randn(2, 2),))

    def test_python_function_attribute(self):
        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        mod = M(torch.sigmoid)

        self.checkModule(mod, (torch.randn(2, 2),))

    def test_failed_function_compilation(self):
        def fn(x):
            return i_dont_exist  # noqa: F821

        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        m = M(fn)
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "failed to compile", "i_dont_exist"
        ):
            torch.jit.script(m)

    def test_init_error(self):
        class M(nn.Module):
            def __init__(self) -> None:
                self.x = 2

            def forward(self):
                pass

        with self.assertRaisesRegex(RuntimeError, "has not been initialized"):
            torch.jit.script(M())

    def test_script_after_eval(self):
        class M(nn.Module):
            def forward(self):
                if self.training:
                    return 2
                else:
                    return 0

        m = M()
        sm1 = torch.jit.script(m)
        m.eval()
        sm2 = torch.jit.script(m)

        # m is in eval mode, training should be False
        self.assertFalse(m.training)

        # sm1 was created while m had training = True
        self.assertTrue(sm1.training)
        self.assertEqual(sm1.training, sm1._c.getattr("training"))
        self.assertEqual(sm1(), 2)

        # sm2 was created after m was eval'ed
        self.assertFalse(sm2.training)
        self.assertEqual(sm2.training, sm2._c.getattr("training"))
        self.assertEqual(sm2(), 0)

    def test_module_name(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        m = torch.jit.script(MyModule())
        FileCheck().check("MyModule").run(m.graph)

    def test_repeated_error_stack(self):
        def d(x):
            return "a" - 2

        def c(x):
            return d(x)

        def b(x):
            return c(x)

        def a(x):
            return b(x)

        try:
            torch.jit.script(a)
        except Exception as e:
            FileCheck().check_count("is being compiled", 2).run(str(e))

        try:
            torch.jit.script(a)
        except Exception as e:
            # Make sure that no entries are left over from the previous failure
            FileCheck().check_count("is being compiled", 2).run(str(e))

    def test_constants_with_final(self):
        class M1(torch.nn.Module):
            x: torch.jit.Final[int]

            def __init__(self) -> None:
                super().__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        self.checkModule(M1(), (torch.randn(2, 2),))

        class M2(torch.nn.Module):
            x: typing_extensions.Final[int]

            def __init__(self) -> None:
                super().__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        self.checkModule(M2(), (torch.randn(2, 2),))

        class M3(torch.nn.Module):
            x: typing.Final[int]

            def __init__(self) -> None:
                super().__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        self.checkModule(M3(), (torch.randn(2, 2),))

    def test_ignore_class(self):
        @torch.jit.ignore
        class MyScriptClass:
            def unscriptable(self):
                return "a" + 200

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return MyScriptClass()

        with self.assertRaisesRegexWithHighlight(
            torch.jit.frontend.FrontendError,
            "Cannot instantiate class",
            "MyScriptClass",
        ):
            t = torch.jit.script(TestModule())

    def test_method_call(self):
        class M(nn.Module):
            def test(self, x):
                return x

            def forward(self, z):
                y = self.test(z)
                return z + 20 + y

        self.checkModule(M(), (torch.randn(2, 2),))

    def test_module_repr(self):
        class Submodule(nn.Module):
            def forward(self, x):
                return x

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(10, 10, 3)
                self.lin = nn.Linear(10, 10)
                self.sub = Submodule()

            def forward(self, x):
                return self.lin(x) + self.sub(x) + self.conv(x)

        m = torch.jit.script(MyModule())

        with self.capture_stdout() as out:
            print(m)

        f = FileCheck()
        f.check("MyModule")
        f.check("Conv2d")
        f.check("Linear")
        f.check("Submodule")
        f.run(out[0])

        self.assertEqual(m.original_name, "MyModule")

    def test_dir(self):
        def test_module_dir(mod):
            dir_set = dir(mod)
            scripted_mod = torch.jit.script(mod)
            dir_scripted = set(dir(scripted_mod))
            # set not currently copied over
            ignore_set = [
                "training",
                "__delitem__",
                "__setitem__",
                "clear",
                "items",
                "keys",
                "pop",
                "update",
                "values",
            ]
            for attr in dir_set:
                if attr in ignore_set:
                    continue
                self.assertTrue(attr in dir_scripted, attr)

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(10, 10, 3)
                self.lin = nn.Linear(10, 10)

            def forward(self, x):
                return self.lin(x) + self.conv(x)

        test_module_dir(MyModule())

        # test custom __dir__ for containers
        conv = nn.Conv2d(10, 10, 3)
        linear = nn.Linear(10, 10)

        test_module_dir(nn.Sequential(conv, linear))
        test_module_dir(
            nn.ModuleDict(OrderedDict([("conv", conv), ("linear", linear)]))
        )

    def test_class_compile(self):
        def other_fn(a: int, b: Tensor) -> Tensor:
            return a * b

        class B:
            def __init__(self, x):
                self.x = 2

            def helper(self, a):
                return self.x + a + other_fn(self.x, a)

        class N(torch.nn.Module):
            def forward(self, x):
                b = B(x)
                return b.helper(x)

        self.checkModule(N(), (torch.randn(2, 2),))

    def test_error_stack(self):
        def d(x: int) -> int:
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
        def d(x: int) -> int:
            return x + 10

        def c(x):
            return d("hello") + d(x)

        def b(x):
            return c(x)

        class Submodule(torch.nn.Module):
            def forward(self, x):
                return b(x)

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
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
            checker.check("'c' is being compiled since it was called from 'b'")
            checker.check("'b' is being compiled since it was called from")
            checker.run(str(e))

    @_tmp_donotuse_dont_inline_everything
    def test_script_basic(self):
        def a_python_fn(a, b, c):
            return a + b + c

        @torch.jit.script
        def a_script_fn(d, e, f):
            return a_python_fn(d, e, f)

        graph = str(a_script_fn.graph)
        FileCheck().check("prim::CallFunction").run(graph)
        FileCheck().check_not("^a_python_fn").run(graph)
        t = torch.ones(2, 2)
        self.assertEqual(a_script_fn(t, t, t), t + t + t)

    def test_error_stack_class(self):
        class X:
            def bad_fn(self):
                import pdb  # noqa: F401

        def fn(x) -> X:
            return X(10)

        try:
            torch.jit.script(fn)
        except Exception as e:
            checker = FileCheck()
            checker.check("import statements")
            checker.check("is being compiled since it was called from")
            checker.run(str(e))

    def test_error_stack_annotation(self):
        class X:
            def bad_fn(self):
                import pdb  # noqa: F401

        def fn(x) -> X:
            return X(10)

        try:
            torch.jit.script(fn)
        except Exception as e:
            checker = FileCheck()
            checker.check("import statements")
            checker.check("is being compiled since it was called from")
            checker.check("-> X")
            checker.run(str(e))

    def test_module_basic(self):
        class Other(torch.nn.Module):
            __constants__ = ["x"]

            def __init__(self, x):
                super().__init__()
                self.x = x
                self.param = torch.nn.Parameter(torch.ones(2, 2))

            def some_unscriptable_method(self):
                a = 2
                a = [2]
                return a

            def forward(self, t):
                return t + self.x + self.param

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.other = Other(200)

            def forward(self, t):
                return self.other(t) * 2

        self.checkModule(M(), (torch.ones(2, 2),))

    def test_module_function_export(self):
        class Other(torch.nn.Module):
            __constants__ = ["x"]

            def __init__(self, x):
                super().__init__()
                self.x = x
                self.param = torch.nn.Parameter(torch.ones(2, 2))

            @torch.jit.export
            def some_entry_point(self, y):
                return y + 20

            def forward(self, t):
                return t + self.x + self.param

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.other = Other(200)

            def forward(self, t):
                return self.other(t) * 2

        self.checkModule(M(), (torch.ones(2, 2),))

    def test_iterable_modules(self):
        class Inner(torch.nn.Module):
            def forward(self, x):
                return x + 10

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sequential = nn.Sequential(
                    Inner(), Inner(), nn.Sequential(Inner(), Inner())
                )
                self.module_list = nn.ModuleList([Inner(), Inner()])

            def forward(self, x):
                for mod in self.module_list:
                    x += mod(x)
                x += self.sequential(x)
                return x

        self.checkModule(M(), (torch.randn(5, 5),))

    def test_prepare_scriptable_basic(self):
        class SeluButReluWhenScripted(torch.nn.SELU):
            def __prepare_scriptable__(self):
                return nn.ReLU()

        t = torch.randn(5, 5)
        m = SeluButReluWhenScripted()
        sm = torch.jit.script(m)
        eager_out = m(t)
        script_out = sm(t)
        self.assertNotEqual(eager_out, script_out)

    def test_prepare_scriptable_iterable_modules(self):
        class SeluButReluWhenScripted(torch.nn.SELU):
            def __prepare_scriptable__(self):
                return nn.ReLU()

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                shared = SeluButReluWhenScripted()
                self.sequential = nn.Sequential(
                    SeluButReluWhenScripted(),
                    SeluButReluWhenScripted(),
                    nn.Sequential(
                        SeluButReluWhenScripted(), shared, SeluButReluWhenScripted()
                    ),
                    shared,
                )
                self.module_list = nn.ModuleList(
                    [SeluButReluWhenScripted(), shared, SeluButReluWhenScripted()]
                )

            def forward(self, x):
                for mod in self.module_list:
                    x += mod(x)
                x += self.sequential(x)
                return x

        t = torch.randn(5, 5)
        m = M()
        eager_out = m(t.clone())
        sm = torch.jit.script(m)
        script_out = sm(t.clone())
        self.assertNotEqual(eager_out, script_out)

    def test_prepare_scriptable_cycle(self):
        t = torch.randn(5, 5)
        c = torch.nn.Module()
        p = torch.nn.Module()
        c.__dict__["_p"] = p
        p.__dict__["_c"] = c

        sm = torch.jit.script(p)

    def test_prepare_scriptable_escape_hatch(self):
        class NonJitableClass:
            def __call__(self, int1, int2, *args):
                total = int1 + int2
                for arg in args:
                    total += arg
                return total

        obj = NonJitableClass()

        self.assertEqual(obj(1, 2), 3)
        self.assertEqual(obj(1, 2, 3, 4), 10)
        with self.assertRaisesRegex(
            torch.jit.frontend.NotSupportedError,
            expected_regex="can't take variable number of arguments",
        ):
            torch.jit.script(obj)

        def escape_hatch(int1: int, int2: int) -> int:
            return int1 + int2

        class NonJitableClassWithEscapeHatch(NonJitableClass):
            def __prepare_scriptable__(self):
                return escape_hatch

        jit_obj = torch.jit.script(NonJitableClassWithEscapeHatch())

        self.assertEqual(jit_obj(1, 2), 3)
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "expected at most 2 argument(s) but received 4 argument(s)"
            ),
        ):
            jit_obj(1, 2, 3, 4)

    def test_attributes(self):
        @torch.jit.script
        class Inner2:
            def __init__(self) -> None:
                self.b = "a string"

        @torch.jit.script
        class Foo:
            def __init__(self) -> None:
                self.a = 4
                self.inner = Inner2()

        @torch.jit.script
        class SFoo:
            def __init__(self) -> None:
                self.a = 4
                self.inner = Inner2()

            def __setstate__(self, obj: Tuple[int, Inner2]) -> None:
                a, inner = obj
                self.a = a
                self.inner = inner

            def __getstate__(self):
                return (self.a, self.inner)

        untyped_values = (
            ("my_dict", {"I": "am", "a test": "test"}),
            ("my_float", 2.3),
            ("my_int", 99),
            ("my_bool", False),
            ("my_tuple", (1, 2, 3, 4)),
            ("my_list", [(1, 2), (3, 4)]),
            # ('my_tensor', torch.randn(2, 2)),
            ("my_int_list", [1, 2, 3, 4]),
            # ('my_tensor_list', [torch.ones(2, 2) + i for i in range(4)]),
            ("my_bool_list", [True, True, False, True]),
            ("my_float_list", [1.0, 2.0, 3.0, 4.0]),
            ("my_str_list", ["hello", "bye"]),
        )
        typed_values = (
            ("my_empty_list", []),
            ("my_empty_dict", {}),
            ("my_none", None),
            ("my_object", Foo()),
            ("my_object2", SFoo()),
        )

        class M(torch.nn.Module):
            # TODO: re-enable this once this test is in a Python 3-only syntax
            # file
            # my_empty_list : List[int]
            # my_empty_dict : Dict[str, int]
            # my_none : Optional[int]

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
            "my_empty_list": List[int],
            "my_empty_dict": Dict[str, int],
            "my_none": Optional[int],
            "my_object": Foo,
            "my_object2": SFoo,
        }

        m = M()
        for name, value in untyped_values + typed_values:
            setattr(m, name, value)

        self.checkModule(m, (torch.randn(5, 5),))

    def test_function_attribute_in_submodule(self):
        class N(nn.Module):
            def __init__(self, norm):
                super().__init__()
                self.activation = torch.nn.functional.relu
                self.norm = norm

            def forward(self, src):
                output = src
                output = self.norm(output)
                return output

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                encoder_norm = nn.ReLU()
                self.encoder = N(encoder_norm)

            def forward(self, x):
                return self.encoder(x)

        m = M()
        self.checkModule(m, (torch.randn(5, 5),))

    def test_inner_traced_module(self):
        class Dummy(nn.Module):
            def forward(self, x):
                return x

        class Model(nn.Module):
            def __init__(self, dummies):
                super().__init__()
                self._dummies = dummies

            def forward(self, x):
                out = []
                for dummy in self._dummies:
                    out.append(dummy(x))
                return out

        dummy = torch.jit.trace(Dummy(), torch.randn(1, 2))
        dummies = nn.ModuleList([dummy])
        model = Model(dummies)
        self.checkModule(model, (torch.rand(5, 5),))

    def test_script_loaded_module(self):
        """
        Test that we can hold a loaded ScriptModule as a submodule.
        """

        class Dummy(nn.Module):
            def forward(self, x):
                return x

        dummy = torch.jit.script(Dummy())
        dummy = self.getExportImportCopy(dummy)

        class ContainsLoaded(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = dummy

            def forward(self, input):
                return self.encoder(input)

        self.checkModule(ContainsLoaded(), (torch.rand(2, 3),))

    def test_optional_module(self):
        class Dummy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = nn.Linear(2, 2)

            def forward(self, x):
                if self.foo is not None:
                    return self.foo(x)
                return x

        mod = Dummy()
        self.checkModule(mod, (torch.rand(2, 2),))
        mod.foo = None
        self.checkModule(mod, (torch.rand(2, 2),))

    def test_override_instance_method_ignore(self):
        class M(torch.nn.Module):
            @torch.jit.ignore
            def i_am_ignored(self):
                return "old"

        m = M()

        # Override the ignored method by binding a new method to this instance.
        @torch.jit.ignore
        def i_am_ignored(self):
            return "new"

        m.i_am_ignored = types.MethodType(i_am_ignored, m)
        self.assertEqual(m.i_am_ignored(), "new")

        # ScriptModule should correctly reflect the override.
        s = torch.jit.script(m)
        self.assertEqual(s.i_am_ignored(), "new")
