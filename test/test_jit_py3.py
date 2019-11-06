from common_utils import run_tests
from jit_utils import JitTestCase
from torch.testing import FileCheck
from typing import NamedTuple, List, Optional, Any
from jit.test_module_interface import TestModuleInterface  # noqa: F401
import unittest
import sys
import torch
import torch.nn as nn
import jit_utils

from torch import Tensor

class TestScriptPy3(JitTestCase):
    def test_joined_str(self):
        def func(x):
            hello, test = "Hello", "test"
            print(f"{hello + ' ' + test}, I'm a {test}") # noqa E999
            print(f"format blank")
            hi = 'hi'
            print(f"stuff before {hi}")
            print(f"{hi} stuff after")
            return x + 1

        x = torch.arange(4., requires_grad=True)
        # TODO: Add support for f-strings in string parser frontend
        # self.checkScript(func, [x], optimize=True, capture_output=True)

        with self.capture_stdout() as captured:
            out = func(x)

        scripted = torch.jit.script(func)
        with self.capture_stdout() as captured_script:
            out_script = func(x)

        self.assertAlmostEqual(out, out_script)
        self.assertEqual(captured, captured_script)

    def test_named_tuple(self):
        class FeatureVector(NamedTuple):
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x) -> float:
            fv = FeatureVector(3.0, [3.0], 3.0)  # noqa
            rv = fv.float_features
            for val in fv.sequence_features:
                rv += val
            rv *= fv.time_since_first
            return rv

        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    @unittest.skipIf(sys.version_info[0] < 3 and sys.version_info[1] < 6, "dict not ordered")
    def test_dict_preserves_order(self):
        def dict_ordering():
            a : Dict[int, int] = {}
            for i in range(1000):
                a[i] = i + 1
            return a

        self.checkScript(dict_ordering, ())
        di = torch.jit.script(dict_ordering)()
        res = list(di.items())
        for i in range(1000):
            key, value = res[i]
            self.assertTrue(key == i and value == i + 1)

    def test_return_named_tuple(self):
        class FeatureVector(NamedTuple):
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x):
            fv = FeatureVector(3.0, [3.0], 3.0)
            return fv

        out = foo(torch.rand(3, 4))
        out = foo(torch.rand(3, 4))
        self.assertEqual(out.float_features, 3.0)
        self.assertEqual(out.sequence_features, [3.0])
        self.assertEqual(out.time_since_first, 3.0)

    def test_named_tuple_slice_unpack(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        @torch.jit.script
        def foo(a : int, b : float, c : List[int]):
            tup = MyCoolNamedTuple(a, b, c)  # noqa
            my_a, my_b, my_c = tup
            return tup[:1], my_a, my_c

        self.assertEqual(foo(3, 3.5, [6]), ((3,), 3, [6]))

    def test_named_tuple_lower(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        @torch.jit.script
        def foo(a : int):
            tup = MyCoolNamedTuple(a, 3.14, [9])  # noqa
            return tup

        FileCheck().check('TupleConstruct').run(foo.graph)
        torch._C._jit_pass_lower_all_tuples(foo.graph)
        FileCheck().check_not('TupleConstruct').run(foo.graph)

    def test_named_tuple_type_annotation(self):
        global MyCoolNamedTuple  # see [local resolution in python]

        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        @torch.jit.script
        def foo(x : MyCoolNamedTuple) -> MyCoolNamedTuple:
            return x

        mnt = MyCoolNamedTuple(42, 420.0, [666])
        self.assertEqual(foo(mnt), mnt)

    def test_named_tuple_wrong_types(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'int' for argument 'a'"
                                                  " but instead found type 'str'"):
            @torch.jit.script
            def foo():
                tup = MyCoolNamedTuple('foo', 'bar', 'baz')  # noqa
                return tup

    def test_named_tuple_kwarg_construct(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        @torch.jit.script
        def foo():
            tup = MyCoolNamedTuple(c=[1, 2, 3], b=3.5, a=9)  # noqa
            return tup

        tup = foo()
        self.assertEqual(tup.a, 9)
        self.assertEqual(tup.b, 3.5)
        self.assertEqual(tup.c, [1, 2, 3])

    def test_named_tuple_default_error(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int] = [3, 4, 5]

        with self.assertRaisesRegex(RuntimeError, 'Default values are currently not supported'):
            @torch.jit.script
            def foo():
                tup = MyCoolNamedTuple(c=[1, 2, 3], b=3.5, a=9)  # noqa
                return tup

    @unittest.skipIf(True, "broken while these tests were not in CI")
    def test_named_tuple_serialization(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self):
                return MyCoolNamedTuple(3, 3.5, [3, 4, 5])

        mm = MyMod()
        mm.save('foo.zip')
        jit_utils.clear_class_registry()
        loaded = torch.jit.load('foo.zip')

        out = mm()
        out_loaded = loaded()

        for name in ['a', 'b', 'c']:
            self.assertEqual(getattr(out_loaded, name), getattr(out, name))

    def test_type_annotate_py3(self):
        def fn():
            a : List[int] = []
            b : torch.Tensor = torch.ones(2, 2)
            c : Optional[torch.Tensor] = None
            d : Optional[torch.Tensor] = torch.ones(3, 4)
            for _ in range(10):
                a.append(4)
                c = torch.ones(2, 2)
                d = None
            return a, b, c, d

        self.checkScript(fn, ())

        def wrong_type():
            wrong : List[int] = [0.5]
            return wrong

        with self.assertRaisesRegex(RuntimeError, "Lists must contain only a single type"):
            torch.jit.script(wrong_type)

    def test_parser_bug(self):
        def parser_bug(o: Optional[torch.Tensor]):
            pass

    def test_mismatched_annotation(self):
        with self.assertRaisesRegex(RuntimeError, 'annotated with type'):
            @torch.jit.script
            def foo():
                x : str = 4
                return x

    def test_reannotate(self):
        with self.assertRaisesRegex(RuntimeError, 'declare and annotate'):
            @torch.jit.script
            def foo():
                x = 5
                if True:
                    x : Optional[int] = 7

    def test_module_annotation(self):
        class Foo(nn.Module):
            def __init__(self):
                super(Foo, self).__init__()

            def forward(self, x):
                return x + 2

        class Bar(nn.Module):
            foo: Foo

            def __init__(self):
                super(Bar, self).__init__()
                self.foo = Foo()

            def forward(self, x):
                return self.foo(x)

        scripted_mod = torch.jit.script(Bar())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), input + 2)

    def test_any_in_class_fails(self):
        class MyCoolNamedTuple(NamedTuple):
            a : Any
            b : float
            c : List[int]
        with self.assertRaisesRegex(RuntimeError, "contains an Any"):
            @torch.jit.script
            def foo():
                return MyCoolNamedTuple(4, 5.5, [3])
            print(foo.graph)

    def test_module_interface(self):
        global OneTwo, OneTwoClass, Foo
        @torch.jit.interface
        class OneTwo(nn.Module):
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def two(self, x):
                # type: (Tensor) -> Tensor
                pass

            def forward(self, x):
                # type: (Tensor) -> Tensor
                pass

        @torch.jit.interface
        class OneTwoClass(object):
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def two(self, x):
                # type: (Tensor) -> Tensor
                pass

        class FooMod(nn.Module):
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                return x + y

            def two(self, x):
                # type: (Tensor) -> Tensor
                return 2 * x

            def forward(self, x):
                # type: (Tensor) -> Tensor
                return self.one(self.two(x), x)

        class BarMod(nn.Module):
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                return x * y

            def two(self, x):
                # type: (Tensor) -> Tensor
                return 2 / x

            def forward(self, x):
                # type: (Tensor) -> Tensor
                return self.two(self.one(x, x))

            @torch.jit.export
            def forward2(self, x):
                # type: (Tensor) -> Tensor
                return self.two(self.one(x, x)) + 1

        def use_module_interface(mod_list, x):
            # type: (List[OneTwo], Tensor) -> Tensor
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        def use_class_interface(mod_list, x):
            # type: (List[OneTwoClass], Tensor) -> Tensor
            return mod_list[0].two(x) + mod_list[1].one(x, x)

        scripted_foo_mod = torch.jit.script(FooMod())
        scripted_bar_mod = torch.jit.script(BarMod())
        self.checkScript(use_module_interface,
                         ([scripted_foo_mod, scripted_bar_mod], torch.rand(3, 4),))
        self.checkScript(use_class_interface,
                         ([scripted_foo_mod, scripted_bar_mod], torch.rand(3, 4),))

        @torch.jit.script
        def as_module_interface(x):
            # type: (OneTwo) -> OneTwo
            return x

        @torch.jit.script
        def as_class_interface(x):
            # type: (OneTwoClass) -> OneTwoClass
            return x

        @torch.jit.script
        class Foo(object):
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                return x + y

            def two(self, x):
                # type: (Tensor) -> Tensor
                return 2 * x

            def forward(self, x):
                # type: (Tensor) -> Tensor
                return self.one(self.two(x), x)

        # check class object is not a subtype of module interface
        with self.assertRaisesRegex(RuntimeError, "ScriptModule class can be subtype of module interface"):
            as_module_interface(Foo())

        def call_module_interface_on_other_method(mod_interface, x):
            # type: (OneTwo, Tensor) -> Tensor
            return mod_interface.forward2(x)

        # ensure error out when we call the module on the method other than the interface specified.
        with self.assertRaisesRegex(RuntimeError, "Tried to access nonexistent attribute or method"):
            self.checkScript(call_module_interface_on_other_method, (scripted_bar_mod, torch.rand(3, 4),))

        class WrongMod(nn.Module):
            def two(self, x):
                # type: (int) -> int
                return 2 * x

            def forward(self, x):
                # type: (Tensor) -> Tensor
                return x + torch.randn(3, self.two(3))

        scripted_wrong_mod = torch.jit.script(WrongMod())

        # wrong module that is not compatible with module interface
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            as_module_interface(scripted_wrong_mod)

        with self.assertRaisesRegex(RuntimeError, "does not support inheritance yet. Please directly"):
            @torch.jit.interface
            class InheritMod(FooMod):
                def three(self, x):
                    # type: (Tensor) -> Tensor
                    return 3 * x

if __name__ == '__main__':
    run_tests()
