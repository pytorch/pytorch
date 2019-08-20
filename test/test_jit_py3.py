from common_utils import run_tests
from jit_utils import JitTestCase
from torch.testing import FileCheck
from typing import NamedTuple, List, Optional

import torch


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
        torch._C._jit_clear_class_registry()
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
            for _ in range(10):
                a.append(4)
                c = torch.ones(2, 2)
            return a, b, c

        self.checkScript(fn, ())

        def wrong_type():
            wrong : List[int] = [0.5]
            return wrong

        with self.assertRaisesRegex(RuntimeError, "Lists must contain only a single type"):
            torch.jit.script(wrong_type)

    def test_parser_bug(self):
        def parser_bug(o: Optional[torch.Tensor]):
            pass



if __name__ == '__main__':
    run_tests()
