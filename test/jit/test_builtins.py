import os
import sys
import unittest
from collections import namedtuple
from textwrap import dedent
from typing import List

import torch
from torch._six import PY2

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from jit_utils import JitTestCase, execWrapper

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestBuiltins(JitTestCase):
    """
    Tests for TorchScript support of Python builtin functions.
    See: https://docs.python.org/3/library/functions.html
    """
    def test_has_attr(self):
        class HasA(torch.nn.Module):
            def __init__(self):
                super(HasA, self).__init__()
                self.a = 0

        class HasB(torch.nn.Module):
            def __init__(self):
                super(HasB, self).__init__()
                self.b = 1

        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.mods = torch.nn.ModuleList([HasA(), HasB()])

            def forward(self):
                # use a list to encode hasattr results
                l = torch.jit.annotate(List[int], [])
                for mod in self.mods:
                    l.append(int(hasattr(mod, "a")))
                    l.append(int(hasattr(mod, "b")))
                    # actually retrieve the attr to test static refinement
                    if hasattr(mod, "a"):
                        l.append(mod.a)
                    if hasattr(mod, "b"):
                        l.append(mod.b)
                return l

        self.checkModule(Mod(), ())

    def test_has_attr_invalid_args(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.mod = torch.nn.Linear(1, 1)

            def forward(self, name):
                # not allowed, `name` must be static.
                return hasattr(self.mod, name)

        with self.assertRaisesRegex(RuntimeError, "hasattr"):
            torch.jit.script(Mod())

        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()

            def forward(self, name):
                # not allowed, `torch.rand` is not a class type
                return hasattr(torch.rand(2, 3), name)

        with self.assertRaisesRegex(RuntimeError, "hasattr"):
            torch.jit.script(Mod())

    def test_ord(self):
        def fn(x):
            # type: (str) -> int
            return ord(x)

        self.checkScript(fn, ("h"))
        self.checkScript(fn, ("y"))

        def index_str_to_tensor(s):
            # type: (str) -> int
            return torch.tensor(ord(s))  # noqa T484

        s = u'\u00a3'.encode('utf8')[:1]
        self.checkScript(index_str_to_tensor, (s,))

    def test_chr(self):
        def fn(x):
            # type: (int) -> str
            return chr(x)

        self.checkScript(fn, (1,))
        self.checkScript(fn, (97,))

    def test_round(self):
        def round_float(x):
            # type: (float) -> float
            return round(x)

        def round_int(x):
            # type: (int) -> float
            return round(x)

        self.checkScript(round_float, (1.5,))
        self.checkScript(round_int, (2,))

    def test_range_args(self):
        with self.assertRaisesRegex(RuntimeError, r'range expected at least 1 arguments, got 0'):
            @torch.jit.script
            def range_no_arg(x):
                for _ in range():
                    x += 1
                return x
        with self.assertRaisesRegex(RuntimeError, r'found float'):
            @torch.jit.script
            def range_non_float():
                for i in range(.5):
                    print(i)

    def test_number_all(self):
        def int1():
            return all(torch.tensor([1, 2, 3], dtype=torch.uint8))

        def int2():
            return all(torch.tensor([1, 0, 3], dtype=torch.uint8))

        self.checkScript(int1, ())
        self.checkScript(int2, ())

    @unittest.skipIf(PY2, "oct() format changed from PY2 to PY3")
    def test_convert_base(self):
        def test_hex(x):
            # type: (int) -> str
            return hex(x)

        def test_oct(x):
            # type: (int) -> str
            return oct(x)

        def test_bin(x):
            # type: (int) -> str
            return bin(x)

        numbers = [-1000, -10, 0, 1, 10, 2343]
        for n in numbers:
            self.checkScript(test_bin, (n,))
            self.checkScript(test_oct, (n,))
            self.checkScript(test_hex, (n,))

    @unittest.skipIf(PY2, "tuple printing in py2 is different than torchscript")
    def test_print(self):
        def func(x, y):
            q = (x + y).sigmoid()
            print(q, 1, 2, [1, 2], [1.0, 2.0])
            w = -q
            return w * w

        x = torch.arange(4., requires_grad=True)
        y = torch.arange(0., 8, 2, requires_grad=True)
        self.checkScript(func, [x, y], optimize=True, capture_output=True)

    def test_print_format(self):
        def func(x):
            print("{}, I'm a {}".format("Hello", "test"))
            print("format blank".format())
            print("stuff before {}".format("hi"))
            print("{} stuff after".format("hi"))
            return x + 1

        x = torch.arange(4., requires_grad=True)
        self.checkScript(func, [x], optimize=True, capture_output=True)

        @torch.jit.script
        def bad_func():
            print("{0}".format("hello"))
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            bad_func()

    def test_type_cast(self):
        template = dedent('''
        def func(v):
            # type: ({from_type}) -> {to_type}
            return {to_type}(v)
        ''')

        def check_cast(from_type, to_type, value, raises=False):
            code = template.format(from_type=from_type, to_type=to_type)
            self.checkScript(code, (value,))

        check_cast('int', 'float', 1)
        check_cast('int', 'bool', 1)
        check_cast('int', 'bool', 0)

        check_cast('float', 'int', 1.)
        check_cast('float', 'bool', 1.)
        check_cast('float', 'bool', 0.)

        check_cast('bool', 'int', True)
        check_cast('bool', 'float', True)

    def test_print_kwargs(self):
        with self.assertRaisesRegex(RuntimeError, 'print doesn\'t accept any keyword arguments'):
            cu = torch.jit.CompilationUnit('''
            def print_kwargs(x):
                print(x, flush=True)
                return x
            ''')

    def test_isinstance_metacompile(self):
        @torch.jit.script
        def test_primitive_type(x):
            # type: (int) -> int
            if isinstance(x, int):
                return x + 1
            else:
                return x - 1

        self.assertEqual(test_primitive_type(1), 2)
        with self.assertRaisesRegex(Exception, "Expected a value of type"):
            test_primitive_type(1.5)

        _MyNamedTuple = namedtuple('_MyNamedTuple', ['value'])

        @torch.jit.script
        def test_non_primitive_types(x):
            # type: (_MyNamedTuple) -> Tensor
            if isinstance(1, _MyNamedTuple):
                return 10

            if isinstance(x, _MyNamedTuple):
                return x.value + 1
            else:
                return 1

        out = test_non_primitive_types(_MyNamedTuple(value=torch.tensor(5.0)))
        self.assertEqual(out, torch.tensor(6.0))

    def test_isinstance_dynamic(self):
        @torch.jit.script
        def foo(a):
            # type: (Optional[List[int]]) -> int
            b = 0
            if isinstance(a, (int, (float,), list, str)):
                b += 1
            if isinstance(a, (int, str)):
                b += 1
            if isinstance(a, List[int]):
                b += 1
            return b
        self.assertEqual(foo([3, 4]), 2)
        self.assertEqual(foo(None), 0)

    def test_isinstance_refinement(self):
        @torch.jit.script
        def foo(a):
            # type: (Optional[int]) -> int
            if isinstance(a, int):
                return a + 3
            else:
                return 4
        self.assertEqual(foo(4), 7)
        self.assertEqual(foo(None), 4)
        @torch.jit.script
        def foo2(a, b):
            # type: (Optional[int], Optional[int]) -> int
            if not isinstance(a, int) or not isinstance(b, int):
                return 0
            else:
                return a + b
        self.assertEqual(foo2(3, 4), 7)
        self.assertEqual(foo2(None, 4), 0)
        self.assertEqual(foo2(4, None), 0)

        @torch.jit.script
        def any_refinement(a, b):
            # type: (Any, Any) -> int
            if isinstance(a, int) and isinstance(b, int):
                return a + b
            return 0

        self.assertEqual(any_refinement(3, 4), 7)
        self.assertEqual(any_refinement(3, "hi"), 0)

    def test_isinstance(self):
        # test isinstance operator for static type checking
        template = dedent('''
        def func(x):
            # type: ({type_hint}) -> bool
            return isinstance(x, {typ})
        ''')

        def test(inp, typ, type_hint):
            code = template.format(typ=typ, type_hint=type_hint)
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            self.assertEqual(
                cu.func(inp),
                scope['func'](inp),
                "Failed with typ: {}"
                .format(typ)
            )

        inputs = [True, 1, 1.0, torch.tensor(1), [1, 2], (1.0,), [1, 2], 1]
        type_literals = ['bool', 'int', 'float', 'torch.Tensor', 'list', 'tuple',
                         '(list, tuple)', '(int, float, bool)']
        type_annotations = ['bool', 'int', 'float', 'Tensor', 'List[int]', 'Tuple[float]',
                            'List[int]', 'int']

        # do zipping to try different types
        for inp, typ, type_hint in zip(inputs, type_literals, type_annotations):
            test(inp, typ, type_hint)

        # test optional isinstance check
        @torch.jit.script
        def opt_func(x):
            # type: (Optional[int]) -> bool
            return isinstance(x, int)
        self.assertTrue(opt_func(3))
        self.assertFalse(opt_func(None))
