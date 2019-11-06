import os
import sys
from typing import List

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from jit_utils import JitTestCase

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
                self.b = 0

        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.mods = torch.nn.ModuleList([HasA(), HasB()])

            def forward(self):
                # use a list to encode hasattr results
                l : List[int] = []
                for mod in self.mods:
                    l.append(int(hasattr(mod, "a")))
                    l.append(int(hasattr(mod, "b")))
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
