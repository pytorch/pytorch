import sys
import torch
from jit_utils import JitTestCase
from contextlib import contextmanager
from torch import Tensor
from typing import Tuple

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

    def test_matmul(self):
        def fn(a, b):
            return a @ b

        a = torch.rand(4, 3, requires_grad=True)
        b = torch.rand(3, 2, requires_grad=True)
        self.checkScript(fn, (a, b), optimize=True)

    def test_python_frontend(self):
        def fn():
            raise Exception("hello")
        ast = torch.jit.frontend.get_jit_def(fn)
        self.assertExpected(str(ast))

    def _make_scalar_vars(self, arr, dtype):
        return [torch.tensor(val, dtype=dtype) for val in arr]

    def test_string_print(self):
        def func(a):
            print(a, "a" 'b' '''c''' """d""", 2, 1.5)
            return a

        inputs = self._make_scalar_vars([1], torch.int64)
        self.checkScript(func, inputs, capture_output=True)

    def test_type_annotation(self):
        def fn(x : torch.Tensor, y : Tensor, z) -> Tuple[Tensor, Tensor, Tensor]:
            return (x, y + z, z)

        with self.assertRaisesRegex(RuntimeError, r"expected a value of type 'Tensor' for argument"
                                                  r" '0' but instead found type 'Tuple\[Tensor,"):
            @torch.jit.script
            def bad_fn(x):
                x, y = fn((x, x), x, x)
                return y

        with self.assertRaisesRegex(RuntimeError, r"too many values .* need 2 but found 3"):
            @torch.jit.script
            def bad_fn2(x):
                x, y = fn(x, x, x)
                return y

        with self.assertRaisesRegex(RuntimeError, r"need 4 values .* found only 3"):
            @torch.jit.script
            def bad_fn3(x):
                x, y, z, w = fn(x, x, x)
                return y

        def good_fn(x):
            y, z, w = fn(x, x, x)
            return y, z, w

        self.checkScript(good_fn, (torch.ones(2, 2),), optimize=True)
