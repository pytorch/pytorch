import sys
import torch
from jit_utils import JitTestCase
from contextlib import contextmanager

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

    def test_matmul_py3(self):
        def fn(a, b):
            return a @ b

        a = torch.rand(4, 3, requires_grad=True)
        b = torch.rand(3, 2, requires_grad=True)
        self.checkScript(fn, (a, b), optimize=True)
