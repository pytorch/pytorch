# Owner(s): ["oncall: jit"]

import os
import sys
import unittest

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
from torch.jit.frontend import _IS_ASTUNPARSE_INSTALLED

if __name__ == "__main__":
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestIgnoreContextManager(JitTestCase):
    @unittest.skipUnless(_IS_ASTUNPARSE_INSTALLED, "astunparse package is required")
    def test_with_ignore_context_manager_with_inp_out(self):
        class A(torch.nn.Module):
            def forward(self):
                a: int = 4
                b: int = 5
                c: int = 0
                d: int = 6
                with torch.jit._IgnoreContextManager(a="inp:int", b="inp:int", c="out:int", d="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                    d = 9
                return c + d
        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), model())
        self.assertEqual(s(), 20)

        class B(torch.nn.Module):
            def forward(self):
                a: int = 4
                b: int = 5
                c: int = 0
                with torch.jit._IgnoreContextManager(a="inp:int", b="inp:int", c="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                return c
        model = B()
        s = torch.jit.script(model)
        self.assertEqual(s(), 11)
        self.assertEqual(s(), model())

        class C(torch.nn.Module):
            def forward(self):
                a: int = 4
                b: int = 5
                with torch.jit._IgnoreContextManager(a="inp:int", b="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    b = l[0] + a
                return b
        model = C()
        s = torch.jit.script(model)
        self.assertEqual(s(), 6)
        self.assertEqual(s(), model())

    @unittest.skipUnless(_IS_ASTUNPARSE_INSTALLED, "astunparse package is required")
    def test_with_ignore_context_manager_with_just_inp(self):
        class A(torch.nn.Module):
            def forward(self):
                a: int = 4
                b: int = 5
                with torch.jit._IgnoreContextManager(a="inp:int", b="inp:int"):
                    l = [2 + b for i in range(a) if i > 2]
                return a
        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), 4)
        self.assertEqual(s(), model())

    @unittest.skipUnless(_IS_ASTUNPARSE_INSTALLED, "astunparse package is required")
    def test_with_ignore_context_manager_with_just_out(self):
        class A(torch.nn.Module):
            def forward(self):
                with torch.jit._IgnoreContextManager(c="out:List[int]"):
                    c = [2 for i in range(7) if i > 2]
                c[0] = 3
                return c[0] + c[1]
        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), 5)
        self.assertEqual(s(), model())
