import os
import sys
import inspect
import unittest
from typing import Dict, List

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")


class TestBuiltins(JitTestCase):
    """
    Tests for TorchScript support of Python builtin functions.
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

        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            torch.jit.script(Mod())

        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()

            def forward(self, name):
                # not allowed, `torch.rand` is not a class type
                return hasattr(torch.rand(2, 3), name)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            torch.jit.script(Mod())

    def test_del(self):
        def fn(x: List[int]) -> List[int]:
            a = x * 2
            del a
            return x

        self.checkScript(fn, ([1, 2, 3],))

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):
            @torch.jit.script
            def fn(x):
                a = x ** 2
                del a
                return a

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):
            @torch.jit.script
            def fn(x):
                a = x ** 2
                if a:
                    del a
                return a

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "b"):
            @torch.jit.script
            def fn(x):
                a = x ** 2
                del b
                return a

    def test_del_multiple_operands(self):
        def fn(x: List[int]) -> List[int]:
            a, b, c = x[0], x[1], x[2]
            del a, b, c
            return x

        self.checkScript(fn, ([1, 2, 3],))

        def del_list_multiple_operands(x: List[int]) -> List[int]:
            del x[0], x[1]
            return x

        py_out = del_list_multiple_operands([0, 1, 2])
        jit_out = torch.jit.script(del_list_multiple_operands)([0, 1, 2])
        self.assertEquals(py_out, jit_out)

        def del_dict_multiple_operands(x: Dict[str, int]) -> Dict[str, int]:
            del x['hi'], x['there']
            return x

        py_out = del_dict_multiple_operands({"hi": 5, "there": 6})
        jit_out = torch.jit.script(del_dict_multiple_operands)({"hi": 5, "there": 6})
        self.assertEquals(py_out, jit_out)


class TestTensorBuiltins(JitTestCase):
    def test_tensor_properties(self):
        def should_keep(tensor, name):
            if inspect.isroutine(getattr(tensor, name)):
                return False
            if name.startswith('_'):
                return False
            return True

        tensor = torch.arange(4, dtype=torch.float).view(2, 2)
        keys = dir(tensor)

        # real and imag are only implemented for complex tensors.
        self.assertRaises(RuntimeError, lambda: should_keep(tensor, 'imag'))
        keys.remove('imag')
        self.assertRaises(RuntimeError, lambda: should_keep(tensor, 'real'))
        keys.remove('real')

        properties = [p for p in keys if should_keep(tensor, p)]

        code_template = """
        def fn(x):
            return x.{}
        """

        EQUALITY_MISMATCH = set([
            # TorchScript doesn't have real enums so they return an int instead
            # of the actual value
            'dtype',
            'layout',
        ])
        MISSING_PROPERTIES = set([
            'grad_fn',
            # This is an undocumented property so it's not included
            "output_nr",
            # This has a longer implementation, maybe not worth copying to
            # TorchScript if named tensors don't work there anyways
            'names',
        ])

        for p in properties:
            if p in MISSING_PROPERTIES:
                continue
            code = code_template.format(p)
            cu = torch.jit.CompilationUnit()
            cu.define(code)
            if p in EQUALITY_MISMATCH:
                continue
            self.assertEqual(getattr(tensor, p), cu.fn(tensor))

    def test_tensor_subscript_assign(self):
        def fn1(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[torch.tensor(0)] = torch.tensor(2, dtype=torch.uint8)
            return a

        def fn2(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[0] = 2
            return a

        def fn3(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[torch.tensor(0)] = 2
            return a

        def fn4(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[0] = torch.tensor(2, dtype=torch.uint8)
            return a

        def fn5(x):
            a = torch.zeros_like(x, dtype=torch.float32)
            a[torch.tensor(0)] = 2
            return a

        for fn in (fn1, fn2, fn3, fn4, fn5):
            self.checkScript(fn, (torch.zeros(2, dtype=torch.uint8),))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_tensor_subscript_assign_device(self):
        def fn6(x):
            a = torch.zeros_like(x, dtype=torch.float32, device="cuda")
            a[torch.tensor(0)] = 2
            return a

        self.checkScript(fn6, (torch.zeros(2, dtype=torch.float32, device="cuda"),))
