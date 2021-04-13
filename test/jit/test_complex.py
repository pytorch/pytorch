import torch
import os
import sys
from torch.testing._internal.jit_utils import JitTestCase, execWrapper
from typing import List, Dict
from itertools import product
from textwrap import dedent
import cmath  # noqa

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

class TestComplex(JitTestCase):
    def test_script(self):
        def fn(a: complex):
            return a

        self.checkScript(fn, (3 + 5j,))

    def test_complexlist(self):
        def fn(a: List[complex], idx: int):
            return a[idx]

        input = [1j, 2, 3 + 4j, -5, -7j]
        self.checkScript(fn, (input, 2))

    def test_complexdict(self):
        def fn(a: Dict[complex, complex], key: complex) -> complex:
            return a[key]

        input = {2 + 3j : 2 - 3j, -4.3 - 2j: 3j}
        self.checkScript(fn, (input, -4.3 - 2j))

    def test_pickle(self):
        class ComplexModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.a = 3 + 5j
                self.b = [2 + 3j, 3 + 4j, 0 - 3j, -4 + 0j]
                self.c = {2 + 3j : 2 - 3j, -4.3 - 2j: 3j}

            @torch.jit.script_method
            def forward(self, b: int):
                return b + 2j

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded.a, 3 + 5j)
        self.assertEqual(loaded.b, [2 + 3j, 3 + 4j, -3j, -4])
        self.assertEqual(loaded.c, {2 + 3j : 2 - 3j, -4.3 - 2j: 3j})
        self.assertEqual(loaded(2), 2 + 2j)

    def test_complex_parse(self):
        def fn(a: int, b: torch.Tensor, dim: int):
            # verifies `emitValueToTensor()` 's behavior
            b[dim] = 2.4 + 0.5j
            return (3 * 2j) + a + 5j - 7.4j - 4

        t1 = torch.tensor(1)
        t2 = torch.tensor([0.4, 1.4j, 2.35])

        self.checkScript(fn, (t1, t2, 2))

    def test_complex_constants_and_ops(self):
        vals = ([0.0, 1.0, 2.2, -1.0, -0.0, -2.2, 1, 0, 2]
                + [10.0 ** i for i in range(2)] + [-(10.0 ** i) for i in range(2)])
        complex_vals = tuple(complex(x, y) for x, y in product(vals, vals))

        funcs_template = dedent('''
            def func(a: complex):
                return cmath.{func_or_const}(a)
            ''')

        def checkCmath(func_name, funcs_template=funcs_template):
            funcs_str = funcs_template.format(func_or_const=func_name)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            f = scope['func']

            for a in complex_vals:
                res_python = None
                res_script = None
                try:
                    res_python = f(a)
                except Exception as e:
                    res_python = e
                try:
                    res_script = f_script(a)
                except Exception as e:
                    res_script = e

                if res_python != res_script:
                    if isinstance(res_python, Exception):
                        continue

                    msg = f"Failed on {func_name} with input {a}. Python: {res_python}, Script: {res_script}"
                    self.assertEqual(res_python, res_script, msg=msg)

        unary_ops = ['log', 'log10', 'sqrt', 'exp', 'sin', 'cos', 'asin', 'acos', 'atan', 'sinh', 'cosh',
                     'tanh', 'asinh', 'acosh', 'atanh', 'phase']

        # --- Unary ops ---
        for op in unary_ops:
            checkCmath(op)

        def fn(x: complex):
            return abs(x)

        for val in complex_vals:
            self.checkScript(fn, (val, ))

        func_constants_template = dedent('''
            def func():
                return cmath.{func_or_const}
            ''')
        float_consts = ['pi', 'e', 'tau', 'inf', 'nan']
        complex_consts = ['infj', 'nanj']
        for x in (float_consts + complex_consts):
            checkCmath(x, funcs_template=func_constants_template)


    def test_infj_nanj_pickle(self):
        class ComplexModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.a = 3 + 5j

            @torch.jit.script_method
            def forward(self, infj: int, nanj: int):
                if infj == 2:
                    return infj + cmath.infj
                else:
                    return nanj + cmath.nanj

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded(2, 3), 2 + cmath.infj)
        self.assertEqual(loaded(3, 4), 4 + cmath.nanj)
