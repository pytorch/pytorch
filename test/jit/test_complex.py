import torch
import os
import sys
from torch.testing._internal.jit_utils import JitTestCase, execWrapper
from typing import List, Dict
from itertools import product
from textwrap import dedent
import cmath  # noqa
from torch.testing._internal.common_utils import (IS_WINDOWS, IS_MACOS)

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

            def forward(self, b: int):
                return b + 2j

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded.a, 3 + 5j)
        self.assertEqual(loaded.b, [2 + 3j, 3 + 4j, -3j, -4])
        self.assertEqual(loaded.c, {2 + 3j : 2 - 3j, -4.3 - 2j: 3j})

    def test_complex_parse(self):
        def fn(a: int, b: torch.Tensor, dim: int):
            # verifies `emitValueToTensor()` 's behavior
            b[dim] = 2.4 + 0.5j
            return (3 * 2j) + a + 5j - 7.4j - 4

        t1 = torch.tensor(1)
        t2 = torch.tensor([0.4, 1.4j, 2.35])

        self.checkScript(fn, (t1, t2, 2))

    def test_complex_math_ops(self):
        inf = float("inf")
        nan = float("nan")
        vals = ([inf, nan, 0.0, 1.0, 2.2, -1.0, -0.0, -2.2, -inf, 1, 0, 2]
                + [10.0 ** i for i in range(5)] + [-(10.0 ** i) for i in range(5)])
        complex_vals = tuple(complex(x, y) for x, y in product(vals, vals))

        def checkMath(func_name, vals=None):
            funcs_template = dedent('''
            def func(a):
                return cmath.{func}(a)
            ''')

            funcs_str = funcs_template.format(func=func_name)
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

                    if (a == complex(inf, nan) and IS_WINDOWS) or (a == complex(-inf, inf) and IS_MACOS):
                        continue

                    msg = ("Failed on {func_name} with input {a}. Python: {res_python}, Script: {res_script}"
                           .format(func_name=func_name, a=a, res_python=res_python, res_script=res_script))
                    self.assertEqual(res_python, res_script, msg=msg)

        unary_ops = ['log', 'log10', 'sqrt', 'exp', 'sin', 'cos', 'asin', 'acos', 'atan', 'sinh', 'cosh',
                     'tanh', 'asinh', 'acosh', 'atanh']

        # --- Unary ops with complex valued output ---
        for op in unary_ops:
            checkMath(op)

        # --- Unary ops with floating point output --- (cmath.phase(), abs())
        checkMath('phase')

        def fn(x: complex):
            return abs(x)

        for val in complex_vals:
            self.checkScript(fn, (val, ))

    def test_cmath_constants(self):
        def checkCmathConst(const_name, ret_type="float"):
            funcs_template = dedent('''
                def func():
                    # type: () -> {ret_type}
                    return cmath.{const}
                ''')

            funcs_str = funcs_template.format(const=const_name, ret_type=ret_type)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            f = scope['func']

            res_python = f()
            res_script = f_script()

            if res_python != res_script:
                msg = ("Failed on {const_name}. Python: {res_python}, Script: {res_script}"
                    .format(const_name=const_name, res_python=res_python, res_script=res_script))
                self.assertEqual(res_python, res_script, msg=msg)

        float_consts = ['pi', 'e', 'tau', 'inf', 'nan']
        complex_consts = ['infj', 'nanj']
        for x in float_consts:
            checkCmathConst(x)
        for x in complex_consts:
            checkCmathConst(x, "complex")
