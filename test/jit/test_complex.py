# Owner(s): ["oncall: jit"]

import cmath
import os
import sys
from itertools import product
from textwrap import dedent
from typing import Dict, List

import torch
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.jit_utils import execWrapper, JitTestCase

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

        input = {2 + 3j: 2 - 3j, -4.3 - 2j: 3j}
        self.checkScript(fn, (input, -4.3 - 2j))

    def test_pickle(self):
        class ComplexModule(torch.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = 3 + 5j
                self.b = [2 + 3j, 3 + 4j, 0 - 3j, -4 + 0j]
                self.c = {2 + 3j: 2 - 3j, -4.3 - 2j: 3j}

            @torch.jit.script_method
            def forward(self, b: int):
                return b + 2j

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded.a, 3 + 5j)
        self.assertEqual(loaded.b, [2 + 3j, 3 + 4j, -3j, -4])
        self.assertEqual(loaded.c, {2 + 3j: 2 - 3j, -4.3 - 2j: 3j})
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
        vals = (
            [0.0, 1.0, 2.2, -1.0, -0.0, -2.2, 1, 0, 2]
            + [10.0**i for i in range(2)]
            + [-(10.0**i) for i in range(2)]
        )
        complex_vals = tuple(complex(x, y) for x, y in product(vals, vals))

        funcs_template = dedent(
            """
            def func(a: complex):
                return cmath.{func_or_const}(a)
            """
        )

        def checkCmath(func_name, funcs_template=funcs_template):
            funcs_str = funcs_template.format(func_or_const=func_name)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            f = scope["func"]

            if func_name in ["isinf", "isnan", "isfinite"]:
                new_vals = vals + ([float("inf"), float("nan"), -1 * float("inf")])
                final_vals = tuple(
                    complex(x, y) for x, y in product(new_vals, new_vals)
                )
            else:
                final_vals = complex_vals

            for a in final_vals:
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

        unary_ops = [
            "log",
            "log10",
            "sqrt",
            "exp",
            "sin",
            "cos",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "phase",
            "isinf",
            "isnan",
            "isfinite",
        ]

        # --- Unary ops ---
        for op in unary_ops:
            checkCmath(op)

        def fn(x: complex):
            return abs(x)

        for val in complex_vals:
            self.checkScript(fn, (val,))

        def pow_complex_float(x: complex, y: float):
            return pow(x, y)

        def pow_float_complex(x: float, y: complex):
            return pow(x, y)

        self.checkScript(pow_float_complex, (2, 3j))
        self.checkScript(pow_complex_float, (3j, 2))

        def pow_complex_complex(x: complex, y: complex):
            return pow(x, y)

        for x, y in zip(complex_vals, complex_vals):
            # Reference: https://github.com/pytorch/pytorch/issues/54622
            if x == 0:
                continue
            self.checkScript(pow_complex_complex, (x, y))

        if not IS_MACOS:
            # --- Binary op ---
            def rect_fn(x: float, y: float):
                return cmath.rect(x, y)

            for x, y in product(vals, vals):
                self.checkScript(
                    rect_fn,
                    (
                        x,
                        y,
                    ),
                )

        func_constants_template = dedent(
            """
            def func():
                return cmath.{func_or_const}
            """
        )
        float_consts = ["pi", "e", "tau", "inf", "nan"]
        complex_consts = ["infj", "nanj"]
        for x in float_consts + complex_consts:
            checkCmath(x, funcs_template=func_constants_template)

    def test_infj_nanj_pickle(self):
        class ComplexModule(torch.jit.ScriptModule):
            def __init__(self) -> None:
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

    def test_complex_constructor(self):
        # Test all scalar types
        def fn_int(real: int, img: int):
            return complex(real, img)

        self.checkScript(
            fn_int,
            (
                0,
                0,
            ),
        )
        self.checkScript(
            fn_int,
            (
                -1234,
                0,
            ),
        )
        self.checkScript(
            fn_int,
            (
                0,
                -1256,
            ),
        )
        self.checkScript(
            fn_int,
            (
                -167,
                -1256,
            ),
        )

        def fn_float(real: float, img: float):
            return complex(real, img)

        self.checkScript(
            fn_float,
            (
                0.0,
                0.0,
            ),
        )
        self.checkScript(
            fn_float,
            (
                -1234.78,
                0,
            ),
        )
        self.checkScript(
            fn_float,
            (
                0,
                56.18,
            ),
        )
        self.checkScript(
            fn_float,
            (
                -1.9,
                -19.8,
            ),
        )

        def fn_bool(real: bool, img: bool):
            return complex(real, img)

        self.checkScript(
            fn_bool,
            (
                True,
                True,
            ),
        )
        self.checkScript(
            fn_bool,
            (
                False,
                False,
            ),
        )
        self.checkScript(
            fn_bool,
            (
                False,
                True,
            ),
        )
        self.checkScript(
            fn_bool,
            (
                True,
                False,
            ),
        )

        def fn_bool_int(real: bool, img: int):
            return complex(real, img)

        self.checkScript(
            fn_bool_int,
            (
                True,
                0,
            ),
        )
        self.checkScript(
            fn_bool_int,
            (
                False,
                0,
            ),
        )
        self.checkScript(
            fn_bool_int,
            (
                False,
                -1,
            ),
        )
        self.checkScript(
            fn_bool_int,
            (
                True,
                3,
            ),
        )

        def fn_int_bool(real: int, img: bool):
            return complex(real, img)

        self.checkScript(
            fn_int_bool,
            (
                0,
                True,
            ),
        )
        self.checkScript(
            fn_int_bool,
            (
                0,
                False,
            ),
        )
        self.checkScript(
            fn_int_bool,
            (
                -3,
                True,
            ),
        )
        self.checkScript(
            fn_int_bool,
            (
                6,
                False,
            ),
        )

        def fn_bool_float(real: bool, img: float):
            return complex(real, img)

        self.checkScript(
            fn_bool_float,
            (
                True,
                0.0,
            ),
        )
        self.checkScript(
            fn_bool_float,
            (
                False,
                0.0,
            ),
        )
        self.checkScript(
            fn_bool_float,
            (
                False,
                -1.0,
            ),
        )
        self.checkScript(
            fn_bool_float,
            (
                True,
                3.0,
            ),
        )

        def fn_float_bool(real: float, img: bool):
            return complex(real, img)

        self.checkScript(
            fn_float_bool,
            (
                0.0,
                True,
            ),
        )
        self.checkScript(
            fn_float_bool,
            (
                0.0,
                False,
            ),
        )
        self.checkScript(
            fn_float_bool,
            (
                -3.0,
                True,
            ),
        )
        self.checkScript(
            fn_float_bool,
            (
                6.0,
                False,
            ),
        )

        def fn_float_int(real: float, img: int):
            return complex(real, img)

        self.checkScript(
            fn_float_int,
            (
                0.0,
                1,
            ),
        )
        self.checkScript(
            fn_float_int,
            (
                0.0,
                -1,
            ),
        )
        self.checkScript(
            fn_float_int,
            (
                1.8,
                -3,
            ),
        )
        self.checkScript(
            fn_float_int,
            (
                2.7,
                8,
            ),
        )

        def fn_int_float(real: int, img: float):
            return complex(real, img)

        self.checkScript(
            fn_int_float,
            (
                1,
                0.0,
            ),
        )
        self.checkScript(
            fn_int_float,
            (
                -1,
                1.7,
            ),
        )
        self.checkScript(
            fn_int_float,
            (
                -3,
                0.0,
            ),
        )
        self.checkScript(
            fn_int_float,
            (
                2,
                -8.9,
            ),
        )

    def test_torch_complex_constructor_with_tensor(self):
        tensors = [torch.rand(1), torch.randint(-5, 5, (1,)), torch.tensor([False])]

        def fn_tensor_float(real, img: float):
            return complex(real, img)

        def fn_tensor_int(real, img: int):
            return complex(real, img)

        def fn_tensor_bool(real, img: bool):
            return complex(real, img)

        def fn_float_tensor(real: float, img):
            return complex(real, img)

        def fn_int_tensor(real: int, img):
            return complex(real, img)

        def fn_bool_tensor(real: bool, img):
            return complex(real, img)

        for tensor in tensors:
            self.checkScript(fn_tensor_float, (tensor, 1.2))
            self.checkScript(fn_tensor_int, (tensor, 3))
            self.checkScript(fn_tensor_bool, (tensor, True))

            self.checkScript(fn_float_tensor, (1.2, tensor))
            self.checkScript(fn_int_tensor, (3, tensor))
            self.checkScript(fn_bool_tensor, (True, tensor))

        def fn_tensor_tensor(real, img):
            return complex(real, img) + complex(2)

        for x, y in product(tensors, tensors):
            self.checkScript(
                fn_tensor_tensor,
                (
                    x,
                    y,
                ),
            )

    def test_comparison_ops(self):
        def fn1(a: complex, b: complex):
            return a == b

        def fn2(a: complex, b: complex):
            return a != b

        def fn3(a: complex, b: float):
            return a == b

        def fn4(a: complex, b: float):
            return a != b

        x, y = 2 - 3j, 4j
        self.checkScript(fn1, (x, x))
        self.checkScript(fn1, (x, y))
        self.checkScript(fn2, (x, x))
        self.checkScript(fn2, (x, y))

        x1, y1 = 1 + 0j, 1.0
        self.checkScript(fn3, (x1, y1))
        self.checkScript(fn4, (x1, y1))

    def test_div(self):
        def fn1(a: complex, b: complex):
            return a / b

        x, y = 2 - 3j, 4j
        self.checkScript(fn1, (x, y))

    def test_complex_list_sum(self):
        def fn(x: List[complex]):
            return sum(x)

        self.checkScript(fn, (torch.randn(4, dtype=torch.cdouble).tolist(),))

    def test_tensor_attributes(self):
        def tensor_real(x):
            return x.real

        def tensor_imag(x):
            return x.imag

        t = torch.randn(2, 3, dtype=torch.cdouble)
        self.checkScript(tensor_real, (t,))
        self.checkScript(tensor_imag, (t,))

    def test_binary_op_complex_tensor(self):
        def mul(x: complex, y: torch.Tensor):
            return x * y

        def add(x: complex, y: torch.Tensor):
            return x + y

        def eq(x: complex, y: torch.Tensor):
            return x == y

        def ne(x: complex, y: torch.Tensor):
            return x != y

        def sub(x: complex, y: torch.Tensor):
            return x - y

        def div(x: complex, y: torch.Tensor):
            return x - y

        ops = [mul, add, eq, ne, sub, div]

        for shape in [(1,), (2, 2)]:
            x = 0.71 + 0.71j
            y = torch.randn(shape, dtype=torch.cfloat)
            for op in ops:
                eager_result = op(x, y)
                scripted = torch.jit.script(op)
                jit_result = scripted(x, y)
                self.assertEqual(eager_result, jit_result)
