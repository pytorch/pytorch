# Owner(s): ["oncall: jit"]

from itertools import product as product
import io
import os
import random
import sys
import unittest

import torch
# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# Legacy test cases for dynamic operator versioning
class TestLegacyUpgraders(JitTestCase):
    reason: str = "Skipped due to new global operator versioning for operators"

    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_symbols(self):
        """
        Tests Torchscript symbol versioning. See note [Versioned Symbols].
        This test uses an undocumented, test-only function
        torch._test_serialization_subcmul.
        This function is implemented as (a - alpha * b) with a default value
        of 1 for alpha. In file format version 2, however, it was implemented
        as (b - alpha * a) with a default value of 2 for alpha.
        This test verifies a module seralized with file format version 2
        exhibits the old behavior, and that the same module newly serialized
        exhibits the current behavior.
        """
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, a, b, alpha: float):
                no_alpha = torch._test_serialization_subcmul(a, b)
                with_alpha = torch._test_serialization_subcmul(a, b, alpha)
                return no_alpha, with_alpha

        def historic_subcmul(a, b, alpha=2):
            return b - alpha * a

        def current_subcmul(a, b, alpha=1):
            return a - alpha * b

        # Loads and verifies the historic behavior of the module
        # that was serialized with version 2
        module_v2 = torch.jit.load(pytorch_test_dir + "/jit/fixtures/_test_serialization_subcmul_v2.pt")
        a = torch.randn((5,))
        b = torch.randn((5,))
        alpha = random.random()
        args = (a, b, alpha)
        no_alpha_v2, with_alpha_v2 = module_v2(*args)
        self.assertEqual(no_alpha_v2, historic_subcmul(a, b))
        self.assertEqual(with_alpha_v2, historic_subcmul(*args))

        # Scripts, saves, loads and verifies the current behavior of the module
        scripted_module = torch.jit.script(MyModule())
        buffer = io.BytesIO()
        torch.jit.save(scripted_module, buffer)
        buffer.seek(0)
        module_current = torch.jit.load(buffer)
        no_alpha_current, with_alpha_current = module_current(*args)
        self.assertEqual(no_alpha_current, current_subcmul(a, b))
        self.assertEqual(with_alpha_current, current_subcmul(*args))

    # Helper that returns the module after saving and loading
    def _save_load_module(self, m):
        scripted_module = torch.jit.script(m())
        buffer = io.BytesIO()
        torch.jit.save(scripted_module, buffer)
        buffer.seek(0)
        return torch.jit.load(buffer)

    # Helper which returns the result of a function or the exception the
    #   function threw.
    def _try_fn(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e

    def _verify_no(self, kind, m):
        self._verify_count(kind, m, 0)

    def _verify_count(self, kind, m, count):
        node_count = sum(str(n).count(kind) for n in m.graph.nodes())
        self.assertEqual(node_count, count)

    """
    Tests that verify Torchscript remaps aten::div(_) from versions 0-3
    to call either aten::true_divide(_), if an input is a float type,
    or truncated aten::divide(_) otherwise.
    NOTE: currently compares against current div behavior, too, since
      div behavior has not yet been updated.
    """

    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_div_tensor(self):
        def historic_div(self, other):
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide(other)
            return self.divide(other, rounding_mode='trunc')

        # Tensor x Tensor
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, a, b):
                result_0 = a / b
                result_1 = torch.div(a, b)
                result_2 = a.div(b)

                return result_0, result_1, result_2

        # Loads historic module
        try:
            v3_module = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_tensor_v3.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        self._verify_count("aten::div", v3_module, 6)  # true_divide and divide alias to div
        self._verify_count('prim::Constant[value="trunc"]', v3_module, 1)  # rounding_mode argument

        current_module = self._save_load_module(MyModule)
        self._verify_count("aten::div", current_module, 3)

        vals = (2., 3., 2, 3)
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            def _helper(m, fn):
                m_results = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                if isinstance(m_results, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                else:
                    for result in m_results:
                        self.assertEqual(result, fn_result)

            _helper(v3_module, historic_div)
            _helper(current_module, torch.div)

    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_div_tensor_inplace(self):
        def historic_div_(self, other):
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide_(other)
            return self.divide_(other, rounding_mode='trunc')

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, a, b):
                a /= b
                return a

        try:
            v3_module = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_tensor_inplace_v3.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        self._verify_count("aten::div", v3_module, 2)  # true_divide and divide both alias to div
        self._verify_count('prim::Constant[value="trunc"]', v3_module, 1)  # rounding_mode argument

        current_module = self._save_load_module(MyModule)
        self._verify_count("aten::div", current_module, 1)

        vals = (2., 3., 2, 3)
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            def _helper(m, fn):
                fn_result = self._try_fn(fn, a.clone(), b)
                m_result = self._try_fn(m, a, b)

                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)
                    self.assertEqual(m_result, a)

            _helper(v3_module, historic_div_)

            # Recreates a since it was modified in place
            a = torch.tensor((val_a,))
            _helper(current_module, torch.Tensor.div_)

    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_div_tensor_out(self):
        def historic_div_out(self, other, out):
            if self.is_floating_point() or other.is_floating_point() or out.is_floating_point():
                return torch.true_divide(self, other, out=out)
            return torch.divide(self, other, out=out, rounding_mode='trunc')

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, a, b, out):
                return a.div(b, out=out)

        try:
            v3_module = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_tensor_out_v3.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        self._verify_count("aten::div", v3_module, 2)  # true_divide and divide alias to div
        self._verify_count('prim::Constant[value="trunc"]', v3_module, 1)  # rounding_mode argument

        current_module = self._save_load_module(MyModule)
        self._verify_count("aten::div", current_module, 1)

        vals = (2., 3., 2, 3)
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            for out in (torch.empty((1,)), torch.empty((1,), dtype=torch.long)):
                def _helper(m, fn):
                    fn_result = None
                    if fn is torch.div:
                        fn_result = self._try_fn(fn, a, b, out=out.clone())
                    else:
                        fn_result = self._try_fn(fn, a, b, out.clone())
                    m_result = self._try_fn(m, a, b, out)

                    if isinstance(m_result, Exception):
                        self.assertTrue(fn_result, Exception)
                    else:
                        self.assertEqual(m_result, fn_result)
                        self.assertEqual(m_result, out)

                _helper(v3_module, historic_div_out)
                _helper(current_module, torch.div)

    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_div_scalar(self):
        def historic_div_scalar_float(self, other: float):
            return torch.true_divide(self, other)

        def historic_div_scalar_int(self, other: int):
            if self.is_floating_point():
                return torch.true_divide(self, other)
            return torch.divide(self, other, rounding_mode='trunc')

        class MyModuleFloat(torch.nn.Module):
            def __init__(self):
                super(MyModuleFloat, self).__init__()

            def forward(self, a, b: float):
                return a / b

        class MyModuleInt(torch.nn.Module):
            def __init__(self):
                super(MyModuleInt, self).__init__()

            def forward(self, a, b: int):
                return a / b

        try:
            v3_module_float = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_scalar_float_v3.pt")
            v3_module_int = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_scalar_int_v3.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        for m in (v3_module_float, v3_module_int):
            self._verify_count("aten::div", m, 2)  # true_divide and divide alias to div
            self._verify_count('prim::Constant[value="trunc"]', m, 1)  # rounding_mode argument

        current_module_float = self._save_load_module(MyModuleFloat)
        current_module_int = self._save_load_module(MyModuleInt)

        for m in (current_module_float, current_module_int):
            self._verify_count("aten::div", m, 1)

        vals = (2., 3., 2, 3)
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                m_result = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)

            if isinstance(b, float):
                _helper(v3_module_float, historic_div_scalar_float)
                _helper(current_module_float, torch.div)
            else:
                _helper(v3_module_int, historic_div_scalar_int)
                _helper(current_module_int, torch.div)

    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_div_scalar_reciprocal(self):
        def historic_div_scalar_float_reciprocal(self, other: float):
            return other / self

        def historic_div_scalar_int_reciprocal(self, other: int):
            if self.is_floating_point():
                return other / self
            return torch.divide(other, self, rounding_mode='trunc')

        class MyModuleFloat(torch.nn.Module):
            def __init__(self):
                super(MyModuleFloat, self).__init__()

            def forward(self, a, b: float):
                return b / a

        class MyModuleInt(torch.nn.Module):
            def __init__(self):
                super(MyModuleInt, self).__init__()

            def forward(self, a, b: int):
                return b / a

        try:
            v3_module_float = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_scalar_reciprocal_float_v3.pt")
            v3_module_int = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_scalar_reciprocal_int_v3.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        # NOTE: number / tensor is rewritten to torch.reciprocal(a) * b
        #  so true_divide and floor_divide do not appear in their graphs
        for m in (v3_module_float, v3_module_int):
            self._verify_no("aten::div", m)
            self._verify_no("aten::true_divide", m)
            self._verify_no("aten::floor_divide", m)
            self._verify_count("aten::reciprocal", m, 1)

        current_module_float = self._save_load_module(MyModuleFloat)
        current_module_int = self._save_load_module(MyModuleInt)

        vals = (2., 3., 2, 3)
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                m_result = self._try_fn(m, a, b)
                fn_result = None
                # Reverses argument order for torch.div
                if fn is torch.div:
                    fn_result = self._try_fn(torch.div, b, a)
                else:
                    fn_result = self._try_fn(fn, a, b)

                if isinstance(m_result, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                elif fn is torch.div or a.is_floating_point():
                    self.assertEqual(m_result, fn_result)
                else:
                    # Skip when fn is not torch.div and a is integral because
                    # historic_div_scalar_int performs floored division
                    pass

            if isinstance(b, float):
                _helper(v3_module_float, historic_div_scalar_float_reciprocal)
                _helper(current_module_float, torch.div)
            else:
                _helper(v3_module_int, historic_div_scalar_int_reciprocal)
                _helper(current_module_int, torch.div)

    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_div_scalar_inplace(self):
        def historic_div_scalar_float_inplace(self, other: float):
            return self.true_divide_(other)

        def historic_div_scalar_int_inplace(self, other: int):
            if self.is_floating_point():
                return self.true_divide_(other)

            return self.divide_(other, rounding_mode='trunc')

        class MyModuleFloat(torch.nn.Module):
            def __init__(self):
                super(MyModuleFloat, self).__init__()

            def forward(self, a, b: float):
                a /= b
                return a

        class MyModuleInt(torch.nn.Module):
            def __init__(self):
                super(MyModuleInt, self).__init__()

            def forward(self, a, b: int):
                a /= b
                return a

        try:
            v3_module_float = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_scalar_inplace_float_v3.pt")
            v3_module_int = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_scalar_inplace_int_v3.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        for m in (v3_module_float, v3_module_int):
            self._verify_count("aten::div_", m, 2)  # true_divide and divide alias to div
            self._verify_count('prim::Constant[value="trunc"]', m, 1)  # rounding_mode argument

        current_module_float = self._save_load_module(MyModuleFloat)
        current_module_int = self._save_load_module(MyModuleInt)

        for m in (current_module_float, current_module_int):
            self._verify_count("aten::div", m, 1)

        for m in (current_module_float, current_module_int):
            self._verify_count("aten::div", m, 1)

        vals = (2., 3., 2, 3)
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                m_result = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)

            if isinstance(b, float):
                _helper(v3_module_float, historic_div_scalar_float_inplace)
                _helper(current_module_float, torch.Tensor.div_)
            else:
                _helper(v3_module_int, historic_div_scalar_int_inplace)
                _helper(current_module_int, torch.Tensor.div_)

    # NOTE: Scalar division was already true division in op version 3,
    #   so this test verifies the behavior is unchanged.
    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_div_scalar_scalar(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, a: float, b: int, c: float, d: int):
                result_0 = a / b
                result_1 = a / c
                result_2 = b / c
                result_3 = b / d
                return (result_0, result_1, result_2, result_3)

        try:
            v3_module = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_div_scalar_scalar_v3.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        self._verify_count("aten::div", v3_module, 4)

        current_module = self._save_load_module(MyModule)
        self._verify_count("aten::div", current_module, 4)

        def _helper(m, fn):
            vals = (5., 3, 2., 7)
            m_result = m(*vals)
            fn_result = fn(*vals)
            for mr, hr in zip(m_result, fn_result):
                self.assertEqual(mr, hr)

        _helper(v3_module, current_module)

    # NOTE: the JIT was incapable of handling boolean fill values when
    #   PyTorch produced file format versions 0-4
    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_full_integer_value(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, int_fill: int):
                size = torch.Size(2, 2)
                a = torch.full(size, int_fill)
                b = torch.full(size, 1)
                return (a, b)

        try:
            v4_module = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_full_integer_value_v4.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        self._verify_count("aten::full", v4_module, 2)

        current_module = self._save_load_module(MyModule)
        self._verify_count("aten::full", current_module, 2)

        # Verifies historic integer type inference is float
        # NOTE: only verifies floating point, not exact dtype, due to
        #   https://github.com/pytorch/pytorch/issues/40470
        results = v4_module(2)
        for result in results:
            self.assertTrue(result.is_floating_point())

        # Verifies values are correct
        a, b = results
        self.assertTrue((a == 2.).all())
        self.assertTrue((b == 1.).all())

    # Tests that torch.full behavior which is the same from prior versions
    #   to version 5 is preserved.
    # NOTE: while torch.full in eager PyTorch accepts a requires_grad argument,
    #   it does not in Torchscript (see https://github.com/pytorch/pytorch/issues/40363)
    @unittest.skipIf(torch._C._is_upgraders_enabled(), reason)
    def test_versioned_full_preserved(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, float_fill: float):
                size = (2, 2)
                a = torch.full(size, 1.)
                b = torch.full(size, float_fill)
                c = torch.full(size, float_fill, dtype=torch.long)

                out = torch.empty(size, dtype=torch.long)
                d = torch.full(size, float_fill, out=out)

                e = torch.full(size, float_fill, dtype=torch.float16, pin_memory=None,
                               layout=torch.strided, device='cpu')
                return (a, b, c, d, e)

        try:
            v4_module = torch.jit.load(pytorch_test_dir + "/jit/fixtures/test_versioned_full_preserved_v4.pt")
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        self._verify_count("aten::full", v4_module, 5)

        current_module = self._save_load_module(MyModule)
        self._verify_count("aten::full", current_module, 5)

        self.assertEqual(v4_module(2.), current_module(2.))
