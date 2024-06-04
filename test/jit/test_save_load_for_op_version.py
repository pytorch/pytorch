# Owner(s): ["oncall: jit"]

import io
import os
import sys
from itertools import product as product
from typing import Union

import hypothesis.strategies as st
from hypothesis import example, given, settings

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestSaveLoadForOpVersion(JitTestCase):
    # Helper that returns the module after saving and loading
    def _save_load_module(self, m):
        scripted_module = torch.jit.script(m())
        buffer = io.BytesIO()
        torch.jit.save(scripted_module, buffer)
        buffer.seek(0)
        return torch.jit.load(buffer)

    def _save_load_mobile_module(self, m):
        scripted_module = torch.jit.script(m())
        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        return _load_for_lite_interpreter(buffer)

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

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_tensor(self, sample_input):
        def historic_div(self, other):
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide(other)
            return self.divide(other, rounding_mode="trunc")

        # Tensor x Tensor
        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                result_0 = a / b
                result_1 = torch.div(a, b)
                result_2 = a.div(b)

                return result_0, result_1, result_2

        # Loads historic module
        try:
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module = self._save_load_mobile_module(MyModule)

        for val_a, val_b in product(sample_input, sample_input):
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

            _helper(v3_mobile_module, historic_div)
            _helper(current_mobile_module, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_tensor_inplace(self, sample_input):
        def historic_div_(self, other):
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide_(other)
            return self.divide_(other, rounding_mode="trunc")

        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                a /= b
                return a

        try:
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_inplace_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module = self._save_load_mobile_module(MyModule)

        for val_a, val_b in product(sample_input, sample_input):
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

            _helper(v3_mobile_module, historic_div_)

            # Recreates a since it was modified in place
            a = torch.tensor((val_a,))
            _helper(current_mobile_module, torch.Tensor.div_)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_tensor_out(self, sample_input):
        def historic_div_out(self, other, out):
            if (
                self.is_floating_point()
                or other.is_floating_point()
                or out.is_floating_point()
            ):
                return torch.true_divide(self, other, out=out)
            return torch.divide(self, other, out=out, rounding_mode="trunc")

        class MyModule(torch.nn.Module):
            def forward(self, a, b, out):
                return a.div(b, out=out)

        try:
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_out_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module = self._save_load_mobile_module(MyModule)

        for val_a, val_b in product(sample_input, sample_input):
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

                _helper(v3_mobile_module, historic_div_out)
                _helper(current_mobile_module, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_scalar(self, sample_input):
        def historic_div_scalar_float(self, other: float):
            return torch.true_divide(self, other)

        def historic_div_scalar_int(self, other: int):
            if self.is_floating_point():
                return torch.true_divide(self, other)
            return torch.divide(self, other, rounding_mode="trunc")

        class MyModuleFloat(torch.nn.Module):
            def forward(self, a, b: float):
                return a / b

        class MyModuleInt(torch.nn.Module):
            def forward(self, a, b: int):
                return a / b

        try:
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/jit/fixtures/test_versioned_div_scalar_float_v2.ptl"
            )
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_int_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module_float = self._save_load_mobile_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_mobile_module(MyModuleInt)

        for val_a, val_b in product(sample_input, sample_input):
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
                _helper(v3_mobile_module_float, current_mobile_module_float)
                _helper(current_mobile_module_float, torch.div)
            else:
                _helper(v3_mobile_module_int, historic_div_scalar_int)
                _helper(current_mobile_module_int, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_scalar_reciprocal(self, sample_input):
        def historic_div_scalar_float_reciprocal(self, other: float):
            return other / self

        def historic_div_scalar_int_reciprocal(self, other: int):
            if self.is_floating_point():
                return other / self
            return torch.divide(other, self, rounding_mode="trunc")

        class MyModuleFloat(torch.nn.Module):
            def forward(self, a, b: float):
                return b / a

        class MyModuleInt(torch.nn.Module):
            def forward(self, a, b: int):
                return b / a

        try:
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_reciprocal_float_v2.ptl"
            )
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_reciprocal_int_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module_float = self._save_load_mobile_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_mobile_module(MyModuleInt)

        for val_a, val_b in product(sample_input, sample_input):
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
                _helper(v3_mobile_module_float, current_mobile_module_float)
                _helper(current_mobile_module_float, torch.div)
            else:
                _helper(v3_mobile_module_int, current_mobile_module_int)
                _helper(current_mobile_module_int, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_scalar_inplace(self, sample_input):
        def historic_div_scalar_float_inplace(self, other: float):
            return self.true_divide_(other)

        def historic_div_scalar_int_inplace(self, other: int):
            if self.is_floating_point():
                return self.true_divide_(other)

            return self.divide_(other, rounding_mode="trunc")

        class MyModuleFloat(torch.nn.Module):
            def forward(self, a, b: float):
                a /= b
                return a

        class MyModuleInt(torch.nn.Module):
            def forward(self, a, b: int):
                a /= b
                return a

        try:
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_inplace_float_v2.ptl"
            )
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_inplace_int_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module_float = self._save_load_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_module(MyModuleInt)

        for val_a, val_b in product(sample_input, sample_input):
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
                _helper(current_mobile_module_float, torch.Tensor.div_)
            else:
                _helper(current_mobile_module_int, torch.Tensor.div_)

    # NOTE: Scalar division was already true division in op version 3,
    #   so this test verifies the behavior is unchanged.
    def test_versioned_div_scalar_scalar(self):
        class MyModule(torch.nn.Module):
            def forward(self, a: float, b: int, c: float, d: int):
                result_0 = a / b
                result_1 = a / c
                result_2 = b / c
                result_3 = b / d
                return (result_0, result_1, result_2, result_3)

        try:
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_scalar_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module = self._save_load_mobile_module(MyModule)

        def _helper(m, fn):
            vals = (5.0, 3, 2.0, 7)
            m_result = m(*vals)
            fn_result = fn(*vals)
            for mr, hr in zip(m_result, fn_result):
                self.assertEqual(mr, hr)

        _helper(v3_mobile_module, current_mobile_module)

    def test_versioned_linspace(self):
        class Module(torch.nn.Module):
            def forward(
                self, a: Union[int, float, complex], b: Union[int, float, complex]
            ):
                c = torch.linspace(a, b, steps=5)
                d = torch.linspace(a, b, steps=100)
                return c, d

        scripted_module = torch.jit.load(
            pytorch_test_dir + "/jit/fixtures/test_versioned_linspace_v7.ptl"
        )

        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v7_mobile_module = _load_for_lite_interpreter(buffer)

        current_mobile_module = self._save_load_mobile_module(Module)

        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        for a, b in sample_inputs:
            (output_with_step, output_without_step) = v7_mobile_module(a, b)
            (current_with_step, current_without_step) = current_mobile_module(a, b)
            # when no step is given, should have used 100
            self.assertTrue(output_without_step.size(dim=0) == 100)
            self.assertTrue(output_with_step.size(dim=0) == 5)
            # outputs should be equal to the newest version
            self.assertEqual(output_with_step, current_with_step)
            self.assertEqual(output_without_step, current_without_step)

    def test_versioned_linspace_out(self):
        class Module(torch.nn.Module):
            def forward(
                self,
                a: Union[int, float, complex],
                b: Union[int, float, complex],
                out: torch.Tensor,
            ):
                return torch.linspace(a, b, steps=100, out=out)

        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_linspace_out_v7.ptl"
        )
        loaded_model = torch.jit.load(model_path)
        buffer = io.BytesIO(loaded_model._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v7_mobile_module = _load_for_lite_interpreter(buffer)
        current_mobile_module = self._save_load_mobile_module(Module)

        sample_inputs = (
            (
                3,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                -10,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                4.0,
                6.0,
                torch.empty((100,), dtype=torch.float64),
                torch.empty((100,), dtype=torch.float64),
            ),
            (
                3 + 4j,
                4 + 5j,
                torch.empty((100,), dtype=torch.complex64),
                torch.empty((100,), dtype=torch.complex64),
            ),
        )
        for start, end, out_for_old, out_for_new in sample_inputs:
            output = v7_mobile_module(start, end, out_for_old)
            output_current = current_mobile_module(start, end, out_for_new)
            # when no step is given, should have used 100
            self.assertTrue(output.size(dim=0) == 100)
            # "Upgraded" model should match the new version output
            self.assertEqual(output, output_current)

    def test_versioned_logspace(self):
        class Module(torch.nn.Module):
            def forward(
                self, a: Union[int, float, complex], b: Union[int, float, complex]
            ):
                c = torch.logspace(a, b, steps=5)
                d = torch.logspace(a, b, steps=100)
                return c, d

        scripted_module = torch.jit.load(
            pytorch_test_dir + "/jit/fixtures/test_versioned_logspace_v8.ptl"
        )

        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v8_mobile_module = _load_for_lite_interpreter(buffer)

        current_mobile_module = self._save_load_mobile_module(Module)

        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        for a, b in sample_inputs:
            (output_with_step, output_without_step) = v8_mobile_module(a, b)
            (current_with_step, current_without_step) = current_mobile_module(a, b)
            # when no step is given, should have used 100
            self.assertTrue(output_without_step.size(dim=0) == 100)
            self.assertTrue(output_with_step.size(dim=0) == 5)
            # outputs should be equal to the newest version
            self.assertEqual(output_with_step, current_with_step)
            self.assertEqual(output_without_step, current_without_step)

    def test_versioned_logspace_out(self):
        class Module(torch.nn.Module):
            def forward(
                self,
                a: Union[int, float, complex],
                b: Union[int, float, complex],
                out: torch.Tensor,
            ):
                return torch.logspace(a, b, steps=100, out=out)

        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_logspace_out_v8.ptl"
        )
        loaded_model = torch.jit.load(model_path)
        buffer = io.BytesIO(loaded_model._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v8_mobile_module = _load_for_lite_interpreter(buffer)
        current_mobile_module = self._save_load_mobile_module(Module)

        sample_inputs = (
            (
                3,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                -10,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                4.0,
                6.0,
                torch.empty((100,), dtype=torch.float64),
                torch.empty((100,), dtype=torch.float64),
            ),
            (
                3 + 4j,
                4 + 5j,
                torch.empty((100,), dtype=torch.complex64),
                torch.empty((100,), dtype=torch.complex64),
            ),
        )
        for start, end, out_for_old, out_for_new in sample_inputs:
            output = v8_mobile_module(start, end, out_for_old)
            output_current = current_mobile_module(start, end, out_for_new)
            # when no step is given, should have used 100
            self.assertTrue(output.size(dim=0) == 100)
            # "Upgraded" model should match the new version output
            self.assertEqual(output, output_current)
