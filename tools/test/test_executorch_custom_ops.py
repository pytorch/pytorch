from __future__ import annotations

import tempfile
import unittest
from typing import Any
from unittest.mock import ANY, Mock, patch

import expecttest

import torchgen
from torchgen.executorch.api.custom_ops import ComputeNativeFunctionStub
from torchgen.executorch.model import ETKernelIndex
from torchgen.gen_executorch import gen_headers
from torchgen.model import Location, NativeFunction
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import FileManager


SPACES = "    "


def _get_native_function_from_yaml(yaml_obj: dict[str, object]) -> NativeFunction:
    native_function, _ = NativeFunction.from_yaml(
        yaml_obj,
        loc=Location(__file__, 1),
        valid_tags=set(),
    )
    return native_function


class TestComputeNativeFunctionStub(expecttest.TestCase):
    """
    Could use torch.testing._internal.common_utils to reduce boilerplate.
    GH CI job doesn't build torch before running tools unit tests, hence
    manually adding these parametrized tests.
    """

    def _test_function_schema_generates_correct_kernel(
        self, obj: dict[str, Any], expected: str
    ) -> None:
        func = _get_native_function_from_yaml(obj)

        gen = ComputeNativeFunctionStub()
        res = gen(func)
        self.assertIsNotNone(res)
        self.assertExpectedInline(
            str(res),
            expected,
        )

    def test_function_schema_generates_correct_kernel_tensor_out(self) -> None:
        obj = {"func": "custom::foo.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"}
        expected = """
at::Tensor & wrapper_CPU_out_foo_out(const at::Tensor & self, at::Tensor & out) {
    return out;
}
    """
        self._test_function_schema_generates_correct_kernel(obj, expected)

    def test_function_schema_generates_correct_kernel_no_out(self) -> None:
        obj = {"func": "custom::foo.Tensor(Tensor self) -> Tensor"}
        expected = """
at::Tensor wrapper_CPU_Tensor_foo(const at::Tensor & self) {
    return self;
}
    """
        self._test_function_schema_generates_correct_kernel(obj, expected)

    def test_function_schema_generates_correct_kernel_no_return(self) -> None:
        obj = {"func": "custom::foo.out(Tensor self, *, Tensor(a!)[] out) -> ()"}
        expected = f"""
void wrapper_CPU_out_foo_out(const at::Tensor & self, at::TensorList out) {{
{SPACES}
}}
    """
        self._test_function_schema_generates_correct_kernel(obj, expected)

    def test_function_schema_generates_correct_kernel_3_returns(self) -> None:
        obj = {
            "func": "custom::foo(Tensor self, Tensor[] other) -> (Tensor, Tensor, Tensor)"
        }
        expected = """
::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper_CPU__foo(const at::Tensor & self, at::TensorList other) {
    return ::std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                at::Tensor(), at::Tensor(), at::Tensor()
            );
}
    """
        self._test_function_schema_generates_correct_kernel(obj, expected)

    def test_function_schema_generates_correct_kernel_1_return_no_out(self) -> None:
        obj = {"func": "custom::foo(Tensor[] a) -> Tensor"}
        expected = """
at::Tensor wrapper_CPU__foo(at::TensorList a) {
    return at::Tensor();
}
    """
        self._test_function_schema_generates_correct_kernel(obj, expected)

    def test_schema_has_no_return_type_argument_throws(self) -> None:
        func = _get_native_function_from_yaml(
            {"func": "custom::foo.bool(Tensor self) -> bool"}
        )

        gen = ComputeNativeFunctionStub()
        with self.assertRaisesRegex(Exception, "Can't handle this return type"):
            gen(func)


class TestGenCustomOpsHeader(unittest.TestCase):
    @patch.object(torchgen.utils.FileManager, "write_with_template")
    @patch.object(torchgen.utils.FileManager, "write")
    def test_fm_writes_custom_ops_header_when_boolean_is_true(
        self, unused: Mock, mock_method: Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            fm = FileManager(tempdir, tempdir, False)
            gen_headers(
                native_functions=[],
                gen_custom_ops_header=True,
                custom_ops_native_functions=[],
                selector=SelectiveBuilder.get_nop_selector(),
                kernel_index=ETKernelIndex(index={}),
                cpu_fm=fm,
                use_aten_lib=False,
            )
            mock_method.assert_called_once_with(
                "CustomOpsNativeFunctions.h", "NativeFunctions.h", ANY
            )

    @patch.object(torchgen.utils.FileManager, "write_with_template")
    @patch.object(torchgen.utils.FileManager, "write")
    def test_fm_doesnot_writes_custom_ops_header_when_boolean_is_false(
        self, unused: Mock, mock_method: Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            fm = FileManager(tempdir, tempdir, False)
            gen_headers(
                native_functions=[],
                gen_custom_ops_header=False,
                custom_ops_native_functions=[],
                selector=SelectiveBuilder.get_nop_selector(),
                kernel_index=ETKernelIndex(index={}),
                cpu_fm=fm,
                use_aten_lib=False,
            )
            mock_method.assert_not_called()
