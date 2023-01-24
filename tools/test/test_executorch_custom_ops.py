from typing import Any, Dict

import expecttest

from torchgen.executorch.api.custom_ops import ComputeNativeFunctionStub
from torchgen.model import Location, NativeFunction

SPACES = "    "


def _get_native_function_from_yaml(yaml_obj: Dict[str, object]) -> NativeFunction:
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
        self, obj: Dict[str, Any], expected: str
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
        obj = {"func": "custom::foo(Tensor self, *, Tensor(a!)[] out) -> ()"}
        expected = f"""
void wrapper_CPU__foo_out(const at::Tensor & self, at::TensorList out) {{
{SPACES}
}}
    """
        self._test_function_schema_generates_correct_kernel(obj, expected)

    def test_schema_has_no_return_type_argument_throws(self) -> None:
        func = _get_native_function_from_yaml(
            {"func": "custom::foo.bool(Tensor self) -> bool"}
        )

        gen = ComputeNativeFunctionStub()
        with self.assertRaisesRegex(Exception, "Can't handle this return type"):
            gen(func)
