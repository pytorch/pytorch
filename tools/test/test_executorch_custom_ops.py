#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest
from typing import Dict

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
    TestCase,
)

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


@instantiate_parametrized_tests
class TestComputeNativeFunctionStub(TestCase):
    @parametrize(
        "obj, expected",
        [
            subtest(
                (
                    {
                        "func": "custom::foo.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"
                    },
                    """
at::Tensor & wrapper_out_foo_out(const at::Tensor & self, at::Tensor & out) {
    return out;
}
    """,
                ),
                name="tensor_out",
            ),
            subtest(
                (
                    {"func": "custom::foo.Tensor(Tensor self) -> Tensor"},
                    """
at::Tensor wrapper_Tensor_foo(const at::Tensor & self) {
    return self;
}
    """,
                ),
                name="no_out",
            ),
            subtest(
                (
                    {"func": "custom::foo(Tensor self, *, Tensor(a!)[] out) -> ()"},
                    f"""
void wrapper__foo_out(const at::Tensor & self, at::TensorList out) {{
{SPACES}
}}
    """,
                ),
                name="no_return",
            ),
        ],
    )
    def test_function_schema_generates_correct_kernel(self, obj, expected) -> None:
        func = _get_native_function_from_yaml(obj)

        gen = ComputeNativeFunctionStub()
        res = gen(func)
        self.assertEquals(
            res,
            expected,
        )

    def test_schema_has_no_return_type_argument_throws(self) -> None:
        func = _get_native_function_from_yaml(
            {"func": "custom::foo.bool(Tensor self) -> bool"}
        )

        gen = ComputeNativeFunctionStub()
        with self.assertRaisesRegex(Exception, "Can't handle this return type"):
            gen(func)
