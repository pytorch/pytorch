"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_eager_mode_functionalization)
"""

# Owner(s): ["module: dynamo"]

from unittest.mock import patch
from typing import List

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd


class TestFuntionalAssertions(TestCase):
    def test_functional_assert_async_msg(self) -> None:
        dep_token = torch.ops.aten._make_dep_token()
        self.assertEqual(
            torch.ops.aten._functional_assert_async.msg(
                torch.tensor(1), "test msg", dep_token
            ),
            dep_token,
        )
        with self.assertRaisesRegex(RuntimeError, "test msg"):
            torch.ops.aten._functional_assert_async.msg(
                torch.tensor(0), "test msg", dep_token
            ),

    def test_functional_sym_constrain_range(self) -> None:
        dep_token = torch.ops.aten._make_dep_token()
        self.assertEqual(
            torch.ops.aten._functional_sym_constrain_range(
                3, min=2, max=5, dep_token=dep_token
            ),
            dep_token,
        )

    def test_eager_mode_functionalization(self) -> None:
        def my_compiler(
            gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor],
        ):
            self.assertExpectedInline(gm.code.strip(), """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1)
    select = torch.ops.aten.select.int(arg0_1, 0, 0)
    eq = torch.ops.aten.eq.Scalar(select, 3);  select = None
    _make_dep_token = torch.ops.aten._make_dep_token.default()
    _functional_assert_async = torch.ops.aten._functional_assert_async.msg(eq, 'assertion error', _make_dep_token);  eq = _make_dep_token = None
    select_1 = torch.ops.aten.select.int(arg0_1, 0, 2)
    eq_1 = torch.ops.aten.eq.Scalar(select_1, 5);  select_1 = None
    _functional_assert_async_1 = torch.ops.aten._functional_assert_async.msg(eq_1, 'assertion error', _functional_assert_async);  eq_1 = _functional_assert_async = None
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
    return (add, _functional_assert_async_1)""")  # noqa: B950
            return make_boxed_func(gm.forward)

        my_compiler = aot_autograd(fw_compiler=my_compiler)

        def f(x):
            b = x.sin()
            assert x[0] == 3
            assert x[2] == 5
            return x.cos() + b

        with patch("functorch.compile.config.functionalize_assertion_ops", True), patch(
            "functorch.compile.config.functionalize_rng_ops", False
        ):
            compiled = torch.compile(f, backend=my_compiler)
            inp = torch.Tensor([3, 4, 5])
            self.assertTrue(torch._dynamo.utils.same(compiled(inp), f(inp)))


if __name__ == "__main__":
    run_tests()
