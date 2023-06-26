"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_eager_mode_functionalization)
"""

# Owner(s): ["module: dynamo"]

from typing import List

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from functorch.compile import make_boxed_func
import torch._functorch.config as config
from contextlib import contextmanager
from torch._dynamo.backends.common import aot_autograd
from torch.testing import FileCheck


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
        def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
            FileCheck().check_count(
                "aten._make_dep_token.default", 1, exactly=True,
            ).run(gm.code)
            FileCheck().check_count(
                "aten._functional_assert_async.msg", 1, exactly=True,
            ).run(gm.code)
            return make_boxed_func(gm.forward)

        my_compiler = aot_autograd(fw_compiler=my_compiler)

        def f(x):
            b = x.sin()
            assert x[0] == 3
            return x.cos() + b

        with _enable_functionalization():
            compiled = torch.compile(f, backend=my_compiler)
            inp = torch.Tensor([3, 4, 5])
            self.assertTrue(torch._dynamo.utils.same(compiled(inp), f(inp)))


@contextmanager
def _enable_functionalization():
    functionalize = config.functionalize_assertion_ops
    config.functionalize_assertion_ops = True
    try:
        yield
    finally:
        config.functionalize_assertion_ops = functionalize


if __name__ == "__main__":
    run_tests()
