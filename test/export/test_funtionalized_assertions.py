# Owner(s): ["module: dynamo"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


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


if __name__ == "__main__":
    run_tests()
