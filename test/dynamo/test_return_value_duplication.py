# Owner(s): ["module: dynamo"]
import typing

import torch
import torch.testing._internal.common_utils


class TestReturnValueDuplication(torch.testing._internal.common_utils.TestCase):
    """Tests that compiled functions returning identically constructed but otherwise
    separate tensors still return separate tensors.

    The original bug that prompted this test was triggered by requires_grad=True, so add
    explicit tests for that case as well."""

    @staticmethod
    def _test_func(val: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return val * 2, val * 2

    def test_return_value_duplication_nograd(self):
        x = torch.randn(2)

        expect = self._test_func(x)
        self.assertFalse(expect[0].requires_grad)
        self.assertFalse(expect[1].requires_grad)
        self.assertIsNot(expect[0], expect[1])

        actual = torch.compile(self._test_func)(x)
        self.assertFalse(actual[0].requires_grad)
        self.assertFalse(actual[1].requires_grad)
        self.assertIsNot(actual[0], actual[1])

    def test_return_value_duplication_grad(self):
        x = torch.randn(2, requires_grad=True)

        expect = self._test_func(x)
        self.assertTrue(expect[0].requires_grad)
        self.assertTrue(expect[1].requires_grad)
        self.assertIsNot(expect[0], expect[1])

        actual = torch.compile(self._test_func)(x)
        self.assertTrue(actual[0].requires_grad)
        self.assertTrue(actual[1].requires_grad)
        self.assertIsNot(actual[0], actual[1])


if __name__ == "__main__":
    import torch._dynamo.test_case

    torch._dynamo.test_case.run_tests()
