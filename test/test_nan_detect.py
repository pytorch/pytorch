# Owner(s): ["module: autograd"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.nan_detect import NanDetectMode


class TestNanDetectMode(TestCase):
    def test_nan_detected(self):
        with self.assertRaisesRegex(RuntimeError, "returned NaN"):
            with NanDetectMode():
                x = torch.tensor([1.0, float("nan")])
                x + 1

    def test_clean_tensors_pass(self):
        with NanDetectMode():
            x = torch.randn(4, 4)
            y = x @ x.t()
            z = y.relu()
        self.assertFalse(torch.isnan(z).any())

    def test_inf_not_detected_by_default(self):
        with NanDetectMode():
            x = torch.tensor([1.0, float("inf")])
            y = x + 1
        self.assertTrue(torch.isinf(y).any())

    def test_inf_detected_when_enabled(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            with NanDetectMode(check_inf=True):
                x = torch.tensor([1.0, float("inf")])
                x + 1

    def test_integer_tensors_skipped(self):
        with NanDetectMode():
            x = torch.tensor([1, 2, 3])
            y = x + 1
        self.assertEqual(y.tolist(), [2, 3, 4])

    def test_empty_tensors_skipped(self):
        with NanDetectMode():
            x = torch.empty(0)
            y = x + 1
        self.assertEqual(y.numel(), 0)

    def test_nn_module(self):
        with self.assertRaisesRegex(RuntimeError, "returned NaN"):
            with NanDetectMode():
                m = torch.nn.Linear(4, 4)
                x = torch.tensor([[1.0, float("nan"), 3.0, 4.0]])
                m(x)

    def test_context_manager_restores(self):
        x = torch.tensor([float("nan")])
        try:
            with NanDetectMode():
                x + 1
        except RuntimeError:
            pass
        y = x + 1
        self.assertTrue(torch.isnan(y).any())


if __name__ == "__main__":
    run_tests()
