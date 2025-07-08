# Owner(s): ["module: cpp"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class BinomialDatatypeErrors(TestCase):
    def test_binomial_dtype_errors(self):
        dtypes = [torch.int, torch.long, torch.short]

        for count_dtype in dtypes:
            total_count = torch.tensor([10, 10], dtype=count_dtype)
            total_prob = torch.tensor([0.5, 0.5], dtype=torch.float)

            with self.assertRaisesRegex(
                ValueError,
                "binomial only supports floating-point dtypes for count.*",
            ):
                torch.binomial(total_count, total_prob)

        for prob_dtype in dtypes:
            total_count = torch.tensor([10, 10], dtype=torch.float)
            total_prob = torch.tensor([0.5, 0.5], dtype=prob_dtype)

            with self.assertRaisesRegex(
                ValueError,
                "binomial only supports floating-point dtypes for prob.*",
            ):
                torch.binomial(total_count, total_prob)


if __name__ == "__main__":
    run_tests()
