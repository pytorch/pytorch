# Owner(s): ["module: dynamo"]
import torch
from torch._dynamo import utils
from torch._inductor.test_case import TestCase


class TestUtils(TestCase):
    def test_nan(self):
        a = torch.Tensor([float("nan")])
        b = torch.Tensor([float("nan")])
        fp64_ref = torch.DoubleTensor([5.0])
        res = utils.same(a, b, fp64_ref=fp64_ref, equal_nan=True)
        self.assertTrue(res)

    def test_larger_multiplier_for_smaller_tensor(self):
        """
        Tensor numel between (10, 500]
        """
        N = 100
        fp64_ref = torch.full([N], 0.0, dtype=torch.double)
        a = torch.full([N], 1.0)
        tol = 4 * 1e-2
        self.assertTrue(utils.same(a, a * 2, fp64_ref=fp64_ref, tol=tol))
        self.assertFalse(utils.same(a, a * 4, fp64_ref=fp64_ref, tol=tol))
        self.assertTrue(
            utils.same(
                a,
                a * 4,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )
        self.assertFalse(
            utils.same(
                a,
                a * 6,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )

    def test_larger_multiplier_for_even_smaller_tensor(self):
        """
        Tesnor numel <=10
        """
        fp64_ref = torch.DoubleTensor([0.0])
        a = torch.Tensor([1.0])
        tol = 4 * 1e-2
        self.assertTrue(utils.same(a, a * 2, fp64_ref=fp64_ref, tol=tol))
        self.assertFalse(utils.same(a, a * 7, fp64_ref=fp64_ref, tol=tol))
        self.assertTrue(
            utils.same(
                a,
                a * 7,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )
        self.assertFalse(
            utils.same(
                a,
                a * 20,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
