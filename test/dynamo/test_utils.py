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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
