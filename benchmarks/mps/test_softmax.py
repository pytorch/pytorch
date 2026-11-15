"""Check sampled reduction rows from every model shape against CPU float64."""

import math

from softmax_common import CASES

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestSoftmaxModels(TestCase):
    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("case", range(len(CASES)))
    @parametrize("op", ["softmax", "log_softmax"])
    def test_models(self, device, dtype, case, op):
        shape, dim = CASES[case]
        torch.manual_seed(case)
        x = torch.randn(shape, dtype=dtype, device=device)
        output = getattr(torch, op)(x, dim=dim)
        outer = math.prod(shape[:dim])
        inner = math.prod(shape[dim + 1 :])
        rows = torch.tensor([0, outer // 2, outer - 1], device=device)
        cols = torch.tensor([0, inner // 2, inner - 1], device=device)
        samples = x.view(outer, shape[dim], inner)[rows, :, cols].cpu()
        actual = output.view(outer, shape[dim], inner)[rows, :, cols].cpu()
        expected = getattr(samples.double(), op)(-1).to(dtype)
        self.assertEqual(actual, expected)
        del x, output
        torch.mps.empty_cache()


instantiate_device_type_tests(
    TestSoftmaxModels, globals(), only_for="mps", allow_mps=True
)

if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    run_tests()
