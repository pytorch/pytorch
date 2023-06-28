# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import TestCase, run_tests
import unittest

# We cannot import TEST_CUDA from torch.testing._internal.common_cuda here, because if we do that,
# the TEST_CUDNN line from torch.testing._internal.common_cuda will be executed multiple times
# as well during the execution of this test suite, and it will cause
# CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()
try:
    import tabulate  # noqa: F401  # type: ignore[import]
    from torch.utils.benchmark.utils.compile import bench_all
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

@unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
@unittest.skipIf(not HAS_TABULATE, "tabulate not available")
class TestCompileBenchmarkUtil(TestCase):
    def test_training_and_inference(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super(ToyModel, self).__init__()
                self.weight = torch.nn.Parameter(torch.Tensor(2, 2))

            def forward(self, x):
                return x * self.weight

        torchdynamo.reset()
        model = ToyModel().cuda()

        inference_table = bench_all(model, torch.ones(1024, 2, 2).cuda(), 5)
        self.assertTrue("Inference" in inference_table and "Eager" in inference_table and "-" in inference_table)

        training_table = bench_all(model, torch.ones(1024, 2, 2).cuda(), 5, optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
        self.assertTrue("Train" in training_table and "Eager" in training_table and "-" in training_table)

if __name__ == '__main__':
    run_tests()
