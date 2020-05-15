import unittest

import torch
import torch.cuda.nccl as nccl
import torch.cuda

from torch.testing._internal.common_utils import (TestCase, run_tests,
                                                  IS_WINDOWS, load_tests,
                                                  TEST_WITH_ROCM)
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

nGPUs = torch.cuda.device_count()
if not TEST_CUDA:
    print('CUDA not available, skipping tests')
    TestCase = object  # noqa: F811


datatypes = [torch.float, torch.bfloat16] if TEST_WITH_ROCM else [torch.float]

class TestNCCL(TestCase):

    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_unique_id(self, device):
        uid = nccl.unique_id()
        self.assertIsInstance(uid, bytes)
        self.assertGreater(len(uid), 1)

    @unittest.skipIf(TEST_WITH_ROCM, 'Skip NCCL tests for ROCm, see https://github.com/pytorch/pytorch/issues/38833')
    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_broadcast(self, device, dtype):
        expected = torch.zeros(128, dtype=dtype).uniform_()
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            with torch.cuda.device(device):
                tensors.append(torch.zeros(128, dtype=dtype).cuda())

        nccl.broadcast(tensors)
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

    @unittest.skipIf(TEST_WITH_ROCM, 'Skip NCCL tests for ROCm, see https://github.com/pytorch/pytorch/issues/38834')
    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce(self, device, dtype):
        tensors = [torch.zeros(128, dtype=dtype).uniform_() for i in range(nGPUs)]
        expected = torch.zeros(128, dtype=dtype)
        for t in tensors:
            expected.add_(t)

        tensors = [tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tensors)

        self.assertEqual(tensors[0], expected)

    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_reduce(self, device, dtype):
        tensors = [torch.zeros(128, dtype=dtype).uniform_() for i in range(nGPUs)]
        expected = torch.zeros(128, dtype=dtype)
        for t in tensors:
            expected.add_(t)

        tensors = [tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

    @unittest.skipIf(TEST_WITH_ROCM, 'Skip NCCL tests for ROCm')
    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_gather(self, device, dtype):
        inputs = [torch.zeros(128, dtype=dtype).uniform_() for i in range(nGPUs)]
        expected = torch.cat(inputs, 0)

        inputs = [inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(128 * nGPUs, device=i, dtype=dtype)
                   for i in range(nGPUs)]
        nccl.all_gather(inputs, outputs)

        for tensor in outputs:
            self.assertEqual(tensor, expected)

    @unittest.skipIf(TEST_WITH_ROCM, 'Skip NCCL tests for ROCm, see https://github.com/pytorch/pytorch/issues/38835')
    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce_scatter(self, device, dtype):
        in_size = 32 * nGPUs
        out_size = 32

        inputs = [torch.zeros(in_size, dtype=dtype).uniform_() for i in range(nGPUs)]
        expected = torch.zeros(in_size, dtype=dtype)
        for t in inputs:
            expected.add_(t)
        expected = expected.view(nGPUs, 32)

        inputs = [inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype)
                   for i in range(nGPUs)]
        nccl.reduce_scatter(inputs, outputs)

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])


instantiate_device_type_tests(TestNCCL, globals(), only_for='cuda')

if __name__ == '__main__':
    run_tests()
