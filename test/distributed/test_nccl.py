import unittest

import torch
import torch.cuda.nccl as nccl
import torch.cuda

from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS, load_tests
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

nGPUs = torch.cuda.device_count()
if not TEST_CUDA:
    print('CUDA not available, skipping tests')
    TestCase = object  # noqa: F811


class TestNCCL(TestCase):

    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_unique_id(self):
        uid = nccl.unique_id()
        self.assertIsInstance(uid, bytes)
        self.assertGreater(len(uid), 1)

    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_broadcast(self):
        expected = torch.FloatTensor(128).uniform_()
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            with torch.cuda.device(device):
                tensors.append(torch.cuda.FloatTensor(128))

        nccl.broadcast(tensors)
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce(self):
        tensors = [torch.FloatTensor(128).uniform_() for i in range(nGPUs)]
        expected = torch.FloatTensor(128).zero_()
        for t in tensors:
            expected.add_(t)

        tensors = [tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tensors)

        self.assertEqual(tensors[0], expected)

    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_all_reduce(self):
        tensors = [torch.FloatTensor(128).uniform_() for i in range(nGPUs)]
        expected = torch.FloatTensor(128).zero_()
        for t in tensors:
            expected.add_(t)

        tensors = [tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_all_gather(self):
        inputs = [torch.FloatTensor(128).uniform_() for i in range(nGPUs)]
        expected = torch.cat(inputs, 0)

        inputs = [inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.cuda.FloatTensor(128 * nGPUs, device=i)
                   for i in range(nGPUs)]
        nccl.all_gather(inputs, outputs)

        for tensor in outputs:
            self.assertEqual(tensor, expected)

    @unittest.skipIf(IS_WINDOWS, "NCCL doesn't support Windows")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_scatter(self):
        in_size = 32 * nGPUs
        out_size = 32

        inputs = [torch.FloatTensor(in_size).uniform_() for i in range(nGPUs)]
        expected = torch.FloatTensor(in_size).zero_()
        for t in inputs:
            expected.add_(t)
        expected = expected.view(nGPUs, 32)

        inputs = [inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.cuda.FloatTensor(out_size, device=i)
                   for i in range(nGPUs)]
        nccl.reduce_scatter(inputs, outputs)

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])


if __name__ == '__main__':
    run_tests()
