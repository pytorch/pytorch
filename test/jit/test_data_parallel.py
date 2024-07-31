# Owner(s): ["oncall: jit"]

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.parallel as dp

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA_MULTI_GPU

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestDataParallel(JitTestCase):
    class Mpy(torch.nn.Module):
        def __init__(self) -> None:
            super(TestDataParallel.Mpy, self).__init__()
            self.m = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 2)
            )

        @torch.jit.ignore
        def forward(self, input):
            return self.m(input)

    class Mpy1(torch.nn.Module):
        def __init__(self, block):
            super(TestDataParallel.Mpy1, self).__init__()
            self.m = block

        @torch.jit.ignore
        def forward(self, input):
            return self.m.forward(input)

    class Mpy2(torch.nn.Module):
        def __init__(self, block1, block2):
            super(TestDataParallel.Mpy2, self).__init__()
            self.m1 = block1
            self.m2 = block2

        @torch.jit.ignore
        def forward(self, input):
            x = self.m1.forward(input)
            return self.m2(x)

    class Msm(torch.jit.ScriptModule):
        __constants__ = ["m"]

        def __init__(self) -> None:
            super(TestDataParallel.Msm, self).__init__()
            self.m = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 2)
            )

        @torch.jit.script_method
        def forward(self, input):
            return self.m(input)

    class Msm1(torch.jit.ScriptModule):
        def __init__(self, block):
            super(TestDataParallel.Msm1, self).__init__()
            self.block = block

        @torch.jit.script_method
        def forward(self, input):
            x = self.block(input)
            return x

    def check_replicas(self, module, replicas, input_shape=(2, 2)):
        input = torch.randn(input_shape).cuda()
        expected_output = module(input).data
        for i, replica in enumerate(replicas):
            for p in replica.parameters():
                self.assertEqual(p.get_device(), i)
            for b in replica.buffers():
                self.assertEqual(b.get_device(), i)
            replica_input = input.cuda(i)
            self.assertEqual(replica(replica_input).data, expected_output)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_python_submodule_script(self):
        module = self.Mpy1(self.Msm()).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_shared_module(self):
        s = self.Msm()
        p1 = self.Mpy1(s)
        module = self.Mpy2(p1, s).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_traced_module(self):
        module = torch.jit.trace(self.Mpy1(self.Mpy()), torch.ones(2, 2)).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_tensor_sharing(self):
        module = self.Msm1(self.Msm()).cuda()
        replica = dp.replicate(module, {0, 1})

        def assert_share_data(t1, t2):
            # Only checks that they point to the same memory on the same device.
            return (
                t1.device == t2.device
                and t1.storage().data_ptr() == t2.storage().data_ptr()
            )

        for p1, p2 in zip(module.parameters(), replica[0].parameters()):
            self.assertTrue(assert_share_data(p1, p2))

        for p1, p2 in zip(module.buffers(), replica[0].buffers()):
            self.assertTrue(assert_share_data(p1, p2))

        for p1, p2 in zip(module.parameters(), replica[1].parameters()):
            self.assertFalse(assert_share_data(p1, p2))

        for p1, p2 in zip(module.buffers(), replica[1].buffers()):
            self.assertFalse(assert_share_data(p1, p2))

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_tensor_sharing_with_forward(self):
        module = self.Msm1(self.Msm()).cuda()
        replica = dp.replicate(module, {0, 1})
        x = torch.ones(2, 2, requires_grad=True).cuda()
        first_forward = module(x)
        first_forward.sum().backward()
        with torch.no_grad():
            for p in module.parameters():
                # Use .data here to avoid version counter bump.
                # The graph created by the following forward will be wrong but
                # we never backward through them so it's fine
                p.data -= 1.0 * p.grad
        second_forward = module(x)

        # replica which is on the same GPU has a shallow copy of the original
        # params and buffers
        r0_forward = replica[0](x)
        self.assertEqual(second_forward, r0_forward)

        # replica which is on a different GPU has a deep copy of the original
        # params and buffers
        x1 = torch.ones(2, 2, requires_grad=True).cuda(device=1)
        r1_forward = replica[1](x1)
        self.assertEqual(first_forward, r1_forward)
