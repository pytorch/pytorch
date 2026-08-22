# Owner(s): ["oncall: distributed"]


from test_c10d_spawn import TestDistributedNNFunctions

import torch
import torch.distributed as c10d
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsNccl(TestDistributedNNFunctions):
        BACKEND = "nccl"

        def setUp(self):
            if not c10d.is_nccl_available():
                self.skipTest("c10d was not compiled with the NCCL backend")
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                self.skipTest("Need at least 2 CUDA GPUs")
            super().setUp()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        def test_reduce_scatter(self):
            self._init_process_group()
            device = self._get_device()
            x0 = torch.ones(5, 5, device=device) + self.rank
            x1 = torch.ones(5, 5, device=device) + self.rank + 1
            x0.requires_grad = True
            x1.requires_grad = True
            y = torch.empty_like(x0)
            expected = (
                1 + self.world_size
            ) * self.world_size / 2 + self.world_size * self.rank
            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            self.assertEqual(y, torch.ones(5, 5, device=device) * expected)
            z = y.sin().sum()
            z.backward()
            expected_0 = (1 + self.world_size) * self.world_size / 2
            expected_1 = expected_0 + self.world_size
            x_s_0 = (expected_0 * torch.ones(5, 5, device=device)).cos()
            x_s_1 = (expected_1 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x0.grad, x_s_0)
            self.assertEqual(x1.grad, x_s_1)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        def test_reduce_scatter_non_contiguous(self):
            self._init_process_group()
            device = self._get_device()

            class NonContiguousGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    return grad_output.clone().transpose(0, 1)

            x0 = torch.rand(5, 5, device=device, requires_grad=True)
            x1 = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.empty(5, 5, device=device)

            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        def test_all_reduce_non_contiguous(self):
            self._init_process_group()
            device = self._get_device()

            class NonContiguousGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    return grad_output.clone().transpose(0, 1)

            x = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.distributed.nn.all_reduce(x)
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        def test_all_gather_base(self):
            self._init_process_group()

            device = self._get_device()
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True

            output = torch.empty(5 * self.world_size, 5, device=device)
            output = torch.distributed.nn.functional._all_gather_base(output, x)
            self.assertEqual(output.size(), torch.Size((5 * self.world_size, 5)))

            for idx in range(self.world_size):
                self.assertEqual(
                    output[5 * idx : 5 * (idx + 1)],
                    torch.ones(5, 5, device=device) + idx,
                )

            y = torch.sum(output.view(self.world_size, 5, 5), axis=0)
            z = y.sin().sum()
            z.backward()

            x_s = 2 * (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_s)


if __name__ == "__main__":
    run_tests()
