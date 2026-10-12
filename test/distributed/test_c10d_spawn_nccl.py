# Owner(s): ["oncall: distributed"]


from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

import torch
import torch.distributed as c10d
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


NO_NCCL = not hasattr(c10d, "ProcessGroupNCCL")

# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsNccl(TestDistributedNNFunctions):
        # Test Common Ops First.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            self._test_broadcast("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self):
            self._test_reduce("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self):
            self._test_allreduce("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_gather(self):
            self._test_all_gather("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self):
            self._test_all_to_all("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self):
            self._test_all_to_all_single("nccl")

        # Test Ops only supported in NCCL.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce_scatter(self):
            self._test_reduce_scatter("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce_scatter_non_contiguous(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")

            class NonContiguousGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    return grad_output.clone().transpose(0, 1)

            x0 = torch.rand(5, 5, device=device, requires_grad=True)
            x1 = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.empty(5, 5, device=device)

            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_reduce_non_contiguous(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")

            class NonContiguousGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    return grad_output.clone().transpose(0, 1)

            x = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.distributed.nn.all_reduce(x)
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_gather_base(self):
            self._test_all_gather_base("nccl")


if __name__ == "__main__":
    run_tests()
