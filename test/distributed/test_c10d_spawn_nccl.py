# Owner(s): ["oncall: distributed"]

import os
import sys

import test_c10d_spawn
import torch
import torch.distributed as c10d
from test_c10d_spawn import _torch_dist_nn_available
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    sandcastle_skip_if,
    TEST_WITH_DEV_DBG_ASAN,
)

NO_NCCL = not hasattr(c10d, "ProcessGroupNCCL")

# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619
if sys.version_info < (3, 9):

    class ProcessGroupShareTensorTest(
        test_c10d_spawn.AbstractProcessGroupShareTensorTest, TestCase
    ):
        @classmethod
        def _init_pg_nccl(cls, rank, filename, world_size):
            store = c10d.FileStore(filename, world_size)
            return c10d.ProcessGroupNCCL(store, rank, world_size)

        @sandcastle_skip_if(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
        @sandcastle_skip_if(NO_NCCL, "NCCL needed")
        def test_shared_broadcast_nccl(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_broadcast_process,
                [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_nccl,
                1,
            )

        @sandcastle_skip_if(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
        @sandcastle_skip_if(NO_NCCL, "NCCL needed")
        def test_shared_allreduce_nccl(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_allreduce_process,
                [torch.ones(2, 2).to(i) for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_nccl,
                1,
            )

        @classmethod
        def _test_reduce_process(
            cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
        ):
            pg = init_pg(rank, filename, world_size)
            x = shared_tensors[rank]
            pg.reduce(x, root=0, op=c10d.ReduceOp.SUM).wait()
            if rank == 0:
                c2p.put((rank, torch.ones(2, 2) * 2, x.to("cpu")))
            else:
                c2p.put((rank, torch.ones(2, 2), x.to("cpu")))
            p2c.get()

        @sandcastle_skip_if(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
        @sandcastle_skip_if(NO_NCCL, "NCCL needed")
        def test_shared_reduce_nccl(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_reduce_process,
                [torch.ones(2, 2).to(i) for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_nccl,
                1,
            )

        @sandcastle_skip_if(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
        @sandcastle_skip_if(NO_NCCL, "NCCL needed")
        def test_shared_allgather_nccl(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_allgather_process,
                [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_nccl,
                self.world_size,
            )


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctions(MultiProcessTestCase):
        def setUp(self):
            super(TestDistributedNNFunctions, self).setUp()
            self._spawn_processes()

        def tearDown(self):
            super(TestDistributedNNFunctions, self).tearDown()
            try:
                os.remove(self.file_name)
            except OSError:
                pass

        @property
        def op_timeout_sec(self):
            return 1

        @property
        def world_size(self):
            return 2

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @sandcastle_skip_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            y = torch.distributed.nn.broadcast(x, 1)
            self.assertEqual(y, 1 + torch.ones(5, 5))
            z = y.sin().sum()
            z.backward()
            # We can't check the gradient of communications numerically so we have to do some calculations
            if self.rank == 1:
                self.assertEqual(x.grad, 2 * torch.cos(x))
            elif self.rank == 0:
                self.assertEqual(x.grad, torch.zeros(5, 5, device=device))

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_reduce(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='nccl')
            device = torch.device(f"cuda:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            y = torch.distributed.nn.reduce(x, 1, op=c10d.ReduceOp.SUM)

            if self.rank == 1:
                self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

            z = y.sin().sum()
            z.backward()
            # Gradients are broadcasted to both ranks
            x_g = (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_g)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_allreduce(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='nccl')
            device = torch.device(f"cuda:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            y = torch.distributed.nn.all_reduce(x, op=c10d.ReduceOp.SUM)

            self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

            z = y.sin().sum()
            z.backward()
            x_g = 2 * (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_g)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_all_gather(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='nccl')
            device = torch.device(f"cuda:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            tensors = torch.distributed.nn.all_gather(x)
            for i, t in enumerate(tensors):
                self.assertEqual(t, torch.ones(5, 5, device=device) + i)
            y = torch.sum(torch.stack(tensors), axis=0)
            z = y.sin().sum()
            z.backward()

            x_s = 2 * (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_s)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_reduce_scatter(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='nccl')
            device = torch.device(f"cuda:{self.rank}")
            x0 = torch.ones(5, 5, device=device) + self.rank
            x1 = torch.ones(5, 5, device=device) + self.rank + 1
            x0.requires_grad = True
            x1.requires_grad = True
            y = torch.empty_like(x0)
            expected = (1 + self.world_size) * self.world_size / 2 + self.world_size * self.rank
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
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_all_to_all(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='nccl')
            device = torch.device(f"cuda:{self.rank}")
            x0 = torch.ones(5, 5, device=device) + 2 * self.rank
            x1 = torch.ones(5, 5, device=device) + 2 * self.rank
            x0.requires_grad = True
            x1.requires_grad = True
            tensors = torch.distributed.nn.all_to_all(None, [x0, x1])
            for i, t in enumerate(tensors):
                self.assertEqual(t, torch.ones(5, 5, device=device) + 2 * i)
            y = torch.sum(torch.stack(tensors), axis=0)
            z = y.sin().sum()
            z.backward()
            x_s = (4 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x0.grad, x_s)
            self.assertEqual(x1.grad, x_s)


if __name__ == "__main__":
    run_tests()
