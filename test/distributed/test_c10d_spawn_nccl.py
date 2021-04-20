import copy
import os
import sys
import tempfile
import unittest

import torch
import torch.distributed as c10d
import torch.multiprocessing as mp
import torch.nn as nn

from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_distributed import \
    create_device, MultiProcessTestCase, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TestCase, load_tests, \
    run_tests
from torch.testing._internal.common_utils import NO_MULTIPROCESSING_SPAWN, TEST_WITH_TSAN

import test_c10d_spawn
from test_c10d_spawn import _torch_dist_nn_available


NO_NCCL = not hasattr(c10d, "ProcessGroupNCCL")


class ProcessGroupShareTensorTest(test_c10d_spawn.AbstractProcessGroupShareTensorTest, TestCase):

    @classmethod
    def _init_pg_nccl(cls, rank, filename, world_size):
        store = c10d.FileStore(filename, world_size)
        return c10d.ProcessGroupNCCL(store, rank, world_size)

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @unittest.skipIf(NO_NCCL, "NCCL needed")
    def test_shared_broadcast_nccl(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_broadcast_process,
            [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_nccl,
            1)

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @unittest.skipIf(NO_NCCL, "NCCL needed")
    def test_shared_allreduce_nccl(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_allreduce_process,
            [torch.ones(2, 2).to(i) for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_nccl,
            1)

    @classmethod
    def _test_reduce_process(
            cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, filename, world_size)
        x = shared_tensors[rank]
        pg.reduce(x, root=0, op=c10d.ReduceOp.SUM).wait()
        if rank == 0:
            c2p.put((rank, torch.ones(2, 2) * 2, x.to("cpu")))
        else:
            c2p.put((rank, torch.ones(2, 2), x.to("cpu")))
        p2c.get()

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @unittest.skipIf(NO_NCCL, "NCCL needed")
    def test_shared_reduce_nccl(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_reduce_process,
            [torch.ones(2, 2).to(i) for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_nccl,
            1)

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @unittest.skipIf(NO_NCCL, "NCCL needed")
    def test_shared_allgather_nccl(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_allgather_process,
            [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_nccl,
            self.world_size)


if __name__ == '__main__':
    run_tests()
