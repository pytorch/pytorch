from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import fcntl
import multiprocessing
import os
import sys
import time
import unittest
from contextlib import contextmanager
from functools import reduce, wraps
import tempfile

import torch
import torch.cuda
import torch.distributed.deprecated as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common_utils import TestCase, run_tests
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR


BACKEND = os.environ["BACKEND"]
TEMP_DIR = os.environ["TEMP_DIR"]
INIT_METHOD = os.getenv("INIT_METHOD", "env://")
MASTER_PORT = "29500"

DEFAULT_TIMEOUT = 300
CUSTOMIZED_TIMEOUT = {"test_DistributedDataParallel": 500}


def get_timeout(test_id):
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        return DEFAULT_TIMEOUT


if not dist.is_available():
    print("Distributed not available, skipping tests")
    sys.exit(0)

SKIP_IF_NO_CUDA_EXIT_CODE = 75
SKIP_IF_NO_GPU_EXIT_CODE = 76
SKIP_IF_SMALL_WORLDSIZE_EXIT_CODE = 77
SKIP_IF_BACKEND_UNAVAILABLE = 78


def skip_if_no_cuda_distributed(func):
    func.skip_if_no_cuda_distributed = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            sys.exit(SKIP_IF_NO_CUDA_EXIT_CODE)

        return func(*args, **kwargs)

    return wrapper


def skip_if_no_gpu(func):
    """ Nccl multigpu tests requires at least 2 GPUS. Skip if this is not met"""
    func.skip_if_no_gpu = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            sys.exit(SKIP_IF_NO_CUDA_EXIT_CODE)
        if torch.cuda.device_count() < int(os.environ["WORLD_SIZE"]):
            sys.exit(SKIP_IF_NO_GPU_EXIT_CODE)

        return func(*args, **kwargs)

    return wrapper


def skip_if_small_worldsize(func):
    func.skip_if_small_worldsize = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) <= 2:
            sys.exit(SKIP_IF_SMALL_WORLDSIZE_EXIT_CODE)

        return func(*args, **kwargs)

    return wrapper


def apply_hack_for_nccl():
    # This is a hack for a known NCCL issue using multiprocess
    # in conjunction with multiple threads to manage different GPUs which
    # may cause ncclCommInitRank to fail.
    # http://docs.nvidia.com/deeplearning/sdk/nccl-release-notes/rel_2.1.4.html#rel_2.1.4
    # It slows down the performance of collective operations.
    # Without this setting NCCL might throw unhandled error.
    os.environ["NCCL_MAX_NRINGS"] = "1"


@contextmanager
def _lock():
    lockfile = os.path.join(TEMP_DIR, "lockfile")
    with open(lockfile, "w") as lf:
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            lf.close()


def _build_tensor(size, value=None):
    if value is None:
        value = size
    return torch.FloatTensor(size, size, size).fill_(value)


class Barrier(object):
    barrier_id = 0

    @classmethod
    def init(cls):
        cls.barrier_id = 0
        barrier_dir = os.path.join(TEMP_DIR, "barrier")
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))

    @classmethod
    def sync(cls, timeout=5):
        cls.barrier_id += 1
        barrier_dir = os.path.join(TEMP_DIR, "barrier")
        pid = str(os.getpid())
        barrier_file = os.path.join(barrier_dir, pid)
        with _lock():
            with open(barrier_file, "w") as f:
                f.write(str(cls.barrier_id))

        start_time = time.time()
        while True:
            arrived = 0
            with _lock():
                for f_name in os.listdir(barrier_dir):
                    with open(os.path.join(barrier_dir, f_name), "r") as f:
                        data = f.read()
                        if int(data) >= cls.barrier_id:
                            arrived += 1
            if arrived == dist.get_world_size():
                break

            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            time.sleep(0.1)


# The test network must live at top level so we can test pickling it.


class _FC2(nn.Module):

    def __init__(self):
        super(_FC2, self).__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class _DistTestBase(object):
    def _barrier(self, *args, **kwargs):
        Barrier.sync(*args, **kwargs)

    def _init_group_test(self):
        group = [1, 2]
        group_id = dist.new_group(group)
        rank = dist.get_rank()
        if rank not in group:
            return ([], None, rank)

        return (group, group_id, rank)

    def _init_global_test(self):
        group = [i for i in range(0, dist.get_world_size())]
        group_id = dist.group.WORLD
        rank = dist.get_rank()
        return (group, group_id, rank)

    # HELPER FOR MULTIGPU TESTS
    def _init_multigpu_helper(self):
        """Multigpu tests are designed to simulate the multi nodes with multi
        GPUs on each node. Nccl backend requires equal #GPUs in each process.
        On a single node, all visible GPUs are evenly
        divided to subsets, each process only uses a subset.
        """
        nGPUs = torch.cuda.device_count()
        world_size = dist.get_world_size()
        visible_devices = range(nGPUs)

        if BACKEND == "nccl":
            apply_hack_for_nccl()

        nGPUs_per_process = nGPUs // world_size
        rank_to_GPU = {
            i: list(
                visible_devices[i * nGPUs_per_process: (i + 1) * nGPUs_per_process]
            )
            for i in range(world_size)
        }
        return rank_to_GPU

    # GET RANK
    def test_get_rank(self):
        test_dir = os.path.join(TEMP_DIR, "test_dir")
        pid = str(os.getpid())
        num_processes = dist.get_world_size()
        with open(os.path.join(test_dir, pid), "w") as f:
            f.write(str(dist.get_rank()))

        self._barrier()

        all_ranks = set()
        for f_name in os.listdir(test_dir):
            with open(os.path.join(test_dir, f_name), "r") as f:
                all_ranks.add(int(f.read()))
        self.assertEqual(len(all_ranks), num_processes)

        self._barrier()

        if dist.get_rank() == 0:
            for f_name in os.listdir(test_dir):
                os.unlink(os.path.join(test_dir, f_name))

        self._barrier()

    # SEND RECV
    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support send/recv")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support send/recv")
    def test_send_recv(self):
        rank = dist.get_rank()
        tensor = _build_tensor(rank + 1)
        for dest in range(0, dist.get_world_size()):
            if dest == rank:
                continue
            dist.send(tensor, dest)

        for src in range(0, dist.get_world_size()):
            if src == rank:
                continue
            tensor = _build_tensor(src + 1, value=-1)
            expected_tensor = _build_tensor(src + 1)
            dist.recv(tensor, src)
            self.assertEqual(tensor, expected_tensor)

        self._barrier()

    # SEND RECV ANY SOURCE
    @unittest.skipIf(
        BACKEND == "gloo", "Gloo does not support send/recv from any source"
    )
    @unittest.skipIf(
        BACKEND == "nccl", "Nccl does not support send/recv from any source"
    )
    def test_send_recv_any_source(self):
        rank = dist.get_rank()
        tensor = _build_tensor(10, rank)
        for dest in range(0, dist.get_world_size()):
            if dest == rank:
                continue
            dist.send(tensor, dest)

        recv_ranks = set()
        for src in range(0, dist.get_world_size()):
            if src == rank:
                continue
            tensor = _build_tensor(10, value=-1)
            sender = dist.recv(tensor)
            self.assertTrue(tensor.eq(sender).all())
            recv_ranks.add(sender)

        self.assertEqual(len(recv_ranks), dist.get_world_size() - 1)
        self._barrier()

    # ISEND
    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support isend")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support isend")
    def test_isend(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            requests = [
                dist.isend(_build_tensor(dest, 10), dest)
                for dest in range(1, world_size)
            ]
            for request in requests:
                request.wait()
                self.assertTrue(request.is_completed())
        else:
            tensor = _build_tensor(rank, -1)
            dist.recv(tensor, 0)
            self.assertEqual(tensor, _build_tensor(rank, 10))

        self._barrier()

    # IRECV
    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support irecv")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support irecv")
    def test_irecv(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            expected_tensors = [_build_tensor(src, -1) for src in range(1, world_size)]
            requests = [
                dist.irecv(expected_tensors[src - 1], src)
                for src in range(1, world_size)
            ]

            for src in range(1, world_size):
                requests[src - 1].wait()
                self.assertTrue(requests[src - 1].is_completed())
                self.assertEqual(expected_tensors[src - 1], _build_tensor(src, 10))
        else:
            tensor = _build_tensor(rank, 10)
            dist.send(tensor, 0)

        self._barrier()

    # BROADCAST
    def _test_broadcast_helper(
        self, group, group_id, rank, cuda=False, rank_to_GPU=None
    ):
        for ttype, value, requires_cuda in [
            ("torch.FloatTensor", -1e-10, False),
            ("torch.DoubleTensor", -1e-100, False),
            ("torch.HalfTensor", -0.1, True),
            ("torch.CharTensor", -2, False),
            ("torch.ByteTensor", 129, False),
            ("torch.IntTensor", -1e5, False),
            ("torch.LongTensor", -1e15, False),
        ]:
            if requires_cuda and not cuda:
                continue
            for src in group:
                expected_tensor = _build_tensor(src + 1, value).type(ttype)
                if cuda:
                    expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
                if rank == src:
                    dist.broadcast(expected_tensor, src, group_id)
                else:
                    tensor = _build_tensor(src + 1, -1).type(ttype)
                    if cuda:
                        tensor = tensor.cuda(rank_to_GPU[rank][0])
                    dist.broadcast(tensor, src, group_id)
                    self.assertEqual(tensor.size(), expected_tensor.size())
                    self.assertEqual(tensor.ne(expected_tensor).max(), 0)

        self._barrier()

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_broadcast(self):
        group, group_id, rank = self._init_global_test()
        self._test_broadcast_helper(group, group_id, rank)

    @unittest.skipIf(
        BACKEND != "gloo" and BACKEND != "nccl",
        "Only Gloo and Nccl backend supports CUDA allReduce",
    )
    @skip_if_no_cuda_distributed
    @skip_if_no_gpu
    def test_broadcast_cuda(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_broadcast_helper(group, group_id, rank, True, rank_to_GPU)

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_broadcast_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_broadcast_helper(group, group_id, rank)

    # REDUCE
    def _test_reduce_helper(
        self,
        group,
        group_id,
        rank,
        op,
        master_value,
        worker_value,
        expected_value,
        cuda=False,
        rank_to_GPU=None,
    ):
        for src in group:
            if rank == src:
                tensor = _build_tensor(src + 1).fill_(master_value)
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                dist.reduce(tensor, src, op, group_id)
                self.assertEqual(tensor, _build_tensor(src + 1, expected_value))
            else:
                tensor = _build_tensor(src + 1).fill_(worker_value)
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                dist.reduce(tensor, src, op, group_id)

        self._barrier()

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support reduce")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_reduce_sum(self):
        group, group_id, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.SUM,
            2,
            10,
            2 + (10 * (len(group) - 1)),
        )

    @unittest.skipIf(BACKEND != "nccl", "Only Nccl supports CUDA reduce")
    @skip_if_no_cuda_distributed
    @skip_if_no_gpu
    def test_reduce_sum_cuda(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.SUM,
            2,
            10,
            2 + 10 * (len(group) - 1),
            True,
            rank_to_GPU,
        )

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support reduce")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_reduce_product(self):
        group, group_id, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.PRODUCT,
            2,
            10,
            reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
        )

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support reduce")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_reduce_min(self):
        group, group_id, rank = self._init_global_test()
        self._test_reduce_helper(group, group_id, rank, dist.reduce_op.MIN, 1010, 1, 1)

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support reduce")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_reduce_max(self):
        group, group_id, rank = self._init_global_test()
        self._test_reduce_helper(group, group_id, rank, dist.reduce_op.MAX, -1, 10, 10)

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support reduce")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_reduce_group_sum(self):
        group, group_id, rank = self._init_group_test()
        self._test_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.SUM,
            2,
            10,
            2 + (10 * (len(group) - 1)),
        )

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support reduce")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_reduce_group_product(self):
        group, group_id, rank = self._init_group_test()
        self._test_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.PRODUCT,
            2,
            10,
            reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
        )

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support reduce")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_reduce_group_min(self):
        group, group_id, rank = self._init_group_test()
        self._test_reduce_helper(group, group_id, rank, dist.reduce_op.MIN, 1010, 1, 1)

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support reduce")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_reduce_group_max(self):
        group, group_id, rank = self._init_group_test()
        self._test_reduce_helper(group, group_id, rank, dist.reduce_op.MAX, -1, 10, 10)

    # ALL REDUCE
    def _test_all_reduce_helper(
        self,
        group,
        group_id,
        rank,
        op,
        master_value,
        worker_value,
        expected_value,
        cuda=False,
        rank_to_GPU=None,
    ):
        for src in group:
            if rank == src:
                tensor = _build_tensor(src + 1).fill_(master_value)
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                dist.all_reduce(tensor, op, group_id)
                self.assertEqual(tensor, _build_tensor(src + 1, expected_value))
            else:
                tensor = _build_tensor(src + 1).fill_(worker_value)
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                dist.all_reduce(tensor, op, group_id)
                self.assertEqual(tensor, _build_tensor(src + 1, expected_value))

        self._barrier()

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_all_reduce_sum(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.SUM,
            2,
            10,
            2 + (10 * (len(group) - 1)),
        )

    @unittest.skipIf(
        BACKEND != "gloo" and BACKEND != "nccl",
        "Only Gloo & Nccl backend support CUDA allReduce",
    )
    @skip_if_no_cuda_distributed
    @skip_if_no_gpu
    def test_all_reduce_sum_cuda(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_all_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.SUM,
            2,
            10,
            2 + (10 * (len(group) - 1)),
            True,
            rank_to_GPU,
        )

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_all_reduce_product(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.PRODUCT,
            2,
            10,
            reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
        )

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_all_reduce_min(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.MIN, 1010, 1, 1
        )

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_all_reduce_max(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.MAX, -1, 10, 10
        )

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_all_reduce_group_sum(self):
        group, group_id, rank = self._init_group_test()
        self._test_all_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.SUM,
            2,
            10,
            2 + (10 * (len(group) - 1)),
        )

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_all_reduce_group_product(self):
        group, group_id, rank = self._init_group_test()
        self._test_all_reduce_helper(
            group,
            group_id,
            rank,
            dist.reduce_op.PRODUCT,
            2,
            10,
            reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
        )

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_all_reduce_group_min(self):
        group, group_id, rank = self._init_group_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.MIN, 1010, 1, 1
        )

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_all_reduce_group_max(self):
        group, group_id, rank = self._init_group_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.MAX, -1, 10, 10
        )

    # SCATTER
    def _test_scatter_helper(self, group, group_id, rank):
        for dest in group:
            tensor = _build_tensor(dest + 1, -1)
            expected_tensor = _build_tensor(dest + 1, rank)
            tensors = (
                [_build_tensor(dest + 1, i) for i in group] if rank == dest else []
            )
            dist.scatter(tensor, src=dest, scatter_list=tensors, group=group_id)
            self.assertEqual(tensor, expected_tensor)

        self._barrier()

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support scatter")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support scatter")
    def test_scatter(self):
        group, group_id, rank = self._init_global_test()
        self._test_scatter_helper(group, group_id, rank)

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support scatter")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support scatter")
    @skip_if_small_worldsize
    def test_scatter_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_scatter_helper(group, group_id, rank)

    # GATHER
    def _test_gather_helper(self, group, group_id, rank):
        for dest in group:
            tensor = _build_tensor(dest + 1, rank)
            tensors = (
                [_build_tensor(dest + 1, -1) for i in group] if rank == dest else []
            )
            dist.gather(tensor, dst=dest, gather_list=tensors, group=group_id)
            if rank == dest:
                expected_tensors = [_build_tensor(dest + 1, i) for i in group]
                for t1, t2 in zip(tensors, expected_tensors):
                    self.assertEqual(t1, t2)

        self._barrier()

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support gather")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_gather(self):
        group, group_id, rank = self._init_global_test()
        self._test_gather_helper(group, group_id, rank)

    @unittest.skipIf(BACKEND == "gloo", "Gloo does not support gather")
    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_gather_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_gather_helper(group, group_id, rank)

    # ALL GATHER
    def _test_all_gather_helper(
        self, group, group_id, rank, cuda=False, rank_to_GPU=None
    ):
        for dest in group:
            tensor = _build_tensor(dest + 1, rank)
            tensors = [_build_tensor(dest + 1, -1) for i in group]
            if cuda:
                tensor = tensor.cuda(rank_to_GPU[rank][0])
                tensors = [t.cuda(rank_to_GPU[rank][0]) for t in tensors]
            dist.all_gather(tensors, tensor, group_id)

            expected_tensors = [_build_tensor(dest + 1, i) for i in group]
            for t1, t2 in zip(tensors, expected_tensors):
                self.assertEqual(t1, t2)

        self._barrier()

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_all_gather(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_gather_helper(group, group_id, rank)

    @unittest.skipIf(BACKEND != "nccl", "Only Nccl supports CUDA all gather")
    @skip_if_no_cuda_distributed
    @skip_if_no_gpu
    def test_all_gather_cuda(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_all_gather_helper(group, group_id, rank, True, rank_to_GPU)

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_all_gather_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_all_gather_helper(group, group_id, rank)

    # BARRIER
    def _test_barrier_helper(self, group, group_id, rank):
        WAIT_TIME = 0.3  # seconds

        for dest in group:
            expected_time = torch.DoubleTensor(1).fill_(0.0)
            if dest == rank:
                expected_time.fill_(time.time() + WAIT_TIME)
                dist.broadcast(expected_time, dest, group_id)
                time.sleep(WAIT_TIME + 0.1)  # sleep a little bit longer
                dist.barrier(group_id)
            else:
                dist.broadcast(expected_time, dest, group_id)
                dist.barrier(group_id)
                self.assertGreaterEqual(time.time(), expected_time[0])

        self._barrier()

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
    def test_barrier(self):
        group, group_id, rank = self._init_global_test()
        self._test_barrier_helper(group, group_id, rank)

    @unittest.skipIf(BACKEND == "nccl", "Nccl does not support newGroup")
    @skip_if_small_worldsize
    def test_barrier_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_barrier_helper(group, group_id, rank)

    def _test_broadcast_multigpu_helper(self, group, group_id, rank, rank_to_GPU):
        for src in group:
            expected_tensor = _build_tensor(src + 1)
            tensors = [
                _build_tensor(src + 1, -1).cuda(device=i) for i in rank_to_GPU[rank]
            ]
            if rank == src:
                tensors[0] = expected_tensor.cuda(device=rank_to_GPU[rank][0])

            dist.broadcast_multigpu(tensors, src, group_id)
            for tensor in tensors:
                self.assertEqual(tensor, expected_tensor)
        self._barrier()

    @unittest.skipIf(BACKEND != "nccl", "Only Nccl backend supports broadcast multigpu")
    @skip_if_no_gpu
    def test_broadcast_multigpu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_broadcast_multigpu_helper(group, group_id, rank, rank_to_GPU)

    def _test_all_reduce_multigpu_helper(
        self,
        group,
        group_id,
        rank,
        rank_to_GPU,
        op,
        master_value,
        worker_value,
        expected_value,
    ):
        for src in group:
            if rank == src:
                tensors = [
                    _build_tensor(src + 1, master_value).cuda(device=i)
                    for i in rank_to_GPU[rank]
                ]
            else:
                tensors = [
                    _build_tensor(src + 1, worker_value).cuda(device=i)
                    for i in rank_to_GPU[rank]
                ]

            dist.all_reduce_multigpu(tensors, op, group_id)
            expected_tensor = _build_tensor(src + 1, expected_value)
            for tensor in tensors:
                self.assertEqual(tensor, expected_tensor)

        self._barrier()

    @unittest.skipIf(BACKEND != "nccl", "Only Nccl backend supports allreduce multigpu")
    @skip_if_no_gpu
    def test_all_reduce_multigpu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_all_reduce_multigpu_helper(
            group,
            group_id,
            rank,
            rank_to_GPU,
            dist.reduce_op.SUM,
            2,
            10,
            (2 + 10 * (len(group) - 1)) * len(rank_to_GPU[0]),
        )

    def _test_reduce_multigpu_helper(
        self,
        group,
        group_id,
        rank,
        rank_to_GPU,
        op,
        master_value,
        worker_value,
        expected_value,
    ):
        for src in group:
            if rank == src:
                tensors = [
                    _build_tensor(src + 1, master_value).cuda(device=i)
                    for i in rank_to_GPU[rank]
                ]
                dist.reduce_multigpu(tensors, src, op, group_id)
                expected_tensor = _build_tensor(src + 1, expected_value)
                self.assertEqual(tensors[0], expected_tensor)
            else:
                tensors = [
                    _build_tensor(src + 1, worker_value).cuda(device=i)
                    for i in rank_to_GPU[rank]
                ]
                dist.reduce_multigpu(tensors, src, op, group_id)

        self._barrier()

    @unittest.skipIf(BACKEND != "nccl", "Only Nccl backend supports reduce multigpu")
    @skip_if_no_gpu
    def test_reduce_multigpu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_reduce_multigpu_helper(
            group,
            group_id,
            rank,
            rank_to_GPU,
            dist.reduce_op.SUM,
            2,
            10,
            (2 + 10 * (len(group) - 1)) * len(rank_to_GPU[0]),
        )

    def _test_all_gather_multigpu_helper(self, group, group_id, rank, rank_to_GPU):
        for dest in group:
            tensors = [
                _build_tensor(dest + 1).cuda(device=i) for i in rank_to_GPU[rank]
            ]

            # construct expected output along with
            # a place holder to receive all gather results
            output_tensors = []
            expected_output = []
            output_per_gpu = (
                [_build_tensor(dest + 1, -1)] * len(rank_to_GPU[0]) * len(group)
            )
            expected_per_gpu = (
                [_build_tensor(dest + 1)] * len(rank_to_GPU[0]) * len(group)
            )
            for gpu in rank_to_GPU[rank]:
                output_tensors.append([t.cuda(device=gpu) for t in output_per_gpu])
                expected_output.append([t.cuda(device=gpu) for t in expected_per_gpu])

            dist.all_gather_multigpu(output_tensors, tensors, group_id)
            self.assertEqual(output_tensors, expected_output)

        self._barrier()

    @unittest.skipIf(BACKEND != "nccl", "Only Nccl backend supports allgather multigpu")
    @skip_if_no_gpu
    def test_all_gather_multigpu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_all_gather_multigpu_helper(group, group_id, rank, rank_to_GPU)

    def _create_Net(self):
        return Net()

    def _model_step(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.data += param.grad
                param.grad = None

    def _prepare_dummy_data(self, local_bs):
        # global_bs for DDP should be divisible by WORLD_SIZE
        global_bs = int(WORLD_SIZE) * local_bs
        input_cpu = torch.randn(global_bs, 2)
        target = torch.randn(global_bs, 4)
        loss = nn.MSELoss()
        return global_bs, input_cpu, target, loss

    # END TO END TEST FOR DISTRIBUTEDDATAPARALLEL
    def _test_DDP_helper(self, model, input_var, target, loss):
        model.train()
        output = model(input_var)
        l = loss(output, target)
        l.backward()

    def _assert_equal_param(self, param_gpu, param_DDP):
        self.assertEqual(len(param_gpu), len(param_DDP))
        for p_gpu, p_DDP in zip(param_gpu, param_DDP):
            self.assertEqual(p_gpu, p_DDP)

    def _test_DDP_2iter(
        self, model_base, model_DDP, input, target, loss, local_bs, rank, batch_size
    ):
        for _ in range(2):
            # single cpu/gpu training
            self._test_DDP_helper(model_base, input, target, loss)

            # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
            self._test_DDP_helper(
                model_DDP,
                input[rank * local_bs: (rank + 1) * local_bs],
                target[rank * local_bs: (rank + 1) * local_bs],
                loss,
            )

            # Update weights and run a second iteration to shake out errors
            self._model_step(model_base)
            self._model_step(model_DDP)
            self._assert_equal_param(
                list(model_base.parameters()), list(model_DDP.module.parameters())
            )

            # Shuffle the input so that DDP input is different
            input = input[torch.randperm(batch_size)]

        # Test that saving and loading work
        # gloo serialization doesn't work, see #12261
        if BACKEND != "gloo":
            with tempfile.TemporaryFile() as tmp_file:
                torch.save(model_DDP, tmp_file)
                tmp_file.seek(0)
                saved_model_DDP = torch.load(tmp_file)

    @unittest.skipIf(
        BACKEND != "nccl" and BACKEND != "gloo",
        "Only Nccl & Gloo backend support DistributedDataParallel",
    )
    @skip_if_no_cuda_distributed
    @skip_if_no_gpu
    def test_DistributedDataParallel(self):
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()

        # cpu training setup
        model = self._create_Net()

        # single gpu training setup
        model_gpu = copy.deepcopy(model)
        gpu_subset = list(rank_to_GPU[rank])
        model_gpu.cuda(gpu_subset[0])

        # DDP training setup
        model_DDP = copy.deepcopy(model)
        model_DDP.cuda(gpu_subset[0])
        model_DDP = nn.parallel.deprecated.DistributedDataParallel(
            model_DDP, device_ids=gpu_subset
        )

        # dummy data initialization
        local_bs = len(gpu_subset)
        global_bs, input_cpu, target, loss = self._prepare_dummy_data(local_bs)

        # check two model parameters over 2 iterations
        self._test_DDP_2iter(
            model_gpu,
            model_DDP,
            input_cpu.cuda(gpu_subset[0]),
            target.cuda(gpu_subset[0]),
            loss,
            local_bs,
            rank,
            global_bs,
        )
        self._barrier()

    @unittest.skipIf(
        BACKEND == "nccl", "nccl does not support DistributedDataParallelCPU"
    )
    def test_DistributedDataParallelCPU(self):
        # Run a simple end to end DDP-CPU model, use result of single node
        # model as baseline
        group, group_id, rank = self._init_global_test()

        # cpu training setup
        model_base = self._create_Net()

        # DDP-CPU training setup
        model_DDP = copy.deepcopy(model_base)
        model_DDP = nn.parallel.deprecated.DistributedDataParallelCPU(model_DDP)

        # dummy data initialization
        local_bs = 2
        global_bs, input_cpu, target, loss = self._prepare_dummy_data(local_bs)

        # check two model parameters over 2 iterations
        self._test_DDP_2iter(
            model_base, model_DDP, input_cpu, target, loss, local_bs, rank, global_bs
        )
        self._barrier()


if BACKEND == "tcp" or BACKEND == "gloo" or BACKEND == "nccl":
    WORLD_SIZE = os.environ["WORLD_SIZE"]

    class TestDistBackend(TestCase, _DistTestBase):
        MANAGER_PROCESS_RANK = -1

        @staticmethod
        def manager_join(fn):
            @wraps(fn)
            def wrapper(self):
                if self.rank == self.MANAGER_PROCESS_RANK:
                    self._join_and_reduce(fn)
                else:
                    fn(self)

            return wrapper

        @classmethod
        def setUpClass(cls):
            os.environ["MASTER_ADDR"] = MASTER_ADDR
            os.environ["MASTER_PORT"] = MASTER_PORT
            os.environ["WORLD_SIZE"] = WORLD_SIZE
            for attr in dir(cls):
                if attr.startswith("test"):
                    fn = getattr(cls, attr)
                    setattr(cls, attr, cls.manager_join(fn))

        def setUp(self):
            self.processes = []
            self.rank = self.MANAGER_PROCESS_RANK
            Barrier.init()
            for rank in range(int(WORLD_SIZE)):
                self.processes.append(self._spawn_process(rank))

        def tearDown(self):
            for p in self.processes:
                p.terminate()

        def _spawn_process(self, rank):
            os.environ["RANK"] = str(rank)
            name = "process " + str(rank)
            process = multiprocessing.Process(target=self._run, name=name, args=(rank,))
            process.start()
            return process

        def _run(self, rank):
            self.rank = rank
            try:
                dist.init_process_group(
                    init_method=INIT_METHOD, backend=BACKEND, world_size=int(WORLD_SIZE)
                )
            except RuntimeError as e:
                if "recompile" in e.args[0]:
                    sys.exit(SKIP_IF_BACKEND_UNAVAILABLE)
                    # sys.exit(0)
                raise
            # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
            # We're retreiving a corresponding test and executing it.
            getattr(self, self.id().split(".")[2])()
            sys.exit(0)

        def _join_and_reduce(self, fn):
            skip_ok = (
                getattr(fn, "skip_if_no_cuda_distributed", False) or
                getattr(fn, "skip_if_no_gpu", False) or
                getattr(fn, "skip_if_small_worldsize", False)
            )
            join_timeout = get_timeout(self.id())
            for rank, process in enumerate(self.processes):
                process.join(join_timeout)
                self.assertFalse(
                    process.is_alive(),
                    "Timeout waiting for rank %d to terminate" % rank)

            first_process = self.processes[0]
            for p in self.processes:
                self.assertEqual(p.exitcode, first_process.exitcode)

            if first_process.exitcode == SKIP_IF_BACKEND_UNAVAILABLE:
                raise unittest.SkipTest("Compiled without the " + BACKEND + " backend")

            if skip_ok:
                # do this first so we don't give an error message about
                # mismatched exit codes if the first isn't valid
                assert (
                    first_process.exitcode == 0 or
                    first_process.exitcode == SKIP_IF_NO_CUDA_EXIT_CODE or
                    first_process.exitcode == SKIP_IF_NO_GPU_EXIT_CODE or
                    first_process.exitcode == SKIP_IF_SMALL_WORLDSIZE_EXIT_CODE
                )

                if first_process.exitcode == SKIP_IF_NO_CUDA_EXIT_CODE:
                    raise unittest.SkipTest("cuda is not available")
                if first_process.exitcode == SKIP_IF_NO_GPU_EXIT_CODE:
                    raise unittest.SkipTest(
                        "One unique gpu per process is not available"
                    )
                if first_process.exitcode == SKIP_IF_SMALL_WORLDSIZE_EXIT_CODE:
                    raise unittest.SkipTest("worldsize is too small to run group tests")

            self.assertEqual(first_process.exitcode, 0)


elif BACKEND == "mpi":
    WORLD_SIZE = os.environ["WORLD_SIZE"]
    dist.init_process_group(init_method=INIT_METHOD, backend="mpi")

    class TestMPI(TestCase, _DistTestBase):
        pass


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
