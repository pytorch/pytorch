import copy
from collections import namedtuple
import itertools
import random
import math
import os
import sys
import time
import tempfile
import unittest
from contextlib import contextmanager, suppress
from datetime import timedelta
from functools import reduce
from typing import Union, NamedTuple

import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.distributed_c10d import _get_default_group, AllreduceOptions, GroupMember
from torch.testing._internal.common_utils import FILE_SCHEMA
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
    initialize_temp_directories,
    cleanup_temp_dir,
    simple_sparse_reduce_tests,
    skip_if_rocm,
    skip_if_small_worldsize,
    skip_if_lt_x_gpu,
    skip_if_no_gpu,
    require_n_gpus_for_nccl_backend,
    requires_nccl_version,
    captured_output,
)
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

class Foo:
    def __init__(self, x):
        # Can be tensor or int
        self.x = x

    def __eq__(self, other):
        def eq(value, other):
            if isinstance(value, torch.Tensor):
                return torch.equal(value, other)
            return value == other

        for attr, value in self.__dict__.items():
            other_value = other.__dict__[attr]
            if not eq(value, other_value):
                return False
        return True

f = Foo(10)
f.bar = 1

foo_cpu_tensor = Foo(torch.randn(3, 3))


COLLECTIVES_OBJECT_TEST_LIST = [
    {"key1": 3, "key2": 4, "key3": {"nested": True}},
    f,
    foo_cpu_tensor,
    "foo",
    [1, 2, True, "string", [4, 5, "nested"]],
]

# Allowlist of distributed backends where profiling collectives is supported.
PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.NCCL,
    dist.Backend.GLOO,
]

# Allowlist of distributed backends where profiling is supported with use_cuda=True
CUDA_PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.GLOO
]

# Dummy NamedTuple data structures to test DDP support for NamedTuple types.
EXPECTED_FIELDS = ("a", "b")
TestNamedTupleInput_0 = namedtuple("NamedTuple", EXPECTED_FIELDS)

class TestNamedTupleInput_1(NamedTuple):
    a: torch.tensor
    b: torch.tensor

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

BACKEND = os.environ["BACKEND"]
INIT_METHOD = os.getenv("INIT_METHOD", "env://")

DEFAULT_TIMEOUT = 300
CUSTOMIZED_TIMEOUT = {"test_DistributedDataParallel": 500}


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
        self.no_grad_param = nn.Parameter(torch.tensor([2, 2]).long(),
                                          requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.fc1 = nn.Linear(1000, 2000, bias=False)
        self.fc2 = nn.Linear(2000, 500, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Task(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        return self.p + x


class BatchNormNet(nn.Module):

    def __init__(self):
        super(BatchNormNet, self).__init__()
        self.fc1 = nn.Linear(2, 40, bias=False)
        self.bn = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(40, 4, bias=False)

    def forward(self, x):
        x = torch.reshape(self.fc1(x), (-1, 4, 10))
        x = self.bn(x)
        x = torch.reshape(x, (-1, 40))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


DDP_NET = Net()
BN_NET = BatchNormNet()
ONLY_SBN_NET = nn.SyncBatchNorm(2, momentum=0.99)

def get_timeout(test_id):
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        return DEFAULT_TIMEOUT


def require_backend(backends):
    if BACKEND not in backends:
        return unittest.skip("Test requires backend to be one of %s" % backends)
    return lambda func: func


def require_backends_available(backends):
    def check(backend):
        if backend == dist.Backend.GLOO:
            return dist.is_gloo_available()
        if backend == dist.Backend.NCCL:
            return dist.is_nccl_available()
        if backend == dist.Backend.MPI:
            return dist.is_mpi_available()
        return False
    if not all(check(dist.Backend(backend)) for backend in backends):
        return unittest.skip(
            "Test requires backends to be available %s" % backends)
    return lambda func: func


def require_world_size(world_size):
    if int(os.environ["WORLD_SIZE"]) < world_size:
        return unittest.skip("Test requires world size of %d" % world_size)
    return lambda func: func


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
    TEMP_DIR = os.environ["TEMP_DIR"]
    lockfile = os.path.join(TEMP_DIR, "lockfile")
    with open(lockfile, "w") as lf:
        try:
            if sys.platform == 'win32':
                msvcrt.locking(lf.fileno(), msvcrt.LK_RLCK, 1)
                yield
            else:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            if sys.platform == 'win32':
                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            lf.close()


def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, size, size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, size, size, dtype=dtype).fill_(value).cuda(device_id)


def _build_multidim_tensor(dim, dim_size, value=None, dtype=torch.float):
    if value is None:
        value = size
    return torch.empty(size=[dim_size for _ in range(dim)], dtype=dtype).fill_(value)


class Barrier(object):
    barrier_id = 0

    @classmethod
    def init(cls):
        cls.barrier_id = 0
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))

    @classmethod
    def sync(cls, wait_for=None, timeout=10):
        if wait_for is None:
            wait_for = dist.get_world_size()
        cls.barrier_id += 1
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
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
            if arrived == wait_for:
                break

            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            time.sleep(0.1)


class TestDistBackend(MultiProcessTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        # os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        super().setUpClass()

    def setUp(self):
        super().setUp()
        # initialize temp directories
        initialize_temp_directories()
        # initialize Barrier
        Barrier.init()

    def tearDown(self):
        cleanup_temp_dir()
        super().tearDown()

    @property
    def init_method(self):
        return "{}{file_name}".format(FILE_SCHEMA, file_name=self.file_name)

    @classmethod
    def _run(cls, rank, test_name, file_name):
        if BACKEND == 'nccl' and not torch.cuda.is_available():
            sys.exit(TEST_SKIPS['no_cuda'].exit_code)
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        if torch.cuda.is_available() and torch.cuda.device_count() < int(self.world_size):
            sys.exit(TEST_SKIPS['multi-gpu'].exit_code)
        try:
            timeout = timedelta(seconds=60)
            dist.init_process_group(
                init_method=self.init_method,
                backend=BACKEND,
                world_size=int(self.world_size),
                rank=self.rank,
                timeout=timeout,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        self._barrier()

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retreiving a corresponding test and executing it.
        getattr(self, test_name)()
        self._barrier()
        dist.destroy_process_group()
        sys.exit(0)

    # Needed since MultiProcessTestCase assumes a world_size of 4, but we
    # run these tests under other various world_sizes.
    @property
    def world_size(self):
        return os.environ["WORLD_SIZE"]


class DistributedTest:
    class _DistTestBase:
        def _barrier(self, *args, **kwargs):
            Barrier.sync(*args, **kwargs)

        def _init_group_test(self, **kwargs):
            group = [1, 2]
            group_id = dist.new_group(group, **kwargs)
            rank = dist.get_rank()
            if rank not in group:
                return ([], None, rank)

            return (group, group_id, rank)

        def _init_full_group_test(self, **kwargs):
            group = list(range(0, dist.get_world_size()))
            group_id = dist.new_group(**kwargs)
            rank = dist.get_rank()
            return (group, group_id, rank)

        def _init_global_test(self):
            group = list(range(0, dist.get_world_size()))
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

            # If rank is lesser than or equal to number of available GPU's
            # then each rank can be mapped to corresponding GPU.
            nGPUs_per_process = 1
            if world_size > nGPUs:
                nGPUs_per_process = nGPUs // world_size
            rank_to_GPU = {
                i: list(
                    visible_devices[i * nGPUs_per_process: (i + 1) * nGPUs_per_process]
                )
                for i in range(world_size)
            }
            return rank_to_GPU

        def test_dump_DDP_relevant_env_vars(self):
            with captured_output() as (out, _):
                _dump_DDP_relevant_env_vars()
                lines = out.getvalue().splitlines()

            def format_line(var):
                return "env:%s=%s" % (var, os.environ[var] if var in os.environ else "N/A")

            # Check relevant env vars
            vars = [
                "MASTER_ADDR",
                "MASTER_PORT",
                "WORLD_SIZE",
                "NCCL_TOPO_DUMP_FILE",  # N/A
            ]
            for var in vars:
                line = format_line(var)
                self.assertIn(line, lines)
            # Check irrelevant env vars
            vars = [
                "xxx",
                "yyy",
                "zzz",
            ]
            for var in vars:
                line = format_line(var)
                self.assertNotIn(line, lines)

        # GET RANK
        def test_get_rank(self):
            test_dir = os.path.join(os.environ["TEMP_DIR"], "test_dir")
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

        def test_get_backend(self):
            if dist.get_world_size() > 2:
                group = [1, 2]
            else:
                group = [0, 1]
            group_id = dist.new_group(group)
            backend_str = BACKEND.lower()
            self.assertEqual(dist.get_backend(), backend_str)
            if dist.get_rank() in group:
                self.assertEqual(dist.get_backend(group_id), backend_str)
            else:
                with self.assertRaisesRegex(RuntimeError, "Invalid process group specified"):
                    dist.get_backend(group_id)

        def test_Backend_enum_class(self):
            # test parsing
            backend = BACKEND.lower()
            self.assertEqual(dist.Backend(BACKEND.upper()), backend)
            self.assertEqual(dist.Backend(BACKEND), backend)
            with self.assertRaisesRegex(ValueError, "Invalid backend: 'undefined'"):
                dist.Backend("undefined")
            with self.assertRaisesRegex(ValueError, "Invalid backend: 'xYz'"):
                dist.Backend("xYz")
            with self.assertRaises(ValueError):
                dist.Backend(None)
            with self.assertRaises(ValueError):
                dist.Backend(3)
            with self.assertRaises(ValueError):
                dist.Backend(["gloo"])

        # Test destroy
        def test_destroy_group(self):
            if dist.get_world_size() > 2:
                group = [1, 2]
            else:
                group = [0, 1]
            group_id = dist.new_group(group)
            self._barrier()
            dist.destroy_process_group(group_id)

        # Test get rank and size of group
        def test_get_rank_size_group(self):
            if dist.get_world_size() > 2:
                group = [1, 2]
            else:
                group = [0, 1]
            group_id = dist.new_group(group)
            if dist.get_rank() in group:
                self.assertEqual(dist.get_world_size(group_id), 2)
                self.assertTrue(dist.get_rank(group_id) in list(range(2)))
            else:
                self.assertEqual(dist.get_world_size(group_id), -1)
                self.assertEqual(dist.get_rank(group_id), -1)

        # Test destroy full groups
        def test_destroy_full_group(self):
            _, group_id, _ = self._init_full_group_test()
            self._barrier()
            dist.destroy_process_group(group_id)

        # Test get rank and size of full group
        def test_get_rank_size_full_group(self):
            _, group_id, _ = self._init_full_group_test()
            self.assertEqual(dist.get_world_size(group_id), dist.get_world_size())
            self.assertEqual(dist.get_rank(group_id), dist.get_rank())

        def _test_barrier_timeout(self, group_id, timeout):
            local_rank = dist.get_rank(group_id)

            # Only execute barrier on rank == 0, causing it to timeout
            if local_rank == 0:
                expected_time = time.time() + timeout.total_seconds()
                with self.assertRaisesRegex(Exception, " (Timed out|closed|timeout) "):
                    dist.barrier(group_id)
                self.assertGreaterEqual(time.time(), expected_time)
            else:
                time.sleep(timeout.total_seconds())

        @unittest.skipIf(BACKEND != "gloo", "Only gloo backend supports timeouts")
        @unittest.skipIf(
            not INIT_METHOD.startswith("file://"),
            "Requires file:// initialization method. " +
            "Both tcp:// and env:// rely on the TCP store for which "
            "reinitialization has proven racy."
        )
        def test_barrier_timeout_global(self):
            dist.destroy_process_group()

            # Explicitly pass world size to the barrier because we've
            # just destroyed any state in torch.distributed.
            self._barrier(wait_for=int(os.environ["WORLD_SIZE"]))

            # Reinitialize global process group
            timeout = timedelta(seconds=1)
            dist.init_process_group(
                init_method=INIT_METHOD,
                backend=BACKEND,
                world_size=int(os.environ["WORLD_SIZE"]),
                rank=self.rank,
                timeout=timeout,
            )
            self._test_barrier_timeout(dist.group.WORLD, timeout)

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND != "gloo", "Only gloo backend supports timeouts")
        def test_barrier_timeout_group(self):
            timeout = timedelta(seconds=5)
            _, group_id, _ = self._init_group_test(timeout=timeout)
            if group_id is not None:
                self._test_barrier_timeout(group_id, timeout)

        @unittest.skipIf(BACKEND != "gloo", "Only gloo backend supports timeouts")
        def test_barrier_timeout_full_group(self):
            timeout = timedelta(seconds=1)
            _, group_id, _ = self._init_full_group_test(timeout=timeout)
            if group_id is not None:
                self._test_barrier_timeout(group_id, timeout)

        # This test helper can only be used when using the Gloo or NCCL backend
        # **and** both the Gloo and NCCL backends are available.
        # See the @skip annotations below.
        def _test_group_override_backend(self, initializer):
            if BACKEND == "gloo":
                new_backend = "nccl"
            if BACKEND == "nccl":
                new_backend = "gloo"

            group, group_id, rank = initializer(backend=new_backend)
            if group_id is None:
                return

            if new_backend == "gloo":
                self.assertTrue(isinstance(group_id, dist.ProcessGroupGloo))
            if new_backend == "nccl":
                self.assertTrue(isinstance(group_id, dist.ProcessGroupNCCL))

            self.assertEqual(rank, group[dist.get_rank(group_id)])
            self.assertEqual(len(group), dist.get_world_size(group_id))

            # Pin device (so we avoid NCCL race conditions/deadlocks).
            group_rank = dist.get_rank(group_id)
            torch.cuda.set_device(group_rank)

            # Run broadcast of CUDA tensor (so it works for both Gloo and NCCL).
            tensor = _build_tensor(2, value=group_rank).cuda()
            dist.broadcast(tensor, src=group[0], group=group_id)
            self.assertEqual(_build_tensor(2, value=0), tensor.to("cpu"))

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @require_world_size(3)
        @skip_if_lt_x_gpu(2)
        def test_backend_group(self):
            self._test_group_override_backend(self._init_group_test)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(3)
        def test_backend_full_group(self):
            self._test_group_override_backend(self._init_full_group_test)

        # NCCL Batch SEND RECV
        @skip_if_no_gpu
        @unittest.skipIf(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_nccl(self):
            self._barrier()
            rank = dist.get_rank()
            rank_to_GPU = self._init_multigpu_helper()
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            p2p_op_list = []

            for val in ["1", "0"]:
                os.environ["NCCL_BLOCKING_WAIT"] = val
                for src in range(0, dist.get_world_size()):
                    send_tensor = _build_tensor(rank + 1, device_id=device_id)
                    recv_tensor = _build_tensor(src + 1, value=-1, device_id=device_id)
                    recv_op = dist.P2POp(dist.irecv, recv_tensor, src)
                    p2p_op_list.append(recv_op)
                    send_op = dist.P2POp(dist.isend, send_tensor, src)
                    p2p_op_list.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()

            self._barrier()

        @skip_if_no_gpu
        @unittest.skipIf(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_self_nccl(self):
            self._barrier()
            rank = dist.get_rank()
            rank_to_GPU = self._init_multigpu_helper()
            device_id = rank_to_GPU[rank][0]
            p2p_op_list = []

            if rank == 0:
                send_tensor = _build_tensor(rank + 1, device_id=device_id)
                recv_tensor = _build_tensor(rank + 1, value=-1, device_id=device_id)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, 0)
                p2p_op_list.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()

            self._barrier()

        @skip_if_no_gpu
        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_no_rank_zero_nccl(self):
            self._barrier()
            rank = dist.get_rank()
            rank_to_GPU = self._init_multigpu_helper()
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            p2p_op_list = []

            if rank == 1:
                peer = 2
            elif rank == 2:
                peer = 1

            if rank in [1, 2]:
                send_tensor = _build_tensor(rank + 1, device_id=device_id)
                recv_tensor = _build_tensor(peer + 1, value=-1, device_id=device_id)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, peer)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, peer)
                p2p_op_list.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()


            self._barrier()

        # GLOO Batch SEND RECV CPU
        @unittest.skipIf(BACKEND != "gloo", "GLOO Batch Send Recv CPU")
        def test_batch_isend_irecv_gloo(self):
            self._barrier()
            rank = dist.get_rank()
            p2p_op_list = []

            for src in range(0, dist.get_world_size()):
                if src == rank:
                    continue
                send_tensor = _build_tensor(rank + 1)
                recv_tensor = _build_tensor(src + 1, value=-1)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, src)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, src)
                p2p_op_list.append(send_op)

            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()

            self._barrier()

        # GLOO Batch SEND RECV CPU with provided tags
        @unittest.skipIf(BACKEND != "gloo", "GLOO Batch Send Recv CPU")
        def test_batch_isend_irecv_gloo_tags(self):
            self._barrier()
            rank = dist.get_rank()
            p2p_op_list = []

            for src in range(0, dist.get_world_size()):
                if src == rank:
                    continue
                send_tensor = _build_tensor(rank + 1)
                recv_tensor = _build_tensor(src + 1, value=-1)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, src, tag=src)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, src, tag=rank)
                p2p_op_list.append(send_op)

            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()

            self._barrier()

        # NCCL Batch SEND RECV Tensor Error
        @unittest.skipIf(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_tensor_err(self):
            self._barrier()
            rank = dist.get_rank()
            if rank == 0:
                rank_to_GPU = self._init_multigpu_helper()
                device_id = rank_to_GPU[rank][0]
                with self.assertRaisesRegex(
                    RuntimeError, "Tensors must be CUDA and dense"
                ):
                    send_tensor = _build_tensor(rank + 1)
                    send_op = dist.P2POp(dist.isend, send_tensor, 1)
                    req = dist.batch_isend_irecv([send_op])
                    req.wait()

        # NCCL Batch SEND RECV Op Error
        @unittest.skipIf(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_op_err(self):
            self._barrier()
            rank = dist.get_rank()
            if rank == 0:
                rank_to_GPU = self._init_multigpu_helper()
                device_id = rank_to_GPU[rank][0]
                with self.assertRaisesRegex(
                    RuntimeError, "^Invalid ``op``"
                ):
                    send_tensor = _build_tensor(rank + 1, device_id=device_id)
                    send_op = dist.P2POp(dist.broadcast, send_tensor, 1)
                    req = dist.batch_isend_irecv([send_op])
                    req.wait()

        # NCCL Batch SEND RECV p2p_op_list Error
        @unittest.skipIf(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_op_list_err(self):
            self._barrier()
            rank = dist.get_rank()
            if rank == 0:
                rank_to_GPU = self._init_multigpu_helper()
                device_id = rank_to_GPU[rank][0]
                with self.assertRaisesRegex(
                    RuntimeError, "^Invalid ``p2p_op_list``"
                ):
                    send_tensor = _build_tensor(rank + 1)
                    req = dist.batch_isend_irecv([1, 2])
                    req.wait()

        # NCCL Batch SEND RECV Mixed Backend Error
        @unittest.skipIf(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_mixed_backend_err(self):
            self._barrier()
            rank = dist.get_rank()
            rank_to_GPU = self._init_multigpu_helper()
            device_id = rank_to_GPU[rank][0]
            group_gloo = dist.new_group(ranks=[0, 1], backend="gloo")
            group_nccl = dist.new_group(ranks=[0, 1], backend="nccl")
            if rank == 0:
                with self.assertRaisesRegex(
                    RuntimeError, "All groups need to use the same backend"
                ):
                    send_tensor = _build_tensor(rank + 1)
                    send_op_gloo = dist.P2POp(dist.isend, send_tensor, 1, group_gloo)
                    send_op_nccl = dist.P2POp(dist.isend, send_tensor, 1, group_nccl)
                    req = dist.batch_isend_irecv([send_op_gloo, send_op_nccl])
                    req.wait()

        # NCCL SEND RECV
        @skip_if_no_gpu
        @unittest.skipIf(BACKEND != "nccl", "NCCL Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_send_recv_nccl(self):
            rank = dist.get_rank()
            rank_to_GPU = self._init_multigpu_helper()
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)

            tensor = _build_tensor(rank + 1, device_id=device_id)

            for src in range(0, dist.get_world_size()):
                if src == rank:
                    # Send mode
                    for dst in range(0, dist.get_world_size()):
                        if dst == rank:
                            continue
                        dist.send(tensor, dst)
                else:
                    # Recv mode
                    expected_tensor = _build_tensor(src + 1)
                    output_tensor = _build_tensor(src + 1, value=-1, device_id=device_id)
                    dist.recv(output_tensor, src)
                    self.assertEqual(output_tensor, expected_tensor)

            self._barrier()

        # SEND RECV
        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support send/recv")
        def test_send_recv(self):
            rank = dist.get_rank()
            tensor = _build_tensor(rank + 1)

            for src in range(0, dist.get_world_size()):
                if src == rank:
                    # Send mode
                    for dst in range(0, dist.get_world_size()):
                        if dst == rank:
                            continue
                        dist.send(tensor, dst)
                else:
                    # Recv mode
                    expected_tensor = _build_tensor(src + 1)
                    output_tensor = _build_tensor(src + 1, value=-1)
                    dist.recv(output_tensor, src)
                    self.assertEqual(output_tensor, expected_tensor)

            self._barrier()

        # SEND RECV ANY SOURCE
        @unittest.skipIf(
            BACKEND == "nccl", "Nccl does not support send/recv from any source"
        )
        def test_send_recv_any_source(self):
            rank = dist.get_rank()
            tensor = _build_tensor(10, value=rank)
            recv_ranks = list()
            irecv_ranks = list()

            for dst in range(0, dist.get_world_size()):
                if dst == rank:
                    # Recv mode
                    for dst in range(0, dist.get_world_size()):
                        if dst == rank:
                            continue

                        for recv in ["recv", "irecv"]:
                            output_tensor = _build_tensor(10, value=-1)

                            if recv == "recv":
                                sender = dist.recv(output_tensor)
                                recv_ranks.append(sender)
                            elif recv == "irecv":
                                work = dist.irecv(output_tensor)
                                work.wait()
                                sender = work._source_rank()
                                irecv_ranks.append(sender)

                            # Assert the scalar value "sender" that should be
                            # equal to the rank of the sender is equal to all
                            # values in the received tensor.
                            self.assertTrue(output_tensor.eq(sender).all())
                else:
                    # Send mode
                    dist.send(tensor, dst)  # recv
                    dist.send(tensor, dst)  # irecv

            # Each rank would have 2 * (world_size - 1) sends, verify that
            # globally we receive the same amount on the other end.
            recv_ranks_tensor = torch.cat((torch.tensor(recv_ranks), torch.tensor(irecv_ranks)), 0)
            global_recv_ranks = [torch.empty_like(recv_ranks_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(global_recv_ranks, recv_ranks_tensor)
            global_recv_ranks_list = []
            for tensor in global_recv_ranks:
                global_recv_ranks_list += tensor.tolist()

            from itertools import groupby
            global_recv_ranks_list.sort()
            frequency = [len(list(group)) for key, group in groupby(global_recv_ranks_list)]
            self.assertEqual(dist.get_world_size(), len(frequency))
            self.assertEqual([2 * (dist.get_world_size() - 1)] * dist.get_world_size(), frequency)
            self._barrier()

        # SEND RECV WITH TAG
        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support send/recv")
        def test_send_recv_with_tag(self):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            tensor = _build_tensor(10, value=rank)

            for dst in range(0, world_size):
                if dst == rank:
                    # Recv mode
                    for src in range(0, world_size):
                        if src == rank:
                            continue
                        output_tensor = _build_tensor(10, value=-1)
                        dist.recv(output_tensor, src, tag=src)
                        self.assertTrue(output_tensor.eq(src).all())
                else:
                    # Send mode
                    dist.send(tensor, dst, tag=rank)

        # ISEND
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
            self, group, group_id, rank, cuda=False, rank_to_GPU=None, with_options=False
        ):
            for dtype, value, requires_cuda in [
                (torch.float, -1e-10, False),
                (torch.double, -1e-100, False),
                (torch.half, -0.1, True),
                (torch.int8, -2, False),
                (torch.uint8, 129, False),
                (torch.int, -1e5, False),
                (torch.long, -1e15, False),
            ]:
                if requires_cuda and not cuda:
                    continue
                for src in group:
                    expected_tensor = _build_tensor(src + 1, value, dtype)
                    if cuda:
                        expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
                    if rank == src:
                        if with_options:
                            opts = dist.BroadcastOptions()
                            opts.rootTensor = 0
                            opts.rootRank = src
                            self.call_dist_op(":broadcast", True, group_id.broadcast, [expected_tensor], opts)
                        else:
                            self.call_dist_op(":broadcast", False, dist.broadcast, expected_tensor, src, group_id)
                    else:
                        tensor = _build_tensor(src + 1, -1, dtype)
                        if cuda:
                            tensor = tensor.cuda(rank_to_GPU[rank][0])
                        if with_options:
                            opts = dist.BroadcastOptions()
                            opts.rootTensor = 0
                            opts.rootRank = src
                            self.call_dist_op(":broadcast", True, group_id.broadcast, [tensor], opts)
                        else:
                            self.call_dist_op(":broadcast", False, dist.broadcast, tensor, src, group_id)
                        self.assertEqual(tensor.size(), expected_tensor.size())
                        self.assertEqual(tensor.ne(expected_tensor).max(), torch.tensor(False))

            self._barrier()

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_broadcast(self):
            group, group_id, rank = self._init_global_test()
            self._test_broadcast_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "gloo" and BACKEND != "nccl",
            "Only Gloo and Nccl backend supports CUDA allReduce",
        )
        @skip_if_no_gpu
        def test_broadcast_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_broadcast_helper(group, group_id, rank, True, rank_to_GPU)

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_broadcast_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_broadcast_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_broadcast_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_broadcast_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "nccl",
            "Only NCCL backend supports high priority stream",
        )
        @skip_if_no_gpu
        def test_nccl_high_priority_stream(self):
            group, _, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()

            new_port = str(MASTER_PORT + 1)
            os.environ['MASTER_PORT'] = new_port
            gen_iterator = dist.rendezvous('env://', rank, dist.get_world_size())
            store, rank, size = next(gen_iterator)
            store = dist.PrefixStore(new_port, store)

            opts = dist.ProcessGroupNCCL.Options()
            opts.is_high_priority = False
            group_id = dist.ProcessGroupNCCL(store, rank, size, opts)

            self._test_broadcast_helper(group, group_id, rank, True, rank_to_GPU, True)

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
                tensor = _build_tensor(src + 1).fill_(master_value if rank == src else worker_value)
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                self.call_dist_op(":reduce", False, dist.reduce, tensor, src, op, group_id)
                if rank == src:
                    self.assertEqual(tensor, _build_tensor(src + 1, expected_value))

            self._barrier()

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_sum(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @unittest.skipIf(BACKEND != "nccl", "Only Nccl supports CUDA reduce")
        @skip_if_no_gpu
        def test_reduce_sum_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + 10 * (len(group) - 1),
                True,
                rank_to_GPU,
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_product(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_min(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_helper(group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_max(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_helper(group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        @skip_if_small_worldsize
        def test_reduce_group_sum(self):
            group, group_id, rank = self._init_group_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        @skip_if_small_worldsize
        def test_reduce_group_product(self):
            group, group_id, rank = self._init_group_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        @skip_if_small_worldsize
        def test_reduce_group_min(self):
            group, group_id, rank = self._init_group_test()
            self._test_reduce_helper(group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        @skip_if_small_worldsize
        def test_reduce_group_max(self):
            group, group_id, rank = self._init_group_test()
            self._test_reduce_helper(group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_full_group_sum(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_full_group_product(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_full_group_min(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_reduce_helper(group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_full_group_max(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_reduce_helper(group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10)

        # REDUCE TWICE
        def _test_reduce_twice_helper(
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
                tensors = [_build_tensor(src + 1).fill_(master_value if rank == src else worker_value) for i in range(2)]
                if cuda:
                    for i in range(2):
                        tensors[i] = tensors[i].cuda(rank_to_GPU[rank][0])
                self.call_dist_op(":reduce", False, dist.reduce, tensors[0], src, op, group_id,
                                  secondary_op_call=lambda: dist.reduce(tensors[1], src, op, group_id))
                if rank == src:
                    for tensor in tensors:
                        self.assertEqual(tensor, _build_tensor(src + 1, expected_value))

            self._barrier()

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_reduce_sum_twice(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_twice_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @unittest.skipIf(BACKEND != "nccl", "Only Nccl supports CUDA reduce")
        @skip_if_no_gpu
        def test_reduce_sum_cuda_twice(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_reduce_twice_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + 10 * (len(group) - 1),
                True,
                rank_to_GPU,
            )


        @skip_if_no_gpu
        @require_backend({"gloo", "nccl"})
        def test_all_reduce_result_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            for src in group:
                if rank == src:
                    tensor = _build_tensor(src + 1, 2)
                else:
                    tensor = _build_tensor(src + 1, 10)
                tensor = tensor.cuda(rank_to_GPU[rank][0])

                opts = AllreduceOptions()
                opts.reduceOp = dist.ReduceOp.SUM

                if group_id == GroupMember.WORLD:
                    work = _get_default_group().allreduce([tensor], opts)
                else:
                    work = group_id.allreduce([tensor], opts)


                if BACKEND == "gloo":
                    # Calling result right the work is finished should throw exception.
                    # Here we have a race condition, we may not assume the work is not
                    # finished by the time we run next lines.
                    try:
                        with self.assertRaisesRegex(
                                RuntimeError,
                                "Work needs to be completed before calling result"):
                            work.result()
                    except AssertionError:
                        # Exception was not raised, ensure is_completed()
                        self.assertTrue(work.is_completed())

                    work.wait()
                    result = work.result()
                else:
                    # In case of NCCL we should be able to retrieve pointer to the result
                    # even before work is finished.
                    result = work.result()
                    work.wait()

                expected_value = 2 + (10 * (len(group) - 1))
                self.assertEqual(result, [_build_tensor(src + 1, expected_value)])
            self._barrier()

        def call_dist_op(
            self,
            profiling_title_postfix,
            is_async,
            op,
            *args,
            expect_event=True,
            secondary_op_call=None,
            profile_cuda=False,
            **kwargs,
        ):
            op_calls = [lambda: op(*args, **kwargs)]
            if secondary_op_call is not None:
                op_calls.append(secondary_op_call)

            with torch.autograd.profiler.profile(use_cuda=profile_cuda) as prof:
                works = [op_call() for op_call in op_calls]
                if is_async:
                    for work in works:
                        work.wait()

            def get_event(postfix):
                return [event for event in prof.function_events if event.name.endswith(postfix)]

            if expect_event and dist.get_backend() in PROFILING_SUPPORTED_BACKENDS:
                events = get_event(profiling_title_postfix)
                self.assertEqual(len(events), len(op_calls))
                for e in events:
                    self.assertEqual(e.count, 1)
                    self.assertGreaterEqual(e.cpu_time, 0)

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
            dtype=torch.float,
            async_op=False,
        ):
            for src in group:
                curr_value = master_value if rank == src else worker_value

                tensor = _build_tensor(src + 1, dtype=dtype).fill_(curr_value)
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                self.call_dist_op(":all_reduce", async_op, dist.all_reduce, tensor, op, group_id, async_op=async_op)
                # Currently, only Gloo backend has profiling tested with CUDA enabled.
                # Only run cuda profiling test for one rank to speed up since
                # running with different src_rank does not affect the correctness.
                if (
                    src == 0
                    and cuda
                    and dist.get_backend() in CUDA_PROFILING_SUPPORTED_BACKENDS
                ):
                    self.call_dist_op(
                        ":all_reduce",
                        async_op,
                        dist.all_reduce,
                        tensor,
                        op,
                        group_id,
                        async_op=async_op,
                        profile_cuda=True,
                    )

            self._barrier()

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_sum(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_sum_async(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
                async_op=True
            )

        @unittest.skipIf(
            BACKEND != "gloo" and BACKEND != "nccl",
            "Only Gloo and NCCL backends will have CUDA allReduce tested",
        )
        @skip_if_no_gpu
        def test_all_reduce_sum_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
                True,
                rank_to_GPU,
            )

        @unittest.skipIf(
            BACKEND != "gloo" and BACKEND != "nccl",
            "Only Gloo and NCCL backends will have CUDA allReduce tested",
        )
        @skip_if_no_gpu
        def test_all_reduce_sum_cuda_async(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
                True,
                rank_to_GPU,
                async_op=True
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_sum_complex(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                complex(2, 3),
                complex(10, 11),
                complex(2, 3) + (complex(10, 11) * (len(group) - 1)),
                dtype=torch.cfloat,
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_complex_unsupported_ops(self):
            unsupported_ops = [dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.PRODUCT,
                               dist.ReduceOp.BAND, dist.ReduceOp.BOR, dist.ReduceOp.BXOR]
            group, group_id, rank = self._init_global_test()
            for unsupported_op in unsupported_ops:
                with self.assertRaisesRegex(RuntimeError, "all_reduce does not support"):
                    dist.all_reduce(_build_tensor(1, dtype=torch.cfloat), unsupported_op, group_id)

        @unittest.skipIf(
            BACKEND != "gloo" and BACKEND != "nccl",
            "Only Gloo and NCCL backends will have CUDA allReduce tested",
        )
        @skip_if_no_gpu
        def test_all_reduce_sum_cuda_complex(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                complex(2, 3),
                complex(10, 11),
                complex(2, 3) + (complex(10, 11) * (len(group) - 1)),
                True,
                rank_to_GPU,
                dtype=torch.cfloat,
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_product(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_min(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_max(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10
            )

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_group_sum(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_group_product(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
            )

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_group_min(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1
            )

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_group_max(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_full_group_sum(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_full_group_product(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2),
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_full_group_min(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_full_group_max(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10
            )

        # SPARSE ALL REDUCE
        def _test_sparse_all_reduce_sum(self, fn):
            group, group_id, rank = self._init_global_test()

            tests = simple_sparse_reduce_tests(
                rank,
                dist.get_world_size(),
                num_inputs=1)
            for (inputs, outputs) in tests:
                tensors = [fn(input) for input in inputs]
                dist.all_reduce(tensors[0], dist.ReduceOp.SUM, group_id)
                self.assertEqual(tensors[0], outputs[0])

        @unittest.skipIf(BACKEND != "gloo", "Only Gloo backend support sparse all reduce")
        def test_sparse_all_reduce_sum(self):
            self._test_sparse_all_reduce_sum(lambda t: t)

        @unittest.skipIf(BACKEND != "gloo", "Only Gloo backend support sparse all reduce")
        @skip_if_no_gpu
        def test_sparse_all_reduce_sum_cuda(self):
            self._test_sparse_all_reduce_sum(lambda t: t.clone().cuda())

        # ALL REDUCE - COALESCED
        @staticmethod
        def _all_reduce_coalesced_sum_test_cases(group_size):
            return (
                [2, 3, complex(2, 3)],
                [10, 11, complex(10, 11)],
                [2 + 10 * (group_size - 1), 3 + 11 * (group_size - 1), complex(2, 3) + complex(10, 11) * (group_size - 1)],
                [torch.float, torch.float, torch.cfloat],
            )

        @staticmethod
        def _all_reduce_coalesced_product_test_cases(group_size):
            return (
                [1, 2],
                [3, 4],
                [1 * 3 ** (group_size - 1), 2 * 4 ** (group_size - 1)],
                [torch.float, torch.float],
            )

        @staticmethod
        def _all_reduce_coalesced_min_test_cases(group_size):
            return (
                [1, 4],
                [2, 3],
                [1, 3],
                [torch.float, torch.float],
            )

        @staticmethod
        def _all_reduce_coalesced_max_test_cases(group_size):
            return (
                [1, 4],
                [2, 3],
                [2, 4],
                [torch.float, torch.float],
            )

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_reduce_coalesced_max_complex_unsupported(self):
            group, group_id, rank = self._init_global_test()
            with self.assertRaisesRegex(RuntimeError, "all_reduce does not support"):
                dist.all_reduce_coalesced([_build_tensor(1, dtype=torch.cfloat)], dist.ReduceOp.MAX, group_id)

        def _test_all_reduce_coalesced_helper(
            self,
            group,
            group_id,
            rank,
            op,
            cuda=False,
            rank_to_GPU=None,
        ):
            test_case_func = {
                dist.ReduceOp.SUM: self._all_reduce_coalesced_sum_test_cases,
                dist.ReduceOp.PRODUCT: self._all_reduce_coalesced_product_test_cases,
                dist.ReduceOp.MIN: self._all_reduce_coalesced_min_test_cases,
                dist.ReduceOp.MAX: self._all_reduce_coalesced_max_test_cases
            }[op]

            master_values, worker_values, expected_values, dtypes = test_case_func(len(group))

            for src in group:
                curr_values = master_values if rank == src else worker_values
                tensors = [
                    _build_tensor(src + 1, val, dtype=dtype)
                    for dtype, val in zip(dtypes, curr_values)
                ]
                if cuda:
                    tensors = [t.cuda(rank_to_GPU[rank][0]) for t in tensors]
                self.call_dist_op(":all_reduce", False, dist.all_reduce_coalesced, tensors, op, group_id)
                expected_tensors = [
                    _build_tensor(src + 1, expected_value, dtype=dtype)
                    for dtype, expected_value in zip(dtypes, expected_values)
                ]
                self.assertEqual(
                    tensors,
                    expected_tensors
                )

            self._barrier()

        @require_backend({"gloo"})
        def test_all_reduce_coalesced_sum(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                cuda=False,
                rank_to_GPU=None,
            )

        @require_backend({"gloo"})
        def test_all_reduce_coalesced_product(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                cuda=False,
                rank_to_GPU=None,
            )

        @require_backend({"gloo"})
        def test_all_reduce_coalesced_min(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.MIN,
                cuda=False,
                rank_to_GPU=None,
            )

        @require_backend({"gloo"})
        def test_all_reduce_coalesced_max(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.MAX,
                cuda=False,
                rank_to_GPU=None
            )

        @skip_if_small_worldsize
        @require_backend({"gloo"})
        def test_all_reduce_coalesced_group_sum(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                cuda=False,
                rank_to_GPU=None
            )

        @skip_if_small_worldsize
        @require_backend({"gloo"})
        def test_all_reduce_coalesced_group_product(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                cuda=False,
                rank_to_GPU=None
            )

        @skip_if_small_worldsize
        @require_backend({"gloo"})
        def test_all_reduce_coalesced_group_min(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.MIN,
                cuda=False,
                rank_to_GPU=None
            )

        @skip_if_small_worldsize
        @require_backend({"gloo"})
        def test_all_reduce_coalesced_group_max(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.MAX,
                cuda=False,
                rank_to_GPU=None
            )

        @require_backend({"gloo"})
        def test_all_reduce_coalesced_full_group_sum(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                cuda=False,
                rank_to_GPU=None
            )

        @require_backend({"gloo"})
        def test_all_reduce_coalesced_full_group_product(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                cuda=False,
                rank_to_GPU=None
            )

        @require_backend({"gloo"})
        def test_all_reduce_coalesced_full_group_min(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.MIN,
                cuda=False,
                rank_to_GPU=None,
            )

        @require_backend({"gloo"})
        def test_all_reduce_coalesced_full_group_max(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_reduce_coalesced_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.MAX,
                cuda=False,
                rank_to_GPU=None
            )

        # SCATTER
        def _test_scatter_helper(self, group, group_id, rank):
            for dest in group:
                tensor = _build_tensor(dest + 1, -1)
                expected_tensor = _build_tensor(dest + 1, rank)
                tensors = (
                    [_build_tensor(dest + 1, i) for i in group] if rank == dest else []
                )
                self.call_dist_op(":scatter", False, dist.scatter, tensor, src=dest, scatter_list=tensors, group=group_id)
                self.assertEqual(tensor, expected_tensor)

            self._barrier()

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_scatter_checks(self):
            group, group_id, rank = self._init_global_test()
            one = torch.ones([1])

            # Specify scatter_list argument only on source rank.
            output = one.clone() * -1
            if rank == 0:
                scatter_list = [one.clone() * i for i in group]
                dist.scatter(output, src=0, scatter_list=scatter_list)
            else:
                dist.scatter(output, src=0)
            self.assertEqual(output, one * rank)

            # Don't specify src argument.
            output = one.clone() * -1
            if rank == 0:
                scatter_list = [one.clone() * i for i in group]
                dist.scatter(output, scatter_list=scatter_list)
            else:
                dist.scatter(output)
            self.assertEqual(output, one * rank)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support scatter")
        def test_scatter(self):
            group, group_id, rank = self._init_global_test()
            self._test_scatter_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support scatter")
        @skip_if_small_worldsize
        def test_scatter_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_scatter_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support scatter")
        def test_scatter_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_scatter_helper(group, group_id, rank)

        # GATHER
        def _test_gather_helper(self, group, group_id, rank):
            for dest in group:
                tensor = _build_tensor(dest + 1, rank)
                tensors = (
                    [_build_tensor(dest + 1, -1) for i in group] if rank == dest else []
                )
                self.call_dist_op(":gather", False, dist.gather, tensor, dst=dest, gather_list=tensors, group=group_id)
                if rank == dest:
                    expected_tensors = [_build_tensor(dest + 1, i) for i in group]
                    for t1, t2 in zip(tensors, expected_tensors):
                        self.assertEqual(t1, t2)

            self._barrier()

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_gather_checks(self):
            group, group_id, rank = self._init_global_test()
            one = torch.ones([1])

            # Specify gather_list argument only on destination rank.
            if rank == 0:
                gather_list = [one.clone() for _ in group]
                dist.gather(one * rank, dst=0, gather_list=gather_list)
                for i in group:
                    self.assertEqual(gather_list[i], one * i)
            else:
                dist.gather(one * rank, dst=0)

            # Don't specify dst argument.
            if rank == 0:
                gather_list = [one.clone() for _ in group]
                dist.gather(one * rank, gather_list=gather_list)
                for i in group:
                    self.assertEqual(gather_list[i], one * i)
            else:
                dist.gather(one * rank)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_gather(self):
            group, group_id, rank = self._init_global_test()
            self._test_gather_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        @skip_if_small_worldsize
        def test_gather_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_gather_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_gather_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_gather_helper(group, group_id, rank)

        # ALL GATHER
        def _test_all_gather_helper(
            self, group, group_id, rank, cuda=False, rank_to_GPU=None, dtype=torch.float
        ):

            for dest in group:
                tensor = _build_tensor(dest + 1, rank, dtype=dtype)
                tensors = [_build_tensor(dest + 1, -1, dtype=dtype) for i in group]
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                    tensors = [t.cuda(rank_to_GPU[rank][0]) for t in tensors]
                self.call_dist_op(":all_gather", False, dist.all_gather, tensors, tensor, group_id)

                expected_tensors = [_build_tensor(dest + 1, i, dtype=dtype) for i in group]
                for t1, t2 in zip(tensors, expected_tensors):
                    self.assertEqual(t1, t2)

            self._barrier()

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_gather(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_gather_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND != "nccl", "Only Nccl supports CUDA all gather")
        @unittest.skipIf(BACKEND == "nccl", "CUDA all gather skipped for NCCL")
        @skip_if_no_gpu
        def test_all_gather_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_gather_helper(group, group_id, rank, True, rank_to_GPU)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_gather_complex(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_gather_helper(group, group_id, rank, dtype=torch.cfloat)

        @unittest.skipIf(BACKEND != "nccl", "Only Nccl supports CUDA all gather")
        @unittest.skipIf(BACKEND == "nccl", "CUDA all gather skipped for NCCL")
        @skip_if_no_gpu
        def test_all_gather_cuda_complex(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_gather_helper(group, group_id, rank, True, rank_to_GPU, dtype=torch.cfloat)

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_gather_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_gather_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support CPU tensors")
        def test_all_gather_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_gather_helper(group, group_id, rank)

        def _run_all_gather_coalesced_and_verify(
            self, output_tensor_lists, input_tensors, expected_tensors, group_id
        ):
            """
            Helper that runs all_gather_coalesced and returns true if output
            matches expectations.
            """
            self.call_dist_op(":all_gather", False, dist.all_gather_coalesced,
                              output_tensor_lists, input_tensors, group_id)

            for l1, l2 in zip(output_tensor_lists, expected_tensors):
                for t1, t2 in zip(l1, l2):
                    if not torch.equal(t1, t2):
                        return False
            return True

        def _test_all_gather_coalesced_helper(
            self, group, group_id, rank, dtype=torch.float
        ):
            # TODO: Instead we should probably go through _rank_not_in_group
            # mechanism to disable sending tensors
            if group_id is not None:
                for test_case_id in range(2, 5):
                    # Make sure we create tensors of incompatible sizes, e.g.
                    # [1], [2x2], [3x3x3] ... to be sent in one batch
                    input_tensors = [
                        _build_multidim_tensor(
                            tensor_id,
                            tensor_id,
                            rank + tensor_id,
                            dtype=dtype) for tensor_id in range(
                                1, test_case_id)
                    ]
                    output_tensor_lists = [
                        [
                            _build_multidim_tensor(
                                tensor_id, tensor_id, -1, dtype=dtype) for tensor_id in range(
                                    1, test_case_id)
                        ] for _ in group
                    ]
                    expected_tensors = [
                        [
                            _build_multidim_tensor(
                                tensor_id,
                                tensor_id,
                                rank_iter + tensor_id,
                                dtype=dtype) for tensor_id in range(
                                    1, test_case_id)
                        ] for rank_iter in group
                    ]
                    assert self._run_all_gather_coalesced_and_verify(
                        output_tensor_lists, input_tensors,
                        expected_tensors, group_id
                    ), "output tensors do not match expected ouputs"

            self._barrier()

        @unittest.skipIf(BACKEND == "nccl", "all_gather_coalesced does not support NCCL")
        @unittest.skipIf(BACKEND == "mpi", "all_gather_coalesced does not support MPI")
        def test_all_gather_coalesced_simple(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_gather_coalesced_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "all_gather_coalesced does not support NCCL")
        @unittest.skipIf(BACKEND == "mpi", "all_gather_coalesced does not support MPI")
        def test_all_gather_coalesced_complex(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_gather_coalesced_helper(group, group_id, rank, dtype=torch.cfloat)

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND == "nccl", "all_gather_coalesced does not support NCCL")
        @unittest.skipIf(BACKEND == "mpi", "all_gather_coalesced does not support MPI")
        def test_all_gather_coalesced_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_gather_coalesced_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "all_gather_coalesced does not support NCCL")
        @unittest.skipIf(BACKEND == "mpi", "all_gather_coalesced does not support MPI")
        def test_all_gather_coalesced_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_gather_coalesced_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "all_gather_coalesced does not support NCCL")
        @unittest.skipIf(BACKEND == "mpi", "all_gather_coalesced does not support MPI")
        def test_all_gather_coalesced_with_empty(self):
            group, group_id, rank = self._init_global_test()
            input_tensors = [
                rank * torch.ones([2, 2]),
                torch.ones([0]),
                (rank + 1) * torch.ones([3, 3]),
                torch.ones([0]),
                torch.ones([0])
            ]
            output_tensors_lists = [
                [
                    -1 * torch.ones([2, 2]),
                    -1 * torch.ones([0]),
                    -1 * torch.ones([3, 3]),
                    -1 * torch.ones([0]),
                    -1 * torch.ones([0])
                ] for _ in group
            ]
            expected_tensors = [
                [
                    r * torch.ones([2, 2]),
                    torch.ones([0]),
                    (r + 1) * torch.ones([3, 3]),
                    torch.ones([0]),
                    torch.ones([0])
                ] for r in group
            ]
            assert self._run_all_gather_coalesced_and_verify(
                output_tensors_lists, input_tensors, expected_tensors, group_id)
            self._barrier()

        # AllToAll
        def _test_all_to_all_single_equal_split_helper(
            self,
            group,
            group_id,
            rank,
            cuda=False,
            rank_to_GPU=None,
        ):
            if group_id is not None:
                size = len(group)
                in_tensor = torch.ones([size, size]) * rank
                expected_tensor = torch.cat([torch.ones([1, size]) * i for i in group])
                out_tensor = torch.ones([size, size]) * -1
                if cuda:
                    in_tensor = in_tensor.cuda(rank_to_GPU[rank][0])
                    expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
                    out_tensor = out_tensor.cuda(rank_to_GPU[rank][0])
                self.call_dist_op(":all_to_all", False, dist.all_to_all_single, out_tensor, in_tensor, group=group_id)
                self.assertEqual(out_tensor, expected_tensor)
            self._barrier()

        def _test_all_to_all_single_unequal_split_helper(
            self,
            group,
            group_id,
            rank,
            cuda=False,
            rank_to_GPU=None,
        ):
            if group_id is not None:
                size = len(group)
                in_splits = [i + 1 for i in group]
                out_splits = [rank + 1 for _ in group]
                in_tensor = torch.ones([sum(in_splits), size]) * rank
                out_tensor = torch.ones([(rank + 1) * size, size])
                expected_tensor = torch.cat([torch.ones([rank + 1, size]) * i for i in group])
                if cuda:
                    in_tensor = in_tensor.cuda(rank_to_GPU[rank][0])
                    expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
                    out_tensor = out_tensor.cuda(rank_to_GPU[rank][0])
                dist.all_to_all_single(
                    out_tensor, in_tensor, out_splits, in_splits, group=group_id)
                self.assertEqual(out_tensor, expected_tensor)
            self._barrier()

        def _test_all_to_all_helper(
            self,
            group,
            group_id,
            rank,
            cuda=False,
            rank_to_GPU=None,
        ):
            if group_id is not None:
                size = len(group)
                in_splits = [i + 1 for i in group]
                in_tensors = [
                    torch.ones([in_splits[i], size]) * rank for i, _ in enumerate(group)
                ]
                out_tensors = [torch.ones([(rank + 1), size]) for _ in group]
                expected_tensors = [torch.ones([rank + 1, size]) * i for i in group]
                if cuda:
                    in_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in in_tensors]
                    expected_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in expected_tensors]
                    out_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in out_tensors]
                dist.all_to_all(out_tensors, in_tensors, group=group_id)
                for t1, t2 in zip(out_tensors, expected_tensors):
                    self.assertEqual(t1, t2)
            self._barrier()

        @unittest.skipIf(
            BACKEND != "mpi", "Only MPI supports CPU all_to_all_single"
        )
        def test_all_to_all_single_equal_split(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_to_all_single_equal_split_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single"
        )
        @skip_if_no_gpu
        def test_all_to_all_single_equal_split_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_single_equal_split_helper(
                group,
                group_id,
                rank,
                True,
                rank_to_GPU,
            )

        @unittest.skipIf(
            BACKEND != "mpi", "Only MPI supports CPU all_to_all_single"
        )
        def test_all_to_all_single_unequal_split(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_to_all_single_unequal_split_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single"
        )
        @skip_if_no_gpu
        def test_all_to_all_single_unequal_split_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_single_unequal_split_helper(
                group,
                group_id,
                rank,
                True,
                rank_to_GPU,
            )

        @unittest.skipIf(BACKEND != "mpi", "Only MPI supports all_to_all")
        def test_all_to_all(self):
            group, group_id, rank = self._init_global_test()
            self._test_all_to_all_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND != "nccl", "Only NCCL supports CUDA all_to_all")
        @skip_if_rocm
        def test_all_to_all_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_helper(group, group_id, rank, True, rank_to_GPU)

        @unittest.skipIf(
            BACKEND != "mpi", "Only MPI supports CPU all_to_all_single"
        )
        @skip_if_small_worldsize
        def test_all_to_all_single_equal_split_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_to_all_single_equal_split_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single"
        )
        @skip_if_no_gpu
        @skip_if_small_worldsize
        def test_all_to_all_single_equal_split_group_cuda(self):
            group, group_id, rank = self._init_group_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_single_equal_split_helper(
                group,
                group_id,
                rank,
                True,
                rank_to_GPU,
            )

        @unittest.skipIf(
            BACKEND != "mpi", "Only MPI supports CPU all_to_all_single"
        )
        @skip_if_small_worldsize
        def test_all_to_all_single_unequal_split_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_to_all_single_unequal_split_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single"
        )
        @skip_if_no_gpu
        @skip_if_small_worldsize
        def test_all_to_all_single_unequal_split_group_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_single_unequal_split_helper(
                group,
                group_id,
                rank,
                True,
                rank_to_GPU,
            )

        @unittest.skipIf(BACKEND != "mpi", "Only MPI supports all_to_all")
        @skip_if_small_worldsize
        def test_all_to_all_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_all_to_all_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single"
        )
        @skip_if_small_worldsize
        @skip_if_rocm
        def test_all_to_all_group_cuda(self):
            group, group_id, rank = self._init_group_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_helper(
                group,
                group_id,
                rank,
                True,
                rank_to_GPU)

        @unittest.skipIf(
            BACKEND != "mpi", "Only MPI supports CPU all_to_all_single"
        )
        def test_all_to_all_single_equal_split_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_to_all_single_equal_split_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single"
        )
        @skip_if_no_gpu
        def test_all_to_all_single_equal_split_full_group_cuda(self):
            group, group_id, rank = self._init_full_group_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_single_equal_split_helper(
                group,
                group_id,
                rank,
                True,
                rank_to_GPU,
            )

        @unittest.skipIf(
            BACKEND != "mpi", "Only MPI supports CPU all_to_all_single"
        )
        def test_all_to_all_single_unequal_split_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_to_all_single_unequal_split_helper(group, group_id, rank)

        @unittest.skipIf(
            BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single"
        )
        @skip_if_no_gpu
        def test_all_to_all_single_unequal_split_full_group_cuda(self):
            group, group_id, rank = self._init_full_group_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_single_unequal_split_helper(
                group,
                group_id,
                rank,
                True,
                rank_to_GPU,
            )

        @unittest.skipIf(BACKEND != "mpi", "Only MPI supports all_to_all")
        def test_all_to_all_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_all_to_all_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND != "nccl", "Only NCCL supports CUDA all_to_all")
        @skip_if_rocm
        def test_all_to_all_full_group_cuda(self):
            group, group_id, rank = self._init_full_group_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_to_all_helper(group, group_id, rank, True, rank_to_GPU)

        # BARRIER
        def _test_barrier_helper(
                self, group, group_id, rank, cuda=False, rank_to_GPU=None):
            WAIT_TIME = 0.3  # seconds

            for dest in group:
                expected_time = torch.DoubleTensor(1).fill_(0.0)
                if cuda:
                    expected_time = expected_time.cuda(rank_to_GPU[rank][0])
                if dest == rank:
                    expected_time.fill_(time.time() + WAIT_TIME)
                    dist.broadcast(expected_time, dest, group_id)
                    time.sleep(WAIT_TIME + 0.1)  # sleep a little bit longer
                    dist.barrier(group_id)
                else:
                    dist.broadcast(expected_time, dest, group_id)
                    dist.barrier(group_id)
                    self.assertGreaterEqual(
                        float(time.time()),
                        float(expected_time[0]),
                        "destination rank: %d, my rank: %d" % (dest, rank) +
                        " (if you see this failure, please report in #14554)")

            # Use higher timeout for the instance where the test runs
            # against a subgroup and uses a CUDA tensor for expected time.
            # The CUDA initialization for the participating processes can
            # take long enough for the barrier timeout to trigger on the
            # process that doesn't participate in the group.
            self._barrier(timeout=20)

        @skip_if_no_gpu
        @unittest.skipIf(BACKEND == "mpi", "MPI doesn't supports GPU barrier")
        def test_barrier_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_barrier_helper(group, group_id, rank, True, rank_to_GPU)

        @skip_if_small_worldsize
        @skip_if_no_gpu
        @unittest.skipIf(BACKEND == "mpi", "MPI doesn't supports GPU barrier")
        def test_barrier_group_cuda(self):
            group, group_id, rank = self._init_group_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_barrier_helper(group, group_id, rank, True, rank_to_GPU)

        @skip_if_small_worldsize
        @skip_if_no_gpu
        @unittest.skipIf(BACKEND == "mpi", "MPI doesn't supports GPU barrier")
        def test_barrier_full_group_cuda(self):
            group, group_id, rank = self._init_full_group_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_barrier_helper(group, group_id, rank, True, rank_to_GPU)

        @unittest.skipIf(BACKEND == "nccl", "NCCL does not support CPU barrier")
        def test_barrier(self):
            group, group_id, rank = self._init_global_test()
            self._test_barrier_helper(group, group_id, rank)

        @skip_if_small_worldsize
        @unittest.skipIf(BACKEND == "nccl", "NCCL does not support CPU barrier")
        def test_barrier_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_barrier_helper(group, group_id, rank)

        @unittest.skipIf(BACKEND == "nccl", "NCCL does not support CPU barrier")
        def test_barrier_full_group(self):
            group, group_id, rank = self._init_full_group_test()
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

        @unittest.skipIf(BACKEND == "mpi", "MPI doesn't support broadcast multigpu")
        @unittest.skipIf(BACKEND == "nccl", "NCCL broadcast multigpu skipped")
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
            dtype=torch.float,
        ):
            for src in group:
                curr_value = master_value if rank == src else worker_value
                tensors = [
                    _build_tensor(src + 1, curr_value, dtype=dtype).cuda(device=i)
                    for i in rank_to_GPU[rank]
                ]
                self.call_dist_op(":all_reduce", False, dist.all_reduce_multigpu, tensors, op, group_id)
                expected_tensor = _build_tensor(src + 1, expected_value, dtype=dtype)
                for tensor in tensors:
                    self.assertEqual(tensor, expected_tensor)

            self._barrier()

        @unittest.skipIf(BACKEND == "mpi", "MPI doesn't support broadcast multigpu")
        @unittest.skipIf(BACKEND == "nccl", "CUDA all_reduce multigpu skipped for NCCL")
        @skip_if_no_gpu
        def test_all_reduce_multigpu(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_reduce_multigpu_helper(
                group,
                group_id,
                rank,
                rank_to_GPU,
                dist.ReduceOp.SUM,
                2,
                10,
                (2 + 10 * (len(group) - 1)) * len(rank_to_GPU[0]),
            )

        @unittest.skipIf(BACKEND == "mpi", "MPI doesn't support broadcast multigpu")
        @unittest.skipIf(BACKEND == "nccl", "CUDA all_reduce multigpu skipped for NCCL")
        @skip_if_no_gpu
        def test_all_reduce_multigpu_complex(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_reduce_multigpu_helper(
                group,
                group_id,
                rank,
                rank_to_GPU,
                dist.ReduceOp.SUM,
                complex(2, 3),
                complex(10, 11),
                (complex(2, 3) + complex(10, 11) * (len(group) - 1)) * len(rank_to_GPU[0]),
                dtype=torch.cfloat,
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
                tensor_value = master_value if rank == src else worker_value
                tensors = [
                    _build_tensor(src + 1, tensor_value).cuda(device=i)
                    for i in rank_to_GPU[rank]
                ]
                self.call_dist_op(
                    "reduce", False, dist.reduce_multigpu, tensors, src, op, group_id,
                    expect_event=len(tensors) == 1)
                if rank == src:
                    expected_tensor = _build_tensor(src + 1, expected_value)
                    self.assertEqual(tensors[0], expected_tensor)

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
                dist.ReduceOp.SUM,
                2,
                10,
                (2 + 10 * (len(group) - 1)) * len(rank_to_GPU[0]),
            )

        def _test_all_gather_multigpu_helper(self, group, group_id, rank, rank_to_GPU, dtype=torch.float):
            for dest in group:
                tensors = [
                    _build_tensor(dest + 1, dtype=dtype).cuda(device=i) for i in rank_to_GPU[rank]
                ]

                # construct expected output along with
                # a place holder to receive all gather results
                output_tensors = []
                expected_output = []
                output_per_gpu = (
                    [_build_tensor(dest + 1, -1, dtype=dtype)] * len(rank_to_GPU[0]) * len(group)
                )
                expected_per_gpu = (
                    [_build_tensor(dest + 1, dtype=dtype)] * len(rank_to_GPU[0]) * len(group)
                )
                for gpu in rank_to_GPU[rank]:
                    output_tensors.append([t.cuda(device=gpu) for t in output_per_gpu])
                    expected_output.append([t.cuda(device=gpu) for t in expected_per_gpu])
                self.call_dist_op(
                    "all_gather", False,
                    dist.all_gather_multigpu, output_tensors, tensors, group_id,
                    expect_event=len(expected_output) == 1)
                self.assertEqual(output_tensors, expected_output)

            self._barrier()

        @unittest.skipIf(BACKEND != "nccl", "Only Nccl backend supports allgather multigpu")
        @skip_if_no_gpu
        def test_all_gather_multigpu(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_gather_multigpu_helper(group, group_id, rank, rank_to_GPU)

        @unittest.skipIf(BACKEND != "nccl", "Only Nccl backend supports allgather multigpu")
        @skip_if_no_gpu
        def test_all_gather_multigpu_complex(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            self._test_all_gather_multigpu_helper(group, group_id, rank, rank_to_GPU, dtype=torch.cfloat)

        def _model_step(self, model):
            for param in model.parameters():
                if param.grad is not None:
                    with torch.no_grad():
                        param += param.grad
                    param.grad = None

        def _model_step_with_zero_grad(self, model):
            for param in model.parameters():
                if param.grad is not None:
                    with torch.no_grad():
                        param += param.grad
                    param.grad.requires_grad_(False)
                    param.grad.zero_()

        def _prepare_dummy_data(self, local_bs):
            # global_bs for DDP should be divisible by WORLD_SIZE
            world_size = int(os.environ["WORLD_SIZE"])
            global_bs = world_size * local_bs
            input_cpu = torch.randn(global_bs, 2)
            target = torch.randn(global_bs, 4)
            loss = nn.MSELoss()
            return global_bs, input_cpu, target, loss

        # END TO END TEST FOR DISTRIBUTEDDATAPARALLEL
        def _test_DDP_helper(self, model, input_var, target, loss, scale_factor=1.0):
            model.train()
            output = model(input_var)
            l = loss(output, target) * scale_factor
            l.backward()

        def _assert_equal_param(self, param_gpu, param_DDP):
            self.assertEqual(len(param_gpu), len(param_DDP))
            for p_gpu, p_DDP in zip(param_gpu, param_DDP):
                self.assertEqual(p_gpu, p_DDP)

        def _test_DDP_5iter(
            self, model_base, model_DDP, input, target, loss, local_bs, rank, batch_size, test_save,
            offset=None, world_size=0, zero_grad=False
        ):
            for idx in range(5):
                # single cpu/gpu training
                self._test_DDP_helper(model_base, input, target, loss)

                if offset is None:
                    offset = rank * local_bs

                # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
                self._test_DDP_helper(
                    model_DDP,
                    input[offset: offset + local_bs],
                    target[offset: offset + local_bs],
                    loss,
                    world_size * local_bs / batch_size if world_size != 0 else 1,
                )

                # Update weights and run a second iteration to shake out errors
                if zero_grad:
                    self._model_step_with_zero_grad(model_base)
                    self._model_step_with_zero_grad(model_DDP)
                else:
                    self._model_step(model_base)
                    self._model_step(model_DDP)
                self._assert_equal_param(
                    list(model_base.parameters()), list(model_DDP.module.parameters())
                )

                # Shuffle the input so that DDP input is different
                input = input[torch.randperm(batch_size)]

                # save the model in the middle and reload
                if test_save and idx == 2 and INIT_METHOD.startswith("file://"):
                    with tempfile.NamedTemporaryFile() as tmp:
                        if sys.platform == 'win32':
                            torch.save(model_DDP, tmp)
                            tmp.seek(0)
                            model_DDP = torch.load(tmp)
                        else:
                            torch.save(model_DDP, tmp.name)
                            model_DDP = torch.load(tmp.name)

            with tempfile.TemporaryFile() as tmp_file:
                torch.save(model_DDP, tmp_file)
                tmp_file.seek(0)
                saved_model = torch.load(tmp_file)
            for k in model_DDP.state_dict():
                self.assertEqual(model_DDP.state_dict()[k], saved_model.state_dict()[k])

        def _test_DistributedDataParallel(self, gpu_subset, rank, output_device=None, gradient_as_bucket_view=False):
            # Run a simple end to end DDP model, use result of single node model
            # as baseline

            # cpu training setup
            model = DDP_NET

            # single gpu training setup
            model_gpu = copy.deepcopy(model)
            model_gpu.cuda(gpu_subset[0])

            # DDP training setup
            model_DDP = copy.deepcopy(model)
            model_DDP.cuda(gpu_subset[0])
            model_DDP = nn.parallel.DistributedDataParallel(
                model_DDP, device_ids=gpu_subset, gradient_as_bucket_view=gradient_as_bucket_view
            )

            # test serializable/unserializable
            with tempfile.NamedTemporaryFile() as tmp:
                if sys.platform == 'win32':
                    torch.save(model_DDP, tmp)
                    tmp.seek(0)
                    model_DDP = torch.load(tmp)
                else:
                    torch.save(model_DDP, tmp.name)
                    model_DDP = torch.load(tmp.name)

            # dummy data initialization
            local_bs = len(gpu_subset)
            global_bs, input_cpu, target, loss = self._prepare_dummy_data(local_bs)

            # check two model parameters over 5 iterations
            self._test_DDP_5iter(
                model_gpu,
                model_DDP,
                input_cpu.cuda(gpu_subset[0]),
                target.cuda(gpu_subset[0]),
                loss,
                local_bs,
                rank,
                global_bs,
                True,
            )
            self._barrier()

        def _test_DistributedDataParallelCPU(self, gradient_as_bucket_view=False):
            # Run a simple end to end DDP-CPU model, use result of single node
            # model as baseline
            group, group_id, rank = self._init_global_test()

            # cpu training setup
            model_base = DDP_NET

            # DDP-CPU training setup
            model_DDP = copy.deepcopy(model_base)
            model_DDP = nn.parallel.DistributedDataParallel(
                model_DDP, gradient_as_bucket_view=gradient_as_bucket_view)

            # dummy data initialization
            local_bs = 2
            global_bs, input_cpu, target, loss = self._prepare_dummy_data(local_bs)

            # check two model parameters over 5 iterations
            self._test_DDP_5iter(
                model_base, model_DDP, input_cpu, target, loss, local_bs, rank, global_bs, False, zero_grad=True
            )
            self._barrier()

        @unittest.skipIf(
            BACKEND == "nccl", "nccl does not support DDP on CPU models"
        )
        def test_DistributedDataParallelCPU(self):
            self._test_DistributedDataParallelCPU()

        @unittest.skipIf(
            BACKEND == "nccl", "nccl does not support DDP on CPU models"
        )
        def test_DistributedDataParallelCPU_grad_is_view(self):
            self._test_DistributedDataParallelCPU(gradient_as_bucket_view=True)

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        def test_DistributedDataParallel_requires_grad(self):
            # a module without gradients shouldn't be accepted
            self.assertRaises(AssertionError, lambda: nn.parallel.DistributedDataParallel(nn.Module()))
            self._barrier()

        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_DistributedDataParallel_non_default_stream(self):
            stream = torch.cuda.Stream(self.rank)
            rank = self.rank
            with torch.cuda.stream(stream):
                net = torch.nn.parallel.DistributedDataParallel(
                    torch.nn.Linear(1, 1, bias=False).cuda(rank), device_ids=[rank]
                )
                for i in range(1000):
                    # Clear gradients manually
                    grad = net.module.weight.grad
                    if grad is not None:
                        grad.requires_grad_(False)
                        grad.zero_()
                    # Forward + BW
                    batch = torch.tensor([rank]).float().cuda(rank)
                    loss = net(batch).sum()
                    loss.backward()
                    # For each worker, the gradient on the weight should be worker_rank.
                    grad = net.module.weight.grad
                    avg = grad.clone()
                    # All-reducing the gradient averages should give us the gradient
                    # average. If not, then one of the workers has not correctly
                    # written back the averaged gradient before this all-reduce call.
                    dist.all_reduce(avg)
                    world_size = int(os.environ["WORLD_SIZE"])
                    avg.div_(world_size)
                    expected_grad = sum(i for i in range(world_size)) / world_size
                    self.assertEqual(
                        avg[0, 0],
                        expected_grad,
                        msg=f"Expected gradient of {expected_grad} but got {avg} on rank {self.rank}",
                    )

        def _test_ddp_hook_parity(self, state, hook):
            rank = self.rank
            m = torch.nn.Linear(1, 5)
            try:
                process_group = state.process_group
            except AttributeError:
                process_group = state

            net_with_hook = torch.nn.parallel.DistributedDataParallel(
                copy.deepcopy(m).to(rank), device_ids=[rank], process_group=process_group
            )
            net_with_hook.register_comm_hook(state=state, hook=hook)
            net_without_hook = torch.nn.parallel.DistributedDataParallel(
                copy.deepcopy(m).to(rank), device_ids=[rank], process_group=process_group
            )
            for i in range(100):
                # Clear gradients manually.
                for g in [net_without_hook.module.weight.grad, net_with_hook.module.weight.grad]:
                    if g is not None:
                        g.requires_grad_(False)
                        g.zero_()
                # Forward + BW
                batch = torch.tensor([rank]).float().cuda(rank)
                loss = net_without_hook(batch).sum()
                loss.backward()
                # For each worker, the gradient on the weight should be worker_rank.
                grad = net_without_hook.module.weight.grad
                avg = grad.clone()
                expected_grad = sum(i for i in range(dist.get_world_size())) / dist.get_world_size()
                loss_hook = net_with_hook(batch).sum()
                loss_hook.backward()
                grad_hook = net_with_hook.module.weight.grad
                avg_hook = grad_hook.clone()
                # Verify hook grad with expected.
                # Cannot use exact match here due to a very small accuracy loss,
                # e.g. 1e-05, for powerSGD hook case.
                assert_func = self.assertEqual if hook == default.allreduce_hook else torch.testing.assert_allclose
                assert_func(
                    avg_hook[0, 0],
                    expected_grad,
                    msg=f"Expected hook grad of {expected_grad} but got {avg_hook[0, 0]}"
                )
                # Verify hook grad with vanilla allreduce
                assert_func(
                    avg_hook[0, 0],
                    avg[0, 0],
                    msg=f"Expected hook grad to be close to allreduce {avg[0, 0]}, but got {avg_hook[0, 0]}"
                )

        @unittest.skipIf(
            BACKEND != "nccl",
            "Only NCCL backend supports DDP communication hook",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm
        def test_ddp_hook_parity_allreduce(self):
            self._test_ddp_hook_parity(state=None, hook=default.allreduce_hook)

        @unittest.skipIf(
            BACKEND != "nccl",
            "Only NCCL backend supports DDP communication hook",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm
        def test_ddp_hook_parity_allreduce_process_group(self):
            # process_group is passed in to both DDP and comm. hook
            rank_to_GPU = self._init_multigpu_helper()
            gpus = [rank_to_GPU[int(r)][0] for r in range(dist.get_world_size())]
            process_group = torch.distributed.new_group(gpus)
            self._test_ddp_hook_parity(state=process_group, hook=default.allreduce_hook)

        @unittest.skipIf(
            BACKEND != "nccl",
            "Only NCCL backend supports DDP communication hook",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm
        def test_ddp_hook_parity_powerSGD(self):
            for warm_start in [True, False]:
                powersgd_state = powerSGD.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=1,
                    warm_start=warm_start,
                )
                self._test_ddp_hook_parity(state=powersgd_state, hook=powerSGD.powerSGD_hook)


        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            gpus = list(rank_to_GPU[rank])
            self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank)

            # test output_device
            self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank, output_device=torch.device('cuda'))

            # test device_ids
            gpus = [torch.device('cuda:' + str(i)) for i in gpus]
            self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank, output_device=torch.device('cuda'))

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel_with_grad_is_view(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            gpus = list(rank_to_GPU[rank])
            self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank, gradient_as_bucket_view=True)

            # test output_device
            self._test_DistributedDataParallel(
                gpu_subset=gpus, rank=rank, output_device=torch.device('cuda'), gradient_as_bucket_view=True)

            # test device_ids
            gpus = [torch.device('cuda:' + str(i)) for i in gpus]
            self._test_DistributedDataParallel(
                gpu_subset=gpus, rank=rank, output_device=torch.device('cuda'), gradient_as_bucket_view=True)

        def _test_DistributedDataParallel_SyncBatchNorm(self, gpu_subset, rank, local_bs, global_bs, offset, output_device=None):
            # Run a simple end to end DDP model, use result of single node model
            # as baseline

            # cpu training setup
            model = BN_NET

            # single gpu training setup
            model_gpu = copy.deepcopy(model)
            model_gpu.cuda(gpu_subset[0])

            # DDP training setup
            model_DDP = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(model))
            model_DDP.cuda(gpu_subset[0])
            model_DDP = nn.parallel.DistributedDataParallel(
                model_DDP, device_ids=gpu_subset
            )

            # test serializable/unserializable
            with tempfile.NamedTemporaryFile() as tmp:
                if sys.platform == 'win32':
                    torch.save(model_DDP, tmp)
                    tmp.seek(0)
                    model_DDP = torch.load(tmp)
                else:
                    torch.save(model_DDP, tmp.name)
                    model_DDP = torch.load(tmp.name)

            # data initialization
            input_cpu = torch.randn(global_bs, 2)
            target = torch.randn(global_bs, 4)
            loss = nn.MSELoss()

            # check two model parameters over 5 iterations
            self._test_DDP_5iter(
                model_gpu,
                model_DDP,
                input_cpu.cuda(gpu_subset[0]),
                target.cuda(gpu_subset[0]),
                loss,
                local_bs,
                rank,
                global_bs,
                True,
                offset,
                dist.get_world_size()
            )
            self._barrier()

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel_SyncBatchNorm(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            # DDP does not support replicating BN layers within a process, hence
            # testing with one module replica per process
            gpus = [rank]

            num_processes = dist.get_world_size()
            local_bs = 2
            bs_offset = int(rank * 2)
            global_bs = int(num_processes * 2)

            self._test_DistributedDataParallel_SyncBatchNorm(
                gpu_subset=gpus,
                rank=rank,
                local_bs=local_bs,
                global_bs=global_bs,
                offset=bs_offset)

            # test output_device
            self._test_DistributedDataParallel_SyncBatchNorm(
                gpu_subset=gpus,
                rank=rank,
                local_bs=local_bs,
                global_bs=global_bs,
                offset=bs_offset,
                output_device=torch.device('cuda'))

            # test device_ids
            gpus = [torch.device('cuda:' + str(i)) for i in gpus]
            self._test_DistributedDataParallel_SyncBatchNorm(
                gpu_subset=gpus,
                rank=rank,
                local_bs=local_bs,
                global_bs=global_bs,
                offset=bs_offset,
                output_device=torch.device('cuda'))

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel_SyncBatchNorm_2D_Input(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            # DDP does not support replicating BN layers within a process, hence
            # testing with one module replica per process
            gpus = [rank]

            model = nn.BatchNorm1d(2)

            # single gpu training setup
            model_gpu = copy.deepcopy(model)
            model_gpu.cuda(gpus[0])

            # DDP training setup
            model_DDP = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(model))
            model_DDP.cuda(gpus[0])
            model_DDP = nn.parallel.DistributedDataParallel(
                model_DDP, device_ids=gpus
            )

            local_bs = len(gpus) * 2
            global_bs = dist.get_world_size() * local_bs
            input_cpu = torch.randn(global_bs, 2)
            target = torch.randn(global_bs, 2)
            loss = nn.MSELoss()

            # disabling cudnn.
            # SyncBatchNorm goes through native_batch_norm kernel, this avoids the
            # numerical issue created by the divergent code path.
            with torch.backends.cudnn.flags(False):
                # check two model parameters over 5 iterations
                self._test_DDP_5iter(
                    model_gpu,
                    model_DDP,
                    input_cpu.cuda(gpus[0]),
                    target.cuda(gpus[0]),
                    loss,
                    local_bs,
                    rank,
                    global_bs,
                    True
                )
                self._barrier()

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        @require_world_size(2)
        def test_DistributedDataParallel_SyncBatchNorm_Single_Input_Per_Process(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            # DDP does not support replicating BN layers within a process, hence
            # testing with one module replica per process
            gpus = [rank]

            model = nn.BatchNorm1d(2)

            # single gpu training setup
            model_gpu = copy.deepcopy(model)
            model_gpu.cuda(gpus[0])

            # DDP training setup
            model_DDP = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(model))
            model_DDP.cuda(gpus[0])
            model_DDP = nn.parallel.DistributedDataParallel(
                model_DDP, device_ids=gpus
            )

            local_bs = 1
            global_bs = dist.get_world_size()
            input_cpu = torch.randn(global_bs, 2)
            target = torch.randn(global_bs, 2)
            loss = nn.MSELoss()

            # disabling cudnn.
            # SyncBatchNorm goes through native_batch_norm kernel, this avoids the
            # numerical issue created by the divergent code path.
            with torch.backends.cudnn.flags(False):
                # check two model parameters over 5 iterations
                self._test_DDP_5iter(
                    model_gpu,
                    model_DDP,
                    input_cpu.cuda(gpus[0]),
                    target.cuda(gpus[0]),
                    loss,
                    local_bs,
                    rank,
                    global_bs,
                    True
                )
                self._barrier()

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Running_Value(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            model = nn.parallel.DistributedDataParallel(ONLY_SBN_NET.cuda(rank), device_ids=[rank])

            input_var = []
            for i in range(dist.get_world_size()):
                input_var_rank = torch.cat([
                    torch.ones(2, 1, 10 ** (i + 1)) * (0.1 ** (i - 1)),
                    torch.ones(2, 1, 10 ** (i + 1)) * (0.3 ** (i - 1))
                ], dim=1)
                input_var.append(input_var_rank)

            all_input_var = torch.cat(
                [x.permute(1, 0, 2).contiguous().view(ONLY_SBN_NET.num_features, -1) for x in input_var],
                dim=1
            ).cuda(rank)

            for i in range(100):
                y = model(input_var[rank].cuda(rank))
                y.mean().backward()

            running_mean, running_var = model.module.running_mean, model.module.running_var
            torch.testing.assert_allclose(running_mean, all_input_var.mean(1))
            torch.testing.assert_allclose(running_var, all_input_var.var(1))

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_gradient(self):
            group, group_id, rank = self._init_global_test()
            # only do single GPU per process
            gpus = [rank]

            # cpu training setup
            model = BN_NET

            num_processes = dist.get_world_size()
            local_bs = rank + 2
            bs_offset = int((rank + 3) * rank / 2)
            global_bs = int((num_processes + 3) * num_processes / 2)

            self._test_DistributedDataParallel_SyncBatchNorm(
                gpu_subset=gpus,
                rank=rank,
                local_bs=local_bs,
                global_bs=global_bs,
                offset=bs_offset)

        @unittest.skipIf(
            BACKEND == "nccl", "nccl does not support DDP on CPU models"
        )
        def test_ddp_logging_data_cpu(self):
            def parse_env(var):
                return os.environ[var] if var in os.environ else "N/A"

            group, group_id, rank = self._init_global_test()
            model_DDP = copy.deepcopy(DDP_NET)
            model_DDP = nn.parallel.DistributedDataParallel(model_DDP, bucket_cap_mb=0.001)
            ddp_logging_data = model_DDP.logger.get_ddp_logging_data()
            self.assertEqual(ddp_logging_data.world_size, dist.get_world_size())
            self.assertEqual(ddp_logging_data.rank, dist.get_rank())
            self.assertEqual(ddp_logging_data.module_name, 'Net')
            self.assertEqual(ddp_logging_data.device_ids, [])
            # output_device is -1 in default if it is not set, e.g.
            # output_device of CPU training is -1.
            self.assertEqual(ddp_logging_data.output_device, -1)
            self.assertEqual(ddp_logging_data.broadcast_buffers, True)
            self.assertEqual(ddp_logging_data.bucket_cap_mb, 0.001)
            self.assertEqual(ddp_logging_data.find_unused_parameters, False)
            self.assertEqual(ddp_logging_data.gradient_as_bucket_view, False)
            self.assertEqual(ddp_logging_data.backend_name, dist.get_backend(group_id))
            self.assertEqual(ddp_logging_data.iteration, 0)
            params = list(model_DDP.parameters())
            num_params = 0
            param_size = 0
            params = list(parameter for parameter in filter(lambda parameter: parameter.requires_grad, params))
            for p in params:
                num_params += 1
                param_size += p.numel() * p.element_size()
            self.assertEqual(ddp_logging_data.dtype, "float")
            self.assertEqual(ddp_logging_data.total_parameter_size_bytes, param_size)
            self.assertEqual(ddp_logging_data.num_parameter_tensors, num_params)
            self.assertEqual(ddp_logging_data.bucket_sizes, [param_size])
            self.assertEqual(ddp_logging_data.master_port, parse_env("MASTER_PORT"))
            self.assertEqual(ddp_logging_data.master_addr, parse_env("MASTER_ADDR"))
            self.assertEqual(ddp_logging_data.cuda_visible_devices, parse_env("CUDA_VISIBLE_DEVICES"))
            self.assertEqual(ddp_logging_data.gloo_socket_ifname, parse_env("GLOO_SOCKET_IFNAME"))
            self.assertEqual(ddp_logging_data.gloo_device_transport, parse_env("GLOO_DEVICE_TRANSPORT"))
            self.assertEqual(ddp_logging_data.nccl_socket_ifname, parse_env("NCCL_SOCKET_IFNAME"))
            self.assertEqual(ddp_logging_data.nccl_blocking_wait, parse_env("NCCL_BLOCKING_WAIT"))
            self.assertEqual(ddp_logging_data.nccl_debug, parse_env("NCCL_DEBUG"))
            self.assertEqual(ddp_logging_data.nccl_nthreads, parse_env("NCCL_NTHREADS"))
            self.assertEqual(ddp_logging_data.nccl_ib_timeout, parse_env("NCCL_IB_TIMEOUT"))
            # test larger net and verify multiple bucket sizes
            model = LargeNet()
            model_DDP = nn.parallel.DistributedDataParallel(model, bucket_cap_mb=1)
            ddp_logging_data = model_DDP.logger.get_ddp_logging_data()
            params = list(model_DDP.parameters())
            self.assertEqual(
                ddp_logging_data.bucket_sizes,
                [params[1].numel() * params[1].element_size(), params[0].numel() * params[0].element_size()])

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_ddp_logging_data_gpu(self):
            group, group_id, rank = self._init_global_test()
            model_DDP = copy.deepcopy(DDP_NET)
            model_DDP.cuda(rank)
            model_DDP = nn.parallel.DistributedDataParallel(model_DDP, device_ids=[rank])
            ddp_logging_data = model_DDP.logger.get_ddp_logging_data()
            self.assertEqual(ddp_logging_data.device_ids, [rank])
            self.assertEqual(ddp_logging_data.output_device, rank)

        @skipIfNoTorchVision
        def test_SyncBatchNorm_process_group(self):
            # When adopting `convert_sync_batchnorm` to convert a `nn.modules`,
            # it need to recursively pass the `process_group` in the module when the `SyncBatchNorm`
            # is nested in a sub-module or sub-sub-module (e.g. resnet50 in torchvision.models).

            process_ids = 0
            process_group = torch.distributed.new_group([process_ids])
            res50_model = torchvision.models.resnet50()
            res50_model_sync = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(res50_model), process_group)
            process_group_sync = res50_model_sync.layer1[0].bn1.process_group
            self.assertEqual(process_group_sync, process_group)

        def _run_reduction_test(
                self, tensor, expected_tensor, op, reduction_fn=dist.all_reduce, dst=None
        ):
            if reduction_fn != dist.all_reduce and dst is None:
                raise ValueError(f"Reduction fn {reduction_fn} must specify dst!")
            if dst is not None:
                reduction_fn(tensor, dst, op)
                # Only destination rank tensor is expected to have final result.
                if dist.get_rank() == dst:
                    self.assertEqual(tensor, expected_tensor)
            else:
                reduction_fn(tensor, op)
                self.assertEqual(tensor, expected_tensor)

        @require_backend({"nccl"})
        @require_backends_available({"nccl"})
        @skip_if_lt_x_gpu(2)
        def test_nccl_backend_bool_allreduce(self):
            torch.cuda.set_device(self.rank)
            # Run all_reduce with PRODUCT
            element = self.rank % 2 == 0
            for op in [dist.ReduceOp.PRODUCT, dist.ReduceOp.MIN]:
                input_tensor = torch.tensor([element, element]).to(self.rank)
                self._run_reduction_test(
                    input_tensor, torch.tensor([False, False]).to(self.rank), op
                )
                # Ensure that all ranks contributing True (cast to 1) results in the
                # correct reduction.
                input_tensor = torch.tensor([True, True]).to(self.rank)
                expected_tensor = input_tensor.clone()
                self._run_reduction_test(
                    input_tensor, expected_tensor, op
                )

            # Run all_reduce with SUM
            for op in [dist.ReduceOp.SUM, dist.ReduceOp.MAX]:
                input_tensor = torch.tensor([element, element]).to(self.rank)
                self._run_reduction_test(
                    input_tensor, torch.tensor([True, True]).to(self.rank), op
                )
            # TODO: NCCL backend does not work correctly for bitwise reduction ops
            # (see https://github.com/pytorch/pytorch/issues/41362). Add tests for
            # these once it is supported.

        @require_backend({"nccl"})
        @require_backends_available({"nccl"})
        @skip_if_lt_x_gpu(2)
        def test_nccl_backend_bool_allgather(self):
            torch.cuda.set_device(self.rank)
            inp = {0: [True, True], 1: [False, True]}
            input_tensor = torch.tensor(inp[self.rank % 2]).to(self.rank)
            # Preserve a copy of the tensor to compare against after allgather.
            input_tensor_copy = input_tensor.clone()
            tensor_list = [
                torch.tensor([False, False]).to(self.rank)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list, input_tensor)

            self.assertEqual(len(tensor_list), dist.get_world_size())
            for i, t in enumerate(tensor_list):
                expected = torch.tensor(inp[i % 2]).to(self.rank)
                self.assertEqual(t, expected)
            # Ensure that the input tensor is not modified, since this collective
            # does not modify its input.
            self.assertEqual(input_tensor_copy, input_tensor)

        @require_backend({"nccl"})
        @require_backends_available({"nccl"})
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_nccl_backend_bool_reduce(self):
            torch.cuda.set_device(self.rank)
            inp = {0: [True, True], 1: [False, False]}
            # Run reduce() with product op
            for op in [dist.ReduceOp.PRODUCT, dist.ReduceOp.MIN]:
                input_tensor = torch.tensor(inp[self.rank % 2]).to(self.rank)
                expected = torch.tensor([False, False]).to(self.rank)
                self._run_reduction_test(
                    input_tensor, expected, op, dist.reduce, dst=0
                )
                # Ensure that all ranks contributing True (cast to 1) results in the
                # correct reduction.
                input_tensor = torch.tensor([True, True]).to(self.rank)
                expected_tensor = input_tensor.clone()
                self._run_reduction_test(
                    input_tensor, expected_tensor, op, dist.reduce, dst=0
                )

            for op in [dist.ReduceOp.SUM, dist.ReduceOp.MAX]:
                input_tensor = torch.tensor(inp[self.rank % 2]).to(self.rank)
                expected = (
                    torch.tensor([True, True]).to(self.rank)
                    if self.rank == 0
                    else input_tensor.clone()
                )
                self._run_reduction_test(
                    input_tensor, expected, op, dist.reduce, dst=0
                )

        @require_backend({"nccl"})
        @require_backends_available({"nccl"})
        @skip_if_lt_x_gpu(2)
        def test_nccl_backend_bool_broadcast(self):
            tensor_size = 10
            bcast_tensor = torch.tensor(
                [
                    (random.random() < 0.5 if self.rank == 0 else False)
                    for _ in range(tensor_size)
                ]
            ).to(self.rank)
            dist.broadcast(bcast_tensor, src=0)
            # Now allgather and ensure the tensors are equal.
            tensor_list = [
                torch.tensor([False for _ in range(tensor_size)]).to(self.rank)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list, bcast_tensor)
            expected = tensor_list[0]
            for tensor in tensor_list[1:]:
                self.assertEqual(tensor, expected)

        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_DistributedSampler_padding(self):
            # Tests padding of distributed sampler.
            world_size = dist.get_world_size()

            # Simulates the 'casual' dataset size
            dataset_size = 100 + world_size + 1
            dataset = [torch.ones(1).to(self.rank) * i for i in range(dataset_size)]

            # Simulates the 'tiny' dataset size
            dataset_tiny_size = max(world_size // 2 - 1, 1)
            dataset_tiny = [torch.ones(1).to(self.rank) * i for i in range(dataset_tiny_size)]

            # Specifying drop_last=True will cause the tail of the data to be dropped.
            dist_sampler = DistributedSampler(dataset=dataset, drop_last=True)
            local_num_samples, local_dataset_size = (
                dist_sampler.num_samples,
                dist_sampler.total_size,
            )
            # The effective dataset size should be the greatest integer that is <=
            # dataset_size that is divisible by the world_size. This is to ensure each
            # rank processes the same number of samples.
            effective_dataset_size = (
                math.ceil((dataset_size - world_size) / world_size)
                if dataset_size % world_size != 0
                else dataset_size / world_size
            )
            self.assertEqual(local_num_samples, effective_dataset_size)
            self.assertEqual(local_dataset_size, local_num_samples * world_size)
            indices_list = list(iter(dist_sampler))
            self.assertEqual(len(indices_list), local_num_samples)

            def validate_global_samples(local_num_samples):
                # Ensure that each rank processes the same number of samples.
                world_samples = [
                    torch.LongTensor([0]).to(self.rank) for _ in range(world_size)
                ]
                dist.all_gather(world_samples, torch.tensor([local_num_samples]).to(self.rank))
                world_samples = [sample.item() for sample in world_samples]
                self.assertEqual(len(set(world_samples)), 1)

            validate_global_samples(local_num_samples)

            # drop_last=False is the default and will add additional indices to be sampled,
            # increasing the effective dataset size.
            dist_sampler_added_samples = DistributedSampler(dataset=dataset)
            local_num_samples, local_dataset_size = (
                dist_sampler_added_samples.num_samples,
                dist_sampler_added_samples.total_size,
            )
            # The effective dataset size is the smallest integer that is >= dataset_size
            # and divisible by the world size.
            self.assertEqual(
                local_num_samples, math.ceil(dataset_size / world_size)
            )
            self.assertEqual(local_dataset_size, local_num_samples * world_size)
            indices_list = list(iter(dist_sampler_added_samples))
            self.assertEqual(len(indices_list), local_num_samples)

            # Ensure that each rank processes the same number of samples.
            validate_global_samples(local_num_samples)

            # Ensure additional samples are padded even when
            # the extremely small dataset is given.
            dist_sampler_added_samples_tiny = DistributedSampler(dataset=dataset_tiny)
            local_num_samples, local_dataset_size = (
                dist_sampler_added_samples_tiny.num_samples,
                dist_sampler_added_samples_tiny.total_size,
            )
            self.assertEqual(
                local_num_samples, math.ceil(dataset_tiny_size / world_size)
            )
            self.assertEqual(local_dataset_size, local_num_samples * world_size)
            indices_list = list(iter(dist_sampler_added_samples_tiny))
            self.assertEqual(len(indices_list), local_num_samples)
            validate_global_samples(local_num_samples)


        @require_backend({"nccl", "gloo"})
        @require_n_gpus_for_nccl_backend(int(os.environ["WORLD_SIZE"]), os.environ["BACKEND"])
        def test_allgather_object(self):
            # Only set device for NCCL backend since it must use GPUs.
            backend = os.environ["BACKEND"]
            if backend == "nccl":
                # Case where rank != GPU device.
                next_rank = (self.rank + 1) % int(self.world_size)
                torch.cuda.set_device(next_rank)

            # If GPU test, add object with GPU tensor
            if backend == "nccl":
                COLLECTIVES_OBJECT_TEST_LIST.append(Foo(torch.randn(3, 3, device=0)))

            gather_objects = COLLECTIVES_OBJECT_TEST_LIST

            output_gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(
                output_gathered, gather_objects[self.rank % len(gather_objects)]
            )

            for i, val in enumerate(output_gathered):
                expected = gather_objects[i % len(gather_objects)]
                self.assertEqual(val, expected)

                output_gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(
                    output_gathered, gather_objects[self.rank % len(gather_objects)]
                )

        @require_backend({"gloo"})
        @unittest.skipIf(BACKEND == "nccl", "NCCL does not support gather")
        def test_gather_object(self):
            # Ensure stateful objects can be gathered
            gather_objects = COLLECTIVES_OBJECT_TEST_LIST
            output_gathered = [None for _ in range(dist.get_world_size())]
            gather_on_rank = 0
            my_rank = dist.get_rank()
            dist.gather_object(
                gather_objects[self.rank % len(gather_objects)],
                object_gather_list=output_gathered if my_rank == gather_on_rank else None,
                dst=gather_on_rank,
            )
            if my_rank != gather_on_rank:
                self.assertEqual(
                    output_gathered, [None for _ in range(dist.get_world_size())]
                )
            else:
                for i, val in enumerate(output_gathered):
                    expected = gather_objects[i % len(gather_objects)]
                    self.assertEqual(val, expected)

            # Validate errors when objects can't be pickled.
            class Bar:
                pass

            b = Bar()
            gather_objects = [b for _ in range(dist.get_world_size())]
            with self.assertRaisesRegex(AttributeError, "Can't pickle local object"):
                dist.all_gather_object(
                    [None for _ in range(dist.get_world_size())], gather_objects[self.rank]
                )

        @require_backend({"nccl"})
        @require_backends_available({"nccl"})
        @skip_if_lt_x_gpu(2)
        def test_nccl_gather_object_err(self):
            output_gathered = [None for _ in range(dist.get_world_size())]
            gather_on_rank = 0
            # Case where rank != GPU device.
            my_rank = dist.get_rank()
            next_rank = (my_rank + 1) % dist.get_world_size()
            torch.cuda.set_device(next_rank)
            with self.assertRaisesRegex(
                RuntimeError, "ProcessGroupNCCL does not support gather"
            ):
                dist.gather_object(
                    "foo",
                    object_gather_list=output_gathered
                    if my_rank == gather_on_rank
                    else None,
                    dst=gather_on_rank,
                )

        def validate_net_equivalence(self, net):
            # Helper to validate synchronization of nets across ranks.
            net_module_states = list(net.module.state_dict().values())
            # Check that all tensors in module's state_dict() are equal.
            for t in net_module_states:
                tensor_list = [
                    torch.zeros_like(t) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list, t)
                for tensor in tensor_list:
                    self.assertEqual(tensor, t)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_sync_params_and_buffers(self):
            # Test that after calling _sync_params_and_buffers, models across ranks
            # are the same and are equal to the model on the input rank.
            dim = 2
            rank = self.rank
            rank_to_broadcast = 1
            # Seed to ensure that ranks are initialized with different initial models.
            torch.manual_seed(rank)
            model = nn.Linear(dim, dim, bias=False)
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(rank), device_ids=[self.rank], bucket_cap_mb=1
            )
            new_model = nn.Linear(dim, dim, bias=False).cuda(rank)
            net.module = copy.deepcopy(new_model)
            # Assert params are different
            net_module_states = list(net.module.state_dict().values())
            for t in net_module_states:
                tensor_list = [
                    torch.zeros_like(t) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list, t)
                for i, tensor in enumerate(tensor_list):
                    if i == rank:
                        self.assertEqual(t, tensor)
                    else:
                        # tensor from another rank should be different.
                        self.assertNotEqual(t, tensor)

            net._sync_params_and_buffers(authoritative_rank=rank_to_broadcast)
            # Now all model params should be the same.
            self.validate_net_equivalence(net)
            # Since the network params were broadcast from rank_to_broadcast, validate that
            # they are the same as new_model on rank_to_broadcast.
            if rank == rank_to_broadcast:
                expected_states = new_model.state_dict().values()
                for t, expected in zip(net_module_states, expected_states):
                    self.assertEqual(t, expected)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_grad_div_uneven_inputs(self):
            # Test gradient division during training with join() API. If
            # divide_by_initial_world_size=False, we scale by the effective world
            # size when allreducing grads.
            dim = 5
            batch = 1
            grad_scale = 50
            rank = self.rank
            model = nn.Linear(dim, dim, bias=False)
            inp = torch.ones(batch, dim, device=self.rank) * grad_scale
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(rank), device_ids=[self.rank], bucket_cap_mb=1
            )
            n_iters = 3
            if self.rank > 0:
                n_iters += 2

            with net.join(divide_by_initial_world_size=False):
                for _ in range(n_iters):
                    loss = net(inp).sum()
                    loss.backward()
                    # The grad is always expected_grad, since we divide by the number
                    # of currently active processes and inactive processes contribute
                    # zero gradient. If we kept dividing by static initial world
                    # size as processes leave, the grad would be smaller.
                    expected_grad = torch.ones(dim, dim, device=self.rank) * grad_scale
                    param = list(net.parameters())[0]
                    self.assertEqual(expected_grad, param.grad)
                    # Avoid accumulating grads so that it's the same every iteration
                    net.zero_grad()
                    torch.cuda.synchronize(device=self.rank)

            # If divide_by_initial_world_size=True (default), we always scale grads
            # by the initial world_size.
            with net.join(divide_by_initial_world_size=True):
                for i in range(n_iters):
                    loss = net(inp).sum()
                    loss.backward()
                    effective_ws = dist.get_world_size()
                    if i >= 3:
                        effective_ws -= 1
                    expected_grad = (
                        torch.ones(dim, dim, device=self.rank) * grad_scale * effective_ws
                    ) / dist.get_world_size()
                    param = list(net.parameters())[0]
                    self.assertEqual(expected_grad, param.grad)
                    # Avoid accumulating grad so that it's the same every iteration.
                    net.zero_grad()
                    torch.cuda.synchronize(device=self.rank)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_join_model_equivalence(self):
            # Verifies equivalence with model training locally and with DDP under
            # the join context manager.
            batch = 3
            dim = 10
            learning_rate = 0.03
            model = nn.Linear(dim, dim, bias=False)
            inp = torch.rand(batch, dim, device=self.rank)
            local_model = copy.deepcopy(model)
            local_model = local_model.cuda(self.rank)
            rank_to_iter_mapping = {rank : 2 * (rank + 1) for rank in range(dist.get_world_size())}
            # run local model
            local_iters = sum(rank_to_iter_mapping.values())
            local_optim = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
            for _ in range(local_iters):
                local_optim.zero_grad()
                out = local_model(inp)
                loss = out.sum()
                loss.backward()
                local_optim.step()

            # run DDP model with join API
            num_iters = rank_to_iter_mapping[self.rank]
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(self.rank), device_ids=[self.rank]
            )
            ddp_optim = torch.optim.SGD(
                model.parameters(), lr=learning_rate * dist.get_world_size()
            )
            with net.join():
                for i in range(num_iters):
                    ddp_optim.zero_grad()
                    out = net(inp)
                    loss = out.sum()
                    loss.backward()
                    torch.cuda.synchronize(device=self.rank)
                    ddp_optim.step()

            # Validate model state dicts are equal
            for (_, local_tensor), (_, dist_tensor) in zip(
                local_model.state_dict().items(), net.module.state_dict().items()
            ):
                self.assertEqual(local_tensor, dist_tensor)

        def _run_uneven_inputs_test(
            self, test_case, iteration_mapping, find_unused_params,
        ):
            model = test_case.model
            inp = test_case.inp
            rank = self.rank
            sync_interval = test_case.sync_interval
            # Ensure all outsanding GPU work is comlete so this test runs independently.
            dist.barrier()
            # Bucket_cap_mb is intentionally low to test allreduce scheduling when
            # there are many buckets.
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(rank),
                device_ids=[rank],
                bucket_cap_mb=1,
                find_unused_parameters=find_unused_params,
            )

            # Determine num iters for this rank via the passed in mapping.
            num_iters = iteration_mapping[rank]
            with net.join():
                for i in range(num_iters):
                    # Use model.no_sync() to disable grad synchronization every
                    # sync_interval.
                    if i % sync_interval != 0:
                        context = net.no_sync()
                    else:
                        context = suppress()
                    with context:
                        if isinstance(inp, tuple):
                            loss = net(*inp).sum()
                        else:
                            loss = net(inp).sum()
                        loss.backward()
                        self._model_step(net)
                        # Ensure completion of GPU kernels (including allreduce). If the
                        # join API is not properly implemented, then this should hang
                        # since the allreduce will hang.
                        torch.cuda.synchronize(device=rank)

            # Ensure completion of all GPU kernels.
            torch.cuda.synchronize(device=rank)
            self.assertTrue(net._authoritative_rank)
            # All ranks should have agreed on the same authoritative_rank!
            final_rank_tensor = torch.tensor([net._authoritative_rank], device=self.rank)
            tensor_list = [
                torch.zeros_like(final_rank_tensor)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list, final_rank_tensor)
            max_rank = dist.get_world_size() - 1
            self.assertSetEqual({max_rank}, set(tensor.item() for tensor in tensor_list))
            # Ensure that all models are the same across ranks after all have joined.
            self.validate_net_equivalence(net)
            dist.barrier()

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_uneven_inputs(self):
            class DDPUnevenTestInput(NamedTuple):
                name: str
                model: nn.Module
                inp: Union[torch.tensor, tuple]
                sync_interval: int

            dim = 1000
            batch = 1
            # Create a variety of models to run uneven input tests on.
            large_model = nn.Sequential(
                nn.Conv2d(1, 20, 5),
                nn.ReLU(),
                nn.Conv2d(20, 32, 5),
                nn.ReLU(),
                nn.Conv2d(32, 256, 5),
                nn.ReLU(),
            )
            small_model = nn.Linear(dim, dim, bias=False)
            bn_net = BatchNormNet()

            class UnusedParamModule(nn.Module):
                def __init__(self, unused_params_rank):
                    super().__init__()
                    self.t0 = Task()
                    self.t1 = Task()
                    self.unused_params_rank = unused_params_rank

                def task_parameters(self):
                    return (self.t0.p, self.t1.p)

                def forward(self, x, rank):
                    return (
                        self.t1(self.t0(x))
                        if rank != self.unused_params_rank
                        else self.t1(x)
                    )

            unjoined_rank_with_unused_params_model = UnusedParamModule(1)
            joined_rank_with_unused_params_model = UnusedParamModule(0)

            rank = self.rank
            models_to_test = [
                # Network with batchnorm
                DDPUnevenTestInput(
                    name="batch_norm_net",
                    model=bn_net,
                    inp=torch.ones(batch, 2, device=rank),
                    sync_interval=1
                ),
                DDPUnevenTestInput(
                    name="large_conv_model",
                    model=large_model,
                    inp=torch.ones(batch, batch, dim, dim, device=rank),
                    sync_interval=1,
                ),
                DDPUnevenTestInput(
                    name="small_model",
                    model=small_model,
                    inp=torch.ones(batch, dim, device=rank),
                    sync_interval=1,
                ),
                # Unused parameter test where rank that does not join early has unused params
                DDPUnevenTestInput(
                    name="unjoined_rank_with_unused_params_model",
                    model=unjoined_rank_with_unused_params_model,
                    inp=(torch.ones(batch, 2, device=rank), rank),
                    sync_interval=1,
                ),
                # Unused parameter test where rank that does join early has unused params
                DDPUnevenTestInput(
                    name="joined_rank_with_unused_params_model",
                    model=joined_rank_with_unused_params_model,
                    inp=(torch.ones(batch, 2, device=rank), rank),
                    sync_interval=1,
                ),
            ]

            # Add resnet model if we have torchvision installed.
            if HAS_TORCHVISION:
                resnet_model = torchvision.models.resnet50()
                models_to_test.append(
                    DDPUnevenTestInput(
                        name="resnet_model",
                        model=resnet_model,
                        inp=torch.ones(1, 3, 1000, 1000),
                        sync_interval=1,
                    )
                )

            # Test with no_sync every 2, 3, 4, ... iterations.
            models_with_sync = []
            for i, test_input in enumerate(models_to_test):
                models_with_sync.append(
                    DDPUnevenTestInput(
                        name=test_input.name,
                        model=test_input.model,
                        inp=test_input.inp,
                        sync_interval=i + 2,
                    )
                )

            models_to_test.extend(models_with_sync)

            # 0 iteration tests for when one process does not train model at all, so
            # we must shadow the broadcast calls made when rebuilding buckets.
            baseline_num_iters = [0, 5]
            iteration_offsets = [2, 3, 10]
            num_uneven_ranks = [1]
            if dist.get_world_size() > 2:
                num_uneven_ranks.append(2)
            iteration_mappings = []
            # Generate rank : num_iters mappings for various uneven input scenarios.
            # This includes cases where rank 0 joins early and all other ranks join
            # later, and scenarios where multiple ranks join early, but at different
            # iterations, and later ranks join later.
            for num_early_join_ranks in num_uneven_ranks:
                for baseline_iter in baseline_num_iters:
                    for offset in iteration_offsets:
                        mapping = {
                            rank: baseline_iter for rank in range(0, num_early_join_ranks)
                        }
                        # if num_early_join_ranks > 1, ranks > 0 that will join early
                        # iterate offset//2 more times than rank 0, to test nodes
                        # depleting inputs at different times.
                        if num_early_join_ranks > 1:
                            for rank in mapping.keys():
                                if rank > 0:
                                    mapping[rank] += offset // 2
                        mapping.update(
                            {
                                rank: baseline_iter + offset
                                for rank in range(
                                    num_early_join_ranks, dist.get_world_size()
                                )
                            }
                        )
                        iteration_mappings.append(mapping)

            for (test_case, iteration_mapping) in itertools.product(
                models_to_test, iteration_mappings
            ):
                if self.rank == 0:
                    print(
                        f"""Running test: {test_case.name} sync interval
                        {test_case.sync_interval} with iteration mapping
                        {iteration_mapping}"""
                    )
                self._run_uneven_inputs_test(
                    test_case,
                    iteration_mapping,
                    find_unused_params=("unused_params_model" in test_case.name),
                )

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_uneven_input_join_disable(self):
            # tests that if net.join() with enable=False is specified, DDP works as
            # expected with even inputs.
            torch.manual_seed(self.rank)
            net = torch.nn.parallel.DistributedDataParallel(
                torch.nn.Linear(1, 1).cuda(self.rank), device_ids=[self.rank]
            )
            inp = torch.ones(1) * self.rank
            n_iters = 5
            world_size = dist.get_world_size()
            with net.join(enable=False):
                for _ in range(n_iters):
                    # Clear grads
                    grad = net.module.weight.grad
                    if grad is not None:
                        grad.requires_grad_(False)
                        grad.zero_()
                    out = net(inp)
                    loss = out.sum()
                    loss.backward()
                    # Validate gradients to ensure that we divide by the correct
                    # world_size when join mode is disabled.
                    expected_grad = sum(i for i in range(world_size)) / world_size
                    self.assertEqual(
                        net.module.weight.grad.item(), expected_grad
                    )

            join_config = net.ddp_uneven_inputs_config
            self.assertFalse(join_config.ddp_join_enabled)
            self.validate_net_equivalence(net)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_uneven_input_exception(self):
            # Tests that exceptions during training are correctly propagated by the
            # context manager.
            error_str = "Intentional error"

            class ExceptionModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.param = nn.Parameter(torch.ones(1, requires_grad=True))

                def forward(self, _):
                    raise ValueError(error_str)

            exception_module = ExceptionModule()
            net = torch.nn.parallel.DistributedDataParallel(
                exception_module.cuda(self.rank), device_ids=[self.rank]
            )
            inp = torch.ones(1)
            with self.assertRaisesRegex(ValueError, error_str):
                with net.join():
                    out = net(inp)
                    loss = out.sum()
                    loss.backward()

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(4)
        def test_ddp_uneven_inputs_replicated_error(self):
            # Tests that the context manager errors out in SPMD mode.
            group = dist.new_group([0, 1])
            if self.rank < 2:
                model = nn.Linear(1, 1, bias=False)
                rank_to_device = {0: [0, 1], 1: [2, 3]}

                devices = rank_to_device[self.rank]
                net = torch.nn.parallel.DistributedDataParallel(
                    model.cuda(devices[0]), device_ids=devices, process_group=group
                )
                with self.assertRaisesRegex(
                    ValueError, r"DDP join\(\) API does not support Single-Process Multi-GPU"
                ):
                    with net.join():
                        pass
            # We need a barrier since otherwise non-participating processes exit too early
            # and cause a timeout.
            self._barrier(timeout=60)

        @require_backend({"nccl", "gloo"})
        @require_n_gpus_for_nccl_backend(int(os.environ["WORLD_SIZE"]), os.environ["BACKEND"])
        def test_broadcast_object_list(self):
            # Only set device for NCCL backend since it must use GPUs.
            backend = os.environ["BACKEND"]
            if backend == "nccl":
                # Case where rank != GPU device.
                next_rank = (self.rank + 1) % int(self.world_size)
                torch.cuda.set_device(next_rank)

            src_rank = 0
            # If GPU test, add object with GPU tensor
            if backend == "nccl":
                COLLECTIVES_OBJECT_TEST_LIST.append(Foo(torch.randn(3, 3, device=0)))

            objects = (
                COLLECTIVES_OBJECT_TEST_LIST
                if self.rank == src_rank
                else [None for _ in COLLECTIVES_OBJECT_TEST_LIST]
            )

            # Single object test
            single_obj_list = [objects[0]]
            if self.rank != src_rank:
                self.assertNotEqual(single_obj_list[0], COLLECTIVES_OBJECT_TEST_LIST[0])
            dist.broadcast_object_list(single_obj_list, src=0)
            self.assertEqual(single_obj_list[0], COLLECTIVES_OBJECT_TEST_LIST[0])

            # Multiple input objects test
            if self.rank != src_rank:
                self.assertNotEqual(objects, COLLECTIVES_OBJECT_TEST_LIST)
            dist.broadcast_object_list(objects, src=0)
            self.assertEqual(objects, COLLECTIVES_OBJECT_TEST_LIST)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_ignore_params_arg(self):
            class TestModel(nn.Module):
                def __init__(self, rank):
                    self.rank = rank
                    super(TestModel, self).__init__()
                    self.fc1 = nn.Linear(1, 1, bias=False)
                    # Proxy that will be materialized to another architecture later.
                    # (after wrapping model with DDP)
                    if self.rank == 0:
                        self.fc2 = nn.Linear(1, 10, bias=False)
                    else:
                        self.fc2 = nn.Linear(10, 10, bias=False)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.fc2(x)
                    return x

            device_id = self.rank
            # Ensure the test works for both find_unused_parameter and broadcast_buffer settings.
            for (find_unused, broadcast_buffers) in itertools.product([False, True], [False, True]):
                model = TestModel(self.rank).float().to(device_id)
                # Note that the model can have different shape buffers if we pass
                # them in to be ignored as well.
                model.fc2.register_buffer(
                    "ignore_buffer", torch.zeros(5 + self.rank, device=self.rank)
                )
                proxy_params = list(model.fc2.parameters())
                proxy_buffers = list(model.fc2.buffers())
                model_fc2_name = [
                    module_name
                    for module_name, module in model.named_modules()
                    if module is model.fc2
                ][0]
                proxy_param_names = [
                    f"{model_fc2_name}.{param_name}"
                    for param_name, _ in model.fc2.named_parameters()
                ]
                proxy_buffer_names = [
                    f"{model_fc2_name}.{buf_name}"
                    for buf_name, _ in model.fc2.named_buffers()
                ]
                # Specify that we should ignore proxy_params since it will be
                # materialized later.
                torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
                    model, proxy_param_names + proxy_buffer_names
                )
                ddp = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[device_id],
                    find_unused_parameters=find_unused,
                    broadcast_buffers=broadcast_buffers,
                )
                # Materialize new params. These are not registered in DDP and thus
                # don't have autograd hooks installed on them.
                ddp.module.fc2 = nn.Linear(1, 1, bias=False).to(device_id)
                # local model with the new materialized parameters.
                local_model = copy.deepcopy(ddp.module).cuda(self.rank)

                inp = torch.ones(1, dtype=torch.float).to(device_id) * (self.rank + 1)
                for i in range(6):
                    ddp(inp).sum().backward()
                    local_model(inp).sum().backward()
                    # materialized param grad is not touched by DDP, so its grad should
                    # be the same as if running locally.
                    for materialized_param, local_param in zip(
                        ddp.module.fc2.parameters(), local_model.fc2.parameters()
                    ):
                        self.assertEqual(materialized_param.grad, local_param.grad)

                    # fc1 parameter grad should still be different, due to allreduce.
                    for synced_param, local_param in zip(
                        ddp.module.fc1.parameters(), local_model.fc1.parameters()
                    ):
                        self.assertFalse(synced_param.grad == local_param.grad)

                    # Proxy module grad should not be touched
                    for proxy_param in proxy_params:
                        self.assertTrue(proxy_param.grad is None)

                # Synchronize since we run multiple iterations of this test, to
                # isolate failure hangs.
                torch.cuda.synchronize(device=self.rank)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_unused_params_rebuild_buckets_exception(self):
            class ToyModel(nn.Module):
                def __init__(self):
                    super(ToyModel, self).__init__()
                    self.net1 = nn.Linear(10, 10, bias=False)
                    self.net2 = nn.Linear(10, 10, bias=False)

                def forward(self, x):
                    return self.net1(x)

            ddp = torch.nn.parallel.DistributedDataParallel(
                ToyModel().cuda(self.rank), device_ids=[self.rank]
            )
            for i in range(2):
                inp = torch.rand(1, 10)
                if i > 0:
                    # On 2nd iteration, this will fail during rebuild_buckets,
                    # but we should report an error regarding unused parameters
                    # since that is the underlying root cause.
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Expected to have finished reduction in the prior iteration",
                    ):
                        ddp(inp).sum().backward()
                else:
                    ddp(inp).sum().backward()

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_shared_grad_acc_unused_params(self):
            # When find_unused_parameters=True, ensure we mark unused parameters
            # even if they share gradient accumulators.
            class ToyModel(nn.Module):
                def __init__(self):
                    super(ToyModel, self).__init__()
                    # net1, bias, and net1.bias are all unused params.
                    self.net1 = nn.Linear(10, 5, bias=False)
                    self.bias = nn.Parameter(torch.zeros(5))
                    # net1.bias and self.bias are names for the same underlying
                    # parameter, so they share the same grad acc. This caused
                    # the bug reported in https://github.com/pytorch/pytorch/issues/41324.
                    self.net1.bias = self.bias
                    self.net2 = nn.Linear(10, 5)

                def forward(self, x):
                    return self.net2(x)

            torch.cuda.set_device(self.rank)
            model = ToyModel().to(torch.cuda.current_device())
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.rank], find_unused_parameters=True
            )
            inp = torch.randn(20, 10, device=self.rank)
            for i in range(6):
                out = ddp_model(inp)
                loss = out.sum()
                loss.backward()

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_device(self):
            m = nn.Linear(10, 10).to(self.rank)
            expected_len = 2

            class TensorWrapper:
                __slots__ = ['t', 'moved_to_gpu']

                def __init__(self, t):
                    self.t = t
                    self.moved_to_gpu = False

            # Handlers for specific types of validation we want to do based on
            # the input type.

            def tuple_and_list_validator(x):
                self.assertTrue(len(x), expected_len)
                self.assertEqual(1, len(set(t.device for t in x)))
                self.assertEqual(x[0].device.index, self.rank)
                return x[0] + x[1]

            def namedtuple_validator(x):
                self.assertEqual(x._fields, EXPECTED_FIELDS)
                self.assertEqual(x.a.device.index, x.b.device.index)
                self.assertEqual(x.a.device.index, self.rank)
                return x.a + x.b

            def custom_type_validator(x):
                self.assertTrue(x.moved_to_gpu or (str(x.t.device) == "cpu"))
                x.t = x.t.to(self.rank)
                x.moved_to_gpu = True
                return x.t

            def dict_validator(x):
                self.assertTrue(EXPECTED_FIELDS[0] in x.keys())
                self.assertTrue(EXPECTED_FIELDS[1] in x.keys())
                self.assertEqual(1, len(set(t.device for t in x.values())))
                self.assertEqual(x[EXPECTED_FIELDS[0]].device.index, self.rank)
                return x[EXPECTED_FIELDS[0]] + x[EXPECTED_FIELDS[1]]

            validators = {
                TensorWrapper: custom_type_validator,
                tuple: tuple_and_list_validator,
                list: tuple_and_list_validator,
                TestNamedTupleInput_0: namedtuple_validator,
                TestNamedTupleInput_1: namedtuple_validator,
                dict: dict_validator,
            }

            class ToyModel(torch.nn.Module):
                def __init__(_self):  # noqa: B902
                    super().__init__()
                    _self.lin = nn.Linear(10, 10, bias=False)

                def forward(_self, x, expected_type):  # noqa: B902
                    # Similar to scatter, the recursive to in the single-device
                    # case does not move tensors if they are in a custom type.
                    self.assertTrue(isinstance(x, expected_type))
                    fwd_tensor = validators[expected_type](x)
                    return _self.lin(fwd_tensor)

            model = torch.nn.parallel.DistributedDataParallel(
                ToyModel().to(self.rank), device_ids=[self.rank]
            )

            def train_iter(inp, input_type):
                for _ in range(4):
                    out = model(inp, input_type)
                    out.sum().backward()

            # CPU tuple input, should be moved to the proper device before call
            # to forward.
            inp = tuple(torch.randn(10, 10) for _ in range(expected_len))
            train_iter(inp, tuple)

            # List CPU input, should be moved to proper device before call to
            # forward.
            inp = [torch.randn(10, 10) for _ in range(expected_len)]
            train_iter(inp, list)
            # Custom type containing tensor. The type is maintained, but the
            # device is not propagated (which is what happens with scatter too)
            inp = TensorWrapper(torch.randn(10, 10))
            train_iter(inp, TensorWrapper)
            # NamedTuple input. The type should be maintained and tensor inputs
            # should be moved to the correct device as in scatter.
            batch = 5
            dim = 10
            a = torch.rand(batch, dim)
            b = torch.rand(batch, dim)

            inp = TestNamedTupleInput_0(a, b)
            train_iter(inp, type(inp))

            inp = TestNamedTupleInput_1(a, b)
            train_iter(inp, type(inp))

            # dictionary input.
            inp = {
                EXPECTED_FIELDS[0]: a,
                EXPECTED_FIELDS[1]: b,
            }
            train_iter(inp, type(inp))

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_namedtuple(self):
            batch = 5
            dim = 10

            a = torch.rand(batch, dim, device=self.rank)
            b = torch.rand(batch, dim, device=self.rank)

            class NamedTupleModule(torch.nn.Module):
                def __init__(_self):  # noqa
                    super().__init__()
                    _self.lin = nn.Linear(10, 1)

                def forward(_self, input, expected_type):  # noqa
                    # Without NamedTuple support, this would be of type tuple.
                    self.assertTrue(
                        isinstance(input, expected_type),
                        f"Expected type {expected_type} but got {type(input)}",
                    )
                    self.assertEqual(input._fields, EXPECTED_FIELDS)
                    self.assertEqual(a, input.a)
                    self.assertEqual(b, input.b)
                    return _self.lin(torch.mul(input.a, input.b))

            model = torch.nn.parallel.DistributedDataParallel(
                NamedTupleModule().cuda(self.rank), device_ids=[self.rank]
            )
            inp = TestNamedTupleInput_0(a, b)
            # The following would fail if DDP does not propagate NamedTuples correctly.
            model(inp, type(inp))

            inp = TestNamedTupleInput_1(a, b)
            model(inp, type(inp))

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        @skip_if_rocm
        def test_ddp_control_flow_same_across_ranks(self):
            # Control flow that is the same across ranks.
            batch = 20
            dim = 10

            class ToyModel(nn.Module):
                def __init__(self):
                    super(ToyModel, self).__init__()
                    self.lin1 = nn.Linear(10, 10, bias=False)
                    self.lin2 = nn.Linear(10, 10, bias=False)

                def forward(self, x):
                    # Second layer is used dependent on input x.
                    use_second_layer = torch.equal(
                        x, torch.ones(batch, dim, device=x.device)
                    )
                    if use_second_layer:
                        return self.lin2(F.relu(self.lin1(x)))
                    else:
                        return F.relu(self.lin1(x))

            world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            model = torch.nn.parallel.DistributedDataParallel(
                ToyModel().cuda(self.rank),
                device_ids=[self.rank],
                find_unused_parameters=True,
            )
            random_input = torch.randn(batch, dim, device=self.rank)
            ones_input = torch.ones(batch, dim, device=self.rank)
            for i in range(6):
                if i % 2 == 0:
                    out = model(random_input)
                else:
                    out = model(ones_input)
                loss = out.sum()
                loss.backward()
                # On even iterations, 2nd param goes unused, on odd iterations,
                # it is used.
                local_used_maps = model.reducer._get_local_used_maps()
                if i % 2 == 0:
                    expected = torch.tensor([world_size, 0], device=self.rank, dtype=torch.int32)
                else:
                    expected = torch.tensor([world_size, world_size], device=self.rank, dtype=torch.int32)

                # Validate parameter usage.
                variable_usage_tensor = local_used_maps[0]
                self.assertEqual(variable_usage_tensor, expected)

            # Validate appropriate error message when DDP is used with
            # find_unused_parameters=False.
            model = torch.nn.parallel.DistributedDataParallel(
                ToyModel().cuda(self.rank),
                device_ids=[self.rank],
                find_unused_parameters=False,
            )
            for i in range(2):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Expected to have finished reduction in the prior iteration before starting a new one",
                ) if i == 1 else suppress():
                    loss = model(random_input).sum()
                    loss.backward()

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        @skip_if_rocm
        def test_ddp_control_flow_different_across_ranks(self):
            # Control flow that is different across ranks.
            batch = 20
            dim = 10

            class ToyModel(nn.Module):
                def __init__(self, rank):
                    super(ToyModel, self).__init__()
                    self.lin1 = nn.Linear(10, 10, bias=False)
                    self.lin2 = nn.Linear(10, 10, bias=False)
                    self.rank = rank

                def forward(self, x):
                    # Control-flow that is rank and input dependent for the
                    # model.
                    use_second_layer = (
                        torch.equal(x, torch.ones(batch, dim, device=x.device))
                        and self.rank == 1
                    )

                    if use_second_layer:
                        return self.lin2(F.relu(self.lin1(x)))
                    else:
                        return F.relu(self.lin1(x))

            world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            model = torch.nn.parallel.DistributedDataParallel(
                ToyModel(self.rank).cuda(self.rank),
                device_ids=[self.rank],
                find_unused_parameters=True,
            )
            random_input = torch.randn(batch, dim, device=self.rank)
            ones_input = torch.ones(batch, dim, device=self.rank)
            for i in range(6):
                if i % 2 == 0:
                    out = model(random_input)
                else:
                    out = model(ones_input)
                loss = out.sum()
                loss.backward()
                # On even iterations, 2nd param goes unused, on odd iterations,
                # it is used only on rank 1.
                local_used_maps = model.reducer._get_local_used_maps()

                if i % 2 == 0:
                    expected = torch.tensor([world_size, 0], device=self.rank, dtype=torch.int32)
                else:
                    expected = torch.tensor([world_size, 1], device=self.rank, dtype=torch.int32)

                variable_usage_tensor = local_used_maps[0]
                # Validate parameter usage. On odd iterations, 2nd param is only
                # used on rank 1.
                self.assertEqual(variable_usage_tensor, expected)

            # Validate appropriate error message when DDP is used with
            # find_unused_parameters=False.
            model = torch.nn.parallel.DistributedDataParallel(
                ToyModel(self.rank).cuda(self.rank),
                device_ids=[self.rank],
                find_unused_parameters=False,
            )
            for i in range(2):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Expected to have finished reduction in the prior iteration before starting a new one",
                ) if i == 1 else suppress():
                    loss = model(random_input).sum()
                    loss.backward()

        @require_backend({"gloo"})
        @unittest.skipIf(BACKEND == "nccl", "NCCL does not support scatter")
        def test_scatter_object_list(self):
            src_rank = 0
            scatter_list = (
                COLLECTIVES_OBJECT_TEST_LIST
                if self.rank == src_rank
                else [None for _ in COLLECTIVES_OBJECT_TEST_LIST]
            )
            world_size = dist.get_world_size()
            scatter_list = scatter_list[: world_size]
            i = 0
            while len(scatter_list) < world_size:
                scatter_list.append(scatter_list[i])
                i += 1

            output_obj_list = [None]
            dist.scatter_object_list(output_obj_list, scatter_list, src=src_rank)
            self.assertEqual(
                output_obj_list[0],
                COLLECTIVES_OBJECT_TEST_LIST[self.rank % len(COLLECTIVES_OBJECT_TEST_LIST)],
            )
            # Ensure errors are raised upon incorrect arguments.
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected argument scatter_object_output_list to be a list of size at least 1.",
            ):
                dist.scatter_object_list([], scatter_list, src=src_rank)
