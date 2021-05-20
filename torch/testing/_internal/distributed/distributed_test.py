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
from torch.testing._internal.common_utils import (
    IS_MACOS, IS_WINDOWS, FILE_SCHEMA, IS_FBCODE
)

import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
from torch.distributed.algorithms.ddp_comm_hooks import quantization as quantization_hooks
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.distributed_c10d import get_world_size, _get_default_group, AllreduceOptions, GroupMember
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
    initialize_temp_directories,
    cleanup_temp_dir,
    simple_sparse_reduce_tests,
    skip_if_rocm,
    skip_if_small_worldsize,
    skip_if_lt_x_gpu,
    nccl_skip_if_lt_x_gpu,
    skip_if_no_gpu,
    require_n_gpus_for_nccl_backend,
    requires_nccl_version,
    captured_output,
    with_nccl_blocking_wait,
    with_dist_debug_levels,
)
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.cuda.amp import GradScaler, autocast

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
    dist.Backend.MPI,
]

# Allowlist of distributed backends where profiling is supported with use_cuda=True
CUDA_PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.GLOO,
    dist.Backend.MPI,
    dist.Backend.NCCL,
]

# Allowlist of distributed backends where profiling is supported for p2p ops
SEND_RECV_PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.MPI,
    dist.Backend.GLOO,
    dist.Backend.NCCL,
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

def get_profiling_event(postfix, profiler):
    event_list = (
        profiler.events()
        if isinstance(profiler, torch.profiler.profile)
        else profiler.function_events
    )
    return [
        event for event in event_list if event.name.endswith(postfix)
    ]

# Base error message substring on unfinished reductions.
ddp_prev_reduction_unfinished_str = "Expected to have finished reduction in the prior iteration"
# Error message substring when find_unused_parameters=True has not been passed
ddp_recommend_find_unused_params_str = "passing the keyword argument `find_unused_parameters=True`"
# Error message substring when find_unused_parameters=True is enabled
ddp_find_unused_params_enabled_str = "Since `find_unused_parameters=True` is enabled"
# Error message substring for possibility of not all model outputs being used
# in loss computation
ddp_outputs_not_used_in_loss_str = "`forward` function outputs participate in calculating loss"
# Error message substring suggesting to use TORCH_DISTRIBUTED_DEBUG
ddp_suggest_debug_mode_str = "set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL"


class DDPUnevenTestInput(NamedTuple):
    name: str
    model: nn.Module
    inp: Union[torch.tensor, tuple]
    sync_interval: int
    throw_on_early_termination: bool = False


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

    def __init__(self, affine=True):
        super(BatchNormNet, self).__init__()
        self.fc1 = nn.Linear(2, 40, bias=False)
        self.bn = nn.BatchNorm1d(4, affine=affine)
        self.fc2 = nn.Linear(40, 4, bias=False)

    def forward(self, x):
        x = torch.reshape(self.fc1(x), (-1, 4, 10))
        x = self.bn(x)
        x = torch.reshape(x, (-1, 40))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class UnusedParamTwoLinLayerNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10, bias=False)
        self.b = nn.Linear(10, 10, bias=False)
        self.c = nn.Linear(5, 5, bias=False)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)

class DictOutputModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = UnusedParamTwoLinLayerNet()

    def forward(self, x):
        predictions = self.module(x)
        loss = (predictions[0] + predictions[1]).sum()
        return {
            "predictions": predictions,
            "loss": loss,
        }

class TwoLinLayerNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10, bias=False)
        self.b = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)

class EmbeddingNet(nn.Module):

    def __init__(self, rank):
        super().__init__()
        embedding_dim = 500 if rank == 0 else 50
        self.embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=embedding_dim
        )
        self.lin = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        return self.lin(x)

class ControlFlowToyModel(nn.Module):
    def __init__(self):
        super(ControlFlowToyModel, self).__init__()
        self.lin1 = nn.Linear(10, 10, bias=False)
        self.lin2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        # Second layer is used dependent on input x.
        use_second_layer = torch.equal(
            x, torch.ones(20, 10, device=x.device)
        )
        if use_second_layer:
            return self.lin2(F.relu(self.lin1(x)))
        else:
            return F.relu(self.lin1(x))

DDP_NET = Net()
BN_NET = BatchNormNet()
BN_NET_NO_AFFINE = BatchNormNet(affine=False)
ONLY_SBN_NET = nn.SyncBatchNorm(2, momentum=0.99)

def get_timeout(test_id):
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        return DEFAULT_TIMEOUT

default_pg_timeout = 60

CUSTOM_PG_TIMEOUT = {
    # This test runs slowly and needs additional time to complete, otherwise can
    # be taken down by NCCL_ASYNC_ERROR_HANDLING
    "test_ddp_uneven_inputs": 300,
    # This test has a short timeout since it tests being taken down by
    # NCCL_ASYNC_ERROR_HANDLING which we want to happen quickly.
    "test_ddp_model_diff_across_ranks": 5,
}


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

def _create_autograd_profiler():
    return torch.autograd.profiler.profile(record_shapes=True)

def _create_torch_profiler():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True
    )

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
        # NCCL_BLOCKING_WAIT overrides NCCL_ASYNC_ERROR_HANDLING hence tests
        # such as test_batch_isend_irecv_nccl will test NCCL_BLOCKING_WAIT as
        # expected.
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        super().setUpClass()

    def setUp(self):
        super().setUp()
        # initialize temp directories
        initialize_temp_directories()
        # initialize Barrier
        Barrier.init()
        # Skip return code checking for following tests as they are expected to
        # crash a process due to NCCL_ASYNC_ERROR_HANDLING.
        self.skip_return_code_checks = [
            self.test_ddp_model_diff_across_ranks.__wrapped__,
        ]

    def tearDown(self):
        cleanup_temp_dir()
        super().tearDown()

    @property
    def init_method(self):
        return "{}{file_name}".format(FILE_SCHEMA, file_name=self.file_name)

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        if BACKEND == 'nccl' and not torch.cuda.is_available():
            sys.exit(TEST_SKIPS['no_cuda'].exit_code)
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        if torch.cuda.is_available() and torch.cuda.device_count() < int(self.world_size):
            sys.exit(TEST_SKIPS['multi-gpu'].exit_code)
        try:
            pg_timeout_seconds = CUSTOM_PG_TIMEOUT.get(
                test_name, default_pg_timeout
            )
            timeout = timedelta(seconds=pg_timeout_seconds)
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

        self.run_test(test_name, pipe)
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
                self.assertGreaterAlmostEqual(time.time(), expected_time, delta=0.1)
            else:
                pass

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
        def _test_send_recv_nccl(self, profiler_ctx=None):
            # TODO: now that nccl send/recv is supported, there does not seem to
            # be a need to have nccl send/recv be tested separately.
            rank = dist.get_rank()
            rank_to_GPU = self._init_multigpu_helper()
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)

            tensor = _build_tensor(rank + 1, device_id=device_id)
            profiler_cls = profiler_ctx if profiler_ctx is not None else suppress()
            with profiler_cls as prof:
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

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    for event_name in [f"{backend}:send", f"{backend}:recv"]:
                        events = get_profiling_event(event_name, prof)
                        self.assertTrue(events)
                        # Event order is not deterministic, so simply assert their shape
                        # is found in the following list.
                        expected_shapes = [
                            [[rank + 1] * 3] for rank in range(dist.get_world_size())
                        ]
                        for event in events:
                            self.assertTrue(event.input_shapes in expected_shapes)

        @skip_if_no_gpu
        @unittest.skipIf(BACKEND != "nccl", "NCCL Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_send_recv_nccl(self):
            self._test_send_recv_nccl()

        @skip_if_no_gpu
        @unittest.skipIf(BACKEND != "nccl", "NCCL Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        def test_send_recv_nccl_autograd_profiler(self):
            profiler_ctx = torch.autograd.profiler.profile(record_shapes=True)
            self._test_send_recv_nccl(profiler_ctx)

        @skip_if_no_gpu
        @unittest.skipIf(BACKEND != "nccl", "NCCL Send Recv Only")
        @requires_nccl_version(2700, "Need NCCL 2.7+ for send/recv")
        @unittest.skipIf(IS_FBCODE, "Kineto in fbcode causes hang")
        @unittest.skipIf(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124"
        )
        def test_send_recv_nccl_torch_profiler(self):
            profiler_ctx = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True
            )
            self._test_send_recv_nccl(profiler_ctx)

        # SEND RECV
        def _test_send_recv(self, profiler_ctx):
            rank = dist.get_rank()
            send_size = rank + 1
            tensor = _build_tensor(send_size)
            ctx = profiler_ctx if profiler_ctx is not None else suppress()
            with ctx as prof:
                for src in range(0, dist.get_world_size()):
                    if src == rank:
                        # Send mode
                        for dst in range(0, dist.get_world_size()):
                            if dst == rank:
                                continue
                            dist.send(tensor, dst)
                    else:
                        # Recv mode
                        recv_size = src + 1
                        expected_tensor = _build_tensor(recv_size)
                        output_tensor = _build_tensor(recv_size, value=-1)
                        dist.recv(output_tensor, src)
                        self.assertEqual(output_tensor, expected_tensor)

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    for event_name in [f"{backend}:send", f"{backend}:recv"]:
                        events = get_profiling_event(event_name, prof)
                        # Each rank sends/recvs from all other ranks.
                        event_count = sum(e.count for e in events)
                        expected_event_count = dist.get_world_size() - 1
                        self.assertEqual(event_count, expected_event_count)
                        # Event order is not deterministic, so simply assert their shape
                        # is found in the following list.
                        expected_shapes = [
                            [[rank + 1] * 3] for rank in range(dist.get_world_size())
                        ]
                        for event in events:
                            self.assertTrue(event.is_async)
                            self.assertTrue(event.input_shapes in expected_shapes)

        @unittest.skipIf(BACKEND == "nccl", "Nccl send/recv tested by test_send_recv_nccl")
        def test_send_recv(self):
            self._test_send_recv(profiler_ctx=None)

        @unittest.skipIf(BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl")
        def test_send_recv_autograd_profiler(self):
            autograd_profiler_ctx = _create_autograd_profiler()
            self._test_send_recv(profiler_ctx=autograd_profiler_ctx)

        @unittest.skipIf(BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl")
        @unittest.skipIf(IS_FBCODE, "Kineto in fbcode causes hang")
        @unittest.skipIf(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124"
        )
        def test_send_recv_torch_profiler(self):
            torch_profiler_ctx = _create_torch_profiler()
            return self._test_send_recv(profiler_ctx=torch_profiler_ctx)

        # SEND RECV ANY SOURCE
        def _test_send_recv_any_source(self, profiler_ctx):
            rank = dist.get_rank()
            send_recv_size = 10
            tensor = _build_tensor(send_recv_size, value=rank)
            recv_ranks = list()
            irecv_ranks = list()

            ctx = profiler_ctx if profiler_ctx is not None else suppress()
            with ctx as prof:
                for dst in range(0, dist.get_world_size()):
                    if dst == rank:
                        # Recv mode
                        for dst in range(0, dist.get_world_size()):
                            if dst == rank:
                                continue

                            for recv in ["recv", "irecv"]:
                                output_tensor = _build_tensor(send_recv_size, value=-1)

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

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    for event_name in [f"{backend}:send", f"{backend}:recvAnySource"]:
                        events = get_profiling_event(event_name, prof)
                        # Each rank sends/recvs from other rank twice.
                        self.assertEqual(sum(event.count for event in events), 2 * (dist.get_world_size() - 1))
                        for event in events:
                            self.assertTrue(event.is_async)
                            self.assertEqual(event.input_shapes, [[send_recv_size] * 3])

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

        @unittest.skipIf(
            BACKEND == "nccl", "Nccl does not support send/recv from any source"
        )
        def test_send_recv_any_source(self):
            self._test_send_recv_any_source(profiler_ctx=None)

        @unittest.skipIf(
            BACKEND == "nccl", "Nccl does not support send/recv from any source"
        )
        def test_send_recv_any_source_autograd_profiler(self):
            autograd_profiler_ctx = _create_autograd_profiler()
            self._test_send_recv_any_source(profiler_ctx=autograd_profiler_ctx)

        @unittest.skipIf(
            BACKEND == "nccl", "Nccl does not support send/recv from any source"
        )
        @unittest.skipIf(IS_FBCODE, "Kineto in fbcode code causes hang")
        @unittest.skipIf(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124"
        )
        def test_send_recv_any_source_torch_profiler(self):
            torch_profiler_ctx = _create_torch_profiler()
            return self._test_send_recv_any_source(profiler_ctx=torch_profiler_ctx)

        # SEND RECV WITH TAG
        def _test_send_recv_with_tag(self, profiler_ctx):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            send_recv_size = 10
            tensor = _build_tensor(send_recv_size, value=rank)
            ctx = profiler_ctx if profiler_ctx is not None else suppress()
            with ctx as prof:
                for dst in range(0, world_size):
                    if dst == rank:
                        # Recv mode
                        for src in range(0, world_size):
                            if src == rank:
                                continue
                            output_tensor = _build_tensor(send_recv_size, value=-1)
                            dist.recv(output_tensor, src, tag=src)
                            self.assertTrue(output_tensor.eq(src).all())
                    else:
                        # Send mode
                        dist.send(tensor, dst, tag=rank)

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    for event_name in [f"{backend}:send", f"{backend}:recv"]:
                        events = get_profiling_event(event_name, prof)
                        # Each rank sends/recvs from all other ranks
                        event_count = sum(e.count for e in events)
                        expected_event_count = dist.get_world_size() - 1
                        self.assertEqual(event_count, expected_event_count)
                        for event in events:
                            self.assertTrue(event.is_async)
                            self.assertEqual(event.name, event_name)
                            self.assertEqual(event.input_shapes, [[send_recv_size] * 3])

        @unittest.skipIf(BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl")
        def test_send_recv_with_tag(self):
            self._test_send_recv_with_tag(profiler_ctx=None)

        @unittest.skipIf(BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl")
        def test_send_recv_with_tag_autograd_profiler(self):
            autograd_profiler_ctx = _create_autograd_profiler()
            return self._test_send_recv_with_tag(profiler_ctx=autograd_profiler_ctx)

        @unittest.skipIf(BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl")
        @unittest.skipIf(IS_FBCODE, "Kineto in fbcode code causes hang")
        @unittest.skipIf(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124"
        )
        def test_send_recv_with_tag_torch_profiler(self):
            torch_profiler_ctx = _create_torch_profiler()
            return self._test_send_recv_with_tag(profiler_ctx=torch_profiler_ctx)

        # ISEND
        def _test_isend(self, profiler_ctx):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            ctx = profiler_ctx if profiler_ctx is not None else suppress()
            with ctx as prof:
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

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    expected_event_name = f"{backend}:send" if rank == 0 else f"{backend}:recv"
                    events = get_profiling_event(expected_event_name, prof)
                    event_count = sum(e.count for e in events)
                    expected_count = dist.get_world_size() - 1 if rank == 0 else 1
                    self.assertEqual(expected_count, event_count)
                    # Event ordering is not guaranteed, so simply ensure the shapes are
                    # found in the following map.
                    expected_shapes = {
                        r: [[r] * 3] for r in range(1, dist.get_world_size())
                    }
                    for event in events:
                        self.assertTrue(event.is_async)
                        self.assertEqual(event.name, expected_event_name)
                        if rank == 0:
                            self.assertTrue(event.input_shapes in expected_shapes.values())
                        else:
                            self.assertEqual(event.input_shapes, expected_shapes[rank])

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support isend")
        def test_isend(self):
            self._test_isend(profiler_ctx=None)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support isend")
        def test_isend_autograd_profiler(self):
            autograd_profiler_ctx = _create_autograd_profiler()
            self._test_isend(profiler_ctx=autograd_profiler_ctx)

        @unittest.skipIf(BACKEND == "nccl", "Nccl does not support isend")
        @unittest.skipIf(IS_FBCODE, "Kineto in fbcode code causes hang")
        @unittest.skipIf(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124"
        )
        def test_isend_torch_profiler(self):
            torch_profiler_ctx = _create_torch_profiler()
            self._test_isend(profiler_ctx=torch_profiler_ctx)

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
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
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
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)

            new_port = str(MASTER_PORT + 1)
            os.environ['MASTER_PORT'] = new_port
            gen_iterator = dist.rendezvous('env://', rank, dist.get_world_size())
            store, rank, size = next(gen_iterator)
            store = dist.PrefixStore(new_port, store)

            opts = dist.ProcessGroupNCCL.Options()
            opts.is_high_priority_stream = False
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
                self.call_dist_op(":reduce", False, dist.reduce, tensor, src, op, group_id, tensor_shapes=[tensor.shape])
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
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
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
                self.call_dist_op(
                    ":reduce",
                    False,
                    dist.reduce,
                    tensors[0],
                    src,
                    op,
                    group_id,
                    secondary_op_call=lambda: dist.reduce(tensors[1], src, op, group_id),
                    tensor_shapes=[tensors[0].shape]
                )
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
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
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
            tensor_shapes=None,
            **kwargs,
        ):
            op_calls = [lambda: op(*args, **kwargs)]
            if secondary_op_call is not None:
                op_calls.append(secondary_op_call)

            autograd_profiler_ctx = torch.autograd.profiler.profile(
                use_cuda=profile_cuda,
                record_shapes=True
            )

            # TODO: move this test to use torch.profiler once kineto issues are
            # fixed internally.
            with autograd_profiler_ctx as prof:
                works = [op_call() for op_call in op_calls]
                if is_async:
                    for work in works:
                        work.wait()

            if expect_event and dist.get_backend() in PROFILING_SUPPORTED_BACKENDS:
                events = get_profiling_event(profiling_title_postfix, autograd_profiler_ctx)
                self.assertEqual(len(events), len(op_calls))
                for e in events:
                    self.assertTrue(e.is_async)
                    self.assertEqual(e.count, 1)
                    self.assertGreaterEqual(e.cpu_time, 0)
                    # Verify tensor shapes if given
                    if tensor_shapes is not None:
                        self.assertEqual(e.input_shapes, tensor_shapes, f"event shape: {e.input_shapes} vs tensor {tensor_shapes}")

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
                if tensor.dtype == torch.complex64:
                    tensor_shapes = [torch.view_as_real(tensor).shape]
                else:
                    tensor_shapes = [tensor.shape]
                self.call_dist_op(
                    ":all_reduce",
                    async_op,
                    dist.all_reduce,
                    tensor,
                    op,
                    group_id,
                    async_op=async_op,
                    tensor_shapes=tensor_shapes
                )
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
                        tensor_shapes=tensor_shapes,
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
            torch.cuda.set_device(self.rank)
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
            torch.cuda.set_device(self.rank)
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
            torch.cuda.set_device(self.rank)
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
                tensor_shapes = []
                for tensor in tensors:
                    if tensor.dtype == torch.complex64:
                        tensor_shapes.append(torch.view_as_real(tensor).shape)
                    else:
                        tensor_shapes.append(tensor.shape)
                self.call_dist_op(
                    ":all_reduce",
                    False,
                    dist.all_reduce_coalesced,
                    tensors,
                    op,
                    group_id,
                    tensor_shapes=tensor_shapes
                )
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
                self.call_dist_op(
                    ":scatter",
                    False,
                    dist.scatter,
                    tensor,
                    src=dest,
                    scatter_list=tensors,
                    group=group_id,
                    tensor_shapes=[t.shape for t in tensors]
                )
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
                self.call_dist_op(
                    ":gather",
                    False,
                    dist.gather,
                    tensor,
                    dst=dest,
                    gather_list=tensors,
                    group=group_id,
                    tensor_shapes=[tensors[0].shape] if len(tensors) > 0 else None
                )
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
                if tensors[0].dtype == torch.complex64:
                    tensor_shapes = [torch.view_as_real(tensors[0]).shape]
                else:
                    tensor_shapes = [tensors[0].shape]
                self.call_dist_op(":all_gather", False, dist.all_gather, tensors, tensor, group_id, tensor_shapes=tensor_shapes)

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
            tensor_shapes = []
            for input_tensor in input_tensors:
                if input_tensor.dtype == torch.complex64:
                    tensor_shapes.append(torch.view_as_real(input_tensor).shape)
                else:
                    tensor_shapes.append(input_tensor.shape)
            self.call_dist_op(":all_gather", False, dist.all_gather_coalesced,
                              output_tensor_lists, input_tensors, group_id, tensor_shapes=tensor_shapes)

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
                self.call_dist_op(
                    ":all_to_all",
                    False,
                    dist.all_to_all_single,
                    out_tensor,
                    in_tensor,
                    group=group_id,
                    tensor_shapes=[in_tensor.shape]
                )
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
                    self.assertGreaterAlmostEqual(
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
                    expect_event=len(tensors) == 1, tensor_shapes=[tensors[0].shape])
                if rank == src:
                    expected_tensor = _build_tensor(src + 1, expected_value)
                    self.assertEqual(tensors[0], expected_tensor)

            self._barrier()

        @unittest.skipIf(BACKEND != "nccl", "Only Nccl backend supports reduce multigpu")
        @skip_if_no_gpu
        def test_reduce_multigpu(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
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
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            self._test_all_gather_multigpu_helper(group, group_id, rank, rank_to_GPU)

        @unittest.skipIf(BACKEND != "nccl", "Only Nccl backend supports allgather multigpu")
        @skip_if_no_gpu
        def test_all_gather_multigpu_complex(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
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
        def _test_DDP_helper(self, model, input_var, target, loss, scale_factor=1.0, memory_format=None):
            model.train()
            output = model(input_var)
            l = loss(output, target) * scale_factor
            l.backward()
            if memory_format is not None:
                self.assertTrue(output.is_contiguous(memory_format=memory_format))

        def _assert_equal_param(self, param_gpu, param_DDP):
            self.assertEqual(len(param_gpu), len(param_DDP))
            for p_gpu, p_DDP in zip(param_gpu, param_DDP):
                self.assertEqual(p_gpu, p_DDP)

        def _test_DDP_niter(
            self, model_base, model_DDP, input, target, loss, local_bs, rank, batch_size, test_save,
            offset=None, world_size=0, zero_grad=False, memory_format=None, n_iter=5,
        ):
            for idx in range(n_iter):
                # single cpu/gpu training
                self._test_DDP_helper(model_base, input, target, loss, memory_format=memory_format)

                if offset is None:
                    offset = rank * local_bs

                # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
                self._test_DDP_helper(
                    model_DDP,
                    input[offset: offset + local_bs],
                    target[offset: offset + local_bs],
                    loss,
                    world_size * local_bs / batch_size if world_size != 0 else 1,
                    memory_format=memory_format
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

        def _test_DistributedDataParallel(self,
                                          gpu_subset,
                                          rank,
                                          output_device=None,
                                          gradient_as_bucket_view=False,
                                          static_graph=False):
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
            if static_graph:
                model_DDP._set_static_graph()

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
            self._test_DDP_niter(
                model_gpu,
                model_DDP,
                input_cpu.cuda(gpu_subset[0]),
                target.cuda(gpu_subset[0]),
                loss,
                local_bs,
                rank,
                global_bs,
                True
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
            self._test_DDP_niter(
                model_base,
                model_DDP,
                input_cpu,
                target,
                loss,
                local_bs,
                rank,
                global_bs,
                False,
                zero_grad=True
            )
            self._barrier()

            return model_DDP

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

        @unittest.skipIf(
            BACKEND != "nccl",
            "Only NCCL backend supports DDP communication hook",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm
        def test_ddp_comm_hook_logging(self):
            hooks = [
                default.allreduce_hook,
                default.fp16_compress_hook,
                powerSGD.powerSGD_hook,
                powerSGD.batched_powerSGD_hook,
                quantization_hooks.quantization_pertensor_hook,
                quantization_hooks.quantization_perchannel_hook,
            ]

            cpp_builtin_hooks = [
                dist.BuiltinCommHookType.ALLREDUCE,
                dist.BuiltinCommHookType.FP16_COMPRESS,
            ]

            for hook in hooks:
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    torch.nn.Linear(1, 1, bias=False).cuda(self.rank),
                    device_ids=[self.rank]
                )
                ddp_logging_data = ddp_model._get_ddp_logging_data()
                # Hook not registered yet, so should be empty
                self.assertEqual(ddp_logging_data.get("comm_hook"), None)
                ddp_model.register_comm_hook(None, hook)
                ddp_logging_data = ddp_model._get_ddp_logging_data()
                self.assertEqual(ddp_logging_data.get("comm_hook"), hook.__qualname__)

            for hook in cpp_builtin_hooks:
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    torch.nn.Linear(1, 1, bias=False).cuda(self.rank),
                    device_ids=[self.rank]
                )
                ddp_logging_data = ddp_model._get_ddp_logging_data()
                # Hook not registered yet, so should be empty
                self.assertEqual(ddp_logging_data.get("comm_hook"), None)
                ddp_model._register_builtin_comm_hook(hook)
                ddp_logging_data = ddp_model._get_ddp_logging_data()
                self.assertEqual(ddp_logging_data.get("comm_hook"), str(hook))

            # No hook registered
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                torch.nn.Linear(1, 1, bias=False).cuda(self.rank),
                device_ids=[self.rank]
            )
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            # Hook not registered yet, so should be empty
            self.assertEqual(ddp_logging_data.get("comm_hook"), None)
            # After second forward pass, hook should still be empty string
            for i in range(2):
                inp = torch.ones(1, 1, device=self.rank)
                loss = ddp_model(inp).sum()
                loss.backward()

            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertEqual(ddp_logging_data.get("comm_hook"), None)

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
                    start_powerSGD_iter=2,
                    warm_start=warm_start,
                )
                self._test_ddp_hook_parity(state=powersgd_state, hook=powerSGD.powerSGD_hook)

        def _prepare_single_device_module(
                self,
                rank,
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view=False,
        ):
            model = Net()
            device = devices[0] if devices else torch.device("cuda:%d" % rank)
            ddp_model = DistributedDataParallel(
                copy.deepcopy(model).to(device),
                device_ids=device_ids,
                process_group=process_group,
                bucket_cap_mb=0.001,
                gradient_as_bucket_view=gradient_as_bucket_view,
            )

            model.to(device)

            input = torch.randn(global_batch_size, 2).to(device)
            target = torch.randn(global_batch_size, 4).to(device)

            return model, ddp_model, input, target

        def _prepare_cpu_module(
            self,
            process_group,
            global_batch_size,
            gradient_as_bucket_view=False,
        ):
            model = Net()
            ddp_model = DistributedDataParallel(
                copy.deepcopy(model),
                process_group=process_group,
                bucket_cap_mb=0.001,
                gradient_as_bucket_view=gradient_as_bucket_view,
            )
            input = torch.randn(global_batch_size, 2)
            target = torch.randn(global_batch_size, 4)
            return model, ddp_model, input, target

        def _test_accumulate_gradients_no_sync(
                self, num_iters=2, ddp_comm_hook=None, gradient_as_bucket_view=False
        ):
            """
            This is the recommended way to implement accumulate grads.
            If ``ddp_comm_hook`` input was specified, it will also register that hook
            to the ``ddp_model``. The hook fed into this function should not change
            the resulting gradients.
            """
            group, group_id, rank = self._init_global_test()
            world_size = get_world_size()

            # FIXME: Add testing for gloo/CUDA
            if BACKEND == "mpi" or BACKEND == "gloo":
                global_batch_size = world_size
                local_batch_size = 1
                model, ddp_model, input, target = self._prepare_cpu_module(
                    group_id, global_batch_size, gradient_as_bucket_view
                )

            if BACKEND == "nccl":
                rank_to_GPU = self._init_multigpu_helper()
                int_devices = rank_to_GPU[rank][:1]
                devices = [torch.device("cuda:" + str(i)) for i in int_devices]
                global_batch_size = world_size
                local_batch_size = len(devices)
                model, ddp_model, input, target = self._prepare_single_device_module(
                    rank, group_id, devices, devices, global_batch_size, gradient_as_bucket_view
                )

            if ddp_comm_hook is not None:
                ddp_model.register_comm_hook(group_id, ddp_comm_hook)

            def step_model(model, input, target):
                model.train()
                output = model(input)
                loss = F.mse_loss(output, target.to(output.device))
                loss.backward()

            # ensure accumulate grads works with no_grad => no grads are accumulated.
            with torch.no_grad():
                with ddp_model.no_sync():
                    ddp_model.train()
                    ddp_model(input)

            # check two model parameters over num_iters iterations
            for iteration in range(num_iters):
                step_model(model, input, target)

                ddp_input = input[
                    rank * local_batch_size: (rank + 1) * local_batch_size
                ]
                ddp_target = target[
                    rank * local_batch_size: (rank + 1) * local_batch_size
                ]

                if iteration % num_iters == 0:
                    # accumulate grads locally
                    with ddp_model.no_sync():
                        step_model(ddp_model, ddp_input, ddp_target)
                else:
                    # sync grads
                    step_model(ddp_model, ddp_input, ddp_target)

                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    if not i.requires_grad:
                        continue
                    if iteration % num_iters == 0:
                        self.assertNotEqual(i.grad, j.grad)
                    else:
                        self.assertEqual(i.grad, j.grad)


                # Shuffle the input so that DDP input is different
                torch.manual_seed(1337 + iteration)
                input = input[torch.randperm(global_batch_size)]


        @unittest.skipIf(
            BACKEND != "mpi" and BACKEND != "nccl" and BACKEND != "gloo",
            "get_future is only supported on mpi, nccl and gloo"
        )
        @nccl_skip_if_lt_x_gpu(BACKEND, 2)
        def test_accumulate_gradients_no_sync(self):
            """
            Runs _test_accumulate_gradients_no_sync using default inputs
            """
            self._test_accumulate_gradients_no_sync()

        @unittest.skipIf(
            BACKEND != "mpi" and BACKEND != "nccl" and BACKEND != "gloo",
            "get_future is only supported on mpi, nccl and gloo"
        )
        @nccl_skip_if_lt_x_gpu(BACKEND, 2)
        def test_accumulate_gradients_no_sync_grad_is_view(self):
            """
            Runs _test_accumulate_gradients_no_sync using default inputs
            """
            self._test_accumulate_gradients_no_sync(gradient_as_bucket_view=True)

        @unittest.skipIf(
            BACKEND != "mpi" and BACKEND != "nccl" and BACKEND != "gloo",
            "get_future is only supported on mpi, nccl and gloo"
        )
        @nccl_skip_if_lt_x_gpu(BACKEND, 2)
        def test_accumulate_gradients_no_sync_allreduce_hook(self):
            """
            Runs multiple iterations on _test_accumulate_gradients_no_sync
            using allreduce hook and validates whether future result was properly
            passed as gradients in reducer.
            """

            world_size = get_world_size()

            def allreduce_hook(
                    group_id: object, bucket: dist.GradBucket
            ) -> torch._C.Future:
                tensors = [bucket.get_tensor() / world_size]
                return group_id.allreduce(tensors).get_future()

            self._test_accumulate_gradients_no_sync(
                num_iters=4, ddp_comm_hook=allreduce_hook
            )

        @unittest.skipIf(
            BACKEND != "mpi" and BACKEND != "nccl" and BACKEND != "gloo",
            "get_future is only supported on mpi, nccl and gloo"
        )
        @nccl_skip_if_lt_x_gpu(BACKEND, 2)
        def test_accumulate_gradients_no_sync_allreduce_with_then_hook(self):
            """
            Runs multiple iterations on _test_accumulate_gradients_no_sync using allreduce
            hook that also uses then callbacks. In first then callback result is multiplied
            by 2, and the second callback divides the result by 2 * world_size. It validates
            whether final result was properly passed as gradients in reducer.
            """

            world_size = get_world_size()

            def allreduce_with_then_hook(
                    group_id: object, bucket: dist.GradBucket
            ) -> torch.futures.Future:
                fut = group_id.allreduce([bucket.get_tensor()]).get_future()

                def mult(fut):
                    # Multiply the result by 2.
                    return [2 * t for t in fut.wait()]

                def div(fut):
                    # Divide the result by 2 * world_size.
                    return [t / (2 * world_size) for t in fut.wait()]

                return fut.then(mult).then(div)

            self._test_accumulate_gradients_no_sync(
                num_iters=4, ddp_comm_hook=allreduce_with_then_hook
            )

        @unittest.skipIf(
            BACKEND != "mpi" and BACKEND != "nccl" and BACKEND != "gloo",
            "get_future is only supported on mpi, nccl and gloo"
        )
        @nccl_skip_if_lt_x_gpu(BACKEND, 2)
        def test_get_future(self):
            def mult(fut):
                return [t * 3 for t in fut.wait()]

            def add(fut):
                return [t + 1 for t in fut.wait()]

            group, group_id, rank = self._init_global_test()
            input = _build_tensor(3, 2)
            if BACKEND == "nccl":
                rank_to_GPU = self._init_multigpu_helper()
                device_id = rank_to_GPU[rank][0]
                input = input.to(device_id)
            fut = group_id.allreduce([input]).get_future()
            res = fut.then(mult).then(add).wait()
            expected = _build_tensor(3, 2 * len(group) * 3 + 1)

            self.assertEqual(res[0], expected)


        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = self._init_multigpu_helper()
            gpus = list(rank_to_GPU[rank])

            for use_bucket_view, static_graph in itertools.product((False, True), (False, True)):
                self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank, gradient_as_bucket_view=use_bucket_view,
                                                   static_graph=static_graph)

                # test output_device
                self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank, output_device=torch.device('cuda'),
                                                   gradient_as_bucket_view=use_bucket_view, static_graph=static_graph)

                # test device_ids
                gpus_list = [torch.device('cuda:' + str(i)) for i in gpus]
                self._test_DistributedDataParallel(gpu_subset=gpus_list, rank=rank, output_device=torch.device('cuda'),
                                                   gradient_as_bucket_view=use_bucket_view, static_graph=static_graph)

        def _test_DistributedDataParallel_with_amp(self, grad_is_view=False):
            torch.manual_seed(31415)
            # Creates model and optimizer in default precision
            model = copy.deepcopy(DDP_NET).cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

            # Creates a GradScaler once at the beginning of training.
            scaler = GradScaler()

            ddp_model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.rank], gradient_as_bucket_view=grad_is_view)

            input = torch.randn(dist.get_world_size() * 2, 2).cuda()
            target = torch.randn(dist.get_world_size() * 2, 4).cuda()
            loss_fn = nn.MSELoss()

            # verify grads are none before training
            for p in ddp_model.parameters():
                self.assertTrue(p is not None)
                self.assertTrue(p.grad is None)

            for idx in range(20):
                optimizer.zero_grad()
                # Runs the forward pass with autocasting.
                with autocast():
                    output = ddp_model(input)
                    loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()

                # verify grads are not none and are valid during training
                for p in ddp_model.parameters():
                    if p.requires_grad:
                        self.assertTrue(p.grad is not None)
                        self.assertFalse(p.grad.isnan().any())
                        self.assertFalse(p.grad.isinf().any())

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                # Shuffle the input so that DDP input is different
                torch.manual_seed(1337 + idx)
                input = input[torch.randperm(dist.get_world_size() * 2)]

            return ddp_model

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel_with_amp_and_grad_is_view(self):
            torch.cuda.set_device(self.rank)
            ddp_model_grad_not_view = self._test_DistributedDataParallel_with_amp(grad_is_view=False)
            ddp_model_grad_is_view = self._test_DistributedDataParallel_with_amp(grad_is_view=True)
            for i, j in zip(ddp_model_grad_not_view.parameters(), ddp_model_grad_is_view.parameters()):
                self.assertEqual(i, j)

        def _test_DistributedDataParallel_SyncBatchNorm(self, gpu_subset, rank, local_bs, global_bs, offset,
                                                        output_device=None, affine=True):
            # Run a simple end to end DDP model, use result of single node model
            # as baseline

            # cpu training setup
            model = BN_NET if affine else BN_NET_NO_AFFINE

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
            self._test_DDP_niter(
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
                dist.get_world_size(),
                5 if affine else 2,
            )
            self._barrier()

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_DistributedDataParallel_SyncBatchNorm_Channels_Last(self):
            group, group_id, rank = self._init_global_test()
            num_processes = dist.get_world_size()
            local_bs = 2
            bs_offset = int(rank * 2)
            global_bs = int(num_processes * 2)

            model = ONLY_SBN_NET
            model_gpu = copy.deepcopy(model).cuda(rank)
            model_DDP = nn.parallel.DistributedDataParallel(
                model_gpu, device_ids=[rank]
            )

            memory_format = torch.channels_last
            input_gpu = torch.randn(global_bs, 2, 4, 4, dtype=torch.float).cuda(rank).to(memory_format=memory_format)
            target_gpu = torch.randn(global_bs, 2, 4, 4, dtype=torch.float).cuda(rank).to(memory_format=memory_format)
            loss = nn.MSELoss()

            # check two model parameters over 5 iterations
            self._test_DDP_niter(
                model_gpu,
                model_DDP,
                input_gpu,
                target_gpu,
                loss,
                local_bs,
                rank,
                global_bs,
                True,
                bs_offset,
                dist.get_world_size(),
                memory_format=memory_format
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
        def test_DistributedDataParallel_SyncBatchNorm_No_Affine(self):
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
                offset=bs_offset,
                affine=False)


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
                self._test_DDP_niter(
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
                self._test_DDP_niter(
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

        def _test_ddp_logging_data(self, is_gpu):
            rank = dist.get_rank()
            model_DDP = copy.deepcopy(DDP_NET)
            if is_gpu:
                model_DDP = nn.parallel.DistributedDataParallel(model_DDP.cuda(rank), device_ids=[rank])
            else:
                model_DDP = nn.parallel.DistributedDataParallel(model_DDP)

            # dummy data initialization
            local_bs = 2
            batch_size, input, target, loss = self._prepare_dummy_data(local_bs)
            if is_gpu:
                input = input.cuda(rank)
                target = target.cuda(rank)

            model_DDP._set_ddp_runtime_logging_sample_rate(2)

            for idx in range(20):
                offset = rank * local_bs

                # DDP training, DDP scatters subsets of input to nodes/GPUs
                self._test_DDP_helper(
                    model_DDP,
                    input[offset: offset + local_bs],
                    target[offset: offset + local_bs],
                    loss,
                    1,
                )

                self._model_step_with_zero_grad(model_DDP)

                # Verify DDP logging data is sampled as expected
                # If it has ran more than 10 iteratons and this is
                # the sampled iteration for measuring run time stats,
                # the run time stats for this idx-th iteration will not
                # be zeros.
                ddp_logging_data = model_DDP._get_ddp_logging_data()
                if (idx > 0 and (idx < 10 or idx % 2 == 0)):
                    self.assertGreaterEqual(ddp_logging_data.get("forward_compute_time"), 1)
                    self.assertGreaterEqual(ddp_logging_data.get("backward_compute_time"), 1)
                    self.assertGreaterEqual(ddp_logging_data.get("backward_comm_time"), 1)
                    self.assertGreaterEqual(
                        ddp_logging_data.get("backward_compute_time"),
                        ddp_logging_data.get("backward_compute_comm_overlap_time"))
                    self.assertGreaterEqual(
                        ddp_logging_data.get("backward_comm_time"),
                        ddp_logging_data.get("backward_compute_comm_overlap_time"))
                    self.assertEqual(ddp_logging_data.get("iteration"), idx)
                elif idx > 0:
                    # if the idx-th iteration is not sampled to set runtime stats,
                    # ddp_logging_data.iteration will not be updated to current
                    # iteration.
                    self.assertNotEqual(ddp_logging_data.get("iteration"), idx)

                # Shuffle the input so that DDP input is different
                input = input[torch.randperm(batch_size)]

            return model_DDP

        @unittest.skipIf(
            BACKEND == "nccl", "nccl does not support DDP on CPU models"
        )
        def test_ddp_logging_data_cpu(self):
            def parse_env(var):
                return os.environ[var] if var in os.environ else "N/A"

            group, group_id, rank = self._init_global_test()
            model_DDP = self._test_ddp_logging_data(is_gpu=False)

            ddp_logging_data = model_DDP._get_ddp_logging_data()
            self.assertEqual(ddp_logging_data.get("world_size"), dist.get_world_size())
            self.assertEqual(ddp_logging_data.get("rank"), dist.get_rank())
            self.assertEqual(ddp_logging_data.get("module_name"), 'Net')
            self.assertEqual(ddp_logging_data.get("device_ids"), "")
            # output_device is -1 in default if it is not set, e.g.
            # output_device of CPU training is -1.
            self.assertEqual(ddp_logging_data.get("output_device"), -1)
            self.assertEqual(ddp_logging_data.get("broadcast_buffers"), 1)
            self.assertEqual(ddp_logging_data.get("bucket_cap_bytes"), 25 * 1024 * 1024)
            self.assertEqual(ddp_logging_data.get("find_unused_parameters"), 0)
            self.assertEqual(ddp_logging_data.get("gradient_as_bucket_view"), 0)
            self.assertEqual(ddp_logging_data.get("backend_name"), dist.get_backend(group_id))
            self.assertEqual(ddp_logging_data.get("iteration"), 18)
            params = list(model_DDP.parameters())
            num_params = 0
            param_size = 0
            params = list(parameter for parameter in filter(lambda parameter: parameter.requires_grad, params))
            for p in params:
                num_params += 1
                param_size += p.numel() * p.element_size()
            self.assertEqual(ddp_logging_data.get("dtypes"), "float")
            self.assertEqual(ddp_logging_data.get("total_parameter_size_bytes"), param_size)
            self.assertEqual(ddp_logging_data.get("num_parameter_tensors"), num_params)
            self.assertEqual(ddp_logging_data.get("bucket_sizes"), str(param_size))
            self.assertEqual(ddp_logging_data.get("master_port"), parse_env("MASTER_PORT"))
            self.assertEqual(ddp_logging_data.get("master_addr"), parse_env("MASTER_ADDR"))
            self.assertEqual(ddp_logging_data.get("cuda_visible_devices"), parse_env("CUDA_VISIBLE_DEVICES"))
            if ddp_logging_data.get("backend_name") == "gloo":
                self.assertEqual(ddp_logging_data.get("gloo_socket_ifname"), parse_env("GLOO_SOCKET_IFNAME"))
                self.assertEqual(ddp_logging_data.get("gloo_device_transport"), parse_env("GLOO_DEVICE_TRANSPORT"))
            self.assertEqual(ddp_logging_data.get("nccl_socket_ifname"), None)
            self.assertEqual(ddp_logging_data.get("nccl_blocking_wait"), None)
            self.assertEqual(ddp_logging_data.get("nccl_async_error_handling"), None)
            self.assertEqual(ddp_logging_data.get("nccl_debug"), None)
            self.assertEqual(ddp_logging_data.get("nccl_nthreads"), None)
            self.assertEqual(ddp_logging_data.get("nccl_ib_timeout"), None)
            # test runtime logging fields
            self.assertEqual(ddp_logging_data.get("unused_parameter_size"), None)
            self.assertEqual(ddp_logging_data.get("has_rebuilt_buckets"), 1)
            self.assertEqual(ddp_logging_data.get("rebuilt_bucket_sizes"), str(param_size))
            # It is hard to test accurate latency, but it can test whether the latency is
            # a valid value and in the expected range.
            self.assertGreaterEqual(ddp_logging_data.get("avg_forward_compute_time"), 1)
            self.assertGreaterEqual(ddp_logging_data.get("avg_backward_compute_time"), 1)
            self.assertGreaterEqual(ddp_logging_data.get("avg_backward_comm_time"), 1)
            self.assertGreaterEqual(
                ddp_logging_data.get("avg_backward_compute_time"),
                ddp_logging_data.get("avg_backward_compute_comm_overlap_time"))
            self.assertGreaterEqual(
                ddp_logging_data.get("avg_backward_comm_time"),
                ddp_logging_data.get("avg_backward_compute_comm_overlap_time"))
            # test larger net with mixed data types, verify multiple bucket sizes
            model = LargeNet()
            model.float()
            model.fc1.double()
            model_DDP = nn.parallel.DistributedDataParallel(model, bucket_cap_mb=1.5)
            ddp_logging_data = model_DDP._get_ddp_logging_data()
            params = list(model_DDP.parameters())
            self.assertEqual(ddp_logging_data.get("bucket_cap_bytes"), int(1.5 * 1024 * 1024))
            bucket_sizes = [params[1].numel() * params[1].element_size(), params[0].numel() * params[0].element_size()]
            self.assertEqual(
                ddp_logging_data.get("bucket_sizes"),
                ', '.join(str(x) for x in bucket_sizes))
            self.assertEqual(ddp_logging_data.get("dtypes"), 'double, float')

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_no_gpu
        def test_ddp_logging_data_gpu(self):
            group, group_id, rank = self._init_global_test()
            model_DDP = self._test_ddp_logging_data(is_gpu=True)
            ddp_logging_data = model_DDP._get_ddp_logging_data()
            self.assertEqual(ddp_logging_data.get("device_ids"), str(rank))
            self.assertEqual(ddp_logging_data.get("output_device"), rank)
            # test runtime logging fields
            # It is hard to test accurate latency, but it can test whether the latency is
            # a valid value and in the expected range.
            self.assertGreaterEqual(ddp_logging_data.get("avg_forward_compute_time"), 1)
            self.assertGreaterEqual(ddp_logging_data.get("avg_backward_compute_comm_overlap_time"), 1)
            self.assertGreaterEqual(
                ddp_logging_data.get("avg_backward_compute_time"),
                ddp_logging_data.get("avg_backward_compute_comm_overlap_time"))
            self.assertGreaterEqual(
                ddp_logging_data.get("avg_backward_comm_time"),
                ddp_logging_data.get("avg_backward_compute_comm_overlap_time"))

        @unittest.skipIf(
            BACKEND == "nccl", "nccl does not support DDP on CPU models"
        )
        def test_static_graph_api_cpu(self):
            model_DDP = nn.parallel.DistributedDataParallel(DDP_NET)
            model_DDP._set_static_graph()
            self.assertEqual(model_DDP._get_ddp_logging_data().get("static_graph"), True)
            with self.assertRaisesRegex(RuntimeError, 'should be called before training loop starts'):
                local_bs = 2
                batch_size, input, target, loss = self._prepare_dummy_data(local_bs)
                offset = dist.get_rank() * local_bs

                # DDP training, DDP scatters subsets of input to nodes/GPUs
                self._test_DDP_helper(
                    model_DDP,
                    input[offset: offset + local_bs],
                    target[offset: offset + local_bs],
                    loss,
                    1,
                )
                model_DDP._set_static_graph()

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

        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
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

        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
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

        def _test_ddp_profiling(self, profiler_ctx):
            torch.cuda.set_device(self.rank)
            batch = 3
            dim = 10
            num_iters = 6
            torch.cuda.set_device(self.rank)
            model = nn.Linear(dim, dim, bias=False)
            inp = torch.rand(batch, dim, device=self.rank)
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(self.rank),
                device_ids=[self.rank],
            )
            profiler_ctx_copy = copy.deepcopy(profiler_ctx)

            with profiler_ctx as prof:
                for i in range(num_iters):
                    loss = net(inp).sum()
                    loss.backward()

            all_reduce_event_name = f"{dist.get_backend()}:all_reduce"
            events = get_profiling_event(all_reduce_event_name, prof)
            event_count = sum(e.count for e in events)
            self.assertEqual(event_count, num_iters)
            for event in events:
                self.assertTrue(event.is_async)
                self.assertEqual(event.name, all_reduce_event_name)

            broadcast_event_name = f"{dist.get_backend()}:broadcast"
            broadcast_events = get_profiling_event(broadcast_event_name, prof)
            event_count = sum(e.count for e in broadcast_events)
            # Broadcast is called during rebuild_buckets
            self.assertGreaterEqual(event_count, 1)
            for event in broadcast_events:
                self.assertEqual(event.name, broadcast_event_name)

            # Run DDP with profiling for a few iterations, then enable profiling
            # for a single pass, and ensure it is recorded. This tests that the
            # thread local state is correctly updated.
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(self.rank),
                device_ids=[self.rank],
                find_unused_parameters=True
            )
            for i in range(3):
                loss = net(inp).sum()
                loss.backward()
            # Now enable the profiler.
            with profiler_ctx_copy as prof:
                loss = net(inp).sum()
                loss.backward()

            events = get_profiling_event(all_reduce_event_name, prof)
            self.assertGreaterEqual(len(events), 1)
            self.assertGreaterEqual(events[0].count, 1)
            self.assertEqual(events[0].name, all_reduce_event_name)
            for event in events:
                self.assertTrue(event.is_async)
            # Ensure searching unused parameters was profiled
            events = get_profiling_event("search_unused_parameters", prof)
            self.assertEqual(len(events), 1)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_profiling_autograd_profiler(self):
            autograd_profiler_ctx = torch.autograd.profiler.profile()
            return self._test_ddp_profiling(profiler_ctx=autograd_profiler_ctx)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(IS_FBCODE, "Kineto in fbcode code causes hang")
        @unittest.skipIf(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124"
        )
        def test_ddp_profiling_torch_profiler(self):
            cpu_act = torch.profiler.ProfilerActivity.CPU
            cuda_act = torch.profiler.ProfilerActivity.CUDA
            torch_profiler_ctx = torch.profiler.profile(
                activities=[cpu_act, cuda_act]
            )
            self._test_ddp_profiling(profiler_ctx=torch_profiler_ctx)

        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
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
            torch.cuda.set_device(rank)
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
            # If we throw when earliest rank terminates, we should ensure
            # that we iterate for that minimum number of times.
            num_iters_tensor = torch.tensor([num_iters], device=torch.cuda.current_device())
            dist.all_reduce(num_iters_tensor, op=dist.ReduceOp.MIN)
            min_num_iters = num_iters_tensor.item()
            total_iters = 0
            if test_case.throw_on_early_termination:
                if min_num_iters == num_iters:
                    # Early termination rank(s)
                    exception_ctx = self.assertRaisesRegex(
                        RuntimeError, f"Rank {self.rank} exhausted all inputs"
                    )
                else:
                    # Non early termination rank
                    exception_ctx = self.assertRaisesRegex(
                        RuntimeError,
                        "Detected at least one rank that exhausted inputs."
                    )
            else:
                exception_ctx = suppress()
            with exception_ctx:
                with net.join(throw_on_early_termination=test_case.throw_on_early_termination):
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
                        total_iters += 1
            if test_case.throw_on_early_termination:
                # Ensure we iterated min_num_iters times.
                self.assertEqual(total_iters, min_num_iters)
            else:
                # Ensure we iterated at least min_num_iters times.
                self.assertGreaterEqual(total_iters, min_num_iters)

            # Ensure completion of all GPU kernels.
            torch.cuda.synchronize(device=rank)
            # When throwing on early rank termination, we do not
            # broadcast model state from an authoritative rank. All models
            # should already be in sync.
            if not test_case.throw_on_early_termination:
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
                # Ensure that running with DDP uneven inputs was logged.
                ddp_logging_data = net._get_ddp_logging_data()
                self.assertTrue(ddp_logging_data.get("join_uneven_inputs"))
                dist.barrier()

        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
        def test_ddp_uneven_inputs_stop_iteration_sync_bn(self):
            # Tests that uneven inputs join handler correctly throws StopIteration
            # for models with SyncBN or general collective comm when
            # throw_on_early_termination=True.
            class ModelWithComm(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = nn.Linear(2, 40, bias=False)

                def forward(self, x):
                    x = self.lin(x)
                    dist.all_reduce(x)
                    return x

            torch.cuda.set_device(self.rank)
            model_bn = BN_NET
            model_bn = nn.SyncBatchNorm.convert_sync_batchnorm(
                copy.deepcopy(model_bn)
            ).cuda(self.rank)
            comm_model = ModelWithComm().cuda(self.rank)
            model_input = torch.randn(10, 2).cuda(torch.cuda.current_device())

            for model in [model_bn, comm_model]:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.rank],
                )
                min_num_iters = 5
                if self.rank != 0:
                    # Early termination rank(s)
                    num_iters = min_num_iters
                    exception_ctx = self.assertRaisesRegex(
                        RuntimeError, f"Rank {self.rank} exhausted all inputs"
                    )
                else:
                    # Non early termination rank
                    num_iters = min_num_iters * 2
                    exception_ctx = self.assertRaisesRegex(
                        RuntimeError,
                        "Detected at least one rank that exhausted inputs."
                    )
                n = 0
                with exception_ctx:
                    with model.join(throw_on_early_termination=True):
                        for i in range(num_iters):
                            loss = model(model_input).sum()
                            loss.backward()
                            self._model_step(model)
                            n += 1

                self.assertEqual(n, min_num_iters)
                # Verify model equivalence
                self.validate_net_equivalence(model)

        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
        def test_ddp_uneven_inputs(self):
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

            throw_on_early_term_tests = []
            for test_input in models_to_test:
                throw_on_early_term_tests.append(
                    DDPUnevenTestInput(
                        name=test_input.name,
                        model=test_input.model,
                        inp=test_input.inp,
                        sync_interval=test_input.sync_interval,
                        throw_on_early_termination=True,
                    )
                )

            models_to_test.extend(models_with_sync)
            models_to_test.extend(throw_on_early_term_tests)

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

        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
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

        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only NCCL and GLOO backend support DistributedDataParallel",
        )
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

        def _test_ddp_ignore_params_arg(self, static_graph=False):
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
                if static_graph:
                    ddp._set_static_graph()
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
        def test_ddp_ignore_params_arg(self):
            self._test_ddp_ignore_params_arg(static_graph=False)
            self._test_ddp_ignore_params_arg(static_graph=True)

        @with_dist_debug_levels(levels=["OFF", "INFO", "DETAIL"])
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
                    try:
                        ddp(inp).sum().backward()
                    except RuntimeError as e:
                        msg = str(e)
                        expected_strs = [
                            ddp_prev_reduction_unfinished_str,
                            ddp_recommend_find_unused_params_str,
                            ddp_outputs_not_used_in_loss_str
                        ]
                        # In debug mode, should show parameters that weren't reduced.
                        # Without debug mode, should show suggestion to use debug mode.
                        if dist._get_debug_mode() == dist._DistributedDebugLevel.OFF:
                            expected_strs.append(ddp_suggest_debug_mode_str)
                        else:
                            unreduced_params = ", ".join(['net2.weight'])
                            expected_strs.append(
                                f"did not receive grad for rank {self.rank}: {unreduced_params}"
                            )
                        for s in expected_strs:
                            self.assertTrue(
                                s in msg,
                                f"Expected {s} to be in {msg}"
                            )
                        self.assertFalse(ddp_find_unused_params_enabled_str in msg)
                    else:
                        self.assertFalse(True, "DDP unused parameters error not raised.")
                else:
                    ddp(inp).sum().backward()

            dist.barrier()

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
                def __init__(_self):  # noqa: B902
                    super().__init__()
                    _self.lin = nn.Linear(10, 1)

                def forward(_self, input, expected_type):  # noqa: B902
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

        @with_dist_debug_levels(levels=["OFF", "INFO", "DETAIL"])
        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_control_flow_same_across_ranks(self):
            # Control flow that is the same across ranks.
            batch = 20
            dim = 10

            world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            model = torch.nn.parallel.DistributedDataParallel(
                ControlFlowToyModel().cuda(self.rank),
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
                ControlFlowToyModel().cuda(self.rank),
                device_ids=[self.rank],
                find_unused_parameters=False,
            )
            for i in range(2):
                if i == 0:
                    loss = model(random_input).sum()
                    loss.backward()
                else:
                    try:
                        loss = model(random_input).sum()
                        loss.backward()
                    except RuntimeError as e:
                        msg = str(e)
                        # 2nd linear layer is unused
                        unused_param_index = 1
                        expected_strs = [
                            ddp_prev_reduction_unfinished_str,
                            ddp_recommend_find_unused_params_str,
                            ddp_outputs_not_used_in_loss_str,
                            f"Parameter indices which did not receive grad for rank {self.rank}: {unused_param_index}"
                        ]
                        # In debug mode, should show parameters that weren't reduced.
                        # Without debug mode, should show suggestion to use debug mode.
                        if dist._get_debug_mode() == dist._DistributedDebugLevel.OFF:
                            expected_strs.append(ddp_suggest_debug_mode_str)
                        else:
                            unreduced_params = ", ".join(['lin2.weight'])
                            expected_strs.append(
                                f"did not receive grad for rank {self.rank}: {unreduced_params}"
                            )
                        for s in expected_strs:
                            self.assertTrue(
                                s in msg,
                                f"Expected {s} to be in {msg}"
                            )
                        self.assertFalse(ddp_find_unused_params_enabled_str in msg)
                    else:
                        self.assertFalse(True, "DDP error not raised")

            dist.barrier()

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_invalid_static_graph(self):
            world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            model = torch.nn.parallel.DistributedDataParallel(
                ControlFlowToyModel().cuda(self.rank),
                device_ids=[self.rank],
            )
            model._set_static_graph()
            random_input = torch.randn(20, 10, device=self.rank)
            ones_input = torch.ones(20, 10, device=self.rank)
            # unused parameter in the first iteration got used
            # in second iteration.
            with self.assertRaisesRegex(
                RuntimeError,
                "Your training graph has changed in this iteration",
            ):
                for i in range(2):
                    if i % 2 == 0:
                        out = model(random_input)
                    else:
                        out = model(ones_input)
                    loss = out.sum()
                    loss.backward()
            # used parameter in the first iteration got unused
            # in second iteration.
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected to have finished reduction in the prior iteration "
                "before starting a new one. This error indicates that your "
                "training graph has changed in this iteration",
            ):
                for i in range(2):
                    if i % 2 != 0:
                        out = model(random_input)
                    else:
                        out = model(ones_input)
                    loss = out.sum()
                    loss.backward()

        @with_dist_debug_levels(levels=["OFF", "INFO", "DETAIL"])
        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
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
                if i == 0:
                    loss = model(random_input).sum()
                    loss.backward()
                else:
                    try:
                        loss = model(random_input).sum()
                        loss.backward()
                    except RuntimeError as e:
                        msg = str(e)
                        unused_param_index = 1
                        expected_strs = [
                            ddp_prev_reduction_unfinished_str,
                            ddp_recommend_find_unused_params_str,
                            ddp_outputs_not_used_in_loss_str,
                            f"Parameter indices which did not receive grad for rank {self.rank}: {unused_param_index}"
                        ]
                        # In debug mode, should show parameters that weren't reduced.
                        # Without debug mode, should show suggestion to use debug mode.
                        if dist._get_debug_mode() == dist._DistributedDebugLevel.OFF:
                            expected_strs.append(ddp_suggest_debug_mode_str)
                        else:
                            unreduced_params = ", ".join(['lin2.weight'])
                            expected_strs.append(
                                f"did not receive grad for rank {self.rank}: {unreduced_params}"
                            )
                        for s in expected_strs:
                            self.assertTrue(
                                s in msg,
                                f"Expected {s} to be in {msg}"
                            )
                        self.assertFalse(ddp_find_unused_params_enabled_str in msg)
                    else:
                        self.assertFalse(True, "DDP error not raised")

            dist.barrier()

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

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        @skip_if_rocm
        def test_ddp_model_diff_across_ranks(self):
            torch.cuda.set_device(self.rank)
            # Creates network with different sized embedding table on different
            # ranks. This should throw an error during DDP init.
            net = EmbeddingNet(self.rank)
            # When running with NCCL backend, we don't expect an error on rank 0,
            # rather, it will be taken down by NCCL_ASYNC_ERROR_HANDLING. When
            # running with Gloo, we expect the error to be caught inline.
            rank_0_ctx = (
                suppress()
                if dist.get_backend() == dist.Backend.NCCL
                # Gloo can raise various exception messages, so just assert
                # Runtime error here.
                else self.assertRaises(RuntimeError)
            )
            ctx = (
                rank_0_ctx
                if self.rank == 0
                else self.assertRaisesRegex(RuntimeError, "appears not to match")
            )
            with ctx:
                net = torch.nn.parallel.DistributedDataParallel(
                    net.to(self.rank), device_ids=[self.rank]
                )
                dist.barrier()

        def _test_output_unused_in_loss(self, module_cls, gradient_as_bucket_view):
            model = module_cls()
            local_net = copy.deepcopy(model)
            net = torch.nn.parallel.DistributedDataParallel(
                copy.deepcopy(model).cuda(self.rank),
                device_ids=[self.rank],
                find_unused_parameters=True,
            )

            # Tests that certain parameters not getting gradient since the
            # output is unused in loss computation is supported. Specifically,
            # checks that the grads remain unchanged and are the same as local
            # training.
            inp = torch.randn(10, 10)
            a_local_grad = None
            a_dist_grad = None

            # Ensure that if a param is not used in loss computation, its
            # gradient is untouched, i.e. if it is None before it is None after,
            # not zero.
            if module_cls == DictOutputModule:
                a, b = local_net(inp)["predictions"]
                a_dist, b_dist = net(inp)["predictions"]
            else:
                a, b = local_net(inp)
                a_dist, b_dist = net(inp)

            loss_dist = b_dist.sum()
            loss_dist.backward()

            # Ensure that gradient corresponding to parameter "a" was not
            # touched, i.e. it is None and matches the local grad.
            if module_cls == DictOutputModule:
                self.assertTrue(net.module.module.a.weight.grad is None)
                self.assertEqual(
                    net.module.module.a.weight.grad, local_net.module.a.weight.grad
                )
            else:
                self.assertTrue(net.module.a.weight.grad is None)
                self.assertEqual(net.module.a.weight.grad, local_net.a.weight.grad)

            net.zero_grad()
            local_net.zero_grad()
            for i in range(6):
                if module_cls == DictOutputModule:
                    a, b = local_net(inp)["predictions"]
                    a_dist, b_dist = net(inp)["predictions"]
                else:
                    a, b = local_net(inp)
                    a_dist, b_dist = net(inp)
                if i < 2:
                    # Use both params in loss computation. Later, "a" will go
                    # unused and we check to ensure DDP supports this and
                    # gradients remain the same as local training.
                    t = a @ b
                    t_dist = a_dist @ b_dist
                    loss = t.sum()
                    loss_dist = t_dist.sum()
                else:
                    # Model output "a" unused in loss.
                    loss = b.sum()
                    loss_dist = b_dist.sum()
                loss.backward()
                loss_dist.backward()
                if i == 1:
                    # Save grads to compare with them in next iterations.
                    if module_cls == DictOutputModule:
                        a_local_grad = local_net.module.a.weight.grad
                        a_dist_grad = net.module.module.a.weight.grad
                    else:
                        a_local_grad = local_net.a.weight.grad
                        a_dist_grad = net.module.a.weight.grad
                    self.assertEqual(a_local_grad, a_dist_grad)
                elif i >= 2:
                    # parameter "a" of both models should be the same and not change
                    if module_cls == DictOutputModule:
                        self.assertEqual(net.module.module.a.weight.grad, a_dist_grad)
                        self.assertEqual(local_net.module.a.weight.grad, a_local_grad)
                    else:
                        self.assertEqual(net.module.a.weight.grad, a_dist_grad)
                        self.assertEqual(local_net.a.weight.grad, a_local_grad)

                # Verify grads are the same
                for (local_param, dist_param) in zip(
                    local_net.parameters(), net.parameters()
                ):
                    local_grad = local_param.grad
                    dist_grad = dist_param.grad
                    self.assertEqual(local_grad, dist_grad)

            dist.barrier()


        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_lt_x_gpu(2)
        def test_output_unused_in_loss_tuple_module(self):
            module_cls = UnusedParamTwoLinLayerNet
            for grad_as_bucket_view in [True, False]:
                self._test_output_unused_in_loss(
                    module_cls,
                    grad_as_bucket_view
                )

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_lt_x_gpu(2)
        def test_output_unused_in_loss_dict_module(self):
            module_cls = DictOutputModule
            for grad_as_bucket_view in [True, False]:
                self._test_output_unused_in_loss(
                    module_cls,
                    grad_as_bucket_view
                )

        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only Nccl & Gloo backend support DistributedDataParallel",
        )
        @skip_if_lt_x_gpu(2)
        def test_undefined_grad_parity_unused_parameters(self):
            # TODO: enable this for general training use cases:
            # https://github.com/pytorch/pytorch/issues/58511.
            x = torch.ones(1, 2).to(self.rank)
            net = Net().to(self.rank)
            local_net = copy.deepcopy(net)
            net = torch.nn.parallel.DistributedDataParallel(
                net,
                device_ids=[self.rank],
                find_unused_parameters=True,
            )
            out = net(x).sum()
            local_out = local_net(x).sum()
            # Simulates undefined gradients.
            torch._C._functions.UndefinedGrad()(out).backward()
            torch._C._functions.UndefinedGrad()(local_out).backward()
            for (dist_param_name, dist_param), (local_param_name, local_param) in zip(
                net.named_parameters(), local_net.named_parameters()
            ):
                dist_grad = dist_param.grad
                local_grad = local_param.grad
                self.assertEqual(
                    dist_grad,
                    local_grad,
                    f"""DDP param {dist_param_name} with grad {dist_grad}
                    does not match local param {local_param_name} with grad
                    {local_grad}"""
                )

        def _test_different_graph_across_ranks(self,
                                               find_unused_parameters=False,
                                               static_graph=False):
            class ToyModel(nn.Module):
                def __init__(self, rank):
                    super(ToyModel, self).__init__()
                    self.lin1 = nn.Linear(10, 10, bias=False)
                    self.lin2 = nn.Linear(10, 10, bias=False)
                    self.rank = rank

                def forward(self, x):
                    if self.rank == 0:
                        return self.lin2(F.relu(self.lin1(x)))
                    else:
                        return F.relu(self.lin1(x))

            torch.manual_seed(31415)
            world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            model = ToyModel(self.rank).cuda(self.rank)
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=True,
            )
            if static_graph:
                ddp_model._set_static_graph()
            random_input = torch.randn(20, 10, device=self.rank)
            for i in range(10):
                out = ddp_model(random_input)
                loss = out.sum()
                loss.backward()
            return ddp_model

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_different_graph_across_ranks(self):
            base_model = self._test_different_graph_across_ranks(find_unused_parameters=True)
            self.assertFalse(base_model._get_ddp_logging_data().get("has_rebuilt_buckets", 0))
            static_model = self._test_different_graph_across_ranks(static_graph=True)
            self.assertTrue(static_model._get_ddp_logging_data().get("has_rebuilt_buckets", 0))
            for i, j in zip(base_model.parameters(), static_model.parameters()):
                self.assertEqual(i, j)

        @require_backend({"gloo"})
        @require_backends_available({"gloo"})
        @unittest.skipIf(
            IS_MACOS or IS_WINDOWS,
            "MacOS uses uv transport which does not have as robust error handling as tcp transport"
        )
        def test_monitored_barrier_gloo(self):
            tensors = [torch.ones(10) * self.rank]
            # Kick off some allreduce work on all ranks
            for _ in range(10):
                dist.all_reduce(torch.cat(tensors))
            # Run monitored barrier
            timeout = timedelta(seconds=2)
            dist.monitored_barrier(timeout=timeout)
            # All ranks besides 1 call into barrier, rank 0 should report failure
            # while others report gloo error.
            failed_rank = 1
            src_rank = 0
            if self.rank == src_rank:
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"Rank {failed_rank} failed to pass monitoredBarrier"
                ):
                    dist.monitored_barrier(timeout=timeout)
            elif self.rank != failed_rank:
                # Other ranks should not pass barrier since rank 0 failed.
                err_regex = (
                    f"Rank {self.rank} successfully reached monitoredBarrier,"
                    f" but received errors while waiting to be unblocked by rank"
                    f" {src_rank}"
                )
                with self.assertRaisesRegex(RuntimeError, err_regex):
                    dist.monitored_barrier(timeout=timeout)

            # We need a barrier since otherwise failed_rank exits too early
            # and cause a timeout.
            self._barrier(timeout=30)

        @require_backend({"gloo"})
        @require_backends_available({"gloo"})
        def test_monitored_barrier_gloo_subgroup(self):
            # Tests that monitored_barrier works as expected on non-default
            # process groups.
            failed_rank = 1
            timeout = 0.1
            subgroup = dist.new_group(ranks=[0, 1])

            if self.rank == failed_rank:
                return

            if self.rank == 0:
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"Rank {failed_rank} failed to pass monitoredBarrier"
                ):
                    dist.monitored_barrier(subgroup, timeout)
            else:
                # Other ranks call into monitored_barrier, but this should be a
                # noop because they are not part of the subgroup. Verify that
                # there are no errors here.
                dist.monitored_barrier(subgroup, timeout)

        def _test_monitored_barrier_allreduce_hang(self, wait_all_ranks):
            # tests expected behavior when nonzero rank hangs.
            nccl_pg = dist.new_group(
                ranks=list(i for i in range(int(self.world_size))),
                timeout=timedelta(seconds=2),
                backend=dist.Backend.NCCL,
            )
            gloo_pg = dist.new_group(
                ranks=list(i for i in range(int(self.world_size))),
                backend=dist.Backend.GLOO,
            )
            tensors = [
                torch.ones(10, device=self.rank) * self.rank
            ]
            # Let all ranks call allreduce first to set up communicators etc.
            # Directly simulating error here will run into store issue described
            # in https://github.com/pytorch/pytorch/issues/54524.
            nccl_pg.allreduce(tensors).wait()
            # All ranks besides 0 call into allreduce. This is to simulate a
            # desync across the world, where some ranks call into
            # monitored_barrier() and others are stuck in collective comm. In
            # practice, we don't need NCCL_BLOCKING_WAIT, but we use it in this
            # test to ensure it exits cleanly.
            if self.rank != 0:
                with self.assertRaisesRegex(RuntimeError, "Caught collective operation timeout"):
                    nccl_pg.allreduce(tensors).wait(timedelta(seconds=0.1))
            else:
                # Rank 0 should report first (in order) timed out rank or all ranks
                # depending on wait_all_ranks flag passed into monitored_barrier.
                if wait_all_ranks:
                    rank_str = ", ".join([str(i) for i in range(1, int(self.world_size))])
                    err_regex = f"Ranks {rank_str} failed to pass monitoredBarrier"
                else:
                    expected_first_fail_rank = 1
                    err_regex = f"Rank {expected_first_fail_rank} failed to pass monitoredBarrier"
                monitored_barrier_timeout_seconds = timedelta(seconds=0.1)
                with self.assertRaisesRegex(
                    RuntimeError,
                    err_regex
                ):
                    gloo_pg.monitored_barrier(monitored_barrier_timeout_seconds, wait_all_ranks=wait_all_ranks)

        @with_nccl_blocking_wait
        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_rocm
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_monitored_barrier_allreduce_hang(self):
            # tests expected behavior when nonzero rank hangs and we want to
            # report first timed out rank.
            self._test_monitored_barrier_allreduce_hang(wait_all_ranks=False)

        @with_nccl_blocking_wait
        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_rocm
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_monitored_barrier_allreduce_hang_wait_all_ranks(self):
            # tests expected behavior when nonzero rank hangs and we want to
            # report all timed out ranks.
            self._test_monitored_barrier_allreduce_hang(wait_all_ranks=True)

        @require_backend({"gloo"})
        @require_backends_available({"gloo"})
        def test_monitored_barrier_gloo_rank_0_timeout(self):
            # tests error when rank 0 exhausts its given timeout.
            process_group = dist.new_group(
                ranks=list(i for i in range(int(self.world_size)))
            )
            timeout = timedelta(seconds=0)
            if self.rank == 0:
                with self.assertRaisesRegex(
                    RuntimeError, f"Rank {self.rank} timed out in monitoredBarrier"
                ):
                    process_group.monitored_barrier(timeout)

        @require_backend({"gloo"})
        @require_backends_available({"gloo"})
        @skip_if_small_worldsize
        @unittest.skipIf(
            IS_MACOS or IS_WINDOWS,
            "MacOS uses uv transport which does not have as robust error handling as tcp transport"
        )
        def test_monitored_barrier_failure_order(self):
            # Ensure that the first (in sorted order) rank is reported when
            # multiple ranks fail to pass the monitored_barrier.
            # TODO(#54879): Provide ability to wait and report all failed ranks
            expected_first_failed_rank = 2
            timeout = timedelta(seconds=2)
            src_rank = 0
            if self.rank == src_rank:
                with self.assertRaisesRegex(RuntimeError, f"Rank {expected_first_failed_rank}"):
                    dist.monitored_barrier(timeout=timeout)
            elif self.rank == 1:
                err_regex = (
                    f"Rank {self.rank} successfully reached monitoredBarrier,"
                    f" but received errors while waiting to be unblocked by rank"
                    f" {src_rank}"
                )
                with self.assertRaisesRegex(RuntimeError, err_regex):
                    dist.monitored_barrier(timeout=timeout)

        @require_backend({"gloo"})
        @require_backends_available({"gloo"})
        @skip_if_small_worldsize
        def test_monitored_barrier_wait_all_ranks(self):
            # Tests simple case where > 1 rank does not call into monitored
            # barrier and verifies all ranks are reported by rank 0.
            if self.rank == 0:
                timeout = timedelta(seconds=0.1)
                rank_str = ", ".join([str(i) for i in range(1, int(self.world_size))])
                err_regex = f"Ranks {rank_str} failed to pass monitoredBarrier"
                with self.assertRaisesRegex(RuntimeError, err_regex):
                    dist.monitored_barrier(timeout=timeout, wait_all_ranks=True)

        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_build_param_to_name_mapping(self):
            model = TwoLinLayerNet()
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(self.rank),
                device_ids=[self.rank],
            )
            expected_mapping = {0: "a.weight", 1: "b.weight"}
            net_params, _ = net._build_params_for_reducer()
            param_to_name_mapping = net._build_param_to_name_mapping(net_params)
            self.assertDictEqual(expected_mapping, param_to_name_mapping)

            # Test when DDP is used with ignored parameters.
            model = TwoLinLayerNet()
            # Parameters to ignore are in the format {module_name}.{param_name}
            params_to_ignore = ["a.weight"]
            torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
                model, params_to_ignore
            )
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(self.rank),
                device_ids=[self.rank],
            )
            expected_mapping = {0: "b.weight"}
            net_params, _ = net._build_params_for_reducer()
            param_to_name_mapping = net._build_param_to_name_mapping(net_params)
            self.assertDictEqual(expected_mapping, param_to_name_mapping)

            # Test errors are raised when DDP and module parameters mismatch.
            # This generally indicates a bug with DDP and is not expected to
            # happen in user applications.
            model = TwoLinLayerNet()
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(self.rank),
                device_ids=[self.rank],
            )
            net_params, _ = net._build_params_for_reducer()
            if self.rank == 0:
                print(type(net_params[0][0]))

            net_params[0].extend([
                torch.nn.Parameter(torch.ones(1)),
                torch.nn.Parameter(torch.ones(1)),
            ])

            with self.assertRaisesRegex(
                ValueError,
                "Expected param to name mapping"
            ):
                net._build_param_to_name_mapping(net_params)

            net_params[0] = net_params[0][:-3]
            with self.assertRaisesRegex(ValueError, "Param with name"):
                net._build_param_to_name_mapping(net_params)

            net_params[0].extend([
                torch.nn.Parameter(torch.ones(1)),
                torch.nn.Parameter(torch.ones(1)),
            ])

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_lt_x_gpu(2)
        def test_ddp_build_param_to_name_mapping_requires_grad(self):
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = nn.Linear(10, 10)
                    # Is not tracked by DDP and should not show up in param to
                    # name mapping.
                    self.lin.bias.requires_grad_(False)

                def forward(self, x):
                    return self.lin(x)

            model = Net()
            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(self.rank),
                device_ids=[self.rank]
            )
            expected_mapping = {
                0: 'lin.weight',
            }
            net_params, _ = net._build_params_for_reducer()
            param_to_name_mapping = net._build_param_to_name_mapping(net_params)
            self.assertEqual(param_to_name_mapping, expected_mapping)

        def _test_ddp_multiple_nested_unused_params_error(self, ignore_sparse):
            debug_mode_off = dist._get_debug_mode() == dist._DistributedDebugLevel.OFF

            class SubModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding_net = EmbeddingNet(0)
                    self.lin = TwoLinLayerNet()
                    self.bn = BatchNormNet()
                    self.lin_layer = nn.Linear(4, 10, bias=False)

                def forward(self, x):
                    x = self.bn(x)
                    x = self.lin_layer(x)
                    x = self.lin.a(x)  # self.lin.b param unused
                    # EmbeddingNet entirely unused: self.embedding_net.embedding and
                    # self.embedding_net.lin unused.
                    return x

            class MyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.sub_module = SubModule()

                def forward(self, x):
                    return self.sub_module(x)

            model = MyModel()
            sparse_embedding_fqns = []
            if ignore_sparse:
                for module_name, module in model.named_modules():
                    if module == model.sub_module.embedding_net.embedding:
                        for parameter_name, param in module.named_parameters(
                            recurse=False
                        ):
                            fqn = f"{module_name}.{parameter_name}"
                            sparse_embedding_fqns.append(fqn)

                torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
                    model, sparse_embedding_fqns
                )
                unused_modules = [
                    model.sub_module.embedding_net.lin,
                    model.sub_module.lin.b,
                ]
            else:
                unused_modules = list(model.sub_module.embedding_net.modules()) + [
                    model.sub_module.lin.b,
                ]


            expected_unused_param_fqns = []
            used_param_fqns = []  # Validate that these don't mistakenly show up.
            fqn_to_param_index = {}
            index = 0
            for module_name, module in model.named_modules():
                for parameter_name, param in module.named_parameters(recurse=False):
                    fqn = f"{module_name}.{parameter_name}"
                    fqn_to_param_index[fqn] = index
                    if fqn not in sparse_embedding_fqns:
                        index += 1
                    if module in unused_modules:
                        expected_unused_param_fqns.append(fqn)
                    else:
                        if (
                            not ignore_sparse
                            or module != model.sub_module.embedding_net.embedding
                        ):
                            used_param_fqns.append(fqn)

            net = torch.nn.parallel.DistributedDataParallel(
                model.cuda(self.rank),
                device_ids=[self.rank],
            )
            batch, dim = 10, 2
            inp = torch.ones(batch, dim)
            for i in range(2):
                if i == 0:
                    out = net(inp)
                    loss = out.sum()
                    loss.backward()
                else:
                    try:
                        out = net(inp)
                        loss = out.sum()
                        loss.backward()
                    except RuntimeError as e:
                        e = str(e)

                        unused_param_substr = e[e.find("did not receive grad") :]
                        # Validate that each unused param fully qualified name
                        # shows up in error logs. We do this instead of
                        # constructing a joined string since order of parameters
                        # can be different in Reducer. In addition, validate
                        # param indices show up as well.
                        for unused_param_fqn in expected_unused_param_fqns:
                            self.assertTrue(unused_param_fqn in unused_param_substr or debug_mode_off)
                            self.assertTrue(
                                str(fqn_to_param_index[unused_param_fqn]) in unused_param_substr,
                                f"Did not find index {fqn_to_param_index[unused_param_fqn]} for {unused_param_fqn}"
                            )

                        # Validate that used param fqns don't show up in error
                        # logs.
                        for used_param_fqn in used_param_fqns:
                            self.assertFalse(used_param_fqn in unused_param_substr)
                        # Validate that ignored param fqns don't show up as unused
                        # (since DDP does not track them)
                        for sparse_param_fqn in sparse_embedding_fqns:
                            self.assertFalse(sparse_param_fqn in unused_param_substr)
                    else:
                        self.assertTrue(False, "Expected error was not raised!")

        @with_dist_debug_levels(levels=["OFF", "INFO", "DETAIL"])
        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_multiple_nested_unused_params_error(self):
            self._test_ddp_multiple_nested_unused_params_error(ignore_sparse=False)

        @with_dist_debug_levels(levels=["OFF", "INFO", "DETAIL"])
        @require_backend({"gloo", "nccl"})
        @require_backends_available({"gloo", "nccl"})
        @skip_if_lt_x_gpu(2)
        def test_ddp_multiple_nested_unused_params_err_ignore_params(self):
            # Tests unused parameter reporting when DDP is configured to ignore
            # certain parameters.
            self._test_ddp_multiple_nested_unused_params_error(ignore_sparse=True)

        @unittest.skipIf(BACKEND != 'nccl' and BACKEND != 'gloo',
                         "Only Nccl & Gloo backend support DistributedDataParallel")
        @skip_if_lt_x_gpu(2)
        def test_ddp_sync_bn_training_vs_eval(self):
            rank = self.rank
            torch.cuda.set_device(rank)
            # Need to set track_running_stats=False, when track_running_stats=True,
            # bn_training is False and sync could not occur in eval model.
            model = nn.SyncBatchNorm(
                2, momentum=0.99, track_running_stats=False
            ).cuda(rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank]
            )
            # Test sync occurs in training mode.
            with torch.autograd.profiler.profile() as prof:
                for i in range(6):
                    inp = torch.randn(10, 2, 4, 4).cuda(rank)
                    out = model(inp)
                    loss = out.sum()
                    loss.backward()

            # SyncBN allgathers stats across all ranks, so verify call to
            # all_gather in profiler.
            all_gather_calls = get_profiling_event("all_gather", prof)
            self.assertNotEqual([], all_gather_calls)

            # Only do inference on one rank. If SyncBN did collective stats sync,
            # this would hang/error.
            model_inference = model.module
            if self.rank == 0:
                model_inference.eval()
                with torch.autograd.profiler.profile() as prof:
                    for i in range(6):
                        inp = torch.randn(10, 2, 4, 4).cuda(rank)
                        out = model_inference(inp)
                        loss = out.sum()
                        loss.backward()

                # Ensure sync does not occur in eval() mode.
                all_gather_calls = get_profiling_event("all_gather", prof)
                self.assertEqual([], all_gather_calls)

        @skip_if_lt_x_gpu(2)
        @unittest.skipIf(
            BACKEND != "nccl" and BACKEND != "gloo",
            "Only Nccl & Gloo backend support DistributedDataParallel",
        )
        def test_ddp_static_graph_nested_types(self):
            # Tests for static graph training when outputs are not just tensors
            # but can be (nested) tuple, list, dict, etc.
            rank = self.rank
            torch.cuda.set_device(rank)

            class NestedOutputModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = nn.Linear(100, 1, bias=False)

                def forward(self, inp, output_type):
                    if output_type == "tuple":
                        return (
                            self.lin(inp),
                            (
                                self.lin(inp),
                                self.lin(inp),
                            ),
                        )
                    elif output_type == "list":
                        return [
                            self.lin(inp),
                            [
                                self.lin(inp),
                                self.lin(inp),
                            ],
                        ]
                    elif output_type == "dict":
                        return {
                            "a": self.lin(inp),
                            "b": {
                                "c": self.lin(inp),
                            },
                        }

            def get_loss(model_output):
                loss = 0.0
                if isinstance(model_output, torch.Tensor):
                    return model_output.sum()
                elif isinstance(model_output, dict):
                    for value in model_output.values():
                        loss += get_loss(value)
                elif isinstance(model_output, tuple) or isinstance(model_output, list):
                    for x in model_output:
                        loss += get_loss(x)
                else:
                    raise ValueError(f"Unknown model output type {type(model_output)}")
                return loss

            model = NestedOutputModule().cuda(rank)
            model_static_graph = copy.deepcopy(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
            )
            model_static_graph = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
            )
            model_static_graph._set_static_graph()
            inp = torch.randn(10, 100)
            type_mapping = {
                "list": list,
                "tuple": tuple,
                "dict": dict,
            }
            for output_type in type_mapping.keys():
                for i in range(6):
                    out = model(inp, output_type=output_type)
                    loss = get_loss(out)
                    loss.backward()
                    self._model_step(model)
                    out_static = model_static_graph(inp, output_type=output_type)
                    self.assertTrue(isinstance(out_static, type_mapping[output_type]))
                    loss_static = get_loss(out_static)
                    loss_static.backward()
                    self._model_step(model_static_graph)
                    for (p, p_static) in zip(
                        model.parameters(), model_static_graph.parameters()
                    ):
                        self.assertEqual(p, p_static)
