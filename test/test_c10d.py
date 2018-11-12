import copy
import math
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from datetime import timedelta

from functools import wraps
from collections import namedtuple

import torch
import common_utils as common
from torch import nn
import torch.nn.functional as F
import torch.distributed as c10d
from torch.nn.parallel import DistributedDataParallel

from common_utils import TestCase, load_tests, run_tests

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not c10d.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


TIMEOUT_DEFAULT = 15
TIMEOUT_OVERRIDE = {}

TestSkip = namedtuple('TestSkip', 'exit_code, message')

TEST_SKIPS = {
    "multi-gpu": TestSkip(75, "Need at least 2 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "known_issues": TestSkip(77, "Test skipped due to known issues")
}


def skip_if_not_multigpu(func):
    """Multi-GPU tests requires at least 2 GPUS. Skip if this is not met."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS['multi-gpu'].exit_code)

    return wrapper


def skip_if_not_nccl(func):
    """Skips a test if NCCL is not available (for c10d)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(c10d, "ProcessGroupNCCL"):
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS['nccl'].exit_code)

    return wrapper


def skip_for_known_issues(func):
    """Skips a test due to known issues (for c10d)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        sys.exit(TEST_SKIPS['known_issues'].exit_code)

    return wrapper


def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)


def gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    visible_devices = list(range(torch.cuda.device_count()))
    gpus_per_process = torch.cuda.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(visible_devices[rank * gpus_per_process: (rank + 1) * gpus_per_process])
    return gpus_for_rank


def simple_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            torch.Tensor([rank + 1.0]),
            torch.Tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.Tensor([rank + 1.0]),
            torch.Tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.Tensor([rank + 1.0]),
            torch.Tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.Tensor([rank + 1.0]),
            torch.Tensor([world_size]),
        ),
    ]


class StoreTestBase(object):
    def _create_store(self, i):
        raise RuntimeError("not implemented")

    def _test_set_get(self, fs):
        fs.set("key0", "value0")
        fs.set("key1", "value1")
        fs.set("key2", "value2")
        self.assertEqual(b"value0", fs.get("key0"))
        self.assertEqual(b"value1", fs.get("key1"))
        self.assertEqual(b"value2", fs.get("key2"))

    def test_set_get(self):
        self._test_set_get(self._create_store())


class FileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile()

    def tearDown(self):
        self.file.close()

    def _create_store(self):
        store = c10d.FileStore(self.file.name)
        store.set_timeout(timedelta(seconds=300))
        return store


class PrefixFileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile()
        self.filestore = c10d.FileStore(self.file.name)
        self.prefix = "test_prefix"
        self.filestore.set_timeout(timedelta(seconds=300))

    def tearDown(self):
        self.file.close()

    def _create_store(self):
        return c10d.PrefixStore(self.prefix, self.filestore)


def create_tcp_store(addr):
    """
    Creates a TCP store. Retries if the chosen port is already in use.
    """
    ports = []
    for _ in range(10):
        try:
            port = common.find_free_port()
            ports.append(port)
            return c10d.TCPStore(addr, port, True)
        except RuntimeError as error:
            if str(error) == "Address already in use":
                continue
            raise
    raise RuntimeError("Unable to find free port (tried %s)" % ", ".join(ports))


class TCPStoreTest(TestCase, StoreTestBase):
    def _create_store(self):
        store = create_tcp_store('localhost')
        store.set_timeout(timedelta(seconds=300))
        return store

    def test_address_already_in_use(self):
        with self.assertRaisesRegex(RuntimeError, "^Address already in use$"):
            addr = 'localhost'
            port = common.find_free_port()

            # Use noqa to silence flake8.
            # Need to store in an unused variable here to ensure the first
            # object is not destroyed before the second object is created.
            store1 = c10d.TCPStore(addr, port, True)  # noqa: F841
            store2 = c10d.TCPStore(addr, port, True)  # noqa: F841


class PrefixTCPStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        self.tcpstore = create_tcp_store('localhost')
        self.prefix = "test_prefix"
        self.tcpstore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return c10d.PrefixStore(self.prefix, self.tcpstore)


class RendezvousTest(TestCase):
    def test_unknown_handler(self):
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            c10d.rendezvous('invalid://')


class RendezvousEnvTest(TestCase):
    def test_common_errors(self):
        vars = {
            "WORLD_SIZE": "2",
            "RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": common.find_free_port(),
        }

        class Env(object):
            def __init__(self, vars):
                self.vars = vars

            def __enter__(self):
                for key, value in self.vars.items():
                    os.environ[key] = str(value)

            def __exit__(self, type, value, traceback):
                for key in self.vars.keys():
                    del os.environ[key]

        def without(d, key):
            d = d.copy()
            d.pop(key)
            return d

        with Env(without(vars, 'WORLD_SIZE')):
            with self.assertRaisesRegex(ValueError, 'WORLD_SIZE expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
        with Env(without(vars, 'RANK')):
            with self.assertRaisesRegex(ValueError, 'RANK expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
        with Env(without(vars, 'MASTER_ADDR')):
            with self.assertRaisesRegex(ValueError, 'MASTER_ADDR expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
        with Env(without(vars, 'MASTER_PORT')):
            with self.assertRaisesRegex(ValueError, 'MASTER_PORT expected'):
                gen = c10d.rendezvous('env://')
                next(gen)

    def test_nominal(self):
        os.environ['WORLD_SIZE'] = '2'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(common.find_free_port())

        # First rank
        os.environ['RANK'] = '0'
        gen0 = c10d.rendezvous('env://')
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(2, size0)

        # Second rank
        os.environ['RANK'] = '1'
        gen1 = c10d.rendezvous('env://')
        store1, rank1, size1 = next(gen1)
        self.assertEqual(1, rank1)
        self.assertEqual(2, size1)

        # Set value on both stores
        store0.set("key0", "value0")
        store1.set("key1", "value1")

        # Cross check with get
        self.assertEqual(b"value0", store1.get("key0"))
        self.assertEqual(b"value1", store0.get("key1"))


class RendezvousFileTest(TestCase):
    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, 'path missing'):
            gen = c10d.rendezvous('file://?rank=0&world_size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            gen = c10d.rendezvous('file:///tmp/foo?world_size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'size parameter missing'):
            gen = c10d.rendezvous('file:///tmp/foo?rank=0')
            next(gen)

    def test_nominal(self):
        with tempfile.NamedTemporaryFile() as file:
            url = 'file://%s?world_size=%d' % (file.name, 2)
            gen0 = c10d.rendezvous(url + "&rank=0")
            store0, rank0, size0 = next(gen0)
            self.assertEqual(0, rank0)
            self.assertEqual(2, size0)
            gen1 = c10d.rendezvous(url + "&rank=1")
            store1, rank1, size1 = next(gen1)
            self.assertEqual(1, rank1)
            self.assertEqual(2, size1)

            # Set value on both stores
            store0.set("key0", "value0")
            store1.set("key1", "value1")

            # Cross check with get
            self.assertEqual(b"value0", store1.get("key0"))
            self.assertEqual(b"value1", store0.get("key1"))


class RendezvousTCPTest(TestCase):
    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, 'port number missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1?rank=0&world_size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1:23456?world_size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'size parameter missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1:23456?rank=0')
            next(gen)

    def test_nominal(self):
        addr = 'localhost'
        port = common.find_free_port()
        url = 'tcp://%s:%d?world_size=%d' % (addr, port, 2)
        gen0 = c10d.rendezvous(url + "&rank=0")
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(2, size0)
        gen1 = c10d.rendezvous(url + "&rank=1")
        store1, rank1, size1 = next(gen1)
        self.assertEqual(1, rank1)
        self.assertEqual(2, size1)

        # Set value on both stores
        store0.set("key0", "value0")
        store1.set("key1", "value1")

        # Cross check with get
        self.assertEqual(b"value0", store1.get("key0"))
        self.assertEqual(b"value1", store0.get("key1"))


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1

    @property
    def world_size(self):
        return 4

    @staticmethod
    def join_or_run(fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn(self)
        return wrapper

    # The main process spawns N subprocesses that run the test.
    # This function patches overwrites every test function to either
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    @classmethod
    def setUpClass(cls):
        for attr in dir(cls):
            if attr.startswith('test'):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.file = tempfile.NamedTemporaryFile()
        self.processes = [self._spawn_process(rank) for rank in range(int(self.world_size))]

    def tearDown(self):
        for p in self.processes:
            p.terminate()
        self.file.close()

    def _spawn_process(self, rank):
        name = 'process ' + str(rank)
        process = multiprocessing.Process(target=self._run, name=name, args=(rank,))
        process.start()
        return process

    def _run(self, rank):
        self.rank = rank

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retreiving a corresponding test and executing it.
        getattr(self, self.id().split(".")[2])()
        sys.exit(0)

    def _join_processes(self, fn):
        timeout = get_timeout(self.id())
        start_time = time.time()
        for p in self.processes:
            p.join(timeout)
        elapsed_time = time.time() - start_time
        self._check_return_codes(elapsed_time)

    def _check_return_codes(self, elapsed_time):
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        first_process = self.processes[0]
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} terminated or timed out after {} seconds'.format(i, elapsed_time))
            self.assertEqual(p.exitcode, first_process.exitcode)
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)
        self.assertEqual(first_process.exitcode, 0)

    @property
    def is_master(self):
        return self.rank == 0


class ProcessGroupGlooTest(MultiProcessTestCase):
    def opts(self, threads=2):
        opts = c10d.ProcessGroupGloo.Options()
        opts.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        opts.timeout = 1.0
        opts.threads = threads
        return opts

    def test_broadcast_checks(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = -1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t3], opts)

    def _test_broadcast_basics(self, fn):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.Tensor([self.rank]))
            broadcast([x], i, 0)
            self.assertEqual(torch.Tensor([i]), x)

            # Run with 2 input tensors
            num = 2
            for j in range(num):
                xs = [
                    fn(torch.Tensor([self.rank * num + 0.0])),
                    fn(torch.Tensor([self.rank * num + 1.0])),
                ]

                broadcast(xs, i, j)
                self.assertEqual(torch.Tensor([i * num + j]), xs[0])
                self.assertEqual(torch.Tensor([i * num + j]), xs[1])

        # Test overloaded convenience function
        x = torch.Tensor([self.rank + 1.0])
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.Tensor([1.0]), x)

    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_broadcast_basics_cuda(self):
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    def _test_broadcast_stress(self, inputs):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(threads=8))
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.Tensor([
                    (i * self.world_size) + (i % self.world_size)
                ]),
                inputs[i],
                "Mismatch in iteration %d" % i,
            )

    def test_broadcast_stress(self):
        inputs = [torch.Tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    @skip_if_not_multigpu
    def test_broadcast_stress_cuda(self):
        inputs = [torch.Tensor([i * self.world_size + self.rank]).cuda() for i in range(1000)]
        self._test_broadcast_stress(inputs)

    def test_allreduce_checks(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t3], opts)

    def _test_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())
        for (op, input, output) in simple_reduce_tests(self.rank, self.world_size):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tmp = fn(input)
            work = pg.allreduce([tmp], opts)
            work.wait()
            self.assertEqual(output, tmp)

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.Tensor([self.rank + 1.0]))
        work = pg.allreduce(x)
        work.wait()
        self.assertEqual(torch.Tensor([float(self.world_size * (self.world_size + 1) / 2)]), x)

    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_allreduce_basics_cuda(self):
        self._test_allreduce_basics(lambda t: t.clone().cuda())

    def _test_allreduce_stress(self, inputs):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(threads=8))
        work_handles = [pg.allreduce(inputs[i]) for i in range(len(inputs))]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.Tensor([
                    (i * self.world_size) +
                    (self.world_size * (self.world_size - 1) / 2)
                ]),
                inputs[i],
                "Mismatch in iteration %d" % i,
            )

    def test_allreduce_stress(self):
        inputs = [torch.Tensor([i + self.rank]) for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @skip_if_not_multigpu
    def test_allreduce_stress_cuda(self):
        inputs = [torch.Tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allreduce_stress(inputs)

    def test_scatter_checks(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output tensor list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output tensor list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([t1, t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * self.world_size, [t1] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * (self.world_size - 1)], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * (self.world_size + 1)], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * (self.world_size + 1)], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t2] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t3] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "requires empty input on non-root"):
            opts = c10d.ScatterOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.scatter([t1], [[t1] * self.world_size], opts)

    def test_scatter_basics(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        input = [torch.Tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.Tensor([-1]) for _ in range(self.world_size)]

        # Take turns being the scatter root and accumulate work items
        work = []
        for i in range(self.world_size):
            opts = c10d.ScatterOptions()
            opts.rootRank = i
            if i == self.rank:
                work.append(pg.scatter([outputs[i]], [input], opts))
            else:
                work.append(pg.scatter([outputs[i]], [], opts))

        # Wait for work to complete
        for i in range(self.world_size):
            work[i].wait()
            self.assertEqual(torch.Tensor([i]), outputs[i])

    def test_gather_checks(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input tensor list"):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input tensor list"):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [t1, t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output list"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output list"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * self.world_size, [t1] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output list"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * (self.world_size - 1)], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output list"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * (self.world_size + 1)], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t2] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t3] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires empty output on non-root"):
            opts = c10d.GatherOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.gather([[t1] * self.world_size], [t1], opts)

    def test_gather_basics(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        input = [torch.Tensor([self.rank])]
        outputs = [torch.Tensor([-1]) for _ in range(self.world_size)]

        # Take turns being the gather root and accumulate work items
        work = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                work.append(pg.gather([outputs], input, opts))
            else:
                work.append(pg.gather([], input, opts))

        # Wait for work to complete
        expected = [torch.Tensor([rank]) for rank in range(self.world_size)]
        for i in range(self.world_size):
            work[i].wait()
            if i == self.rank:
                self.assertEqual(expected, outputs)

    def test_allgather_checks(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires non-empty input tensor list"):
            pg.allgather([], [])

        with self.assertRaisesRegex(ValueError, "requires input/output tensor lists to have the same length"):
            pg.allgather([], [t1])

        with self.assertRaisesRegex(ValueError, "requires input/output tensor lists to have the same length"):
            pg.allgather([[t1] * self.world_size, [t1] * self.world_size], [t1])

        with self.assertRaisesRegex(ValueError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size - 1)], [t1])

        with self.assertRaisesRegex(ValueError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size + 1)], [t1])

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            pg.allgather([[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t2])

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            pg.allgather([[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t3])

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            pg.allgather([([t1, t2] * (self.world_size))[:self.world_size]], [t1])

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            pg.allgather([([t1, t3] * (self.world_size))[:self.world_size]], [t1])

    def test_allgather_basics(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Run with N input tensor per rank
        for n in [1, 2, 3]:
            input = [
                torch.Tensor([n * self.rank + i]) for i in range(n)
            ]
            output = [
                [
                    torch.Tensor([-1]) for _ in range(n * self.world_size)
                ] for _ in range(n)
            ]
            expected_output = [
                [
                    torch.Tensor([i]) for i in range(n * self.world_size)
                ] for _ in range(n)
            ]
            work = pg.allgather(output, input)
            work.wait()
            self.assertEqual(expected_output, output)

    def test_reduce_checks(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element tensor list"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.reduce([t1, t1], opts)

    def test_reduce_basics(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())
        for (op, input, output) in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = input.clone()
                work = pg.reduce([tmp], opts)
                work.wait()
                if root == self.rank:
                    self.assertEqual(output, tmp)

    def test_send_recv_all_to_all(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        inputs = [torch.Tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.Tensor([-1]) for _ in range(self.world_size)]

        # Issue sends
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))

        # Issue recvs
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # Wait for sends to complete
        for work in send_work:
            work.wait()

        # Wait for recvs to complete
        for work in recv_work:
            work.wait()

        # Test that every output other than our own contains the respective rank
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.Tensor([i]), outputs[i])

    def test_timeout_kwarg(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=0.5))

        # Wait on barrier
        self.assertTrue(pg.barrier().wait())

        # Sleep on one of the processes to trigger barrier timeout
        if self.rank == 0:
            time.sleep(0.6)

        # The barrier will now time output
        self.assertFalse(pg.barrier().wait())


class ProcessGroupNCCLTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        if not hasattr(c10d, "ProcessGroupNCCL"):
            raise unittest.SkipTest("C10D is not built with NCCL process group,"
                                    " skipping test")

        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile()
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")

    def tearDown(self):
        self.file.close()

    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.num_gpus):
            tensors = []
            for i in range(self.num_gpus):
                tensors.append(torch.Tensor([i]).cuda(i))

            broadcast(tensors, self.rank, rt)

            for i in range(self.num_gpus):
                self.assertEqual(tensors[i], tensors[rt])

    def test_allreduce_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce(tensors, op):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()

        # Sum
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.SUM)

        for i in range(self.num_gpus):
            self.assertEqual(
                torch.Tensor([float(self.num_gpus * (self.num_gpus + 1) / 2)]),
                tensors[i])

        # Product
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.PRODUCT)

        for i in range(self.num_gpus):
            self.assertEqual(
                torch.Tensor([float(math.factorial(self.num_gpus))]),
                tensors[i])

        # Min
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.MIN)

        for i in range(self.num_gpus):
            self.assertEqual(torch.Tensor([1.0]), tensors[i])

        # Max
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.MAX)

        for i in range(self.num_gpus):
            self.assertEqual(torch.Tensor([self.num_gpus]), tensors[i])

    def test_reduce_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def reduce(xs, rootRank, rootTensor):
            opts = c10d.ReduceOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.reduce(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.num_gpus):
            tensors = []
            for i in range(self.num_gpus):
                tensors.append(torch.Tensor([i + 1]).cuda(i))

            reduce(tensors, self.rank, rt)

            self.assertEqual(
                torch.Tensor([float(self.num_gpus * (self.num_gpus + 1) / 2)]),
                tensors[rt])

    def test_allgather_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allgather(output_ts, input_ts):
            work = pg.allgather(output_ts, input_ts)
            work.wait()

        tensors = []
        output_ts = [[] for _ in range(self.num_gpus)]

        for idx, ls in enumerate(output_ts):
            for _ in range(self.world_size * self.num_gpus):
                ls.append(torch.Tensor([0]).cuda(idx))

        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i]).cuda(i))

        allgather(output_ts, tensors)

        # Verification
        for device_ts in output_ts:
            for s_idx, t in enumerate(device_ts):
                self.assertEqual(torch.Tensor([s_idx]), t)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class DistributedDataParallelTest(MultiProcessTestCase):

    @property
    def world_size(self):
        return 2

    def _test_ddp_with_process_group(self, process_group, gpus):
        model = Net()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(model).cuda(gpus[0]),
            device_ids=gpus,
            process_group=process_group,
            bucket_cap_mb=0.001)

        model.cuda(gpus[0])

        local_batch_size = len(gpus)
        global_batch_size = self.world_size * local_batch_size
        input = torch.randn(global_batch_size, 2).cuda(gpus[0])
        target = torch.randn(global_batch_size, 4).cuda(gpus[0])

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target)
            loss.backward()

        def update_parameters(model):
            for param in model.parameters():
                param.data -= param.grad
                param.grad = None

        # check two model parameters over 2 iterations
        for iteration in range(2):
            # single cpu/gpu training
            step_model(model, input, target)

            # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
            step_model(ddp_model,
                       input[self.rank * local_batch_size: (self.rank + 1) * local_batch_size],
                       target[self.rank * local_batch_size: (self.rank + 1) * local_batch_size])

            # Update weights and run a second iteration to shake out errors
            update_parameters(model)
            update_parameters(ddp_model)
            self.assertEqual(len(list(model.parameters())), len(list(ddp_model.parameters())))
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                self.assertEqual(i, j)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    @skip_if_not_multigpu
    def test_gloo_backend(self):
        store = c10d.FileStore(self.file.name)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_ddp_with_process_group(process_group, gpus)
        self._test_ddp_with_process_group(process_group, list(map(lambda i: torch.device('cuda:' + str(i)), gpus)))

    @skip_if_not_multigpu
    @skip_if_not_nccl
    def test_nccl_backend(self):
        store = c10d.FileStore(self.file.name)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_ddp_with_process_group(process_group, gpus)
        self._test_ddp_with_process_group(process_group, list(map(lambda i: torch.device('cuda:' + str(i)), gpus)))

    @skip_if_not_multigpu
    @skip_if_not_nccl
    @skip_for_known_issues
    def test_dist_broadcast_coalesced_nccl(self):
        store = c10d.FileStore(self.file.name)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        device = torch.device('cuda')

        for fine_grained in [False, True]:
            target = torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

            if self.is_master:
                # All processes should have these tensors in the end.
                tensors = target
            else:
                # Non-master processes start with empty tensors and should be
                # filled with the tensors from the master.
                tensors = torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float32, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float64, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float32, device=device).chunk(5)

            c10d._dist_broadcast_coalesced(
                process_group,
                tensors,
                buffer_size=256,
                fine_grained=fine_grained)

            self.assertEqual(tensors, target)

    @skip_if_not_multigpu
    def test_dist_broadcast_coalesced_gloo(self):
        store = c10d.FileStore(self.file.name)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)

        device = torch.device('cuda')

        for fine_grained in [False, True]:
            target = torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

            if self.is_master:
                # All processes should have these tensors in the end.
                tensors = target
            else:
                # Non-master processes start with empty tensors and should be
                # filled with the tensors from the master.
                tensors = torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float32, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float64, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float32, device=device).chunk(5)

            c10d._dist_broadcast_coalesced(
                process_group,
                tensors,
                buffer_size=128,
                fine_grained=fine_grained)

            self.assertEqual(tensors, target)

    @skip_if_not_multigpu
    def test_sync_params_no_buffers(self):
        store = c10d.FileStore(self.file.name)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)

        # Use all available devices on every process here (data is small, so should be fine).
        devices = gpus_for_rank(self.world_size)[self.rank]
        target = torch.arange(10, dtype=torch.float64, device='cuda:0').chunk(5)
        parameter_data = [target]
        parameter_data += [torch.zeros(10, device=torch.device('cuda', d)).chunk(5) for d in devices[1:]]
        buffer_data = [[]] * len(parameter_data)

        c10d._sync_params(
            process_group,
            parameter_data=parameter_data,
            buffer_data=buffer_data,
            devices=devices,
            broadcast_bucket_size=10,
            broadcast_buffers=False)

        for device_data in parameter_data:
            for i, parameter in enumerate(device_data):
                self.assertEqual(parameter, target[i])

    @skip_if_not_multigpu
    def test_sync_params_with_buffers(self):
        store = c10d.FileStore(self.file.name)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)

        devices = gpus_for_rank(self.world_size)[self.rank]
        target = torch.arange(10, dtype=torch.float64, device='cuda:0').chunk(5)
        parameter_data = [target]
        parameter_data += [torch.zeros(10, device=torch.device('cuda', d)).chunk(5) for d in devices[1:]]

        # sync_params should do a dist_broadcast for buffers, so we only populate the master buffers and
        # then check that other processes' tensors end up matching.

        if self.is_master:
            buffer_data = [target]
            buffer_data += [torch.zeros(10, device=torch.device('cuda', d)).chunk(5) for d in devices[1:]]
        else:
            buffer_data = [torch.zeros(10, device=torch.device('cuda', d)).chunk(5) for d in devices]

        c10d._sync_params(
            process_group,
            parameter_data=parameter_data,
            buffer_data=buffer_data,
            devices=devices,
            broadcast_bucket_size=10,
            broadcast_buffers=True)

        for device_data in parameter_data:
            for i, parameter in enumerate(device_data):
                self.assertEqual(parameter, target[i])

        for device_data in buffer_data:
            for i, buffer in enumerate(device_data):
                self.assertEqual(buffer, target[i])

    @skip_if_not_multigpu
    @skip_if_not_nccl
    def test_fp16(self):
        store = c10d.FileStore(self.file.name)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).cuda(gpus[0]).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[gpus[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
        )

        # Input 2**15, so that the gradients will overflow with a
        # world_size of 2, unless we normalize the gradient by the
        # world_size before the reduction
        input = torch.Tensor([[2**15]]).cuda(gpus[0]).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(
            any(torch.isinf(p.grad).any() for p in ddp_model.parameters())
        )

    @skip_if_not_nccl
    def test_queue_reduction(self):
        # Set up process group.
        store = c10d.FileStore(self.file.name)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # Get this process' split of devices.
        devices = gpus_for_rank(self.world_size)[self.rank]
        grads_batch = [(torch.ones(10, device=torch.device('cuda', d)) *
                       (self.rank + 1)).chunk(5)
                       for d in devices]

        work, local_grad_sum = c10d._queue_reduction(process_group,
                                                     grads_batch,
                                                     devices)
        # The first return value should be the allreduce work item.
        self.assertTrue(isinstance(work, c10d.Work))
        # The second return value will be the finished allreduced gradients.
        self.assertTrue(isinstance(local_grad_sum, torch.Tensor))

        # Wait for the allreduce to finish.
        work.wait()

        # The expected result of the allreduce should be the average
        self.assertEqual(local_grad_sum,
                         torch.ones(10) * (self.world_size + 1) / 2.0)

    @skip_if_not_nccl
    def test_sync_reduction(self):
        # Set up process group.
        store = c10d.FileStore(self.file.name)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # Get this process' split of devices.
        devices = gpus_for_rank(self.world_size)[self.rank]
        grads_batch = [(torch.ones(10, device=torch.device('cuda', d)) *
                       (self.rank + 1)).chunk(5)
                       for d in devices]
        work, local_grad_sum = c10d._queue_reduction(process_group,
                                                     grads_batch,
                                                     devices)
        c10d._sync_reduction(work, grads_batch[0], local_grad_sum)
        # The expected result of the allreduce should be the average
        self.assertEqual(grads_batch[0], (torch.ones(10) * (self.world_size + 1) / 2.0).chunk(5))


if __name__ == '__main__':
    assert not torch.cuda._initialized, "test_distributed must not have initialized CUDA context on main process"

    run_tests()
