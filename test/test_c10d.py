import copy
import math
import multiprocessing
import socket
import sys
import tempfile
import unittest

from functools import wraps
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed import c10d
from torch.nn.parallel import distributed_c10d

from common import TestCase


if not c10d.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


TIMEOUT_DEFAULT = 5
TIMEOUT_OVERRIDE = {}

TestSkip = namedtuple('TestSkip', 'exit_code, message')

TEST_SKIPS = {
    "multi-gpu": TestSkip(75, "Need at least 2 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
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


def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]


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
        return c10d.FileStore(self.file.name)


class TCPStoreTest(TestCase, StoreTestBase):
    def _create_store(self):
        addr = 'localhost'
        port = find_free_port()
        return c10d.TCPStore(addr, port, True)


class RendezvousTest(TestCase):
    def test_unknown_handler(self):
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            c10d.rendezvous('invalid://')


class RendezvousFileTest(TestCase):
    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, 'path missing'):
            gen = c10d.rendezvous('file://?rank=0&size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            gen = c10d.rendezvous('file:///tmp/foo?size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'size parameter missing'):
            gen = c10d.rendezvous('file:///tmp/foo?rank=0')
            next(gen)

    def test_nominal(self):
        with tempfile.NamedTemporaryFile() as file:
            url = 'file://%s?size=%d' % (file.name, 2)
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
            gen = c10d.rendezvous('tcp://127.0.0.1?rank=0&size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1:23456?size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'size parameter missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1:23456?rank=0')
            next(gen)

    def test_nominal(self):
        addr = 'localhost'
        port = find_free_port()
        url = 'tcp://%s:%d?size=%d' % (addr, port, 2)
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
        self.port = find_free_port()
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
        for p in self.processes:
            p.join(timeout)
        self._check_return_codes()

    def _check_return_codes(self):
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        first_process = self.processes[0]
        for p in self.processes:
            self.assertEqual(p.exitcode, first_process.exitcode)
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)
        self.assertEqual(first_process.exitcode, 0)


class ProcessGroupGlooTest(MultiProcessTestCase):
    def opts(self):
        opts = c10d.ProcessGroupGloo.Options()
        opts.timeout = 1.0
        opts.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        return opts

    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # Every rank is root once, every tensor index is root once
        for i in range(self.world_size):
            for j in range(2):
                xs = [
                    torch.Tensor([self.rank * self.world_size + 0.0]),
                    torch.Tensor([self.rank * self.world_size + 1.0]),
                ]

                broadcast(xs, i, j)
                self.assertEqual(torch.Tensor([i * self.world_size + j]), xs[0])
                self.assertEqual(torch.Tensor([i * self.world_size + j]), xs[1])

        # Test overloaded convenience function
        x = torch.Tensor([self.rank + 1.0])
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.Tensor([1.0]), x)

    def test_allreduce_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        def allreduce(x, op):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce([x], opts)
            work.wait()

        # Sum
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.SUM)
        self.assertEqual(torch.Tensor([float(self.world_size * (self.world_size + 1) / 2)]), x)

        # Product
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.PRODUCT)
        self.assertEqual(torch.Tensor([float(math.factorial(self.world_size))]), x)

        # Min
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.MIN)
        self.assertEqual(torch.Tensor([1.0]), x)

        # Max
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.MAX)
        self.assertEqual(torch.Tensor([self.world_size]), x)

        # Test overloaded convenience function (defaults to using sum)
        x = torch.Tensor([self.rank + 1.0])
        work = pg.allreduce(x)
        work.wait()
        self.assertEqual(torch.Tensor([float(self.world_size * (self.world_size + 1) / 2)]), x)


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

    def tearDown(self):
        self.file.close()

    @skip_if_not_nccl
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

    @skip_if_not_nccl
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
            for _ in range(self.world_size):
                ls.append(torch.Tensor([0]).cuda(idx))

        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i]).cuda(i))

        allgather(output_ts, tensors)

        # Verification
        for idx, device_ts in enumerate(output_ts):
            for t in device_ts:
                self.assertEqual(torch.Tensor([idx]), t)


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

    def _test_ddp_with_process_group(self, process_group):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = Net()
        ddp_model = distributed_c10d._DistributedDataParallelC10d(
            copy.deepcopy(model).cuda(gpus[0]),
            process_group,
            device_ids=gpus)
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
        store = c10d.TCPStore('localhost', self.port, self.rank == 0)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)
        self._test_ddp_with_process_group(process_group)

    @skip_if_not_multigpu
    @skip_if_not_nccl
    def test_nccl_backend(self):
        store = c10d.TCPStore('localhost', self.port, self.rank == 0)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        self._test_ddp_with_process_group(process_group)

if __name__ == '__main__':
    assert not torch.cuda._initialized, "test_distributed must not have initialized CUDA context on main process"

    unittest.main()
