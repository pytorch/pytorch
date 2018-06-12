import math
import multiprocessing
import sys
import tempfile
import unittest
from functools import wraps

import torch
import torch.distributed.c10d as c10d

from common import TestCase


TCP_ADDR = '127.0.0.1'
TCP_PORT = 29500

TIMEOUT_DEFAULT = 5
TIMEOUT_OVERRIDE = {}


def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)


if not c10d.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


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
        return c10d.TCPStore(TCP_ADDR, TCP_PORT, True)


class ProcessGroupGlooTest(TestCase):
    MAIN_PROCESS_RANK = -1

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
        self.size = 4
        self.file = tempfile.NamedTemporaryFile()
        self.processes = [self._spawn_process(rank) for rank in range(int(self.size))]

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

    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # Every rank is root once, every tensor index is root once
        for i in range(self.size):
            for j in range(2):
                xs = [
                    torch.Tensor([self.rank * self.size + 0.0]),
                    torch.Tensor([self.rank * self.size + 1.0]),
                ]

                broadcast(xs, i, j)
                self.assertEqual(torch.Tensor([i * self.size + j]), xs[0])
                self.assertEqual(torch.Tensor([i * self.size + j]), xs[1])

        # Test overloaded convenience function
        x = torch.Tensor([self.rank + 1.0])
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.Tensor([1.0]), x)

    def test_allreduce_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.size)

        def allreduce(x, op):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce([x], opts)
            work.wait()

        # Sum
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.SUM)
        self.assertEqual(torch.Tensor([float(self.size * (self.size + 1) / 2)]), x)

        # Product
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.PRODUCT)
        self.assertEqual(torch.Tensor([float(math.factorial(self.size))]), x)

        # Min
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.MIN)
        self.assertEqual(torch.Tensor([1.0]), x)

        # Max
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.MAX)
        self.assertEqual(torch.Tensor([self.size]), x)

        # Test overloaded convenience function (defaults to using sum)
        x = torch.Tensor([self.rank + 1.0])
        work = pg.allreduce(x)
        work.wait()
        self.assertEqual(torch.Tensor([float(self.size * (self.size + 1) / 2)]), x)


if __name__ == '__main__':
    unittest.main()
