import fcntl
import multiprocessing
import os
import sys
import time
import unittest
from functools import wraps
from contextlib import contextmanager

import torch
import torch.distributed as dist
from common import TestCase

BACKEND = os.environ['BACKEND']
TEMP_DIR = os.environ['TEMP_DIR']
MASTER_PORT = '29500'
MASTER_ADDR = '127.0.0.1:' + MASTER_PORT


@contextmanager
def _lock():
    lockfile = os.path.join(TEMP_DIR, 'lockfile')
    with open(lockfile, 'w') as lf:
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
        barrier_dir = os.path.join(TEMP_DIR, 'barrier')
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))

    @classmethod
    def sync(cls, timeout=5):
        cls.barrier_id += 1
        barrier_dir = os.path.join(TEMP_DIR, 'barrier')
        pid = str(os.getpid())
        barrier_file = os.path.join(barrier_dir, pid)
        with _lock():
            with open(barrier_file, 'w') as f:
                f.write(str(cls.barrier_id))

        start_time = time.time()
        while True:
            arrived = 0
            with _lock():
                for f_name in os.listdir(barrier_dir):
                    with open(os.path.join(barrier_dir, f_name), 'r') as f:
                        data = f.read()
                        if int(data) >= cls.barrier_id:
                            arrived += 1
            if arrived == dist.get_num_processes():
                break

            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            time.sleep(0.1)


class _DistTestBase(object):

    def _barrier(self, *args, **kwargs):
        Barrier.sync(*args, **kwargs)

    def _reduce(self, reduce_op, expected_tensor, dest, special=None,
                common=None, broadcast=False):
        rank = dist.get_rank()
        world_size = dist.get_num_processes()
        if rank == dest:
            tensor = _build_tensor(rank + 1)
            dist.reduce(tensor, dest, reduce_op)
            self.assertEqual(tensor, expected_tensor)
        elif rank == (dest + 1) % world_size:
            tensor = _build_tensor(dest + 1, value=special)
            dist.reduce(tensor, dest, reduce_op)
        else:
            tensor = _build_tensor(dest + 1, value=common)
            dist.reduce(tensor, dest, reduce_op)
        self._barrier()
        if broadcast:
            if rank == dest:
                dist.broadcast(tensor, dest)
            else:
                tensor.fill_(-1)
                dist.broadcast(tensor, dest)
                self.assertEqual(tensor, expected_tensor)
            self._barrier()

    def test_get_rank(self):
        test_dir = os.path.join(TEMP_DIR, 'test_dir')
        pid = str(os.getpid())
        num_processes = dist.get_num_processes()
        with open(os.path.join(test_dir, pid), 'w') as f:
            f.write(str(dist.get_rank()))

        self._barrier()

        all_ranks = set()
        for f_name in os.listdir(test_dir):
            with open(os.path.join(test_dir, f_name), 'r') as f:
                all_ranks.add(int(f.read()))
        self.assertEqual(len(all_ranks), num_processes)

        self._barrier()

        if dist.get_rank() == 0:
            for f_name in os.listdir(test_dir):
                os.unlink(os.path.join(test_dir, f_name))

        self._barrier()

    def test_send_recv(self):
        rank = dist.get_rank()
        tensor = _build_tensor(rank + 1)
        for dest in range(0, dist.get_num_processes()):
            if dest == rank:
                continue
            dist.send(tensor, dest)

        for src in range(0, dist.get_num_processes()):
            if src == rank:
                continue
            tensor = _build_tensor(src + 1, value=-1)
            expected_tensor = _build_tensor(src + 1)
            dist.recv(tensor, src)
            self.assertEqual(tensor, expected_tensor)

        self._barrier()

    def test_broadcast(self):
        rank = dist.get_rank()
        for src in range(0, dist.get_num_processes()):
            tensor = _build_tensor(src + 1)
            if rank == src:
                dist.broadcast(tensor, src)

            if rank != src:
                tensor.fill_(-1)
                expected_tensor = _build_tensor(src + 1)
                dist.broadcast(tensor, src)
                self.assertEqual(tensor, expected_tensor)
            self._barrier()

    def test_reduce_max(self):
        rank = dist.get_rank()
        world_size = dist.get_num_processes()
        for dest in range(0, world_size):
            tensor = _build_tensor(rank + 1, value=world_size+1)
            self._reduce(dist.reduce_op.MAX, tensor, dest,
                         special=world_size+1)

    def test_reduce_min(self):
        rank = dist.get_rank()
        world_size = dist.get_num_processes()
        for dest in range(0, world_size):
            tensor = _build_tensor(rank + 1, value=-1)
            self._reduce(dist.reduce_op.MIN, tensor, dest, special=-1)

    def test_reduce_product(self):
        rank = dist.get_rank()
        world_size = dist.get_num_processes()
        for dest in range(0, world_size):
            tensor = _build_tensor(rank + 1).mul_(2)
            self._reduce(dist.reduce_op.PRODUCT, tensor, dest, special=2,
                         common=1)

    def test_reduce_sum(self):
        rank = dist.get_rank()
        world_size = dist.get_num_processes()
        for dest in range(0, world_size):
            tensor = _build_tensor(rank + 1).mul_(world_size)
            self._reduce(dist.reduce_op.SUM, tensor, dest)

    def test_all_reduce_max(self):
        world_size = dist.get_num_processes()
        for dest in range(0, world_size):
            tensor = _build_tensor(dest + 1, value=world_size+1)
            self._reduce(dist.reduce_op.MAX, tensor, dest,
                         special=world_size+1, broadcast=True)

    def test_all_reduce_min(self):
        world_size = dist.get_num_processes()
        for dest in range(0, world_size):
            tensor = _build_tensor(dest + 1, value=-1)
            self._reduce(dist.reduce_op.MIN, tensor, dest, special=-1,
                         broadcast=True)

    def test_all_reduce_product(self):
        rank = dist.get_rank()
        world_size = dist.get_num_processes()
        for dest in range(0, world_size):
            tensor = _build_tensor(dest + 1).mul_(2)
            self._reduce(dist.reduce_op.PRODUCT, tensor, dest, special=2,
                         common=1, broadcast=True)

    def test_all_reduce_sum(self):
        world_size = dist.get_num_processes()
        for dest in range(0, world_size):
            tensor = _build_tensor(dest + 1).mul_(world_size)
            self._reduce(dist.reduce_op.SUM, tensor, dest, broadcast=True)

if BACKEND == 'tcp':
    WORLD_SIZE = os.environ['WORLD_SIZE']

    class TestTCP(TestCase, _DistTestBase):

        MANAGER_PROCESS_RANK = -1
        JOIN_TIMEOUT = 5

        @staticmethod
        def manager_join(fn):
            @wraps(fn)
            def wrapper(self):
                if self.rank == self.MANAGER_PROCESS_RANK:
                    self._join_and_reduce()
                else:
                    fn(self)
            return wrapper

        @classmethod
        def setUpClass(cls):
            os.environ['MASTER_ADDR'] = MASTER_ADDR
            os.environ['MASTER_PORT'] = MASTER_PORT
            os.environ['WORLD_SIZE'] = WORLD_SIZE
            for attr in dir(cls):
                if attr.startswith('test'):
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
            os.environ['RANK'] = str(rank)
            name = 'process ' + str(rank)
            process = multiprocessing.Process(target=self._run, name=name,
                                              args=(rank,))
            process.start()
            return process

        def _run(self, rank):
            self.rank = rank
            dist.init_process_group(backend=BACKEND)
            # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
            # We're retreiving a corresponding test and executing it.
            getattr(self, self.id().split(".")[2])()
            sys.exit(0)

        def _join_and_reduce(self):
            for p in self.processes:
                p.join(self.JOIN_TIMEOUT)
                self.assertEqual(p.exitcode, 0)

elif BACKEND == 'mpi':
    dist.init_process_group(backend='mpi')

    class TestMPI(TestCase, _DistTestBase):
        pass


if __name__ == '__main__':
    unittest.main()
