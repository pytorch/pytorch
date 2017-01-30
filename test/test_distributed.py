import fcntl
import multiprocessing
import os
import sys
import time
import unittest
from functools import wraps, reduce
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

    def _init_group_test(self):
        group = [1, 2]
        group_id = dist.new_group(group)
        rank = dist.get_rank()
        if rank not in group:
            return ([], None, rank)

        return (group, group_id, rank)

    def _init_global_test(self):
        group = [i for i in range(0, dist.get_num_processes())]
        group_id = dist.group.WORLD
        rank = dist.get_rank()
        return (group, group_id, rank)

    # GET RANK
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

    # SEND RECV
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

    # SEND RECV ANY SOURCE
    def test_send_recv_any_source(self):
        rank = dist.get_rank()
        tensor = _build_tensor(10, rank)
        for dest in range(0, dist.get_num_processes()):
            if dest == rank:
                continue
            dist.send(tensor, dest)

        recv_ranks = set()
        for src in range(0, dist.get_num_processes()):
            if src == rank:
                continue
            tensor = _build_tensor(10, value=-1)
            dist.recv(tensor)
            recv_ranks.add(tensor.resize_(1)[0])

        self.assertEqual(len(recv_ranks), dist.get_num_processes() - 1)
        self._barrier()

    # ISEND
    def test_isend(self):
        rank = dist.get_rank()
        world_size = dist.get_num_processes()

        if rank == 0:
            requests = [
                dist.isend(_build_tensor(dest, 10), dest) for dest in range(1, world_size)
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
    def test_irecv(self):
        rank = dist.get_rank()
        world_size = dist.get_num_processes()

        if rank == 0:
            expected_tensors = [_build_tensor(src, -1) for src in range(1, world_size)]
            requests = [
                dist.irecv(expected_tensors[src - 1], src) for src in range(1, world_size)
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
    def _test_broadcast_helper(self, group, group_id, rank):
        for src in group:
            expected_tensor = _build_tensor(src + 1)
            if rank == src:
                dist.broadcast(expected_tensor, src, group_id)
            else:
                tensor = _build_tensor(src + 1, -1)
                dist.broadcast(tensor, src, group_id)
                self.assertEqual(tensor, expected_tensor)

        self._barrier()

    def test_broadcast(self):
        group, group_id, rank = self._init_global_test()
        self._test_broadcast_helper(group, group_id, rank)

    def test_broadcast_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_broadcast_helper(group, group_id, rank)

    # REDUCE
    def _test_reduce_helper(self, group, group_id, rank, op, master_value, worker_value, expected_value):
        for src in group:
            if rank == src:
                tensor = _build_tensor(src + 1).fill_(master_value)
                dist.reduce(tensor, src, op, group_id)
                self.assertEqual(tensor, _build_tensor(src + 1, expected_value))
            else:
                tensor = _build_tensor(src + 1).fill_(worker_value)
                dist.reduce(tensor, src, op, group_id)

        self._barrier()

    def test_reduce_sum(self):
        group, group_id, rank = self._init_global_test()
        self._test_reduce_helper(
            group, group_id, rank, dist.reduce_op.SUM, 2, 10, 2 + (10 * (len(group) - 1))
        )

    def test_reduce_product(self):
        group, group_id, rank = self._init_global_test()
        self._test_reduce_helper(
            group, group_id, rank, dist.reduce_op.PRODUCT,
            2, 10, reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2)
        )

    def test_reduce_min(self):
        group, group_id, rank = self._init_global_test()
        self._test_reduce_helper(
            group, group_id, rank, dist.reduce_op.MIN, 1010, 1, 1
        )

    def test_reduce_max(self):
        group, group_id, rank = self._init_global_test()
        self._test_reduce_helper(
            group, group_id, rank, dist.reduce_op.MAX, -1, 10, 10
        )

    def test_reduce_group_sum(self):
        group, group_id, rank = self._init_group_test()
        self._test_reduce_helper(
            group, group_id, rank, dist.reduce_op.SUM, 2, 10, 2 + (10 * (len(group) - 1))
        )

    def test_reduce_group_product(self):
        group, group_id, rank = self._init_group_test()
        self._test_reduce_helper(
            group, group_id, rank, dist.reduce_op.PRODUCT,
            2, 10, reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2)
        )

    def test_reduce_group_min(self):
        group, group_id, rank = self._init_group_test()
        self._test_reduce_helper(
            group, group_id, rank, dist.reduce_op.MIN, 1010, 1, 1
        )

    def test_reduce_group_max(self):
        group, group_id, rank = self._init_group_test()
        self._test_reduce_helper(
            group, group_id, rank, dist.reduce_op.MAX, -1, 10, 10
        )

    # ALL REDUCE
    def _test_all_reduce_helper(self, group, group_id, rank, op, master_value, worker_value, expected_value):
        for src in group:
            if rank == src:
                tensor = _build_tensor(src + 1).fill_(master_value)
                dist.all_reduce(tensor, op, group_id)
                self.assertEqual(tensor, _build_tensor(src + 1, expected_value))
            else:
                tensor = _build_tensor(src + 1).fill_(worker_value)
                dist.all_reduce(tensor, op, group_id)
                self.assertEqual(tensor, _build_tensor(src + 1, expected_value))

        self._barrier()

    def test_all_reduce_sum(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.SUM, 2, 10, 2 + (10 * (len(group) - 1))
        )

    def test_all_reduce_product(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.PRODUCT,
            2, 10, reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2)
        )

    def test_all_reduce_min(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.MIN, 1010, 1, 1
        )

    def test_all_reduce_max(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.MAX, -1, 10, 10
        )

    def test_all_reduce_group_sum(self):
        group, group_id, rank = self._init_group_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.SUM, 2, 10, 2 + (10 * (len(group) - 1))
        )

    def test_all_reduce_group_product(self):
        group, group_id, rank = self._init_group_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.PRODUCT,
            2, 10, reduce((lambda x, y: x * y), [10] * (len(group) - 1), 2)
        )

    def test_all_reduce_group_min(self):
        group, group_id, rank = self._init_group_test()
        self._test_all_reduce_helper(
            group, group_id, rank, dist.reduce_op.MIN, 1010, 1, 1
        )

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
            if rank == dest:
                tensors = [_build_tensor(dest + 1, i) for i in group]
                dist.scatter_send(tensors, tensor, group_id)
                self.assertEqual(tensor, expected_tensor)
            else:
                dist.scatter_recv(tensor, dest, group_id)
                self.assertEqual(tensor, expected_tensor)

        self._barrier()

    def test_scatter(self):
        group, group_id, rank = self._init_global_test()
        self._test_scatter_helper(group, group_id, rank)

    def test_scatter_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_scatter_helper(group, group_id, rank)

    # GATHER
    def _test_gather_helper(self, group, group_id, rank):
        for dest in group:
            tensor = _build_tensor(dest + 1, rank)
            if rank == dest:
                tensors = [_build_tensor(dest + 1, -1) for i in group]
                dist.gather_recv(tensors, tensor, group_id)

                expected_tensors = [_build_tensor(dest + 1, i) for i in group]
                for t1, t2 in zip(tensors, expected_tensors):
                    self.assertEqual(t1, t2)
            else:
                dist.gather_send(tensor, dest, group_id)

        self._barrier()

    def test_gather(self):
        group, group_id, rank = self._init_global_test()
        self._test_gather_helper(group, group_id, rank)

    def test_gather_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_gather_helper(group, group_id, rank)

    # ALL GATHER
    def _test_all_gather_helper(self, group, group_id, rank):
        for dest in group:
            tensor = _build_tensor(dest + 1, rank)
            tensors = [_build_tensor(dest + 1, -1) for i in group]
            dist.all_gather(tensors, tensor, group_id)

            expected_tensors = [_build_tensor(dest + 1, i) for i in group]
            for t1, t2 in zip(tensors, expected_tensors):
                self.assertEqual(t1, t2)

        self._barrier()

    def test_all_gather(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_gather_helper(group, group_id, rank)

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

    def test_barrier(self):
        group, group_id, rank = self._init_global_test()
        self._test_barrier_helper(group, group_id, rank)

    def test_barrier_group(self):
        group, group_id, rank = self._init_group_test()
        self._test_barrier_helper(group, group_id, rank)

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
