import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from common_cuda import TEST_MULTIGPU
from common_utils import TestCase, load_tests, run_tests
from common_utils import NO_MULTIPROCESSING_SPAWN

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not dist.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


if NO_MULTIPROCESSING_SPAWN:
    print('spawn not available, skipping tests')
    sys.exit(0)

# tests with scalar args requires https://github.com/pytorch/pytorch/pull/22817
class RpcTest(TestCase):

    world_size = 2

    @classmethod
    def opts(cls, threads=2):
        opts = dist.ProcessGroupGloo.Options()
        opts.devices = [dist.ProcessGroupGloo.create_tcp_device(interface="lo")]
        opts.timeout = 5.0
        opts.threads = threads
        return opts

    @classmethod
    def _init_rpc(cls, rank, filename, world_size):
        store = dist.FileStore(filename, world_size)
        dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size, store=store)
        dist.init_rpc('worker%d' % rank)

    @classmethod
    def _destroy_rpc(cls):
        dist.destroy_rpc()
        dist.destroy_process_group(dist.group.WORLD)

    def _test_multiprocess(self, f, n_output, world_size):
        # file store will delete the test file on destruction
        file = tempfile.NamedTemporaryFile(delete=False)
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(10)
        p2c = ctx.Queue(10)
        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, file.name, c2p, p2c))

            p.start()
            ps.append(p)

        for _ in range(world_size * n_output):
            pid, expected, result = c2p.get()
            self.assertEqual(
                expected,
                result,
                (
                    "Expect rank {} to receive tensor {} but got {}."
                ).format(pid, expected, result)
            )

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join(2)

    # Why classmethod? multiprocessing cannot pickle TestCase subclass when in
    # spawn mode. See https://bugs.python.org/issue33884.
    @classmethod
    def _test_add(cls, rank, filename, c2p, p2c):
        RpcTest._init_rpc(rank, filename, 2)

        if rank == 0:
            ret = dist.rpc_sync('worker1', 'aten::add', torch.ones(2, 2), torch.ones(2, 2))
            c2p.put((rank, torch.ones(2, 2) * 2, ret))
        else:
            ret = dist.rpc_sync('worker0', 'aten::add', torch.ones(3, 3), torch.zeros(3, 3))
            c2p.put((rank, torch.ones(3, 3), ret))
        p2c.get()

        RpcTest._destroy_rpc()

    def test_add(self):
        self._test_multiprocess(RpcTest._test_add, 1, self.world_size)

    @classmethod
    def _test_nonzero(cls, rank, filename, c2p, p2c):
        RpcTest._init_rpc(rank, filename, 2)

        if rank == 0:
            x = torch.ones(2, 2)
            x[0][0] = 0
            ret = dist.rpc_sync('worker1', 'aten::nonzero', x)
            c2p.put((rank, x.nonzero(), ret))
        else:
            x = torch.ones(3, 3)
            x[1][1] = 0
            ret = dist.rpc_sync('worker0', 'aten::nonzero', x)
            c2p.put((rank, x.nonzero(), ret))
        p2c.get()

        RpcTest._destroy_rpc()

    def test_nonezero(self):
        self._test_multiprocess(RpcTest._test_nonzero, 1, self.world_size)

    @classmethod
    def _test_multi_rpc(cls, rank, filename, c2p, p2c):
        RpcTest._init_rpc(rank, filename, 2)

        rpc_num = 20
        if rank == 0:
            for _ in range(rpc_num):
                ret = dist.rpc_sync('worker1', 'aten::add', torch.ones(2, 2), torch.ones(2, 2))
                c2p.put((rank, torch.ones(2, 2) * 2, ret))
        else:
            for _ in range(rpc_num):
                ret = dist.rpc_sync('worker0', 'aten::add', torch.ones(3, 3), torch.zeros(3, 3))
                c2p.put((rank, torch.ones(3, 3), ret))
        p2c.get()

        RpcTest._destroy_rpc()

    def test_multi_rpc(self):
        self._test_multiprocess(RpcTest._test_multi_rpc, 20, self.world_size)

    @classmethod
    def _test_4_workers(cls, rank, filename, c2p, p2c):
        RpcTest._init_rpc(rank, filename, 4)

        if rank == 0:
            ret = dist.rpc_sync('worker1', 'aten::add', torch.ones(2, 2), torch.ones(2, 2))
            c2p.put((rank, torch.ones(2, 2) * 2, ret))
        elif rank == 1:
            ret = dist.rpc_sync('worker0', 'aten::add', torch.ones(3, 3), torch.zeros(3, 3))
            c2p.put((rank, torch.ones(3, 3), ret))
        elif rank == 2:
            ret = dist.rpc_sync('worker1', 'aten::add', torch.ones(4, 4), torch.ones(4, 4))
            c2p.put((rank, torch.ones(4, 4) * 2, ret))
        else:
            ret = dist.rpc_sync('worker0', 'aten::add', torch.ones(5, 5), torch.zeros(5, 5))
            c2p.put((rank, torch.ones(5, 5), ret))

        p2c.get()

        RpcTest._destroy_rpc()

    def test_4_workers(self):
        self._test_multiprocess(RpcTest._test_4_workers, 1, 4)


if __name__ == '__main__':
    run_tests()
