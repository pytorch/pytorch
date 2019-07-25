import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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

    # Why classmethod? multiprocessing cannot pickle TestCase subclass when in
    # spawn mode. See https://bugs.python.org/issue33884.
    @classmethod
    def _test_process(cls, rank, func, filename, c2p, p2c, world_size):
        RpcTest._init_rpc(rank, filename, world_size)

        result, expected = func(rank, world_size)
        c2p.put((rank, expected, result))
        p2c.get()

        RpcTest._destroy_rpc()

    def _test_multiprocess(self, func, world_size):
        # file store will delete the test file on destruction
        file = tempfile.NamedTemporaryFile(delete=False)
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(10)
        p2c = ctx.Queue(10)

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=RpcTest._test_process,
                args=(i, func, file.name, c2p, p2c, world_size))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, expected, result = c2p.get()
            self.assertEqual(
                expected,
                result,
                (
                    "Expect rank {} to receive tensor {} but got {}."
                ).format(rank, expected, result)
            )

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join(2)

    @classmethod
    def _test_add(cls, rank, world_size):
        n = rank + 1
        dstRank = n % world_size
        ret = dist.rpc_sync('worker%d' % dstRank,
                            'aten::add', torch.ones(n, n), torch.ones(n, n))
        return ret, torch.ones(n, n) * 2

    def test_add(self):
        self._test_multiprocess(RpcTest._test_add, self.world_size)

    @classmethod
    def _test_async_add(cls, rank, world_size):
        n = rank + 1
        dstRank = n % world_size
        fut = dist.rpc_async('worker%d' % dstRank,
                             'aten::add',
                             torch.ones(n, n),
                             torch.ones(n, n))
        fut.wait()
        return fut.get(), torch.ones(n, n) * 2

    def test_async_add(self):
        self._test_multiprocess(RpcTest._test_async_add, self.world_size)

    @classmethod
    def _test_nonzero(cls, rank, world_size):
        n = rank + 1
        dstRank = n % world_size
        x = torch.ones(2, 2)
        x[rank][rank] = 0
        ret = dist.rpc_sync('worker%d' % dstRank, 'aten::nonzero', x)
        return ret, x.nonzero()

    def test_nonezero(self):
        self._test_multiprocess(RpcTest._test_nonzero, self.world_size)

    @classmethod
    def _test_multi_rpc(cls, rank, world_size):
        dstRank = (rank + 1) % world_size
        result = []
        expected = []
        for i in range(20):
            n = i + rank + 1
            ret = dist.rpc_sync('worker%d' % dstRank,
                                'aten::add', torch.ones(n, n), torch.ones(n, n))
            result.append(ret)
            expected.append(torch.ones(n, n) * 2)

        return result, expected

    def test_multi_rpc(self):
        self._test_multiprocess(RpcTest._test_multi_rpc, self.world_size)

    @classmethod
    def _test_4_workers(cls, rank, world_size):
        n = rank + 1
        dstRank = n % world_size
        ret = dist.rpc_sync('worker%d' % dstRank,
                            'aten::add', torch.ones(n, n), torch.ones(n, n))
        return ret, torch.ones(n, n) * 2

    def test_4_workers(self):
        world_size = 4
        self._test_multiprocess(RpcTest._test_4_workers, world_size)


if __name__ == '__main__':
    run_tests()
