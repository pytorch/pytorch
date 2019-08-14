import sys

import torch
import torch.distributed as dist

from common_distributed import MultiProcessTestCase
from common_utils import load_tests, run_tests


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


if not dist.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


def _wrap_with_rpc(func):
    def wrapper(self):
        store = dist.FileStore(self.file.name, self.world_size)
        dist.init_process_group(backend='gloo', rank=self.rank,
                                world_size=self.world_size, store=store)
        dist.init_rpc('worker%d' % self.rank)
        func(self)
        dist.join_rpc()

    return wrapper


class RpcTest(MultiProcessTestCase):

    @property
    def world_size(self):
        return 4

    @_wrap_with_rpc
    def test_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, torch.add,
                       args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @_wrap_with_rpc
    def test_scalar_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, torch.add,
                       args=(torch.ones(n, n), n))
        self.assertEqual(ret, (torch.ones(n, n) + n))

    @_wrap_with_rpc
    def test_async_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        fut = dist.rpc('worker%d' % dstRank,
                       torch.add,
                       args=(torch.ones(n, n), torch.ones(n, n)),
                       async_call=True)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @_wrap_with_rpc
    def test_nonzero(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        ret = dist.rpc('worker%d' % dstRank, torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @_wrap_with_rpc
    def test_multi_rpc(self):
        dstRank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            ret = dist.rpc('worker%d' % dstRank, torch.add,
                           args=(torch.ones(n, n), torch.ones(n, n)))
            self.assertEqual(ret, torch.ones(n, n) * 2)

    @_wrap_with_rpc
    def test_sync_rpc(self):
        dstRank = (self.rank + 1) % self.world_size
        for i in range(20):
            dist.sync_rpc()
            n = i + self.rank + 1
            ret1 = dist.rpc('worker%d' % dstRank, torch.add,
                            args=(torch.ones(n, n), torch.ones(n, n)))
            dist.sync_rpc()
            ret2 = dist.rpc('worker%d' % dstRank, torch.add,
                            args=(torch.ones(n, n), 2))
            dist.sync_rpc()
            self.assertEqual(ret1, torch.ones(n, n) * 2)
            self.assertEqual(ret2, torch.ones(n, n) * 3)

    @_wrap_with_rpc
    def test_join_rpc(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, torch.add,
                       args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(ret, torch.ones(n, n) * 2)
        dist.join_rpc()

        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            dist.rpc('worker%d' % dstRank, torch.add,
                     args=(torch.ones(n, n), torch.ones(n, n)))

        # it's safe to call join_rpc() multiple times
        dist.join_rpc()

if __name__ == '__main__':
    run_tests()
