import sys
import tempfile
import time

from functools import wraps


import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from common_distributed import MultiProcessTestCase
from common_utils import TestCase, load_tests, run_tests
from common_utils import NO_MULTIPROCESSING_SPAWN


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not dist.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


def _init_rpc(func):
    def wrapper(self):
        store = dist.FileStore(self.file.name, self.world_size)
        dist.init_process_group(backend='gloo', rank=self.rank,
                                world_size=self.world_size, store=store)
        dist.init_rpc('worker%d' % self.rank)
        func(self)
        dist.destroy_rpc()
        dist.destroy_process_group(dist.group.WORLD)

    return wrapper

class RpcTest(MultiProcessTestCase):

    @property
    def world_size(self):
        return 4

    @_init_rpc
    def test_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc_sync('worker%d' % dstRank, torch.add,
                            args=(torch.ones(n, n), torch.ones(n, n)))
        dist.barrier()
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @_init_rpc
    def test_scalar_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc_sync('worker%d' % dstRank, torch.add,
                            args=(torch.ones(n, n), n))
        self.assertEqual(ret, (torch.ones(n, n) + n))

    @_init_rpc
    def test_async_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        fut = dist.rpc_async('worker%d' % dstRank, torch.add,
                             args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)


    @_init_rpc
    def test_nonzero(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        ret = dist.rpc_sync('worker%d' % dstRank, torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @_init_rpc
    def test_multi_rpc(self):
        dstRank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            ret = dist.rpc_sync('worker%d' % dstRank, torch.add,
                                args=(torch.ones(n, n), torch.ones(n, n)))

            self.assertEqual(ret, torch.ones(n, n) * 2)

        dist.barrier()


if __name__ == '__main__':
    run_tests()
