from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import torch
import torch.distributed as dist

from common_distributed import MultiProcessTestCase
from common_utils import load_tests, run_tests

def my_function(a, b, c):
    return a + b + c

def no_result():
    print("do nothing")

class my_class:
    def __init__(self, a):
        self.a = a

    def my_instance_method(self, b):
        return self.a + b

    @classmethod
    def my_class_method(cls, d, e):
        return d + e

    @staticmethod
    def my_static_method(f):
        return f > 10

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

    @_wrap_with_rpc
    def test_py_built_in(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, min, args=(n, n + 1, n + 2))
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @_wrap_with_rpc
    def test_py_user_defined(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, my_function, args=(n, n + 1, n + 2))
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    @_wrap_with_rpc
    def test_py_class_constructor(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, my_class, args=(n,))
        self.assertEqual(ret.a, n)

    @_wrap_with_rpc
    def test_py_class_instance_method(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, my_class(2).my_instance_method, args=(n,))
        self.assertEqual(ret, my_class(2).my_instance_method(n))

    @_wrap_with_rpc
    def test_py_class_method(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, my_class.my_class_method, args=(n, n + 1))
        self.assertEqual(ret, my_class.my_class_method(n, n + 1))

    @_wrap_with_rpc
    def test_py_class_static_method(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, my_class.my_static_method, args=(n + 10,))
        self.assertEqual(ret, my_class.my_static_method(n + 10))

    @_wrap_with_rpc
    def test_py_multi_aync_call(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        fut1 = dist.rpc('worker%d' % dstRank, my_class.my_static_method, args=(n + 10,), async_call=True)
        fut2 = dist.rpc('worker%d' % dstRank, min, args=(n, n + 1, n + 2), async_call=True)
        self.assertEqual(fut1.wait(), my_class.my_static_method(n + 10))
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))

    @_wrap_with_rpc
    def test_py_no_return_result(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, no_result)
        self.assertEqual(ret, no_result())

    @_wrap_with_rpc
    def test_py_function_exception(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc('worker%d' % dstRank, no_result, args=(10,))
        expected = "run_python_udf_internal caught exception: no_result() takes 0 positional arguments but 1 was given"
        self.assertEqual(ret, expected)

if __name__ == '__main__':
    run_tests()
