from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import functools
import sys

import torch
import torch.distributed as dist
from common_distributed import MultiProcessTestCase, skip_for_known_issues
from common_utils import load_tests, run_tests


class WorkerContext(object):
    _INITIALIZED = False
    _WORKER_ID = -1
    _WORLD_SIZE = 0
    _WORKER_NAMES = None

    def __new__(cls, worker_id, world_size):
        cls._WORKER_ID = worker_id
        cls._WORLD_SIZE = world_size
        cls._WORKER_NAMES = ["worker%d" % i for i in range(world_size)]
        cls._INITIALIZED = True
        return cls  # Never create instance.

    @classmethod
    def worker_id(cls):
        assert cls._INITIALIZED is True
        return cls._WORKER_ID

    @classmethod
    def world_size(cls):
        assert cls._INITIALIZED is True
        return cls._WORLD_SIZE

    @classmethod
    def worker_name(cls):
        assert cls._INITIALIZED is True
        if cls._WORKER_ID == -1:
            return ""
        return cls._WORKER_NAMES[cls._WORKER_ID]


def sprint(*args, **kwargs):
    date_str = datetime.datetime.today().strftime("%m%d %H:%M:%S.%f")
    print(
        "{date_str}{worker_indent}{worker_name}:".format(
            date_str=date_str,
            worker_indent=" " * (WorkerContext.worker_id() + 1) * 2,
            worker_name=WorkerContext.worker_name(),
        ),
        *args,
        **kwargs
    )


def watch_call(old_func):
    """Decorate on functions that we would liked to have traces.
        e.g. `dist.rpc = watch_call(dist.rpc)``
    """

    @functools.wraps(old_func)
    def new_func(*args, **kwargs):

        func_name = old_func.__name__
        sprint(
            "-> {func_name} Enter ...\n"
            "  args={args}\n"
            "  kwargs={kwargs}".format(func_name=func_name, args=args, kwargs=kwargs)
        )
        return_value = old_func(*args, **kwargs)
        sprint("<- {func_name} Exit ...".format(func_name=func_name))
        return return_value

    return new_func


def watch_rpc(old_func):
    @functools.wraps(old_func)
    def new_func(to, func, args=None, kwargs=None, async_call=False):
        sprint(
            "RPCall {func_name} to {to},\n"
            "  args={args}\n"
            "  kwargs={kwargs}".format(
                func_name=func.__name__, to=to, args=args, kwargs=kwargs
            )
        )
        return old_func(to, func, args=args, kwargs=kwargs, async_call=async_call)

    return new_func


dist.rpc = watch_rpc(dist.rpc)


def my_function(a, b, c):
    return a + b + c


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


def no_result():
    print("do nothing")


global_var = False


@watch_call
def modify_global_var(global_var_in):
    global global_var
    global_var = global_var_in


@watch_call
def nested_rpc(src_name, ttl, reduce_ttl_on):
    if WorkerContext.worker_name() == reduce_ttl_on:
        ttl -= 1
    if ttl > 0:
        dist.rpc(
            to=src_name,
            func=nested_rpc,
            args=(WorkerContext.worker_name(), ttl, reduce_ttl_on),
        )


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not dist.is_available():
    print("c10d not available, skipping tests")
    sys.exit(0)


def _wrap_with_rpc(func):
    def wrapper(self):
        store = dist.FileStore(self.file.name, self.world_size)
        dist.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )
        WorkerContext(worker_id=self.rank, world_size=self.world_size)
        dist.init_rpc(
            name=WorkerContext.worker_name(),
            num_send_recv_threads=self.num_send_recv_threads,
        )
        func(self)
        dist.join_rpc()

    return wrapper


class RpcTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4

    @property
    def num_send_recv_threads(self):
        return 2

    @_wrap_with_rpc
    def test_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc(
            "worker%d" % dstRank, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @_wrap_with_rpc
    def test_scalar_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, torch.add, args=(torch.ones(n, n), n))
        self.assertEqual(ret, (torch.ones(n, n) + n))

    @_wrap_with_rpc
    def test_async_add(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        fut = dist.rpc(
            "worker%d" % dstRank, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @_wrap_with_rpc
    def test_nonzero(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        ret = dist.rpc("worker%d" % dstRank, torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @_wrap_with_rpc
    def test_multi_rpc(self):
        dstRank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            ret = dist.rpc(
                "worker%d" % dstRank,
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )
            self.assertEqual(ret, torch.ones(n, n) * 2)

    @_wrap_with_rpc
    def test_sync_rpc(self):
        dstRank = (self.rank + 1) % self.world_size
        for i in range(20):
            dist.sync_rpc()
            n = i + self.rank + 1
            ret1 = dist.rpc(
                "worker%d" % dstRank,
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )
            dist.sync_rpc()
            ret2 = dist.rpc("worker%d" % dstRank, torch.add, args=(torch.ones(n, n), 2))
            dist.sync_rpc()
            self.assertEqual(ret1, torch.ones(n, n) * 2)
            self.assertEqual(ret2, torch.ones(n, n) * 3)

    @_wrap_with_rpc
    def test_join_rpc(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc(
            "worker%d" % dstRank, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)
        dist.join_rpc()

        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            dist.rpc(
                "worker%d" % dstRank,
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )

        # it's safe to call join_rpc() multiple times
        dist.join_rpc()

    @_wrap_with_rpc
    def test_py_built_in(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, min, args=(n, n + 1, n + 2))
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @_wrap_with_rpc
    def test_py_user_defined(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, my_function, args=(n, n + 1, n + 2))
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    @_wrap_with_rpc
    def test_py_class_constructor(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, my_class, args=(n,))
        self.assertEqual(ret.a, n)

    @_wrap_with_rpc
    def test_py_class_instance_method(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, my_class(2).my_instance_method, args=(n,))
        self.assertEqual(ret, my_class(2).my_instance_method(n))

    @_wrap_with_rpc
    def test_py_class_method(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, my_class.my_class_method, args=(n, n + 1))
        self.assertEqual(ret, my_class.my_class_method(n, n + 1))

    @_wrap_with_rpc
    def test_py_class_static_method(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, my_class.my_static_method, args=(n + 10,))
        self.assertEqual(ret, my_class.my_static_method(n + 10))

    @_wrap_with_rpc
    def test_py_multi_aync_call(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        fut1 = dist.rpc(
            "worker%d" % dstRank,
            my_class.my_static_method,
            args=(n + 10,),
            async_call=True,
        )
        fut2 = dist.rpc(
            "worker%d" % dstRank, min, args=(n, n + 1, n + 2), async_call=True
        )
        self.assertEqual(fut1.wait(), my_class.my_static_method(n + 10))
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))

    @_wrap_with_rpc
    def test_py_no_return_result(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, no_result)
        self.assertEqual(ret, no_result())

    @_wrap_with_rpc
    def test_py_function_exception(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, no_result, args=(10,))
        expected = "run_python_udf_internal caught exception: no_result() takes 0 positional arguments but 1 was given"
        self.assertEqual(ret, expected)

    @skip_for_known_issues  # No garentee that peer's request futures are all resovled.
    @_wrap_with_rpc
    def test_py_modify_global_var(self):
        n = self.rank + 1
        dstRank = n % self.world_size
        ret = dist.rpc("worker%d" % dstRank, modify_global_var, args=(True,))
        dist.barrier()  # Need an API, dist.rpc_wait_request_futures()
        self.assertEqual(global_var, True)

    @_wrap_with_rpc
    def test_py_nested_rpc(self):
        assert self.world_size >= 2, "Requires 2 workers to reproduce this issue."
        if WorkerContext.worker_name() == "worker0":
            to = "worker1"
            ret = dist.rpc(to, nested_rpc, args=(WorkerContext.worker_name(), self.num_send_recv_threads, to), async_call=False)
            assert ret is None, str(ret)

    @skip_for_known_issues  # Deadlock if all threads are taken for processing RPC.
    @_wrap_with_rpc
    def test_py_nested_rpc_more_than_recv_threads(self):
        assert self.world_size >= 2, "Requires 2 workers to reproduce this issue."
        if WorkerContext.worker_name() == "worker0":
            to = "worker1"
            ret = dist.rpc(to, nested_rpc, args=(WorkerContext.worker_name(), self.num_send_recv_threads + 1, to), async_call=False)
            assert ret is None, str(ret)

if __name__ == "__main__":
    run_tests()
