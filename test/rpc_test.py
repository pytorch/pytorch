from __future__ import absolute_import, division, print_function, unicode_literals

import concurrent.futures
from datetime import timedelta
import sys
import unittest
from collections import namedtuple
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from common_utils import load_tests
from dist_utils import INIT_METHOD_TEMPLATE, TEST_CONFIG, dist_init
from torch.distributed.rpc.internal import PythonUDF, _internal_rpc_pickler


def requires_process_group_agent(message=""):
    def decorator(old_func):
        return unittest.skipUnless(
            TEST_CONFIG.rpc_backend_name == "PROCESS_GROUP", message
        )(old_func)

    return decorator


VALUE_FUTURE = concurrent.futures.Future()


def stub_start_rpc_backend_handler(
    store, self_name, self_rank, worker_name_to_id, *args, **kwargs
):
    return mock.Mock()  # RpcAgent.


def set_value(value):
    VALUE_FUTURE.set_result(value)


# it is used to test python user defined function over rpc
# classes and functions are used to test python user defined class and
# methods over rpc
TensorClass = namedtuple("TensorClass", ["tensors"])


class MyPickleClass:
    def __init__(self):
        self.t = None

    def __getstate__(self):
        (pickled_python_udf, tensors) = _internal_rpc_pickler.serialize(
            PythonUDF(my_tensor_function, (torch.ones(2, 2), torch.ones(2, 2)), None)
        )
        return (pickled_python_udf, tensors)

    def __setstate__(self, obj):
        python_udf = _internal_rpc_pickler.deserialize(obj[0], obj[1])
        result = python_udf.func(python_udf.args[0], python_udf.args[1])
        self.t = result

    def set(self, val):
        self.t = val


class MyClass:
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

    def increment_value(self, increment):
        self.a += increment

    def get_value(self):
        return self.a


def _call_method_on_rref(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def run_nested_pickle(pickle_cls_instance, tensor):
    return pickle_cls_instance.t + tensor


def build_complex_tensors():
    a = torch.ones(3, 3)
    b = [a, a]
    c = [b, b]
    d = [a, b]
    e = {a: d}
    return [a, b, c, d, e]


def my_function(a, b, c):
    return a + b + c


def my_tensor_function(a, b):
    return a + b


def my_complex_tensor_function(list_input, tensor_class_input, dict_input):
    res = list_input[0]
    for t in list_input:
        res += t
    for k, v in dict_input.items():
        res += v
    complex_tensors = tensor_class_input.tensors
    return (res, complex_tensors[0], complex_tensors[1], complex_tensors[2])


def my_rref_function(rref_a, rref_b):
    return rref_a.to_here() + rref_b.to_here()


def no_result():
    print("do nothing")


def nested_rpc(dst):
    return rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))


def multi_layer_nested_async_rpc(dst, world_size, ttl):
    # this method returns immediately without blocking the callee, but will
    # generate additional requests.
    if ttl > 0:
        current_dst = "worker{}".format(dst)
        next_dst = (dst + 1) % world_size
        rpc.rpc_async(
            current_dst,
            multi_layer_nested_async_rpc,
            args=(next_dst, world_size, ttl - 1),
        )
        return 0


def nested_rref(dst):
    return (
        rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1)),
        rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 2)),
    )


def nested_remote(dst):
    rref = rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 3))
    return rref.to_here()


def rref_forward_chain(dst, world_size, rref, ttl):
    if ttl > 0:
        current_dst = "worker{}".format(dst)
        next_dst = (dst + 1) % world_size
        ret_rref = rpc.remote(
            current_dst, rref_forward_chain, args=(next_dst, world_size, rref, ttl - 1)
        )
        return [ret_rref]
    else:
        return rref.to_here()


def rpc_return_rref(dst):
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))


def light_rpc():
    return 0


def heavy_rpc(tensor):
    for i in range(1, 100):
        tensor *= i
        tensor /= i + 1
    return 0


def raise_func():
    raise ValueError("Expected error")


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


@unittest.skipIf(
    sys.version_info < (3, 0),
    "Pytorch distributed rpc package " "does not support python2",
)
class RpcTest(object):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return INIT_METHOD_TEMPLATE.format(file_name=self.file_name)

    @dist_init
    def test_worker_id(self):
        n = self.rank + 1
        peer_rank = n % self.world_size
        self_worker_info = rpc.get_worker_info()
        peer_worker_info = rpc.get_worker_info("worker{}".format(peer_rank))

        self.assertEqual(self_worker_info.name, "worker{}".format(self.rank))
        self.assertEqual(peer_worker_info.name, "worker{}".format(peer_rank))

        with self.assertRaisesRegex(RuntimeError, "Unknown destination worker"):
            unknown_worker_id = rpc.get_worker_info("WorkerUnknown")

    @dist_init
    def test_self_add(self):
        self_worker_info = rpc.get_worker_info()
        self_worker_name = "worker{}".format(self.rank)

        with self.assertRaisesRegex(
            RuntimeError, "does not support making RPC calls to self"
        ):
            rpc.rpc_sync(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))

        with self.assertRaisesRegex(
            RuntimeError, "does not support making RPC calls to self"
        ):
            rpc.rpc_sync(self_worker_name, torch.add, args=(torch.ones(2, 2), 1))

    @mock.patch.object(torch.distributed.autograd, "_init")
    @mock.patch.object(torch.distributed.rpc.api, "_start_rpc_agent")
    @dist_init(setup_model_parallel=False)
    def test_register_rpc_backend_and_start_rpc_backend(
        self, mock_rpc_agent, mock_dist_autograd_init
    ):
        backend_name = "stub_backend"

        backend = rpc.backend_registry.register_backend(
            backend_name, stub_start_rpc_backend_handler
        )

        with self.assertRaisesRegex(
            RuntimeError, "^RPC backend .+: already registered$"
        ):
            rpc.backend_registry.register_backend(
                backend_name, stub_start_rpc_backend_handler
            )

        rpc.init_model_parallel(
            self_name="worker1",
            backend=backend,
            init_method=self.init_method,
            self_rank=self.rank,
            worker_name_to_id=self.worker_name_to_id,
        )

    @requires_process_group_agent("PROCESS_GROUP rpc backend specific test, skip")
    @dist_init(setup_model_parallel=False)
    def test_duplicate_name(self):
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            store, _, _ = next(torch.distributed.rendezvous(
                self.init_method, rank=self.rank, world_size=self.world_size
            ))
            rpc._init_rpc(
                backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
                store=store,
                self_name="duplicate_name",
                self_rank=self.rank,
                worker_name_to_id=self.worker_name_to_id,
            )
        rpc.join_rpc()

    @dist_init(setup_model_parallel=False)
    def test_reinit(self):
        rpc.init_model_parallel(
            self_name="worker{}".format(self.rank),
            backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
            init_method=self.init_method,
            self_rank=self.rank,
            worker_name_to_id=self.worker_name_to_id,
        )

        # This is for the below `dist.barrier`.
        # For `RpcAgent` other than `ProcessGroupAgent`,
        # no `_default_pg` is initialized.
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method=self.init_method,
                rank=self.rank,
                world_size=self.world_size,
            )
        # Wait for all init to complete.
        dist.barrier()

        with self.assertRaisesRegex(RuntimeError, "is already initialized"):
            rpc.init_model_parallel(
                self_name="worker{}".format(self.rank),
                backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
                init_method=self.init_method,
                self_rank=self.rank,
                worker_name_to_id=self.worker_name_to_id,
            )
        rpc.join_rpc()

    @dist_init(setup_model_parallel=False)
    def test_invalid_names(self):
        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            store, _, _ = next(torch.distributed.rendezvous(
                self.init_method, rank=self.rank, world_size=self.world_size
            ))
            rpc._init_rpc(
                backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
                store=store,
                self_name="abc*",
                self_rank=self.rank,
                worker_name_to_id=self.worker_name_to_id,
                num_send_recv_threads=16,
            )

        base_file_name = self.file_name
        # Use a different file path for FileStore to avoid rendezvous mismatch.
        self.file_name = base_file_name + "1"

        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            store, _, _ = next(torch.distributed.rendezvous(
                self.init_method, rank=self.rank, world_size=self.world_size
            ))
            rpc._init_rpc(
                backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
                store=store,
                self_name=" ",
                self_rank=self.rank,
                worker_name_to_id=self.worker_name_to_id,
                num_send_recv_threads=16,
            )

        # Use a different file path for FileStore to avoid rendezvous mismatch.
        self.file_name = base_file_name + "2"
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            store, _, _ = next(torch.distributed.rendezvous(
                self.init_method, rank=self.rank, world_size=self.world_size
            ))
            rpc._init_rpc(
                backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
                store=store,
                self_name="",
                self_rank=self.rank,
                worker_name_to_id=self.worker_name_to_id,
                num_send_recv_threads=16,
            )

        # Use a different file path for FileStore to avoid rendezvous mismatch.
        self.file_name = base_file_name + "3"
        # If the number in the message does not match, it is likely that the
        # value of MAX_NAME_LEN in RPC WorkerInfo has changed.
        with self.assertRaisesRegex(RuntimeError, "shorter than 128"):
            store, _, _ = next(torch.distributed.rendezvous(
                self.init_method, rank=self.rank, world_size=self.world_size
            ))
            rpc._init_rpc(
                backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
                store=store,
                self_name="".join(["a" for i in range(500)]),
                self_rank=self.rank,
                worker_name_to_id=self.worker_name_to_id,
                num_send_recv_threads=16,
            )

        from torch.distributed.rpc.api import _agent
        self.assertEqual(_agent, None)
        # join_rpc() should not do anything as _agent is None
        rpc.join_rpc()
        # We need this barrier here because although init_process_group is
        # blocking, it does not guarantee that all ranks are done with
        # initialization after the call. We did run into issues with it where
        # rank 3 crashed with "connection closed by peer" RuntimeError, which is
        # caused by other ranks exit before rank 3 is ready. This can be fixed
        # by adding a collective call to sync all processes.
        #
        # We decided not fixing this issue in init_process_group because it
        # would add extra overhead to the call, and normal use cases won't
        # create a progress group and exit without doing anything. Hence, it is
        # not worthy to introduce the overhead just for this test case.

    @dist_init
    def test_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_add_with_id(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        workder_info = rpc.get_worker_info("worker{}".format(dst_rank))

        ret = rpc.rpc_sync(
            workder_info, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_scalar_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), n)
        )
        self.assertEqual(ret, (torch.ones(n, n) + n))

    @dist_init
    def test_async_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_nonzero(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        ret = rpc.rpc_sync("worker{}".format(dst_rank), torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @dist_init
    def test_multi_rpc(self):
        dst_rank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            ret = rpc.rpc_sync(
                "worker{}".format(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )
            self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_sync_rpc(self):
        dst_rank = (self.rank + 1) % self.world_size
        for i in range(20):
            rpc.sync_rpc()
            n = i + self.rank + 1
            ret1 = rpc.rpc_sync(
                "worker{}".format(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )
            rpc.sync_rpc()
            ret2 = rpc.rpc_sync(
                "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), 2)
            )
            rpc.sync_rpc()
            self.assertEqual(ret1, torch.ones(n, n) * 2)
            self.assertEqual(ret2, torch.ones(n, n) * 3)

    @dist_init(setup_model_parallel=False)
    def test_join_rpc(self):
        # Initialize RPC.
        rpc.init_model_parallel(
            self_name="worker%d" % self.rank,
            backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
            init_method=self.init_method,
            self_rank=self.rank,
            worker_name_to_id=self.worker_name_to_id,
        )

        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)
        rpc.join_rpc()

        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            rpc.rpc_sync(
                "worker{}".format(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )

        # it's safe to call join_rpc() multiple times
        rpc.join_rpc()

    @dist_init
    def test_expected_src(self):
        dst_rank = (self.rank + 1) % self.world_size
        expected_src_rank = (self.rank - 1) % self.world_size
        ret = rpc.rpc_sync("worker{}".format(dst_rank), set_value, args=(self.rank,))
        value = VALUE_FUTURE.result()
        self.assertEqual(value, expected_src_rank)

    @dist_init
    def test_py_built_in(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync("worker{}".format(dst_rank), min, args=(n, n + 1, n + 2))
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @dist_init
    def test_py_user_defined(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    @dist_init
    def test_py_class_constructor(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync("worker{}".format(dst_rank), MyClass, args=(n,))
        self.assertEqual(ret.a, n)

    @dist_init
    def test_py_class_instance_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank), MyClass(2).my_instance_method, args=(n,)
        )
        self.assertEqual(ret, MyClass(2).my_instance_method(n))

    @dist_init
    def test_py_class_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank), MyClass.my_class_method, args=(n, n + 1)
        )
        self.assertEqual(ret, MyClass.my_class_method(n, n + 1))

    @dist_init
    def test_py_class_static_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank), MyClass.my_static_method, args=(n + 10,)
        )
        self.assertEqual(ret, MyClass.my_static_method(n + 10))

    @dist_init
    def test_py_multi_async_call(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker_info = rpc.get_worker_info("worker{}".format(dst_rank))
        fut1 = rpc.rpc_async(dst_worker_info, MyClass.my_static_method, args=(n + 10,))
        fut2 = rpc.rpc_async(dst_worker_info, min, args=(n, n + 1, n + 2))
        self.assertEqual(fut1.wait(), MyClass.my_static_method(n + 10))
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))

    @dist_init
    def test_py_no_return_result(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync("worker{}".format(dst_rank), no_result)
        self.assertEqual(ret, no_result())

    @dist_init
    def test_py_tensors(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank),
            my_tensor_function,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, my_tensor_function(torch.ones(n, n), torch.ones(n, n)))

    @dist_init
    def test_py_tensors_multi_async_call(self):
        futs = []
        n = self.rank + 1
        dst_rank = n % self.world_size
        for i in range(100):
            fut = rpc.rpc_async(
                "worker{}".format(dst_rank),
                my_tensor_function,
                args=(torch.ones(i, i), torch.ones(i, i)),
            )
            futs.append(fut)

        j = 0
        for fut in futs:
            self.assertEqual(
                fut.wait(), my_tensor_function(torch.ones(j, j), torch.ones(j, j))
            )
            j += 1

    @dist_init
    def test_py_tensors_in_container(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        a = [torch.ones(n, n), torch.ones(n, n)]
        b = TensorClass(build_complex_tensors())
        c = {"foo": torch.ones(n, n), "bar": torch.ones(n, n)}
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank), my_complex_tensor_function, args=(a, b, c)
        )
        self.assertEqual(ret, my_complex_tensor_function(a, b, c))

    @dist_init
    def test_py_nested_pickle(self):
        n = self.rank + 1
        dst_rank = n % self.world_size

        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank),
            run_nested_pickle,
            args=(MyPickleClass(), torch.ones(2, 2)),
        )

        m = MyPickleClass()
        m.set(my_tensor_function(torch.ones(2, 2), torch.ones(2, 2)))
        self.assertEqual(ret, run_nested_pickle(m, torch.ones(2, 2)))

    @dist_init
    def test_py_function_exception(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        with self.assertRaisesRegex(Exception, "TypeError"):
            ret = rpc.rpc_sync("worker{}".format(dst_rank), no_result, args=(10,))

    @dist_init
    def test_py_raise_in_user_func(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async("worker{}".format(dst_rank), raise_func)
        with self.assertRaisesRegex(Exception, "ValueError"):
            fut.wait()

    @dist_init
    def test_nested_rpc(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank),
            nested_rpc,
            args=("worker{}".format(self.rank),),
        )
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    def _stress_test_rpc(self, f, repeat=1000, args=()):
        import time

        n = self.rank + 1
        dst_rank = n % self.world_size
        futs = []
        tik = time.time()
        for _ in range(repeat):
            fut = rpc.rpc_async("worker{}".format(dst_rank), f, args=args)
            futs.append(fut)

        for fut in futs:
            self.assertEqual(fut.wait(), 0)
        tok = time.time()
        print(
            "Rank {} finished testing {} {} times in {} seconds.".format(
                self.rank, f.__name__, repeat, tok - tik
            )
        )

    @dist_init
    def test_stress_light_rpc(self):
        self._stress_test_rpc(light_rpc)

    @dist_init
    def test_stress_heavy_rpc(self):
        self._stress_test_rpc(heavy_rpc, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_builtin_remote_ret(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(rref.to_here(), torch.ones(n, n) * 2)

    @dist_init
    def test_asymmetric_load_with_join(self):
        """Test graceful termination."""
        # worker0 drives and waits for worker1 and worker2
        # throughout the test.
        if self.rank == 0:
            assert self.world_size >= 3

            num_repeat = 100
            futs = []

            # Phase 1: Only worker1 has workload.
            dst = "worker1"
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, heavy_rpc, args=(torch.ones(100, 100),))
                futs.append(fut)

            for fut in futs:
                fut.wait()
                self.assertEqual(fut.wait(), 0)

            # Phase 2: Only worker2 has workload.
            # If join is not correctly implemented,
            # worker2 should be closed by now.
            dst = "worker2"
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, heavy_rpc, args=(torch.ones(100, 100),))
                futs.append(fut)

            for fut in futs:
                fut.wait()
                self.assertEqual(fut.wait(), 0)

    def _test_multi_remote_call(self, fn, args_fn=lambda x: (), kwargs_fn=lambda x: {}):
        m = 10
        n = self.rank + 1
        dst_rank = n % self.world_size
        rrefs = []
        expected = []
        for i in range(m):
            n = n + i
            rrefs.append(
                rpc.remote(
                    "worker{}".format(dst_rank),
                    fn,
                    args=args_fn(n),
                    kwargs=kwargs_fn(n),
                )
            )
            expected.append(fn(*args_fn(n), **kwargs_fn(n)))

        for i in range(m):
            self.assertEqual(rrefs[i].to_here(), expected[i])

    @dist_init
    def test_multi_builtin_remote_ret(self):
        def args_fn(n):
            return (torch.ones(n, n), torch.ones(n, n))

        self._test_multi_remote_call(torch.add, args_fn=args_fn)

    @dist_init
    def test_py_udf_remote(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            "worker{}".format(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        self.assertEqual(rref.to_here(), my_function(n, n + 1, n + 2))

    @dist_init
    def test_multi_py_udf_remote(self):
        def kwargs_fn(n):
            return {"a": torch.ones(n, n), "b": torch.ones(n, n), "c": torch.ones(n, n)}

        self._test_multi_remote_call(my_function, kwargs_fn=kwargs_fn)

    @dist_init
    def test_py_rref_args(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), 2)
        )
        rref_b = rpc.remote(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), 1)
        )
        rref_c = rpc.remote(
            "worker{}".format(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), torch.ones(n, n) + 4)

    @dist_init
    def test_py_rref_args_user_share(self):
        n = self.rank + 1
        owner_rank = n % self.world_size
        user_rank = (n + 1) % self.world_size
        rref_a = rpc.remote(
            "worker{}".format(owner_rank), my_function, args=(torch.ones(n, n), 2, 0)
        )
        rref_b = rpc.remote(
            "worker{}".format(owner_rank), my_function, args=(torch.ones(n, n), 1, 0)
        )
        rref_c = rpc.remote(
            "worker{}".format(user_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), torch.ones(n, n) + 4)

    @dist_init
    def test_py_rpc_rref_args(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(
            "worker{}".format(dst_rank), my_function, args=(torch.ones(n, n), 2, 0)
        )
        rref_b = rpc.remote(
            "worker{}".format(dst_rank), my_function, args=(torch.ones(n, n), 1, 0)
        )

        c = rpc.rpc_sync(
            "worker{}".format(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )

        self.assertEqual(c, torch.ones(n, n) + 4)

    @dist_init
    def test_nested_remote(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size

        rref = rpc.remote(
            "worker{}".format(dst_rank1),
            nested_remote,
            args=("worker{}".format(dst_rank2),),
        )
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 3)

    @dist_init
    def test_nested_rref(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref_of_rrefs = rpc.remote(
            "worker{}".format(dst_rank1),
            nested_rref,
            args=("worker{}".format(dst_rank2),),
        )
        rrefs = rref_of_rrefs.to_here()
        self.assertEqual(len(rrefs), 2)
        self.assertEqual(rrefs[0].to_here(), torch.ones(2, 2) + 1)
        self.assertEqual(rrefs[1].to_here(), torch.ones(2, 2) + 2)

    @dist_init
    def test_nested_rref_stress(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        all_rrefs = []
        for _ in range(20):
            all_rrefs.append(
                rpc.remote(
                    "worker{}".format(dst_rank1),
                    nested_rref,
                    args=("worker{}".format(dst_rank2),),
                )
            )

        for i in range(20):
            rref_of_rrefs = all_rrefs[i]
            rrefs = rref_of_rrefs.to_here()
            self.assertEqual(len(rrefs), 2)
            self.assertEqual(rrefs[0].to_here(), torch.ones(2, 2) + 1)
            self.assertEqual(rrefs[1].to_here(), torch.ones(2, 2) + 2)

    @dist_init
    def test_multi_layer_nested_async_rpc(self):
        # This test will exit right away, but there will be a chain of async
        # RPCs. The termination algorithm should detect those messages properly.
        # Otherwise, some peer could exit early, leaving others to timeout
        # errors or connection closed errors.
        ttl = 20
        n = self.rank + 1
        dst_rank = n % self.world_size

        multi_layer_nested_async_rpc(dst_rank, self.world_size, ttl)

    @dist_init
    def test_remote_with_exception(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote("worker{}".format(dst_rank), raise_func)
        with self.assertRaisesRegex(Exception, "ValueError"):
            rref.to_here()

    @dist_init
    def test_rpc_return_rref(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref = rpc.rpc_sync(
            "worker{}".format(dst_rank1),
            rpc_return_rref,
            args=("worker{}".format(dst_rank2),),
        )
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_rref_forward_chain(self):
        ttl = 8
        n = self.rank + 1
        dst_rank = n % self.world_size

        rref = rpc.remote(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), 1)
        )

        ret_rref = rref_forward_chain(dst_rank, self.world_size, rref, ttl)

        for i in range(ttl):
            self.assertEqual(len(ret_rref), 1)
            ret_rref = ret_rref[0].to_here()

        ret = ret_rref
        self.assertEqual(ret, torch.add(torch.ones(n, n), 1))

    @dist_init
    def test_remote_same_worker(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), 2)
        )
        rref_b = rpc.remote(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), 1)
        )
        rref_c = rpc.remote(
            "worker{}".format(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), torch.ones(n, n) + 4)

    @unittest.skip("Test is flaky on ASAN, see https://github.com/pytorch/pytorch/issues/29117")
    @dist_init(setup_model_parallel=True)
    def test_call_method_on_rref(self):
        """
        Tests that it is possible to call an instance method on a remote objet
        by using rref.owner() as destination of the call.
        """
        vals = [10, 2, 5, 7]
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = "worker{}".format(dst_rank)

        # creates a remote object
        rref = rpc.remote(dst_worker, MyClass, args=(vals[0], ))

        # modifies state of the remote object
        rpc.rpc_sync(rref.owner(), _call_method_on_rref, args=(
            MyClass.increment_value, rref, vals[1]))
        rpc.rpc_async(rref.owner(), _call_method_on_rref, args=(
            MyClass.increment_value, rref, vals[2])).wait()
        rpc.remote(rref.owner(), _call_method_on_rref, args=(
            MyClass.increment_value, rref, vals[3])).to_here()

        # queries state of the remote object
        result = rpc.rpc_sync(dst_worker, _call_method_on_rref, args=(
            MyClass.get_value, rref))

        self.assertEqual(result, sum(vals))

    @dist_init
    def test_get_default_rpc_timeout(self):
        timeout = rpc.get_rpc_timeout()
        self.assertEqual(timeout, rpc.constants.DEFAULT_RPC_TIMEOUT)

    @dist_init(setup_model_parallel=False)
    def test_set_rpc_timeout(self):
        timeout = timedelta(seconds=1)
        rpc.init_model_parallel(
            self_name="worker{}".format(self.rank),
            backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
            init_method=self.init_method,
            self_rank=self.rank,
            worker_name_to_id=self.worker_name_to_id,
            rpc_timeout=timeout
        )
        set_timeout = rpc.get_rpc_timeout()
        self.assertEqual(timeout, set_timeout)
        rpc.join_rpc()


    def test_requires_process_group_agent_decorator(self):
        @requires_process_group_agent("test_func did not run")
        def test_func():
            return "expected result"

        if TEST_CONFIG.rpc_backend_name == "PROCESS_GROUP":
            self.assertEqual(test_func(), "expected result")

    def test_dist_init_decorator(self):
        @dist_init(setup_model_parallel=False)
        def test_func(self):
            return "expected result"

        self.assertEqual(test_func(self), "expected result")

        @dist_init
        def test_func(self):
            return "expected result"

        self.assertEqual(test_func(self), "expected result")
