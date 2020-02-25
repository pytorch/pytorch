import concurrent.futures
import sys
import time
import unittest
from collections import namedtuple
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info
from torch.distributed.rpc.api import _use_rpc_pickler
from torch.distributed.rpc.internal import PythonUDF, RPCExecMode, _internal_rpc_pickler
from torch.testing._internal.common_utils import IS_MACOS, load_tests
from torch.testing._internal.dist_utils import (
    dist_init,
    get_shutdown_error_regex,
    initialize_pg,
    wait_until_node_failure,
    wait_until_pending_users_flushed,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


def foo_add():
    return torch.add(torch.ones(1), torch.ones(1))


def requires_process_group_agent(message=""):
    def decorator(old_func):
        return unittest.skipUnless(
            dist_utils.TEST_CONFIG.rpc_backend_name == "PROCESS_GROUP", message
        )(old_func)

    return decorator


VALUE_FUTURE = concurrent.futures.Future()
DONE_FUTURE = concurrent.futures.Future()


class StubRpcAgent:
    def __init__(self, world_size):
        self.world_size = world_size

    def get_worker_infos(self):
        return {
            rpc.WorkerInfo(name="worker{}".format(rank), id=rank)
            for rank in range(self.world_size)
        }


def _stub_construct_rpc_backend_options_handler(**kwargs):
    return mock.Mock()  # RpcBackendOptions.


def _stub_init_rpc_backend_handler(store, name, rank, world_size, rpc_backend_options):
    return StubRpcAgent(world_size=world_size)


def set_value(value):
    VALUE_FUTURE.set_result(value)


def wait_for_value_future():
    return VALUE_FUTURE.result()


def set_and_check_done(value):
    VALUE_FUTURE.set_result(value)
    return DONE_FUTURE.result()


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


def get_rref_list(values):
    return [RRef(MyClass(a)) for a in values]


def add_rref_to_value(rref, value):
    return rref.to_here() + value


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


def my_sleep_func(seconds=1):
    time.sleep(seconds)


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


@torch.jit.script
def heavy_rpc_torchscript(tensor):
    for i in range(1, 100):
        tensor *= i
        tensor /= i + 1
    return 0


def raise_func():
    raise ValueError("Expected error")


global_rref = None


def set_global_rref(rref):
    global global_rref
    global_rref = rref


def clear_global_rref():
    global global_rref
    global_rref = None


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed rpc package does not support python2"
)
class RpcTest(RpcAgentTestFixture):
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
    def test_get_worker_infos(self):
        worker_infos = rpc.api._get_current_rpc_agent().get_worker_infos()

        worker_names = {worker_info.name for worker_info in worker_infos}
        expected_worker_names = {
            "worker{}".format(rank) for rank in range(self.world_size)
        }
        self.assertEqual(worker_names, expected_worker_names)

        worker_ids = {worker_info.id for worker_info in worker_infos}
        expected_worker_ids = set(range(self.world_size))
        self.assertEqual(worker_ids, expected_worker_ids)

    @dist_init
    def test_self_add(self):
        self_worker_info = rpc.get_worker_info()
        self_worker_name = "worker{}".format(self.rank)
        fut = rpc.rpc_async(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        ret = rpc.rpc_sync(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    @dist_init
    def test_self_py_udf_remote(self):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(torch.ones(2, 2), 1, 3))
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1 + 3)

    def _test_self_remote_rref_as_rpc_arg(self, dst):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(torch.ones(2, 2), 1, 3))
        fut = rpc.rpc_async(dst, add_rref_to_value, args=(rref, torch.ones(2, 2)))
        ret = rpc.rpc_sync(dst, add_rref_to_value, args=(rref, torch.ones(2, 2) + 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 3 + torch.ones(2, 2) + 1)
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1 + 3 + torch.ones(2, 2))

    @dist_init
    def test_self_remote_rref_as_rpc_arg(self):
        dst = "worker{}".format((self.rank + 1) % self.world_size)
        self._test_self_remote_rref_as_rpc_arg(dst)

    @dist_init
    def test_self_remote_rref_as_self_rpc_arg(self):
        self._test_self_remote_rref_as_rpc_arg(rpc.get_worker_info())

    def _test_self_remote_rref_as_remote_arg(self, dst):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(torch.ones(2, 2), 1, 3))
        ret_rref = rpc.remote(dst, add_rref_to_value, args=(rref, torch.ones(2, 2)))
        self.assertEqual(
            ret_rref.to_here(), torch.ones(2, 2) + 1 + 3 + torch.ones(2, 2)
        )

    @dist_init
    def test_self_remote_rref_as_remote_arg(self):
        dst = "worker{}".format((self.rank + 1) % self.world_size)
        self._test_self_remote_rref_as_remote_arg(dst)

    @dist_init
    def test_self_remote_rref_as_self_remote_arg(self):
        self._test_self_remote_rref_as_remote_arg(rpc.get_worker_info())

    @mock.patch.object(torch.distributed.autograd, "_init")
    @mock.patch.object(torch.distributed.rpc.api, "_set_and_start_rpc_agent")
    @dist_init(setup_rpc=False)
    def test_register_rpc_backend_and_set_and_start_rpc_backend(
        self, mock_rpc_agent, mock_dist_autograd_init
    ):
        backend_name = "stub_backend"

        backend = rpc.backend_registry.register_backend(
            backend_name,
            _stub_construct_rpc_backend_options_handler,
            _stub_init_rpc_backend_handler,
        )

        with self.assertRaisesRegex(
            RuntimeError, "^RPC backend .+: already registered$"
        ):
            backend = rpc.backend_registry.register_backend(
                backend_name,
                _stub_construct_rpc_backend_options_handler,
                _stub_init_rpc_backend_handler,
            )

        rpc.init_rpc(
            name="worker1",
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

    @requires_process_group_agent("PROCESS_GROUP rpc backend specific test, skip")
    @dist_init(setup_rpc=False)
    def test_duplicate_name(self):
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            store, _, _ = next(
                torch.distributed.rendezvous(
                    self.init_method, rank=self.rank, world_size=self.world_size
                )
            )
            rpc.api._init_rpc_backend(
                backend=self.rpc_backend,
                store=store,
                name="duplicate_name",
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_reinit(self):
        rpc.init_rpc(
            name="worker{}".format(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        initialize_pg(self.init_method, self.rank, self.world_size)
        # Wait for all init to complete.
        dist.barrier()

        with self.assertRaisesRegex(RuntimeError, "is already initialized"):
            rpc.init_rpc(
                name="worker{}".format(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_invalid_names(self):
        from torch.distributed.rpc import WorkerInfo

        worker_id = 0
        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            info = WorkerInfo("abc*", worker_id)

        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            info = WorkerInfo(" ", worker_id)

        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            info = WorkerInfo("", worker_id)

        # If the number in the message does not match, it is likely that the
        # value of MAX_NAME_LEN in RPC WorkerInfo has changed.
        with self.assertRaisesRegex(RuntimeError, "shorter than 128"):
            info = WorkerInfo("".join(["a" for i in range(500)]), worker_id)

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

    def _run_uneven_workload(self, num_repeat=30):
        # worker0 drives and waits for worker1 and worker2
        # throughout the test.
        if self.rank == 0:
            self.assertTrue(self.world_size >= 3)

            # Phase 1: Only worker1 has workload.
            dst = "worker1"
            futs = []
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
            futs = []
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, heavy_rpc, args=(torch.ones(100, 100),))
                futs.append(fut)

            for fut in futs:
                fut.wait()
                self.assertEqual(fut.wait(), 0)

    def test_wait_all_workers(self):
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        self._run_uneven_workload()

        # worker0 calls this at the end after waiting for RPC responses.
        # worker1/2 calls this immediately and has some works after it.
        # worker3 calls this immediately and has no more work.
        rpc.api._wait_all_workers()
        rpc.shutdown(graceful=False)

    def test_wait_all_workers_twice(self):
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        self._run_uneven_workload()

        # worker0 calls this at the end after waiting for RPC responses.
        # worker1/2 calls this immediately and has some works after it.
        # worker3 calls this immediately and has no more work.
        rpc.api._wait_all_workers()
        rpc.api._wait_all_workers()
        rpc.shutdown(graceful=False)

    @dist_init
    def test_graceful_shutdown_with_uneven_workload(self):
        """Test graceful termination."""
        self._run_uneven_workload()

    @dist_init(setup_rpc=False)
    def test_shutdown_followed_by_rpc(self):
        # Initialize RPC.
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)
        rpc.shutdown()

        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            rpc.rpc_sync(
                "worker{}".format(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )

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

    def _profiler_test_with_rpc(self, rpc_exec_mode, func, args):
        dst = (self.rank + 1) % self.world_size
        # only run profiler on rank 1.
        if self.rank == 1:
            with torch.autograd.profiler.profile() as prof:
                if rpc_exec_mode == RPCExecMode.SYNC:
                    rpc.rpc_sync("worker{}".format(dst), func, args=args)
                elif rpc_exec_mode == RPCExecMode.ASYNC:
                    fut = rpc.rpc_async("worker{}".format(dst), func, args=args)
                    fut.wait()
                else:
                    self.assertTrue(rpc_exec_mode == RPCExecMode.REMOTE)
                    rref = rpc.remote("worker{}".format(dst), func, args=args)
                    rref.to_here()
                    # We need to wait for the instance to be created on
                    # the owner, and get back a positive confirmation.
                    # Calling to_here does not ensure that we have finished
                    # processing the Owner's confirmation of this RRef. To do
                    # this, we wait until the current RRef context doesn't have
                    # any pending users, which indicates that the confirmation
                    # was processed on this worker.
                    wait_until_pending_users_flushed()

            events = prof.function_events
            rpc_event = [
                event for event in events if rpc_exec_mode.value in event.name
            ][0]
            # the sender, dest worker, function run, and type of RPC should all
            # be recorded.
            self_worker_name = "worker{}".format(self.rank)
            dst_worker_name = "worker{}".format(dst)
            self.assertTrue(self_worker_name in rpc_event.name)
            self.assertTrue(dst_worker_name in rpc_event.name)
            self.assertTrue(func.__name__ in rpc_event.name)
            self.assertTrue(rpc_exec_mode.value in rpc_event.name)
            self.assertEqual(rpc_event.count, 1)

    @dist_init
    def test_profiler_with_sync_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,))

    @dist_init
    def test_profiler_with_sync_rpc_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, torch.add, args=(torch.ones(1), torch.ones(1))
        )

    @dist_init
    def test_profiler_with_async_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,))

    @dist_init
    def test_profiler_with_async_rpc_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, torch.add, args=(torch.ones(1), torch.ones(1))
        )

    @dist_init
    def test_profiler_with_remote_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_sleep_func, args=(1,))

    @dist_init
    def test_profiler_with_remote_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, torch.add, args=(torch.ones(1), torch.ones(1))
        )

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
        with self.assertRaises(TypeError):
            ret = rpc.rpc_sync("worker{}".format(dst_rank), no_result, args=(10,))

    @dist_init
    def test_py_raise_in_user_func(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async("worker{}".format(dst_rank), raise_func)
        with self.assertRaises(ValueError):
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
            "Rank {} finished testing {} times in {} seconds.".format(
                self.rank, repeat, tok - tik
            )
        )

    @dist_init
    def test_stress_light_rpc(self):
        self._stress_test_rpc(light_rpc)

    @dist_init
    def test_stress_heavy_rpc(self):
        self._stress_test_rpc(heavy_rpc, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_stress_heavy_rpc_torchscript(self):
        self._stress_test_rpc(
            heavy_rpc_torchscript, repeat=20, args=(torch.ones(100, 100),)
        )

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
        # check ref to other workers
        rref = rpc.remote("worker{}".format(dst_rank), raise_func)
        with self.assertRaises(ValueError):
            rref.to_here()
        # check ref to itself
        rref = rpc.remote("worker{}".format(self.rank), no_result, args=(10,))
        with self.assertRaises(TypeError):
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
    def test_local_rref_no_fork(self):
        local_rref = RRef(35)
        self.assertEqual(local_rref.local_value(), 35)

    @dist_init
    def test_local_value_not_on_owner(self):
        # ensure that an error message is thrown if a user tries to call
        # local_value() on a non-owning node.
        next_rank = (self.rank + 1) % self.world_size
        rref = rpc.remote(
            "worker{}".format(next_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        with self.assertRaisesRegex(
            RuntimeError, "Call it on worker{}".format(next_rank)
        ):
            rref.local_value()

    @dist_init
    def test_return_local_rrefs(self):
        n = self.rank + 1
        dst_rank = n % self.world_size

        rref_list = rpc.rpc_sync(
            "worker{}".format(dst_rank), get_rref_list, args=([1, 2, 3],)
        )

        for rref in rref_list:
            rpc.rpc_sync(
                rref.owner(),
                _call_method_on_rref,
                args=(MyClass.increment_value, rref, 10),
            )

        rets = [
            rpc.rpc_sync(
                rref.owner(), _call_method_on_rref, args=(MyClass.get_value, rref)
            )
            for rref in rref_list
        ]

        self.assertEqual(rets, [11, 12, 13])

    @dist_init
    def test_owner_equality(self):
        a = RRef(40)
        b = RRef(50)

        other_rank = (self.rank + 1) % self.world_size
        other_a = rpc.remote(
            "worker{}".format(other_rank), torch.add, args=(torch.ones(1), 1)
        )
        other_b = rpc.remote(
            "worker{}".format(other_rank), torch.add, args=(torch.ones(1), 1)
        )
        other_a.to_here()  # to ensure clean termination
        other_b.to_here()

        self.assertNotEqual(a.owner(), 23)
        self.assertEqual(other_a.owner(), other_b.owner())
        self.assertNotEqual(a.owner(), other_a.owner())
        self.assertEqual(other_a.owner(), other_a.owner())
        self.assertEqual(other_a.owner(), other_b.owner())
        self.assertEqual(a.owner(), a.owner())
        self.assertEqual(a.owner(), b.owner())
        self.assertEqual(a.owner(), rpc.get_worker_info())
        x = dict()
        x[a.owner()] = a
        x[other_a.owner()] = other_a
        self.assertEqual(x[a.owner()], a)
        self.assertEqual(x[b.owner()], a)
        self.assertEqual(x[other_a.owner()], other_a)
        self.assertEqual(x[other_b.owner()], other_a)
        self.assertEqual(len(x), 2)

    @dist_init
    def test_pass_local_rrefs(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker = "worker{}".format(dst_rank)

        rref = RRef(40)
        self.assertEqual(
            rpc.rpc_sync(dst_worker, add_rref_to_value, args=(rref, 50)), 90
        )
        self.assertEqual(
            rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 50)).wait(), 90
        )
        self.assertEqual(
            rpc.remote(dst_worker, add_rref_to_value, args=(rref, 50)).to_here(), 90
        )

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

    @dist_init(setup_rpc=True)
    def test_call_method_on_rref(self):
        """
        Tests that it is possible to call an instance method on a remote objet
        by using rref.owner() as destination of the call.
        """
        vals = [10, 2, 5, 7]
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = "worker{}".format(dst_rank)

        # creates a remote object
        rref = rpc.remote(dst_worker, MyClass, args=(vals[0],))

        # modifies state of the remote object
        rpc.rpc_sync(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[1]),
        )
        rpc.rpc_async(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[2]),
        ).wait()
        rpc.remote(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[3]),
        ).to_here()

        # queries state of the remote object
        result = rpc.rpc_sync(
            dst_worker, _call_method_on_rref, args=(MyClass.get_value, rref)
        )

        self.assertEqual(result, sum(vals))

    def _test_rref_leak(self, ignore_leak):
        rpc.init_rpc(
            name="worker{}".format(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        initialize_pg(self.init_method, self.rank, self.world_size)
        # Wait for all init to complete.
        dist.barrier()

        rref = rpc.remote(
            "worker{}".format((self.rank + 1) % self.world_size),
            torch.add,
            args=(torch.ones(2, 2), 1),
        )

        import torch.distributed.rpc.api as api

        if ignore_leak:
            api._ignore_rref_leak = True
            rpc.shutdown(graceful=True)
        else:
            api._ignore_rref_leak = False
            with self.assertRaisesRegex(RuntimeError, "Leaking RRef"):
                rpc.shutdown(graceful=True)

    @dist_init(setup_rpc=False)
    def test_rref_leak(self):
        self._test_rref_leak(ignore_leak=False)

    @dist_init(setup_rpc=False)
    def test_ignore_rref_leak(self):
        self._test_rref_leak(ignore_leak=True)

    @dist_init
    def test_rref_str(self):
        rref1 = RRef(self.rank)
        id_class = "GloballyUniqueId"
        self.assertEqual(
            "OwnerRRef({}({}, 0))".format(id_class, self.rank), rref1.__str__()
        )

        dst_rank = (self.rank + 1) % self.world_size
        rref2 = rpc.remote(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        self.assertEqual(
            rref2.__str__(),
            "UserRRef(RRefId = {0}({1}, 1), ForkId = {0}({1}, 2))".format(
                id_class, self.rank
            ),
        )

    @dist_init
    def test_rref_context_debug_info(self):
        # This test checks local states that are modified by remote workers.
        # This means that we would need barrier before and after every check.
        # The barrier before the check makes sure that all previous states are
        # cleared globally, the barrier after ensures that no following states
        # change gets into the current check.
        initialize_pg(self.init_method, self.rank, self.world_size)

        # Check 1: local RRef does not update owners_ map or add a pending user.
        #################################################

        rref1 = RRef(self.rank)

        # don't need a barrier here as local RRef is handled by this thread
        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertIn("num_pending_users", info)
        # RRef on local value is not added to context until shared across RPC
        self.assertEqual(0, int(info["num_owner_rrefs"]))
        self.assertEqual(0, int(info["num_pending_users"]))
        # barrier after the check 1
        dist.barrier()

        # Check 2: Sharing RRef as an arg should update owners_ map
        ###########################################################

        dst_rank = (self.rank + 1) % self.world_size
        rpc.rpc_sync("worker{}".format(dst_rank), set_global_rref, args=(rref1,))

        # barrier before check 2
        dist.barrier()

        wait_until_pending_users_flushed()
        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertEqual(1, int(info["num_owner_rrefs"]))
        # no pending users since the fork is finished
        self.assertEqual(0, int(info["num_pending_users"]))
        # barrier after check 2
        dist.barrier()

        # clear states for check 2
        rpc.rpc_sync("worker{}".format(dst_rank), clear_global_rref)

        # Check 3: rpc.remote call should update owners_ map
        ####################################################
        rref2 = rpc.remote(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        rref3 = rpc.remote(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        rref2.to_here()
        rref3.to_here()

        # barrier before check 3
        dist.barrier()

        wait_until_pending_users_flushed()
        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertEqual(2, int(info["num_owner_rrefs"]))
        # no pending users since the fork is finished
        self.assertEqual(0, int(info["num_pending_users"]))

        # barrier after check 3
        dist.barrier()

    @dist_init
    def test_disable_gil_profiling(self):
        # test that rpc.enable_gil_profilig(false) will result in
        # GIL wait time not being recorded.

        # GIL profiling should be disabled by default.
        dst_rank = (self.rank + 1) % self.world_size
        rpc.rpc_sync(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertRaises(KeyError, lambda: info["agent.gil_average_wait_time_us"])
        rpc.enable_gil_profiling(True)
        rpc.rpc_sync(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertIn("agent.gil_average_wait_time_us", info)

    @dist_init
    @requires_process_group_agent("PROCESS_GROUP rpc backend specific test, skip")
    def test_process_group_debug_info(self):
        rpc.enable_gil_profiling(True)
        initialize_pg(self.init_method, self.rank, self.world_size)
        NUM_THREAD = self.rpc_backend_options.num_send_recv_threads

        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertIn("agent.num_pending_requests", info)
        self.assertIn("agent.thread_pool_size", info)
        self.assertIn("agent.num_idle_threads", info)
        self.assertIn("agent.gil_average_wait_time_us", info)
        self.assertEqual(int(info["agent.num_pending_requests"]), 0)
        self.assertEqual(int(info["agent.thread_pool_size"]), NUM_THREAD)
        self.assertEqual(int(info["agent.num_idle_threads"]), NUM_THREAD)
        # for the above check, add a barrier to ensure that another worker
        # cannot send a request before we check num_idle_threads, since we'd
        # use up an idle thread if we start processing that request.
        dist.barrier()
        dst_rank = (self.rank + 1) % self.world_size
        fut = rpc.rpc_async(
            "worker{}".format(dst_rank), set_and_check_done, args=(dst_rank,)
        )
        # blocks until the request arrives
        self.assertEqual(self.rank, VALUE_FUTURE.result())

        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertIn("agent.num_pending_requests", info)
        self.assertIn("agent.thread_pool_size", info)
        self.assertIn("agent.num_idle_threads", info)
        self.assertIn("agent.gil_average_wait_time_us", info)
        self.assertGreaterEqual(float(info["agent.gil_average_wait_time_us"]), 0)
        self.assertEqual(int(info["agent.num_pending_requests"]), 1)
        self.assertEqual(int(info["agent.thread_pool_size"]), NUM_THREAD)
        num_idle_threads = int(info["agent.num_idle_threads"])
        # as we cannot know for sure whether the send thread has returned, there
        # might be either 1 or 2 busy threads
        self.assertTrue(num_idle_threads in [NUM_THREAD - 1, NUM_THREAD - 2])

        # add a barrier to make sure the request is not finished before checking
        # num_pending_requests
        dist.barrier()

        DONE_FUTURE.set_result(self.rank)
        self.assertEqual(dst_rank, fut.wait())

        # add a barrier to make sure the dst_rank has finished processing the
        # request
        dist.barrier()

        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertIn("agent.num_pending_requests", info)
        self.assertIn("agent.thread_pool_size", info)
        self.assertIn("agent.num_idle_threads", info)
        self.assertEqual(int(info["agent.num_pending_requests"]), 0)
        self.assertEqual(int(info["agent.thread_pool_size"]), NUM_THREAD)

        for retry in range(3):
            # even if the future has completed, there is no guarantee that
            # the local send/recv threads would have finished. We try three
            # times. (NB: this might potentially be flaky. If flakiness does
            # occur, then we have to relax the assert.)
            info = rpc.api._get_current_rpc_agent().get_debug_info()
            if int(info["agent.num_idle_threads"]) == NUM_THREAD:
                break
            time.sleep(0.1)
        self.assertEqual(int(info["agent.num_idle_threads"]), NUM_THREAD)

        # add a barrier to make sure SHUTDOWN message is not sent
        dist.barrier()

    @dist_init(setup_rpc=False)
    @requires_process_group_agent("PROCESS_GROUP rpc backend specific test, skip")
    def test_local_shutdown(self):
        # test that we can start RPC and then immediately locally shutdown
        # without sending any messages.
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        # pass in graceful=False to ensure that we don't wait for other workers.
        rpc.shutdown(graceful=False)

    @dist_init
    @unittest.skip("Test is flaky. see https://github.com/pytorch/pytorch/issues/31846")
    def test_debug_info(self):
        # only test keys in this test case. Values should be covered by
        # individual module debug info tests
        import torch.distributed.autograd as dist_autograd

        info = _get_debug_info()
        rref_info = _rref_context_get_debug_info()
        agent_info = rpc.api._get_current_rpc_agent().get_debug_info()
        autograd_info = dist_autograd._get_debug_info()
        common_keys = rref_info.keys() & agent_info.keys() & autograd_info.keys()
        self.assertEqual(0, len(common_keys))
        expected = {}
        expected.update(rref_info)
        expected.update(agent_info)
        expected.update(autograd_info)
        self.assertEqual(expected.keys(), info.keys())

    @dist_init(setup_rpc=False)
    @unittest.skipIf(
        IS_MACOS,
        "Test is flaky on MacOS, see https://github.com/pytorch/pytorch/issues/32019",
    )
    def test_handle_send_exceptions(self):
        # test that if a callee node has gone down, we raise an appropriate
        # exception instead of just crashing.
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        # This barrier is needed to ensure that some workers do not exit before
        # others have been brought up, for non ProcessGroupAgent backends.
        initialize_pg(self.init_method, self.rank, self.world_size)
        dist.barrier()

        if self.rank == 1:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker = "worker{}".format(dst_rank)
            # allow destination worker to exit without joining
            wait_until_node_failure(dst_rank)
            fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.ones(1), 3))
            # Shutdown sequence is not very well defined and as a result
            # we can see any of these error messages.
            error_str = (
                "Encountered exception in ProcessGroupAgent::enqueueSend"
                if self.rpc_backend == rpc.backend_registry.BackendType.PROCESS_GROUP
                else get_shutdown_error_regex()
            )
            with self.assertRaisesRegex(RuntimeError, error_str):
                fut.wait()
        # exit all workers non-gracefully.
        rpc.shutdown(graceful=False)

    @dist_init(setup_rpc=False)
    @requires_process_group_agent("PROCESS_GROUP rpc backend specific test, skip")
    def test_local_shutdown_with_rpc(self):
        # test that we can start RPC, send RPCs, and then run local shutdown.
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        n = self.rank + 1
        dst_rank = n % self.world_size
        rpc.rpc_sync(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # A barrier is needed to ensure that all RPCs are processed.
        # Otherwise, some RPCs can timeout since the receiving end
        # has terminated.
        initialize_pg(self.init_method, self.rank, self.world_size)
        dist.barrier()
        # pass in graceful=False to ensure that we don't wait for other workers.
        rpc.shutdown(graceful=False)

    @dist_init(setup_rpc=False)
    def test_get_rpc_timeout(self):
        timeout = timedelta(seconds=1)

        # A new `RpcBackendOptions` is constructed
        # when accessing `self.rpc_backend_options`.
        rpc_backend_options = self.rpc_backend_options
        rpc_backend_options.rpc_timeout = timeout

        rpc.init_rpc(
            name="worker{}".format(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )
        set_timeout = rpc.get_rpc_timeout()
        self.assertEqual(timeout, set_timeout)
        rpc.shutdown()

    @dist_init
    @requires_process_group_agent("PROCESS_GROUP rpc backend specific test, skip")
    def test_rpc_timeouts(self):
        dst_rank = (self.rank + 1) % self.world_size
        rpc._set_rpc_timeout(timedelta(milliseconds=1))
        # futures should time out and be marked with an exception indicating it as such.
        futs = [
            rpc.rpc_async("worker{}".format(dst_rank), my_sleep_func, args=())
            for _ in range(10)
        ]
        for fut in futs:
            with self.assertRaisesRegex(RuntimeError, "RPC ran for more than"):
                fut.wait()

        # ensure that if a new timeout is set old futures don't time out but new ones do.
        rpc._set_rpc_timeout(timedelta(seconds=200))
        # create a longstanding RPC.
        fut1 = rpc.rpc_async("worker{}".format(dst_rank), my_sleep_func, args=(1,))
        # now, set a short timeout.
        rpc._set_rpc_timeout(timedelta(milliseconds=1))
        # f2 should time out, f should not.
        fut2 = rpc.rpc_async("worker{}".format(dst_rank), my_sleep_func, args=(1,))
        with self.assertRaises(RuntimeError):
            fut2.wait()
        fut1.wait()

        # future should run to completion if the timeout is zero.
        rpc._set_rpc_timeout(timedelta(seconds=0))
        rpc.rpc_async("worker{}".format(dst_rank), my_sleep_func, args=()).wait()

        # reset to default timeout so shutdown messages can process cleanly.
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT)

    def test_requires_process_group_agent_decorator(self):
        @requires_process_group_agent("test_func did not run")
        def test_func():
            return "expected result"

        if dist_utils.TEST_CONFIG.rpc_backend_name == "PROCESS_GROUP":
            self.assertEqual(test_func(), "expected result")

    def test_dist_init_decorator(self):
        @dist_init(setup_rpc=False)
        def test_func(self):
            return "expected result"

        self.assertEqual(test_func(self), "expected result")

        @dist_init
        def test_func(self):
            return "expected result"

        self.assertEqual(test_func(self), "expected result")

    def test_use_rpc_pickler(self):
        class TestPickler:
            pass

        test_pickler = TestPickler()
        with _use_rpc_pickler(test_pickler):
            self.assertTrue(torch.distributed.rpc.api._default_pickler is test_pickler)
        self.assertTrue(
            torch.distributed.rpc.api._default_pickler is _internal_rpc_pickler
        )

    @dist_init
    def test_function_not_on_callee(self):
        # test that if a function does not exist on a callee, we don't crash,
        # instead we get an AttributeError indicating that the func does not exist.
        this_module = sys.modules[__name__]
        caller_worker = "worker0"
        callee_worker = "worker1"

        if self.rank == 1:
            # Use delattr to remove the binding of a func on this nodes
            delattr(this_module, "foo_add")
            # notify remote end that we have removed it.
            rpc.rpc_sync(caller_worker, set_value, args=(self.rank,))

        if self.rank == 0:
            # func exists on caller, but not callee.
            # wait for remote end to remove the binding of foo_add func.
            wait_for_value_future()
            # Ensure that we have the attribute on this module. Otherwise, the test could fail due to a caller-side pickling error.
            self.assertTrue(hasattr(this_module, "foo_add"))
            with self.assertRaisesRegex(
                AttributeError, "RPC pickler does not serialize"
            ):
                rpc.rpc_sync(callee_worker, foo_add, args=())
