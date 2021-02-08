import concurrent.futures
import contextlib
import json
import logging
import os
import sys
from threading import Lock
import time
import unittest
from collections import namedtuple
from functools import partial
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs, _use_rpc_pickler, _thread_local_var, _wait_all
from torch.distributed.rpc.internal import (
    PythonUDF,
    RPCExecMode,
    _internal_rpc_pickler,
    _build_rpc_profiling_key,
)
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
    captured_output,
)
from torch.testing._internal.common_utils import IS_MACOS, load_tests
from torch.testing._internal.dist_utils import (
    dist_init,
    get_function_event,
    initialize_pg,
    wait_until_node_failure,
    wait_until_pending_futures_and_users_flushed,
    wait_until_owners_and_forks_on_rank,
    worker_name,
    single_threaded_process_group_agent,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.testing._internal.common_utils import TemporaryFileName


def foo_add():
    return torch.add(torch.ones(1), torch.ones(1))

def udf_with_torch_ops(device=-1, use_record_function=False):
    device_ctx = contextlib.suppress() if device == -1 else torch.cuda.device(device)
    record_function_ctx = (
        torch.autograd.profiler.record_function("##forward##")
        if use_record_function
        else contextlib.suppress()
    )
    with device_ctx, record_function_ctx:
        t1, t2 = torch.ones(1), torch.ones(1)
        t = torch.add(t1, t2)
        t = torch.mul(t, t)
        t = t.relu()
        t = t.sigmoid()

# Events (operator invocations) that are expected to be ran as part of the above
# function.
EXPECTED_REMOTE_EVENTS = [
    "aten::ones",
    "aten::ones",
    "aten::add",
    "aten::mul",
    "aten::relu",
    "aten::threshold",
    "aten::sigmoid",
]

# Remote operations are prefixed with the following string for RPC profiling.
REMOTE_OP_STR = "#remote_op: "


VALUE_FUTURE = concurrent.futures.Future()
DONE_FUTURE = concurrent.futures.Future()

FIFTY_MIL_CYCLES = 50000000

class StubRpcAgent:
    def __init__(self, world_size):
        self.world_size = world_size

    def get_worker_infos(self):
        return {
            rpc.WorkerInfo(name=worker_name(rank), id=rank)
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


class SlowPickleClass:
    def __init__(self, t):
        self.t = t

    def __getstate__(self):
        time.sleep(self.t)
        return (self.t, )

    def __setstate__(self, obj):
        self.t = obj[0]
        time.sleep(self.t)


class MyClass:
    def __init__(self, a, delay=False):
        self.a = a
        # delay initialization to simulate errors if specified
        if delay:
            time.sleep(2)

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

    def my_slow_method(self, my_tensor_arg):
        time.sleep(5)
        return torch.add(self.a, my_tensor_arg)


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

def non_cont_test(t_view, t_cont):
    if t_view.is_contiguous():
        raise Exception('t_view is contiguous!')
    if not t_cont.is_contiguous():
        raise Exception('t_cont is not contiguous!')
    if not torch.equal(t_view, t_cont):
        raise Exception('t_view is not equal to t_cont!')
    return t_view

def my_function(a, b, c):
    return a + b + c


def my_tensor_function(a, b):
    return a + b


def my_sleep_func(seconds=1):
    time.sleep(seconds)
    return torch.mul(torch.tensor(1), torch.tensor(1))


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


def delayed_add(a, b, seconds=0.05):
    time.sleep(seconds)
    return a + b


def no_result():
    print("do nothing")

def raise_or_inc(value):
    if value.numel() == 2:
        raise ValueError("Expected error")
    return value + 1

def nested_rpc(dst):
    return rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))


def multi_layer_nested_async_rpc(dst, world_size, ttl):
    # this method returns immediately without blocking the callee, but will
    # generate additional requests.
    if ttl > 0:
        current_dst = worker_name(dst)
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
        current_dst = worker_name(dst)
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


@torch.jit.script
def my_script_func(tensor):
    return torch.add(tensor, tensor)


expected_err = "Expected error"
def raise_func():
    raise ValueError(expected_err)

expected_err_escape = "\nFirst line of error \n next line of error \n last line of error"
def raise_func_escape():
    raise ValueError(expected_err_escape)


global_rref = None


def set_global_rref(rref):
    global global_rref
    global_rref = rref


def clear_global_rref():
    global global_rref
    global_rref = None


def check_rref_confirmed(rref):
    return rref.confirmed_by_owner()


def get_rref_debug_info():
    return _rref_context_get_debug_info()


def add_use_future_cb(to, x, y, z):
    out = concurrent.futures.Future()

    def callback(fut):
        out.set_result(fut.wait() + z)

    fut = rpc.rpc_async(to, torch.add, args=(x, y))
    fut.then(callback)
    return out.result()


def get_events_from_profile(profile_rref):
    return profile_rref.local_value().process_global_function_events


def add_use_future_set_result(to, x, y, z):
    out = torch.futures.Future()
    fut = rpc.rpc_async(to, torch.add, args=(x, y))
    fut.then(lambda fut : out.set_result(fut.wait() + z))
    return out.wait()


def add_use_future_nested_cb(to, x, y, z):
    out = torch.futures.Future()

    def callback(fut1):
        fut2 = rpc.rpc_async(to, torch.add, args=(fut1.wait(), z))
        fut2.then(lambda fut2 : out.set_result(fut2.wait()))

    fut1 = rpc.rpc_async(to, torch.add, args=(x, y))
    fut1.then(callback)
    return out.wait()


def fail_on_fut(fut):
    pass


@rpc.functions.async_execution
def async_raise_func():
    raise RuntimeError("Expected error")


@rpc.functions.async_execution
def async_wrong_type():
    return torch.zeros(2, 2)


@rpc.functions.async_execution
def async_add(to, x, y):
    return rpc.rpc_async(to, torch.add, args=(x, y))


def slow_add(x, y, device="cpu"):
    time.sleep(1)
    x = x.to(device)
    y = y.to(device)
    return torch.add(x, y).cpu()


@rpc.functions.async_execution
def slow_async_add(to, x, y, device="cpu"):
    return rpc.rpc_async(to, slow_add, args=(x, y, device))


@rpc.functions.async_execution
def async_add_with_future_ctor(to, x, y, z):
    fut = torch.futures.Future()
    rpc.rpc_async(to, torch.add, args=(x, y)).then(
        lambda fut1: fut.set_result(fut1.wait() + z)
    )
    return fut


@rpc.functions.async_execution
def async_add_chained(to, x, y, z):
    return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        lambda fut: fut.wait() + z
    )


@rpc.functions.async_execution
def async_add_chained_multi(to, x, num, step):
    fut = rpc.rpc_async(to, torch.add, args=(x, 0))
    for _ in range(num):
        fut = fut.then(lambda fut: fut.wait() + step)
    return fut


@rpc.functions.async_execution
def async_add_nested(to, x, y, z):
    return rpc.rpc_async(to, async_add, args=(to, x, y)).then(
        lambda fut: fut.wait() + z
    )


@rpc.functions.async_execution
def async_add_multi_fanout(to, x, num, step):
    futs = []
    for i in range(num):
        if i == 0:
            futs.append(rpc.rpc_async(to, torch.add, args=(x, step)))
        else:
            futs.append(rpc.rpc_async(to, torch.add, args=(0, step)))

    # TODO: use torch.futures.collect_all
    lock = Lock()
    state = {"cnt": 0, "ret": torch.zeros_like(x)}
    ret_future = torch.futures.Future()

    def inc_and_set(fut):
        with lock:
            state["cnt"] += 1
            state["ret"] += fut.wait()
            if state["cnt"] >= len(futs):
                ret_future.set_result(state["ret"])

    for fut in futs:
        fut.then(inc_and_set)

    return ret_future


class AsyncExecutionClass:

    @staticmethod
    @rpc.functions.async_execution
    def static_async_add(to, x, y, z):
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
        )

    @classmethod
    @rpc.functions.async_execution
    def class_async_add(cls, to, x, y, z):
        ret_fut = torch.futures.Future()
        rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: ret_fut.set_result(fut.wait() + z)
        )
        return ret_fut

    @rpc.functions.async_execution
    def bound_async_add(self, to, x, y, z):
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
        )


def return_future():
    return torch.futures.Future()


class FooBackendOptions(rpc.RpcBackendOptions):
    def __init__(self, init_method):
        # Must call the __init__ of the superclass (and do so directly,
        # without using super()) because... pybind.
        rpc.RpcBackendOptions.__init__(self)
        self.init_method = init_method


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class RpcTest(RpcAgentTestFixture):
    @dist_init
    def test_worker_id(self):
        n = self.rank + 1
        peer_rank = n % self.world_size
        self_worker_info = rpc.get_worker_info()
        peer_worker_info = rpc.get_worker_info(worker_name(peer_rank))

        self.assertEqual(self_worker_info.name, worker_name(self.rank))
        self.assertEqual(peer_worker_info.name, worker_name(peer_rank))

        with self.assertRaisesRegex(RuntimeError, "Unknown destination worker"):
            unknown_worker_id = rpc.get_worker_info("WorkerUnknown")

    @dist_init
    def test_get_worker_infos(self):
        worker_infos = rpc.api._get_current_rpc_agent().get_worker_infos()

        worker_names = {worker_info.name for worker_info in worker_infos}
        expected_worker_names = {
            worker_name(rank) for rank in range(self.world_size)
        }
        self.assertEqual(worker_names, expected_worker_names)

        worker_ids = {worker_info.id for worker_info in worker_infos}
        expected_worker_ids = set(range(self.world_size))
        self.assertEqual(worker_ids, expected_worker_ids)

    @dist_init
    def test_self_add(self):
        self_worker_info = rpc.get_worker_info()
        self_worker_name = worker_name(self.rank)
        fut = rpc.rpc_async(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        ret = rpc.rpc_sync(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    @dist_init
    def test_send_to_rank(self):
        dst_rank = (self.rank + 1) % self.world_size

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
            self.assertEqual(ret, torch.ones(2, 2) + 1)

        # Test invalid ranks
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(RuntimeError):
                self._run_func_in_mode(self.world_size + 1, torch.add, exec_mode, args=(torch.ones(2, 2), 1))

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(RuntimeError):
                self._run_func_in_mode(-1, torch.add, exec_mode, args=(torch.ones(2, 2), 1))

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(ValueError):
                self._run_func_in_mode(dst_rank + 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1))

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(ValueError):
                self._run_func_in_mode(dst_rank - 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1))

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
        dst = worker_name((self.rank + 1) % self.world_size)
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
        dst = worker_name((self.rank + 1) % self.world_size)
        self._test_self_remote_rref_as_remote_arg(dst)

    @dist_init
    def test_rref_proxy_non_exist(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))
        msg = "has no attribute \'non_exist\'"
        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_sync().non_exist()

        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_async().non_exist()

        with self.assertRaisesRegex(AttributeError, msg):
            rref.remote().non_exist()

    def _test_rref_proxy_tensor(self, dst):
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))

        expected = torch.ones(2, 2) + 1 + 3
        self.assertEqual(expected.size(), rref.rpc_sync().size())
        self.assertEqual(expected + 1, rref.rpc_async().add(1).wait())
        self.assertEqual(expected.view(1, 4), rref.remote().view(1, 4).to_here())

    @dist_init
    def test_rref_proxy_tensor(self):
        self._test_rref_proxy_tensor(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_tensor_self(self):
        self._test_rref_proxy_tensor(rpc.get_worker_info())

    @dist_init
    def test_rref_proxy_reuse(self):
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            my_function,
            args=(torch.ones(2, 2), 1, 3)
        )
        expected = torch.ones(2, 2) + 1 + 3

        proxy_rpc_sync = rref.rpc_sync()
        proxy_rpc_async = rref.rpc_async()
        proxy_remote = rref.remote()

        self.assertEqual(expected.size(), proxy_rpc_sync.size())
        self.assertEqual(expected + 1, proxy_rpc_sync.add(1))
        self.assertEqual(expected.view(1, 4), proxy_rpc_sync.view(1, 4))

        self.assertEqual(expected.size(), proxy_rpc_async.size().wait())
        self.assertEqual(expected + 3, proxy_rpc_async.add(3).wait())
        self.assertEqual(expected.view(4, 1), proxy_rpc_async.view(4, 1).wait())

        self.assertEqual(expected.size(), proxy_remote.size().to_here())
        self.assertEqual(expected + 5, proxy_remote.add(5).to_here())
        self.assertEqual(expected.view(-1), proxy_remote.view(-1).to_here())

    def _test_rref_proxy_class(self, dst):
        rref = rpc.remote(dst, MyClass, args=(7,))
        expected = MyClass(7)
        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())

        expected.increment_value(3)
        self.assertEqual(None, rref.rpc_sync().increment_value(1))
        self.assertEqual(None, rref.rpc_async().increment_value(1).wait())
        self.assertEqual(None, rref.remote().increment_value(1).to_here())

        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())

        self.assertEqual(
            expected.my_instance_method(2),
            rref.rpc_sync().my_instance_method(2)
        )
        self.assertEqual(
            expected.my_instance_method(3),
            rref.rpc_async().my_instance_method(3).wait()
        )
        self.assertEqual(
            expected.my_instance_method(4),
            rref.remote().my_instance_method(4).to_here()
        )

        self.assertEqual(
            expected.my_static_method(9),
            rref.rpc_sync().my_static_method(9)
        )
        self.assertEqual(
            expected.my_static_method(10),
            rref.rpc_async().my_static_method(10).wait()
        )
        self.assertEqual(
            expected.my_static_method(11),
            rref.remote().my_static_method(11).to_here()
        )

        self.assertEqual(
            expected.my_class_method(2, torch.zeros(2, 2)),
            rref.rpc_sync().my_class_method(2, torch.zeros(2, 2))
        )
        self.assertEqual(
            expected.my_class_method(2, torch.ones(3, 3)),
            rref.rpc_async().my_class_method(2, torch.ones(3, 3)).wait()
        )
        self.assertEqual(
            expected.my_class_method(2, torch.ones(4, 4)),
            rref.remote().my_class_method(2, torch.ones(4, 4)).to_here()
        )

    @dist_init
    def test_rref_proxy_class(self):
        self._test_rref_proxy_class(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_class_self(self):
        self._test_rref_proxy_class(rpc.get_worker_info())

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

    @dist_init(setup_rpc=False)
    def test_duplicate_name(self):
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            store, _, _ = next(
                torch.distributed.rendezvous(
                    self.init_method, rank=self.rank, world_size=self.world_size
                )
            )
            rpc._init_rpc_backend(
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
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # Wait for all init to complete.
        dist.barrier()

        # TODO: with TCP init, rank 0 raises Address already in use because
        # rank 0 is the start daemon and the store is created before checking if
        # RPC is already initialized in init_rpc.
        if os.environ.get("RPC_INIT_WITH_TCP", None) == "1" and self.rank == 0:
            expected_reinit_err = "Address already in use"
        else:
            expected_reinit_err = "is already initialized"

        with self.assertRaisesRegex(RuntimeError, expected_reinit_err):
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        rpc.shutdown()

    def test_world_size_one(self):
        if self.rank == 0:
            rpc.init_rpc(
                name="me",
                backend=self.rpc_backend,
                rank=0,
                world_size=1,
                rpc_backend_options=self.rpc_backend_options,
            )

            expect = torch.ones(2, 2) * 2
            result = rpc.rpc_sync(
                "me",
                my_tensor_function,
                args=(torch.ones(2, 2), torch.ones(2, 2))
            )
            self.assertEqual(expect, result)

            expect = torch.ones(3, 3) * 2
            result = rpc.rpc_async(
                "me",
                my_tensor_function,
                args=(torch.ones(3, 3), torch.ones(3, 3))
            ).wait()
            self.assertEqual(expect, result)

            expect = torch.ones(4, 4) * 2
            result = rpc.remote(
                "me",
                my_tensor_function,
                args=(torch.ones(4, 4), torch.ones(4, 4))
            ).to_here()
            self.assertEqual(expect, result)

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
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_add_with_id(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        workder_info = rpc.get_worker_info(worker_name(dst_rank))

        ret = rpc.rpc_sync(
            workder_info, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_scalar_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), n)
        )
        self.assertEqual(ret, (torch.ones(n, n) + n))

    @dist_init
    def test_async_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(
            worker_name(dst_rank),
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
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @dist_init
    def test_multi_rpc(self):
        dst_rank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            ret = rpc.rpc_sync(
                worker_name(dst_rank),
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

            for fut in torch.futures.collect_all(futs).wait():
                self.assertEqual(fut.wait(), 0)

            # Phase 2: Only worker2 has workload.
            # If join is not correctly implemented,
            # worker2 should be closed by now.
            dst = "worker2"
            futs = []
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, heavy_rpc, args=(torch.ones(100, 100),))
                futs.append(fut)

            for val in torch.futures.wait_all(futs):
                self.assertEqual(val, 0)

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
    def test_all_gather(self):
        info = rpc.get_worker_info()
        results = rpc.api._all_gather(info.id)
        expected = {}
        for info in rpc._get_current_rpc_agent().get_worker_infos():
            expected[info.name] = info.id

        self.assertEqual(expected, results)

    @dist_init
    def test_all_gather_timeout(self):
        rpc._set_rpc_timeout(0.1)

        if self.rank == 0:
            with self.assertRaisesRegex(
                RuntimeError,
                "timed out in _all_gather after 0\\.10 seconds"
            ):
                rpc.api._all_gather(SlowPickleClass(0.5))
        else:
            expected_error = self.get_timeout_error_regex()
            with self.assertRaisesRegex(RuntimeError, expected_error):
                rpc.api._all_gather(SlowPickleClass(0.5))

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
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)
        rpc.shutdown()

        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            rpc.rpc_sync(
                worker_name(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )

    @dist_init
    def test_expected_src(self):
        dst_rank = (self.rank + 1) % self.world_size
        expected_src_rank = (self.rank - 1) % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), set_value, args=(self.rank,))
        value = VALUE_FUTURE.result()
        self.assertEqual(value, expected_src_rank)

    @dist_init
    def test_py_built_in(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), min, args=(n, n + 1, n + 2))
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @dist_init
    def test_py_user_defined(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    def test_build_rpc_profiling_key(self):
        # Tests that the name that shows up as an Event in profiling RPCs has all
        # the necessary information.
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            rpc_profiling_key = _build_rpc_profiling_key(
                exec_mode, "foo", "worker0", "worker1"
            )
            self.assertIn(exec_mode.value, rpc_profiling_key)
            self.assertIn("foo", rpc_profiling_key)
            self.assertIn("worker0", rpc_profiling_key)
            self.assertIn("worker1", rpc_profiling_key)

    def check_profiling_info(self, self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode):
        self.assertTrue(self_worker_name in rpc_event.name)
        self.assertTrue(dst_worker_name in rpc_event.name)
        if isinstance(func, torch.jit.ScriptFunction):
            self.assertTrue(torch._jit_internal._qualified_name(func) in rpc_event.name)
        else:
            self.assertTrue(func.__name__ in rpc_event.name)
        self.assertTrue(rpc_exec_mode.value in rpc_event.name)
        self.assertEqual(rpc_event.count, 1)

    @dist_init
    def test_profiler_rpc_record_shapes(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        t1, t2 = torch.ones(100), torch.ones(100)
        with torch.autograd.profiler.profile(record_shapes=True) as prof:
            rpc.rpc_sync(dst_worker, torch.add, args=(t1, t2))

        function_events = prof.function_events
        remote_events = [event for event in function_events if event.is_remote]
        remote_add_event = [
            event for event in remote_events if "aten::add" in event.name
        ][0]
        remote_add_input_shapes = remote_add_event.input_shapes
        # Run profiler on equivalent local op and validate shapes are the same.
        with torch.autograd.profiler.profile(record_shapes=True) as prof:
            torch.add(t1, t2)

        local_function_events = prof.function_events
        local_add_event = [
            event for event in local_function_events if "aten::add" in event.name
        ][0]
        local_add_input_shapes = local_add_event.input_shapes
        self.assertEqual(remote_add_input_shapes, local_add_input_shapes)

    @dist_init
    def test_profiler_rpc_memory(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        with torch.autograd.profiler.profile(profile_memory=True) as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            res = fut.wait()

        function_events = p.function_events
        event_cpu_mem_usages = set(event.cpu_memory_usage for event in function_events)
        # if cpu_memory_usage was not propagated over the wire, this set would
        # only contain 0 (indicates no memory being profiled)
        self.assertNotEqual({0}, event_cpu_mem_usages)
        # No memory profiled if profile_memory=False
        with torch.autograd.profiler.profile(profile_memory=False) as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            res = fut.wait()

        function_events = p.function_events
        event_cpu_mem_usages = set(event.cpu_memory_usage for event in function_events)
        self.assertEqual({0}, event_cpu_mem_usages)

    @skip_if_lt_x_gpu(2)
    @dist_init
    def test_profiler_remote_cuda(self):
        if self.rank != 1:
            return

        dst_cuda_0 = (self.rank + 1) % self.world_size
        dst_cuda_1 = (self.rank + 2) % self.world_size
        dst_worker_cuda_0 = worker_name(dst_cuda_0)
        dst_worker_cuda_1 = worker_name(dst_cuda_1)

        with torch.autograd.profiler.profile(use_cuda=True) as p:
            fut1 = rpc.rpc_async(dst_worker_cuda_0, udf_with_torch_ops, args=(0, ))
            fut2 = rpc.rpc_async(dst_worker_cuda_1, udf_with_torch_ops, args=(1, ))
            fut1.wait()
            fut2.wait()

        def get_name(event):
            return event.name[event.name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR):]

        function_events = p.function_events
        for event in function_events:
            if event.is_async:
                self.assertEqual(0, event.cuda_time_total)
                self.assertEqual([], event.kernels)
                self.assertEqual(0, event.cuda_time)
            else:
                if event.node_id == 1:
                    continue
                self.assertTrue(event.node_id in [dst_cuda_0, dst_cuda_1])
                if get_name(event) in EXPECTED_REMOTE_EVENTS:
                    self.assertGreater(event.cuda_time_total, 0)
                    self.assertEqual(1, len(event.kernels))
                    kernel = event.kernels[0]
                    if event.node_id == dst_cuda_0:
                        self.assertEqual(kernel.device, 0)
                    if event.node_id == dst_cuda_1:
                        self.assertEqual(kernel.device, 1)
                    self.assertGreater(event.cuda_time, 0)

        # Validate that EXPECTED_REMOTE_EVENTS is a subset of remotely profiled
        # events.
        remote_events = [event for event in function_events if event.is_remote]
        remote_event_names = [get_name(event) for event in remote_events if get_name(event) in EXPECTED_REMOTE_EVENTS]
        self.assertEqual(set(remote_event_names), set(EXPECTED_REMOTE_EVENTS))

    @dist_init
    def test_profiler_export_trace(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        with torch.autograd.profiler.profile() as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            res = fut.wait()

        events = p.function_events
        with TemporaryFileName() as fname:
            path = fname
            p.export_chrome_trace(path)
            with open(path) as f:
                trace = json.load(f)
                event_names = [event['name'] for event in trace]
                for expected_event_name in EXPECTED_REMOTE_EVENTS + [RPCExecMode.ASYNC.value]:
                    event_exists = any([expected_event_name in event_name for event_name in event_names])
                    self.assertTrue(event_exists)

    @dist_init
    def test_profiler_rpc_key_names(self):
        # tests that remote events are properly prefixed with the RPC profiling key.
        if self.rank != 1:
            return

        # Spawn multiple threads that send RPCs to ensure keys are correctly
        # prefixied when there are multiple RPCs being created/in flight at the
        # same time.
        dst_ranks = [rank for rank in range(0, self.world_size) if rank != self.rank]

        def rpc_with_profiling(dst_worker):
            with torch.autograd.profiler.profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                fut.wait()

            events = prof.function_events
            remote_event_names = {
                event.name: event for event in events if event.is_remote
            }
            rpc_profiling_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC,
                udf_with_torch_ops.__qualname__,
                worker_name(self.rank),
                dst_worker,
            )

            remote_event_name_set = set(EXPECTED_REMOTE_EVENTS)
            for name, event in remote_event_names.items():
                # Ensure that we have the expected key as part of the remote
                # event.
                self.assertTrue(name.startswith(rpc_profiling_key))
                self.assertTrue(event.is_remote)
                self.assertTrue(event.node_id == rpc.get_worker_info(dst_worker).id)
                # Ensure that the remote event name also contains the operator.
                operator_name_substr = name[len(rpc_profiling_key) :]
                # Note: we don't assert that every remote event needs to be
                # in the above set, the set is just a representative set of
                # what we expect to see. The profiler can change and add more
                # events, but we should always expect to see this representative
                # set.
                matching_event = {
                    remote_event_name
                    for remote_event_name in remote_event_name_set
                    if remote_event_name in operator_name_substr
                }
                remote_event_name_set -= matching_event

            # The set should be empty, otherwise its contained elements did
            # not show up in the remote profiler output.
            self.assertTrue(
                remote_event_name_set == set(),
                f"Expected {remote_event_name_set} to be included in remote profiler output.",
            )

        for dst in dst_ranks:
            dst_worker = worker_name(dst)
            num_parallel_rpcs = 2
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_parallel_rpcs
            ) as executor:
                futs = [
                    executor.submit(rpc_with_profiling, dst_worker)
                    for _ in range(num_parallel_rpcs)
                ]
                # Wait for workers to finish test
                for fut in futs:
                    fut.result()

    def _run_test_profiler_remote_events_profiled(self):
        # Tests that we can successfully invoke the profiler on a remote node,
        # and collect the remote events back in the local profiler.
        if self.rank != 1:
            return

        dst_ranks = [rank for rank in range(0, self.world_size) if rank != self.rank]
        for dst in dst_ranks:
            dst_worker = worker_name(dst)
            with torch.autograd.profiler.profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                ret = fut.wait()

            events = prof.function_events

            rpc_event = get_function_event(events, RPCExecMode.ASYNC.value)
            self.check_profiling_info(
                worker_name(self.rank),
                dst_worker,
                udf_with_torch_ops,
                rpc_event,
                RPCExecMode.ASYNC,
            )

            remote_events = {event.name: event for event in events if event.is_remote}
            rpc_profiling_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC,
                udf_with_torch_ops.__qualname__,
                worker_name(self.rank),
                worker_name(dst),
            )

            for expected_remote_event_name in EXPECTED_REMOTE_EVENTS:
                expected_key = rpc_profiling_key + REMOTE_OP_STR + expected_remote_event_name
                self.assertTrue(expected_key in remote_events)
                remote_event = remote_events[expected_key]
                # Remote event should have a node ID corresponding to the worker
                # it ran on.
                self.assertEqual(remote_event.node_id, dst)

            # Validate order remote events show up in profiling output.
            def convert_remote_to_local(event_name):
                remote_op_key = rpc_profiling_key + REMOTE_OP_STR
                return event_name[
                    event_name.find(remote_op_key)
                    + len(remote_op_key) :
                ]

            remote_events_list = [
                convert_remote_to_local(event.name)
                for event in events
                if convert_remote_to_local(event.name) in EXPECTED_REMOTE_EVENTS
            ]
            self.assertEqual(
                set(remote_events_list),
                set(EXPECTED_REMOTE_EVENTS),
                f"Mismatch between profiled events: {set(remote_events_list)} and expected events: {set(EXPECTED_REMOTE_EVENTS)}",
            )

    @dist_init
    def test_profiler_remote_events_profiled(self):
        self._run_test_profiler_remote_events_profiled()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_remote_events_profiled_single_threaded(self):
        self._run_test_profiler_remote_events_profiled()

    def run_profiling_workload(self, dst):
        fut = rpc.rpc_async(
            worker_name(dst),
            torch.mul,
            args=(
                torch.tensor(1.0, requires_grad=True),
                torch.tensor(1.0, requires_grad=True),
            ),
        )
        fut.wait()

    def _run_rpc_profiling_async_function(self, device="cpu"):
        if self.rank != 1:
            return

        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        x = torch.ones(2)
        y = torch.ones(2)
        with torch.autograd.profiler.profile() as prof:
            ret = rpc.rpc_async(
                dst1, slow_async_add, args=(dst2, x, y, device), timeout=20
            )
            out = ret.wait()

        function_events = prof.function_events
        # slow_async_add resulted in an RPC from dst1 -> dst2, so this should be
        # recorded.
        key_prefix = _build_rpc_profiling_key(
            RPCExecMode.ASYNC, slow_async_add.__qualname__, worker_name(self.rank), dst1
        )

        nested_rpc_key_prefix = _build_rpc_profiling_key(
            RPCExecMode.ASYNC, slow_add.__qualname__, dst1, dst2
        )
        expected_key = key_prefix + REMOTE_OP_STR + nested_rpc_key_prefix
        remote_events = [event for event in function_events if event.is_remote]
        rpc_remote_event = [
            event for event in remote_events if event.name == expected_key
        ]
        self.assertEqual(1, len(rpc_remote_event))
        rpc_remote_event = rpc_remote_event[0]
        self.assertEqual(rpc_remote_event.node_id, (self.rank + 1) % self.world_size)
        # slow_async_add's RPC does an add on dst2, which should be reflected as well.
        remote_add_key = (
            expected_key + REMOTE_OP_STR + torch.jit._builtins._find_builtin(torch.add)
        )
        remote_add_event = [
            event for event in remote_events if event.name == remote_add_key
        ]
        self.assertEqual(1, len(remote_add_event))
        remote_add_event = remote_add_event[0]
        # Validate that node_id is dst2.
        self.assertEqual(remote_add_event.node_id, (self.rank + 2) % self.world_size)

    @dist_init
    def test_rpc_profiling_async_function(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        self._run_rpc_profiling_async_function()
        if torch.cuda.is_available():
            dist.barrier()
            self._run_rpc_profiling_async_function(device="cuda:0")

    @single_threaded_process_group_agent
    @dist_init
    def test_rpc_profiling_async_function_single_threaded(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        self._run_rpc_profiling_async_function()
        if torch.cuda.is_available():
            dist.barrier()
            self._run_rpc_profiling_async_function(device="cuda:0")

    @dist_init
    def test_rpc_profiling_remote_record_function(self):
        # test that functions run over RPC with record_function show the expected
        # profiled block.
        if self.rank != 1:
            return
        dst_ranks = [i for i in range(self.world_size) if i != self.rank]
        for dst_rank in dst_ranks:
            dst_worker = worker_name(dst_rank)
            with torch.autograd.profiler.profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=(-1, True))
                fut.wait()

            function_events = prof.function_events
            record_function_remote_event = [
                evt for evt in function_events if "##forward##" in evt.name
            ]
            self.assertEqual(1, len(record_function_remote_event))
            record_function_remote_event = record_function_remote_event[0]
            self.assertEqual(record_function_remote_event.node_id, dst_rank)
            remaining_remote_events = {
                evt for evt in function_events if evt.node_id == dst_rank
            } - {record_function_remote_event}
            # These ops are created by the hack of casting record_function to a
            # tensor, so they should not count in the actual UDF profiled time.
            # TODO remove after https://github.com/pytorch/pytorch/issues/43868
            # is resolved.
            remote_events_denylist = [
                "aten::zeros",
                "aten::empty",
                "aten::zero_",
                "aten::fill_",
            ]

            REMOTE_OP_STR = "#remote_op: "

            def convert_remote_to_local(event_name):
                remote_op_key = REMOTE_OP_STR
                return event_name[event_name.find(remote_op_key) + len(remote_op_key) :]

            # Ideally, we should validate that the sum of remote operations within
            # record_function are less than record_function's CPU time. However,
            # there is a known bug in profiling
            # (https://github.com/pytorch/pytorch/issues/45160) due to which we
            # can't do this. So, we just validate they are child events.
            prof.key_averages()

            # cpu_children only returns direct children, so here we get all
            # children recursively.
            def get_cpu_children(event):
                if not event.cpu_children:
                    return []
                cpu_children = event.cpu_children
                for e in event.cpu_children:
                    cpu_children.extend(get_cpu_children(e))
                return cpu_children

            record_function_children_names = [
                convert_remote_to_local(c.name)
                for c in get_cpu_children(record_function_remote_event)
            ]
            for evt in remaining_remote_events:
                local_name = convert_remote_to_local(evt.name)
                if local_name not in remote_events_denylist:
                    self.assertTrue(
                        local_name in record_function_children_names,
                        f"{local_name} not in {record_function_children_names}",
                    )

    def validate_profiling_workload(self, dst, prof):

        def convert_remote_to_local(event_name):
            return event_name[event_name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR) :]

        events = prof.function_events
        remote_events = {
            convert_remote_to_local(event.name): event
            for event in events
            if event.is_remote
        }
        self.assertTrue("aten::mul" in remote_events)
        remote_mul_event = remote_events["aten::mul"]
        self.assertEqual(remote_mul_event.node_id, dst)
        self.check_profiling_info(
            worker_name(self.rank),
            worker_name(dst),
            torch.mul,
            remote_mul_event,
            RPCExecMode.ASYNC,
        )

    def _run_test_profiler_with_autograd_context(self):
        dst = (self.rank + 1) % self.world_size
        if self.rank == 1:
            # Cases where we can double wrap messages with profiling information and autograd info.
            with dist_autograd.context() as context_id:
                with torch.autograd.profiler.profile() as prof:
                    self.run_profiling_workload(dst)

            self.validate_profiling_workload(dst, prof)

            # Ensure that flipped order of ctx managers results in events being
            # recorded as expected.
            with torch.autograd.profiler.profile() as prof:
                with dist_autograd.context() as context_id:
                    self.run_profiling_workload(dst)

            self.validate_profiling_workload(dst, prof)

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_autograd_context_single_threaded(self):
        self._run_test_profiler_with_autograd_context()

    @dist_init
    def test_profiler_with_autograd_context(self):
        self._run_test_profiler_with_autograd_context()

    def _profiler_test_with_rpc(self, rpc_exec_mode, func, args, use_record_function=False, dst=None):
        dst = dst if dst is not None else (self.rank + 1) % self.world_size

        # only run profiler on rank 1.
        if self.rank == 1:
            with torch.autograd.profiler.profile() as prof:
                record_function_ctx_mgr = (
                    contextlib.suppress()
                    if not use_record_function
                    else torch.autograd.profiler.record_function(
                        "foo"
                    )
                )
                with record_function_ctx_mgr as rf:
                    if rpc_exec_mode == RPCExecMode.SYNC:
                        rpc.rpc_sync(worker_name(dst), func, args=args)
                    elif rpc_exec_mode == RPCExecMode.ASYNC:
                        fut = rpc.rpc_async(worker_name(dst), func, args=args)
                        fut.wait()
                    else:
                        self.assertTrue(rpc_exec_mode == RPCExecMode.REMOTE)
                        rref = rpc.remote(worker_name(dst), func, args=args)
                        rref.to_here()
                        # To avoid flakiness, wait for the RRef to be profiled. This
                        # means that we received the acknowledgement of successful
                        # creation on the owner and ran the callbacks responsible
                        # for recording the profiling event.
                        rref._get_profiling_future().wait()

            events = prof.function_events
            rpc_event = get_function_event(events, rpc_exec_mode.value)
            # verify Node ID for this rpc event.
            self.assertEqual(rpc_event.node_id, self.rank)
            # Ensure recording of remote events.
            remote_events = {event for event in events if event.node_id == dst} - {rpc_event}
            self.assertGreaterEqual(len(remote_events), 1)
            for remote_event in remote_events:
                self.assertEqual(remote_event.node_id, dst)

            if use_record_function:
                scope_event = get_function_event(events, "foo")
                # Since RPC call is within the scope, its CPU interval should be
                # contained within foo's interval.
                self.assertLessEqual(scope_event.time_range.start, rpc_event.time_range.start)
                self.assertGreaterEqual(scope_event.time_range.end, rpc_event.time_range.end)
            # the sender, dest worker, function run, and type of RPC should all
            # be recorded.
            self_worker_name = worker_name(self.rank)
            dst_worker_name = worker_name(dst)
            self.check_profiling_info(self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode)
            if use_record_function:
                # verify order by ensuring that the outer context comes
                # before the rpc event.
                foo_event_ix = next(i for i, event in enumerate(events) if "foo" in event.name)
                rpc_event_idx = next(i for i, event in enumerate(events) if rpc_exec_mode.value in event.name)
                self.assertLess(foo_event_ix, rpc_event_idx)

    def _run_test_profiler_with_sync_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,),
                                     use_record_function=True)

    @dist_init
    def test_profiler_with_sync_rpc_udf(self):
        self._run_test_profiler_with_sync_rpc_udf()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_sync_rpc_udf_single_threaded(self):
        self._run_test_profiler_with_sync_rpc_udf()

    def _run_test_profiler_with_sync_rpc_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, torch.mul, args=(torch.ones(1), torch.ones(1)),
            use_record_function=True
        )

    @dist_init
    def test_profiler_with_sync_rpc_builtin(self):
        self._run_test_profiler_with_sync_rpc_builtin()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_sync_rpc_builtin_single_threaded(self):
        self._run_test_profiler_with_sync_rpc_builtin()

    def _run_test_profiler_with_async_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,),
                                     use_record_function=True)

    @dist_init
    def test_profiler_with_async_rpc_udf(self):
        self._run_test_profiler_with_async_rpc_udf()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_async_rpc_udf_single_threaded(self):
        self._run_test_profiler_with_async_rpc_udf()

    def _run_test_profiler_with_async_rpc_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, torch.mul, args=(torch.ones(1), torch.ones(1)),
            use_record_function=True
        )

    @dist_init
    def test_profiler_with_async_rpc_builtin(self):
        self._run_test_profiler_with_async_rpc_builtin()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_async_rpc_builtin_single_threaded(self):
        self._run_test_profiler_with_async_rpc_builtin()

    def _run_test_profiler_with_remote_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_sleep_func, args=(1,), use_record_function=True
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_sleep_func, args=(1,), dst=self.rank
        )

    @dist_init
    def test_profiler_with_remote_udf(self):
        self._run_test_profiler_with_remote_udf()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_remote_udf_single_threaded(self):
        self._run_test_profiler_with_remote_udf()

    def _run_test_profiler_with_remote_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1)),
            use_record_function=True
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            dst=self.rank,
        )

    @dist_init
    def test_profiler_with_remote_builtin(self):
        self._run_test_profiler_with_remote_builtin()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_remote_builtin_single_threaded(self):
        self._run_test_profiler_with_remote_builtin()

    def _run_test_profiler_with_script_async_rpc(self):
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_script_async_rpc(self):
        self._run_test_profiler_with_script_async_rpc()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_script_async_rpc_single_threaded(self):
        self._run_test_profiler_with_script_async_rpc()

    def _run_test_profiler_with_script_sync_rpc(self):
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_script_sync_rpc(self):
        self._run_test_profiler_with_script_sync_rpc()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_script_sync_rpc_single_threaded(self):
        self._run_test_profiler_with_script_sync_rpc()

    def _run_test_profiler_with_script_remote_rpc(self):
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),), dst=self.rank
        )

    @dist_init
    def test_profiler_with_script_remote_rpc(self):
        self._run_test_profiler_with_script_remote_rpc()

    @single_threaded_process_group_agent
    @dist_init
    def test_profiler_with_script_remote_rpc_single_threaded(self):
        self._run_test_profiler_with_script_remote_rpc()

    def _assert_top_level_events(self, process_global_events, expected_top_level_event_names):
        top_level_event_names = []
        for thread_local_events in process_global_events:
            # Get top-level events from all events happened on a thread.
            last_end_time = 0
            for event in thread_local_events:
                event_name = event.name
                time_range = event.time_range
                if time_range.start > last_end_time:
                    top_level_event_names.append(event_name)
                    last_end_time = time_range.end
        top_level_event_names = sorted(top_level_event_names)
        expected_top_level_event_names = sorted(expected_top_level_event_names)
        self.assertEqual(
            top_level_event_names,
            expected_top_level_event_names,
            f"Expected events {expected_top_level_event_names}, but got {top_level_event_names}",
        )

    @dist_init
    def test_server_process_global_profiler(self):
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = worker_name(dst_rank)

        x = torch.tensor(1)
        y = torch.tensor(2)

        outer_profile_rref = rpc.remote(dst_worker_name, rpc._server_process_global_profile)
        outer_profile_rref.rpc_sync().__enter__()
        rpc.rpc_sync(dst_worker_name, torch.add, (x, y))
        inner_profile_rref = rpc.remote(dst_worker_name, rpc._server_process_global_profile)
        inner_profile_rref.rpc_sync().__enter__()
        rpc.rpc_sync(dst_worker_name, torch.sub, (x, y))
        inner_profile_rref.rpc_sync().__exit__(None, None, None)
        outer_profile_rref.rpc_sync().__exit__(None, None, None)

        inner_events = rpc.rpc_sync(dst_worker_name, get_events_from_profile, (inner_profile_rref,))
        expected_inner_events = ['aten::sub']
        expected_outer_events = expected_inner_events + ['aten::add']

        self._assert_top_level_events(inner_events, expected_inner_events)
        outer_events = rpc.rpc_sync(dst_worker_name, get_events_from_profile, (outer_profile_rref,))
        self._assert_top_level_events(outer_events, expected_outer_events)

        inner_profile_rref.rpc_sync().key_averages()
        outer_profile_rref.rpc_sync().key_averages()

    @dist_init
    def test_async_record_function_double_end_callbacks(self):
        num_sleep_seconds = 1
        if self.rank == 1:
            # Validate that calling the function twice results in an error.
            with torch.autograd.profiler.profile() as pf:
                with torch.autograd.profiler.record_function("foo") as rf:
                    fut = rpc.rpc_async(
                        worker_name(0), my_sleep_func, args=(num_sleep_seconds,)
                    )
                    rf._call_end_callbacks_on_future(fut)
                    with self.assertRaisesRegex(
                        RuntimeError, "can only be called once."
                    ):
                        rf._call_end_callbacks_on_future(fut)
                fut.wait()

    @dist_init
    def test_async_record_function_cbs_jit_call(self):
        if self.rank == 1:
            with torch.autograd.profiler.profile() as pf:
                key = _build_rpc_profiling_key(
                    RPCExecMode.ASYNC,
                    torch._jit_internal._qualified_name(my_script_func),
                    "worker1",
                    "worker0",
                )
                with torch.autograd.profiler.record_function(key) as rf:
                    fut = rpc.rpc_async(
                        worker_name(0), my_script_func, args=(torch.tensor(1),)
                    )
                    # Intentionally calling record_function internals
                    fut = torch.ops.profiler._call_end_callbacks_on_jit_fut(rf.handle, fut)
                result = fut.wait()
                # Validate that the profiling future returns the same value as the RPC
                # future.
                expected = torch.add(torch.tensor(1), torch.tensor(1))
                self.assertEqual(result, expected)
            events = pf.function_events
            rpc_event = get_function_event(
                events, torch._jit_internal._qualified_name(my_script_func)
            )
            self.assertTrue(torch._jit_internal._qualified_name(my_script_func) in rpc_event.name)

    @dist_init
    def test_py_class_constructor(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass, args=(n,))
        self.assertEqual(ret.a, n)

    @dist_init
    def test_py_class_instance_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass(2).my_instance_method, args=(n,)
        )
        self.assertEqual(ret, MyClass(2).my_instance_method(n))

    @dist_init
    def test_py_class_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass.my_class_method, args=(n, n + 1)
        )
        self.assertEqual(ret, MyClass.my_class_method(n, n + 1))

    @dist_init
    def test_py_class_static_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass.my_static_method, args=(n + 10,)
        )
        self.assertEqual(ret, MyClass.my_static_method(n + 10))

    @dist_init
    def test_py_multi_async_call(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker_info = rpc.get_worker_info(worker_name(dst_rank))
        fut1 = rpc.rpc_async(dst_worker_info, MyClass.my_static_method, args=(n + 10,))
        fut2 = rpc.rpc_async(dst_worker_info, min, args=(n, n + 1, n + 2))
        self.assertEqual(fut1.wait(), MyClass.my_static_method(n + 10))
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))

    @dist_init
    def test_py_no_return_result(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), no_result)
        self.assertEqual(ret, no_result())

    @dist_init
    def test_py_tensors(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
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
                worker_name(dst_rank),
                my_tensor_function,
                args=(torch.ones(i, i), torch.ones(i, i)),
            )
            futs.append(fut)

        j = 0
        for val in torch.futures.wait_all(futs):
            self.assertEqual(
                val, my_tensor_function(torch.ones(j, j), torch.ones(j, j))
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
            worker_name(dst_rank), my_complex_tensor_function, args=(a, b, c)
        )
        self.assertEqual(ret, my_complex_tensor_function(a, b, c))

    @dist_init
    def test_py_nested_pickle(self):
        n = self.rank + 1
        dst_rank = n % self.world_size

        ret = rpc.rpc_sync(
            worker_name(dst_rank),
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
            ret = rpc.rpc_sync(worker_name(dst_rank), no_result, args=(10,))

    @dist_init
    def test_py_raise_in_user_func(self):
        with captured_output() as (_, err):
            # This barrier prevents a race condition where the main thread has
            # not entered the context manager when the remote function runs.
            initialize_pg(self.file_init_method, self.rank, self.world_size)
            dist.barrier()
            n = self.rank + 1
            dst_rank = n % self.world_size
            fut = rpc.rpc_async(worker_name(dst_rank), raise_func)
            with self.assertRaisesRegex(ValueError, expected_err):
                fut.wait()
            # This barrier prevents a race condition where the main thread exits
            # context manager before the remote function has ran.
            dist.barrier()

        # Validate that trainers log errors when running functions.
        stderr_lines = err.getvalue()
        self.assertTrue(expected_err in stderr_lines)

    @dist_init
    def test_py_raise_in_user_func_escaped_str(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(worker_name(dst_rank), raise_func_escape)
        try:
            fut.wait()
        except ValueError as e:
            msg = str(e)
            # Ensure newlines are unescaped to provide a better repr of error.
            self.assertEqual(msg, msg.encode("utf-8").decode("unicode_escape"))
        else:
            self.assertTrue(False, "expected raise_func_escape to raise ValueError.")

    @dist_init
    def test_nested_rpc(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            nested_rpc,
            args=(worker_name(self.rank),),
        )
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    def _stress_test_rpc(self, f, repeat=1000, args=()):
        n = self.rank + 1
        dst_rank = n % self.world_size
        futs = []
        tik = time.time()
        for _ in range(repeat):
            fut = rpc.rpc_async(worker_name(dst_rank), f, args=args)
            futs.append(fut)

        for val in torch.futures.wait_all(futs):
            self.assertEqual(val, 0)
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
        self._stress_test_rpc(heavy_rpc_torchscript, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_builtin_remote_ret(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(rref.to_here(), torch.ones(n, n) * 2)

    @dist_init
    def test_builtin_remote_self(self):
        rref = rpc.remote(
            worker_name(self.rank),
            torch.add,
            args=(torch.ones(2, 2), torch.ones(2, 2)),
        )
        self.assertEqual(rref.local_value(), torch.ones(2, 2) * 2)

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
                    worker_name(dst_rank),
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
            worker_name(dst_rank),
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
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 2)
        )
        rref_b = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1)
        )
        rref_c = rpc.remote(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), torch.ones(n, n) + 4)

    @dist_init
    def test_py_rref_args_user_share(self):
        n = self.rank + 1
        owner_rank = n % self.world_size
        user_rank = (n + 1) % self.world_size
        rref_a = rpc.remote(
            worker_name(owner_rank), my_function, args=(torch.ones(n, n), 2, 0)
        )
        rref_b = rpc.remote(
            worker_name(owner_rank), my_function, args=(torch.ones(n, n), 1, 0)
        )
        rref_c = rpc.remote(
            worker_name(user_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), torch.ones(n, n) + 4)

    @dist_init
    def test_py_rpc_rref_args(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(
            worker_name(dst_rank), my_function, args=(torch.ones(n, n), 2, 0)
        )
        rref_b = rpc.remote(
            worker_name(dst_rank), my_function, args=(torch.ones(n, n), 1, 0)
        )

        c = rpc.rpc_sync(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )

        self.assertEqual(c, torch.ones(n, n) + 4)

    @dist_init
    def test_nested_remote(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size

        rref = rpc.remote(
            worker_name(dst_rank1),
            nested_remote,
            args=(worker_name(dst_rank2),),
        )
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 3)

    @dist_init
    def test_nested_rref(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref_of_rrefs = rpc.remote(
            worker_name(dst_rank1),
            nested_rref,
            args=(worker_name(dst_rank2),),
        )

        # Say C has 2 OwnerRRefs.
        # B has 2 UserRRefs to those 2 OwnerRRefs, respectively.
        # This call is effectively A asking B to share its 2 UserRRefs.
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
                    worker_name(dst_rank1),
                    nested_rref,
                    args=(worker_name(dst_rank2),),
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
        rref = rpc.remote(worker_name(dst_rank), raise_func)
        with self.assertRaises(ValueError):
            rref.to_here()
        # check ref to itself
        rref = rpc.remote(worker_name(self.rank), no_result, args=(10,))
        with self.assertRaises(TypeError):
            rref.to_here()

    @dist_init
    def test_rpc_return_rref(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref = rpc.rpc_sync(
            worker_name(dst_rank1),
            rpc_return_rref,
            args=(worker_name(dst_rank2),),
        )
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_rref_forward_chain(self):
        ttl = 8
        n = self.rank + 1
        dst_rank = n % self.world_size

        rref = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1)
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
            worker_name(next_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        with self.assertRaisesRegex(
            RuntimeError, (
                fr"For UserRRef\(rref_id=GloballyUniqueId\(created_on={self.rank}, local_id=0\), "
                fr"fork_id=GloballyUniqueId\(created_on={self.rank}, local_id=1\)\), "
                r"can't call localValue\(\) on user "
                fr"WorkerInfo\(id={self.rank}, name={worker_name(self.rank)}\). "
                fr"Call it on owner WorkerInfo\(id={next_rank}, name={worker_name(next_rank)}\)"
            )
        ):
            rref.local_value()

    @dist_init
    def test_return_local_rrefs(self):
        n = self.rank + 1
        dst_rank = n % self.world_size

        rref_list = rpc.rpc_sync(
            worker_name(dst_rank), get_rref_list, args=([1, 2, 3],)
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
    def _test_rref_type(self, blocking):

        def launched_rpc(events):
            expected_name = f"rpc_{RPCExecMode.ASYNC.value}#_rref_typeof_on_owner"
            return any([e.name.startswith(expected_name) for e in events])

        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, torch.add, args=(torch.ones(2), 1))

        with torch.autograd.profiler.profile() as p:
            t = rref._get_type(blocking=blocking)
            if not blocking:
                t = t.wait()

        self.assertTrue(launched_rpc(p.function_events))
        expected_type = type(torch.ones(2))
        self.assertEqual(t, expected_type)

        futs = []

        def verify(fut):
            self.assertEqual(fut.value(), expected_type)

        with torch.autograd.profiler.profile() as p:
            for _ in range(10):
                t = rref._get_type(blocking=blocking)
                if not blocking:
                    futs.append(t)
                    t.add_done_callback(verify)
                    t = t.wait()
                self.assertEqual(t, expected_type)

        if not blocking:
            # Note that cached calls with blocking=False all return the same
            # cached original future.
            first_fut = futs[0]
            for f in futs[1:]:
                self.assertTrue(f is first_fut)
        # Ensure we never launch another RPC, other than for the very
        # first call.
        self.assertFalse(launched_rpc(p.function_events))
        self.assertEqual(t, type(torch.ones(2)))

        rref = rpc.remote(dst, MyClass, args=(0,))
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, MyClass)

    def test_rref_type_blocking(self):
        self._test_rref_type(blocking=True)

    def test_rref_type_non_blocking(self):
        self._test_rref_type(blocking=False)

    @dist_init
    def _test_rref_type_with_error(self, blocking):
        dst = worker_name((self.rank + 1) % self.world_size)
        # 10 ms timeout
        rref = rpc.remote(dst, raise_func)
        # Blocking: error raised inline
        if blocking:
            with self.assertRaisesRegex(ValueError, "Expected error"):
                rref._get_type(blocking=blocking)
        else:
            # Non-blocking: Immediately return future, block on wait
            fut = rref._get_type(blocking=blocking)
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut.wait()


    def test_rref_type_with_error_blocking(self):
        self._test_rref_type_with_error(blocking=True)

    def test_rref_type_with_error_non_blocking(self):
        self._test_rref_type_with_error(blocking=False)

    @dist_init
    def _test_rref_type_owner(self, blocking):
        rref = RRef(torch.ones(2) + 1)
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, type(torch.ones(2)))

        rref = RRef(MyClass(0))
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, MyClass)

    def test_rref_type_owner_blocking(self):
        self._test_rref_type_owner(blocking=True)

    def test_rref_type_owner_non_blocking(self):
        self._test_rref_type_owner(blocking=False)

    @staticmethod
    def _slow_add(x, y):
        time.sleep(1)
        return x + y

    @dist_init
    def test_rref_type_slow_init(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, RpcTest._slow_add, args=(torch.ones(2), 1))
        self.assertEqual(rref._get_type(), type(torch.ones(2)))

    @dist_init
    def test_owner_equality(self):
        a = RRef(40)
        b = RRef(50)

        other_rank = (self.rank + 1) % self.world_size
        other_a = rpc.remote(
            worker_name(other_rank), torch.add, args=(torch.ones(1), 1)
        )
        other_b = rpc.remote(
            worker_name(other_rank), torch.add, args=(torch.ones(1), 1)
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
        dst_worker = worker_name(dst_rank)

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
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 2)
        )
        rref_b = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1)
        )
        rref_c = rpc.remote(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
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
        dst_worker = worker_name(dst_rank)

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

    # Notice `rpc.api.shutdown()` accesses
    # `_delete_all_user_and_unforked_owner_rrefs` through
    # `torch.distributed.rpc.api`, so patching
    # `torch.distributed.rpc._delete_all_user_and_unforked_owner_rrefs` will
    # not help.
    @mock.patch.object(torch.distributed.rpc.api, "_delete_all_user_and_unforked_owner_rrefs")
    def _test_rref_leak(self, _mock_delete_all_user_and_unforked_owner_rrefs, ignore_leak):
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # Wait for all init to complete.
        dist.barrier()

        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
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
            "OwnerRRef({}(created_on={}, local_id=0))".format(id_class, self.rank), rref1.__str__()
        )

        dst_rank = (self.rank + 1) % self.world_size
        rref2 = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        self.assertEqual(
            rref2.__str__(),
            "UserRRef(RRefId = {0}(created_on={1}, local_id=1), ForkId = {0}(created_on={1}, local_id=2))".format(
                id_class, self.rank
            ),
        )

    @dist_init
    def test_rref_get_future(self):
        # Tests that we can obtain the future corresponding to the creation of
        # the RRef on remote end
        if self.rank == 0:
            # Builtin
            rref = rpc.remote(worker_name(1), torch.add, args=(1, 1))
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)

            # UDF
            rref = rpc.remote(worker_name(1), foo_add, args=())
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)

            # Script
            rref = rpc.remote(worker_name(1), my_script_func, args=(torch.tensor(1), ))
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)


    @dist_init
    def test_rref_context_debug_info(self):
        # This test checks local states that are modified by remote workers.
        # This means that we would need barrier before and after every check.
        # The barrier before the check makes sure that all previous states are
        # cleared globally, the barrier after ensures that no following states
        # change gets into the current check.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

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
        rpc.rpc_sync(worker_name(dst_rank), set_global_rref, args=(rref1,))

        # barrier before check 2
        wait_until_pending_futures_and_users_flushed()
        dist.barrier()

        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertEqual(1, int(info["num_owner_rrefs"]))
        # no pending users since the fork is finished
        self.assertEqual(0, int(info["num_pending_users"]))
        # barrier after check 2
        dist.barrier()

        # clear states for check 2
        rpc.rpc_sync(worker_name(dst_rank), clear_global_rref)

        # Check 3: rpc.remote call should update owners_ map
        ####################################################
        rref2 = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        rref3 = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        rref2.to_here()
        rref3.to_here()

        # barrier before check 3
        wait_until_pending_futures_and_users_flushed()
        dist.barrier()

        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertEqual(2, int(info["num_owner_rrefs"]))
        # no pending users since the fork is finished
        self.assertEqual(0, int(info["num_pending_users"]))

        # barrier after check 3
        dist.barrier()

    @dist_init
    def test_disable_gil_profiling(self):
        # test that rpc.enable_gil_profiling(false) will result in
        # GIL wait time not being recorded.

        # GIL profiling should be disabled by default.
        dst_rank = (self.rank + 1) % self.world_size
        rpc.rpc_sync(
            worker_name(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertRaises(KeyError, lambda: info["agent.gil_average_wait_time_us"])
        rpc.enable_gil_profiling(True)
        rpc.rpc_sync(
            worker_name(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertIn("agent.gil_average_wait_time_us", info)

    @dist_init(setup_rpc=False)
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
        # NB: Key ordering is only preserved in python 3.6+. So here, we
        # manually check keys are equal.
        for key in expected.keys():
            self.assertIn(key, info.keys())

        for key in info.keys():
            self.assertIn(key, expected.keys())

    @dist_init(setup_rpc=False)
    @unittest.skipIf(
        IS_MACOS,
        "Test is flaky on MacOS since libuv error handling is not as robust as TCP",
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
        rpc._set_rpc_timeout(10)
        # This barrier is needed to ensure that some workers do not exit before
        # others have been brought up, for non ProcessGroupAgent backends.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        if self.rank == 1:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker = worker_name(dst_rank)
            # allow destination worker to exit without joining
            error_str = self.get_shutdown_error_regex()
            wait_until_node_failure(dst_rank, error_str)
            fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.ones(1), 3))
            # Shutdown sequence is not very well defined and as a result
            # we can see any of the error messages defined in get_shutdown_error_regex.
            with self.assertRaisesRegex(RuntimeError, error_str):
                fut.wait()
        # exit all workers non-gracefully.
        rpc.shutdown(graceful=False)

    @dist_init
    def test_deadlock(self):
        # this test is copied from https://github.com/pytorch/pytorch/issues/45089
        if self.rank == 1:
            dst1 = worker_name((self.rank + 1) % self.world_size)
            x = torch.ones(2)
            y = torch.ones(2)
            rpc.rpc_async(dst1, RpcTest._slow_add, args=(x, y), timeout=15).wait()

        dist_initialized = dist.is_initialized()
        if not dist_initialized:
            dist.init_process_group(
                backend="gloo",
                init_method=self.file_init_method,
                rank=self.rank,
                world_size=self.world_size,
            )

    @dist_init(setup_rpc=False)
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
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # A barrier is needed to ensure that all RPCs are processed.
        # Otherwise, some RPCs can timeout since the receiving end
        # has terminated.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        # pass in graceful=False to ensure that we don't wait for other workers.
        rpc.shutdown(graceful=False)

    @dist_init(setup_rpc=False)
    def test_set_and_get_default_rpc_timeout(self):
        timeout = 0.5

        # A new `RpcBackendOptions` is constructed
        # when accessing `self.rpc_backend_options`.
        rpc_backend_options = self.rpc_backend_options
        rpc_backend_options.rpc_timeout = timeout

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )
        set_timeout = rpc.get_rpc_timeout()
        self.assertEqual(timeout, set_timeout)
        rpc.shutdown()

    @dist_init
    def test_default_timeout_used(self):
        """
        Tests that if no timeout is passed into rpc_async and rpc_sync, then the
        default timeout is used.
        """
        dst_rank = (self.rank + 1) % self.world_size
        rpc._set_rpc_timeout(0.001)  # 1 ms
        # futures should time out and be marked with an exception indicating it as such.
        futs = [
            rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=())
            for _ in range(10)
        ]
        expected_error = self.get_timeout_error_regex()
        for fut in futs:
            with self.assertRaisesRegex(RuntimeError, expected_error):
                fut.wait()

        # ensure that if a new timeout is set old futures don't time out but new ones do.
        rpc._set_rpc_timeout(200)  # 200 seconds
        # create a longstanding RPC.
        fut1 = rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=(1,))
        # now, set a short timeout.
        rpc._set_rpc_timeout(0.001)
        # fut2 should time out, fut1 should not.
        fut2 = rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=(1,))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut2.wait()
        fut1.wait()

        # Zero timeout means infinity, so future should run to completion.
        rpc._set_rpc_timeout(0)
        rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=()).wait()

        # reset to default timeout so shutdown messages can process cleanly.
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init
    def test_rpc_timeouts(self):
        # TODO: enable timeouts for rpc.remote/RRef (https://github.com/pytorch/pytorch/issues/33803)
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst_rank)
        timeout = 0.1  # 100 ms
        expected_error = self.get_timeout_error_regex()
        # Test async UDF
        fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=timeout)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if there is no timeout and we use the default
        # RPC timeout.
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,)).wait()

        # Test sync UDF
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=timeout)

        # Ensure run to completion if there is no timeout and we use the default
        # RPC timeout.
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))

        # If we set a default timeout for RPCs, it should be respected, though
        # still overridden if we pass in a different timeout to the APIs.
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))

        # The RPCs should run to completion since we override the timeout.
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=5).wait()
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=5)
        # Passing in a zero timeout should ensure that the RPC won't time out.
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=0).wait()
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=0)
        # Reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

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
    def test_wait_all(self):
        with _wait_all():
            self.assertTrue(_thread_local_var.future_list == [])
            dst = worker_name((self.rank + 1) % self.world_size)
            fut = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
            self.assertTrue(len(_thread_local_var.future_list) == 1)
            self.assertTrue(isinstance(_thread_local_var.future_list[0], torch._C.Future))
        self.assertTrue(fut.done())
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_multiple_call(self):
        with _wait_all():
            self.assertTrue(_thread_local_var.future_list == [])
            dst = worker_name((self.rank + 1) % self.world_size)
            for i in range(20):
                fut = rpc.rpc_async(dst, torch.add, (torch.ones(i, i), 1))
                res = rpc.rpc_sync(dst, torch.add, (torch.ones(i, i), 1))
                self.assertEqual(res, torch.ones(i, i) + 1)
                self.assertEqual(fut.wait(), torch.ones(i, i) + 1)
            self.assertTrue(len(_thread_local_var.future_list) == 20)
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_timeout(self):
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            with _wait_all():
                self.assertTrue(_thread_local_var.future_list == [])
                dst = worker_name((self.rank + 1) % self.world_size)
                timeout = 0.1  # 100 ms
                fut = rpc.rpc_async(dst, my_sleep_func, args=(1,), timeout=timeout)
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_raise_in_user_func(self):
        with self.assertRaises(ValueError):
            with _wait_all():
                self.assertTrue(_thread_local_var.future_list == [])
                dst = worker_name((self.rank + 1) % self.world_size)
                fut = rpc.rpc_async(dst, raise_func)
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_raise_in_body(self):
        with self.assertRaises(ValueError):
            with _wait_all():
                raise_func()
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

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

    @dist_init
    def test_non_garbage_collected_user_rref_due_to_local_circular_dependency(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        a = MyClass(1)
        b = MyClass(2)

        # This is to make Python not garbage collect a and b.
        a.other = b
        b.other = a

        n = self.rank
        a.rref = rpc.remote(
            dst_worker_name,
            torch.add,
            args=(torch.ones(n, n), 2)
        )

    @dist_init(setup_rpc=False)
    def test_use_rref_after_shutdown(self):
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # pass in graceful=True to ensure that local UserRRefs are deleted.
        rpc.shutdown(graceful=True)

        with self.assertRaisesRegex(
            RuntimeError, "Cannot call to_here\\(\\) on it after deletion."
        ):
            rref.to_here()

        with self.assertRaisesRegex(
            RuntimeError, "Cannot call fork an UserRRef after deletion."
        ):
            import torch.distributed.rpc.internal as internal
            internal.serialize(rref)

    @staticmethod
    def _return_gpu_tensor():
        return torch.rand(3, 3).cuda(0)

    @staticmethod
    def _return_gpu_tensor_list():
        return [torch.rand(3, 3).cuda(0), torch.rand(3, 3).cuda(1)]

    @staticmethod
    def _gpu_tensor_list_arg(tensor_list):
        return torch.rand(3, 3)

    def _create_rref(self):
        owner_rank = (self.rank + 2) % self.world_size
        return rpc.remote(
            worker_name(owner_rank),
            torch.add,
            args=(torch.zeros(2, 2), 1)
        )

    @dist_init
    def test_user_rrefs_confirmed(self):
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            check_rref_confirmed,
            args=(rref,)
        )
        self.assertEqual(ret, True)

    @dist_init
    def test_user_rrefs_confirmed_remote(self):
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret_rref = rpc.remote(
            worker_name(dst_rank),
            check_rref_confirmed,
            args=(rref,)
        )
        self.assertEqual(ret_rref.to_here(), True)

    @dist_init
    def test_rref_py_pickle_not_supported(self):
        local_rref = RRef(35)
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, "Can not pickle rref in python pickler"):
                torch.save(local_rref, fname)

    @dist_init
    def test_remote_throw(self):
        rref = rpc.remote(worker_name((self.rank + 1) % self.world_size),
                          raise_or_inc,
                          args=(torch.ones(2),))
        with self.assertRaisesRegex(Exception, ".*Expected error.*"):
            rref.to_here()

    @dist_init
    def test_non_cont_tensors(self):
        if self.rank == 0:
            # Create a non-contiguous tensor.
            t = torch.rand(5, 5)
            t_view = t.narrow(1, 2, 2)
            self.assertFalse(t_view.is_contiguous())
            t_cont = t_view.contiguous()
            self.assertTrue(t_cont.is_contiguous())
            self.assertEqual(t_view, t_cont)

            # Send non-cont tensor over RPC.
            next_rank = (self.rank + 1) % self.world_size
            t_ret = rpc.rpc_sync(worker_name(next_rank), non_cont_test, args=(t_view, t_cont))

            # Verify the returned tensor.
            self.assertEqual(t_view, t_ret)
            self.assertFalse(t_ret.is_contiguous())

    @dist_init
    def test_callback_simple(self):
        set_by_cb = concurrent.futures.Future()
        n = self.rank + 1

        def callback(fut):
            ret = fut.wait()
            self.assertEqual(ret, torch.ones(n, n) * 2)
            set_by_cb.set_result(ret.clone() + 1)

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n))
        )

        fut.then(callback)

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
        self.assertEqual(set_by_cb.result(), torch.ones(n, n) * 2 + 1)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_callback_wrong_arg_num(self):
        set_by_cb = concurrent.futures.Future()
        n = self.rank + 1

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n))
        )

        cb_fut = fut.then(my_function)

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "my\\_function\\(\\) missing 2 required positional arguments"
        ):
            cb_fut.wait()

    @dist_init
    def test_callback_wrong_arg_type(self):
        dst = worker_name((self.rank + 1) % self.world_size)

        fut0 = rpc.rpc_async(dst, torch.add, args=(torch.ones(2, 2), 1))
        fut1 = fut0.then(lambda x: x + 1)

        with self.assertRaisesRegex(
            RuntimeError,
            "unsupported operand type\\(s\\) for \\+"
        ):
            fut1.wait()

    @dist_init
    def test_callback_multi(self):
        num_cbs = 10
        n = self.rank + 1

        def callback(idx, fut):
            ret = fut.wait()
            self.assertEqual(ret, torch.ones(n, n) * 2)
            return ret + idx

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n))
        )

        cb_futs = []
        for idx in range(num_cbs):
            cb_futs.append(fut.then(partial(callback, idx)))

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        for idx in range(num_cbs):
            self.assertEqual(
                cb_futs[idx].wait(),
                torch.ones(n, n) * 2 + idx
            )

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_callback_chain(self):
        n = self.rank + 1
        dst = worker_name(n % self.world_size)

        def callback(fut):
            return fut.wait() + 1

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), 1)
        )

        num_cbs = 20
        for _ in range(num_cbs):
            fut = fut.then(callback)

        self.assertEqual(fut.wait(), torch.ones(n, n) + 1 + num_cbs)

    @dist_init
    def test_callback_in_rpc(self):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        ret = rpc.rpc_sync(
            dst1,
            add_use_future_cb,
            args=(dst2, torch.ones(2, 2), 1, 2)
        )
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 2)

    @dist_init
    def test_callback_with_ret(self):
        dst = worker_name((self.rank + 1) % self.world_size)

        def callback(fut0):
            fut2 = rpc.rpc_async(
                dst,
                torch.add,
                args=(fut0.wait(), 1)
            ).then(lambda fut1: fut1.wait() + 1)

            return fut2.wait()

        fut3 = rpc.rpc_async(
            dst,
            torch.add,
            args=(torch.ones(2, 2), 1)
        ).then(callback)

        self.assertEqual(fut3.wait(), torch.ones(2, 2) + 3)

    @dist_init
    def test_callback_with_error(self):
        dst = worker_name((self.rank + 1) % self.world_size)

        def callback(fut0):
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut0.wait()
            raise RuntimeError("Another expected error")

        fut1 = rpc.rpc_async(dst, raise_func).then(callback)
        with self.assertRaisesRegex(RuntimeError, "Another expected error"):
            fut1.wait()

    @dist_init
    def test_callback_none(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(
            TypeError,
            "incompatible function arguments."
        ):
            rpc.rpc_async(dst, raise_func).then(None)

    @dist_init
    def test_add_done_callback(self):
        set_by_cb = False
        n = self.rank + 1

        def callback(fut):
            nonlocal set_by_cb
            fut.wait()
            set_by_cb = True

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n))
        )

        fut.add_done_callback(callback)
        fut_then = fut.then(lambda _: True)

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        # We have no guarantee that the add_done_callback fn will execute before the test finishes.
        # Adding a 'then' callback that runs afterwards to guarantee we wait for the first callback
        fut_then.wait()
        self.assertTrue(set_by_cb)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_mark_future_twice(self):
        fut = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            torch.add,
            args=(torch.zeros(2, 2), 1)
        )
        self.assertEqual(fut.wait(), torch.zeros(2, 2) + 1)
        with self.assertRaisesRegex(
            RuntimeError,
            "Future can only be marked completed once"
        ):
            fut.set_result(1)

    @dist_init
    def test_pickle_future(self):
        fut = torch.futures.Future()
        errMsg = "Can not pickle torch.futures.Future"

        dst = worker_name((self.rank + 1) % self.world_size)
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.rpc_sync(dst, fail_on_fut, args=(fut,))

        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.rpc_async(dst, fail_on_fut, args=(fut,))

        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.remote(dst, fail_on_fut, args=(fut,))

    @dist_init
    def test_future_done(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        fut = rpc.rpc_async(dst, torch.add, args=(torch.zeros(2), 1))
        fut.wait()
        self.assertTrue(fut.done())

    @dist_init
    def test_future_done_exception(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        fut = rpc.rpc_async(dst, raise_func)
        with self.assertRaisesRegex(ValueError, "Expected error"):
            fut.wait()
        self.assertTrue(fut.done())

    def _test_future_cb(self, func):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        ret = rpc.rpc_sync(
            dst1,
            func,
            args=(dst2, torch.ones(2, 2), 1, 2)
        )
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 2)

    @dist_init
    def test_future_in_rpc(self):
        self._test_future_cb(add_use_future_set_result)

    @dist_init
    def test_future_nested_callback(self):
        self._test_future_cb(add_use_future_nested_cb)

    def _run_func_in_mode(self, to, fn, mode, args=None, kwargs=None):
        if mode == RPCExecMode.SYNC:
            return rpc.rpc_sync(to, fn, args=args, kwargs=kwargs)
        elif mode == RPCExecMode.ASYNC:
            return rpc.rpc_async(to, fn, args=args, kwargs=kwargs).wait()
        elif mode == RPCExecMode.REMOTE:
            return rpc.remote(to, fn, args=args, kwargs=kwargs).to_here()

    def _test_async_function_raise(self, mode):
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size),
                async_raise_func,
                mode
            )

    @dist_init
    def test_async_function_raise(self):
        self._test_async_function_raise(RPCExecMode.SYNC)

    @dist_init
    def test_async_function_raise_async(self):
        self._test_async_function_raise(RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_raise_remote(self):
        self._test_async_function_raise(RPCExecMode.REMOTE)

    def _test_async_function_wrong_return_type(self, mode):
        errMsg = (
            "Functions decorated with @rpc\\.async_function must return a "
            "torch\\.futures\\.Future object,"
        )
        with self.assertRaisesRegex(RuntimeError, errMsg):
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size),
                async_wrong_type,
                mode
            )

    @dist_init
    def test_async_function_wrong_return_type(self):
        self._test_async_function_wrong_return_type(RPCExecMode.SYNC)

    @dist_init
    def test_async_function_wrong_return_type_async(self):
        self._test_async_function_wrong_return_type(RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_wrong_return_type_remote(self):
        self._test_async_function_wrong_return_type(RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_simple(self):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        ret = rpc.rpc_sync(dst1, async_add, args=(dst2, torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    def _test_async_function(self, fn, mode=RPCExecMode.SYNC):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        args = (dst2, torch.ones(2, 2), 1, 2)
        ret = self._run_func_in_mode(dst1, fn, mode, args=args)
        self.assertEqual(ret, torch.ones(2, 2) + 3)

    @dist_init
    def test_async_function_with_future_ctor(self):
        self._test_async_function(async_add_with_future_ctor)

    @dist_init
    def test_async_function_with_future_ctor_remote(self):
        self._test_async_function(
            async_add_with_future_ctor,
            RPCExecMode.REMOTE
        )

    @dist_init
    def test_async_function_chained(self):
        self._test_async_function(async_add_chained)

    @dist_init
    def test_async_function_chained_remote(self):
        self._test_async_function(async_add_chained, RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_nested(self):
        self._test_async_function(async_add_nested)

    @dist_init
    def test_async_function_nested_remote(self):
        self._test_async_function(async_add_nested, RPCExecMode.REMOTE)

    @dist_init
    def test_async_static_method(self):
        self._test_async_function(AsyncExecutionClass.static_async_add)

    @dist_init
    def test_async_static_method_remote(self):
        self._test_async_function(
            AsyncExecutionClass.static_async_add,
            RPCExecMode.REMOTE
        )

    @dist_init
    def test_async_class_method(self):
        self._test_async_function(AsyncExecutionClass.class_async_add)

    @dist_init
    def test_async_class_method_remote(self):
        self._test_async_function(
            AsyncExecutionClass.class_async_add,
            RPCExecMode.REMOTE
        )

    def _test_test_async_class_rref_proxy(self, mode=RPCExecMode.SYNC):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        rref = rpc.remote(dst1, AsyncExecutionClass)

        x = torch.ones(2, 2)
        y = torch.ones(2, 2) + 1
        if mode == RPCExecMode.SYNC:
            ret = rref.rpc_sync().static_async_add(dst2, x, x, y)
            ret += rref.rpc_sync().class_async_add(dst2, x, x, y)
            ret += rref.rpc_sync().bound_async_add(dst2, x, x, y)
        elif mode == RPCExecMode.ASYNC:
            ret = rref.rpc_async().static_async_add(dst2, x, x, y).wait()
            ret += rref.rpc_async().class_async_add(dst2, x, x, y).wait()
            ret += rref.rpc_async().bound_async_add(dst2, x, x, y).wait()
        elif mode == RPCExecMode.REMOTE:
            ret = rref.remote().static_async_add(dst2, x, x, y).to_here()
            ret += rref.remote().class_async_add(dst2, x, x, y).to_here()
            ret += rref.remote().bound_async_add(dst2, x, x, y).to_here()

        self.assertEqual(ret, 3 * 4 * x)

    @dist_init
    def test_async_class_rref_proxy(self):
        self._test_test_async_class_rref_proxy()

    @dist_init
    def test_async_class_rref_proxy_async(self):
        self._test_test_async_class_rref_proxy(mode=RPCExecMode.ASYNC)

    @dist_init
    def test_async_class_rref_proxy_remote(self):
        self._test_test_async_class_rref_proxy(mode=RPCExecMode.REMOTE)

    def _test_async_function_multi(self, fn, mode=RPCExecMode.SYNC):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        num = 20
        step = 3
        args = (dst2, torch.ones(2, 2), num, step)
        ret = self._run_func_in_mode(dst1, fn, mode, args=args)
        self.assertEqual(ret, torch.ones(2, 2) + num * step)

    @dist_init
    def test_async_function_multi_chained(self):
        self._test_async_function_multi(async_add_chained_multi)

    @dist_init
    def test_async_function_multi_chained_async(self):
        self._test_async_function_multi(
            async_add_chained_multi,
            RPCExecMode.ASYNC
        )

    @dist_init
    def test_async_function_multi_chained_remote(self):
        self._test_async_function_multi(
            async_add_chained_multi,
            RPCExecMode.REMOTE
        )

    @dist_init
    def test_async_function_multi_fanout(self):
        self._test_async_function_multi(async_add_multi_fanout)

    @dist_init
    def test_async_function_multi_fanout_async(self):
        self._test_async_function_multi(
            async_add_multi_fanout,
            RPCExecMode.ASYNC
        )

    @dist_init
    def test_async_function_multi_fanout_remote(self):
        self._test_async_function_multi(
            async_add_multi_fanout,
            RPCExecMode.REMOTE
        )

    def _test_return_future(self, mode):
        with self.assertRaisesRegex(
            RuntimeError,
            "Can not pickle torch.futures.Future"
        ):
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size),
                return_future,
                mode
            )

    @dist_init
    def test_return_future(self):
        self._test_return_future(RPCExecMode.SYNC)

    @dist_init
    def test_return_future_async(self):
        self._test_return_future(RPCExecMode.ASYNC)

    @dist_init
    def test_return_future_remote(self):
        self._test_return_future(RPCExecMode.REMOTE)

    @dist_init
    def test_rref_timeout(self):
        # This test is similar to ones in FaultyProcessGroupTest, but is meant to be
        # run with other backends besides ProcessGroup.
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = "worker{}".format(dst_rank)
        # 10 ms timeout
        rref = rpc.remote(dst_worker, my_sleep_func, args=(2, ), timeout=0.01)
        # Future corresponding to the remote creation should time out.
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()
        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()

        wait_until_owners_and_forks_on_rank(1, 1, rank=1)

    @dist_init(setup_rpc=False)
    @unittest.skipIf(
        os.environ.get("RPC_INIT_WITH_TCP", None) == "1",
        "init_pg_then_rpc does not work with TCP init, see https://github.com/pytorch/pytorch/issues/41614."
    )
    def test_init_pg_then_rpc(self):
        dist.init_process_group(
            backend="gloo",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # Test RPC.
        next_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)

        # Test PG
        dist.barrier()

        rpc.shutdown()

    @dist_init(setup_rpc=False)
    @unittest.skipIf(
        os.environ.get("RPC_INIT_WITH_TCP", None) == "1",
        "init_rpc_then_pg does not work with TCP init, see https://github.com/pytorch/pytorch/issues/41614."
    )
    def test_init_rpc_then_pg(self):
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        dist.init_process_group(
            backend="gloo",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Test RPC.
        next_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)

        # Test PG
        dist.barrier()

        rpc.shutdown()

    @dist_init
    def test_wait_all_with_exception(self):
        futs = []
        dst = worker_name((self.rank + 1) % self.world_size)
        for _ in range(10):
            futs.append(rpc.rpc_async(dst, raise_func))

        with self.assertRaisesRegex(ValueError, "Expected error"):
            ret = torch.futures.wait_all(futs)

    @dist_init
    def test_wait_all_with_partial_exception(self):
        futs = []
        dst = worker_name((self.rank + 1) % self.world_size)
        for _ in range(10):
            futs.append(rpc.rpc_async(dst, torch.add, args=(torch.ones(2), 1)))

        futs.append(rpc.rpc_async(dst, raise_func))

        with self.assertRaisesRegex(ValueError, "Expected error"):
            ret = torch.futures.wait_all(futs)

    @dist_init(setup_rpc=False)
    @unittest.skipIf(
        os.environ.get("RPC_INIT_WITH_TCP", None) == "1",
        "Test does not work with TCP init, see https://github.com/pytorch/pytorch/issues/46491",
    )
    def test_init_rpc_twice(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        rpc.shutdown()

        # Wait for all init to complete.
        dist.barrier()

        # Ensure rpc initialization works again.
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # Verify RPCs work after re-init.
        dst = worker_name((self.rank + 1) % self.world_size)
        rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))
        rpc.rpc_sync(dst, foo_add, args=())

        rpc.shutdown()

    def test_wrong_types(self):
        with self.assertRaisesRegex(
            TypeError,
            "Argument backend must be a member of BackendType",
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend="TENSORPIPE",
            )

        with self.assertRaisesRegex(
            TypeError,
            "Argument rpc_backend_options must be an instance of RpcBackendOptions",
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend=self.rpc_backend,
                rpc_backend_options={"init_method": self.init_method}
            )

    def test_cannot_infer_backend_from_options(self):
        # An exception should be raised if the backend isn't specified but
        # options are given which are not an instance of any of the known
        # agents' option classes.
        rpc_backend_options = FooBackendOptions(self.init_method)

        with self.assertRaisesRegex(TypeError, "Could not infer backend for options"):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                # Do _not_ pass backend.
                rpc_backend_options=rpc_backend_options,
            )

    @dist_init
    def test_owner_rref_backward(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        t1 = torch.rand(10, 10, requires_grad=True)
        rref = rpc.RRef(t1.sum() + t1.sum())
        rref.backward()
        expected_grad = torch.ones_like(t1) * 2
        self.assertEqual(expected_grad, t1.grad)

        with dist_autograd.context() as context_id:
            t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
            rref = rpc.RRef(t2.sum())
            rref.backward(context_id)
            self.assertEqual(expected_grad, dist_autograd.get_gradients(context_id)[t1])

        # Double backward.
        with dist_autograd.context() as context_id:
            t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
            rref = rpc.RRef(t2.sum())
            rref.backward(context_id, retain_graph=True)
            rref.backward(context_id)
            self.assertEqual(expected_grad * 2, dist_autograd.get_gradients(context_id)[t1])

        # Test errors.
        with self.assertRaisesRegex(RuntimeError, "tensors does not require grad and does not have a grad_fn"):
            rpc.RRef(torch.rand(10)).backward()

        with self.assertRaisesRegex(RuntimeError, "grad can be implicitly created only for scalar outputs"):
            rpc.RRef(torch.rand(10, requires_grad=True)).backward()

        with self.assertRaisesRegex(RuntimeError, "Could not find autograd context with id: 100"):
            rpc.RRef(torch.rand(10, requires_grad=True).sum()).backward(100)

        with self.assertRaisesRegex(RuntimeError, "RRef should contain a tensor for .backward()"):
            rpc.RRef("foo").backward()

    @staticmethod
    def _sum(x):
        return x.sum()

    @staticmethod
    def _identity(x):
        return x

    @dist_init
    def test_user_rref_backward(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        t = torch.rand(10, requires_grad=True)
        with dist_autograd.context() as context_id:
            rref = rpc.remote(dst, RpcTest._sum, args=(t,))
            rref.backward(context_id, retain_graph=True)
            rref.backward(context_id)
            self.assertEqual(torch.ones_like(t) * 2, dist_autograd.get_gradients(context_id)[t])

        with dist_autograd.context() as context_id:
            rref = rpc.remote(dst, RpcTest._identity, args=("foo",))
            with self.assertRaisesRegex(RuntimeError, "RRef should contain a tensor for .backward()"):
                rref.backward(context_id)

            with self.assertRaisesRegex(RuntimeError, "User RRefs require 'dist_autograd_ctx_id' to be specified"):
                rref.backward()

class ProcessGroupAgentRpcTest(RpcAgentTestFixture):

    def test_mismatched_type_for_options(self):
        # An exception should be raised if the options are not an instance of
        # ProcessGroupRpcBackendOptions.
        rpc_backend_options = FooBackendOptions(self.init_method)

        with self.assertRaisesRegex(
            TypeError, "`rpc_backend_options` must be a `ProcessGroupRpcBackendOptions`"
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend=rpc.BackendType.PROCESS_GROUP,
                rpc_backend_options=rpc_backend_options,
            )

    def test_infer_backend_from_options(self):
        rpc_backend_options = rpc.ProcessGroupRpcBackendOptions(
            init_method=self.init_method
        )

        with self.assertLogs("torch.distributed.rpc", logging.WARNING) as cm:
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                # Do _not_ pass backend.
                rpc_backend_options=rpc_backend_options,
            )
        self.assertIn(
            "To silence this warning pass `backend=BackendType.PROCESS_GROUP` explicitly.",
            "\n".join(cm.output),
        )

        self.assertIsInstance(rpc.api._get_current_rpc_agent(), rpc.ProcessGroupAgent)

    def test_logs_deprecation_warning(self):
        with self.assertLogs("torch.distributed.rpc", logging.WARNING) as cm:
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend=rpc.BackendType.PROCESS_GROUP,
                rpc_backend_options=self.rpc_backend_options,
            )
        self.assertIn(
            "It is recommended to migrate to the TENSORPIPE backend.",
            "\n".join(cm.output),
        )

    @skip_if_lt_x_gpu(2)
    @dist_init
    def test_cuda(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        t1 = torch.rand(3, 3).cuda(0)
        t2 = torch.rand(3, 3).cuda(1)
        t3 = torch.rand(3, 3)

        # cuda tensors as args fail.
        with self.assertRaisesRegex(RuntimeError, "RPC backend only supports CPU tensors.*Found tensor on device: cuda:0"):
            rpc.rpc_sync(dst, torch.add, args=(t1, t2))

        # mix of cpu and cuda tensors as args fail.
        with self.assertRaisesRegex(RuntimeError, "RPC backend only supports CPU tensors.*Found tensor on device: cuda:0"):
            rpc.rpc_sync(dst, torch.add, args=(t1, t3))

        # gpu tensor list as args fails.
        with self.assertRaisesRegex(RuntimeError, "RPC backend only supports CPU tensors.*Found tensor on device: cuda:0"):
            rpc.rpc_sync(dst, RpcTest._gpu_tensor_list_arg, args=([t1, t2]))

        # cuda tensors as return values fail.
        with self.assertRaisesRegex(RuntimeError, "RPC backend only supports CPU tensors.*Found tensor on device: cuda:0"):
            rpc.rpc_sync(dst, RpcTest._return_gpu_tensor, args=())

        # cuda tensors as a list of return value fails
        with self.assertRaisesRegex(RuntimeError, "RPC backend only supports CPU tensors.*Found tensor on device: cuda:0"):
            rpc.rpc_sync(dst, RpcTest._return_gpu_tensor_list, args=())

        # Sending to self should fail too.
        with self.assertRaisesRegex(RuntimeError, "RPC backend only supports CPU tensors.*Found tensor on device: cuda:0"):
            rpc.rpc_sync(worker_name(self.rank), torch.add, args=(t1, t2))


    def test_single_threaded_rref_owner(self):
        # We need a process group in order to perform a barrier at the end.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # This test aims to verify if the server can handle all internal RPC
        # messages using just one thread.
        caller_rank = 0
        callee_rank = 1
        rpc_backend_options = rpc.ProcessGroupRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_send_recv_threads=1
        ) if self.rank == callee_rank else self.rpc_backend_options

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        if self.rank == caller_rank:
            dst = worker_name(callee_rank)
            rrefs = []

            # makes sure there is no existing OwnerRRefs on dst
            info = rpc.rpc_sync(dst, get_rref_debug_info)
            self.assertEqual(0, int(info["num_owner_rrefs"]))

            # creating RRefs on dst
            for i in range(20):
                rrefs.append(
                    rpc.remote(dst, delayed_add, args=(torch.zeros(2, 2), i))
                )

            # using RRefs on dst
            futs = []
            for i in range(len(rrefs)):
                futs.append(
                    rpc.rpc_async(dst, my_rref_function, args=(rrefs[i], rrefs[i]))
                )

            # wait for results and check
            for i in range(len(futs)):
                self.assertEqual(2 * (torch.zeros(2, 2) + i), futs[i].wait())

            # check we created the expected number of RRefs on dst
            info = rpc.rpc_sync(dst, get_rref_debug_info)
            num_owner_rrefs = int(info["num_owner_rrefs"])
            self.assertEqual(len(futs), num_owner_rrefs)

            # trigger RRef deletion
            del futs
            del rrefs

            # wait until OwnerRRefs are cleared on dst
            while num_owner_rrefs > 0:
                info = rpc.rpc_sync(dst, get_rref_debug_info)
                num_owner_rrefs = int(info["num_owner_rrefs"])
                time.sleep(0.01)

        # use a barrier to prevent messages sent during shutdown occupies the
        # only thread on callee (rank == 1) too early.
        dist.barrier()
        rpc.shutdown()

    def test_single_threaded_rref_to_here(self):
        # We need a process group in order to perform a barrier at the end.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # This test aims to verify if the server can handle all internal RPC
        # messages using just one thread.
        caller_rank = 0
        callee_rank = 1
        rpc_backend_options = rpc.ProcessGroupRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_send_recv_threads=1
        ) if self.rank == callee_rank else self.rpc_backend_options

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        if self.rank == caller_rank:
            dst = worker_name(callee_rank)
            rrefs = []

            # makes sure there is no existing OwnerRRefs on dst
            info = rpc.rpc_sync(dst, get_rref_debug_info)
            self.assertEqual(0, int(info["num_owner_rrefs"]))

            # creating RRefs on dst
            for i in range(20):
                rrefs.append(
                    rpc.remote(dst, delayed_add, args=(torch.zeros(2, 2), i))
                )

            # wait for results and check
            for i in range(len(rrefs)):
                self.assertEqual(torch.zeros(2, 2) + i, rrefs[i].to_here())

            # check we created the expected number of RRefs on dst
            info = rpc.rpc_sync(dst, get_rref_debug_info)
            num_owner_rrefs = int(info["num_owner_rrefs"])
            self.assertEqual(len(rrefs), num_owner_rrefs)

            # trigger RRef deletion
            del rrefs

            # wait until OwnerRRefs are cleared on dst
            while num_owner_rrefs > 0:
                info = rpc.rpc_sync(dst, get_rref_debug_info)
                num_owner_rrefs = int(info["num_owner_rrefs"])
                time.sleep(0.01)

        # use a barrier to prevent messages sent during shutdown occupies the
        # only thread on callee (rank == 1) too early.
        dist.barrier()
        rpc.shutdown()

    @dist_init
    def test_process_group_debug_info(self):
        rpc.enable_gil_profiling(True)
        initialize_pg(self.file_init_method, self.rank, self.world_size)
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
            worker_name(dst_rank), set_and_check_done, args=(dst_rank,)
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
    def test_set_and_get_num_send_recv_threads(self):
        NUM_THREADS = 27
        rpc_backend_options = rpc.ProcessGroupRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_send_recv_threads=NUM_THREADS
        )
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertEqual(int(info["agent.thread_pool_size"]), NUM_THREADS)
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_process_group_set_default_timeout(self):
        timeout = 0.5
        rpc_backend_options = rpc.ProcessGroupRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_send_recv_threads=self.rpc_backend_options.num_send_recv_threads,
            rpc_timeout=timeout
        )
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        default_timeout = rpc.get_rpc_timeout()
        self.assertEqual(default_timeout, timeout)
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_process_group_options_throw_on_timedelta_timeout(self):
        from datetime import timedelta

        timeout = timedelta()
        # Ensure that constructing ProcessGroupRpcBackendOptions with timedelta fails
        with self.assertRaisesRegex(TypeError, "incompatible constructor arguments"):
            rpc_backend_options = rpc.ProcessGroupRpcBackendOptions(
                init_method=self.rpc_backend_options.init_method,
                num_send_recv_threads=self.rpc_backend_options.num_send_recv_threads,
                rpc_timeout=timeout,
            )


class FaultyAgentRpcTest(RpcAgentTestFixture):

    # no faulty_messages defined so this fails all retryable messages - see
    # faulty_rpc_agent_test_fixture.py for the list of retryable messages.
    @dist_init(messages_to_delay={})
    def test_check_failed_messages(self):
        if self.rank == 0:
            dst_worker_b = worker_name((self.rank + 1) % self.world_size)
            dst_worker_c = worker_name((self.rank + 2) % self.world_size)

            # Worker0 sends RPC to Worker1 and creates an RRef there
            rref = rpc.remote(dst_worker_b, torch.add, args=(torch.ones(2, 2), torch.ones(2, 2)))
            # Worker0 sends an RPC to Worker2 with the RRef as an arg
            rpc.remote(dst_worker_c, add_rref_to_value, args=(rref, torch.ones(2, 2)))
            # check if the output is as expected
            self.assertEqual(rref.to_here(), torch.add(torch.ones(2, 2), torch.ones(2, 2)))
        # explicitly delete all User RRefs
        _delete_all_user_and_unforked_owner_rrefs()

    @dist_init
    def test_verify_backend_options(self):
        self.assertEqual(self.rpc_backend, rpc.backend_registry.BackendType.FAULTY_PROCESS_GROUP)
        self.assertEqual(self.rpc_backend_options.num_send_recv_threads, 8)
        self.assertEqual(self.rpc_backend_options.num_fail_sends, 3)
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 4)
        self.assertEqual(len(self.rpc_backend_options.messages_to_delay), 2)
        self.assertEqual(self.rpc_backend_options.rpc_timeout, rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=["RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT"])
    def test_custom_faulty_messages(self):
        self.assertEqual(
            set(["RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT"]),
            set(self.rpc_backend_options.messages_to_fail),
        )

    @dist_init(faulty_messages=[])
    def test_no_faulty_messages(self):
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 0)

    @dist_init(messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_custom_messages_to_delay(self):
        self.assertEqual(self.rpc_backend_options.messages_to_delay, {"SCRIPT_CALL": 1.5})

    def _test_remote_message_dropped_pickle(self, dst=None):
        if self.rank != 0:
            return
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = "worker{}".format(dst_rank)
        # Since we fail python_remote_call messages synchronously, the future
        # corresponding to this remote call will be marked with an error when
        # this function returns.
        rref = rpc.remote(dst_worker, my_sleep_func, args=(1,))
        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Attempt to fork the RRef should raise an error indicating the rpc.remote timeout.
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref._serialize()
        # Test that using RRef as arg over RPC (which forks) results in the same
        # error
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 1))

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_remote_message_dropped_pickle(self):
        self._test_remote_message_dropped_pickle()

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_remote_message_dropped_pickle_to_self(self):
        self._test_remote_message_dropped_pickle(self.rank)


    def _test_remote_message_dropped_timeout(self, func, args, dst=None):
        if self.rank != 0:
            return

        # test the case where rpc.remote() message creation is completely dropped.
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = "worker{}".format(dst_rank)
        # Since we fail python_remote_call messages synchronously, the future
        # corresponding to this remote call will be marked with an error when
        # this function returns.
        rref = rpc.remote(dst_worker, func, args=args)
        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()
        # Note: during shutdown, logs will indicate "Could not find OwnerRRef..."
        # on the owning nodes, this is expected because the OwnerRRef was never
        # successfully created. Therefore, delAllUsers will work as expected.

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_builtin_remote_message_dropped_timeout(self):
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_builtin_remote_message_dropped_timeout_to_self(self):
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_udf_remote_message_dropped_timeout(self):
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_dropped_timeout(func, args)

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_udf_remote_message_dropped_timeout_to_self(self):
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    def _test_remote_message_delay_timeout(self, func, args, dst=None):
        if self.rank != 0:
            return
        # Test the case where remote message is eventually processed on the owner,
        # but the future on the creator times out before the response comes back.
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = "worker{}".format(dst_rank)
        # 10 ms timeout
        rref = rpc.remote(dst_worker, func, args=args, timeout=0.001)
        # Future corresponding to the remote creation should time out.
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()

        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # to_here() should now pick up that rpc.remote() creation has failed.
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()

        # Test the case where rpc.remote() times out, but to_here() has already
        # started blocking before.
        # NOTE: we only test this when not sending to self, as to_here() calls
        # calls localValue(), which does not send an RPC and thus does not have
        # a timeout. This can be supported by allowing future.wait() to
        # take in an optional timeout (https://github.com/pytorch/pytorch/issues/39280)
        if dst_rank != self.rank:
            slow_rref = rpc.remote(dst_worker, func, args=args, timeout=2)

            with self.assertRaisesRegex(RuntimeError, expected_error):
                # to_here() should raise timeout error, since it does not know about the
                # status of rpc.remote().
                slow_rref.to_here(0.001)
        # Note: If we proceed with shutdown, UserRRef will send out a RRefUserDelete
        # but this can be a noop since it may not exist on the owner yet. Later,
        # the owner can process the RRef creation and wait for the delete message,
        # thus leading to a timeout.
        # Therefore, we wait until we get notification that pending owners have
        # been confirmed before sending out RRefUserDeletes.
        if dst_rank != self.rank:
            wait_until_owners_and_forks_on_rank(2, 2, rank=dst_rank)

    @dist_init(faulty_messages=[], messages_to_delay={"PYTHON_REMOTE_CALL": 2})
    def test_udf_remote_message_delay_timeout(self):
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(faulty_messages=[], messages_to_delay={"PYTHON_REMOTE_CALL": 2})
    def test_udf_remote_message_delay_timeout_to_self(self):
        func = my_sleep_func
        args = (1,)
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_builtin_delay_timeout(self):
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_builtin_delay_timeout_to_self(self):
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_script_delay_timeout(self):
        func = my_script_func
        args = (torch.tensor(1),)
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_script_delay_timeout_to_self(self):
        func = my_script_func
        args = (torch.tensor(1),)
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_RREF_FETCH_CALL": 1})
    def test_rref_to_here_timeout(self):
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = "worker{}".format(dst_rank)
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref.to_here(0.01)

        rref.to_here()

    @dist_init(faulty_messages=[])
    def test_rpc_builtin_timeout(self):
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = self.get_timeout_error_regex()
        # PYTHON_CALL message types which correspond to Python UDF over RPC
        # by default get a delay (see faulty_rpc_agent_test_fixture)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(
                dst_worker,
                torch.add,
                args=(torch.tensor(1), torch.tensor(1)),
                timeout=1,
            )

        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=1
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure that the currently set default timeout is large enough such
        # that RPCs with delays still complete.
        self.assertEqual(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC, rpc.get_rpc_timeout())
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        fut.wait()

        # Ensure timeout if we set a new default and don't override
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if we specify timeout of 0
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=0
        )
        fut.wait()
        # Reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_rpc_script_timeout(self):
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)

        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure that the currently set default timeout is large enough such
        # that RPCs with delays still complete.
        self.assertEqual(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC, rpc.get_rpc_timeout())
        fut = rpc.rpc_async(
            dst_worker, my_script_func, args=(torch.tensor(1),)
        )
        fut.wait()

        # Ensure timeout if we set a new default and don't override
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(
            dst_worker, my_script_func, args=(torch.tensor(1),)
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if we specify timeout of 0
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(
            dst_worker, my_script_func, args=(torch.tensor(1),), timeout=0
        )
        fut.wait()
        # Reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

class TensorPipeAgentRpcTest(RpcAgentTestFixture):

    def test_mismatched_type_for_options(self):
        # An exception should be raised if the options are not an instance of
        # TensorPipeRpcBackendOptions.
        rpc_backend_options = FooBackendOptions(self.init_method)

        with self.assertRaisesRegex(
            TypeError, "`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`"
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend=rpc.BackendType.TENSORPIPE,
                rpc_backend_options=rpc_backend_options,
            )

    def test_infer_backend_from_options(self):
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=self.init_method
        )

        rpc.init_rpc(
            name=worker_name(self.rank),
            rank=self.rank,
            world_size=self.world_size,
            # Do _not_ pass backend.
            rpc_backend_options=rpc_backend_options,
        )

        self.assertIsInstance(rpc.api._get_current_rpc_agent(), rpc.TensorPipeAgent)

    # FIXME Merge this test with the corresponding one in RpcTest.
    @dist_init(setup_rpc=False)
    def test_set_and_get_num_worker_threads(self):
        NUM_THREADS = 27
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_worker_threads=NUM_THREADS
        )
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertEqual(int(info["agent.thread_pool_size"]), NUM_THREADS)
        rpc.shutdown()

    # FIXME Merge this test with the corresponding one in RpcTest.
    @dist_init(setup_rpc=False)
    def test_tensorpipe_set_default_timeout(self):
        timeout = 0.5
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_worker_threads=self.rpc_backend_options.num_worker_threads,
            rpc_timeout=timeout
        )
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        default_timeout = rpc.get_rpc_timeout()
        self.assertEqual(default_timeout, timeout)
        rpc.shutdown()

    # FIXME Merge this test with the corresponding one in RpcTest.
    @dist_init(setup_rpc=False)
    def test_tensorpipe_options_throw_on_timedelta_timeout(self):
        from datetime import timedelta

        timeout = timedelta()
        # Ensure that constructing TensorPipeRpcBackendOptions with timedelta fails
        with self.assertRaisesRegex(TypeError, "incompatible constructor arguments"):
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                init_method=self.rpc_backend_options.init_method,
                num_worker_threads=self.rpc_backend_options.num_worker_threads,
                rpc_timeout=timeout,
            )

    def _test_device_maps(self, options, errMsg="Invalid device_map"):
        with self.assertRaisesRegex(ValueError, errMsg):
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options,
            )

        self.assertFalse(rpc.api._is_current_rpc_agent_set())

    @skip_if_lt_x_gpu(2)
    def test_device_maps_wrong_worker_name(self):
        options = self.rpc_backend_options
        options.set_device_map("none_exist", {0: 1})
        self._test_device_maps(options, "Wrong worker names")

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_max_local_device(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {torch.cuda.device_count(): 0})

        self._test_device_maps(options)

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_max_remote_device(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {0: torch.cuda.device_count()})

        self._test_device_maps(options)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_many_to_one(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {1: 0})
        options.set_device_map(dst, {0: 0})

        self._test_device_maps(options)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_one_to_many(self):
        if self.rank == 0:
            options = self.rpc_backend_options
            dst = worker_name((self.rank + 1) % self.world_size)
            options.set_device_map(dst, {0: 1})
            with self.assertRaisesRegex(
                ValueError, "`set_device_map` only supports 1-to-1 mapping"
            ):
                options.set_device_map(dst, {0: 0})

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_min_device(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(
            RuntimeError, "Device index must not be negative"
        ):
            options.set_device_map(dst, {-1: 0})

        with self.assertRaisesRegex(
            RuntimeError, "Device index must not be negative"
        ):
            options.set_device_map(dst, {0: -1})

    @staticmethod
    def _gpu_add(x, y):
        if all([x.is_cuda, x.device.index == 1, y.is_cuda, y.device.index == 1]):
            return (x + y).to(0)
        else:
            raise ValueError("Wrong device affinity")

    @skip_if_lt_x_gpu(2)
    def test_device_maps_gpu(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {0: 1, 1: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentRpcTest._gpu_add,
            args=(torch.zeros(2).to(0), torch.ones(2).to(0))
        )
        self.assertEqual(ret.device, torch.device(1))
        self.assertEqual(ret, (torch.zeros(2) + torch.ones(2)).to(1))
        rpc.shutdown()

    @staticmethod
    def _gpu_add_given_gpus(x, y, x_to, y_to, z_to):
        if all([
            x.is_cuda,
            x.device.index == x_to,
            y.is_cuda,
            y.device.index == y_to
        ]):
            return x.to(z_to) + y.to(z_to)
        else:
            raise ValueError("Wrong device affinity")

    def _test_device_maps_gpu(self, x_from, y_from, z_to, device_map, dst=None):
        x_to = device_map[x_from]
        y_to = device_map[y_from]

        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size) if dst is None else dst
        options.set_device_map(dst, device_map)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        x = torch.zeros(2).to(x_from)
        y = torch.ones(2).to(y_from)

        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentRpcTest._gpu_add_given_gpus,
            args=(x, y, x_to, y_to, z_to)
        )

        reverse_device_map = {device_map[k] : k for k in device_map}
        z_from = reverse_device_map[z_to]

        self.assertEqual(ret.device.index, z_from)
        self.assertEqual(ret, torch.ones(2).to(z_from))

        rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_default(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=0,
            z_to=0,
            device_map={0 : 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_non_default(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=1,
            z_to=1,
            device_map={1 : 1}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_default_to_non_default(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=0,
            z_to=1,
            device_map={0 : 1}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_non_default_to_default(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=1,
            z_to=0,
            device_map={1 : 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_1(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=0,
            device_map={0 : 0, 1 : 1}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_2(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=1,
            device_map={0 : 0, 1 : 1}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_3(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=0,
            device_map={0 : 0, 1 : 1}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_4(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=1,
            device_map={0 : 0, 1 : 1}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_5(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=0,
            device_map={0 : 1, 1 : 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_6(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=1,
            device_map={0 : 1, 1 : 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_7(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=0,
            device_map={0 : 1, 1 : 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_8(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=1,
            device_map={0 : 1, 1 : 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_1(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=0,
            device_map={0 : 0, 1 : 1},
            dst=worker_name(self.rank)
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_2(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=1,
            device_map={0 : 0, 1 : 1},
            dst=worker_name(self.rank)
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_3(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=0,
            device_map={0 : 0, 1 : 1},
            dst=worker_name(self.rank)
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_4(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=1,
            device_map={0 : 0, 1 : 1},
            dst=worker_name(self.rank)
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_5(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=0,
            device_map={0 : 1, 1 : 0},
            dst=worker_name(self.rank)
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_6(self):
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=1,
            device_map={0 : 1, 1 : 0},
            dst=worker_name(self.rank)
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_7(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=0,
            device_map={0 : 1, 1 : 0},
            dst=worker_name(self.rank)
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_8(self):
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=1,
            device_map={0 : 1, 1 : 0},
            dst=worker_name(self.rank)
        )

    @staticmethod
    def _gpu_add_multi_gpu(x, y):
        if all([x.is_cuda, x.device.index == 1, y.is_cuda, y.device.index == 0]):
            return x.to(0) + y, x - y.to(1)
        else:
            raise ValueError("Wrong device affinity")

    def _test_device_maps_multi_gpu(self, dst):
        options = self.rpc_backend_options
        options.set_device_map(dst, {0: 1})
        options.set_device_map(dst, {1: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        x = torch.zeros(2).to(0)
        y = torch.ones(2).to(1)
        rets = rpc.rpc_sync(
            dst,
            TensorPipeAgentRpcTest._gpu_add_multi_gpu,
            args=(x, y)
        )

        self.assertEqual(rets[0].device, torch.device(1))
        self.assertEqual(rets[1].device, torch.device(0))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(1))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_device_maps_multi_gpu(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._test_device_maps_multi_gpu(dst)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_multi_gpu_self(self):
        dst = worker_name(self.rank)
        self._test_device_maps_multi_gpu(dst)

    @staticmethod
    def _gpu_add_return_to_gpu(x, y):
        if x.device.type == 'cpu' and y.device.type == 'cpu':
            return (x + y).to(0), (x - y).to(1), (x * y).to(2), (x / y).to(3)
        else:
            raise ValueError("Wrong device affinity")

    @skip_if_lt_x_gpu(2)
    def test_device_maps_in_options(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=options.init_method,
                num_worker_threads=options.num_worker_threads,
                device_maps={dst: {0: 1, 1: 0}}
            )
        )

        rets = rpc.rpc_sync(
            dst,
            TensorPipeAgentRpcTest._gpu_add_multi_gpu,
            args=(torch.zeros(2).to(0), torch.ones(2).to(1))
        )
        self.assertEqual(rets[0].device, torch.device(1))
        self.assertEqual(rets[1].device, torch.device(0))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(1))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        rpc.shutdown()

    def _test_device_maps_return_to_gpu(self, dst):
        options = self.rpc_backend_options

        options.set_device_map(dst, {0: 1})
        options.set_device_map(dst, {1: 2})
        options.set_device_map(dst, {2: 3})
        options.set_device_map(dst, {3: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        rets = rpc.rpc_sync(
            dst,
            TensorPipeAgentRpcTest._gpu_add_return_to_gpu,
            args=(torch.zeros(2), torch.ones(2))
        )
        for i in range(len(rets)):
            self.assertEqual(rets[i].device, torch.device((3 + i) % 4))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(3))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        self.assertEqual(rets[2], (torch.zeros(2) * torch.ones(2)).to(1))
        self.assertEqual(rets[3], (torch.zeros(2) / torch.ones(2)).to(2))
        rpc.shutdown()

    @skip_if_lt_x_gpu(4)
    def test_device_maps_return_to_gpu(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._test_device_maps_return_to_gpu(dst)

    @skip_if_lt_x_gpu(4)
    def test_device_maps_return_to_gpu_self(self):
        dst = worker_name(self.rank)
        self._test_device_maps_return_to_gpu(dst)

    @staticmethod
    def _add_to_gpu(x, y):
        return (x + y).to(0)

    def _test_device_maps_missing_config(self, mode):
        dst = worker_name((self.rank + 1) % self.world_size)
        errMsg = (
            "TensorPipe RPC backend only supports CPU tensors by default.*"
            "`set_device_map` on `TensorPipeRpcBackendOptions`"
        )

        with self.assertRaisesRegex(RuntimeError, errMsg):
            if mode == RPCExecMode.SYNC:
                rpc.rpc_sync(dst, torch.add, args=(torch.zeros(2).to(0), 1))
            elif mode == RPCExecMode.REMOTE:
                rpc.remote(dst, torch.add, args=(torch.zeros(2).to(0), 1)).to_here()
            else:
                raise ValueError(f"unexpected mode {mode}")

        # make sure RPC is still functioning
        ret = rpc.rpc_sync(dst, torch.add, args=(torch.ones(2), 1))
        self.assertEqual(ret, torch.ones(2) + 1)

    def _test_device_maps_missing_config_response(self, mode):
        dst = worker_name((self.rank + 1) % self.world_size)
        errMsg = "Response device mapping is not available"

        with self.assertRaisesRegex(RuntimeError, errMsg):
            if mode == RPCExecMode.SYNC:
                rpc.rpc_sync(
                    dst,
                    TensorPipeAgentRpcTest._add_to_gpu,
                    args=(torch.zeros(2), 1)
                )
            elif mode == RPCExecMode.REMOTE:
                rpc.remote(
                    dst,
                    TensorPipeAgentRpcTest._add_to_gpu,
                    args=(torch.zeros(2), 1)
                ).to_here()
            else:
                raise ValueError(f"unexpected mode {mode}")

        # make sure RPC is still functioning
        ret = rpc.rpc_sync(dst, torch.add, args=(torch.ones(2), 1))
        self.assertEqual(ret, torch.ones(2) + 1)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config(self):
        self._test_device_maps_missing_config(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    def test_device_maps_missing_config_not_timeout(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options
        )

        timeout = rpc.get_rpc_timeout()

        tik = time.time()
        self._test_device_maps_missing_config(RPCExecMode.SYNC)
        rpc.shutdown()
        tok = time.time()

        self.assertTrue(tok - tik < timeout)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_loop(self):
        for _ in range(self.rpc_backend_options.num_worker_threads + 5):
            self._test_device_maps_missing_config(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_response(self):
        self._test_device_maps_missing_config_response(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_response_loop(self):
        for _ in range(self.rpc_backend_options.num_worker_threads + 5):
            self._test_device_maps_missing_config_response(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_remote(self):
        self._test_device_maps_missing_config(RPCExecMode.REMOTE)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_remote_response(self):
        self._test_device_maps_missing_config_response(RPCExecMode.REMOTE)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_remote(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {1: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        rref = rpc.remote(
            dst,
            TensorPipeAgentRpcTest._add_to_gpu,
            args=(torch.zeros(2), 1)
        )

        self.assertEqual(rref.to_here().device.index, 1)
        self.assertEqual(rref.to_here(), torch.ones(2).to(1))

        rpc.shutdown()

    @staticmethod
    def _slow_add_on_user_stream(x, y):
        s0 = torch.cuda.current_stream(x.device)
        s1 = torch.cuda.Stream(device=x.device)
        with torch.cuda.stream(s1):
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            s1.wait_stream(s0)
            z = x + y
            event = torch.cuda.Event()
            event.record(s1)
        event.wait(s0)
        return z

    def _test_custom_stream(self, fn, device_map):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, device_map)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        fn(dst)

        rpc.shutdown()

    def _test_stream_sync(self, dst):
        x = torch.ones(2, 2).to(0)
        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentRpcTest._slow_add_on_user_stream,
            args=(x, x)
        )
        self.assertEqual(ret, 2 * x)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream(self):
        self._test_custom_stream(self._test_stream_sync, {"cuda:0": "cuda:1"})

    def _test_stream_multi_async(self, dst):
        futs = []
        for i in range(20):
            x = torch.ones(2, 2).to(0) * i
            futs.append(
                rpc.rpc_async(
                    dst,
                    TensorPipeAgentRpcTest._slow_add_on_user_stream,
                    args=(x, x)
                )
            )

        for i in range(20):
            self.assertEqual(futs[i].wait(), 2 * torch.ones(2, 2).to(0) * i)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_multi(self):
        self._test_custom_stream(
            self._test_stream_multi_async,
            {"cuda:0": "cuda:1"}
        )

    @staticmethod
    def _nested_slow_add_on_user_stream(dst, x, y, z):
        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentRpcTest._slow_add_on_user_stream,
            args=(x, y)
        )

        return TensorPipeAgentRpcTest._slow_add_on_user_stream(ret, z)

    def _test_stream_nested_sync(self, dst):
        x = torch.ones(2, 2).to(0)
        y = torch.ones(2, 2).to(0) * 2
        z = torch.ones(2, 2).to(0) * 3
        nested_dst = worker_name((self.rank + 2) % self.world_size)
        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentRpcTest._nested_slow_add_on_user_stream,
            args=(nested_dst, x, y, z)
        )
        self.assertEqual(ret, 6 * x)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_nested(self):
        self._test_custom_stream(
            self._test_stream_nested_sync,
            {"cuda:0": "cuda:1", "cuda:1": "cuda:0"}
        )

    def _test_stream_nested_multi_async(self, dst):
        if self.rank == 0:
            futs = []
            n = 5
            xs, ys, zs = [], [], []
            for i in range(n):
                x = torch.ones(2, 2).to(0) * (i - 1)
                y = torch.ones(2, 2).to(0) * i
                z = torch.ones(2, 2).to(0) * (i + 1)
                xs.append(x)
                ys.append(y)
                zs.append(z)
                nested_dst = worker_name((self.rank + 2) % self.world_size)
                futs.append(
                    rpc.rpc_async(
                        dst,
                        TensorPipeAgentRpcTest._nested_slow_add_on_user_stream,
                        args=(nested_dst, x, y, z)
                    )
                )

            for i in range(n):
                self.assertEqual(futs[i].wait(), xs[i] + ys[i] + zs[i])

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_nested_multi(self):
        self._test_custom_stream(
            self._test_stream_nested_multi_async,
            {"cuda:0": "cuda:1", "cuda:1": "cuda:0"}
        )

    @dist_init
    def _test_rref_get_type_timeout(self, blocking):
        # Test where we try to get the type of a RRef from an owner, but RRef
        # creation is slower than timeout passed into _get_type.
        dst_rank = (self.rank + 1) % self.world_size
        dst = worker_name(dst_rank)
        slow_rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), True))
        timeout = 0.5
        expected_err = self.get_timeout_error_regex()
        # Blocking: blocks on inline call
        if blocking:
            with self.assertRaisesRegex(RuntimeError, expected_err):
                slow_rref._get_type(timeout=timeout, blocking=blocking)
        # Non-blocking: blocks on wait
        else:
            fut = slow_rref._get_type(timeout=timeout, blocking=blocking)
            with self.assertRaisesRegex(RuntimeError, expected_err):
                fut.wait()

    def test_rref_get_type_timeout_blocking(self):
        self._test_rref_get_type_timeout(blocking=True)

    def test_rref_get_type_timeout_non_blocking(self):
        self._test_rref_get_type_timeout(blocking=False)

    @dist_init
    def test_op_with_invalid_args(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(
            RuntimeError, "Overloaded torch operator invoked from Python failed to many any schema"
        ):
            rpc.rpc_sync(dst, torch.add, args=())

    def _test_rref_proxy_timeout(self, rref_proxy_api):
        dst_rank = (self.rank + 1) % self.world_size
        dst = worker_name(dst_rank)
        rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), ))
        # Ensure RRef is created on remote node.
        rref.to_here()
        rref_api = getattr(rref, rref_proxy_api)
        self.assertTrue(rref_api is not None, f"Failed to get RRef proxy api: {rref_proxy_api}")
        expected_error = self.get_timeout_error_regex()
        timeout = 2
        with self.assertRaisesRegex(RuntimeError, expected_error):
            result = rref_api(timeout=timeout).my_slow_method(torch.ones(2, 2))
            if rref_api == rref.rpc_async:
                result.wait()
            elif rref_api == rref.remote:
                result._get_future().wait()

        # Case where rpc.remote() is stuck and exceeds timeout
        slow_rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), True))
        timeout = 0.01
        rref_api = getattr(slow_rref, rref_proxy_api)
        # Note that even when we call rref.rpc_async() in this case, we
        # time out in future creation, not waiting for future. This is because
        # rref proxy function calls rref._get_type before returning future,
        # which blocks on the RRef being created on owner node, until the
        # specified timeout.
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref_api(timeout=timeout).my_instance_method(torch.ones(2, 2))

    @dist_init
    def test_rref_proxy_timeout(self):
        for rpc_api in ["rpc_sync", "rpc_async", "remote"]:
            self._test_rref_proxy_timeout(rpc_api)
