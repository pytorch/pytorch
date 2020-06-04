import concurrent.futures
import sys
import time
import unittest
from collections import namedtuple
from functools import partial
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info
from torch.distributed.rpc.api import _delete_all_user_rrefs, _use_rpc_pickler
from torch.distributed.rpc.internal import (
    PythonUDF,
    RPCExecMode,
    _internal_rpc_pickler,
    _build_rpc_profiling_key,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import IS_MACOS, load_tests
from torch.testing._internal.dist_utils import (
    dist_init,
    get_function_event,
    get_shutdown_error_regex,
    get_timeout_error_regex,
    initialize_pg,
    wait_until_node_failure,
    wait_until_pending_futures_and_users_flushed,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.distributed.rpc.faulty_rpc_agent_test_fixture import (
    FaultyRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
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


def raise_func():
    raise ValueError("Expected error")


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


@rpc.functions.async_execution
def async_add_with_future_ctor(to, x, y, z):
    fut = torch.futures.Future()
    rpc.rpc_async(to, torch.add, args=(x, y)).then(
        lambda fut1: fut.set_result(fut1.wait() + z)
    ).wait()
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
    import threading
    lock = threading.Lock()
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


def return_future():
    return torch.futures.Future()


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class RpcTest(RpcAgentTestFixture):
    def _skip_if_tensorpipe_agent(old_func):  # noqa: B902
        def decorator(self):
            return unittest.skipIf(
                self.rpc_backend == rpc.backend_registry.BackendType.TENSORPIPE,
                "This test is not yet supported in the Tensorpipe Agent"
            )(old_func)

        return decorator

    @requires_process_group_agent("PROCESS_GROUP rpc backend specific test, skip")
    def test_world_size_one(self):
        if self.rank == 0:
            rpc.init_rpc(
                name="me",
                backend=self.rpc_backend,
                rank=0,
                world_size=1,
                rpc_backend_options=self.rpc_backend_options,
            )

            expect = torch.ones(4, 4) * 2
            result = rpc.remote(
                "me",
                my_tensor_function,
                args=(torch.ones(4, 4), torch.ones(4, 4))
            ).to_here()
            self.assertEqual(expect, result)

            rpc.shutdown()


class FaultyAgentRpcTest(FaultyRpcAgentTestFixture):

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
        _delete_all_user_rrefs()

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

    @dist_init(faulty_messages=[])
    def test_rpc_builtin_timeout(self):
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = get_timeout_error_regex(
            dist_utils.TEST_CONFIG.rpc_backend_name
        )
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
        expected_error = get_timeout_error_regex(dist_utils.TEST_CONFIG.rpc_backend_name)
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

class TensorPipeAgentRpcTest(TensorPipeRpcAgentTestFixture, RpcTest):

    @dist_init
    def test_verify_backend_options(self):
        self.assertEqual(self.rpc_backend, rpc.backend_registry.BackendType.TENSORPIPE)
