from __future__ import absolute_import, division, print_function, unicode_literals

import concurrent.futures
from datetime import timedelta
import sys
import time
import unittest
from collections import namedtuple
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.testing._internal.common_utils import load_tests, IS_MACOS
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info
import torch.testing._internal.dist_utils
from torch.testing._internal.dist_utils import dist_init, wait_until_node_failure, initialize_pg
from torch.distributed.rpc.api import _use_rpc_pickler
from torch.distributed.rpc.internal import PythonUDF, _internal_rpc_pickler, RPCExecMode
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
from torch._jit_internal import _qualified_name


def requires_process_group_agent(message=""):
    def decorator(old_func):
        return unittest.skipUnless(
            torch.testing._internal.dist_utils.TEST_CONFIG.rpc_backend_name == "PROCESS_GROUP", message
        )(old_func)

    return decorator


VALUE_FUTURE = concurrent.futures.Future()
DONE_FUTURE = concurrent.futures.Future()


class StubRpcAgent:
    def __init__(self, world_size):
        self.world_size = world_size

    def get_worker_infos(self):
        return {
            rpc.WorkerInfo(
                name="worker{}".format(rank),
                id=rank,
            ) for rank in range(self.world_size)
        }


def _stub_construct_rpc_backend_options_handler(
    **kwargs
):
    return mock.Mock()  # RpcBackendOptions.


def _stub_start_rpc_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    return StubRpcAgent(world_size=world_size)


def set_value(value):
    VALUE_FUTURE.set_result(value)


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


def raise_func():
    raise ValueError("Expected error")

global_rref = None

def set_global_rref(rref):
    global global_rref
    global_rref = rref

def clear_global_rref():
    global global_rref
    global_rref = None


@torch.jit.script
class MyScriptClass:
    def __init__(self):
        self.a = 10


class MyScriptModule(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.a = 10

    @torch.jit.script_method
    def my_method(self):
        self.a = 11


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


@unittest.skipIf(
    sys.version_info < (3, 0),
    "Pytorch distributed rpc package " "does not support python2",
)
class RpcTest(RpcAgentTestFixture):
    @dist_init
    def test_nested_rref_stress(self):
        # if self.rank != 0:
        #     return

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

        print("A got children forks.")
