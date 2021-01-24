#!/usr/bin/env python3
import os
import unittest
from enum import Flag, auto
from typing import Dict, List, Type

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
    find_free_port,
)
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
    DdpComparisonTest,
    DdpUnderDistAutogradTest,
)
from torch.testing._internal.distributed.pipe_with_ddp_test import (
    PipeWithDDPTest,
)
from torch.testing._internal.distributed.nn.api.remote_module_test import (
    RemoteModuleTest,
)
from torch.testing._internal.distributed.rpc.dist_autograd_test import (
    DistAutogradTest,
    FaultyAgentDistAutogradTest,
)
from torch.testing._internal.distributed.rpc.dist_optimizer_test import (
    DistOptimizerTest,
)
from torch.testing._internal.distributed.rpc.jit.dist_autograd_test import (
    JitDistAutogradTest,
)
from torch.testing._internal.distributed.rpc.jit.rpc_test import JitRpcTest
from torch.testing._internal.distributed.rpc.jit.rpc_test_faulty import (
    JitFaultyAgentRpcTest,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc.rpc_test import (
    FaultyAgentRpcTest,
    ProcessGroupAgentRpcTest,
    RpcTest,
    TensorPipeAgentRpcTest,
)
from torch.testing._internal.distributed.rpc.examples.batch_update_parameter_server_test import BatchUpdateParameterServerTest
from torch.testing._internal.distributed.rpc.examples.parameter_server_test import ParameterServerTest
from torch.testing._internal.distributed.rpc.examples.rpc_rnn import RPCRNNTest

def _check_and_set_tcp_init():
    # if we are running with TCP init, set main address and port
    # before spawning subprocesses, since different processes could find
    # different ports.
    use_tcp_init = os.environ.get("RPC_INIT_WITH_TCP", None)
    if use_tcp_init == "1":
        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(find_free_port())

def _check_and_unset_tcp_init():
    use_tcp_init = os.environ.get("RPC_INIT_WITH_TCP", None)
    if use_tcp_init == "1":
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]

# The tests for the RPC module need to cover multiple possible combinations:
# - different aspects of the API, each one having its own suite of tests;
# - different agents (ProcessGroup, TensorPipe, ...);
# - and subprocesses launched with either fork or spawn.
# To avoid a combinatorial explosion in code size, and to prevent forgetting to
# add a combination, these are generated automatically by the code in this file.
# Here, we collect all the test suites that we need to cover and the two multi-
# processing methods. We then have one separate file for each agent, from which
# we call the generate_tests function of this file, passing to it a fixture for
# the agent, which then gets mixed-in with each test suite and each mp method.


@unittest.skipIf(TEST_WITH_TSAN, "TSAN and fork() is broken")
class ForkHelper(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        _check_and_set_tcp_init()
        self._fork_processes()

    def tearDown(self):
        _check_and_unset_tcp_init()
        super().tearDown()

@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class SpawnHelper(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        _check_and_set_tcp_init()
        self._spawn_processes()

    def tearDown(self):
        _check_and_unset_tcp_init()
        super().tearDown()


class MultiProcess(Flag):
    FORK = auto()
    SPAWN = auto()


MP_HELPERS_AND_SUFFIXES = {
    MultiProcess.FORK: (ForkHelper, "WithFork"),
    MultiProcess.SPAWN: (SpawnHelper, "WithSpawn"),
}


# This list contains test suites that are agent-agnostic and that only verify
# compliance with the generic RPC interface specification. These tests should
# *not* make use of implementation details of a specific agent (options,
# attributes, ...). These test suites will be instantiated multiple times, once
# for each agent (except the faulty agent, which is special).
GENERIC_TESTS = [
    RpcTest,
    BatchUpdateParameterServerTest,
    ParameterServerTest,
    RPCRNNTest,
    DistAutogradTest,
    DistOptimizerTest,
    JitRpcTest,
    JitDistAutogradTest,
    RemoteModuleTest,
    DdpUnderDistAutogradTest,
    DdpComparisonTest,
    PipeWithDDPTest,
]


# This list contains test suites that will only be run on the ProcessGroupAgent.
# These suites should be standalone, and separate from the ones in the generic
# list (not subclasses of those!).
PROCESS_GROUP_TESTS = [
    ProcessGroupAgentRpcTest
]


# This list contains test suites that will only be run on the TensorPipeAgent.
# These suites should be standalone, and separate from the ones in the generic
# list (not subclasses of those!).
TENSORPIPE_TESTS = [
    TensorPipeAgentRpcTest
]


# This list contains test suites that will only be run on the faulty RPC agent.
# That agent is special as it's only used to perform fault injection in order to
# verify the error handling behavior. Thus the faulty agent will only run the
# suites in this list, which were designed to test such behaviors, and not the
# ones in the generic list.
FAULTY_AGENT_TESTS = [
    FaultyAgentRpcTest,
    FaultyAgentDistAutogradTest,
    JitFaultyAgentRpcTest,
]


def generate_tests(
    prefix: str,
    mixin: Type[RpcAgentTestFixture],
    tests: List[Type[RpcAgentTestFixture]],
    mp_type_filter: MultiProcess,
    module_name: str,
) -> Dict[str, Type[RpcAgentTestFixture]]:
    """Mix in the classes needed to autogenerate the tests based on the params.

    Takes a series of test suites, each written against a "generic" agent (i.e.,
    derived from the abstract RpcAgentTestFixture class), as the `tests` args.
    Takes a concrete subclass of RpcAgentTestFixture, which specializes it for a
    certain agent, as the `mixin` arg. Produces all combinations of them, and of
    the multiprocessing start methods (fork or spawn), possibly filtered using
    the `mp_type_filter`. Returns a dictionary of class names to class type
    objects which can be inserted into the global namespace of the calling
    module. The name of each test will be a concatenation of the `prefix` arg,
    the original name of the test suite, and a suffix of either `WithFork` or
    `WithSpawn`. The `module_name` should be the name of the calling module so
    that the classes can be fixed to make it look like they belong to it, which
    is necessary for pickling to work on them.
    """
    ret: Dict[str, Type[RpcAgentTestFixture]] = {}
    for test_class in tests:
        for mp_type in MultiProcess:
            if mp_type & mp_type_filter:
                mp_helper, suffix = MP_HELPERS_AND_SUFFIXES[mp_type]
                name = f"{prefix}{test_class.__name__}{suffix}"
                class_ = type(name, (test_class, mixin, mp_helper), dict())
                class_.__module__ = module_name
                ret[name] = class_
    return ret
