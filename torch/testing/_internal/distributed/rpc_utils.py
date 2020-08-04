#!/usr/bin/env python3
import unittest
from enum import Flag, auto
from typing import Dict, List, NamedTuple, Type

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
    DdpComparisonTest,
    DdpUnderDistAutogradTest,
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
        self._fork_processes()


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class SpawnHelper(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()


class MultiProcess(Flag):
    FORK = auto()
    SPAWN = auto()


MP_HELPERS_AND_SUFFIXES = {
    MultiProcess.FORK: (ForkHelper, "WithFork"),
    MultiProcess.SPAWN: (SpawnHelper, "WithSpawn"),
}


GENERIC_TESTS = [
    RpcTest,
    DistAutogradTest,
    DistOptimizerTest,
    JitRpcTest,
    JitDistAutogradTest,
    RemoteModuleTest,
    DdpUnderDistAutogradTest,
    DdpComparisonTest,
]


PROCESS_GROUP_TESTS = [
    ProcessGroupAgentRpcTest
]


TENSORPIPE_TESTS = [
    TensorPipeAgentRpcTest
]


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
