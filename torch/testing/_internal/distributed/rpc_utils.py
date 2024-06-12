# mypy: ignore-errors

import os
import sys
import unittest
from typing import Dict, List, Type

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    find_free_port,
    IS_SANDCASTLE,
)
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
    CudaDdpComparisonTest,
    DdpComparisonTest,
    DdpUnderDistAutogradTest,
)
from torch.testing._internal.distributed.nn.api.remote_module_test import (
    CudaRemoteModuleTest,
    RemoteModuleTest,
    ThreeWorkersRemoteModuleTest,
)
from torch.testing._internal.distributed.rpc.dist_autograd_test import (
    DistAutogradTest,
    CudaDistAutogradTest,
    FaultyAgentDistAutogradTest,
    TensorPipeAgentDistAutogradTest,
    TensorPipeCudaDistAutogradTest
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
from torch.testing._internal.distributed.rpc.faulty_agent_rpc_test import (
    FaultyAgentRpcTest,
)
from torch.testing._internal.distributed.rpc.rpc_test import (
    CudaRpcTest,
    RpcTest,
    TensorPipeAgentRpcTest,
    TensorPipeAgentCudaRpcTest,
)
from torch.testing._internal.distributed.rpc.examples.parameter_server_test import ParameterServerTest
from torch.testing._internal.distributed.rpc.examples.reinforcement_learning_rpc_test import (
    ReinforcementLearningRpcTest,
)


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
# To avoid a combinatorial explosion in code size, and to prevent forgetting to
# add a combination, these are generated automatically by the code in this file.
# Here, we collect all the test suites that we need to cover.
# We then have one separate file for each agent, from which
# we call the generate_tests function of this file, passing to it a fixture for
# the agent, which then gets mixed-in with each test suite.

@unittest.skipIf(
    TEST_WITH_DEV_DBG_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class SpawnHelper(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        _check_and_set_tcp_init()
        self._spawn_processes()

    def tearDown(self):
        _check_and_unset_tcp_init()
        super().tearDown()


# This list contains test suites that are agent-agnostic and that only verify
# compliance with the generic RPC interface specification. These tests should
# *not* make use of implementation details of a specific agent (options,
# attributes, ...). These test suites will be instantiated multiple times, once
# for each agent (except the faulty agent, which is special).
GENERIC_TESTS = [
    RpcTest,
    ParameterServerTest,
    DistAutogradTest,
    DistOptimizerTest,
    JitRpcTest,
    JitDistAutogradTest,
    RemoteModuleTest,
    ThreeWorkersRemoteModuleTest,
    DdpUnderDistAutogradTest,
    DdpComparisonTest,
    ReinforcementLearningRpcTest,
]
GENERIC_CUDA_TESTS = [
    CudaRpcTest,
    CudaDistAutogradTest,
    CudaRemoteModuleTest,
    CudaDdpComparisonTest,
]


# This list contains test suites that will only be run on the TensorPipeAgent.
# These suites should be standalone, and separate from the ones in the generic
# list (not subclasses of those!).
TENSORPIPE_TESTS = [
    TensorPipeAgentRpcTest,
    TensorPipeAgentDistAutogradTest,
]
TENSORPIPE_CUDA_TESTS = [
    TensorPipeAgentCudaRpcTest,
    TensorPipeCudaDistAutogradTest,
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
    module_name: str,
) -> Dict[str, Type[RpcAgentTestFixture]]:
    """Mix in the classes needed to autogenerate the tests based on the params.

    Takes a series of test suites, each written against a "generic" agent (i.e.,
    derived from the abstract RpcAgentTestFixture class), as the `tests` args.
    Takes a concrete subclass of RpcAgentTestFixture, which specializes it for a
    certain agent, as the `mixin` arg. Produces all combinations of them.
    Returns a dictionary of class names to class type
    objects which can be inserted into the global namespace of the calling
    module. The name of each test will be a concatenation of the `prefix` arg
    and the original name of the test suite.
    The `module_name` should be the name of the calling module so
    that the classes can be fixed to make it look like they belong to it, which
    is necessary for pickling to work on them.
    """
    ret: Dict[str, Type[RpcAgentTestFixture]] = {}
    for test_class in tests:
        if IS_SANDCASTLE and TEST_WITH_DEV_DBG_ASAN:
            print(
                f'Skipping test {test_class} on sandcastle for the following reason: '
                'Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
            continue

        name = f"{prefix}{test_class.__name__}"
        class_ = type(name, (test_class, mixin, SpawnHelper), {})
        class_.__module__ = module_name
        ret[name] = class_
    return ret
