#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import tempfile
import unittest
from enum import Flag, auto
from typing import Dict, List, Type

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


# libSegFault.so is a library that, when (pre-)loaded into an executable,
# installs a signal handler (for SIGSEGV and possibly others) that prints the
# back trace, the content of registers and possibly other useful info.
def find_libsegfault_path():
    try:
        output = subprocess.check_output(["/sbin/ldconfig", "-p"],
                                         env={"LC_ALL": "C", "LANG": "C"})
    except subprocess.SubprocessError:
        return None
    regex = r"^\s*libSegFault.so\s+\([^)]+\) => (.*)$"
    try:
        match = re.search(os.fsencode(regex), output, re.MULTILINE)
        if not match:
            return None
        return os.fsdecode(match.group(1))
    except OSError:
        return None


class RpcMultiProcessTestCase(MultiProcessTestCase, RpcAgentTestFixture):
    LIBSEGFAULT_PATH = find_libsegfault_path()

    def setUp(self):
        super().setUp()
        self.output_dir = tempfile.mkdtemp()
        self.subprocess_init_data["output_dir"] = self.output_dir

    def subprocess_init(self, data):
        with open(os.path.join(data["output_dir"], f"rank_{self.rank}_stdout"), "wt") as f:
            os.dup2(f.fileno(), 1)
        sys.stdout = os.fdopen(1, "wt")
        with open(os.path.join(data["output_dir"], f"rank_{self.rank}_stderr"), "wt") as f:
            os.dup2(f.fileno(), 2)
        sys.stderr = os.fdopen(2, "wt")

    def subprocess_env_vars(self):
        env_vars = dict()
        if self.LIBSEGFAULT_PATH is not None:
            env_vars["LD_PRELOAD"] = self.LIBSEGFAULT_PATH
            env_vars["SEGFAULT_SIGNALS"] = "segv abrt bus"
        env_vars.update(self.get_env_vars())
        return env_vars

    def on_test_failure(self):
        for rank in range(self.world_size):
            try:
                with open(os.path.join(self.output_dir, f"rank_{rank}_stdout"), "rt") as f:
                    print(f"Process {rank} stdout:")
                    print(f.read())
            except Exception as err:
                print(f"Process {rank} didn't produce stdout ({err})")
            try:
                with open(os.path.join(self.output_dir, f"rank_{rank}_stderr"), "rt") as f:
                    print(f"Process {rank} stderr:")
                    print(f.read())
            except Exception as err:
                print(f"Process {rank} didn't produce stderr ({err})")

    def on_test_timeout(self):
        for rank in range(self.world_size):
            process = self.processes[rank]
            if process.exitcode is not None:
                continue
            try:
                # There's no guarantee that GDB will be available on the machine
                # but this is a best-effort diagnostic and we'll ignore errors.
                subprocess.check_call([
                    "gdb",
                    sys.executable,
                    "--pid",
                    f"{process.pid}",
                    "--batch",
                    # Print the C/C++ stack trace of all threads.
                    "--eval-command",
                    "thread apply all backtrace",
                    # Print the Python stack trace of all threads.
                    # This requires GDB's Python extensions to be installed and
                    # may fail if they aren't. We'll ignore those failures.
                    "--eval-command",
                    "thread apply all py-bt",
                ])
            except subprocess.SubprocessError as err:
                print(f"Issues while retrieving stack traces for process {rank}: {err}")


@unittest.skipIf(TEST_WITH_TSAN, "TSAN and fork() is broken")
class ForkHelper(RpcMultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._fork_processes()


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class SpawnHelper(RpcMultiProcessTestCase):
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


# This list contains test suites that are agent-agnostic and that only verify
# compliance with the generic RPC interface specification. These tests should
# *not* make use of implementation details of a specific agent (options,
# attributes, ...). These test suites will be instantiated multiple times, once
# for each agent (except the faulty agent, which is special).
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
