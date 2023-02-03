#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import multiprocessing as mp
import os
import shutil
import signal
import socket
import tempfile
import time
import unittest
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from unittest import mock
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
import torch.distributed.rpc as rpc
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent,
    TORCHELASTIC_TIMER_FILE,
)
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, record
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.rpc.backend_registry import BackendType
from torch.testing._internal.common_utils import (
    sandcastle_skip_if,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_TSAN,
)


def init_rpc(name, backend):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rpc.init_rpc(
        name=name,
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def rpc_master(msg):
    init_rpc("master", BackendType.TENSORPIPE)
    ret = rpc.rpc_sync(to="worker", func=_echo, args=(msg,))
    rpc.shutdown()
    return f"{ret} from worker"


def rpc_worker():
    init_rpc("worker", BackendType.TENSORPIPE)
    rpc.shutdown()


def _happy_function():
    return


def _sad_function():
    raise RuntimeError("sad because i throw")


def dummy_compute() -> torch.Tensor:
    """
    returns a predefined size random Tensor
    """
    return torch.rand(100, 100)


def _fatal_signal_function(expected_error_index: int, sig: int):
    rank = int(os.environ["RANK"])
    if rank == expected_error_index:
        os.kill(os.getpid(), sig)


def _check_master_port_addr_override(
    expected_master_addr: str, expected_master_port: int
):
    actual_master_addr = os.environ["MASTER_ADDR"]
    actual_master_port = int(os.environ["MASTER_PORT"])
    if (
        expected_master_addr != actual_master_addr
        and expected_master_port != actual_master_port
    ):
        raise RuntimeError(
            f"Expected addr: {expected_master_addr}:{expected_master_port}, got addr: {actual_master_addr}:{actual_master_port}"
        )


def _bipolar_function():
    rank = int(os.environ["RANK"])
    if rank % 2 == 0:
        _happy_function()
    else:
        _sad_function()


def _dist_sum(wait=0):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    t = torch.tensor(rank)

    time.sleep(wait)
    dist.all_reduce(t, op=dist.reduce_op.SUM)

    expected_sum = sum(range(world_size))
    actual = t.item()
    if expected_sum != actual:
        raise RuntimeError(f"Expected rank sum {expected_sum}, got {actual}")


def _sleep(sleep_sec) -> int:
    time.sleep(sleep_sec)
    return int(os.environ["RANK"])


@dataclass
class RankInfo:
    rank: int
    role_rank: int
    group_rank: int
    role_world_size: int
    world_size: int


def _get_role_info() -> RankInfo:
    rank = int(os.environ["RANK"])
    role_rank = int(os.environ["ROLE_RANK"])
    group_rank = int(os.environ["GROUP_RANK"])
    role_world_size = int(os.environ["ROLE_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    return RankInfo(rank, role_rank, group_rank, role_world_size, world_size)


def _echo(msg):
    return msg


def _check_env_function():
    # just check these env vars exist, os.environ[...] will naturally throw
    # if the variable does not exist
    env_vars = [
        "RANK",
        "LOCAL_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "GROUP_RANK",
        "LOCAL_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "WORLD_SIZE",
        "GROUP_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
        "NCCL_ASYNC_ERROR_HANDLING",
    ]
    for var in env_vars:
        _ = os.environ[var]


def _check_env_value(key: str, expected: str):
    # checks if the env var ``key`` matches ``value``
    # this function is intended to be used as the entrypoint to the elastic run
    if key not in os.environ:
        raise RuntimeError(f"Environment variable {key} not found in os.environ")
    else:
        actual = os.getenv(key)
        if expected != actual:
            raise RuntimeError(
                f"os.environ['{key}']={actual}"
                f" does not equal the expected value: {expected}"
            )


def _check_local_watchdog_setup(key: str, should_exist: bool):
    if should_exist and key not in os.environ:
        raise RuntimeError(f"Environment variable {key} not found in os.environ")
    if not should_exist and key in os.environ:
        raise RuntimeError(f"Environment variable {key} found in os.environ")


def acquire_available_port():
    """
    Uses sockets to acquire an available port from the os for use.

    Note: To reduce the race condition where another process grabs the port
          after this function returns an available port, we should aim to use
          the port as quickly as possible.
    """
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )

    for addr in addrs:
        family, type, proto, _, _ = addr
        try:
            s = socket.socket(family, type, proto)
            s.bind(("localhost", 0))
            s.listen(0)
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError as e:
            s.close()
            print(f"Socket creation attempt failed: {e}")

    raise RuntimeError("Failed to create a socket")


@dataclass
class Conf:
    """
    Holds arguments to launch an agent (e.g. simulates an agent run on a node).

    """

    entrypoint: Callable
    local_world_size: int
    args: Tuple = ()
    role: str = "default"
    redirects: Std = Std.NONE
    tee: Std = Std.NONE


class LocalElasticAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def setUp(self):
        self._test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        self._run_id = str(uuid.uuid4()).split("-")[0]

    def tearDown(self):
        shutil.rmtree(self._test_dir)

    def log_dir(self) -> str:
        return tempfile.mkdtemp(prefix="torchelastic_", dir=self._test_dir)

    def get_worker_spec(
        self,
        node_config: Conf,
        min_nodes=1,
        max_nodes=1,
        max_restarts=0,
        monitor_interval=0.01,
        master_addr_override: Optional[str] = None,
        master_port_override: Optional[int] = None,
        is_host=True,
    ):
        rdzv_params = RendezvousParameters(
            backend=self._backend,
            endpoint=self._endpoint,
            run_id=self._run_id,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            is_host=is_host,
        )
        rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_params)
        return WorkerSpec(
            role=node_config.role,
            local_world_size=node_config.local_world_size,
            entrypoint=node_config.entrypoint,
            args=node_config.args,
            rdzv_handler=rdzv_handler,
            max_restarts=max_restarts,
            monitor_interval=monitor_interval,
            redirects=node_config.redirects,
            tee=node_config.tee,
            master_addr=master_addr_override,
            master_port=master_port_override,
        )

    def get_agent(
        self, spec: WorkerSpec, start_method: str = "spawn", exit_barrier_timeout=5
    ) -> LocalElasticAgent:
        return LocalElasticAgent(
            spec,
            start_method=start_method,
            exit_barrier_timeout=exit_barrier_timeout,
            log_dir=self.log_dir(),
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.multiprocessing.errors.record`.
    @record
    def run_agent(
        self,
        conf: Conf,
        agent_results: Optional[mp.Queue] = None,  # (role, agent_result)
        min_nodes=1,
        max_nodes=1,
        start_method: str = "spawn",
        max_restarts: int = 0,
        exit_barrier_timeout=5,
        master_addr_override: Optional[str] = None,
        master_port_override: Optional[int] = None,
        is_host=True,
    ) -> Optional[RunResult]:
        """
        Runs a single agent. This method can be called either on a separate process
        or the main test process. When calling this method on a sparate process make
        sure to pass the ``agent_results`` multiprocessing Queue so that the agent's
        run results can be returned. If ``agent_results`` is omitted, then the
        run result is returned from the method.
        """

        spec = self.get_worker_spec(
            node_config=conf,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            max_restarts=max_restarts,
            master_addr_override=master_addr_override,
            master_port_override=master_port_override,
            is_host=is_host,
        )
        agent = self.get_agent(
            spec=spec,
            start_method=start_method,
            exit_barrier_timeout=exit_barrier_timeout,
        )

        result = agent.run()
        spec.rdzv_handler.shutdown()

        if agent_results:
            agent_results.put((conf.role, result))

        if result.is_failed():
            raise ChildFailedError(spec.get_entrypoint_name(), result.failures)
        else:
            if not agent_results:
                return result

    def run_job(
        self, node_configs: List[Conf], exit_barrier_timeout: int = 5
    ) -> Dict[str, List[RunResult]]:
        """
        Simulates running a distributed job by running multiple agents
        (one on each process). Agent 0 is run on the main process for
        test coverage and ease of debugging
        """

        nnodes = len(node_configs)

        # each element in this queue holds a tuple (role, RunResult) for each agent
        agent_results = mp.Queue()

        # run first agent of first config on main process for test coverage + ease of debugging
        # it is important we loop in reverse order b/c running fn on the main process blocks
        procs = []
        for node_idx in reversed(range(len(node_configs))):
            conf = node_configs[node_idx]
            run_agent_args = {
                "conf": conf,
                "agent_results": agent_results,
                "min_nodes": nnodes,
                "max_nodes": nnodes,
                "start_method": "spawn",
                "max_restarts": 0,
                "exit_barrier_timeout": exit_barrier_timeout,
                "is_host": node_idx == 0,
            }
            p = mp.Process(target=self.run_agent, kwargs=run_agent_args)
            procs.append(p)
            p.start()
        for p in procs:
            p.join()

        results: Dict[str, List[RunResult]] = {}
        while not agent_results.empty():
            role, run_result = agent_results.get()
            results.setdefault(role, []).append(run_result)
        return results

    def run_test_with_backend(self, backend: str, test_to_run: Callable):
        """
        Sets the backend and determines the endpoint before running the
        given test.

        Note: This method must be invoked to run any test functions that spawn
              an agent. This is because this function sets the backend and
              endpoint parameters.
        """
        self._backend = backend

        if self._backend == "etcd-v2" or self._backend == "etcd":
            self._endpoint = self._etcd_server.get_endpoint()
        else:
            # the default is c10d backend
            self._endpoint = f"localhost:{acquire_available_port()}"

        test_to_run()

    def dummy_compute(self):
        res = self.run_agent(Conf(entrypoint=dummy_compute, local_world_size=2))
        self.assertFalse(res.is_failed())
        for return_value in res.return_values.values():
            self.assertIsInstance(return_value, torch.Tensor)
            self.assertEqual((100, 100), return_value.shape)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_dummy_compute_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.dummy_compute)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_dummy_compute_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.dummy_compute)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_dummy_compute_etcd_v2(self):
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.dummy_compute)

    def run_happy_function(self):
        res = self.run_agent(Conf(entrypoint=_happy_function, local_world_size=2))
        self.assertFalse(res.is_failed())
        self.assertIsNone(res.return_values[0])
        self.assertIsNone(res.return_values[1])

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_happy_function_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.run_happy_function)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_happy_function_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.run_happy_function)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_happy_function_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_happy_function
        )

    def check_master_addr_port_override(self):
        master_addr = "test_host"
        master_port = 42
        res = self.run_agent(
            Conf(
                entrypoint=_check_master_port_addr_override,
                args=(master_addr, master_port),
                local_world_size=1,
            ),
            master_addr_override=master_addr,
            master_port_override=master_port,
        )
        self.assertFalse(res.is_failed())
        self.assertIsNone(res.return_values[0])

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_check_master_addr_port_override_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.check_master_addr_port_override
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_check_master_addr_port_override_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.check_master_addr_port_override
        )

    def run_check_env_function(self):
        # just checks that all env vars that we need to set on the user script
        # is actually set
        res = self.run_agent(Conf(entrypoint=_check_env_function, local_world_size=1))
        self.assertFalse(res.is_failed())

    def run_check_nccl_async_error_handling_env(self):
        # make sure NCCL_ASYNC_ERROR_HANDLING set in os.environ is honored
        with patch.dict(os.environ, {"NCCL_ASYNC_ERROR_HANDLING": "0"}):
            res = self.run_agent(
                Conf(
                    entrypoint=_check_env_value,
                    local_world_size=1,
                    args=("NCCL_ASYNC_ERROR_HANDLING", "0"),
                )
            )
            self.assertFalse(res.is_failed())

    def run_check_nccl_async_error_handling_env_default(self):
        # if not present in env var it should default to 1
        res = self.run_agent(
            Conf(
                entrypoint=_check_env_value,
                local_world_size=1,
                args=("NCCL_ASYNC_ERROR_HANDLING", "1"),
            )
        )
        self.assertFalse(res.is_failed())

    def run_agent_local_watchdog_setup_enabled(self):
        # Set the env for watchdog
        watchdog_env_name = TORCHELASTIC_TIMER_FILE
        watchdog_file_path = "/tmp/watchdog_timer_" + str(uuid.uuid4())
        os.environ[watchdog_env_name] = watchdog_file_path
        # Run the agent
        node_conf = Conf(entrypoint=_check_local_watchdog_setup, local_world_size=1, args=(TORCHELASTIC_TIMER_FILE, True))
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec)
        res = agent.run()
        self.assertFalse(res.is_failed())

    def run_agent_local_watchdog_setup_disabled(self):
        # Do not set the env for watchdog
        watchdog_env_name = TORCHELASTIC_TIMER_FILE
        if watchdog_env_name in os.environ:
            del os.environ[watchdog_env_name]
        # Run the agent
        node_conf = Conf(entrypoint=_check_local_watchdog_setup, local_world_size=1, args=(TORCHELASTIC_TIMER_FILE, False))
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec)
        res = agent.run()
        self.assertFalse(res.is_failed())

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_agent_local_watchdog_setup_enabled_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_local_watchdog_setup_enabled
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_agent_local_watchdog_setup_enabled_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_local_watchdog_setup_enabled
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_agent_local_watchdog_setup_disabled_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_local_watchdog_setup_disabled
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_agent_local_watchdog_setup_disabled_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_local_watchdog_setup_disabled
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_check_env_function_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_check_env_function
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_check_nccl_async_error_handling_env_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_check_nccl_async_error_handling_env
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_check_nccl_async_error_handling_env_default_c10d(self):
        self.run_test_with_backend(
            backend="c10d",
            test_to_run=self.run_check_nccl_async_error_handling_env_default,
        )

    def run_function_with_return_value(self):
        res = self.run_agent(Conf(entrypoint=_echo, args=("foo",), local_world_size=2))
        self.assertFalse(res.is_failed())
        self.assertEqual("foo", res.return_values[0])
        self.assertEqual("foo", res.return_values[1])

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_function_with_return_value_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_function_with_return_value
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_function_with_return_value_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_function_with_return_value
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_function_with_return_value_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_function_with_return_value
        )

    def simple_dist_sum(self):
        res = self.run_agent(Conf(entrypoint=_dist_sum, local_world_size=2))
        self.assertFalse(res.is_failed())
        # _dist_sum internally checks that the sum computed is valid

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_simple_dist_sum_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.simple_dist_sum)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_simple_dist_sum_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.simple_dist_sum)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_simple_dist_sum_etcd_v2(self):
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.simple_dist_sum)

    def run_distributed_sum_homogeneous(self):
        node_configs = [
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=4),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=4),
        ]
        # When the process method is spawn, the coverage collector hangs
        # due to getting stuck on the _dist_sum in waiting for TCPStore workers
        # to join the cluster
        # TODO(aivanou): t83447589 come up with the proper fix
        res = self.run_job(node_configs)
        self.assertEqual(2, len(res["sum"]))
        ranks = set()
        for run_results in res["sum"]:
            self.assertFalse(run_results.is_failed())
            ranks.update(run_results.return_values.keys())
        self.assertSetEqual(set(range(4 + 4)), ranks)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_distributed_sum_homogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_distributed_sum_homogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_distributed_sum_homogeneous
        )

    def run_distributed_sum_heterogeneous(self):
        # sums all ranks on 3 agents; each running 1, 2, 3 workers respectively
        # sum should be equal to 0 + (1 + 2) + (3 + 4 + 5) = 15
        # sum asserted inside _dist_sum()
        node_configs = [
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=1),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=2),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=3),
        ]
        # When the process method is spawn, the coverage collector hangs
        # due to getting stuck on the _dist_sum in waiting for TCPStore workers
        # to join the cluster
        # TODO(aivanou): t83447589 come up with the proper fix
        res = self.run_job(node_configs)
        self.assertEqual(3, len(res["sum"]))
        ranks = set()
        for run_results in res["sum"]:
            self.assertFalse(run_results.is_failed())
            ranks.update(run_results.return_values.keys())
        self.assertSetEqual(set(range(1 + 2 + 3)), ranks)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_distributed_sum_heterogeneous_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_distributed_sum_heterogeneous
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_distributed_sum_heterogeneous_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_distributed_sum_heterogeneous
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_distributed_sum_heterogeneous_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_distributed_sum_heterogeneous
        )

    def run_sad_function(self):
        """
        checks error propagation logic
        """
        replyfile = os.path.join(self._test_dir, "error.json")
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": replyfile}):
            with self.assertRaises(ChildFailedError) as cm:
                self.run_agent(Conf(entrypoint=_sad_function, local_world_size=2))

            rank, failure = cm.exception.get_first_failure()
            failure_data = failure.error_file_data["message"]
            with open(replyfile, "r") as fp:
                data = json.load(fp)["message"]

                # ran two; both failed; first failure is either rank 0 or 1
                self.assertTrue(rank in {0, 1})
                self.assertTrue(failure.local_rank in {0, 1})
                self.assertEqual(1, failure.exitcode)
                self.assertEqual(data["message"], failure_data["message"])
                self.assertEqual(int(data["extraInfo"]["timestamp"]), failure.timestamp)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_sad_function_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.run_sad_function)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_sad_function_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.run_sad_function)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_sad_function_etcd_v2(self):
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.run_sad_function)

    def run_bipolar_function(self):
        """
        checks agent failure handling logic
        """
        node_conf = Conf(entrypoint=_bipolar_function, local_world_size=4)
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec)
        run_result = agent.run()
        self.assertTrue(run_result.is_failed())
        self.assertEqual(0, agent._remaining_restarts)
        self.assertEqual(WorkerState.FAILED, agent.get_worker_group().state)
        self.assertTrue(agent._total_execution_time > 0)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_bipolar_function_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_bipolar_function
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_bipolar_function_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_bipolar_function
        )

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_run_bipolar_function_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_bipolar_function
        )

    def correct_rank_assignment_heterogeneous(self):
        node_configs = [
            Conf(role="master", entrypoint=_get_role_info, local_world_size=8),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=1),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=2),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=3),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=5),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=2),
        ]
        results = self.run_job(node_configs)
        print(f"heterogeneous job result: {results}")
        self.assertEqual(1, len(results["master"]))
        self.assertEqual(4, len(results["trainer"]))
        self.assertEqual(2, len(results["ps"]))
        self.assert_rank_consistency(
            results,
            expected_role_world_sizes={
                "master": 8,
                "trainer": 1 + 2 + 3 + 4,
                "ps": 5 + 2,
            },
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_heterogeneous_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.correct_rank_assignment_heterogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_heterogeneous_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.correct_rank_assignment_heterogeneous
        )

    def correct_rank_assignment_homogeneous(self):
        node_configs = [
            Conf(role="master", entrypoint=_get_role_info, local_world_size=1),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=3),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=3),
        ]
        results = self.run_job(node_configs)
        print(f"homogeneous job result: {results}")
        self.assertEqual(1, len(results["master"]))
        self.assertEqual(4, len(results["trainer"]))
        self.assertEqual(2, len(results["ps"]))
        self.assert_rank_consistency(
            results,
            expected_role_world_sizes={"master": 1, "trainer": 4 * 4, "ps": 3 * 2},
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_homogeneous_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.correct_rank_assignment_homogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_homogeneous_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.correct_rank_assignment_homogeneous
        )

    def assert_rank_consistency(
        self,
        run_results: Dict[str, List[RunResult]],
        expected_role_world_sizes: Dict[str, int],
    ):
        """
        Asserts that ranks are consecutive w.r.t role_rank. If local world sizes are 4:
        role_rank_0 -> ranks: 0,1,2,3
        role_rank_1 -> ranks: 4,5,6,7
        ... etc ...
        """

        global_ranks: List[int] = []
        # role -> [role_rank,...]
        role_ranks: Dict[str, List[int]] = {}
        # group rank -> [(rank, role_rank),...]
        grouped_ranks: Dict[int, List[Tuple[int, int]]] = {}

        # global world size == sum of all the role world sizes
        expected_world_size = sum(expected_role_world_sizes.values())
        for role, run_results in run_results.items():
            for result in run_results:
                res = result.return_values
                for role_info in res.values():
                    rank = role_info.rank
                    role_rank = role_info.role_rank
                    group_rank = role_info.group_rank
                    role_world_size = role_info.role_world_size
                    world_size = role_info.world_size

                    self.assertEqual(expected_world_size, world_size)
                    self.assertEqual(expected_role_world_sizes[role], role_world_size)
                    grouped_ranks.setdefault(group_rank, []).append((rank, role_rank))
                    role_ranks.setdefault(role, []).append(role_rank)
                    global_ranks.append(rank)

        global_ranks = sorted(global_ranks)
        self.assertEqual(list(range(expected_world_size)), global_ranks)
        for role, expected_role_world_size in expected_role_world_sizes.items():
            self.assertEqual(
                list(range(expected_role_world_size)), sorted(role_ranks[role])
            )
        # Make sure that each agent assigns consecutive ranks to workers
        # The first argument is the global_rank and the second argument
        # is role_rank
        for ranks_lst in grouped_ranks.values():
            self.assert_ranks_sequential(ranks_lst, 0)
            self.assert_ranks_sequential(ranks_lst, 1)

    def assert_ranks_sequential(self, ranks_pairs, rank_idx):
        ranks = sorted(rank_pair[rank_idx] for rank_pair in ranks_pairs)
        start_rank, end_rank = ranks[0], ranks[-1]
        self.assertEqual(list(range(start_rank, end_rank + 1)), ranks)

    def double_agent_fault_tolerance(self):
        """
        start ``nnodes`` agents, kill and restart odd ones, validate fault-tolerance works
        """
        nnodes = 2
        wait = 2
        node_conf = Conf(entrypoint=_dist_sum, args=(wait,), local_world_size=2)
        agent_results = mp.Queue()
        agent_args = {
            "conf": node_conf,
            "agent_results": agent_results,
            "min_nodes": nnodes,
            "max_nodes": nnodes,
            "max_restarts": 2,
        }

        procs = []
        for _ in range(nnodes):
            p = mp.Process(
                target=self.run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # restart odd agents
        for i in range(nnodes):
            if i % 2 != 0:
                procs[i].kill()
                p = mp.Process(
                    target=self.run_agent,
                    kwargs=agent_args,
                )
                procs[i] = p
                p.start()

        for i in range(nnodes):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_fault_tolerance_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.double_agent_fault_tolerance
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_fault_tolerance_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.double_agent_fault_tolerance
        )

    def double_agent_elastic(self):
        """
        start ``nnodes`` agents, kill odd ones (do not restart), validate
        elasticity (scale-down) works. (scale-up covered in fault_tolerance test)
        """
        min_nodes = 1
        max_nodes = 2
        wait = 2
        node_conf = Conf(entrypoint=_dist_sum, args=(wait,), local_world_size=2)
        agent_results = mp.Queue()
        agent_args = {
            "conf": node_conf,
            "agent_results": agent_results,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "max_restarts": 2,
        }

        procs = []
        for _ in range(max_nodes):
            p = mp.Process(
                target=self.run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # kill odd agents
        for i in range(max_nodes):
            if i % 2 != 0:
                procs[i].kill()

        for i in range(max_nodes):
            p = procs[i]
            p.join()
            if i % 2 == 0:
                self.assertEqual(0, p.exitcode)
            else:
                self.assertEqual(-signal.SIGKILL, p.exitcode)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_c10d(self):
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.double_agent_elastic
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.double_agent_elastic
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.double_agent_elastic
        )

    def torch_rpc(self):
        """
        Simple torch rpc example with torchelastic.
        Creates two agents (to simulate two node job),
        each agent runs a single worker. worker0 calls an rpc_sync on
        worker1.
        """
        msg = "hello world"
        node_configs = [
            Conf(
                role="master",
                entrypoint=rpc_master,
                args=(msg,),
                local_world_size=1,
                tee=Std.ALL,
            ),
            Conf(
                role="worker",
                entrypoint=rpc_worker,
                args=(),
                local_world_size=1,
                tee=Std.ALL,
            ),
        ]

        results = self.run_job(node_configs)
        master_retvals = results["master"][0].return_values
        # there is only one master but the global rank is not stable
        # so compare the master return value as a collection
        self.assertEqual([f"{msg} from worker"], list(master_retvals.values()))

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.torch_rpc)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.torch_rpc)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_etcd_v2(self):
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.torch_rpc)

    def workers_drift_success(self):
        """
        two agents (one worker each) finishes within ``sec`` seconds of each other,
        exit barrier timeout set to ``sec * 2 * 2``.
        """

        sec = 1
        node_configs = [
            Conf(role="zzz", entrypoint=_sleep, args=(0 * sec,), local_world_size=1),
            Conf(role="zzz", entrypoint=_sleep, args=(2 * sec,), local_world_size=1),
        ]
        results = self.run_job(node_configs, exit_barrier_timeout=2 * 2 * sec)
        for i in range(2):
            run_results = results["zzz"][i]
            self.assertFalse(run_results.is_failed())
            for rank, output in run_results.return_values.items():
                # _sleep() returns its own rank
                self.assertEqual(rank, output)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_success_etcd(self):
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.workers_drift_success
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_success_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.workers_drift_success
        )

    def workers_drift_fail(self):
        """
        two agents (one worker each) finishes within ``4 x sec`` seconds of each other,
        exit barrier timeout set to 0. Exit barriers should NOT fail the job.
        """
        sec = 1
        node_configs = [
            Conf(role="zzz", entrypoint=_sleep, args=(0 * sec,), local_world_size=1),
            Conf(role="zzz", entrypoint=_sleep, args=(4 * sec,), local_world_size=1),
        ]
        results = self.run_job(node_configs, exit_barrier_timeout=0)
        for i in range(2):
            run_results = results["zzz"][i]
            self.assertFalse(run_results.is_failed())
            for rank, output in run_results.return_values.items():
                # _sleep() returns its own rank
                self.assertEqual(rank, output)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_fail_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.workers_drift_fail)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_fail_etcd_v2(self):
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.workers_drift_fail
        )

    @patch("torch.distributed.elastic.utils.store.barrier")
    def barrier_failed(self, barrier_mock):
        """
        Failure during the barrier should NOT fail the job.
        """
        barrier_mock.side_effect = RuntimeError("test error")
        res = self.run_agent(Conf(entrypoint=_happy_function, local_world_size=1))
        self.assertFalse(res.is_failed())
        barrier_mock.assert_called_once()

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_barrier_failed_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.barrier_failed)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_barrier_failed_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.barrier_failed)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_barrier_failed_etcd_v2(self):
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.barrier_failed)

    @patch("torch.distributed.elastic.agent.server.local_elastic_agent.start_processes")
    def shutdown_called(self, start_processes_mock):
        pcontext_mock = Mock()
        pcontext_mock.pids.return_value = {0: 0}
        start_processes_mock.return_value = pcontext_mock
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)
        spec = self.get_worker_spec(node_conf, max_restarts=0)
        agent = self.get_agent(spec)
        with patch.object(agent, "_monitor_workers") as monitor_mock:
            monitor_mock.return_value = RunResult(
                state=WorkerState.SUCCEEDED, return_values={0: 0}
            )
            agent.run("worker")
        pcontext_mock.close.assert_called_once()

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_shutdown_called_c10d(self):
        self.run_test_with_backend(backend="c10d", test_to_run=self.shutdown_called)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_shutdown_called_etcd(self):
        self.run_test_with_backend(backend="etcd", test_to_run=self.shutdown_called)

    @sandcastle_skip_if(TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan")
    def test_shutdown_called_etcd_v2(self):
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.shutdown_called)
