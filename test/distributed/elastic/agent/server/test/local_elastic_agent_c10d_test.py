#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid
import tempfile
import shutil
import time
import os
import socket
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional, Tuple

from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, record
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.agent.server.api import (
    WorkerSpec,
    RunResult,
)

from torch.testing._internal.common_utils import (
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
)

import torch
import torch.distributed as dist
import torch.distributed.elastic.rendezvous.registry as rdzv_registry


def _dist_sum(wait=0):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    t = torch.tensor(rank)

    time.sleep(wait)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    expected_sum = sum(range(world_size))
    actual = t.item()
    if expected_sum != actual:
        raise RuntimeError(f"Expected rank sum {expected_sum}, got {actual}")

def acquire_available_port():
    """
    Uses sockets to acquire an available port from the os for use.

    Note: To reduce the race condition where another process grabs the port
          after this function returns an available port, we should aim to use
          the port as quickly as possible.
    """
    addrs = socket.getaddrinfo(
        host="localhost",
        port=None,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM
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


class LocalElasticAgentTest_c10d(unittest.TestCase):
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
        )

    def get_agent(
        self,
        spec: WorkerSpec,
        start_method: str = "spawn",
        exit_barrier_timeout=5
    ) -> LocalElasticAgent:
        return LocalElasticAgent(
            spec,
            start_method=start_method,
            exit_barrier_timeout=exit_barrier_timeout,
            log_dir=self.log_dir(),
        )

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
        is_host=True
    ):
        spec = self.get_worker_spec(
            node_config=conf,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            max_restarts=max_restarts,
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

    # TODO(neelgandhi) T92476229: This file should be merged with
    # local_elastic_agent_test.py in a future diff to combine the duplicate
    # logic into one file. The duplicate logic includes get_agent, run_agent,
    # get_worker_spec, run_job.
    def run_job(
        self,
        node_configs: List[Conf],
        exit_barrier_timeout: int = 5,
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
                "is_host": node_idx == 0
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

        if self._backend == "etcd-v2":
            self._endpoint = self._etcd_server.get_endpoint()
        else:
            # the default is c10d backend
            self._endpoint = f"localhost:{acquire_available_port()}"

        test_to_run()

    def simple_dist_sum(self):
        res = self.run_agent(Conf(entrypoint=_dist_sum, local_world_size=2))
        self.assertFalse(res.is_failed())
        # _dist_sum internally checks that the sum computed is valid

    def multiple_agent_dist_sum(self):
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
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_simple_dist_sum_c10d(self):
        self.run_test_with_backend(
            backend="c10d",
            test_to_run=self.simple_dist_sum
        )

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_simple_dist_sum_etcd(self):
        self.run_test_with_backend(
            backend="etcd-v2",
            test_to_run=self.simple_dist_sum
        )

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_multiple_agent_dist_sum_c10d(self):
        self.run_test_with_backend(
            backend="c10d",
            test_to_run=self.multiple_agent_dist_sum
        )

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_multiple_agent_dist_sum_etcd(self):
        self.run_test_with_backend(
            backend="etcd-v2",
            test_to_run=self.multiple_agent_dist_sum
        )
