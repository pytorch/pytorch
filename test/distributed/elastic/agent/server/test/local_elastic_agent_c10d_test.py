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
from typing import Callable, Optional, Tuple

from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, record
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.agent.server.api import (
    WorkerSpec,
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
    def setUp(self):
        self._host = "localhost"
        self._port = acquire_available_port()
        self._endpoint = f"{self._host}:{self._port}"
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
        is_host=True
    ):
        rdzv_params = RendezvousParameters(
            backend="c10d",
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
    ):
        spec = self.get_worker_spec(
            node_config=conf,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            max_restarts=max_restarts,
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

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_simple_dist_sum(self):
        res = self.run_agent(Conf(entrypoint=_dist_sum, local_world_size=2))
        self.assertFalse(res.is_failed())
        # _dist_sum internally checks that the sum computed is valid
