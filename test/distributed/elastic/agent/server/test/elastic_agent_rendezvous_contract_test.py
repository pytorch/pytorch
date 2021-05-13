#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import unittest
import uuid
from contextlib import closing
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    WorkerState,
)
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.utils import get_socket_with_port
from torch.testing._internal.common_utils import (
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
)
from torch.testing._internal.distributed.elastic_test import (
    Conf,
    run_job_async,
    get_etcd_rdzv_params,
    get_dynamic_rdzv_params,
)


def scaled_work(
    initial_world_size: int, final_world_size: int, response_queues: List[mp.Queue]
) -> int:
    """
    Method is expected to be invoked from the cluster that has `initial_world_size`,
    and after restart the cluster should be `final_world_size`.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    response_queue = response_queues[rank]

    print(f"Worker {rank}-{local_rank} world-size: {world_size} Started")
    response_queue.put(rank)
    if world_size == initial_world_size:
        print(
            f"Worker {rank}-{local_rank} is waiting for scale event, "
            f"current world size: {initial_world_size}, expected world size: {final_world_size}"
        )
        time.slepe(600)  # sleep forever

    dist.init_process_group(backend="gloo")
    t = torch.tensor(rank)
    dist.all_reduce(t, op=dist.reduce_op.SUM)
    output = t.item()
    return output


class ElasticAgentRendezvousContractTest(unittest.TestCase):
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

    def tearDown(self):
        shutil.rmtree(self._test_dir)

    def log_dir(self) -> str:
        return tempfile.mkdtemp(prefix="torchelastic_", dir=self._test_dir)

    def _get_agent_results(self, procs, agent_results_queue):
        for p in procs:
            p.join()

        results: Dict[str, List[RunResult]] = {}
        while not agent_results_queue.empty():
            role, run_result = agent_results_queue.get()
            results.setdefault(role, []).append(run_result)
        return results

    def _get_node_confs(self, num_nodes: int, num_workers: int, args) -> List[Conf]:
        node_confs = []
        for _ in range(num_nodes):
            node_conf = Conf(
                role="worker",
                entrypoint=scaled_work,
                local_world_size=num_workers,
                args=args,
            )
            node_confs.append(node_conf)
        return node_confs

    def _get_and_assert_agent_results(
        self, agent_procs, agent_results_queue, total_workers: int
    ):
        results = self._get_agent_results(agent_procs, agent_results_queue)
        worker_results = results["worker"]
        # import pdb;pdb.set_trace
        self.assertEqual(total_workers, len(worker_results))
        expected_result = sum(total_workers)
        for run_result in worker_results:
            self.assertEqual(WorkerState.SUCCEEDED, run_result.state)
            for worker_rank, worker_result in run_result.return_values.items():
                self.assertEqual(expected_result, worker_result)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_dynamic_rdzv_scale_up(self):
        initial_workers = 1
        new_workers = 1
        total_workers = initial_workers + new_workers
        run_id = str(uuid.uuid4()).split("-")[0]
        sock = get_socket_with_port()
        with closing(sock):
            port = sock.getsockname()[1]

        endpoint = f"localhost:{port}"

        response_queues = [
            mp.get_context("spawn").Queue() for _ in range(total_workers)
        ]

        initial_nodes = self._get_node_confs(
            num_nodes=1, num_workers=1, args=(1, 2, response_queues)
        )
        initial_nodes[0].is_host = True
        new_nodes = self._get_node_confs(
            num_nodes=1, num_workers=1, args=(1, 2, response_queues)
        )

        print(f"Starting initial {initial_workers} nodes")
        rdzv_params = get_dynamic_rdzv_params(
            endpoint, run_id, min_nodes=1, max_nodes=2
        )
        initial_agent_procs, initial_agent_results_queue = run_job_async(
            rdzv_params, self.log_dir(), initial_nodes
        )

        for idx in range(0, initial_workers):
            response_queue = response_queues[idx]
            response_queue.get()

        print(f"Starting new {new_workers} nodes")
        new_agent_procs, new_agent_results_queue = run_job_async(
            rdzv_params, self.log_dir(), new_nodes
        )

        print(f"Cleanup total {total_workers} nodes")
        self._get_agent_results(initial_agent_procs, initial_agent_results_queue)
        self._get_agent_results(new_agent_procs, new_agent_results_queue)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_etcd_scale_up(self):
        initial_workers = 1
        new_workers = 1
        total_workers = initial_workers + new_workers
        run_id = str(uuid.uuid4()).split("-")[0]

        response_queues = [
            mp.get_context("spawn").Queue() for _ in range(total_workers)
        ]

        initial_nodes = self._get_node_confs(
            num_nodes=1, num_workers=1, args=(1, 2, response_queues)
        )
        new_nodes = self._get_node_confs(
            num_nodes=1, num_workers=1, args=(1, 2, response_queues)
        )

        print(f"Starting initial {initial_workers} nodes")
        rdzv_params = get_etcd_rdzv_params(
            self._etcd_server.get_endpoint(), run_id, min_nodes=1, max_nodes=2
        )
        initial_agent_procs, initial_agent_results_queue = run_job_async(
            rdzv_params, self.log_dir(), initial_nodes
        )

        for idx in range(0, initial_workers):
            response_queue = response_queues[idx]
            response_queue.get()

        print(f"Starting new {new_workers} nodes")
        new_agent_procs, new_agent_results_queue = run_job_async(
            rdzv_params, self.log_dir(), new_nodes
        )

        print(f"Cleanup total {total_workers} nodes")
        self._get_agent_results(initial_agent_procs, initial_agent_results_queue)
        self._get_agent_results(new_agent_procs, new_agent_results_queue)
