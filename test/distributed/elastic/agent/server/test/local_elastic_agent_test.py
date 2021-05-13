#!/usr/bin/env python3

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
import tempfile
import time
import unittest
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple
from unittest import mock
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    WorkerState,
)
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.rpc.backend_registry import BackendType
from torch.testing._internal.common_utils import (
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
)
from torch.testing._internal.distributed.elastic_test import (
    Conf,
    run_job,
    get_etcd_rdzv_params,
    get_agent,
    run_agent,
    get_worker_spec,
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
    ]
    for var in env_vars:
        _ = os.environ[var]


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

    def _get_etcd_rdzv_params(self, nnodes: int) -> RendezvousParameters:
        return get_etcd_rdzv_params(
            endpoint=self._etcd_server.get_endpoint(),
            run_id=self._run_id,
            min_nodes=nnodes,
            max_nodes=nnodes,
        )

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_dummy_compute(self):
        res = run_agent(
            self._get_etcd_rdzv_params(1),
            Conf(entrypoint=dummy_compute, local_world_size=2),
            self.log_dir(),
        )
        self.assertFalse(res.is_failed())
        for return_value in res.return_values.values():
            self.assertIsInstance(return_value, torch.Tensor)
            self.assertEqual((100, 100), return_value.shape)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_run_happy_function(self):
        res = run_agent(
            self._get_etcd_rdzv_params(1),
            Conf(entrypoint=_happy_function, local_world_size=2),
            self.log_dir(),
        )
        self.assertFalse(res.is_failed())
        self.assertIsNone(res.return_values[0])
        self.assertIsNone(res.return_values[1])

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_check_master_addr_port_override(self):
        master_addr = "test_host"
        master_port = 42
        res = run_agent(
            self._get_etcd_rdzv_params(1),
            Conf(
                entrypoint=_check_master_port_addr_override,
                args=(master_addr, master_port),
                local_world_size=1,
            ),
            self.log_dir(),
            master_addr_override=master_addr,
            master_port_override=master_port,
        )
        self.assertFalse(res.is_failed())
        self.assertIsNone(res.return_values[0])

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_run_check_env_function(self):
        # just checks that all env vars that we need to set on the user script
        # is actually set
        res = run_agent(
            self._get_etcd_rdzv_params(1),
            Conf(entrypoint=_check_env_function, local_world_size=1),
            self.log_dir(),
        )
        self.assertFalse(res.is_failed())

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_run_function_with_return_value(self):
        res = run_agent(
            self._get_etcd_rdzv_params(1),
            Conf(entrypoint=_echo, args=("foo",), local_world_size=2),
            self.log_dir(),
        )
        self.assertFalse(res.is_failed())
        self.assertEqual("foo", res.return_values[0])
        self.assertEqual("foo", res.return_values[1])

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_run_distributed_sum_homogeneous(self):
        node_configs = [
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=4),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=4),
        ]
        # When the process method is spawn, the coverage collector hangs
        # due to getting stuck on the _dist_sum in waiting for TCPStore workers
        # to join the cluster
        # TODO(aivanou): t83447589 come up with the proper fix
        res = run_job(
            self._get_etcd_rdzv_params(len(node_configs)), self.log_dir(), node_configs
        )
        self.assertEqual(2, len(res["sum"]))
        ranks = set()
        for run_results in res["sum"]:
            self.assertFalse(run_results.is_failed())
            ranks.update(run_results.return_values.keys())
        self.assertSetEqual(set(range(4 + 4)), ranks)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_run_distributed_sum_heterogenous(self):
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
        res = run_job(
            self._get_etcd_rdzv_params(len(node_configs)), self.log_dir(), node_configs
        )
        self.assertEqual(3, len(res["sum"]))
        ranks = set()
        for run_results in res["sum"]:
            self.assertFalse(run_results.is_failed())
            ranks.update(run_results.return_values.keys())
        self.assertSetEqual(set(range(1 + 2 + 3)), ranks)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_run_sad_function(self):
        """
        checks error propagation logic
        """
        replyfile = os.path.join(self._test_dir, "error.json")
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": replyfile}):
            with self.assertRaises(ChildFailedError) as cm:
                run_agent(
                    self._get_etcd_rdzv_params(1),
                    Conf(entrypoint=_sad_function, local_world_size=2),
                    self.log_dir(),
                )

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

    def test_run_bipolar_function(self):
        """
        checks agent failure handling logic
        """
        rdzv_params = self._get_etcd_rdzv_params(1)
        node_conf = Conf(entrypoint=_bipolar_function, local_world_size=4)
        spec = get_worker_spec(rdzv_params, node_conf, max_restarts=2)
        agent = get_agent(spec, self.log_dir())
        run_result = agent.run()
        self.assertTrue(run_result.is_failed())
        self.assertEqual(0, agent._remaining_restarts)
        self.assertEqual(WorkerState.FAILED, agent.get_worker_group().state)
        self.assertTrue(agent._total_execution_time > 0)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_correct_rank_assignment_heterogeneous(self):
        node_configs = [
            Conf(role="master", entrypoint=_get_role_info, local_world_size=8),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=1),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=2),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=3),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=5),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=2),
        ]
        results = run_job(
            self._get_etcd_rdzv_params(len(node_configs)), self.log_dir(), node_configs
        )
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
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_correct_rank_assignment_homogeneous(self):
        node_configs = [
            Conf(role="master", entrypoint=_get_role_info, local_world_size=1),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=3),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=3),
        ]
        results = run_job(
            self._get_etcd_rdzv_params(len(node_configs)), self.log_dir(), node_configs
        )
        print(f"homogeneous job result: {results}")
        self.assertEqual(1, len(results["master"]))
        self.assertEqual(4, len(results["trainer"]))
        self.assertEqual(2, len(results["ps"]))
        self.assert_rank_consistency(
            results,
            expected_role_world_sizes={"master": 1, "trainer": 4 * 4, "ps": 3 * 2},
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

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_double_agent_fault_tolerance(self):
        """
        start ``nnodes`` agents, kill and restart odd ones, validate fault-tolerance works
        """
        nnodes = 2
        wait = 2
        node_conf = Conf(entrypoint=_dist_sum, args=(wait,), local_world_size=2)
        agent_results = mp.Queue()
        agent_args = {
            "rdzv_params": self._get_etcd_rdzv_params(nnodes),
            "conf": node_conf,
            "log_dir": self.log_dir(),
            "agent_results": agent_results,
            "max_restarts": 2,
        }

        procs = []
        for _ in range(nnodes):
            p = mp.Process(
                target=run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # restart odd agents
        for i in range(nnodes):
            if i % 2 != 0:
                procs[i].kill()
                p = mp.Process(
                    target=run_agent,
                    kwargs=agent_args,
                )
                procs[i] = p
                p.start()

        for i in range(nnodes):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_double_agent_elastic(self):
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
            "rdzv_params": self._get_etcd_rdzv_params(1),
            "conf": node_conf,
            "log_dir": self.log_dir(),
            "agent_results": agent_results,
            "max_restarts": 2,
        }

        procs = []
        for _ in range(max_nodes):
            p = mp.Process(
                target=run_agent,
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
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_torch_rpc(self):
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

        results = run_job(
            self._get_etcd_rdzv_params(len(node_configs)), self.log_dir(), node_configs
        )
        master_retvals = results["master"][0].return_values
        # there is only one master but the global rank is not stable
        # so compare the master return value as a collection
        self.assertEqual([f"{msg} from worker"], list(master_retvals.values()))

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_workers_drift_success(self):
        """
        two agents (one worker each) finishes within ``sec`` seconds of each other,
        exit barrier timeout set to ``sec * 2 * 2``.
        """

        sec = 1
        node_configs = [
            Conf(role="zzz", entrypoint=_sleep, args=(0 * sec,), local_world_size=1),
            Conf(role="zzz", entrypoint=_sleep, args=(2 * sec,), local_world_size=1),
        ]
        results = run_job(
            self._get_etcd_rdzv_params(len(node_configs)),
            self.log_dir(),
            node_configs,
            exit_barrier_timeout=2 * 2 * sec,
        )
        for i in range(2):
            run_results = results["zzz"][i]
            self.assertFalse(run_results.is_failed())
            for rank, output in run_results.return_values.items():
                # _sleep() returns its own rank
                self.assertEqual(rank, output)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_workers_drift_fail(self):
        """
        two agents (one worker each) finishes within ``4 x sec`` seconds of each other,
        exit barrier timeout set to 0. Exit barriers should NOT fail the job.
        """
        sec = 1
        node_configs = [
            Conf(role="zzz", entrypoint=_sleep, args=(0 * sec,), local_world_size=1),
            Conf(role="zzz", entrypoint=_sleep, args=(4 * sec,), local_world_size=1),
        ]
        results = run_job(
            self._get_etcd_rdzv_params(len(node_configs)),
            self.log_dir(),
            node_configs,
            exit_barrier_timeout=0,
        )
        for i in range(2):
            run_results = results["zzz"][i]
            self.assertFalse(run_results.is_failed())
            for rank, output in run_results.return_values.items():
                # _sleep() returns its own rank
                self.assertEqual(rank, output)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    @patch("torch.distributed.elastic.utils.store.barrier")
    def test_barrier_failed(self, barrier_mock):
        """
        Failure during the barrier should NOT fail the job.
        """
        barrier_mock.side_effect = RuntimeError("test error")
        res = run_agent(
            self._get_etcd_rdzv_params(1),
            Conf(entrypoint=_happy_function, local_world_size=1),
            self.log_dir(),
        )
        self.assertFalse(res.is_failed())
        barrier_mock.assert_called_once()

    @patch("torch.distributed.elastic.agent.server.local_elastic_agent.start_processes")
    def test_shutdown_called(self, start_processes_mock):
        pcontext_mock = Mock()
        pcontext_mock.pids.return_value = {0: 0}
        start_processes_mock.return_value = pcontext_mock
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)
        rdzv_params = self._get_etcd_rdzv_params(1)
        spec = get_worker_spec(rdzv_params, node_conf, max_restarts=0)
        agent = get_agent(spec, self.log_dir())
        with patch.object(agent, "_monitor_workers") as monitor_mock:
            monitor_mock.return_value = RunResult(
                state=WorkerState.SUCCEEDED, return_values={0: 0}
            )
            agent.run("worker")
        pcontext_mock.close.assert_called_once()
