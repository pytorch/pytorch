#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import os
import signal
import unittest
import uuid
from multiprocessing.pool import ThreadPool
from typing import Any
from unittest.mock import call, patch

import torch.distributed as dist
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic.agent.server.api import (
    _get_fq_hostname,
    _RoleInstanceInfo,
    RunResult,
    SimpleElasticAgent,
    Worker,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.multiprocessing import SignalException
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from torch.distributed.elastic.rendezvous import RendezvousHandler, RendezvousParameters
from torch.distributed.elastic.rendezvous.api import RendezvousGracefulExitError
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.testing._internal.common_utils import run_tests


def do_nothing():
    pass


class WorkerStateTest(unittest.TestCase):
    def test_is_running(self):
        for state in WorkerState:
            if state == WorkerState.HEALTHY or state == WorkerState.UNHEALTHY:
                self.assertTrue(WorkerState.is_running(state))
            else:
                self.assertFalse(WorkerState.is_running(state))


class WorkerGroupTest(unittest.TestCase):
    def test_worker_group_constructor(self):
        spec = WorkerSpec(
            role="test_trainer",
            local_world_size=4,
            fn=do_nothing,
            args=(),
            rdzv_handler=None,
            max_restarts=50,
            monitor_interval=0.1,
        )
        worker_group = WorkerGroup(spec)

        self.assertEqual(WorkerState.INIT, worker_group.state)

        workers = worker_group.workers
        self.assertEqual(4, len(workers))

        # validate full, consecutive local ranks
        self.assertSetEqual(set(range(4)), {w.local_rank for w in workers})

        # global_rank, world_size are assigned after rdzv
        # id is assigned after starting worker (by the agent)
        # validate there are None
        for w in workers:
            self.assertEqual(-1, w.global_rank)
            self.assertEqual(-1, w.world_size)
            self.assertEqual(None, w.id)

        # rank and store are assigned after rdzv; validate that they are None
        self.assertIsNone(worker_group.group_rank)
        self.assertIsNone(worker_group.store)


class RoleInstanceInfoTest(unittest.TestCase):
    def test_compare(self):
        agent_role1 = _RoleInstanceInfo("role", 1, 10)
        agent_role2 = _RoleInstanceInfo("role", 2, 10)
        self.assertEqual(1, _RoleInstanceInfo.compare(agent_role2, agent_role1))
        agent_role1 = _RoleInstanceInfo("role1", 1, 10)
        agent_role2 = _RoleInstanceInfo("role2", 2, 10)
        self.assertEqual(-1, _RoleInstanceInfo.compare(agent_role1, agent_role2))
        agent_role1 = _RoleInstanceInfo("role1", 1, 10)
        agent_role2 = _RoleInstanceInfo("role2", 1, 10)
        self.assertEqual(-1, _RoleInstanceInfo.compare(agent_role1, agent_role2))

    def test_serde(self):
        agent_role = _RoleInstanceInfo("role", 1, 10)
        str_data = agent_role.serialize()
        actual_agent_role = _RoleInstanceInfo.deserialize(str_data)
        self.assertEqual(agent_role.role, actual_agent_role.role)
        self.assertEqual(agent_role.rank, actual_agent_role.rank)
        self.assertEqual(
            agent_role.local_world_size, actual_agent_role.local_world_size
        )

    def test_find_boundaries(self):
        role_infos = [
            _RoleInstanceInfo("trainer", 1, 1),
            _RoleInstanceInfo("trainer", 2, 2),
            _RoleInstanceInfo("trainer", 3, 3),
            _RoleInstanceInfo("parameter_server", 4, 5),
            _RoleInstanceInfo("parameter_server", 0, 4),
        ]
        start_idx, end_idx = _RoleInstanceInfo.find_role_boundaries(
            role_infos, "trainer"
        )
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 2)


class TestAgent(SimpleElasticAgent):
    def __init__(self, spec):
        super().__init__(spec)
        self.stop_workers_call_count = 0
        self.start_workers_call_count = 0

    def _stop_workers(
        self, worker_group: WorkerGroup, is_restart: bool = False
    ) -> None:
        # workers are fake, nothing to stop; just clear the rdzv info
        worker_group.group_rank = None
        worker_group.group_world_size = None
        self.stop_workers_call_count += 1

    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        # crate fake workers; make worker id equal to global rank
        ids = {}
        for worker in worker_group.workers:
            ids[worker.local_rank] = worker.global_rank
        self.start_workers_call_count += 1
        return ids

    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        raise NotImplementedError("mock this method")

    def _shutdown(self):
        pass


def monres(state: WorkerState):
    if state == WorkerState.SUCCEEDED:
        return RunResult(state=state, return_values={0: 0}, failures={})
    elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
        pf = ProcessFailure(local_rank=0, pid=999, exitcode=1, error_file="<none>")
        return RunResult(state=state, return_values={}, failures={0: pf})
    else:
        return RunResult(state=state)


class SimpleElasticAgentTest(unittest.TestCase):
    def _get_worker_spec(
        self,
        max_restarts=1,
        monitor_interval=0.1,
        role="test_trainer",
        local_world_size=8,
        local_addr=None,
    ):
        run_id = str(uuid.uuid4().int)
        port = get_free_port()
        if local_addr is None:
            endpoint = f"127.0.0.1:{port}"
        else:
            endpoint = f"{local_addr}:{port}"

        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint=endpoint,
            run_id=run_id,
            min_nodes=1,
            max_nodes=1,
            rank=0,
        )
        rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_params)
        spec = WorkerSpec(
            role=role,
            local_world_size=local_world_size,
            fn=do_nothing,
            args=(),
            rdzv_handler=rdzv_handler,
            max_restarts=max_restarts,
            monitor_interval=monitor_interval,
            local_addr=local_addr,
        )
        return spec

    def test_agent_constructor(self):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()
        self.assertEqual(WorkerState.INIT, worker_group.state)
        self.assertEqual(spec.max_restarts, agent._remaining_restarts)

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    def test_record_flakiness_metric(self, put_metric_mock):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        agent._record_flakiness_metric()
        put_metric_mock.assert_called_with("workers.test_trainer.flakiness", 0)
        agent._worker_group.spec.max_restarts = 10
        agent._remaining_restarts = 3
        agent._record_flakiness_metric()
        put_metric_mock.assert_called_with("workers.test_trainer.flakiness", 63)

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    def test_record_flakiness_metric_zero_restarts(self, put_metric_mock):
        spec = self._get_worker_spec(max_restarts=1)
        spec.max_restarts = 0
        agent = TestAgent(spec)
        agent._record_flakiness_metric()
        put_metric_mock.assert_called_with("workers.test_trainer.flakiness", 0)

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    def test_record_flakiness_metric_user_exception(self, put_metric_mock):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        agent._record_flakiness_metric(True)
        put_metric_mock.assert_called_with("workers.test_trainer.flakiness", 100)

    @patch.object(TestAgent, "_invoke_run")
    @patch.object(TestAgent, "_record_metrics")
    @patch.object(TestAgent, "_record_worker_events")
    @patch.object(TestAgent, "_shutdown")
    def test_invoke_run(
        self, shutdown_mock, record_events_mock, record_metrics_mock, invoke_run_mock
    ):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        agent.run()
        invoke_run_mock.assert_called_once()
        record_metrics_mock.assert_called_once()
        record_events_mock.assert_called_once()
        shutdown_mock.assert_called_once()

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    def test_record_metrics_success_no_retries(self, put_metric_mock):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        group_result = RunResult({}, {})
        agent._record_metrics(group_result)
        calls = self._get_record_metrics_test_calls(success_no_retries=1)
        put_metric_mock.assert_has_calls(calls, any_order=True)

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    def test_record_metrics_success_with_retries(self, put_metric_mock):
        spec = self._get_worker_spec(max_restarts=10)
        agent = TestAgent(spec)
        agent._remaining_restarts = 2
        group_result = RunResult({}, {})
        agent._record_metrics(group_result)
        calls = self._get_record_metrics_test_calls(success_with_retries=1)
        put_metric_mock.assert_has_calls(calls, any_order=True)

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    def test_record_metrics_failed_with_retries(self, put_metric_mock):
        spec = self._get_worker_spec(max_restarts=10)
        agent = TestAgent(spec)
        agent._remaining_restarts = 2
        group_result = RunResult(
            state=WorkerState.FAILED, return_values={}, failures={0: 0}
        )
        agent._record_metrics(group_result)
        calls = self._get_record_metrics_test_calls(failed_with_retries=1)
        put_metric_mock.assert_has_calls(calls, any_order=True)

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    def test_record_metrics_failed_no_retries(self, put_metric_mock):
        spec = self._get_worker_spec(max_restarts=10)
        agent = TestAgent(spec)
        group_result = RunResult(
            state=WorkerState.FAILED, return_values={}, failures={0: 0}
        )
        agent._record_metrics(group_result)
        calls = self._get_record_metrics_test_calls(failed_no_retries=1)
        put_metric_mock.assert_has_calls(calls, any_order=True)

    def _get_record_metrics_test_calls(
        self,
        success_with_retries=0,
        success_no_retries=0,
        failed_with_retries=0,
        failed_no_retries=0,
    ):
        calls = [
            call("workers.test_trainer.run_success_with_retries", success_with_retries),
            call("workers.test_trainer.run_success_no_retries", success_no_retries),
            call("workers.test_trainer.run_failed_with_retries", failed_with_retries),
            call("workers.test_trainer.run_failed_no_retries", failed_no_retries),
        ]
        return calls

    def test_rendezvous(self):
        hostname = _get_fq_hostname()
        spec = self._get_worker_spec(max_restarts=1, local_addr=hostname)
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()
        agent._rendezvous(worker_group)

        # single agent rdzv
        self.assertEqual(1, worker_group.group_world_size)
        self.assertEqual(0, worker_group.group_rank)

        self.assertEqual(hostname, worker_group.master_addr)
        self.assertTrue(worker_group.master_port > 0)

        rank_set = {w.global_rank for w in worker_group.workers}
        for w in worker_group.workers:
            self.assertIsNone(w.id)
            local_world_size = spec.local_world_size
            group_world_size = worker_group.group_world_size
            group_rank = worker_group.group_rank

            self.assertEqual(local_world_size * group_world_size, w.world_size)
            self.assertEqual(
                local_world_size * group_rank + w.local_rank, w.global_rank
            )
            self.assertSetEqual(set(range(w.world_size)), rank_set)

    def test_rendezvous_default_master_addr(self):
        hostname = _get_fq_hostname()
        spec = self._get_worker_spec(max_restarts=1, local_addr=hostname)
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()
        agent._rendezvous(worker_group)

        self.assertEqual(_get_fq_hostname(), worker_group.master_addr)
        self.assertGreater(worker_group.master_port, 0)

    def test_rendezvous_master_addr_with_local_addr(self):
        spec_local_addr = "127.0.0.1"
        spec = self._get_worker_spec(max_restarts=1, local_addr=spec_local_addr)
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()
        agent._rendezvous(worker_group)

        self.assertNotEqual(_get_fq_hostname(), worker_group.master_addr)
        self.assertEqual(spec_local_addr, worker_group.master_addr)
        self.assertGreater(worker_group.master_port, 0)

    @patch.object(TestAgent, "_construct_event")
    def test_initialize_workers(self, mock_construct_event):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()
        agent._initialize_workers(worker_group)

        self.assertEqual(WorkerState.HEALTHY, worker_group.state)
        for i in range(spec.local_world_size):
            worker = worker_group.workers[i]
            self.assertEqual(worker.id, worker.global_rank)

        mock_construct_event.assert_called()
        self.assertEqual(mock_construct_event.call_count, 10)

    def test_restart_workers(self):
        spec = self._get_worker_spec()
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()

        num_restarts = 3
        for _ in range(0, num_restarts):
            agent._restart_workers(worker_group)
            self.assertEqual(WorkerState.HEALTHY, worker_group.state)

            # test_rendezvous and test_initialize_workers
            # already validates the correctness of these fields
            # simply validate that they are not None
            # (e.g. that they get assigned)
            self.assertIsNotNone(worker_group.group_rank)
            self.assertIsNotNone(worker_group.group_world_size)
            for w in worker_group.workers:
                self.assertIsNotNone(w.id)
                self.assertIsNotNone(w.global_rank)
                self.assertIsNotNone(w.world_size)

        self.assertEqual(num_restarts, agent.start_workers_call_count)
        self.assertEqual(num_restarts, agent.stop_workers_call_count)

    @patch.object(
        TestAgent,
        "_monitor_workers",
        side_effect=[
            monres(WorkerState.HEALTHY),
            monres(WorkerState.HEALTHY),
            monres(WorkerState.SUCCEEDED),
        ],
    )
    @patch.object(TestAgent, "_record_worker_events")
    def test_run_happy_path(self, record_events_mock, mock_monitor_workers):
        # worker starts
        # is always healthy
        # then succeeds
        max_restarts = 10
        spec = self._get_worker_spec(max_restarts)
        agent = TestAgent(spec)

        agent.run()

        # no failure, no membership changes -> no retries
        self.assertEqual(max_restarts, agent._remaining_restarts)
        record_events_mock.assert_called_once()

    @patch.object(TestAgent, "_initialize_workers", side_effect=RuntimeError())
    def test_run_initialization_failure(self, mock_initialize_workers):
        spec = self._get_worker_spec()
        agent = TestAgent(spec)
        worker_group = agent._worker_group

        with self.assertRaises(RuntimeError):
            agent.run()

        self.assertEqual(WorkerState.INIT, worker_group.state)

    def test_run_max_retries_exceeded(self):
        for restartable_state in [
            monres(WorkerState.FAILED),
            monres(WorkerState.UNHEALTHY),
        ]:
            with patch.object(
                TestAgent, "_monitor_workers", return_value=restartable_state
            ) as mock_monitor_workers:
                spec = self._get_worker_spec(max_restarts=3, monitor_interval=0.1)
                agent = TestAgent(spec)
                worker_group = agent._worker_group

                agent.run()
                self.assertEqual(WorkerState.FAILED, worker_group.state)
                self.assertEqual(0, agent._remaining_restarts)
                # one monitor call for each retry + one to monitor the last retry
                self.assertEqual(spec.max_restarts + 1, mock_monitor_workers.call_count)

    @patch.object(
        TestAgent,
        "_monitor_workers",
        side_effect=[
            monres(WorkerState.HEALTHY),
            monres(WorkerState.HEALTHY),
            monres(WorkerState.HEALTHY),
            monres(WorkerState.SUCCEEDED),
        ],
    )
    @patch.object(RendezvousHandler, "num_nodes_waiting", side_effect=[1, 1, 0])
    @patch.object(TestAgent, "_record_worker_events")
    def test_run_membership_change(
        self, record_events_mock, mock_num_nodes_waiting, mock_monitor_workers
    ):
        spec = self._get_worker_spec(max_restarts=1, monitor_interval=0.1)
        agent = TestAgent(spec)
        worker_group = agent._worker_group

        agent.run()
        self.assertEqual(WorkerState.SUCCEEDED, worker_group.state)
        record_events_mock.assert_called_once()

    @patch.object(
        TestAgent, "_monitor_workers", return_value=monres(WorkerState.UNKNOWN)
    )
    def test_run_unknown_state(self, mock_monitor_workers):
        # when the state is unknown we exit immediately; no retries
        spec = self._get_worker_spec(max_restarts=100, monitor_interval=0.1)
        agent = TestAgent(spec)
        worker_group = agent._worker_group

        with self.assertRaises(Exception):
            agent.run()

        self.assertEqual(WorkerState.UNKNOWN, worker_group.state)
        self.assertEqual(1, mock_monitor_workers.call_count)
        self.assertEqual(spec.max_restarts, agent._remaining_restarts)

    def get_worker_assigned(self, store, role_infos_len, info) -> list[Worker]:
        i, role_info = info
        spec = self._get_worker_spec(
            max_restarts=3,
            monitor_interval=0.1,
            role=role_info.role,
            local_world_size=role_info.local_world_size,
        )
        agent = TestAgent(spec)
        workers = agent._assign_worker_ranks(
            store, role_info.rank, role_infos_len, spec
        )
        return [
            (
                w.local_rank,
                w.role_rank,
                w.global_rank,
                w.world_size,
                w.role_world_size,
            )
            for w in workers
        ]

    def test_assign_worker_ranks(self):
        role_infos = [
            _RoleInstanceInfo("parameter_server", 0, 4),
            _RoleInstanceInfo("trainer", 1, 1),
            _RoleInstanceInfo("trainer", 2, 2),
            _RoleInstanceInfo("trainer", 3, 3),
            _RoleInstanceInfo("parameter_server", 4, 5),
        ]
        store = dist.HashStore()

        f = functools.partial(self.get_worker_assigned, store, len(role_infos))

        with ThreadPool(len(role_infos)) as pool:
            out = pool.map(f, enumerate(role_infos))

        self.assertListEqual(
            out,
            [
                [
                    (0, 0, 0, 15, 9),
                    (1, 1, 1, 15, 9),
                    (2, 2, 2, 15, 9),
                    (3, 3, 3, 15, 9),
                ],
                [
                    (0, 0, 4, 15, 6),
                ],
                [
                    (0, 1, 5, 15, 6),
                    (1, 2, 6, 15, 6),
                ],
                [
                    (0, 3, 7, 15, 6),
                    (1, 4, 8, 15, 6),
                    (2, 5, 9, 15, 6),
                ],
                [
                    (0, 4, 10, 15, 9),
                    (1, 5, 11, 15, 9),
                    (2, 6, 12, 15, 9),
                    (3, 7, 13, 15, 9),
                    (4, 8, 14, 15, 9),
                ],
            ],
        )

    def test_assign_worker_ranks_indentical(self):
        os.environ["TORCH_ELASTIC_WORKER_IDENTICAL"] = "1"
        role_infos = [
            _RoleInstanceInfo("trainer", 0, 4),
            _RoleInstanceInfo("trainer", 1, 4),
            _RoleInstanceInfo("trainer", 2, 4),
            _RoleInstanceInfo("trainer", 3, 4),
            _RoleInstanceInfo("trainer", 4, 4),
        ]
        store = dist.HashStore()

        f = functools.partial(self.get_worker_assigned, store, len(role_infos))

        with ThreadPool(len(role_infos)) as pool:
            out = pool.map(f, enumerate(role_infos))

        self.assertListEqual(
            out,
            [
                [
                    (0, 0, 0, 20, 20),
                    (1, 1, 1, 20, 20),
                    (2, 2, 2, 20, 20),
                    (3, 3, 3, 20, 20),
                ],
                [
                    (0, 4, 4, 20, 20),
                    (1, 5, 5, 20, 20),
                    (2, 6, 6, 20, 20),
                    (3, 7, 7, 20, 20),
                ],
                [
                    (0, 8, 8, 20, 20),
                    (1, 9, 9, 20, 20),
                    (2, 10, 10, 20, 20),
                    (3, 11, 11, 20, 20),
                ],
                [
                    (0, 12, 12, 20, 20),
                    (1, 13, 13, 20, 20),
                    (2, 14, 14, 20, 20),
                    (3, 15, 15, 20, 20),
                ],
                [
                    (0, 16, 16, 20, 20),
                    (1, 17, 17, 20, 20),
                    (2, 18, 18, 20, 20),
                    (3, 19, 19, 20, 20),
                ],
            ],
        )
        os.environ["TORCH_ELASTIC_WORKER_IDENTICAL"] = "0"

    def test_get_event(self):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        event = agent.get_event_succeeded()
        self.assertEqual("AGENT", event.source)
        self.assertEqual("static", event.metadata["rdzv_backend"])
        self.assertEqual("SUCCEEDED", event.metadata["state"])
        self.assertEqual(spec.role, event.metadata["role"])

    def test_get_worker_status_event(self):
        spec = self._get_worker_spec(max_restarts=4)
        agent = TestAgent(spec)
        agent._remaining_restarts = spec.max_restarts - 2
        actual_event = agent._construct_event(
            state="SUCCEEDED",
            source="WORKER",
            worker=agent._worker_group.workers[0],
        )
        self.assertEqual("WORKER", actual_event.source)
        self.assertEqual("static", actual_event.metadata["rdzv_backend"])
        self.assertEqual("SUCCEEDED", actual_event.metadata["state"])
        self.assertEqual(spec.role, actual_event.metadata["role"])
        self.assertEqual(2, actual_event.metadata["agent_restarts"])

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    @patch.object(TestAgent, "_invoke_run")
    def test_agent_process_signal_exception(self, invoke_run, _):
        spec = self._get_worker_spec(max_restarts=0)
        agent = TestAgent(spec)
        invoke_run.side_effect = SignalException(
            "signal exception", sigval=signal.SIGTERM
        )
        with patch.object(agent, "_shutdown") as shutdown_mock:
            with self.assertRaises(SignalException):
                agent.run()
            args, _ = shutdown_mock.call_args
            self.assertEqual(signal.SIGTERM, args[0])

    @patch("torch.distributed.elastic.agent.server.api.put_metric")
    @patch.object(TestAgent, "_invoke_run")
    def test_agent_process_handler_graceful_exception(self, invoke_run, _):
        spec = self._get_worker_spec(max_restarts=0)
        agent = TestAgent(spec)
        invoke_run.side_effect = RendezvousGracefulExitError()
        with patch.object(agent, "_shutdown"):
            agent.run()


if __name__ == "__main__":
    run_tests()
