#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
import uuid
from contextlib import closing
from typing import Any, Optional
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.distributed as dist
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState
from torch.distributed.elastic.multiprocessing.api import SignalException
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.utils import get_socket_with_port
from torch.distributed.launcher.api import (
    _get_entrypoint_name,
    elastic_launch,
    launch_agent,
    LaunchConfig,
)
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


def simple_rank_scale():
    rank = int(os.environ["RANK"])
    return 10 + rank


def function_with_bug():
    raise RuntimeError("test error")


def get_test_launch_config(
    rdzv_endpoint: str,
    min_nodes: int,
    max_nodes: int,
    nproc_per_node: int,
    run_id: str = "",
    rdzv_backend: str = "etcd",
    config: Optional[dict[str, Any]] = None,
) -> LaunchConfig:
    rdzv_configs = {}
    if config:
        rdzv_configs.update(config)
    return LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id=run_id,
        rdzv_endpoint=rdzv_endpoint,
        monitor_interval=0.1,
        rdzv_backend=rdzv_backend,
        start_method="spawn",
        max_restarts=0,
        rdzv_configs=rdzv_configs,
    )


def elastic_launch_wrapper(
    test_dir: str,
    rdzv_endpoint: str,
    min_nodes: int,
    max_nodes: int,
    nproc_per_node: int,
    run_id: str,
):
    """A wrapper function for class `elastic_launch.` in order to make multiprocess returns correct exit code."""
    elastic_launch(
        get_test_launch_config(
            rdzv_endpoint, min_nodes, max_nodes, nproc_per_node, run_id
        ),
        sys.executable,
    )("-u", path("bin/test_script.py"), f"--touch-file-dir={test_dir}")


def _dist_sum(wait=0):
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="gloo")
    t = torch.tensor(rank)

    time.sleep(wait)
    dist.all_reduce(t, op=dist.reduce_op.SUM)
    return t.item()


ELASTIC_AGENT_RUN = "torch.distributed.launcher.api.LocalElasticAgent.run"
EVENTS_RECORD = "torch.distributed.launcher.api.events.record"
GET_RDZV_HANDLER = (
    "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
)


class MockException(Exception):
    pass


def short_hash():
    return str(uuid.uuid4()).split("-")[0]


class ElasticLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests.
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()
        cls._etcd_endpoint = cls._etcd_server.get_endpoint()

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server.
        cls._etcd_server.stop()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # remove any lingering environment variables.
        for env in os.environ.keys():  # noqa: SIM118
            if env.startswith("PET_"):
                del os.environ[env]

        # set a sentinel env var on the parent proc.
        # this should be present on the child and gets
        # asserted in ``bin/test_script.py``.
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"
        os.environ["OMP_NUM_THREADS"] = str(1)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def check_works_ran(self, world_size: int):
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_python(self):
        nnodes = 1
        nproc_per_node = 4

        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

        # make sure all the workers ran.
        # each worker touches a file with its global rank as the name.
        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_python_local_rank_transfer(self):
        nnodes = 1
        nproc_per_node = 4

        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

        # make sure all the workers ran.
        # each worker touches a file with its global rank as the name.
        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_bash(self):
        nnodes = 1
        nproc_per_node = 4

        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            path("bin/test_script.sh"),
        )(f"{self.test_dir}")

        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_function(self):
        nnodes = 1
        nproc_per_node = 4

        res = elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            simple_rank_scale,
        )()

        expected_res = [10, 11, 12, 13]
        actual_res = sorted(value for value in res.values())
        self.assertEqual(expected_res, actual_res)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_dist_sum_with_static_rdzv(self):
        nnodes = 1
        nproc_per_node = 4
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        rdzv_endpoint = f"127.0.0.1:{master_port}"
        rank = 0
        rdzv_config = {
            "rank": rank,
        }

        res = elastic_launch(
            get_test_launch_config(
                rdzv_endpoint,
                nnodes,
                nnodes,
                nproc_per_node,
                rdzv_backend="static",
                config=rdzv_config,
            ),
            _dist_sum,
        )()

        expected_res = [sum(range(nproc_per_node))] * nproc_per_node
        actual_res = sorted(value for value in res.values())
        self.assertEqual(expected_res, actual_res)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic(self):
        nproc_per_node = 4

        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, 1, 2, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

        world_size = nproc_per_node
        self.check_works_ran(world_size)

    @mock.patch("torch.distributed.elastic.events.record")
    def test_launch_elastic_worker_raise_exception(self, record_mock):
        """
        Asserts that when the worker program fails and lancher raieses exception
        to indicate that worker process failed.
        """
        nproc_per_node = 4

        with self.assertRaises(ChildFailedError):
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 2, nproc_per_node),
                sys.executable,
            )("-u", path("bin/test_script.py"), "--fail")

        record_mock.assert_called_once()

    @mock.patch("torch.distributed.elastic.events.record")
    @mock.patch(
        "torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent.run"
    )
    def test_launch_elastic_agent_raise_exception(self, record_mock, mock_agent_run):
        """
        Asserts that when the agent raises an exception
        the launcher re-raises the original exception.
        """
        mock_agent_run.side_effect = MockException
        with self.assertRaises(MockException):
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 2, 4),
                sys.executable,
            )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_multiple_agents(self):
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        nnodes = 2
        run_id = str(uuid.uuid4().int)

        procs = []
        ctx = mp.get_context("spawn")
        for _ in range(nnodes - 1):
            p = ctx.Process(
                target=elastic_launch_wrapper,
                args=(
                    self.test_dir,
                    self._etcd_endpoint,
                    min_nodes,
                    max_nodes,
                    nproc_per_node,
                    run_id,
                ),
            )
            procs.append(p)
            p.start()

        elastic_launch_wrapper(
            self.test_dir,
            self._etcd_endpoint,
            min_nodes,
            max_nodes,
            nproc_per_node,
            run_id,
        )

        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        world_size = nnodes * nproc_per_node
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    def test_launch_shutdown(self, agent_mock_cls):
        agent_mock = Mock()
        agent_mock.run.return_value = RunResult(WorkerState.SUCCEEDED)
        agent_mock_cls.return_value = agent_mock
        rdzv_handler_mock = Mock()
        with patch(
            "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
        ) as param_mock:
            param_mock.return_value = rdzv_handler_mock
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 1, 4),
                sys.executable,
            )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

            rdzv_handler_mock.shutdown.assert_called_once()

    def test_get_entrypoint_name(self):
        self.assertEqual(
            "simple_rank_scale", _get_entrypoint_name(simple_rank_scale, [])
        )
        self.assertEqual("", _get_entrypoint_name(sys.executable, []))
        self.assertEqual("", _get_entrypoint_name(sys.executable, ["-u"]))
        self.assertEqual(
            "test_script.py",
            _get_entrypoint_name(sys.executable, ["-u", "test_script.py"]),
        )
        self.assertEqual("", _get_entrypoint_name(None, []))

    @patch(ELASTIC_AGENT_RUN)
    @patch(GET_RDZV_HANDLER)
    def test_rdzv_handler_shutdown_on_agent_signal(self, mock_get_rdzv, mock_agent_run):
        config = get_test_launch_config(
            self._etcd_endpoint, min_nodes=1, max_nodes=1, nproc_per_node=1
        )

        for sigval in [signal.SIGTERM, signal.SIGINT]:
            with patch(EVENTS_RECORD) as record_event_mock:
                rdzv_handler_mock = MagicMock()
                rdzv_handler_mock.get_run_id.return_value = short_hash()
                mock_get_rdzv.return_value = rdzv_handler_mock

                mock_agent_run.side_effect = SignalException("test", sigval)
                with self.assertRaises(SignalException):
                    launch_agent(config, simple_rank_scale, [])
                rdzv_handler_mock.shutdown.assert_not_called()
                record_event_mock.assert_called_once()

    @patch(ELASTIC_AGENT_RUN)
    @patch(GET_RDZV_HANDLER)
    def test_rdzv_handler_shutdown_on_agent_error(self, mock_get_rdzv, mock_agent_run):
        config = get_test_launch_config(
            self._etcd_endpoint, min_nodes=1, max_nodes=1, nproc_per_node=1
        )

        with patch(EVENTS_RECORD) as record_event_mock:
            rdzv_handler_mock = MagicMock()
            rdzv_handler_mock.get_run_id.return_value = short_hash()
            mock_get_rdzv.return_value = rdzv_handler_mock

            mock_agent_run.side_effect = RuntimeError("any other exception")
            with self.assertRaises(RuntimeError):
                launch_agent(config, simple_rank_scale, [])
            rdzv_handler_mock.shutdown.assert_called_once()
            record_event_mock.assert_called_once()


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
