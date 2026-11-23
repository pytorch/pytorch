#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
import tempfile
import time
from unittest.mock import MagicMock, Mock, patch

from torch.distributed.elastic.agent.server.api import SimpleElasticAgent, WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.launcher.api import LaunchConfig
from torch.testing._internal.common_utils import run_tests, TestCase


def _sleep_long(duration: int):
    """Worker function that sleeps for a specified duration."""
    time.sleep(duration)
    return int(os.environ["RANK"])


def _make_mock_spec() -> Mock:
    """A plain Mock WorkerSpec with the attributes the agents touch."""
    mock_spec = Mock()
    mock_spec.role = "test_role"
    mock_spec.local_world_size = 1
    mock_spec.max_restarts = 3
    mock_spec.rdzv_handler = Mock()
    mock_spec.rdzv_handler.use_agent_store = False
    mock_spec.rdzv_handler.get_run_id.return_value = "test_run_id"
    return mock_spec


class _ConcreteAgent(SimpleElasticAgent):
    """Minimal concrete SimpleElasticAgent for testing base-class behavior."""

    def _start_workers(self, worker_group):
        return {}

    def _stop_workers(self, worker_group):
        pass

    def _monitor_workers(self, worker_group):
        raise NotImplementedError

    def _shutdown(self, death_sig=signal.SIGTERM, timeout=30):
        pass


class ShutdownTimeoutTest(TestCase):
    """Tests for the configurable shutdown_timeout feature."""

    def test_launch_config_default_shutdown_timeout(self):
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
        )
        self.assertEqual(config.shutdown_timeout, 30)

    def test_launch_config_custom_shutdown_timeout(self):
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            shutdown_timeout=120,
        )
        self.assertEqual(config.shutdown_timeout, 120)

    def test_launch_config_env_var_shutdown_timeout(self):
        with patch.dict(os.environ, {"TORCH_ELASTIC_SHUTDOWN_TIMEOUT": "600"}):
            config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
            )
            self.assertEqual(config.shutdown_timeout, 600)

    def test_launch_config_explicit_overrides_env(self):
        with patch.dict(os.environ, {"TORCH_ELASTIC_SHUTDOWN_TIMEOUT": "600"}):
            config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                shutdown_timeout=120,
            )
            self.assertEqual(config.shutdown_timeout, 120)

    def test_simple_elastic_agent_receives_shutdown_timeout(self):
        mock_spec = Mock(spec=WorkerSpec)
        mock_spec.max_restarts = 3

        agent = SimpleElasticAgent(
            spec=mock_spec,
            exit_barrier_timeout=300,
            shutdown_timeout=180,
        )

        self.assertEqual(agent._shutdown_timeout, 180)

    def test_simple_elastic_agent_default_shutdown_timeout(self):
        mock_spec = Mock(spec=WorkerSpec)
        mock_spec.max_restarts = 3

        agent = SimpleElasticAgent(
            spec=mock_spec,
            exit_barrier_timeout=300,
        )

        self.assertEqual(agent._shutdown_timeout, 30)

    def test_local_elastic_agent_receives_shutdown_timeout(self):
        mock_spec = Mock(spec=WorkerSpec)
        mock_spec.max_restarts = 3
        mock_spec.rdzv_handler = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_specs = DefaultLogsSpecs(log_dir=tmpdir)

            agent = LocalElasticAgent(
                spec=mock_spec,
                logs_specs=logs_specs,
                start_method="spawn",
                exit_barrier_timeout=300,
                shutdown_timeout=240,
            )

            self.assertEqual(agent._shutdown_timeout, 240)

    def test_shutdown_method_receives_correct_timeout(self):
        mock_spec = Mock(spec=WorkerSpec)
        mock_spec.max_restarts = 3
        mock_spec.rdzv_handler = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_specs = DefaultLogsSpecs(log_dir=tmpdir)

            agent = LocalElasticAgent(
                spec=mock_spec,
                logs_specs=logs_specs,
                start_method="spawn",
                exit_barrier_timeout=300,
                shutdown_timeout=150,
            )

            # Mock the _pcontext to avoid actual process operations
            mock_pcontext = MagicMock()
            agent._pcontext = mock_pcontext

            # Call _shutdown directly
            agent._shutdown(death_sig=signal.SIGTERM, timeout=150)

            # Verify close was called with correct timeout
            mock_pcontext.close.assert_called_once_with(signal.SIGTERM, 150)

    def test_stop_workers_uses_configured_shutdown_timeout(self):
        mock_spec = Mock(spec=WorkerSpec)
        mock_spec.max_restarts = 3
        mock_spec.rdzv_handler = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_specs = DefaultLogsSpecs(log_dir=tmpdir)
            agent = LocalElasticAgent(
                spec=mock_spec,
                logs_specs=logs_specs,
                start_method="spawn",
                shutdown_timeout=150,
            )
            agent._pcontext = MagicMock()

            agent._stop_workers(agent.get_worker_group())

            agent._pcontext.close.assert_called_once_with(signal.SIGTERM, 150)

    @patch("torch.distributed.elastic.multiprocessing.api.PContext")
    def test_pcontext_close_receives_timeout(self, mock_pcontext_class):
        mock_pcontext = MagicMock()
        mock_pcontext_class.return_value = mock_pcontext

        # Simulate calling close with a custom timeout
        mock_pcontext.close(death_sig=signal.SIGTERM, timeout=200)

        # Verify it was called with the correct timeout
        mock_pcontext.close.assert_called_once_with(
            death_sig=signal.SIGTERM, timeout=200
        )

    def test_shutdown_timeout_validation_negative(self):
        with self.assertRaises(ValueError) as cm:
            LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                shutdown_timeout=-5,
            )
        self.assertIn("shutdown_timeout must be non-negative", str(cm.exception))

    def test_shutdown_timeout_zero(self):
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            shutdown_timeout=0,
        )
        self.assertEqual(config.shutdown_timeout, 0)

    def test_launch_config_env_var_float_shutdown_timeout(self):
        with patch.dict(os.environ, {"TORCH_ELASTIC_SHUTDOWN_TIMEOUT": "2.5"}):
            config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
            )
            self.assertEqual(config.shutdown_timeout, 2.5)

    def test_launch_config_env_var_invalid_shutdown_timeout(self):
        with patch.dict(os.environ, {"TORCH_ELASTIC_SHUTDOWN_TIMEOUT": "bogus"}):
            with self.assertRaises(ValueError):
                LaunchConfig(
                    min_nodes=1,
                    max_nodes=1,
                    nproc_per_node=2,
                )

    def test_simple_elastic_agent_env_var_shutdown_timeout(self):
        mock_spec = _make_mock_spec()

        with patch.dict(os.environ, {"TORCH_ELASTIC_SHUTDOWN_TIMEOUT": "45"}):
            agent = _ConcreteAgent(
                spec=mock_spec,
                exit_barrier_timeout=300,
            )
            self.assertEqual(agent._shutdown_timeout, 45)

    def test_simple_elastic_agent_explicit_overrides_env_var(self):
        mock_spec = _make_mock_spec()

        with patch.dict(os.environ, {"TORCH_ELASTIC_SHUTDOWN_TIMEOUT": "45"}):
            agent = _ConcreteAgent(
                spec=mock_spec,
                exit_barrier_timeout=300,
                shutdown_timeout=10,
            )
            self.assertEqual(agent._shutdown_timeout, 10)

    def test_simple_elastic_agent_negative_shutdown_timeout_raises(self):
        mock_spec = _make_mock_spec()

        with self.assertRaises(ValueError) as cm:
            _ConcreteAgent(
                spec=mock_spec,
                exit_barrier_timeout=300,
                shutdown_timeout=-1,
            )
        self.assertIn("shutdown_timeout must be non-negative", str(cm.exception))

    def test_local_elastic_agent_env_var_shutdown_timeout(self):
        mock_spec = _make_mock_spec()

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_specs = DefaultLogsSpecs(log_dir=tmpdir)
            with patch.dict(os.environ, {"TORCH_ELASTIC_SHUTDOWN_TIMEOUT": "90"}):
                agent = LocalElasticAgent(
                    spec=mock_spec,
                    logs_specs=logs_specs,
                    start_method="spawn",
                )
                self.assertEqual(agent._shutdown_timeout, 90)

    def test_start_workers_passes_shutdown_timeout_to_start_processes(self):
        mock_spec = _make_mock_spec()
        mock_spec.entrypoint = _sleep_long
        mock_spec.args = ()
        mock_spec.numa_options = None
        mock_spec.duplicate_stdout_filters = None
        mock_spec.duplicate_stderr_filters = None
        mock_spec.virtual_local_rank = None

        worker = Mock()
        worker.local_rank = 0
        worker.global_rank = 0
        worker.role_rank = 0
        worker.world_size = 1
        worker.role_world_size = 1

        worker_group = Mock()
        worker_group.spec = mock_spec
        worker_group.store = Mock()
        worker_group.workers = [worker]
        worker_group.group_rank = 0
        worker_group.group_world_size = 1
        worker_group.master_addr = "localhost"
        worker_group.master_port = 29500

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_specs = DefaultLogsSpecs(log_dir=tmpdir)
            agent = LocalElasticAgent(
                spec=mock_spec,
                logs_specs=logs_specs,
                start_method="spawn",
                shutdown_timeout=150,
            )
            env = dict(os.environ)
            env.pop("TORCHELASTIC_ENABLE_FILE_TIMER", None)
            env.pop("TORCHELASTIC_TIMER_FILE", None)
            env.pop("TORCHELASTIC_HEALTH_CHECK_PORT", None)
            with patch.dict(os.environ, env, clear=True):
                with patch(
                    "torch.distributed.elastic.agent.server.local_elastic_agent.start_processes"
                ) as mock_start_processes:
                    agent._start_workers(worker_group)

            mock_start_processes.assert_called_once()
            self.assertEqual(
                mock_start_processes.call_args.kwargs["shutdown_timeout"], 150
            )
            # the pcontext created by start_processes is stored on the agent
            self.assertIs(agent._pcontext, mock_start_processes.return_value)


if __name__ == "__main__":
    run_tests()
