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


if __name__ == "__main__":
    run_tests()
