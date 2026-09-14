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
from typing import Any
from unittest.mock import Mock, patch

from torch.distributed.elastic.agent.server.api import (
    RunResult,
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.launcher.api import LaunchConfig
from torch.testing._internal.common_utils import run_tests, TestCase


# ---------------------------------------------------------------------------
# Concrete stub so we can instantiate SimpleElasticAgent without a full env.
# Mirrors the pattern used in shutdown_timeout_test.py
# ---------------------------------------------------------------------------
class _ConcreteAgent(SimpleElasticAgent):
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        return {}

    def _stop_workers(self, worker_group: WorkerGroup, is_restart: bool = False) -> None:
        pass

    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        return RunResult(state=WorkerState.HEALTHY)

    def _shutdown(
        self, death_sig: signal.Signals = signal.SIGTERM, timeout: int = 30
    ) -> None:
        pass


def _make_mock_spec(local_world_size: int = 1) -> Mock:
    """Return a Mock that satisfies WorkerGroup's attribute requirements."""
    spec = Mock(spec=WorkerSpec)
    spec.max_restarts = 3
    spec.local_world_size = local_world_size
    spec.rdzv_handler = Mock()
    spec.role = "default"
    spec.monitor_interval = 0.1
    spec.master_addr = None
    spec.master_port = None
    return spec


class ExitBarrierTimeoutTest(TestCase):
    """Tests for the configurable exit_barrier_timeout feature.

    Mirrors the structure of shutdown_timeout_test.py.
    """

    # ------------------------------------------------------------------
    # LaunchConfig: default / explicit / env-var / env-var-override
    # ------------------------------------------------------------------

    def test_launch_config_default_exit_barrier_timeout(self):
        """When no value is given, LaunchConfig resolves to the 300-second default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT", None)
            config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=2)
        self.assertEqual(config.exit_barrier_timeout, 300.0)

    def test_launch_config_custom_exit_barrier_timeout(self):
        """An explicit value is stored verbatim."""
        config = LaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=2, exit_barrier_timeout=600
        )
        self.assertEqual(config.exit_barrier_timeout, 600.0)

    def test_launch_config_env_var_exit_barrier_timeout(self):
        """TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT is picked up when no explicit value is given."""
        with patch.dict(os.environ, {"TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT": "900"}):
            config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=2)
        self.assertEqual(config.exit_barrier_timeout, 900.0)

    def test_launch_config_explicit_overrides_env(self):
        """An explicit constructor arg wins over the env var."""
        with patch.dict(os.environ, {"TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT": "900"}):
            config = LaunchConfig(
                min_nodes=1, max_nodes=1, nproc_per_node=2, exit_barrier_timeout=120
            )
        self.assertEqual(config.exit_barrier_timeout, 120.0)

    def test_exit_barrier_timeout_validation_negative(self):
        """A negative explicit value raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            LaunchConfig(
                min_nodes=1, max_nodes=1, nproc_per_node=2, exit_barrier_timeout=-1
            )
        self.assertIn("exit_barrier_timeout must be non-negative", str(cm.exception))

    def test_exit_barrier_timeout_zero(self):
        """Zero is a valid (instant) timeout."""
        config = LaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=2, exit_barrier_timeout=0
        )
        self.assertEqual(config.exit_barrier_timeout, 0.0)

    # ------------------------------------------------------------------
    # SimpleElasticAgent (via concrete stub): explicit / env-var / default
    # ------------------------------------------------------------------

    def test_simple_elastic_agent_receives_exit_barrier_timeout(self):
        """An explicit value is stored on the agent."""
        agent = _ConcreteAgent(
            spec=_make_mock_spec(),
            exit_barrier_timeout=450,
            shutdown_timeout=30,
        )
        self.assertEqual(agent._exit_barrier_timeout, 450.0)

    def test_simple_elastic_agent_default_exit_barrier_timeout(self):
        """Without any explicit value or env var, the agent uses 300 seconds."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT", None)
            agent = _ConcreteAgent(spec=_make_mock_spec())
        self.assertEqual(agent._exit_barrier_timeout, 300.0)

    def test_simple_elastic_agent_env_var_exit_barrier_timeout(self):
        """TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT is picked up by SimpleElasticAgent."""
        with patch.dict(os.environ, {"TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT": "750"}):
            agent = _ConcreteAgent(spec=_make_mock_spec())
        self.assertEqual(agent._exit_barrier_timeout, 750.0)

    def test_simple_elastic_agent_explicit_overrides_env(self):
        """Explicit constructor arg wins over env var in SimpleElasticAgent."""
        with patch.dict(os.environ, {"TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT": "750"}):
            agent = _ConcreteAgent(spec=_make_mock_spec(), exit_barrier_timeout=200)
        self.assertEqual(agent._exit_barrier_timeout, 200.0)

    # ------------------------------------------------------------------
    # LocalElasticAgent: explicit / env-var / default
    # ------------------------------------------------------------------

    def test_local_elastic_agent_receives_exit_barrier_timeout(self):
        """LocalElasticAgent stores an explicit exit_barrier_timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = LocalElasticAgent(
                spec=_make_mock_spec(),
                logs_specs=DefaultLogsSpecs(log_dir=tmpdir),
                start_method="spawn",
                exit_barrier_timeout=500,
                shutdown_timeout=30,
            )
        self.assertEqual(agent._exit_barrier_timeout, 500.0)

    def test_local_elastic_agent_default_exit_barrier_timeout(self):
        """LocalElasticAgent defaults to 300 s when nothing is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT", None)
                agent = LocalElasticAgent(
                    spec=_make_mock_spec(),
                    logs_specs=DefaultLogsSpecs(log_dir=tmpdir),
                    start_method="spawn",
                )
        self.assertEqual(agent._exit_barrier_timeout, 300.0)

    def test_local_elastic_agent_env_var_exit_barrier_timeout(self):
        """TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT is picked up by LocalElasticAgent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT": "800"}):
                agent = LocalElasticAgent(
                    spec=_make_mock_spec(),
                    logs_specs=DefaultLogsSpecs(log_dir=tmpdir),
                    start_method="spawn",
                )
        self.assertEqual(agent._exit_barrier_timeout, 800.0)


if __name__ == "__main__":
    run_tests()
