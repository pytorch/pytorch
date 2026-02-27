#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest.mock import MagicMock, patch

import torch.distributed.run as run
from torch.distributed.launcher.api import launch_agent, LaunchConfig
from torch.testing._internal.common_utils import run_tests, TestCase


class RunTest(TestCase):
    def setUp(self):
        super().setUp()
        # Save original environment variable if it exists
        self.original_signals_env = os.environ.get(
            "TORCHELASTIC_SIGNALS_TO_HANDLE", None
        )

    def tearDown(self):
        # Restore original environment variable
        if self.original_signals_env is not None:
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = self.original_signals_env
        elif "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

    def test_signals_to_handle_default(self):
        """Test that the default value for signals_to_handle is correctly set."""
        parser = run.get_args_parser()
        args = parser.parse_args(["dummy_script.py"])
        self.assertEqual(args.signals_to_handle, "SIGTERM,SIGINT,SIGHUP,SIGQUIT")

    def test_signals_to_handle_custom(self):
        """Test that a custom value for signals_to_handle is correctly parsed."""
        parser = run.get_args_parser()
        args = parser.parse_args(
            ["--signals-to-handle=SIGTERM,SIGUSR1,SIGUSR2", "dummy_script.py"]
        )
        self.assertEqual(args.signals_to_handle, "SIGTERM,SIGUSR1,SIGUSR2")

    def test_config_from_args_signals_to_handle(self):
        """Test that the signals_to_handle argument is correctly passed to LaunchConfig."""
        parser = run.get_args_parser()
        args = parser.parse_args(
            ["--signals-to-handle=SIGTERM,SIGUSR1,SIGUSR2", "dummy_script.py"]
        )
        config, _, _ = run.config_from_args(args)
        self.assertEqual(config.signals_to_handle, "SIGTERM,SIGUSR1,SIGUSR2")

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_launch_agent_sets_environment_variable(self, mock_get_handler, mock_agent):
        """Test that launch_agent sets the TORCHELASTIC_SIGNALS_TO_HANDLE environment variable."""
        # Setup
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2",
        )
        entrypoint = "dummy_script.py"
        args = []

        # Make sure the environment variable doesn't exist before the test
        if "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

        # Mock agent.run() to return a MagicMock
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = MagicMock(
            is_failed=lambda: False, return_values={}
        )
        mock_agent.return_value = mock_agent_instance

        # Call launch_agent
        launch_agent(config, entrypoint, args)

        # Verify that the environment variable was set correctly
        self.assertEqual(
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"], "SIGTERM,SIGUSR1,SIGUSR2"
        )

    def _parse_args(self, args):
        parser = run.get_args_parser()
        return parser.parse_args(args)

    @patch("torch.distributed.run.elastic_launch")
    def test_single_node_defaults_to_free_port(self, mock_launch):
        """Single-node with no explicit port should auto-switch to c10d with localhost:0."""
        args = self._parse_args(["dummy_script.py"])
        self.assertIsNone(args.master_port)
        run.run(args)
        self.assertEqual(args.rdzv_backend, "c10d")
        self.assertEqual(args.rdzv_endpoint, "localhost:0")

    @patch("torch.distributed.run.elastic_launch")
    def test_single_node_explicit_port_uses_static(self, mock_launch):
        """Single-node with explicit --master-port should keep static backend."""
        args = self._parse_args(["--master-port=12345", "dummy_script.py"])
        self.assertEqual(args.master_port, 12345)
        run.run(args)
        self.assertEqual(args.rdzv_backend, "static")

    @patch("torch.distributed.run.elastic_launch")
    def test_multi_node_defaults_to_static(self, mock_launch):
        """Multi-node should keep static backend with default port 29500."""
        args = self._parse_args(["--nnodes=2", "dummy_script.py"])
        run.run(args)
        self.assertEqual(args.rdzv_backend, "static")
        self.assertEqual(args.master_port, 29500)

    @patch("torch.distributed.run.elastic_launch")
    def test_standalone_still_works(self, mock_launch):
        """--standalone should still use c10d with localhost:0."""
        args = self._parse_args(["--standalone", "dummy_script.py"])
        run.run(args)
        self.assertEqual(args.rdzv_backend, "c10d")
        self.assertEqual(args.rdzv_endpoint, "localhost:0")


if __name__ == "__main__":
    run_tests()
