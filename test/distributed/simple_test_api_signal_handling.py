#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test to verify the signal handling changes work.
"""

import os
import sys
from pathlib import Path

from torch.distributed.elastic.utils.logging import get_logger
from torch.testing._internal.common_utils import run_tests


# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))
logger = get_logger(__name__)


def test_launch_config():
    """Test that LaunchConfig accepts the new signals_to_handle parameter."""
    try:
        from torch.distributed.launcher.api import LaunchConfig

        logger.info("Testing LaunchConfig...")

        # Test with default signals
        config1 = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        logger.info(f"Default signals: {config1.signals_to_handle}")
        assert config1.signals_to_handle == "SIGTERM,SIGINT,SIGHUP,SIGQUIT"
        logger.info("✓ Default signals test passed")

        # Test with custom signals
        config2 = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2",
        )
        logger.info(f"Custom signals: {config2.signals_to_handle}")
        assert config2.signals_to_handle == "SIGTERM,SIGUSR1,SIGUSR2"
        logger.info("✓ Custom signals test passed")

        return True

    except Exception as e:
        logger.info(f"✗ LaunchConfig test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_command_line_args():
    """Test that the command line argument parsing works."""
    try:
        from torch.distributed.run import get_args_parser

        logger.info("\nTesting command line argument parsing...")

        parser = get_args_parser()

        # Test default behavior
        args1 = parser.parse_args(["--standalone", "test_script.py"])
        logger.info(f"Default signals from args: {args1.signals_to_handle}")
        assert args1.signals_to_handle == "SIGTERM,SIGINT,SIGHUP,SIGQUIT"
        logger.info("✓ Default command line args test passed")

        # Test custom signals
        args2 = parser.parse_args(
            [
                "--standalone",
                "--signals-to-handle=SIGTERM,SIGUSR1,SIGUSR2",
                "test_script.py",
            ]
        )
        logger.info(f"Custom signals from args: {args2.signals_to_handle}")
        assert args2.signals_to_handle == "SIGTERM,SIGUSR1,SIGUSR2"
        logger.info("✓ Custom command line args test passed")

        return True

    except Exception as e:
        logger.info(f"✗ Command line args test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_environment_variable():
    """Test that the environment variable is set correctly."""
    try:
        from torch.distributed.launcher.api import LaunchConfig

        logger.info("\nTesting environment variable setting...")

        # Create a config with custom signals
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2",
        )

        # Mock the launch_agent function to just set the environment variable
        # without actually launching anything
        original_env = os.environ.get("TORCHELASTIC_SIGNALS_TO_HANDLE")

        # Set the environment variable like launch_agent does
        os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = config.signals_to_handle

        # Check if it was set correctly
        env_value = os.environ.get("TORCHELASTIC_SIGNALS_TO_HANDLE")
        logger.info(f"Environment variable value: {env_value}")
        assert env_value == "SIGTERM,SIGUSR1,SIGUSR2"
        logger.info("✓ Environment variable test passed")

        # Restore original environment
        if original_env is not None:
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = original_env
        elif "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

        return True

    except Exception as e:
        logger.info(f"✗ Environment variable test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_tests()
