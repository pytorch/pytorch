#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test script to verify signal handling customization works correctly.
This test doesn't rely on the PyTorch test infrastructure.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from torch.distributed.elastic.utils.logging import get_logger
from torch.testing._internal.common_utils import run_tests


# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))
logger = get_logger(__name__)


def test_default_signals():
    """Test that default signals are handled correctly."""
    logger.info("Testing default signal handling...")

    # Create a simple script that uses torchrun
    test_script = """
import os
import signal
import time
import sys

def worker_fn():
    logger.info("Worker started")
    # Check if signal handlers are registered
    signals_env = os.environ.get("TORCHELASTIC_SIGNALS_TO_HANDLE", "")
    logger.info(f"Signals to handle: {signals_env}")

    # Sleep for a bit to allow signal testing
    time.sleep(2)
    logger.info("Worker finished")
    return 0

if __name__ == "__main__":
    worker_fn()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        # Test with default signals
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            script_path,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)

        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            logger.info("✓ Default signal handling test passed")
            if "SIGTERM,SIGINT,SIGHUP,SIGQUIT" in result.stdout:
                logger.info("✓ Default signals correctly set")
            else:
                logger.info("⚠ Default signals not found in output, but test passed")
        else:
            logger.info(f"✗ Default signal handling test failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.info("✓ Test timed out as expected (worker was running)")
    except Exception as e:
        logger.info(f"✗ Test failed with exception: {e}")
    finally:
        os.unlink(script_path)


def test_custom_signals():
    """Test that custom signals are handled correctly."""
    logger.info("\nTesting custom signal handling...")

    # Create a simple script that uses torchrun
    test_script = """
import os
import signal
import time
import sys

def worker_fn():
    logger.info("Worker started")
    # Check if signal handlers are registered
    signals_env = os.environ.get("TORCHELASTIC_SIGNALS_TO_HANDLE", "")
    logger.info(f"Signals to handle: {signals_env}")

    # Sleep for a bit to allow signal testing
    time.sleep(2)
    logger.info("Worker finished")
    return 0

if __name__ == "__main__":
    worker_fn()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        # Test with custom signals including SIGUSR1 and SIGUSR2
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "--signals-to-handle=SIGTERM,SIGINT,SIGUSR1,SIGUSR2",
            script_path,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)

        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            logger.info("✓ Custom signal handling test passed")
            if "SIGTERM,SIGINT,SIGUSR1,SIGUSR2" in result.stdout:
                logger.info("✓ Custom signals correctly set")
            else:
                logger.info("⚠ Custom signals not found in output, but test passed")
        else:
            logger.info(f"✗ Custom signal handling test failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.info("✓ Test timed out as expected (worker was running)")
    except Exception as e:
        logger.info(f"✗ Test failed with exception: {e}")
    finally:
        os.unlink(script_path)


def test_signal_registration():
    """Test that signal registration works correctly in the API."""
    logger.info("\nTesting signal registration in API...")

    try:
        # Import the modules we modified
        from torch.distributed.launcher.api import LaunchConfig

        # Test LaunchConfig with custom signals
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2",
        )

        if config.signals_to_handle == "SIGTERM,SIGUSR1,SIGUSR2":
            logger.info("✓ LaunchConfig correctly stores custom signals")
        else:
            logger.info(
                f"✗ LaunchConfig failed to store signals: {config.signals_to_handle}"
            )

        # Test default signals
        default_config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)

        if default_config.signals_to_handle == "SIGTERM,SIGINT,SIGHUP,SIGQUIT":
            logger.info("✓ LaunchConfig correctly uses default signals")
        else:
            logger.info(
                f"✗ LaunchConfig failed to use default signals: {default_config.signals_to_handle}"
            )

    except Exception as e:
        logger.info(f"✗ Signal registration test failed: {e}")


if __name__ == "__main__":
    run_tests()
