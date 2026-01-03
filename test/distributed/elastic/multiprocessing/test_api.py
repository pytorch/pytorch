#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
from unittest.mock import MagicMock, patch

from torch.distributed.elastic.multiprocessing.api import (
    _terminate_process_handler,
    PContext,
    SignalException,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class SignalHandlingTest(TestCase):
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

    def test_terminate_process_handler(self):
        """Test that the terminate process handler raises SignalException with the correct signal."""
        signum = signal.SIGTERM
        with self.assertRaises(SignalException) as cm:
            _terminate_process_handler(signum, None)

        self.assertEqual(cm.exception.sigval, signal.SIGTERM)
        # The signal is represented as a number in the string representation
        self.assertIn(f"Process {os.getpid()} got signal: {signum}", str(cm.exception))

    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.signal")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_registers_default_signals(
        self, mock_logger, mock_signal, mock_threading
    ):
        """Test that the start method registers the default signals."""
        # Setup
        mock_threading.current_thread.return_value = (
            mock_threading.main_thread.return_value
        )
        mock_pcontext = MagicMock(spec=PContext)
        # Mock the stdout_tail and stderr_tail
        mock_stdout_tail = MagicMock()
        mock_stderr_tail = MagicMock()
        mock_pcontext._tail_logs = [mock_stdout_tail, mock_stderr_tail]

        # Remove environment variable if it exists to test default behavior
        if "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

        # Call the start method
        PContext.start(mock_pcontext)

        # Verify that the signal handler was registered for the default signals
        expected_signals = ["SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"]

        # Count the number of calls to signal.signal
        signal_calls = 0
        for call in mock_signal.signal.call_args_list:
            args, _ = call
            sig, handler = args
            signal_calls += 1
            # Verify the handler is our _terminate_process_handler
            self.assertEqual(handler, _terminate_process_handler)

        # Verify we registered the expected number of signals
        self.assertEqual(signal_calls, len(expected_signals))

        # Verify _start was called
        mock_pcontext._start.assert_called_once()
        # Verify _stdout_tail.start() and _stderr_tail.start() were called
        mock_stdout_tail.start.assert_called_once()
        mock_stderr_tail.start.assert_called_once()

    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.signal")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_registers_custom_signals(
        self, mock_logger, mock_signal, mock_threading
    ):
        """Test that the start method registers custom signals from the environment variable."""
        # Setup
        mock_threading.current_thread.return_value = (
            mock_threading.main_thread.return_value
        )
        mock_pcontext = MagicMock(spec=PContext)
        # Mock the stdout_tail and stderr_tail
        mock_stdout_tail = MagicMock()
        mock_stderr_tail = MagicMock()
        mock_pcontext._tail_logs = [mock_stdout_tail, mock_stderr_tail]

        # Set custom signals in the environment variable
        os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = "SIGTERM,SIGUSR1,SIGUSR2"

        # Call the start method
        PContext.start(mock_pcontext)

        # Verify that the signal handler was registered for the custom signals
        expected_signals = ["SIGTERM", "SIGUSR1", "SIGUSR2"]

        # Count the number of calls to signal.signal
        signal_calls = 0
        for call in mock_signal.signal.call_args_list:
            args, _ = call
            sig, handler = args
            signal_calls += 1
            # Verify the handler is our _terminate_process_handler
            self.assertEqual(handler, _terminate_process_handler)

        # Verify we registered the expected number of signals
        self.assertEqual(signal_calls, len(expected_signals))

        # Verify _start was called
        mock_pcontext._start.assert_called_once()

    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.signal")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_handles_invalid_signals(
        self, mock_logger, mock_signal, mock_threading
    ):
        """Test that the start method handles invalid signals gracefully."""
        # Setup
        mock_threading.current_thread.return_value = (
            mock_threading.main_thread.return_value
        )
        mock_pcontext = MagicMock(spec=PContext)
        # Mock the stdout_tail and stderr_tail
        mock_stdout_tail = MagicMock()
        mock_stderr_tail = MagicMock()
        mock_pcontext._tail_logs = [mock_stdout_tail, mock_stderr_tail]

        # Set invalid signals in the environment variable
        os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = "SIGTERM,INVALID_SIGNAL"

        # Mock the signal module to not have the INVALID_SIGNAL attribute
        # but have SIGTERM
        mock_signal.SIGTERM = signal.SIGTERM
        # Remove INVALID_SIGNAL attribute if it exists
        if hasattr(mock_signal, "INVALID_SIGNAL"):
            delattr(mock_signal, "INVALID_SIGNAL")

        # Call the start method
        PContext.start(mock_pcontext)

        # Verify that the warning was logged for the invalid signal
        # The exact message may vary, so let's check if warning was called with INVALID_SIGNAL
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "INVALID_SIGNAL" in str(call)
        ]
        self.assertTrue(len(warning_calls) > 0, "Expected warning about INVALID_SIGNAL")

        # Verify _start was called
        mock_pcontext._start.assert_called_once()

    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.signal")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_handles_windows_signals(
        self, mock_logger, mock_signal, mock_threading
    ):
        """Test that the start method handles Windows-specific signal behavior."""
        # Setup
        mock_threading.current_thread.return_value = (
            mock_threading.main_thread.return_value
        )
        mock_pcontext = MagicMock(spec=PContext)
        # Mock the stdout_tail and stderr_tail
        mock_stdout_tail = MagicMock()
        mock_stderr_tail = MagicMock()
        mock_pcontext._tail_logs = [mock_stdout_tail, mock_stderr_tail]

        # Set signals including ones not supported on Windows
        os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = "SIGTERM,SIGHUP,SIGUSR1"

        # Mock signal attributes
        mock_signal.SIGTERM = signal.SIGTERM
        mock_signal.SIGHUP = signal.SIGHUP
        mock_signal.SIGUSR1 = signal.SIGUSR1

        # Mock IS_WINDOWS to be True
        with patch("torch.distributed.elastic.multiprocessing.api.IS_WINDOWS", True):
            # Mock signal.signal to raise RuntimeError for Windows-unsupported signals
            def signal_side_effect(sig, handler):
                if sig in [signal.SIGHUP, signal.SIGUSR1]:
                    raise RuntimeError("Signal not supported on Windows")

            mock_signal.signal.side_effect = signal_side_effect

            # Call the start method
            PContext.start(mock_pcontext)

            # Verify that the info was logged for the unsupported signals
            # Check if any info calls contain the expected messages
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            sighup_logged = any(
                "SIGHUP" in call and "Windows" in call for call in info_calls
            )
            sigusr1_logged = any(
                "SIGUSR1" in call and "Windows" in call for call in info_calls
            )

            self.assertTrue(
                sighup_logged,
                f"Expected SIGHUP Windows message in info calls: {info_calls}",
            )
            self.assertTrue(
                sigusr1_logged,
                f"Expected SIGUSR1 Windows message in info calls: {info_calls}",
            )

            # Verify _start was called
            mock_pcontext._start.assert_called_once()

    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_not_main_thread(self, mock_logger, mock_threading):
        """Test that the start method warns when not called from the main thread."""
        # Setup
        mock_threading.current_thread.return_value = MagicMock()  # Not the main thread
        mock_threading.main_thread.return_value = MagicMock()
        mock_pcontext = MagicMock(spec=PContext)
        # Mock the stdout_tail and stderr_tail
        mock_stdout_tail = MagicMock()
        mock_stderr_tail = MagicMock()
        mock_pcontext._tail_logs = [mock_stdout_tail, mock_stderr_tail]

        # Call the start method
        PContext.start(mock_pcontext)

        # Verify that the warning was logged
        mock_logger.warning.assert_called_with(
            "Failed to register signal handlers since torchelastic is running on a child thread. "
            "This could lead to orphaned worker processes if the torchrun is terminated."
        )

        # Verify _start was called
        mock_pcontext._start.assert_called_once()

    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.signal")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_supports_sigusr1_and_sigusr2(
        self, mock_logger, mock_signal, mock_threading
    ):
        """Test that the start method properly supports SIGUSR1 and SIGUSR2 signals."""
        # Setup
        mock_threading.current_thread.return_value = (
            mock_threading.main_thread.return_value
        )
        mock_pcontext = MagicMock(spec=PContext)
        # Mock the stdout_tail and stderr_tail
        mock_stdout_tail = MagicMock()
        mock_stderr_tail = MagicMock()
        mock_pcontext._tail_logs = [mock_stdout_tail, mock_stderr_tail]

        # Set environment variable to include SIGUSR1 and SIGUSR2
        os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = "SIGUSR1,SIGUSR2"

        # Mock signal attributes to have SIGUSR1 and SIGUSR2
        mock_signal.SIGUSR1 = signal.SIGUSR1
        mock_signal.SIGUSR2 = signal.SIGUSR2

        # Call the start method
        PContext.start(mock_pcontext)

        # Verify that signal.signal was called for both SIGUSR1 and SIGUSR2
        signal_calls = mock_signal.signal.call_args_list
        registered_signals = [
            call[0][0] for call in signal_calls
        ]  # Extract the signal from each call

        # Verify both SIGUSR1 and SIGUSR2 were registered
        self.assertIn(
            signal.SIGUSR1, registered_signals, "SIGUSR1 should be registered"
        )
        self.assertIn(
            signal.SIGUSR2, registered_signals, "SIGUSR2 should be registered"
        )

        # Verify the correct handler was registered for both signals
        for call in signal_calls:
            sig, handler = call[0]
            if sig in [signal.SIGUSR1, signal.SIGUSR2]:
                self.assertEqual(
                    handler,
                    _terminate_process_handler,
                    f"Signal {sig} should use _terminate_process_handler",
                )

        # Verify that info messages were logged for successful registration
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        sigusr1_logged = any(
            "SIGUSR1" in call and "Registered signal handler" in call
            for call in info_calls
        )
        sigusr2_logged = any(
            "SIGUSR2" in call and "Registered signal handler" in call
            for call in info_calls
        )

        self.assertTrue(
            sigusr1_logged,
            f"Expected SIGUSR1 registration message in info calls: {info_calls}",
        )
        self.assertTrue(
            sigusr2_logged,
            f"Expected SIGUSR2 registration message in info calls: {info_calls}",
        )

        # Verify _start was called
        mock_pcontext._start.assert_called_once()
        # Verify _stdout_tail.start() and _stderr_tail.start() were called
        mock_stdout_tail.start.assert_called_once()
        mock_stderr_tail.start.assert_called_once()


if __name__ == "__main__":
    run_tests()
