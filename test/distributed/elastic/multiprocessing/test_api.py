#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
from types import SimpleNamespace
from unittest.mock import call, MagicMock, patch

from torch.distributed.elastic.multiprocessing.api import (
    _terminate_process_handler,
    PContext,
    SignalException,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class SignalHandlingTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    @patch("torch.distributed.elastic.multiprocessing.api.IS_WINDOWS", False)
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

        mock_signal.configure_mock(SIGTERM=15, SIGINT=2, SIGHUP=1, SIGQUIT=3)
        PContext.start(mock_pcontext)

        expected_calls = [
            call(number, _terminate_process_handler) for number in (15, 2, 1, 3)
        ]
        mock_signal.signal.assert_has_calls(expected_calls)
        self.assertEqual(mock_signal.signal.call_count, len(expected_calls))
        mock_logger.warning.assert_not_called()

        # Verify _start was called
        mock_pcontext._start.assert_called_once()
        # Verify _stdout_tail.start() and _stderr_tail.start() were called
        mock_stdout_tail.start.assert_called_once()
        mock_stderr_tail.start.assert_called_once()

    @patch("torch.distributed.elastic.multiprocessing.api.IS_WINDOWS", False)
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

        mock_signal.configure_mock(SIGTERM=15, SIGUSR1=10, SIGUSR2=12)
        PContext.start(mock_pcontext)

        expected_calls = [
            call(number, _terminate_process_handler) for number in (15, 10, 12)
        ]
        mock_signal.signal.assert_has_calls(expected_calls)
        self.assertEqual(mock_signal.signal.call_count, len(expected_calls))
        mock_logger.warning.assert_not_called()

        # Verify _start was called
        mock_pcontext._start.assert_called_once()

    @parametrize("is_windows", [False, True])
    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_handles_invalid_signals(
        self, mock_logger, mock_threading, is_windows
    ):
        mock_threading.current_thread.return_value = (
            mock_threading.main_thread.return_value
        )
        mock_pcontext = MagicMock(spec=PContext)
        mock_pcontext._tail_logs = [MagicMock(), MagicMock()]
        mock_signal = SimpleNamespace(SIGTERM=15, signal=MagicMock())
        os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = " SIGTERM , SIGTREM "

        with (
            patch("torch.distributed.elastic.multiprocessing.api.IS_WINDOWS", is_windows),
            patch("torch.distributed.elastic.multiprocessing.api.signal", mock_signal),
        ):
            PContext.start(mock_pcontext)

        mock_logger.warning.assert_called_once_with(
            "Failed to register signal handler for %s", " SIGTREM ", exc_info=True
        )
        mock_signal.signal.assert_called_once_with(15, _terminate_process_handler)
        mock_pcontext._start.assert_called_once()
        for tail_log in mock_pcontext._tail_logs:
            tail_log.start.assert_called_once()

    @parametrize("is_windows", [False, True])
    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_handles_unsupported_signal_values(
        self, mock_logger, mock_threading, is_windows
    ):
        mock_threading.current_thread.return_value = (
            mock_threading.main_thread.return_value
        )
        mock_pcontext = MagicMock(spec=PContext)
        mock_pcontext._tail_logs = [MagicMock(), MagicMock()]
        mock_signal = SimpleNamespace(
            SIGTERM=15, signal=MagicMock(side_effect=ValueError("unsupported signal"))
        )
        os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = "SIGTERM"

        with (
            patch("torch.distributed.elastic.multiprocessing.api.IS_WINDOWS", is_windows),
            patch("torch.distributed.elastic.multiprocessing.api.signal", mock_signal),
        ):
            PContext.start(mock_pcontext)

        mock_logger.warning.assert_called_once_with(
            "Failed to register signal handler for %s", "SIGTERM", exc_info=True
        )
        mock_signal.signal.assert_called_once_with(15, _terminate_process_handler)
        mock_pcontext._start.assert_called_once()
        for tail_log in mock_pcontext._tail_logs:
            tail_log.start.assert_called_once()

    @parametrize(
        "signals_to_handle, skipped_signals",
        [
            (None, ["SIGHUP", "SIGQUIT"]),
            (
                " SIGTERM , SIGINT , SIGHUP , SIGQUIT , SIGUSR1 , SIGUSR2 ",
                [" SIGHUP ", " SIGQUIT ", " SIGUSR1 ", " SIGUSR2 "],
            ),
        ],
    )
    @patch("torch.distributed.elastic.multiprocessing.api.IS_WINDOWS", True)
    @patch("torch.distributed.elastic.multiprocessing.api.threading")
    @patch("torch.distributed.elastic.multiprocessing.api.logger")
    def test_start_handles_windows_signals(
        self, mock_logger, mock_threading, signals_to_handle, skipped_signals
    ):
        mock_threading.current_thread.return_value = (
            mock_threading.main_thread.return_value
        )
        mock_pcontext = MagicMock(spec=PContext)
        mock_pcontext._tail_logs = [MagicMock(), MagicMock()]
        mock_signal = SimpleNamespace(SIGTERM=15, SIGINT=2, signal=MagicMock())
        if signals_to_handle is None:
            os.environ.pop("TORCHELASTIC_SIGNALS_TO_HANDLE", None)
        else:
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = signals_to_handle

        with patch("torch.distributed.elastic.multiprocessing.api.signal", mock_signal):
            PContext.start(mock_pcontext)

        expected_calls = [
            call(15, _terminate_process_handler), call(2, _terminate_process_handler)
        ]
        mock_signal.signal.assert_has_calls(expected_calls)
        self.assertEqual(mock_signal.signal.call_count, len(expected_calls))
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_has_calls(
            [
                call("Signal %s is not supported on Windows, skipping", sig_name)
                for sig_name in skipped_signals
            ]
        )
        self.assertEqual(mock_logger.info.call_count, 2 + len(skipped_signals))
        registered_signals = (
            ["SIGTERM", "SIGINT"]
            if signals_to_handle is None
            else [" SIGTERM ", " SIGINT "]
        )
        for sig_name in registered_signals:
            mock_logger.info.assert_any_call("Registered signal handler for %s", sig_name)
        mock_pcontext._start.assert_called_once()
        for tail_log in mock_pcontext._tail_logs:
            tail_log.start.assert_called_once()

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

    @patch("torch.distributed.elastic.multiprocessing.api.IS_WINDOWS", False)
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
        mock_signal.SIGUSR1 = 10
        mock_signal.SIGUSR2 = 12

        # Call the start method
        PContext.start(mock_pcontext)

        # Verify that signal.signal was called for both SIGUSR1 and SIGUSR2
        signal_calls = mock_signal.signal.call_args_list
        registered_signals = [
            call[0][0] for call in signal_calls
        ]  # Extract the signal from each call

        # Verify both SIGUSR1 and SIGUSR2 were registered
        self.assertIn(
            10, registered_signals, "SIGUSR1 should be registered"
        )
        self.assertIn(
            12, registered_signals, "SIGUSR2 should be registered"
        )

        # Verify the correct handler was registered for both signals
        for call in signal_calls:
            sig, handler = call[0]
            if sig in [10, 12]:
                self.assertEqual(
                    handler,
                    _terminate_process_handler,
                    lambda msg: f"{msg}\nSignal {sig} should use _terminate_process_handler",
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
            lambda msg: f"{msg}\nExpected SIGUSR1 registration message in info calls: {info_calls}",
        )
        self.assertTrue(
            sigusr2_logged,
            lambda msg: f"{msg}\nExpected SIGUSR2 registration message in info calls: {info_calls}",
        )
        mock_logger.warning.assert_not_called()

        # Verify _start was called
        mock_pcontext._start.assert_called_once()
        # Verify _stdout_tail.start() and _stderr_tail.start() were called
        mock_stdout_tail.start.assert_called_once()
        mock_stderr_tail.start.assert_called_once()


if __name__ == "__main__":
    run_tests()
