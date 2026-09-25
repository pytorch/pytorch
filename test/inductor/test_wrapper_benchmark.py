# Owner(s): ["module: inductor"]

import sys
from unittest.mock import MagicMock, patch

import torch
from torch._inductor.wrapper_benchmark import compiled_module_main
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestCompiledModuleMainPeakMemory(TestCase):
    def _run_main(self, argv, benchmark_fn=None):
        benchmark_fn = benchmark_fn or (lambda times, repeat: 0.001)
        with patch.object(sys, "argv", ["compiled_module.py", *argv]):
            compiled_module_main("test_benchmark", benchmark_fn)

    def test_no_accelerator_skips_peak_memory_stats(self):
        benchmark_fn = MagicMock(return_value=0.001)
        with (
            patch(
                "torch.accelerator.current_accelerator", return_value=None
            ) as mock_acc,
            patch("torch.accelerator.reset_peak_memory_stats") as mock_reset,
            patch("torch.accelerator.max_memory_allocated") as mock_max,
            patch("builtins.print") as mock_print,
        ):
            self._run_main([], benchmark_fn)

        mock_acc.assert_called_once_with(check_available=True)
        mock_reset.assert_not_called()
        mock_max.assert_not_called()
        benchmark_fn.assert_called_once_with(times=10, repeat=10)
        self.assertNotIn("Peak", " ".join(str(c) for c in mock_print.call_args_list))

    @parametrize(
        "device_type,expected",
        [
            ("cuda", "Peak GPU memory usage 2.000 MB"),
            ("xpu", "Peak XPU memory usage 2.000 MB"),
        ],
    )
    def test_accelerator_resets_and_reports_peak_memory(self, device_type, expected):
        events = []
        benchmark_fn = MagicMock(
            side_effect=lambda times, repeat: events.append("benchmark") or 0.001
        )
        with (
            patch(
                "torch.accelerator.current_accelerator",
                return_value=torch.device(device_type),
            ),
            patch(
                "torch.accelerator.reset_peak_memory_stats",
                side_effect=lambda: events.append("reset"),
            ) as mock_reset,
            patch(
                "torch.accelerator.max_memory_allocated",
                side_effect=lambda: events.append("max") or int(2e6),
            ) as mock_max,
            patch("builtins.print") as mock_print,
        ):
            self._run_main([], benchmark_fn)

        self.assertEqual(events, ["reset", "benchmark", "max"])
        mock_reset.assert_called_once_with()
        mock_max.assert_called_once_with()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn(expected, printed)


instantiate_parametrized_tests(TestCompiledModuleMainPeakMemory)


if __name__ == "__main__":
    run_tests()
