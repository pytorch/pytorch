# Owner(s): ["module: inductor"]

import contextlib
import io
import sys
from unittest.mock import MagicMock, patch

import torch
from torch._inductor.wrapper_benchmark import (
    collect_memory_snapshot,
    compiled_module_main,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestCompiledModuleMain(TestCase):
    def _run_main(self, argv, benchmark_fn=None):
        benchmark_fn = benchmark_fn or (lambda times, repeat: 0.001)
        with patch.object(sys, "argv", ["compiled_module.py", *argv]):
            compiled_module_main("test_benchmark", benchmark_fn)

    def test_no_accelerator_skips_memory_stats(self):
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

    def test_accelerator_resets_and_reports_peak_memory(self):
        events = []
        benchmark_fn = MagicMock(
            side_effect=lambda times, repeat: events.append("benchmark") or 0.001
        )
        with (
            patch(
                "torch.accelerator.current_accelerator",
                return_value=torch.device("cuda"),
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
        self.assertIn("Peak CUDA memory usage 2.000 MB", printed)

    @parametrize("flag", ["--memory-snapshot", "--cuda-memory-snapshot"])
    def test_memory_snapshot_flag(self, flag):
        benchmark_fn = MagicMock(return_value=0.001)
        with (
            patch(
                "torch.accelerator.current_accelerator",
                return_value=torch.device("cuda"),
            ),
            patch(
                "torch._inductor.wrapper_benchmark.collect_memory_snapshot"
            ) as mock_collect,
            patch("torch.accelerator.reset_peak_memory_stats"),
            patch("torch.accelerator.max_memory_allocated", return_value=0),
            patch("builtins.print"),
        ):
            self._run_main([flag], benchmark_fn)

        mock_collect.assert_called_once_with(benchmark_fn)

    def test_legacy_memory_snapshot_flag_is_hidden(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(output):
            self._run_main(["--help"])
        self.assertIn("--memory-snapshot", output.getvalue())
        self.assertNotIn("--cuda-memory-snapshot", output.getvalue())

    def test_memory_snapshot_not_dispatched_without_accelerator(self):
        with (
            patch("torch.accelerator.current_accelerator", return_value=None),
            patch(
                "torch._inductor.wrapper_benchmark.collect_memory_snapshot"
            ) as mock_collect,
            patch("builtins.print"),
        ):
            self._run_main(["--memory-snapshot"])
        mock_collect.assert_not_called()


class TestCollectMemorySnapshot(TestCase):
    def test_requires_accelerator(self):
        with patch("torch.accelerator.current_accelerator", return_value=None):
            with self.assertRaisesRegex(AssertionError, "No accelerator is available"):
                collect_memory_snapshot(lambda times, repeat: None)

    def test_unsupported_device_skips(self):
        device_mod = MagicMock(spec=[])
        benchmark_fn = MagicMock()
        with (
            patch(
                "torch.accelerator.current_accelerator",
                return_value=torch.device("mps"),
            ),
            patch("torch.get_device_module", return_value=device_mod),
            patch("builtins.print") as mock_print,
        ):
            collect_memory_snapshot(benchmark_fn)

        benchmark_fn.assert_not_called()
        self.assertIn(
            "not supported on mps",
            " ".join(str(c) for c in mock_print.call_args_list),
        )

    def test_supported_device(self):
        mem_mod = MagicMock(spec=["_record_memory_history", "_dump_snapshot"])
        device_mod = MagicMock()
        device_mod.memory = mem_mod
        benchmark_fn = MagicMock(return_value=None)
        with (
            patch(
                "torch.accelerator.current_accelerator",
                return_value=torch.device("cuda"),
            ),
            patch("torch.get_device_module", return_value=device_mod),
            patch("builtins.print"),
        ):
            collect_memory_snapshot(benchmark_fn)

        self.assertEqual(mem_mod._record_memory_history.call_count, 2)
        self.assertEqual(
            mem_mod._record_memory_history.call_args_list[0].kwargs,
            {"max_entries": 100000},
        )
        mem_mod._record_memory_history.assert_called_with(enabled=None)
        benchmark_fn.assert_called_once_with(times=10, repeat=1)
        snapshot_path = mem_mod._dump_snapshot.call_args.args[0]
        self.assertTrue(snapshot_path.endswith("memory_snapshot.pickle"))

    def test_recording_is_disabled_when_benchmark_fails(self):
        mem_mod = MagicMock(spec=["_record_memory_history", "_dump_snapshot"])
        device_mod = MagicMock()
        device_mod.memory = mem_mod
        benchmark_fn = MagicMock(side_effect=RuntimeError("benchmark failed"))
        with (
            patch(
                "torch.accelerator.current_accelerator",
                return_value=torch.device("cuda"),
            ),
            patch("torch.get_device_module", return_value=device_mod),
        ):
            with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                collect_memory_snapshot(benchmark_fn)

        self.assertEqual(mem_mod._record_memory_history.call_count, 2)
        self.assertEqual(
            mem_mod._record_memory_history.call_args_list[0].kwargs,
            {"max_entries": 100000},
        )
        mem_mod._record_memory_history.assert_called_with(enabled=None)
        mem_mod._dump_snapshot.assert_not_called()


instantiate_parametrized_tests(TestCompiledModuleMain)


if __name__ == "__main__":
    run_tests()
