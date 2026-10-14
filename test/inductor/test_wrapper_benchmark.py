# Owner(s): ["module: inductor"]
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.wrapper_benchmark import compiled_module_main


class TestWrapperBenchmarkPeakMemory(TestCase):
    def test_no_accelerator_skips_peak_memory_stats(self):
        """When no accelerator is available, peak memory stats are not printed."""
        with patch("torch.accelerator.current_accelerator", return_value=None):
            buf = io.StringIO()
            with redirect_stdout(buf):
                sys.argv = ["test", "--times", "1", "--repeat", "1"]
                try:
                    compiled_module_main(
                        "test_bench",
                        lambda times, repeat: 0.001,
                    )
                except SystemExit:
                    pass
            output = buf.getvalue()
            self.assertNotIn("Peak", output)

    @unittest.skipIf(
        not torch.accelerator.is_available(),
        "requires an accelerator",
    )
    def test_accelerator_resets_and_reports_peak_memory(self):
        """When an accelerator is available, peak memory is reset and reported."""
        acc = torch.accelerator.current_accelerator()
        device_label = "GPU" if acc.type == "cuda" else acc.type.upper()

        def mock_benchmark(times, repeat):
            x = torch.randn(100, 100, device=acc)
            del x
            if hasattr(torch, "accelerator"):
                torch.accelerator.synchronize()
            return 0.001

        sys.argv = ["test", "--times", "1", "--repeat", "1"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                compiled_module_main("test_bench", mock_benchmark)
            except SystemExit:
                pass
        output = buf.getvalue()
        self.assertIn(f"Peak {device_label} memory usage", output)

    @unittest.skipIf(
        not torch.accelerator.is_available()
        or torch.accelerator.current_accelerator().type != "cuda",
        "requires CUDA",
    )
    def test_cuda_preserves_gpu_label(self):
        """CUDA output preserves 'Peak GPU memory usage' for backward compatibility."""

        def mock_benchmark(times, repeat):
            x = torch.randn(50, 50, device="cuda")
            del x
            torch.accelerator.synchronize()
            return 0.001

        sys.argv = ["test", "--times", "1", "--repeat", "1"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                compiled_module_main("test_bench", mock_benchmark)
            except SystemExit:
                pass
        output = buf.getvalue()
        self.assertIn("Peak GPU memory usage", output)


if __name__ == "__main__":
    run_tests()
