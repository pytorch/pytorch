import os
import unittest

from .common import parse_args, run
from .torchbench import setup_torchbench_cwd, TorchBenchmarkRunner


try:
    # fbcode only
    from aiplatform.utils.sanitizer_status import is_asan_or_tsan
except ImportError:

    def is_asan_or_tsan():
        return False


class TestDynamoBenchmark(unittest.TestCase):
    def test_dashboard_performance_uses_warm_peak_memory(self) -> None:
        args = parse_args(
            [
                "-dcuda",
                "--inductor",
                "--inference",
                "--performance",
                "--dashboard",
            ]
        )
        self.assertTrue(args.use_warm_peak_memory)

        args = parse_args(
            [
                "-dcuda",
                "--inductor",
                "--inference",
                "--performance",
            ]
        )
        self.assertFalse(args.use_warm_peak_memory)

    def test_detectron2_maskrcnn_uses_iou_for_bool_masks(self) -> None:
        runner = TorchBenchmarkRunner()
        for name in (
            "detectron2_maskrcnn_r_101_fpn",
            "detectron2_maskrcnn_r_50_c4",
        ):
            self.assertTrue(runner.use_iou_for_bool_accuracy(name))
            self.assertEqual(runner.get_iou_threshold(name), 0.99)

    @unittest.skipIf(is_asan_or_tsan(), "ASAN/TSAN not supported")
    def test_benchmark_infra_runs(self) -> None:
        """
        Basic smoke test that TorchBench runs.

        This test is mainly meant to check that our setup in fbcode
        doesn't break.

        If you see a failure here related to missing CPP headers, then
        you likely need to update the resources list in:
            //caffe2:inductor
        """
        original_dir = setup_torchbench_cwd()
        try:
            args = parse_args(
                [
                    "-dcpu",
                    "--inductor",
                    "--training",
                    "--performance",
                    "--only=BERT_pytorch",
                    "-n1",
                    "--batch-size=1",
                ]
            )
            run(TorchBenchmarkRunner(), args, original_dir)
        finally:
            os.chdir(original_dir)
