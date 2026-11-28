import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(__file__))

from auto_classify_skip_reasons import detect_columns
from detect_log_failures import parse_log_file
from generate_summary import (
    add_promoted_failure_stats,
    collect_log_failed_tests,
    write_markdown,
)


class FailureReportingTest(unittest.TestCase):
    def test_auto_classify_accepts_custom_primary_label(self):
        fields = [
            "test_file",
            "status_preview",
            "message_preview",
            "status_cuda",
            "message_cuda",
        ]
        self.assertEqual(
            detect_columns(fields),
            ("status_preview", "status_cuda", "message_preview"),
        )

    def test_log_crash_is_promoted_and_counted(self):
        base = {
            "arch": "preview",
            "platform": "rocm",
            "test_config": "inductor",
            "test_file": "inductor/test_origami",
            "job_shard": "1/2",
            "test_shard": "1/1",
            "reason": (
                "TestOrigami::"
                "test_origami_reduces_compile_work_vs_regular_max_autotune"
            ),
            "job_url": "https://github.com/pytorch/pytorch/actions/runs/1/job/2",
        }
        one_off = dict(
            base, status="FAILED", category="SEGFAULT")
        consistent = dict(
            base, status="FAILED_CONSISTENTLY",
            category="CONSISTENT_FAILURE")

        promoted = collect_log_failed_tests(
            [one_off, consistent], [], "preview")

        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["status_preview"], "FAILED")
        self.assertIn("CONSISTENT_FAILURE", promoted[0]["error_message"])

        rows = [
            ("__section__", "TEST INDUCTOR"),
            ("PREVIEW", [10]),
            ("__section__", "OVERALL"),
            ("FAILED(preview)", [0]),
            ("TOTAL PREVIEW", [10]),
        ]
        add_promoted_failure_stats(rows, ["preview"], promoted, "preview")
        self.assertEqual(rows[1][1], [11])
        self.assertEqual(rows[3][1], [1])
        self.assertEqual(rows[4][1], [11])

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "summary.md")
            markdown = write_markdown(
                rows,
                ["preview"],
                output,
                failed_tests=promoted,
                s1_name="preview",
                s2_name="cuda",
                has_set2=False,
                log_failures=[one_off, consistent],
            )
        self.assertIn("### FAILED TESTS (1)", markdown)
        self.assertIn("CONSISTENT_FAILURE", markdown)
        self.assertNotIn("No failed tests found.", markdown)

    def _parse_shard_log(self, lines):
        running = (
            "2026-10-06T07:38:21.6395057Z Running inductor/test_aot_inductor 6/11"
            " ... [2026-10-06 07:38:21.639003][12082.382829547]"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rocm_1.txt")
            with open(path, "w") as f:
                f.write("\n".join([running] + lines) + "\n")
            results, _, _, _ = parse_log_file(path)
        return results["inductor/test_aot_inductor 6/11"]

    def test_crash_word_in_skip_reason_is_not_a_crash(self):
        info = self._parse_shard_log([
            "2026-10-06T07:49:06.2477712Z inductor/test_aot_inductor.py::"
            "AOTInductorTestABICompatibleCpu::test_nan_cpu SKIPPED [0.0003s] "
            "(Skip this test, only for local test. SIGABRT is produced.) [ 13%]",
            "2026-10-06T07:49:06.2600000Z inductor/test_aot_inductor.py::"
            "TestCheckUpperboundConfig::test_aoti_check_upperbound_codegen "
            "PASSED [1.2000s] [100%]",
        ])
        self.assertEqual(info["crashes"], [])

    def test_real_crash_is_still_detected(self):
        info = self._parse_shard_log([
            "2026-10-06T07:49:06.2477712Z inductor/test_aot_inductor.py::"
            "TestCheckUpperboundConfig::test_aoti_check_upperbound_codegen "
            "Fatal Python error: Aborted",
            "2026-10-06T07:49:06.2600000Z Got exit code -6 (SIGABRT)",
        ])
        self.assertIn("SIGABRT", info["crashes"])
        self.assertIn("FATAL_PYTHON", info["crashes"])

    def test_whole_file_log_failure_dropped_when_named_failure_exists(self):
        xml_failed = [{
            "arch": "mi200",
            "test_config": "default",
            "test_file": "cpp.test_api",
            "test_class": "test_api",
            "test_name": "RNNTest.BidirectionalLSTMReverseForward_CUDA",
        }]
        whole_file = {
            "arch": "mi200",
            "platform": "rocm",
            "test_config": "default",
            "test_file": "cpp/test_api",
            "job_shard": "1/10",
            "test_shard": "1/1",
            "status": "FAILED",
            "category": "FAILED",
            "reason": "",
        }

        promoted = collect_log_failed_tests(
            [whole_file], xml_failed, "mi200")

        self.assertEqual(promoted, [])

    def test_whole_file_log_failure_kept_without_named_failure(self):
        whole_file = {
            "arch": "mi200",
            "platform": "rocm",
            "test_config": "default",
            "test_file": "cpp/test_api",
            "job_shard": "1/10",
            "test_shard": "1/1",
            "status": "FAILED",
            "category": "FAILED",
            "reason": "",
        }
        other_file_failure = {
            "arch": "mi200",
            "test_config": "default",
            "test_file": "inductor.test_torchinductor_opinfo_properties",
            "test_class": "TestOpInfoPropertiesCUDA",
            "test_name": (
                "test_unary_ufunc_numerical_exp_backend_inductor_default_cuda_float32"
            ),
        }

        promoted = collect_log_failed_tests(
            [whole_file], [other_file_failure], "mi200")

        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["test_file"], "cpp/test_api")
        self.assertEqual(promoted[0]["test_name"], "")

    def test_log_failure_matches_xml_row_with_qualified_class(self):
        xml_failed = [{
            "arch": "mi350",
            "test_config": "distributed",
            "test_file": "distributed.test_symmetric_memory",
            "test_class": (
                "test.distributed.test_symmetric_memory.SymmetricMemoryTest"
            ),
            "test_name": "test_rendezvous_after_strict_subgroup",
        }]
        consistent = {
            "arch": "mi350",
            "platform": "rocm",
            "test_config": "distributed",
            "test_file": "distributed/test_symmetric_memory",
            "job_shard": "1/2",
            "test_shard": "1/1",
            "status": "FAILED_CONSISTENTLY",
            "category": "CONSISTENT_FAILURE",
            "reason": "SymmetricMemoryTest::test_rendezvous_after_strict_subgroup",
        }

        promoted = collect_log_failed_tests(
            [consistent], xml_failed, "mi350")

        self.assertEqual(promoted, [])


if __name__ == "__main__":
    unittest.main()
