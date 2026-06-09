import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.stats.upload_test_stats import (
    backfill_test_jsons_while_running,
    get_tests,
    summarize_test_cases,
)


IN_CI = os.environ.get("CI")

_MINIMAL_JUNIT_XML = (
    '<testsuite><testcase classname="C" name="t" time="0"/></testsuite>'
)


class TestUploadTestStats(unittest.TestCase):
    @unittest.skipIf(
        IN_CI,
        "don't run in CI as this does a lot of network calls and uses up GH API rate limit",
    )
    def test_existing_job(self) -> None:
        """Run on a known-good job and make sure we don't error and get basically okay results."""
        test_cases = get_tests(2561394934, 1)
        self.assertEqual(len(test_cases), 609873)
        summary = summarize_test_cases(test_cases)
        self.assertEqual(len(summary), 5068)

    def test_backfill_skips_unexpected_report_layout(self) -> None:
        """ROCm gfx950 jobs store reports under <dir>/test-reports instead of the
        usual <dir>/test/test-reports. Backfill must skip those rather than
        raising ValueError, while still uploading reports in the expected layout.
        """
        # backfill chdir's into a TemporaryDirectory; restore cwd afterwards.
        self.addCleanup(os.chdir, os.getcwd())

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)

            normal_dir = root / "normal"
            normal_xml = (
                normal_dir / "test" / "test-reports" / "python-pytest" / "foo-1.xml"
            )
            normal_xml.parent.mkdir(parents=True)
            normal_xml.write_text(_MINIMAL_JUNIT_XML)

            rocm_dir = root / "rocm"
            rocm_xml = rocm_dir / "test-reports" / "python-pytest" / "bar-1.xml"
            rocm_xml.parent.mkdir(parents=True)
            rocm_xml.write_text(_MINIMAL_JUNIT_XML)

            def fake_download(prefix: str, *_a: object, **_k: object) -> list[str]:
                # No pre-existing test-jsons; two test-report artifacts.
                return ["normal.zip", "rocm.zip"] if prefix == "test-report" else []

            unzip_map = {"normal.zip": normal_dir, "rocm.zip": rocm_dir}

            with (
                mock.patch(
                    "tools.stats.upload_test_stats.download_s3_artifacts",
                    side_effect=fake_download,
                ),
                mock.patch(
                    "tools.stats.upload_test_stats.unzip",
                    side_effect=lambda path: unzip_map[path],
                ),
                mock.patch(
                    "tools.stats.upload_test_stats.get_job_id", return_value=123
                ),
                mock.patch("tools.stats.upload_test_stats.upload_to_s3") as mock_upload,
            ):
                # Must not raise on the ROCm layout.
                backfill_test_jsons_while_running(1, 1)

            uploaded_keys = [call.args[1] for call in mock_upload.call_args_list]
            self.assertEqual(len(uploaded_keys), 1)
            self.assertIn("foo-1", uploaded_keys[0])
            self.assertFalse(any("bar-1" in key for key in uploaded_keys))


if __name__ == "__main__":
    unittest.main()
