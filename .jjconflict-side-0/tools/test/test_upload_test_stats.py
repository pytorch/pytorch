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

    def test_backfill_uploads_both_report_layouts(self) -> None:
        """Reports usually live under <dir>/test/test-reports, but ROCm jobs
        upload them under <dir>/test-reports (no test/ prefix). Backfill keys off
        the nearest test-reports dir and uploads both, while skipping reports
        that have no test-reports dir at all.
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

            other_dir = root / "other"
            other_xml = other_dir / "some" / "dir" / "baz-1.xml"
            other_xml.parent.mkdir(parents=True)
            other_xml.write_text(_MINIMAL_JUNIT_XML)

            zips = ["normal.zip", "rocm.zip", "other.zip"]

            def fake_download(prefix: str, *_a: object, **_k: object) -> list[str]:
                # No pre-existing test-jsons; three test-report artifacts.
                return zips if prefix == "test-report" else []

            unzip_map = {
                "normal.zip": normal_dir,
                "rocm.zip": rocm_dir,
                "other.zip": other_dir,
            }

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
                backfill_test_jsons_while_running(1, 1)

            joined = " ".join(call.args[1] for call in mock_upload.call_args_list)
            self.assertEqual(len(mock_upload.call_args_list), 2)
            self.assertIn("python-pytest_foo-1.json", joined)
            self.assertIn("python-pytest_bar-1.json", joined)
            self.assertNotIn("baz-1", joined)


if __name__ == "__main__":
    unittest.main()
