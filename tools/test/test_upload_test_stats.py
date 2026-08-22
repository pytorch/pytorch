import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.stats.upload_test_stats import (
    backfill_test_jsons_while_running,
    get_tests,
    parse_xml_report,
    sanitize_case_inplace,
    summarize_test_cases,
)


IN_CI = os.environ.get("CI")

_MINIMAL_JUNIT_XML = (
    '<testsuite><testcase classname="C" name="t" time="0"/></testsuite>'
)

# A raw pytest xunit2 testcase: dotted package-path classname and no `file`
# attribute. This is the shape that leaks to ClickHouse with file='' when the
# in-process sanitize pass (common_utils.sanitize_pytest_xml) is skipped because
# the test subprocess was killed (shard timeout, OOM, teardown crash).
_UNSANITIZED_PYTEST_XML = (
    "<testsuite>"
    '<testcase classname="test.test_ops.TestCommonCPU" name="test_foo_cpu" time="1.5"/>'
    "</testsuite>"
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


class TestSanitizeCaseInplace(unittest.TestCase):
    def test_backfills_file_from_dotted_classname(self) -> None:
        case = {"classname": "test.test_ops.TestCommonCPU", "name": "test_foo"}
        sanitize_case_inplace(case)
        self.assertEqual(case["file"], "test_ops.py")
        self.assertEqual(case["classname"], "TestCommonCPU")

    def test_nested_module_path_becomes_directory(self) -> None:
        case = {"classname": "test.inductor.test_torchinductor.TestInductorCPU"}
        sanitize_case_inplace(case)
        self.assertEqual(case["file"], "inductor/test_torchinductor.py")
        self.assertEqual(case["classname"], "TestInductorCPU")

    def test_no_test_prefix(self) -> None:
        # classname without the leading "test." package prefix still resolves.
        case = {"classname": "test_ops.TestCommonCPU"}
        sanitize_case_inplace(case)
        self.assertEqual(case["file"], "test_ops.py")
        self.assertEqual(case["classname"], "TestCommonCPU")

    def test_idempotent(self) -> None:
        case = {"classname": "test.test_ops.TestCommonCPU"}
        sanitize_case_inplace(case)
        first = dict(case)
        sanitize_case_inplace(case)  # second pass must be a no-op
        self.assertEqual(case, first)

    def test_leaves_already_sanitized_case_untouched(self) -> None:
        case = {"classname": "TestCommonCPU", "file": "test_ops.py"}
        before = dict(case)
        sanitize_case_inplace(case)
        self.assertEqual(case, before)

    def test_never_overwrites_a_populated_file(self) -> None:
        # Defensive: even with a dotted classname, an existing file wins.
        case = {"classname": "test.weird.Thing", "file": "real_file.py"}
        before = dict(case)
        sanitize_case_inplace(case)
        self.assertEqual(case, before)

    def test_dotless_classname_without_file_is_left_alone(self) -> None:
        # No dot -> nothing to derive -> leave the row as-is rather than guess.
        case = {"classname": "TestCommonCPU"}
        sanitize_case_inplace(case)
        self.assertNotIn("file", case)
        self.assertEqual(case["classname"], "TestCommonCPU")

    def test_degenerate_dotted_classname_left_alone(self) -> None:
        # Leading/trailing-dot classnames are malformed; don't emit a
        # nonsensical ".py" file or an empty classname -- leave them untouched.
        for bad in (".TestFoo", "test.test_ops."):
            case = {"classname": bad}
            sanitize_case_inplace(case)
            self.assertNotIn("file", case)
            self.assertEqual(case["classname"], bad)

    def test_non_string_classname_ignored(self) -> None:
        case = {"classname": 123}
        sanitize_case_inplace(case)
        self.assertNotIn("file", case)

    def test_missing_classname_ignored(self) -> None:
        case: dict = {"name": "t"}
        sanitize_case_inplace(case)
        self.assertNotIn("file", case)


class TestParseXmlReportSanitizes(unittest.TestCase):
    def test_parse_xml_report_backfills_empty_file(self) -> None:
        """End-to-end: an unsanitized pytest XML parses into a case with `file`
        backfilled and the classname stripped to its bare form.
        """
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "test_ops" / "python-pytest_test_ops.xml"
            report.parent.mkdir(parents=True)
            report.write_text(_UNSANITIZED_PYTEST_XML)

            cases = parse_xml_report("testcase", report, 99, 1, job_id=7)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["file"], "test_ops.py")
            self.assertEqual(cases[0]["classname"], "TestCommonCPU")
            self.assertEqual(cases[0]["job_id"], 7)


if __name__ == "__main__":
    unittest.main()
