import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


# `tools.testing.upload_artifacts` imports `boto3` and `filelock` at module
# load. Stub them out so the test runs without those packages in the env.
for missing in ("boto3", "filelock", "requests"):
    sys.modules.setdefault(missing, mock.MagicMock())


from tools.testing import upload_adhoc_incomplete_tests as helper  # noqa: E402


def _write_cache(dir_: Path, lastrun: Any, made_failing_xml: bool) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "lastrun").write_text(json.dumps(lastrun))
    (dir_ / "made_failing_xml").write_text(json.dumps(made_failing_xml))


def _write_junit_xml(dir_: Path, name: str, testcases: list[tuple[str, str]]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f'<testcase classname="{cls}" name="{n}"/>' for cls, n in testcases
    )
    (dir_ / name).write_text(f'<?xml version="1.0"?><testsuite>{body}</testsuite>')


class TestUploadAdhocIncompleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = self.enterContext(__import__("tempfile").TemporaryDirectory())
        self.repo_root = Path(self.tmp)
        self.stepcurrent = self.repo_root / ".pytest_cache" / "v" / "cache" / "stepcurrent"
        self.test_reports = self.repo_root / "test" / "test-reports"
        # Repoint the helper's module-level paths at the temp dir.
        self.patches = [
            mock.patch.object(helper, "REPO_ROOT", self.repo_root),
            mock.patch.object(helper, "STEPCURRENT_DIR", self.stepcurrent),
            mock.patch.object(helper, "TEST_REPORTS_DIR", self.test_reports),
            mock.patch.dict(
                os.environ,
                {"JOB_ID": "1", "GITHUB_RUN_ID": "2", "GITHUB_RUN_ATTEMPT": "1"},
                clear=False,
            ),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_emits_for_incomplete_test_not_in_xml(self) -> None:
        # An in-flight nodeid with no junit-xml row should produce exactly
        # one adhoc emission.
        _write_cache(
            self.stepcurrent / "key1",
            "test/dynamo/test_foo.py::TestFoo::test_bar",
            made_failing_xml=False,
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        self.assertEqual(up.call_count, 1)
        args, kwargs = up.call_args
        self.assertEqual(args[0], "test/dynamo/test_foo")
        self.assertEqual(args[1], "TestFoo::test_bar")
        self.assertIn("SIGTERM", kwargs["reason"])
        # Deterministic suffix keyed by (classname, testname).
        self.assertEqual(
            kwargs["s3_key_suffix"], "incomplete_TestFoo_test_bar"
        )

    def test_skips_when_made_failing_xml_true(self) -> None:
        # pytest reached sessionfinish with exit != 0 — the existing failure
        # path will have emitted the row (or the XML is correct). Skip.
        _write_cache(
            self.stepcurrent / "key1",
            "test_foo.py::test_bar",
            made_failing_xml=True,
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        up.assert_not_called()

    def test_skips_when_testcase_already_in_junit_xml(self) -> None:
        # Test ran to completion (its testcase is in the xml) and a LATER
        # test was killed mid-run. Don't double-emit for the completed one.
        _write_cache(
            self.stepcurrent / "key1",
            "test_foo.py::TestFoo::test_bar",
            made_failing_xml=False,
        )
        _write_junit_xml(
            self.test_reports,
            "report.xml",
            [("TestFoo", "test_bar")],
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        up.assert_not_called()

    def test_skips_when_lastrun_missing_or_null(self) -> None:
        _write_cache(
            self.stepcurrent / "key1",
            None,  # pytest didn't get to start any test
            made_failing_xml=False,
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        up.assert_not_called()

    def test_handles_partial_xml_gracefully(self) -> None:
        # A partially-written xml is treated as no data so the helper still
        # emits the synthetic row for the in-flight test.
        _write_cache(
            self.stepcurrent / "key1",
            "test_foo.py::TestFoo::test_bar",
            made_failing_xml=False,
        )
        self.test_reports.mkdir(parents=True, exist_ok=True)
        (self.test_reports / "broken.xml").write_text("<testsuite><testc")
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        up.assert_called_once()

    def test_handles_multiple_stepcurrent_dirs(self) -> None:
        _write_cache(
            self.stepcurrent / "key1",
            "test_a.py::TestA::test_one",
            made_failing_xml=False,
        )
        _write_cache(
            self.stepcurrent / "key2",
            "test_b.py::TestB::test_two",
            made_failing_xml=False,
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        self.assertEqual(up.call_count, 2)


if __name__ == "__main__":
    unittest.main()
