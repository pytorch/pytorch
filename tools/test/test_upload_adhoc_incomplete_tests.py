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


def _write_junit_xml(
    dir_: Path, name: str, testcases: list[tuple[str, str, str]]
) -> None:
    """testcases items: (file, classname, name) — xunit2 attributes."""
    dir_.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f'<testcase file="{file_attr}" classname="{cls}" name="{n}"/>'
        for file_attr, cls, n in testcases
    )
    (dir_ / name).write_text(f'<?xml version="1.0"?><testsuite>{body}</testsuite>')


class TestUploadAdhocIncompleteTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.repo_root = Path(tmpdir.name)
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

    def test_skips_when_job_id_missing(self) -> None:
        _write_cache(
            self.stepcurrent / "key1",
            "test_foo.py::TestFoo::test_bar",
            made_failing_xml=False,
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(helper, "upload_adhoc_failure_json") as up,
        ):
            self.assertEqual(helper.main(), 0)
        up.assert_not_called()

    def test_emits_for_incomplete_test_not_in_xml(self) -> None:
        # An in-flight nodeid with no junit-xml row should produce exactly
        # one adhoc emission. Use the run_test.py-style stepcurrent_key
        # (test_file + shard + hex16 suffix) so the helper recovers
        # `invoking_file` from the dir name, not from the nodeid path.
        _write_cache(
            self.stepcurrent / "test_foo_1_aaaabbbbccccdddd",
            "test/dynamo/test_foo.py::TestFoo::test_bar",
            made_failing_xml=False,
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        self.assertEqual(up.call_count, 1)
        args, kwargs = up.call_args
        # invoking_file comes from stepcurrent_key prefix (the run_test.py
        # invocation unit), not the nodeid path.
        self.assertEqual(args[0], "test_foo")
        # current_failure (full nodeid) is passed positionally for the
        # logging/legacy fallback, but explicit kwargs override the splitter.
        self.assertEqual(args[1], "test/dynamo/test_foo.py::TestFoo::test_bar")
        self.assertEqual(kwargs["classname"], "TestFoo")
        self.assertEqual(kwargs["testname"], "test_bar")
        # file_attr is the rootdir-relative path from the nodeid, so the row
        # records the actual source file even when invoking_file differs.
        self.assertEqual(kwargs["file_attr"], "test/dynamo/test_foo.py")
        self.assertIn("SIGTERM", kwargs["reason"])
        # Deterministic suffix: sanitized prefix + 8-char hash of full nodeid.
        self.assertTrue(
            kwargs["s3_key_suffix"].startswith("incomplete_TestFoo_test_bar_"),
            f"suffix={kwargs['s3_key_suffix']!r}",
        )
        # 8 hex chars at the end
        self.assertRegex(kwargs["s3_key_suffix"], r"_[0-9a-f]{8}$")

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
        # Mirrors pytest xunit2 output: classname is dotted module path,
        # `file` is the rootdir-relative test file. Helper bridges xunit2's
        # dotted classname to nodeid's bare classname via last-segment
        # comparison, so the dedup key is (file, last_class_segment, name).
        _write_cache(
            self.stepcurrent / "key1",
            "test_foo.py::TestFoo::test_bar",
            made_failing_xml=False,
        )
        _write_junit_xml(
            self.test_reports,
            "report.xml",
            [("test_foo.py", "test_foo.TestFoo", "test_bar")],
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        up.assert_not_called()

    def test_does_not_skip_when_only_name_collides_across_classes(self) -> None:
        # Two classes in the same file can both define `test_setup`. xunit2
        # distinguishes via classname. The dedup must NOT suppress a real
        # incomplete row in TestBar just because TestFoo.test_setup ran.
        _write_cache(
            self.stepcurrent / "key1",
            "test_foo.py::TestBar::test_setup",
            made_failing_xml=False,
        )
        _write_junit_xml(
            self.test_reports,
            "report.xml",
            [("test_foo.py", "test_foo.TestFoo", "test_setup")],
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        up.assert_called_once()

    def test_does_not_skip_on_classname_only_match(self) -> None:
        # Same classname+name but different file → must still emit.
        _write_cache(
            self.stepcurrent / "key1",
            "test_a.py::TestFoo::test_bar",
            made_failing_xml=False,
        )
        _write_junit_xml(
            self.test_reports,
            "other.xml",
            [("test_b.py", "test_b.TestFoo", "test_bar")],
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        up.assert_called_once()

    def test_function_test_dedup_against_module_only_classname(self) -> None:
        # Function-level test: nodeid has no class segment. xunit2 writes
        # classname=<module path>. The helper's last-segment normalization
        # treats single-segment xunit2 classnames as no-class, matching the
        # nodeid's empty classname.
        _write_cache(
            self.stepcurrent / "key1",
            "test_foo.py::test_top_level",
            made_failing_xml=False,
        )
        _write_junit_xml(
            self.test_reports,
            "report.xml",
            [("test_foo.py", "test_foo", "test_top_level")],
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        up.assert_not_called()

    def test_parametrized_nodeid_with_double_colon(self) -> None:
        # Legal parametrized id like `test_foo.py::test_bar[a::b]` must NOT
        # be mis-split on inner `::`.
        _write_cache(
            self.stepcurrent / "key1",
            "test/test_foo.py::test_bar[a::b]",
            made_failing_xml=False,
        )
        with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
            self.assertEqual(helper.main(), 0)
        self.assertEqual(up.call_count, 1)
        kwargs = up.call_args.kwargs
        self.assertEqual(kwargs["classname"], "")
        self.assertEqual(kwargs["testname"], "test_bar[a::b]")

    def test_lossy_sanitizer_collision_avoided_by_hash(self) -> None:
        # `a:b`, `a::b`, `a/b`, `a b` all sanitize to `a_b`. The 8-char
        # nodeid-hash suffix must keep them distinct.
        for v in ("a:b", "a::b", "a/b", "a b"):
            (self.stepcurrent / "key1").mkdir(parents=True, exist_ok=True)
            (self.stepcurrent / "key1" / "lastrun").write_text(
                json.dumps(f"test_foo.py::test_x[{v}]")
            )
            (self.stepcurrent / "key1" / "made_failing_xml").write_text("false")
            with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
                self.assertEqual(helper.main(), 0)
            self.assertEqual(up.call_count, 1, f"variant={v!r}")
        # All four invocations should have produced distinct s3 keys.
        # (Re-run all and collect.)
        suffixes: set[str] = set()
        for v in ("a:b", "a::b", "a/b", "a b"):
            (self.stepcurrent / "key1" / "lastrun").write_text(
                json.dumps(f"test_foo.py::test_x[{v}]")
            )
            with mock.patch.object(helper, "upload_adhoc_failure_json") as up:
                helper.main()
            suffixes.add(up.call_args.kwargs["s3_key_suffix"])
        self.assertEqual(len(suffixes), 4, suffixes)

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
