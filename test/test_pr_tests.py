# Owner(s): ["module: ci"]

import http.client
import io
import os
import subprocess
import threading
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from email.message import Message
from unittest import mock

import pr_tests

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def make_report_zip(*testcases: list[dict[str, str]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for index, cases in enumerate(testcases):
            testsuite = ET.Element("testsuite")
            for case in cases:
                ET.SubElement(testsuite, "testcase", case)
            archive.writestr(
                f"test-reports/report-{index}.xml",
                ET.tostring(testsuite),
            )
        archive.writestr("unrelated.csv", "value\n")
    return output.getvalue()


def make_s3_page(
    keys: list[str], *, truncated: bool = False, token: str | None = None
) -> bytes:
    contents = "".join(
        f"<Contents><Key>{key}</Key>"
        "<LastModified>2026-07-19T00:00:00Z</LastModified></Contents>"
        for key in keys
    )
    next_token = (
        f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>{str(truncated).lower()}</IsTruncated>
  {next_token}
  {contents}
</ListBucketResult>
""".encode()


class Response(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or Message()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BrokenResponse(Response):
    def read(self, *args):
        raise http.client.IncompleteRead(b"partial")


class TestPRTests(TestCase):
    @parametrize(
        "value",
        [
            "190437",
            "https://github.com/pytorch/pytorch/pull/190437/files",
        ],
    )
    def test_parse_pr(self, value):
        self.assertEqual(pr_tests.parse_pr(value), ("pytorch/pytorch", 190437))

    def test_parse_pr_rejects_invalid_input(self):
        with self.assertRaisesRegex(pr_tests.PRTestsError, "PR must be"):
            pr_tests.parse_pr("pytorch/pytorch#190437")

    def test_pr_info_uses_canonical_repository(self):
        response = {
            "head": {"sha": "sha"},
            "base": {"repo": {"full_name": "pytorch/pytorch"}},
        }
        with mock.patch.object(pr_tests, "_github_json", return_value=response):
            self.assertEqual(
                pr_tests._pr_info("PyTorch/PyTorch", 190437, "token"),
                ("pytorch/pytorch", "sha"),
            )

    def test_workflow_run_pagination(self):
        first_page = [
            {"id": index, "status": "completed", "conclusion": "success"}
            for index in range(100)
        ]
        second_page = [{"id": 100, "status": "completed", "conclusion": "success"}]
        with mock.patch.object(
            pr_tests,
            "_github_json",
            side_effect=[
                {"total_count": 101, "workflow_runs": first_page},
                {"total_count": 101, "workflow_runs": second_page},
            ],
        ) as github_json:
            runs = pr_tests._workflow_runs("pytorch/pytorch", "sha", "token")

        self.assertEqual(len(runs), 101)
        self.assertIn("page=1", github_json.call_args_list[0].args[0])
        self.assertIn("page=2", github_json.call_args_list[1].args[0])

    def test_workflow_run_cap_fails_closed(self):
        with mock.patch.object(
            pr_tests,
            "_github_json",
            return_value={"total_count": 1001, "workflow_runs": []},
        ):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "more than 1,000"):
                pr_tests._workflow_runs("pytorch/pytorch", "sha", "token")

    def test_s3_artifact_pagination(self):
        prefix = "pytorch/pytorch/123/"
        first_page = make_s3_page(
            [
                f"{prefix}1/artifact/test-reports-test-default_11.zip",
                f"{prefix}1/artifact/logs-test-default_11.zip",
            ],
            truncated=True,
            token="next token",
        )
        second_page = make_s3_page(
            [f"{prefix}2/artifact/test-reports-test-dynamo_22.zip"]
        )
        with mock.patch.object(
            pr_tests, "_request_bytes", side_effect=[first_page, second_page]
        ) as request_bytes:
            artifacts = pr_tests._s3_artifacts("pytorch/pytorch", 123)

        self.assertEqual([artifact.job_id for artifact in artifacts], [11, 22])
        second_url = request_bytes.call_args_list[1].args[0]
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(second_url).query)
        self.assertEqual(query["continuation-token"], ["next token"])

    def test_artifact_deduplication_preserves_distinct_copies(self):
        old = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "s3",
            "s3",
            11,
            updated_at=datetime.fromisoformat("2026-07-01T00:00:00+00:00"),
        )
        new = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "github",
            "github",
            11,
            updated_at=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
        )

        selected, expired = pr_tests._deduplicate_artifacts([old], [new])
        self.assertEqual(selected, [old, new])
        self.assertEqual(expired, [])

    def test_tests_from_zip(self):
        data = make_report_zip(
            [
                {
                    "file": "test_as_strided.py",
                    "classname": "TestAsStrided",
                    "name": "test_subset_property",
                },
                {
                    "file": "dynamo\\test_misc.py",
                    "classname": "dynamo.test_misc.TestMisc",
                    "name": "test_parameterized[cpu-float32]",
                },
                {
                    "file": "test_module.py",
                    "classname": "test_module",
                    "name": "test_function",
                },
                {
                    "file": "distributions.py",
                    "classname": "test_constraints",
                    "name": "test_biject_to[param]",
                },
                {"file": "cpp/test_api.cpp", "classname": "C", "name": "test"},
            ],
            [
                {
                    "file": "test_as_strided.py",
                    "classname": "TestAsStrided",
                    "name": "test_subset_property",
                }
            ],
        )
        tests, reports = pr_tests._tests_from_zip(data, "reports.zip")
        self.assertEqual(
            tests,
            {
                "test/test_as_strided.py::TestAsStrided::test_subset_property",
                "test/dynamo/test_misc.py::TestMisc::test_parameterized[cpu-float32]",
                "test/test_module.py::test_function",
                "test/distributions/test_constraints.py::test_biject_to[param]",
            },
        )
        self.assertEqual(reports, 2)

    def test_reports_are_streamed(self):
        data = make_report_zip(
            [{"file": "test_a.py", "classname": "TestA", "name": "test_a"}]
        )
        with mock.patch.object(
            pr_tests.ET,
            "fromstring",
            side_effect=AssertionError("reports must not be buffered"),
        ):
            tests, reports = pr_tests._tests_from_zip(data, "reports.zip")

        self.assertEqual(tests, {"test/test_a.py::TestA::test_a"})
        self.assertEqual(reports, 1)

    @parametrize(
        "file,classname",
        [
            ("sub/test_mod/Outer.py", "Inner"),
            ("sub/test_mod.py", "sub.test_mod.Outer.Inner"),
        ],
    )
    def test_nested_class_reconstruction(self, file, classname):
        testcase = ET.Element(
            "testcase",
            {"file": file, "classname": classname, "name": "test_nested"},
        )
        self.assertEqual(
            pr_tests._qualified_test_name(
                testcase, "reports.zip", report_module="sub.test_mod"
            ),
            "test/sub/test_mod.py::Outer::Inner::test_nested",
        )

    def test_nested_class_reconstruction_with_repository_prefix(self):
        testcase = ET.Element(
            "testcase",
            {
                "file": "sub/test_mod/Outer.py",
                "classname": "Inner",
                "name": "test_nested",
            },
        )
        report_module, report_directory = pr_tests._report_context(
            "test/test-reports/python-pytest/sub.test_mod/report.xml"
        )
        self.assertEqual(
            pr_tests._qualified_test_name(
                testcase, "reports.zip", report_module, report_directory
            ),
            "test/sub/test_mod.py::Outer::Inner::test_nested",
        )

    def test_aot_runner_alias(self):
        testcase = ET.Element(
            "testcase",
            {
                "file": "test_cpp_extensions_aot_ninja.py",
                "classname": "TestCppExtension",
                "name": "test_build",
            },
        )
        self.assertEqual(
            pr_tests._qualified_test_name(testcase, "reports.zip"),
            "test/test_cpp_extensions_aot.py::TestCppExtension::test_build",
        )

    def test_report_directory(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            testsuite = ET.Element("testsuite")
            ET.SubElement(
                testsuite,
                "testcase",
                {
                    "file": "test_custom_backend.py",
                    "classname": "TestCustomBackend",
                    "name": "test_execute",
                },
            )
            archive.writestr(
                "custom_backend/test-reports/python-unittest/test_custom_backend/"
                "TEST-TestCustomBackend.xml",
                ET.tostring(testsuite),
            )

        tests, reports = pr_tests._tests_from_zip(output.getvalue(), "reports.zip")
        self.assertEqual(
            tests,
            {
                "test/custom_backend/test_custom_backend.py::TestCustomBackend::test_execute"
            },
        )
        self.assertEqual(reports, 1)

    def test_invalid_report_zip(self):
        with self.assertRaisesRegex(pr_tests.PRTestsError, "not a valid ZIP"):
            pr_tests._tests_from_zip(b"not a zip", "reports.zip")

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("invalid.xml", "not xml")
        with self.assertRaisesRegex(pr_tests.PRTestsError, "invalid test report"):
            pr_tests._tests_from_zip(output.getvalue(), "reports.zip")

    @parametrize("encoding", ["utf-8", "utf-16"])
    def test_rejects_xml_entities(self, encoding):
        report = f"""<?xml version="1.0" encoding="{encoding}"?>
<!DOCTYPE testsuite [<!ENTITY value "expanded">]>
<testsuite><testcase file="test_a.py" classname="TestA" name="&value;"/></testsuite>
""".encode(encoding)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("test-reports/entities.xml", report)

        with self.assertRaisesRegex(pr_tests.PRTestsError, "unsafe XML"):
            pr_tests._tests_from_zip(output.getvalue(), "reports.zip")

    def test_allows_xml_declaration_text_in_cdata(self):
        report = b"""<testsuite>
<testcase file="test_a.py" classname="TestA" name="test_a">
<system-out><![CDATA[<!DOCTYPE html>]]></system-out>
</testcase>
</testsuite>
"""
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("test-reports/report.xml", report)

        tests, reports = pr_tests._tests_from_zip(output.getvalue(), "reports.zip")
        self.assertEqual(tests, {"test/test_a.py::TestA::test_a"})
        self.assertEqual(reports, 1)

    def test_rejects_unsupported_report_compression(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_BZIP2) as archive:
            archive.writestr("test-reports/report.xml", "<testsuite/>")

        with self.assertRaisesRegex(pr_tests.PRTestsError, "unsupported.*compression"):
            pr_tests._tests_from_zip(output.getvalue(), "reports.zip")

    def test_report_size_limits(self):
        data = make_report_zip(
            [{"file": "test_a.py", "classname": "TestA", "name": "test_a"}]
        )
        with mock.patch.object(pr_tests, "_MAX_REPORT_BYTES", 1):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "oversized"):
                pr_tests._tests_from_zip(data, "reports.zip")

        with (
            mock.patch.object(pr_tests, "_MAX_REPORT_BYTES", 1024),
            mock.patch.object(pr_tests, "_MAX_REPORT_TOTAL_BYTES", 1),
        ):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "too much"):
                pr_tests._tests_from_zip(data, "reports.zip")

    def test_report_depth_limit(self):
        output = io.BytesIO()
        depth = pr_tests._MAX_XML_DEPTH + 1
        report = f"{'<node>' * depth}{'</node>' * depth}"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("test-reports/deep.xml", report)

        with self.assertRaisesRegex(pr_tests.PRTestsError, "excessively nested"):
            pr_tests._tests_from_zip(output.getvalue(), "reports.zip")

    def test_report_collection_size_limit(self):
        data = make_report_zip(
            [{"file": "test_a.py", "classname": "TestA", "name": "test_a"}]
        )
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            report_bytes = sum(
                member.file_size
                for member in archive.infolist()
                if member.filename.endswith(".xml")
            )
        budget = pr_tests._Budget(report_bytes, "collection limit")

        pr_tests._tests_from_zip(data, "first.zip", reserve_report_bytes=budget.reserve)
        with self.assertRaisesRegex(pr_tests.PRTestsError, "collection limit"):
            pr_tests._tests_from_zip(
                data, "second.zip", reserve_report_bytes=budget.reserve
            )

    def test_collection_budget_is_thread_safe(self):
        workers = 8
        budget = pr_tests._Budget(1, "collection limit")
        barrier = threading.Barrier(workers)
        outcomes: list[bool | None] = [None] * workers

        def reserve(index):
            barrier.wait()
            try:
                budget.reserve(1)
            except pr_tests.PRTestsError:
                outcomes[index] = False
            else:
                outcomes[index] = True

        threads = [
            threading.Thread(target=reserve, args=(index,)) for index in range(workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(outcomes.count(True), 1)
        self.assertEqual(outcomes.count(False), workers - 1)

    def test_report_cardinality_limits(self):
        cases = [
            {"file": "test_a.py", "classname": "TestA", "name": f"test_{index}"}
            for index in range(2)
        ]
        with mock.patch.object(pr_tests, "_MAX_TESTCASES_PER_ARTIFACT", 1):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "too many test cases"):
                pr_tests._tests_from_zip(make_report_zip(cases), "reports.zip")

        with mock.patch.object(pr_tests, "_MAX_REPORTS_PER_ARTIFACT", 1):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "too many test reports"):
                pr_tests._tests_from_zip(make_report_zip([], []), "reports.zip")

    def test_zip_member_limit(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("first.txt", "")
            archive.writestr("second.txt", "")

        with mock.patch.object(pr_tests, "_MAX_ZIP_MEMBERS", 1):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "too many files"):
                pr_tests._tests_from_zip(output.getvalue(), "reports.zip")

    def test_rejects_unsafe_test_name(self):
        data = make_report_zip(
            [
                {
                    "file": "test_safe.py",
                    "classname": "TestSafe",
                    "name": "test_safe\nforged",
                }
            ]
        )
        with self.assertRaisesRegex(pr_tests.PRTestsError, "invalid test name"):
            pr_tests._tests_from_zip(data, "reports.zip")

    def test_request_retries_transient_error(self):
        headers = Message()
        error = urllib.error.HTTPError(
            "https://example.com", 503, "unavailable", headers, None
        )
        with (
            mock.patch.object(
                pr_tests,
                "_open_request",
                side_effect=[error, Response(b"ok")],
            ),
            mock.patch.object(pr_tests.time, "sleep") as sleep,
        ):
            self.assertEqual(pr_tests._request_bytes("https://example.com"), b"ok")
        sleep.assert_called_once_with(0.5)

    def test_request_retries_incomplete_read(self):
        with (
            mock.patch.object(
                pr_tests,
                "_open_request",
                side_effect=[BrokenResponse(b""), Response(b"ok")],
            ),
            mock.patch.object(pr_tests.time, "sleep") as sleep,
        ):
            self.assertEqual(pr_tests._request_bytes("https://example.com"), b"ok")
        sleep.assert_called_once_with(0.5)

    def test_request_rejects_oversized_response(self):
        with mock.patch.object(
            pr_tests, "_open_request", return_value=Response(b"large")
        ):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "exceeded 4 bytes"):
                pr_tests._request_bytes("https://example.com", max_bytes=4)

    def test_redirect_drops_github_authorization(self):
        request = urllib.request.Request(
            "https://api.github.com/repos/pytorch/pytorch/actions/artifacts/1/zip",
            headers={"Authorization": "Bearer secret"},
        )
        redirected = pr_tests._SafeRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            Message(),
            "https://signed.example.com/artifact.zip",
        )
        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_redirect_rejects_https_downgrade(self):
        request = urllib.request.Request(
            "https://api.github.com/repos/pytorch/pytorch/actions/artifacts/1/zip",
            headers={"Authorization": "Bearer secret"},
        )
        with self.assertRaisesRegex(pr_tests.PRTestsError, "non-HTTPS"):
            pr_tests._SafeRedirectHandler().redirect_request(
                request,
                None,
                302,
                "Found",
                Message(),
                "http://api.github.com/artifact.zip",
            )

    def test_github_token_pins_hostname(self):
        result = subprocess.CompletedProcess([], 0, stdout="token\n")
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(pr_tests.subprocess, "run", return_value=result) as run,
        ):
            self.assertEqual(pr_tests._github_token(), "token")
        self.assertEqual(
            run.call_args.args[0],
            ["gh", "auth", "token", "--hostname", "github.com"],
        )

    @parametrize(
        "name,is_test_job",
        [
            ("linux / test (default, 1, 1, runner)", True),
            ("linux / test-osdc (default, 1, 1, runner)", True),
            ("dynamo / dynamo-test (3.11) / test (default, 1, 1, runner)", True),
            ("linux / build (default, runner)", False),
            ("linux / build-osdc (default, runner)", False),
            ("cross-compile-linux-test-cuda13 / build-osdc (runner)", False),
            ("linux / test-foo (default, runner)", False),
            ("stable-fa3 / test", False),
            ("linux / test (default, runner) / build", False),
            ("linux / test (default) / build (runner)", False),
        ],
    )
    def test_workflow_job_keys_detect_test_jobs(self, name, is_test_job):
        job = {
            "id": 11,
            "name": name,
            "run_attempt": 1,
            "started_at": "2026-07-01T00:00:00Z",
            "completed_at": "2026-07-01T02:00:00Z",
            "status": "completed",
            "conclusion": "success",
        }
        job_keys, expected = pr_tests._workflow_job_keys(10, [job])

        expected_name = pr_tests._logical_test_job_name(name) if is_test_job else name
        self.assertEqual(job_keys[11].key, (10, expected_name))
        self.assertEqual(job_keys[11].run_attempt, 1 if is_test_job else None)
        self.assertEqual(bool(expected), is_test_job)

    def test_workflow_job_keys_reject_duplicate_attempt_names(self):
        jobs = [
            {
                "id": job_id,
                "name": f"cross / test (aoti, 1, 1, {runner}, win-vs2022-cuda13.0-py3)",
                "run_attempt": 1,
                "started_at": "2026-07-01T00:00:00Z",
                "completed_at": "2026-07-01T02:00:00Z",
                "status": "completed",
                "conclusion": "success",
            }
            for job_id, runner in ((11, "mt-l-x86"), (22, "lf-l-x86"))
        ]
        with self.assertRaisesRegex(pr_tests.PRTestsError, "duplicate test jobs"):
            pr_tests._workflow_job_keys(10, jobs)

    def test_workflow_job_keys_require_run_attempt(self):
        job = {
            "id": 11,
            "name": "linux / test (default, 1, 1, runner)",
            "status": "completed",
            "conclusion": "success",
        }
        with self.assertRaisesRegex(pr_tests.PRTestsError, "invalid run attempt"):
            pr_tests._workflow_job_keys(10, [job])

    def test_workflow_job_keys_match_runner_pool_changes(self):
        jobs = [
            {
                "id": job_id,
                "name": f"cross / test (aoti, 1, 1, {runner}, win-vs2022-cuda13.0-py3)",
                "run_attempt": attempt,
                "started_at": f"2026-07-0{attempt}T00:00:00Z",
                "completed_at": f"2026-07-0{attempt}T02:00:00Z",
                "status": "completed",
                "conclusion": "success",
            }
            for job_id, attempt, runner in (
                (11, 1, "mt-l-x86aavx2-29-113-l4"),
                (22, 2, "lf-l-x86aavx2-29-113-l4"),
            )
        ]

        job_keys, expected = pr_tests._workflow_job_keys(10, jobs)

        self.assertEqual(job_keys[11].key, job_keys[22].key)
        self.assertEqual(len(expected), 1)
        self.assertEqual(next(iter(expected.values())).job_id, 22)

    def test_workflow_job_keys_keep_distinct_runner_hardware(self):
        jobs = [
            {
                "id": job_id,
                "name": f"macos / test (mps, 1, 1, {runner})",
                "run_attempt": 1,
                "started_at": "2026-07-01T00:00:00Z",
                "completed_at": "2026-07-01T02:00:00Z",
                "status": "completed",
                "conclusion": "success",
            }
            for job_id, runner in ((11, "macos-m1-14"), (22, "macos-m2-15"))
        ]

        job_keys, expected = pr_tests._workflow_job_keys(10, jobs)

        self.assertNotEqual(job_keys[11].key, job_keys[22].key)
        self.assertEqual(len(expected), 2)

    def test_rerun_coverage_requires_a_fresh_artifact(self):
        name = "linux / test (default, 1, 1, runner)"
        jobs = [
            {
                "id": 11,
                "name": name,
                "run_attempt": 1,
                "started_at": "2026-07-01T00:00:00Z",
                "completed_at": "2026-07-01T02:00:00Z",
                "status": "completed",
                "conclusion": "cancelled",
            },
            {
                "id": 22,
                "name": name,
                "run_attempt": 2,
                "started_at": "2026-07-02T00:00:00Z",
                "completed_at": "2026-07-02T02:00:00Z",
                "status": "completed",
                "conclusion": "success",
            },
        ]
        job_keys, expected = pr_tests._workflow_job_keys(10, jobs)
        key = (10, pr_tests._logical_test_job_name(name))
        self.assertEqual(expected[key].job_id, 22)
        self.assertEqual(expected[key].latest_run_attempt, 2)

        stale = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "s3",
            "s3",
            11,
            updated_at=datetime.fromisoformat("2026-07-01T01:00:00+00:00"),
        )
        fresh = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "s3",
            "s3",
            11,
            updated_at=datetime.fromisoformat("2026-07-02T01:00:00+00:00"),
        )
        self.assertEqual(pr_tests._covered_job_keys([stale], job_keys, expected), set())
        self.assertEqual(pr_tests._covered_job_keys([fresh], job_keys, expected), {key})
        delayed_exact = pr_tests.Artifact(
            "test-reports-runattempt2-test-default_22.zip",
            "s3-exact",
            "s3",
            22,
            updated_at=datetime.fromisoformat("2026-07-02T04:00:00+00:00"),
        )
        self.assertEqual(
            pr_tests._covered_job_keys([delayed_exact], job_keys, expected), {key}
        )

        jobs[0]["conclusion"] = "success"
        jobs[1]["conclusion"] = "cancelled"
        job_keys, expected = pr_tests._workflow_job_keys(10, jobs)
        self.assertEqual(expected[key].job_id, 11)
        self.assertEqual(expected[key].latest_run_attempt, 2)
        self.assertEqual(pr_tests._covered_job_keys([stale], job_keys, expected), {key})
        self.assertEqual(
            pr_tests._covered_job_keys([stale], job_keys, expected, exact=True), set()
        )
        original = pr_tests.Artifact(
            "test-reports-runattempt1-test-default_11.zip",
            "github-original",
            "github",
            11,
            updated_at=datetime.fromisoformat("2026-07-01T01:30:00+00:00"),
        )
        self.assertEqual(
            pr_tests._select_artifacts([stale, original], job_keys, expected),
            [original],
        )
        overwritten = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "overwritten",
            "s3",
            11,
            updated_at=datetime.fromisoformat("2026-07-02T01:00:00+00:00"),
        )
        self.assertEqual(
            pr_tests._covered_job_keys([overwritten], job_keys, expected), set()
        )
        newer_cancelled = pr_tests.Artifact(
            "test-reports-test-default_22.zip",
            "s3",
            "s3",
            22,
            updated_at=datetime.fromisoformat("2026-07-02T01:00:00+00:00"),
        )
        self.assertEqual(
            pr_tests._covered_job_keys([newer_cancelled], job_keys, expected), set()
        )

    def test_rerun_artifact_selection_uses_one_current_copy(self):
        name = "linux / test (default, 1, 1, runner)"
        jobs = [
            {
                "id": job_id,
                "name": name,
                "run_attempt": attempt,
                "started_at": f"2026-07-0{attempt}T00:00:00Z",
                "completed_at": f"2026-07-0{attempt}T02:00:00Z",
                "status": "completed",
                "conclusion": conclusion,
            }
            for job_id, attempt, conclusion in (
                (11, 1, "cancelled"),
                (22, 2, "success"),
            )
        ]
        job_keys, expected = pr_tests._workflow_job_keys(10, jobs)
        stale = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "stale",
            "s3",
            11,
            updated_at=datetime.fromisoformat("2026-07-01T01:00:00+00:00"),
        )
        mirrored = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "mirror",
            "s3",
            11,
            updated_at=datetime.fromisoformat("2026-07-02T01:00:00+00:00"),
        )
        current = pr_tests.Artifact(
            "test-reports-runattempt2-test-default_22.zip",
            "github",
            "github",
            22,
            updated_at=datetime.fromisoformat("2026-07-02T01:30:00+00:00"),
        )

        self.assertEqual(
            pr_tests._select_artifacts([stale, mirrored, current], job_keys, expected),
            [current],
        )

    def test_collects_and_deduplicates_artifacts(self):
        run = {"id": 10, "name": "pull", "status": "completed", "conclusion": "success"}
        jobs = [
            {
                "id": job_id,
                "name": f"linux / test (default, {index}, 2, runner)",
                "run_attempt": 1,
                "started_at": "2026-07-01T00:00:00Z",
                "completed_at": "2026-07-01T02:00:00Z",
                "status": "completed",
                "conclusion": "success",
            }
            for index, job_id in enumerate((11, 22), 1)
        ]
        s3 = pr_tests.Artifact("test-reports-test-default_11.zip", "s3", "s3", 11)
        duplicate = pr_tests.Artifact(
            "test-reports-runattempt1-test-default_11.zip", "github", "github", 11
        )
        fallback = pr_tests.Artifact(
            "test-reports-runattempt1-test-other_22.zip", "github2", "github", 22
        )
        payloads = {
            "test-reports-test-default_11.zip": make_report_zip(
                [
                    {
                        "file": "test_autograd.py",
                        "classname": "TestAutograd",
                        "name": "test_grad",
                    }
                ]
            ),
            "test-reports-runattempt1-test-other_22.zip": make_report_zip(
                [
                    {
                        "file": "dynamo/test_misc.py",
                        "classname": "TestMisc",
                        "name": "test_misc",
                    }
                ]
            ),
        }
        with (
            mock.patch.object(
                pr_tests, "_pr_info", return_value=("pytorch/pytorch", "sha")
            ),
            mock.patch.object(pr_tests, "_workflow_runs", return_value=[run]),
            mock.patch.object(pr_tests, "_workflow_jobs", return_value=jobs),
            mock.patch.object(pr_tests, "_s3_artifacts", return_value=[s3]),
            mock.patch.object(
                pr_tests, "_github_artifacts", return_value=[duplicate, fallback]
            ),
            mock.patch.object(
                pr_tests,
                "_download_artifact",
                side_effect=lambda artifact, token: payloads[artifact.name],
            ) as download,
        ):
            result = pr_tests.collect_pr_tests("190437", token="token")

        self.assertEqual(
            result.tests,
            frozenset(
                {
                    "test/test_autograd.py::TestAutograd::test_grad",
                    "test/dynamo/test_misc.py::TestMisc::test_misc",
                }
            ),
        )
        self.assertEqual(result.artifacts, 2)
        self.assertEqual(result.reports, 2)
        self.assertEqual(download.call_count, 2)

    def test_exact_github_artifact_wins_over_cross_attempt_s3(self):
        run = {
            "id": 10,
            "name": "pull",
            "status": "completed",
            "conclusion": "success",
        }
        job_name = "linux / test (default, 1, 1, runner)"
        jobs = [
            {
                "id": job_id,
                "name": job_name,
                "run_attempt": run_attempt,
                "started_at": f"2026-07-0{run_attempt}T00:00:00Z",
                "completed_at": f"2026-07-0{run_attempt}T02:00:00Z",
                "status": "completed",
                "conclusion": "success",
            }
            for job_id, run_attempt in ((11, 1), (22, 2))
        ]
        cross_attempt = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "s3",
            "s3",
            11,
            updated_at=datetime.fromisoformat("2026-07-02T01:00:00+00:00"),
        )
        current = pr_tests.Artifact(
            "test-reports-runattempt2-test-default_22.zip",
            "github",
            "github",
            22,
            updated_at=datetime.fromisoformat("2026-07-02T03:00:00+00:00"),
        )
        payloads = {
            "s3": make_report_zip(
                [{"file": "test_old.py", "classname": "TestOld", "name": "test_old"}]
            ),
            "github": make_report_zip(
                [
                    {
                        "file": "test_current.py",
                        "classname": "TestCurrent",
                        "name": "test_current",
                    }
                ]
            ),
        }
        expected_tests = frozenset({"test/test_current.py::TestCurrent::test_current"})
        with (
            mock.patch.object(
                pr_tests, "_pr_info", return_value=("pytorch/pytorch", "sha")
            ),
            mock.patch.object(pr_tests, "_workflow_runs", return_value=[run]),
            mock.patch.object(pr_tests, "_workflow_jobs", return_value=jobs),
            mock.patch.object(pr_tests, "_s3_artifacts", return_value=[cross_attempt]),
            mock.patch.object(
                pr_tests, "_github_artifacts", return_value=[current]
            ) as github_artifacts,
            mock.patch.object(
                pr_tests,
                "_download_artifact",
                side_effect=lambda artifact, token: payloads[artifact.url],
            ) as download,
        ):
            result = pr_tests.collect_pr_tests("190437", token="token")

        self.assertEqual(result.tests, expected_tests)
        self.assertEqual(result.artifacts, 1)
        github_artifacts.assert_called_once()
        download.assert_called_once_with(current, "token")

    def test_cancelled_workflow_uses_valid_prior_attempt(self):
        run = {
            "id": 10,
            "name": "pull",
            "status": "completed",
            "conclusion": "cancelled",
        }
        name = "linux / test (default, 1, 1, runner)"
        jobs = [
            {
                "id": job_id,
                "name": name,
                "run_attempt": attempt,
                "started_at": f"2026-07-0{attempt}T00:00:00Z",
                "completed_at": f"2026-07-0{attempt}T02:00:00Z",
                "status": "completed",
                "conclusion": conclusion,
            }
            for job_id, attempt, conclusion in (
                (11, 1, "success"),
                (22, 2, "cancelled"),
            )
        ]
        s3 = pr_tests.Artifact(
            "test-reports-test-default_11.zip",
            "s3",
            "s3",
            11,
            updated_at=datetime.fromisoformat("2026-07-01T01:00:00+00:00"),
        )
        github = pr_tests.Artifact(
            "test-reports-runattempt1-test-default_11.zip",
            "github",
            "github",
            11,
            updated_at=datetime.fromisoformat("2026-07-01T01:30:00+00:00"),
        )
        payload = make_report_zip(
            [{"file": "test_a.py", "classname": "TestA", "name": "test_a"}]
        )
        with (
            mock.patch.object(pr_tests, "_github_token", return_value=None),
            mock.patch.object(
                pr_tests, "_pr_info", return_value=("pytorch/pytorch", "sha")
            ),
            mock.patch.object(pr_tests, "_workflow_runs", return_value=[run]),
            mock.patch.object(pr_tests, "_workflow_jobs", return_value=jobs),
            mock.patch.object(pr_tests, "_s3_artifacts", return_value=[s3]),
            mock.patch.object(pr_tests, "_github_artifacts", return_value=[github]),
            mock.patch.object(
                pr_tests, "_download_artifact", return_value=payload
            ) as download,
        ):
            result = pr_tests.collect_pr_tests("190437")

        self.assertEqual(result.tests, {"test/test_a.py::TestA::test_a"})
        download.assert_called_once_with(s3, None)

    @parametrize(
        "artifact_job_id,byte_limit,error",
        [
            (22, 1024 * 1024, "associate 1 test report artifact"),
            (11, 1, "aggregate download limit"),
        ],
    )
    def test_rejects_unassociated_or_excessive_artifacts(
        self, artifact_job_id, byte_limit, error
    ):
        run = {
            "id": 10,
            "name": "pull",
            "status": "completed",
            "conclusion": "success",
        }
        job = {
            "id": 11,
            "name": "linux / test (default, 1, 1, runner)",
            "run_attempt": 1,
            "started_at": "2026-07-01T00:00:00Z",
            "completed_at": "2026-07-01T02:00:00Z",
            "status": "completed",
            "conclusion": "success",
        }
        artifact = pr_tests.Artifact(
            f"test-reports-test-default_{artifact_job_id}.zip",
            "s3",
            "s3",
            artifact_job_id,
        )
        payload = make_report_zip(
            [{"file": "test_a.py", "classname": "TestA", "name": "test_a"}]
        )
        with (
            mock.patch.object(
                pr_tests, "_pr_info", return_value=("pytorch/pytorch", "sha")
            ),
            mock.patch.object(pr_tests, "_workflow_runs", return_value=[run]),
            mock.patch.object(pr_tests, "_workflow_jobs", return_value=[job]),
            mock.patch.object(pr_tests, "_s3_artifacts", return_value=[artifact]),
            mock.patch.object(pr_tests, "_github_artifacts", return_value=[]),
            mock.patch.object(pr_tests, "_download_artifact", return_value=payload),
            mock.patch.object(pr_tests, "_MAX_ARTIFACT_TOTAL_BYTES", byte_limit),
        ):
            with self.assertRaisesRegex(pr_tests.PRTestsError, error):
                pr_tests.collect_pr_tests("190437", token="token")

    def test_missing_artifact_fails_closed(self):
        run = {"id": 10, "name": "pull", "status": "completed", "conclusion": "success"}
        job_name = "linux / test-osdc (default, 1, 1, runner)"
        jobs = [
            {
                "id": job_id,
                "name": job_name,
                "run_attempt": run_attempt,
                "started_at": f"2026-07-0{run_attempt}T00:00:00Z",
                "completed_at": f"2026-07-0{run_attempt}T02:00:00Z",
                "status": "completed",
                "conclusion": "success",
            }
            for job_id, run_attempt in ((11, 1), (22, 2))
        ]
        expired = pr_tests.Artifact(
            "test-reports-runattempt1-test-default_11.zip",
            "github",
            "github",
            11,
            expired=True,
            updated_at=datetime.fromisoformat("2026-07-02T01:00:00+00:00"),
        )
        with (
            mock.patch.object(
                pr_tests, "_pr_info", return_value=("pytorch/pytorch", "sha")
            ),
            mock.patch.object(pr_tests, "_workflow_runs", return_value=[run]),
            mock.patch.object(pr_tests, "_workflow_jobs", return_value=jobs),
            mock.patch.object(pr_tests, "_s3_artifacts", return_value=[]),
            mock.patch.object(pr_tests, "_github_artifacts", return_value=[expired]),
        ):
            with self.assertRaisesRegex(
                pr_tests.PRTestsError, r"1 completed test job \(1 expired\)"
            ):
                pr_tests.collect_pr_tests("190437", token="token")

    def test_empty_artifact_is_allowed_when_other_reports_exist(self):
        run = {
            "id": 10,
            "name": "pull",
            "status": "completed",
            "conclusion": "success",
        }
        jobs = [
            {
                "id": job_id,
                "name": f"linux / test (default, {index}, 2, runner)",
                "run_attempt": 1,
                "started_at": "2026-07-01T00:00:00Z",
                "completed_at": "2026-07-01T02:00:00Z",
                "status": "completed",
                "conclusion": "success",
            }
            for index, job_id in enumerate((11, 22), 1)
        ]
        artifacts = [
            pr_tests.Artifact(
                f"test-reports-test-default_{job_id}.zip", url, "s3", job_id
            )
            for job_id, url in ((11, "empty"), (22, "nonempty"))
        ]
        payloads = {
            "empty": make_report_zip(),
            "nonempty": make_report_zip(
                [{"file": "test_a.py", "classname": "TestA", "name": "test_a"}]
            ),
        }
        with (
            mock.patch.object(
                pr_tests, "_pr_info", return_value=("pytorch/pytorch", "sha")
            ),
            mock.patch.object(pr_tests, "_workflow_runs", return_value=[run]),
            mock.patch.object(pr_tests, "_workflow_jobs", return_value=jobs),
            mock.patch.object(pr_tests, "_s3_artifacts", return_value=artifacts),
            mock.patch.object(pr_tests, "_github_artifacts") as github_artifacts,
            mock.patch.object(
                pr_tests,
                "_download_artifact",
                side_effect=lambda artifact, token: payloads[artifact.url],
            ),
        ):
            result = pr_tests.collect_pr_tests("190437", token="token")

        self.assertEqual(result.tests, {"test/test_a.py::TestA::test_a"})
        github_artifacts.assert_not_called()

    def test_no_reports_fails_closed(self):
        run = {
            "id": 10,
            "name": "pull",
            "status": "completed",
            "conclusion": "success",
        }
        job = {
            "id": 11,
            "name": "linux / test (default, 1, 1, runner)",
            "run_attempt": 1,
            "started_at": "2026-07-01T00:00:00Z",
            "completed_at": "2026-07-01T02:00:00Z",
            "status": "completed",
            "conclusion": "success",
        }
        artifact = pr_tests.Artifact("test-reports-test-default_11.zip", "s3", "s3", 11)
        with (
            mock.patch.object(
                pr_tests, "_pr_info", return_value=("pytorch/pytorch", "sha")
            ),
            mock.patch.object(pr_tests, "_workflow_runs", return_value=[run]),
            mock.patch.object(pr_tests, "_workflow_jobs", return_value=[job]),
            mock.patch.object(pr_tests, "_s3_artifacts", return_value=[artifact]),
            mock.patch.object(pr_tests, "_github_artifacts") as github_artifacts,
            mock.patch.object(
                pr_tests, "_download_artifact", return_value=make_report_zip()
            ),
        ):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "no JUnit"):
                pr_tests.collect_pr_tests("190437", token="token")

        github_artifacts.assert_not_called()

    def test_incomplete_workflow_fails_closed(self):
        run = {
            "id": 10,
            "name": "pull",
            "status": "in_progress",
            "conclusion": None,
        }
        with (
            mock.patch.object(
                pr_tests, "_pr_info", return_value=("pytorch/pytorch", "sha")
            ),
            mock.patch.object(pr_tests, "_workflow_runs", return_value=[run]),
        ):
            with self.assertRaisesRegex(pr_tests.PRTestsError, "has not completed"):
                pr_tests.collect_pr_tests("190437", token="token")

    def test_main_prints_no_partial_output_on_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                pr_tests,
                "collect_pr_tests",
                side_effect=pr_tests.PRTestsError("missing artifact"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = pr_tests.main(["190437"])

        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("missing artifact", stderr.getvalue())

    def test_main_sorts_stdout(self):
        result = pr_tests.CollectionResult(
            tests=frozenset(
                {
                    "test/test_z.py::TestZ::test_z",
                    "test/test_a.py::TestA::test_a",
                }
            ),
            workflow_runs=2,
            artifacts=3,
            reports=4,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(pr_tests, "collect_pr_tests", return_value=result),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            return_code = pr_tests.main(["190437"])

        self.assertEqual(return_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            "test/test_a.py::TestA::test_a\ntest/test_z.py::TestZ::test_z\n",
        )
        self.assertIn("Found 2 tests", stderr.getvalue())


instantiate_parametrized_tests(TestPRTests)


if __name__ == "__main__":
    run_tests()
