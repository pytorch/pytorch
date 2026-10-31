from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.check_disabled_tests import process_report, save_results


sys.path.remove(str(REPO_ROOT))

CODE_SKIP = "known failure: https://github.com/pytorch/pytorch/issues/1"
CODE_SKIP_PROPERTY = (
    f'<properties><property name="code_skip" value="{CODE_SKIP}"/></properties>'
)
ENABLED_SKIP = (
    "Test is enabled but --rerun-disabled-tests verification mode is set, "
    "so only disabled tests are run"
)
REPORT = f"""<testsuites><testsuite>
<testcase classname="TestFoo" name="test_disabled" file="test_foo.py"/>
<testcase classname="TestFoo" name="test_disabled" file="test_foo.py"><failure message="boom"/></testcase>
<testcase classname="TestFoo" name="test_known" file="test_foo.py">{CODE_SKIP_PROPERTY}</testcase>
<testcase classname="TestFoo" name="test_known" file="test_foo.py">{CODE_SKIP_PROPERTY}</testcase>
<testcase classname="TestFoo" name="test_known_fails" file="test_foo.py">{CODE_SKIP_PROPERTY}<failure message="boom"/></testcase>
<testcase classname="TestFoo" name="test_known_errors" file="test_foo.py">{CODE_SKIP_PROPERTY}<error message="boom"/></testcase>
<testcase classname="TestFoo" name="test_known_still_skipped" file="test_foo.py">{CODE_SKIP_PROPERTY}<skipped message="requires CUDA"/></testcase>
<testcase classname="TestFoo" name="test_enabled" file="test_foo.py"><skipped message="{ENABLED_SKIP}"/></testcase>
</testsuite></testsuites>
"""


class TestCheckDisabledTests(unittest.TestCase):
    def test_process_report_separates_code_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.xml"
            report.write_text(REPORT)
            tests, code_skips = process_report(report)

        self.assertEqual(
            tests, {"test_disabled;TestFoo;test_foo.py": {"num_green": 1, "num_red": 1}}
        )
        self.assertEqual(
            code_skips,
            {
                "test_known;TestFoo;test_foo.py": {
                    "num_green": 2,
                    "num_red": 0,
                    "code_skip": CODE_SKIP,
                },
                "test_known_fails;TestFoo;test_foo.py": {
                    "num_green": 0,
                    "num_red": 1,
                    "code_skip": CODE_SKIP,
                },
                "test_known_errors;TestFoo;test_foo.py": {
                    "num_green": 0,
                    "num_red": 1,
                    "code_skip": CODE_SKIP,
                },
            },
        )

    @mock.patch("tools.stats.check_disabled_tests.upload_workflow_stats_to_s3")
    def test_save_results_uploads_code_skips_separately(
        self, upload: mock.MagicMock
    ) -> None:
        tests = {"test_disabled;TestFoo;test_foo.py": {"num_green": 50, "num_red": 0}}
        code_skips = {
            "test_known;TestFoo;test_foo.py": {
                "num_green": 50,
                "num_red": 0,
                "code_skip": CODE_SKIP,
            }
        }
        save_results(1, 2, tests, code_skips)

        identity = {
            "workflow_id": 1,
            "workflow_run_attempt": 2,
            "classname": "TestFoo",
            "filename": "test_foo.py",
        }
        self.assertEqual(
            upload.call_args_list,
            [
                mock.call(
                    1,
                    2,
                    "rerun_disabled_tests",
                    [
                        {
                            **identity,
                            "name": "test_disabled",
                            "flaky": False,
                            "num_green": 50,
                            "num_red": 0,
                        }
                    ],
                ),
                mock.call(
                    1,
                    2,
                    "rerun_disabled_code_skips",
                    [
                        {
                            **identity,
                            "name": "test_known",
                            "num_green": 50,
                            "num_red": 0,
                            "code_skip": CODE_SKIP,
                        }
                    ],
                ),
            ],
        )

    @mock.patch("tools.stats.check_disabled_tests.upload_workflow_stats_to_s3")
    def test_save_results_without_code_skips(self, upload: mock.MagicMock) -> None:
        save_results(1, 2, {}, {})
        self.assertEqual(
            upload.call_args_list, [mock.call(1, 2, "rerun_disabled_tests", [])]
        )


if __name__ == "__main__":
    unittest.main()
