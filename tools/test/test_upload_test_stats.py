import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.stats import test_dashboard
from tools.stats.upload_test_stats import (
    backfill_test_jsons_while_running,
    get_tests,
    summarize_test_cases,
)
from tools.testing.target_determination import gen_artifact


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


class TestDashboardExclusions(unittest.TestCase):
    def exclusions(
        self, artifacts: dict[str, dict], jobs: dict[str, str]
    ) -> tuple[dict, list[str]]:
        """get_td_exclusions over `artifacts`, and the artifacts whose jobs it looked up."""
        looked_up: list[str] = []

        def unzip(_: object) -> None:
            for name, content in artifacts.items():
                Path(name).write_text(json.dumps(content))

        def get_job_name(path: Path) -> str:
            looked_up.append(path.name)
            return jobs.get(path.name, "linux / test (default, 1, 2, x)")

        cwd = os.getcwd()
        try:
            with mock.patch.multiple(
                test_dashboard,
                download_s3_artifacts=mock.Mock(return_value=["x"]),
                unzip=unzip,
                get_job_id=lambda path: path,
                get_job_name=get_job_name,
            ):
                exclusions = test_dashboard.get_td_exclusions(1, 1)
        finally:
            os.chdir(cwd)
        return json.loads(json.dumps(exclusions)), sorted(looked_up)

    def test_exclusions_are_grouped_by_source(self) -> None:
        llm = gen_artifact.LLM_SELECTION_SOURCE
        with tempfile.TemporaryDirectory() as d:
            reports = Path(d) / "test" / "test-reports"
            reports.mkdir(parents=True)
            with mock.patch.object(gen_artifact, "REPO_ROOT", Path(d)):
                excluded = [{"test_file": "c"}]
                td_excluded = [{"test_file": "d"}]
                gen_artifact.gen_ci_artifact(
                    [], excluded, source=llm, td_excluded=td_excluded
                )
            (written,) = reports.glob("td_exclusions-*.json")
            llm_artifact = json.loads(written.read_text())

        b = [{"test_file": "b"}]
        artifacts = {
            "td_exclusions-1.json": {"included": [], "excluded": [{"test_file": "a"}]},
            "td_exclusions-2.json": {
                "included": [{"test_file": "c"}],
                "excluded": b,
                "source": "td",
            },
            "td_exclusions-3.json": llm_artifact,
            # An enforced call that excluded nothing.
            "td_exclusions-4.json": {**llm_artifact, "excluded": [], "td_excluded": []},
            "td_exclusions-5.json": {"included": [], "excluded": [], "source": "td"},
            # A call that excluded nothing but ran a file another call excluded.
            "td_exclusions-6.json": {
                "included": [{"test_file": "a"}],
                "excluded": [],
                "source": "td",
            },
            # One that ran only files no enforced config excluded.
            "td_exclusions-7.json": {
                "included": [{"test_file": "z"}],
                "excluded": [],
                "source": "td",
            },
            # One that ran b in another config of the job, which does not cover default's b.
            "td_exclusions-8.json": {
                "included": [{"test_file": "b"}],
                "excluded": [],
                "source": "td",
            },
            # One that ran b in default of another build, which does not cover linux's b either.
            "td_exclusions-9.json": {
                "included": [{"test_file": "b"}],
                "excluded": [],
                "source": "td",
            },
        }
        jobs = {
            "td_exclusions-4.json": "linux / test (slow, 1, 1, x)",
            "td_exclusions-5.json": "linux / test (crossref, 1, 1, x)",
            "td_exclusions-8.json": "linux / test (crossref, 1, 1, x)",
            "td_exclusions-9.json": "win / test (default, 1, 1, x)",
        }
        exclusions, looked_up = self.exclusions(artifacts, jobs)
        self.assertEqual(
            exclusions,
            {
                "td": {"linux": {"default": ["a", "b", "d"]}},
                # What a config with an enforced call excluded and none of its calls ran.
                "llm_selection": {"linux": {"default": ["b"], "slow": []}},
            },
        )
        # A call that excluded nothing is looked up only if it ran a file an enforced config excluded.
        expected = [f"td_exclusions-{i}.json" for i in (1, 2, 3, 4, 6, 8, 9)]
        self.assertEqual(looked_up, expected)
        # Without an enforced call, the calls that excluded nothing are not
        # looked up, and an artifact without a source is TD's.
        td_only = {
            k: artifacts[k] for k in (f"td_exclusions-{i}.json" for i in (1, 2, 5, 6))
        }
        exclusions, looked_up = self.exclusions(td_only, jobs)
        self.assertEqual(exclusions, {"td": {"linux": {"default": ["a", "b"]}}})
        self.assertEqual(looked_up, ["td_exclusions-1.json", "td_exclusions-2.json"])

    def test_a_file_split_by_class_ran_only_if_it_ran_whole(self) -> None:
        llm = gen_artifact.LLM_SELECTION_SOURCE
        a_in, a_out = (
            {"test_file": "a", "included": ["A"]},
            {"test_file": "a", "excluded": ["A"]},
        )
        b_in, b_out = (
            {"test_file": "b", "included": ["B"]},
            {"test_file": "b", "excluded": ["B"]},
        )
        artifacts = {
            "td_exclusions-1.json": {
                "included": [a_in, b_in],
                "excluded": [a_out, b_out],
                "source": llm,
            },
            "td_exclusions-2.json": {
                "included": [{"test_file": "b"}],
                "excluded": [],
                "source": "td",
            },
        }
        exclusions, _ = self.exclusions(artifacts, {})
        self.assertEqual(exclusions["llm_selection"], {"linux": {"default": ["a"]}})

    def test_both_sources_are_uploaded(self) -> None:
        td, llm = {"linux": {"default": ["a"]}}, {"linux": {"default": ["c"]}}
        for exclusions, expected in (
            ({}, [mock.call(1, 2, "additional_info/td_exclusions", [{}])]),
            ({"td": td}, [mock.call(1, 2, "additional_info/td_exclusions", [td])]),
            (
                {"td": td, "llm_selection": llm},
                [
                    mock.call(1, 2, "additional_info/td_exclusions", [td]),
                    mock.call(1, 2, "additional_info/llm_selection_exclusions", [llm]),
                ],
            ),
        ):
            with (
                mock.patch.object(
                    test_dashboard, "get_td_exclusions", return_value=exclusions
                ),
                mock.patch.object(
                    test_dashboard, "upload_workflow_stats_to_s3"
                ) as upload,
            ):
                test_dashboard.upload_additional_info(1, 2)
            self.assertEqual(upload.call_args_list, expected)


if __name__ == "__main__":
    unittest.main()
