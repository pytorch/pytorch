import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from detect_log_failures import (
    _load_expected_shard_totals,
    _shard_total_key,
    classify_log_file,
)


def _load_download_testlogs():
    for var in ("GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        os.environ.setdefault(var, "test")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_testlogs")
    loader = importlib.machinery.SourceFileLoader("download_testlogs", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


dtl = _load_download_testlogs()

THROTTLE_ERROR_DOC = (
    b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?><Error><Code>ServerBusy'
    b"</Code><Message>Egress is over the account limit.\n</Message></Error>"
)


class Response:
    headers = {}

    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class DownloadArtifactsWorkingDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp.name)
        Path("artifacts").mkdir()

    @mock.patch.object(dtl, "_report_missing_shards")
    @mock.patch.object(dtl, "download_xml_files", return_value=[])
    def test_restores_working_directory_after_download(self, _download, _report):
        original_cwd = os.getcwd()

        dtl.download_artifacts({"id": 123}, test_folder="artifacts")

        self.assertEqual(os.getcwd(), original_cwd)

    @mock.patch.object(
        dtl, "download_xml_files", side_effect=RuntimeError("download failed")
    )
    def test_restores_working_directory_after_error(self, _download):
        original_cwd = os.getcwd()

        with self.assertRaisesRegex(RuntimeError, "download failed"):
            dtl.download_artifacts({"id": 123}, test_folder="artifacts")

        self.assertEqual(os.getcwd(), original_cwd)


class CorruptArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp.name)
        dtl.error_msgs.clear()
        self.addCleanup(dtl.error_msgs.clear)

    def _stub_downloads(self, names):
        for name, payload in names.items():
            if payload is None:
                with zipfile.ZipFile(name, "w") as archive:
                    archive.writestr("test-reports/TEST-good.xml", "<testsuite/>")
            else:
                Path(name).write_bytes(payload)

        def fake_s3(prefix, run_id, attempt, allowed_substrings=None):
            return [Path(name) for name in names if name.startswith(prefix)]

        patcher = mock.patch.object(dtl, "download_s3_artifacts", fake_s3)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_four_gpu_directory_is_normalized_as_distributed(self):
        source = Path(
            "unzipped-test-reports-test-distributed_4gpu-2-2-"
            "amd-dpx-linux.rocm.gpu.gfx950.4_110840610613"
        )
        source.mkdir()

        job_ids, shards = dtl._shorten_unzipped_dirs()

        self.assertEqual(job_ids, ["110840610613"])
        self.assertEqual(shards, [("distributed_4gpu", 2, 2)])
        self.assertTrue(Path("test-distributed-2-2_110840610613").is_dir())

    def test_non_zip_artifact_is_skipped_not_fatal(self):
        self._stub_downloads(
            {
                "test-reports-test-default-1-2-rocm.gpu_1.zip": None,
                "test-reports-test-default-2-2-rocm.gpu_2.zip": THROTTLE_ERROR_DOC,
            }
        )

        dtl.download_xml_files(123, 1, prefixes=["test-reports-test-default"])

        self.assertEqual(len(list(Path(".").glob("**/TEST-good.xml"))), 1)
        joined = "\n".join(dtl.error_msgs)
        self.assertIn("test-reports-test-default-2-2-rocm.gpu_2.zip", joined)
        self.assertIn("is not a zip file", joined)
        self.assertIn("Egress is over the account limit", joined)

    def test_all_artifacts_corrupt_records_every_shard(self):
        self._stub_downloads(
            {
                "test-reports-test-default-1-2-rocm.gpu_1.zip": THROTTLE_ERROR_DOC,
                "test-reports-test-default-2-2-rocm.gpu_2.zip": THROTTLE_ERROR_DOC,
            }
        )

        dtl.download_xml_files(123, 1, prefixes=["test-reports-test-default"])

        self.assertEqual(len(dtl.error_msgs), 2)

    def test_github_fallback_fills_each_missing_s3_prefix(self):
        s3_path = Path("test-reports-test-default-1-2-rocm.gpu_1.zip")
        gha_path = Path("test-reports-test-default-2-2-rocm.gpu_2.zip")
        for path in (s3_path, gha_path):
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    f"test-reports/{path.stem}.xml", "<testsuite/>"
                )

        def fake_s3(prefix, run_id, attempt, allowed_substrings=None):
            return [s3_path] if prefix.endswith("1-2") else []

        with (
            mock.patch.object(dtl, "download_s3_artifacts", side_effect=fake_s3),
            mock.patch.object(
                dtl,
                "download_gha_artifacts_filtered",
                return_value=[gha_path],
            ) as gha,
        ):
            dtl.download_xml_files(
                123,
                1,
                prefixes=[
                    "test-reports-test-default-1-2",
                    "test-reports-test-default-2-2",
                ],
            )

        self.assertEqual(len(list(Path(".").glob("**/*.xml"))), 2)
        self.assertEqual(
            gha.call_args.kwargs["prefixes"],
            ["test-reports-test-default-2-2"],
        )


class ApiRetryTest(unittest.TestCase):
    def setUp(self):
        dtl.authentication_headers = {}

    def test_cuda_artifact_allowlist_includes_distributed_jobs(self):
        prefix = "linux-jammy-cuda13.2-py3.11-gcc11"
        jobs = [
            {"id": 1, "name": f"{prefix} / test (default, 1, 14, runner)"},
            {"id": 2, "name": f"{prefix} / test (distributed, 1, 10, runner)"},
            {"id": 3, "name": f"{prefix}-debug / test (default, 1, 7, runner)"},
        ]

        job_ids = dtl.get_cuda_artifact_job_ids(
            jobs, prefix, ("default", "distributed")
        )

        self.assertEqual(job_ids, {"1", "2"})

    def test_jobs_retry_connection_drop_and_secondary_limit(self):
        responses = [
            dtl.requests.exceptions.ConnectionError(),
            Response(403, {}),
            Response(200, {"jobs": [], "total_count": 0}),
        ]
        with (
            mock.patch.object(dtl.requests, "get", side_effect=responses) as get,
            mock.patch.object(dtl.time, "sleep") as sleep,
        ):
            result = dtl._fetch_jobs_page("jobs", {}, max_retries=3)

        self.assertEqual(result["jobs"], [])
        self.assertEqual(get.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_workflow_jobs_use_small_pages(self):
        pages = [
            {"jobs": [{"id": i, "name": "a"} for i in range(50)], "total_count": 120},
            {"jobs": [{"id": i, "name": "b"} for i in range(50)], "total_count": 120},
            {"jobs": [{"id": i, "name": "c"} for i in range(20)], "total_count": 120},
        ]
        with mock.patch.object(dtl, "_fetch_jobs_page", side_effect=pages) as fetch:
            jobs = dtl.get_workflow_jobs({"jobs_url": "jobs"}, all_attempts=True)

        self.assertEqual(len(jobs), 120)
        self.assertEqual(
            [call.args[1] for call in fetch.call_args_list],
            [
                {"per_page": 50, "filter": "all"},
                {"per_page": 50, "filter": "all", "page": 2},
                {"per_page": 50, "filter": "all", "page": 3},
            ],
        )

    def test_workflow_jobs_fall_back_to_smaller_pages(self):
        pages = [
            {"jobs": [{"id": i, "name": "a"} for i in range(50)], "total_count": 60},
            Exception("jobs API did not return a 'jobs' payload (last HTTP 502)"),
            {"jobs": [{"id": i, "name": "b"} for i in range(25)], "total_count": 60},
            {"jobs": [{"id": i, "name": "c"} for i in range(25)], "total_count": 60},
            {"jobs": [{"id": i, "name": "d"} for i in range(10)], "total_count": 60},
        ]
        with mock.patch.object(dtl, "_fetch_jobs_page", side_effect=pages) as fetch:
            jobs = dtl.get_workflow_jobs({"jobs_url": "jobs"})

        self.assertEqual(len(jobs), 60)
        self.assertEqual(
            [call.args[1] for call in fetch.call_args_list],
            [
                {"per_page": 50},
                {"per_page": 50, "page": 2},
                {"per_page": 25},
                {"per_page": 25, "page": 2},
                {"per_page": 25, "page": 3},
            ],
        )

    def test_workflow_runs_retry_bad_response(self):
        responses = [
            Response(503, {}),
            Response(200, {"workflow_runs": [{"id": 1}]}),
        ]
        with (
            mock.patch.object(dtl.requests, "get", side_effect=responses),
            mock.patch.object(dtl.time, "sleep") as sleep,
        ):
            result = dtl._fetch_workflow_runs("trunk", {}, max_retries=2)

        self.assertEqual(result, [{"id": 1}])
        sleep.assert_called_once_with(1)


class CrossSourceComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp.name)
        dtl.error_msgs.clear()
        self.addCleanup(dtl.error_msgs.clear)

    def _args(self, **overrides):
        values = {
            "set1_source": "trunk",
            "set1_sha": "a" * 40,
            "set2_source": "preview",
            "set2_sha": "b" * 40,
            "baseline_sha": None,
            "pr_id": None,
            "ignore_status": True,
            "artifacts_only": True,
            "include_inductor_periodic": False,
            "created": None,
            "max_pages": 10,
            "exclude_default": False,
            "exclude_distributed": True,
            "exclude_inductor": True,
            "exclude_slow": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_side_commands_select_platform_and_arch(self):
        args = self._args()
        trunk = dtl._cross_side_command("trunk", "a" * 40, args)
        preview = dtl._cross_side_command("preview", "b" * 40, args)

        self.assertIn("--no_rocm", trunk)
        self.assertEqual(trunk[trunk.index("--arch") + 1], "mi350")
        self.assertIn("--no_cuda", preview)
        self.assertEqual(preview[preview.index("--arch") + 1], "preview")

        periodic = dtl._cross_side_command(
            "trunk", "a" * 40,
            self._args(include_inductor_periodic=True),
        )
        self.assertIn("--include_inductor_periodic", periodic)

    def test_comparison_downloads_each_side_to_separate_directories(self):
        args = self._args()
        commands = []

        def fake_run(command, check, cwd):
            commands.append(command)
            sha = command[command.index("--sha1") + 1]
            xml_name = "cuda_xml" if "--no_rocm" in command else "rocm_xml"
            source = command[command.index("--arch") + 1]
            folder = Path(cwd) / f"20260930_{sha}"
            xml_dir = folder / xml_name / "shard"
            xml_dir.mkdir(parents=True, exist_ok=True)
            (xml_dir / "TEST-result.xml").write_text(
                f'<testsuite name="{source}"/>'
            )
            (xml_dir / "_wf_run_ids.json").write_text('{"1": "123"}')
            (folder / f"{xml_name[:-4]}1.txt").write_text("test log")
            return SimpleNamespace(returncode=0)

        with mock.patch.object(dtl.subprocess, "run", side_effect=fake_run):
            folder = dtl.run_cross_source_comparison(args)

        self.assertTrue((folder / "set1_xml/shard/TEST-result.xml").is_file())
        self.assertTrue((folder / "set2_xml/shard/TEST-result.xml").is_file())
        self.assertFalse((folder / "comparison_sources.json").exists())
        self.assertTrue(all("--exclude_slow" in command for command in commands))
        self.assertTrue((folder / f"trunk@{'a' * 8}_cuda1.txt").is_file())
        self.assertTrue((folder / f"preview@{'b' * 8}_rocm1.txt").is_file())

    def test_same_sha_rocm_sources_remain_isolated(self):
        sha = "c" * 40
        args = self._args(
            set1_source="mi350", set1_sha=sha,
            set2_source="preview", set2_sha=sha,
        )

        def fake_run(command, check, cwd):
            source = command[command.index("--arch") + 1]
            xml_dir = Path(cwd) / f"20260930_{sha}" / "rocm_xml"
            xml_dir.mkdir(parents=True, exist_ok=True)
            (xml_dir / "TEST-result.xml").write_text(source)
            return SimpleNamespace(returncode=0)

        with mock.patch.object(dtl.subprocess, "run", side_effect=fake_run):
            folder = dtl.run_cross_source_comparison(args)

        self.assertEqual((folder / "set1_xml/TEST-result.xml").read_text(), "mi350")
        self.assertEqual((folder / "set2_xml/TEST-result.xml").read_text(), "preview")

    def test_comparison_requires_all_four_source_fields(self):
        with self.assertRaisesRegex(ValueError, "requires set1_source"):
            dtl.run_cross_source_comparison(self._args(set2_sha=None))

    def test_missing_side_xml_is_recorded_as_failure(self):
        with mock.patch.object(
            dtl.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ):
            dtl.run_cross_source_comparison(self._args())

        self.assertIn(
            f"No trunk XML reports found for {'a' * 40}",
            dtl.error_msgs,
        )
        self.assertIn(
            f"No preview XML reports found for {'b' * 40}",
            dtl.error_msgs,
        )

    def test_legacy_args_do_not_enable_cross_source_mode(self):
        with mock.patch.object(sys, "argv", ["download_testlogs"]):
            args = dtl.parse_args()

        self.assertIsNone(args.set1_source)
        self.assertIsNone(args.set2_source)
        self.assertIsNone(args.set1_sha)
        self.assertIsNone(args.set2_sha)

    def test_cli_accepts_rx7900_from_shared_config(self):
        with mock.patch.object(
            sys, "argv", ["download_testlogs", "--arch", "rx7900"]
        ):
            args = dtl.parse_args()

        self.assertEqual(args.arch, "rx7900")

    def test_mi350_has_trunk_and_four_gpu_distributed_sources(self):
        config = dtl.PARITY_CONFIG["rocm"]["mi350"]

        self.assertEqual(config["distributed"][0]["workflow"], "trunk")
        self.assertEqual(
            config["distributed_4gpu"][0]["workflow"],
            "periodic-rocm-mi350",
        )
        self.assertEqual(config["shard_counts"]["distributed_4gpu"], 2)

    def test_four_gpu_log_is_classified_as_distributed(self):
        self.assertEqual(
            classify_log_file("rocm_dist4gpu2.txt"),
            ("rocm", "distributed", 2),
        )

    def test_cross_source_log_label_is_preserved(self):
        self.assertEqual(
            classify_log_file("trunk@1234abcd_cuda_dist2.txt"),
            ("trunk@1234abcd", "distributed", 2),
        )


class RunScopedCheckRunTest(unittest.TestCase):
    """trunk, trunk-rocm-sandbox and rocm-preview publish identically named
    shards on the same SHA, so anything reasoning about one run must filter on
    the run id in details_url."""

    SHA = "a" * 40
    PREFIX = "linux-noble-rocm-py3.11-mi350"

    def setUp(self):
        dtl.authentication_headers = {}
        self.trunk = {"id": 37000844958, "head_sha": self.SHA, "run_attempt": 1}
        # trunk ran 3 distributed shards; the sandbox run ran 8 under the same
        # names, so a commit-wide modal total would pick the sandbox layout.
        self.check_runs = [
            self._check_run(37000844958, "distributed", i, 3) for i in range(1, 4)
        ] + [
            self._check_run(37005184175, "distributed", i, 8) for i in range(1, 9)
        ]

    def _check_run(self, run_id, config, idx, total):
        return {
            "name": f"{self.PREFIX} / test ({config}, {idx}, {total}, rocm.gpu)",
            "details_url": (
                f"https://github.com/pytorch/pytorch/actions/runs/{run_id}"
                f"/job/{run_id}{idx}"
            ),
        }

    def test_sibling_workflow_check_runs_are_filtered_out(self):
        with mock.patch.object(
            dtl, "get_check_runs_for_commit", return_value=self.check_runs
        ):
            scoped = dtl.get_check_runs_for_run(self.SHA, 37000844958, self.PREFIX)

        self.assertEqual(len(scoped), 3)
        self.assertTrue(
            all("/runs/37000844958/" in cr["details_url"] for cr in scoped)
        )

    def test_shard_count_ignores_sibling_workflow_totals(self):
        # The jobs API has nothing, so the check-run path decides.
        with (
            mock.patch.object(dtl, "get_workflow_jobs", return_value=[]),
            mock.patch.object(
                dtl, "get_check_runs_for_commit", return_value=self.check_runs
            ),
        ):
            total = dtl.derive_shard_count(
                self.trunk, self.PREFIX, "distributed", 99
            )

        self.assertEqual(total, 3)

    def test_job_prefix_is_not_adopted_from_a_sibling_workflow(self):
        sandbox_only = [
            {
                "name": "linux-noble-rocm-py3.12-mi350 / test (default, 1, 8, rocm.gpu)",
                "details_url": (
                    "https://github.com/pytorch/pytorch/actions/runs/37005184175/job/1"
                ),
            }
        ]
        with (
            mock.patch.object(dtl, "get_workflow_jobs", return_value=[]),
            mock.patch.object(
                dtl, "get_check_runs_for_commit", return_value=sandbox_only
            ),
        ):
            resolved = dtl.resolve_job_prefix(self.trunk, "default", self.PREFIX)

        self.assertEqual(resolved, self.PREFIX)


class ShardCompletenessTest(unittest.TestCase):
    def setUp(self):
        dtl.error_msgs.clear()
        self.addCleanup(dtl.error_msgs.clear)

    def test_missing_shard_is_named_in_error_msgs(self):
        dtl._report_missing_shards(
            "rocm",
            37000844958,
            [f"test-reports-test-distributed-{i}-3" for i in (1, 2, 3)],
            [("distributed", 1, 3), ("distributed", 3, 3)],
        )

        self.assertEqual(len(dtl.error_msgs), 1)
        self.assertIn("rocm distributed shards missing", dtl.error_msgs[0])
        self.assertIn("37000844958", dtl.error_msgs[0])
        self.assertIn("2 of 3", dtl.error_msgs[0])

    def test_complete_download_is_silent(self):
        dtl._report_missing_shards(
            "cuda",
            1,
            [
                "test-reports-test-osdc-default-1-2",
                "test-reports-test-osdc-default-2-2",
            ],
            [("default", 1, 2), ("default", 2, 2)],
        )

        self.assertEqual(dtl.error_msgs, [])

    def test_four_gpu_shards_are_tracked_under_their_own_config(self):
        dtl._report_missing_shards(
            "rocm",
            37005713395,
            [f"test-reports-test-distributed_4gpu-{i}-2" for i in (1, 2)],
            [("distributed_4gpu", 1, 2)],
        )

        self.assertEqual(len(dtl.error_msgs), 1)
        self.assertIn("distributed_4gpu shards missing", dtl.error_msgs[0])
        self.assertIn("2 of 2", dtl.error_msgs[0])


class LogShardTotalsTest(unittest.TestCase):
    """download_testlogs records the expected totals; detect_log_failures must
    key into them with the same scheme so a missing log cannot shrink the
    reported "shard N/M" denominator."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_expected_totals_are_recorded_per_log_family(self):
        folder = self.tmp.name
        dtl._record_log_shard_totals(
            folder, [[f"rocm{i}.txt", "key"] for i in range(1, 9)]
        )
        dtl._record_log_shard_totals(
            folder, [[f"rocm_dist4gpu{i}.txt", "key"] for i in range(1, 3)]
        )

        self.assertEqual(
            _load_expected_shard_totals(folder),
            {"rocm": 8, "rocm_dist4gpu": 2},
        )

    def test_log_filenames_key_into_the_recorded_totals(self):
        for name, key in (
            ("rocm3.txt", "rocm"),
            ("rocm_dist4gpu2.txt", "rocm_dist4gpu"),
            ("cuda_dist10.txt", "cuda_dist"),
        ):
            self.assertEqual(_shard_total_key(name), key)


if __name__ == "__main__":
    unittest.main()
