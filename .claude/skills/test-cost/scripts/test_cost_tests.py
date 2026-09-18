import contextlib
import io
import json
import os
import shlex
import tempfile
import unittest
from datetime import date
from html import escape
from pathlib import Path
from unittest.mock import patch

import test_cost

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


CLICKHOUSE = os.environ.get("TEST_COST_CLICKHOUSE") == "1"
SQL_CASES = [
    "snapshots",
    "same_job",
    "different_jobs",
    "zero_time",
    "mixed_zero",
    "paths",
    "cancelled",
    "timed_out",
    "failure",
    "all_cancelled",
]


@instantiate_parametrized_tests
class TestCost(TestCase):
    @parametrize("filename", ["report.html", "report with spaces.html"])
    def test_reproduce_command(self, filename):
        job_columns = ["runner", "workflow", "trigger", "conclusion", "jobs", "wall_s"]
        file_columns = [
            "runner",
            "workflow",
            "trigger",
            "invoking_file",
            "test_s",
            "attr_s",
            "test_rows",
            "jobs",
            "invoking_files",
        ]
        job = ["linux.2xlarge", "pull", "pr", "success", 1]
        file = ["linux.2xlarge", "pull", "pr", "test_ops"]
        old_jobs = [job + [3600]]
        old_files = [file + [3600, 3600, 1, 1, ["test_ops"]]]
        fresh_jobs = [job + [7200]]
        fresh_files = [file + [7200, 7200, 1, 1, ["test_ops"]]]
        queries = [(job_columns, fresh_jobs), (file_columns, fresh_files)] * 2
        scratch = test_cost.REPO_ROOT / "agent_space"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            out_dir = Path(tmp)
            out = out_dir / filename
            with (
                patch.object(test_cost, "OUT_DIR", out_dir),
                patch.object(test_cost, "git_checkout", return_value="fixture"),
                patch.object(test_cost, "run_query", side_effect=queries) as query,
                patch.object(test_cost, "render", wraps=test_cost.render) as render,
                patch.object(test_cost, "upload") as upload,
                patch.object(test_cost, "print_summary"),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                cache = test_cost.cache_path(date(2026, 9, 1))
                cache.parent.mkdir(parents=True)
                cache.write_text(
                    json.dumps(
                        {
                            "query_hash": test_cost.QUERY_HASH,
                            "jobs_columns": job_columns,
                            "jobs": old_jobs,
                            "files_columns": file_columns,
                            "files": old_files,
                        }
                    )
                )
                cached = cache.read_text()
                args = [
                    "--days",
                    "1",
                    "--end",
                    "2026-09-02",
                    "--verify-sample",
                    "0",
                    "--no-cache",
                    "--no-upload",
                    "--out",
                    str(out),
                ]
                self.assertEqual(test_cost.main(args), 0)
                first_report = render.call_args.args[0]
                initial_html = out.read_text()
                replay = shlex.split(first_report.meta.command)[2:]
                self.assertEqual(test_cost.main(replay), 0)
                second_report = render.call_args.args[0]
                self.assertEqual(first_report.wall_h, 2.0)
                self.assertEqual(second_report.wall_h, 2.0)
                self.assertEqual(query.call_count, 4)
                upload.assert_not_called()
                self.assertEqual(test_cost.parse_args(replay).out, out)
                self.assertEqual(out.read_text(), initial_html)
                self.assertEqual(cache.read_text(), cached)

    @unittest.skipUnless(CLICKHOUSE, "requires CI ClickHouse access")
    @parametrize("case", SQL_CASES)
    def test_files_sql(self, case):
        ninja = "test_cpp_extensions_aot_ninja"
        no_ninja = "test_cpp_extensions_aot_no_ninja"
        canonical = "test_cpp_extensions_aot"
        aliases = [ninja, no_ninja]
        if case == "snapshots":
            rows = [
                (1, "test_a", "a", 1, 2),
                (1, "test_a", "a", 1, 2),
                (1, "test_a", "a", 2, 3),
                (1, "test_a", "a", 2, 3),
                (1, "test_a", "a", 3, 3),
                (1, "test_a", "a", 3, 3),
                (1, "test_a", "other", 1, 2),
                (1, "test_b", "b", 1, 2),
            ]
            expected = [
                ("test_a", 8, 2880, 3, 1, ["test_a"]),
                ("test_b", 2, 720, 1, 1, ["test_b"]),
            ]
        elif case in ("same_job", "different_jobs"):
            second_job = 1 if case == "same_job" else 2
            rows = [(1, ninja, "n", 1, 1), (second_job, no_ninja, "nn", 1, 1)]
            expected = [(canonical, 2, 3600 * second_job, 2, second_job, aliases)]
        elif case == "zero_time":
            rows = [
                (1, "test_a", "a", 1, 0),
                (1, "test_a", "a", 1, 0),
                (1, "test_a", "a", 2, 0),
                (1, "test_a", "a", 2, 0),
                (1, "test_b", "b", 1, 0),
            ]
            expected = [
                ("test_a", 0, 2400, 2, 1, ["test_a"]),
                ("test_b", 0, 1200, 1, 1, ["test_b"]),
            ]
        elif case == "mixed_zero":
            rows = [
                (1, "test_a", "a", 1, 0),
                (1, "test_b", "b", 1, 0),
                (2, "test_a", "a", 1, 10),
            ]
            expected = [
                ("test_a", 10, 5400, 2, 2, ["test_a"]),
                ("test_b", 0, 1800, 1, 1, ["test_b"]),
            ]
        elif case in ("cancelled", "timed_out", "failure", "all_cancelled"):
            rows = [(1, "test_a", "a", 1, 0.069), (1, "test_b", "b", 1, 2.104)]
            expected = []
            if case != "all_cancelled":
                rows.append((2, "test_a", "a", 1, 10))
                expected = [("test_a", 10, 3600, 1, 1, ["test_a"])]
        else:
            names = [
                "dynamo.test_deviceguard",
                "dynamo/test_deviceguard",
                ".__w.pytorch.pytorch.test.dynamo.test_deviceguard",
            ]
            rows = [(1, name, name, 1, 1) for name in names]
            expected = [(names[0], 3, 3600, 3, 1, sorted(names))]
        jobs = sorted({row[0] for row in rows})
        unsuccessful = case in ("cancelled", "timed_out", "failure", "all_cancelled")
        conclusion = "cancelled" if case == "all_cancelled" else case
        job_values = ", ".join(
            repr((job, job * 10, conclusion if unsuccessful and job == 1 else "success"))
            for job in jobs
        )
        values = ", ".join(
            repr(
                (
                    job,
                    job * 10,
                    name,
                    ("bucket", source),
                    f"2026-09-01 00:00:00.{snapshot:09}",
                    seconds,
                )
            )
            for job, name, source, snapshot, seconds in rows
        )
        fixture = f"""
WITH jobs AS (
  SELECT * FROM VALUES('id UInt64, run_id UInt64, conclusion String', {job_values})
),
labeled AS (
  SELECT id, conclusion, 'linux.2xlarge' AS runner, 'pull' AS workflow, 'pr' AS trigger,
         3600 AS wall_s FROM jobs
),
test_runs AS (
  SELECT * FROM VALUES(
    'job_id UInt64, workflow_id UInt64, invoking_file String, meta Tuple(String, String),
     time_inserted DateTime64(9), time Float64', {values})
)"""
        sql = fixture + test_cost.FILES_SQL.removeprefix(test_cost.BASE_CTES)
        sql = sql.replace("tests.all_test_runs", "test_runs").format(
            t0="2026-09-01 00:00:00", t1="2026-09-02 00:00:00"
        )
        columns, results = test_cost.run_query(sql, f"synthetic {case}")
        fields = [
            "invoking_file",
            "test_s",
            "attr_s",
            "test_rows",
            "jobs",
            "invoking_files",
        ]
        positions = [columns.index(field) for field in fields]
        actual = [tuple(row[i] for i in positions) for row in results]
        self.assertEqual(actual, expected)
        if unsuccessful:
            sql = fixture + test_cost.JOBS_SQL.removeprefix(test_cost.BASE_CTES)
            columns, results = test_cost.run_query(sql, f"synthetic {case} jobs")
            job_rows = test_cost.to_rows(test_cost.JobRow, columns, results)
            self.assertEqual(sum(row.jobs for row in job_rows), len(jobs))
            self.assertEqual(sum(row.wall_s for row in job_rows), 3600 * len(jobs))

    @unittest.skipUnless(CLICKHOUSE, "requires CI ClickHouse access")
    def test_job_selection(self):
        rows = [
            (1, "linux / test (default, 1, 1)", "success"),
            (2, "linux / test-osdc (default, 1, 1)", "success"),
            (3, "linux / test-osdc (default, 2, 2)", "cancelled"),
            (4, "linux / build", "success"),
            (5, "linux / test-helper (default, 1, 1)", "success"),
        ]
        values = ", ".join(repr(row) for row in rows)
        fixture = f"""(
  SELECT id, 10 AS run_id, name, conclusion, 'trunk' AS workflow_name,
         'main' AS head_branch, ['self-hosted', 'linux.2xlarge'] AS labels,
         toDateTime('2026-09-01 00:00:00', 'UTC') AS started_at,
         toDateTime('2026-09-01 01:00:00', 'UTC') AS completed_at,
         'pytorch/pytorch' AS repository_full_name, 'completed' AS status,
         'runner' AS runner_name
  FROM VALUES('id UInt64, name String, conclusion String', {values})
)"""
        sql = test_cost.SAMPLE_SQL.replace("default.workflow_job", fixture).format(
            t0="2026-09-01 00:00:00", t1="2026-09-02 00:00:00", n=20
        )
        columns, results = test_cost.run_query(sql, "synthetic job selection")
        self.assertEqual(sorted(row[columns.index("id")] for row in results), [1, 2, 3])

    @parametrize("warning", ["2026-09-01: no test jobs", "unknown hardware: <gpu>&"])
    def test_warning_html(self, warning):
        day = date(2026, 9, 1)
        window = test_cost.Window(day, date(2026, 9, 3))
        meta = test_cost.Meta(window, "test", "test", "test")
        runner = "linux.2xlarge"
        name = "test_ops"
        job = test_cost.JobRow(runner, "pull", "pr", "success", 1, 3600)
        file = test_cost.FileRow(runner, "pull", "pr", name, 1, 3600, 1, 1, [name])
        days = [
            test_cost.DayData(day, [], [], settled=True, cached=False),
            test_cost.DayData(
                date(2026, 9, 2),
                [job],
                [file],
                settled=True,
                cached=False,
            ),
        ]
        report = test_cost.aggregate(days, test_cost.REPO_ROOT, meta)
        html = test_cost.render(report)
        self.assertEqual(report.coverage, 1.0)
        self.assertIn(report.warnings[0], html)
        self.assertLess(html.index(report.warnings[0]), html.index('id="charts"'))
        report.warnings = [warning]
        html = test_cost.render(report)
        self.assertIn(escape(warning), html)
        report.warnings = []
        self.assertNotIn('<aside class="warnings"', test_cost.render(report))


if __name__ == "__main__":
    run_tests()
