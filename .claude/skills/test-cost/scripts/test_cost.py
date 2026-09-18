#!/usr/bin/env python3
"""Attribute pytorch/pytorch CI test-job machine-hours to test files and owners.

Job wall time and runner labels come from default.workflow_job in the PyTorch CI
ClickHouse (the archived GitHub Actions webhook payloads); per-test seconds come
from tests.all_test_runs. Each job's wall time is split across the files it ran
in proportion to their test seconds, then rolled up by hardware class and by the
"# Owner(s)" header of each file. The result is one self-contained HTML page, a
stdout summary and, when the px CLI is installed, a Pixelcloud link.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from cost_data import (
    aggregate,
    DayData,
    FileRow,
    hardware_class,
    HW_CLASSES,
    HW_FAMILY,
    HW_PATTERNS,
    INVOKING_ALIASES,
    invoking_file_to_path,
    JobRow,
    LEAKED_PREFIX,
    Meta,
    ratio,
    read_owner,
    Report,
    VerifyRow,
    Window,
)
from cost_html import render, SECTION_IDS


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
WRAPPER = REPO_ROOT / ".claude/skills/ci-metrics/gcx-wrapper.sh"
OUT_DIR = REPO_ROOT / "agent_space/test-cost"
# Test rows can land up to ~2 days after the job; younger days are refetched.
SETTLE_DAYS = 3
QUERY_TIMEOUT_S = 900
SQL_ERROR_MARKERS = ("invalid clickhouse query", "db::exception", "syntax error")

# The read-only Grafana user cannot call dictGet, so the workflow_event ALIAS on
# workflow_job is unusable; the event comes from workflow_run instead. all_test_runs
# is sorted by workflow_id, so its filter must lead with the run ids.
BASE_CTES = """\
WITH jobs AS (
  SELECT id, run_id, workflow_name, head_branch, conclusion,
         arrayFirst(l -> l != 'self-hosted', labels) AS runner,
         dateDiff('second', started_at, completed_at) AS wall_s
  FROM default.workflow_job
  WHERE completed_at >= toDateTime('{t0}', 'UTC') AND completed_at < toDateTime('{t1}', 'UTC')
    AND started_at <= completed_at AND started_at >= toDateTime('{t0}', 'UTC') - INTERVAL 7 DAY
    AND repository_full_name = 'pytorch/pytorch' AND name LIKE '% / test (%'
    AND status = 'completed' AND runner_name != ''
  LIMIT 1 BY id
),
runs AS (
  SELECT id, event FROM default.workflow_run
  WHERE id IN (SELECT run_id FROM jobs)
    AND created_at >= toDateTime('{t0}', 'UTC') - INTERVAL 30 DAY
    AND tupleElement(repository, 'full_name') = 'pytorch/pytorch'
  LIMIT 1 BY id
),
labeled AS (
  SELECT j.id AS id, j.workflow_name AS workflow, j.conclusion AS conclusion,
         j.runner AS runner, j.wall_s AS wall_s,
         multiIf((j.head_branch = 'main' AND ifNull(r.event, '') = 'push')
                 OR (ifNull(r.event, '') = 'workflow_dispatch' AND startsWith(j.head_branch, 'trunk/')), 'main',
                 ifNull(r.event, '') = 'pull_request' OR startsWith(j.head_branch, 'ciflow/'), 'pr',
                 ifNull(r.event, '') = 'schedule', 'scheduled', 'other') AS trigger
  FROM jobs AS j LEFT JOIN runs AS r ON r.id = j.run_id
)"""

JOBS_SQL = (
    BASE_CTES
    + """
SELECT runner, workflow, trigger, conclusion, count() AS jobs, sum(wall_s) AS wall_s
FROM labeled
GROUP BY runner, workflow, trigger, conclusion
ORDER BY runner, workflow, trigger, conclusion"""
)

CANONICAL_FILE_SQL = f"""transform(
    replaceRegexpOne(replaceAll(invoking_file, '/', '.'), {("^" + re.escape(LEAKED_PREFIX))!r}, ''),
    {list(INVOKING_ALIASES)!r}, {list(INVOKING_ALIASES.values())!r})"""

FILES_SQL = (
    BASE_CTES
    + """,
snapshots AS (
  SELECT job_id, invoking_file, meta, time_inserted, sum(time) AS t, count() AS n
  FROM tests.all_test_runs
  WHERE workflow_id IN (SELECT run_id FROM jobs) AND job_id IN (SELECT id FROM jobs)
    AND time_inserted >= toDateTime('{t0}', 'UTC') - INTERVAL 1 DAY
    AND time_inserted < toDateTime('{t1}', 'UTC') + INTERVAL 2 DAY
  GROUP BY job_id, invoking_file, meta, time_inserted
),
reports AS (
  SELECT job_id, invoking_file, meta, argMax(tuple(t, n), time_inserted) AS totals
  FROM snapshots
  GROUP BY job_id, invoking_file, meta
),
ft AS (
  SELECT job_id, invoking_file, sum(totals.1) AS t, sum(totals.2) AS n,
         """
    + CANONICAL_FILE_SQL
    + """ AS canonical_file
  FROM reports
  GROUP BY job_id, invoking_file
),
jt AS (SELECT job_id, sum(t) AS t_tot, sum(n) AS n_tot FROM ft GROUP BY job_id)
SELECT l.runner AS runner, l.workflow AS workflow, l.trigger AS trigger,
       ft.canonical_file AS invoking_file,
       round(sum(ft.t), 3) AS test_s,
       round(sum(l.wall_s * if(jt.t_tot > 0, ft.t / jt.t_tot, ft.n / jt.n_tot)), 3) AS attr_s,
       sum(ft.n) AS test_rows, uniqExact(ft.job_id) AS jobs,
       arraySort(groupUniqArray(ft.invoking_file)) AS invoking_files
FROM ft
JOIN jt ON jt.job_id = ft.job_id
JOIN labeled AS l ON l.id = ft.job_id
GROUP BY runner, workflow, trigger, ft.canonical_file
ORDER BY runner, workflow, trigger, invoking_file"""
)

SAMPLE_SQL = """\
SELECT id, run_id, runner, wall_s
FROM (
  SELECT id, run_id, arrayFirst(l -> l != 'self-hosted', labels) AS runner,
         dateDiff('second', started_at, completed_at) AS wall_s
  FROM default.workflow_job
  WHERE completed_at >= toDateTime('{t0}', 'UTC') AND completed_at < toDateTime('{t1}', 'UTC')
    AND started_at <= completed_at AND started_at >= toDateTime('{t0}', 'UTC') - INTERVAL 7 DAY
    AND repository_full_name = 'pytorch/pytorch' AND name LIKE '% / test (%'
    AND status = 'completed' AND runner_name != ''
  LIMIT 1 BY id
)
ORDER BY cityHash64(id)
LIMIT {n}"""

QUERY_HASH = hashlib.sha256((JOBS_SQL + FILES_SQL).encode()).hexdigest()[:12]


class QueryError(Exception):
    pass


def run_query(sql: str, label: str) -> tuple[list[str], list[list]]:
    cmd = [
        str(WRAPPER),
        "datasources",
        "clickhouse",
        "query",
        sql,
        "--limit",
        "0",
        "-o",
        "json",
    ]
    problem = ""
    for attempt in (1, 2):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=QUERY_TIMEOUT_S
            )
        except FileNotFoundError:
            raise QueryError(f"ClickHouse wrapper not found at {WRAPPER}") from None
        except subprocess.TimeoutExpired:
            problem = f"timed out after {QUERY_TIMEOUT_S} s"
        else:
            try:
                payload = json.loads(proc.stdout)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and "columns" in payload:
                rows = payload.get("rows") or []
                return [c["name"] for c in payload["columns"]], rows
            stderr = proc.stderr.strip()
            problem = (stderr or proc.stdout.strip())[-2000:]
            if isinstance(payload, dict) and payload.get("error"):
                problem = str(payload["error"])[-2000:]
            if any(marker in problem.lower() for marker in SQL_ERROR_MARKERS):
                raise QueryError(f"ClickHouse rejected the {label} query: {problem}")
            if stderr.startswith("error:"):
                raise QueryError(f"ClickHouse wrapper failed: {stderr}")
        if attempt == 1:
            print(
                f"{label} query failed ({problem[-300:]}); retrying in 10 s",
                file=sys.stderr,
            )
            time.sleep(10)
    raise QueryError(f"{label} query failed twice: {problem}")


def to_rows(cls: type, columns: list[str], rows: list[list]) -> list:
    return [cls(**dict(zip(columns, row))) for row in rows]


def cache_path(day: date) -> Path:
    return OUT_DIR / "cache" / QUERY_HASH / f"{day.isoformat()}.json"


def load_day(day: date, today: date, no_cache: bool) -> DayData:
    settled = day <= today - timedelta(days=SETTLE_DAYS)
    use_cache = settled and not no_cache
    path = cache_path(day)
    if use_cache and path.is_file():
        try:
            cached = json.loads(path.read_text())
        except ValueError:
            cached = {}
        if cached.get("query_hash") == QUERY_HASH:
            jobs = to_rows(JobRow, cached["jobs_columns"], cached["jobs"])
            files = to_rows(FileRow, cached["files_columns"], cached["files"])
            return DayData(day, jobs, files, settled=True, cached=True)
    t0 = f"{day.isoformat()} 00:00:00"
    t1 = f"{(day + timedelta(days=1)).isoformat()} 00:00:00"
    print(
        f"fetching {day}{'' if settled else ' (unsettled, not cached)'}...",
        file=sys.stderr,
    )
    jobs_columns, jobs = run_query(JOBS_SQL.format(t0=t0, t1=t1), f"jobs {day}")
    files_columns, files = run_query(FILES_SQL.format(t0=t0, t1=t1), f"files {day}")
    jobs.sort(key=json.dumps)
    files.sort(key=json.dumps)
    if use_cache and jobs and files:
        payload = {
            "day": day.isoformat(),
            "query_hash": QUERY_HASH,
            "jobs_columns": jobs_columns,
            "jobs": jobs,
            "files_columns": files_columns,
            "files": files,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, suffix=".tmp", delete=False
        ) as tmp:
            json.dump(payload, tmp, separators=(",", ":"))
            tmp.write("\n")
        os.chmod(tmp.name, 0o644)
        os.replace(tmp.name, path)
    elif use_cache:
        print(f"warning: {day} returned no data; not cached", file=sys.stderr)
    return DayData(
        day,
        to_rows(JobRow, jobs_columns, jobs),
        to_rows(FileRow, files_columns, files),
        settled=settled,
        cached=False,
    )


def verify_sample(window: Window, n: int) -> list[VerifyRow]:
    sql = SAMPLE_SQL.format(
        t0=f"{window.start} 00:00:00", t1=f"{window.end} 00:00:00", n=n
    )
    columns, rows = run_query(sql, "verification sample")
    results = []
    for raw in rows:
        r = dict(zip(columns, raw))
        row = VerifyRow(int(r["id"]), int(r["run_id"]), r["runner"], int(r["wall_s"]))
        cmd = ["gh", "api", f"repos/pytorch/pytorch/actions/jobs/{row.job_id}"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            row.note = (proc.stderr.strip() or "gh api failed")[-200:]
        else:
            job = json.loads(proc.stdout)
            labels = [lab for lab in job.get("labels", []) if lab != "self-hosted"]
            row.gh_label = labels[0] if labels else ""
            started = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
            completed = datetime.fromisoformat(
                job["completed_at"].replace("Z", "+00:00")
            )
            row.gh_wall_s = int((completed - started).total_seconds())
            same_label = row.gh_label == row.ch_label
            same_wall = abs(row.gh_wall_s - row.ch_wall_s) <= 1
            row.status = "OK" if same_label and same_wall else "MISMATCH"
        results.append(row)
    return sorted(results, key=lambda v: v.job_id)


def git_checkout() -> str:
    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    head = git("rev-parse", "--short=12", "HEAD") or "unknown"
    # Only test/ matters for owners, so the untracked skill files do not count as dirty.
    return f"{head}-dirty" if git("status", "--porcelain", "--", "test") else head


def upload(out: Path, title: str) -> str:
    if shutil.which("px") is None:
        return "skipped (px not on PATH)"
    proc = subprocess.run(
        ["px", "upload", "--title", title, str(out)], capture_output=True, text=True
    )
    urls = [tok for tok in proc.stdout.split() if tok.startswith("https://")]
    if proc.returncode != 0 or not urls:
        lines = (
            proc.stderr.strip() or proc.stdout.strip() or "no URL printed"
        ).splitlines()
        return f"failed ({lines[0][:200]})"
    return urls[0]


def print_summary(report: Report, out: Path, upload_result: str) -> None:
    window = report.meta.window
    hits = sum(1 for d in report.days if d.cached)
    unsettled = [d.day for d in report.days if not d.settled]
    pending = (
        f" ({len(unsettled)} unsettled: {unsettled[0]}..{unsettled[-1]})"
        if unsettled
        else ""
    )
    print(
        f"window: {window.start}..{window.last} ({len(report.days)} UTC days, --end {window.end} exclusive)"
    )
    print(
        f"cache: {hits} days from cache, {len(report.days) - hits} fetched{pending}, query hash {report.meta.query_hash}"
    )
    print(
        f"test jobs: {report.total_jobs:,} | test-job hours: {report.wall_h:,.0f} | attributed: {report.attr_h:,.0f} h ({report.coverage:.1%})"
    )
    by_class = [
        f"{c} {h:,.0f} ({ratio(h, report.wall_h):.1%})"
        for c, h in report.class_wall_h.items()
        if h > 0
    ]
    print("test-job hours by class: " + " | ".join(by_class))
    print(f"accelerator share: {report.accelerator_share:.1%} of test-job hours")
    print("top owners (attributed h):")
    for i, owner in enumerate(report.owners[:10], 1):
        top_class = max(owner.by_class.items(), key=lambda kv: kv[1])[0]
        print(
            f"  {i:>2}. {owner.key:<28} {owner.hours:>10,.0f}  {ratio(owner.hours, report.attr_h):>6.1%}  mostly {top_class}"
        )
    print("top files (attributed h):")
    for i, file_agg in enumerate(report.files[:10], 1):
        print(
            f"  {i:>2}. {file_agg.key:<52} {file_agg.hours:>10,.0f}  [{file_agg.owner}]"
        )
    no_header = next((o for o in report.owners if o.key == "no-header"), None)
    unmapped_h = sum(u.attr_h for u in report.unmapped)
    no_header_text = (
        f"{len(no_header.members)} files, {no_header.hours:,.0f} h"
        if no_header
        else "0 files"
    )
    print(
        f"unmapped: {len(report.unmapped)} invoking files, {unmapped_h:,.0f} h | no-header: {no_header_text}"
    )
    if report.verify_requested:
        ok = sum(1 for v in report.verify if v.status == "OK")
        print(
            f"verify: {ok}/{len(report.verify)} sampled jobs match GitHub (runner label equal, wall seconds within 1 s)"
        )
    for warning in report.warnings:
        print(f"warning: {warning}")
    print(f"report: {out} ({out.stat().st_size / 1024:,.0f} KB)")
    print(f"uploaded: {upload_result}")


SELF_TEST_LABELS = {
    "mt-l-x86iavx512-16-128": "CPU x86",
    "mt-l-x86iavx512-8-64": "CPU x86",
    "mt-l-x86aavx2-29-113-a10g": "A10G",
    "mt-l-x86aavx2-29-113-l4": "L4",
    "mt-l-x86iamx-8-64": "CPU x86",
    "linux.rocm.gpu.gfx950.1": "ROCm",
    "mt-l-x86iavx512-45-172-t4-4": "T4",
    "windows.4xlarge.nonephemeral": "Windows",
    "mt-l-arm64g3-16-62": "CPU arm64",
    "lf-l-x86iavx512-16-128": "CPU x86",
    "mt-l-arm64g4-16-62": "CPU arm64",
    "amd-dpx-linux.rocm.gpu.gfx950.1": "ROCm",
    "macos-m1-stable": "macOS",
    "lf-l-x86aavx2-29-113-l4": "L4",
    "linux.idc.xpu": "XPU",
    "linux.rocm.gpu.gfx950.2": "ROCm",
    "mt-l-x86aavx2-45-167-a10g-4": "A10G",
    "linux.rocm.gpu.mi300.1": "ROCm",
    "lf-l-x86iavx512-8-64": "CPU x86",
    "lf-l-x86iavx512-45-172-t4-4": "T4",
    "linux.rocm.gpu.mi210.1": "ROCm",
    "amd-dpx-linux.rocm.gpu.gfx950.2": "ROCm",
    "lf-l-arm64g3-16-62": "CPU arm64",
    "mt-l-x86iavx512-48-384": "CPU x86",
    "mt-l-x86aavx2-189-704-a10g-8": "A10G",
    "lf-l-x86aavx2-29-113-a10g": "A10G",
    "macos-m2-15": "macOS",
    "lf-l-x86iamx-8-64": "CPU x86",
    "lf-l-arm64g4-16-62": "CPU arm64",
    "linux.dgx.b200": "B200",
    "linux.rocm.gpu.mi300.4": "ROCm",
    "mt-l-bx86iavx512-94-344-t4-8": "T4",
    "mt-l-x86iamx-32-128": "CPU x86",
    "mt-l-bx86iamx-92-167": "CPU x86",
    "linux.rocm.gpu.mi210.2": "ROCm",
    "mt-l-x86iavx512-11-125-a100": "A100",
    "amd-sandbox-linux.rocm.gpu.mi350.1": "ROCm",
    "mt-l-x86iamx-22-225-h100": "H100",
    "linux.s390x": "s390x",
    "lf-l-x86aavx2-45-167-a10g-4": "A10G",
    "mt-l-x86iavx2-8-32": "CPU x86",
    "linux.dgx.b200.8": "B200",
    "mt-l-x86aavx512-125-463": "CPU x86",
    "lf-l-x86aavx2-189-704-a10g-8": "A10G",
    "linux.rocm.gpu.gfx1100": "ROCm",
    "amd-sandbox-linux.rocm.gpu.mi350.2": "ROCm",
    "linux.rocm.gpu.gfx942.1": "ROCm",
    "linux.client.xpu": "XPU",
    "lf-l-bx86iavx512-94-344-t4-8": "T4",
    "macos-m1-14": "macOS",
    "mt-l-x86iavx2-40-160": "CPU x86",
    "mt-l-barm64g4-94-344": "CPU arm64",
    "linux.rocm.gpu.gfx950.1.test-control": "ROCm",
    "linux.google.tpuv7x.1": "TPU",
    "linux.rocm.gpu.rx7900.1": "ROCm",
    "macos-m2-26": "macOS",
    "mt-l-x86iamx-88-900-h100-4": "H100",
    "linux.rocm.gpu.gfx950.4": "ROCm",
    "mt-l-x86iamx-22-225-b200": "B200",
    "lf-l-x86iavx512-11-125-a100": "A100",
    "linux.rocm.gpu.gfx950.2-no-dind": "ROCm",
    "mt-l-x86aavx2-11-41-a10g": "A10G",
    "mt-l-x86iavx512-29-115-t4": "T4",
    "mt-l-bx86iamx-176-1800-h100-8": "H100",
    "lf-l-x86iavx512-29-115-t4": "T4",
    "mt-l-x86aavx2-11-41-l4": "L4",
    "macos-m1-13": "macOS",
    "mt-l-x86iamx-44-450-h100-fab-2": "H100",
    "lf-l-x86iavx512-48-384": "CPU x86",
    "linux.rocm.gpu.gfx950.1-no-dind": "ROCm",
    "linux.arm64.m7g.4xlarge": "CPU arm64",
    "linux.g5.48xlarge.nvidia.gpu": "A10G",
    "amd-dpx-linux.rocm.gpu.gfx950.4": "ROCm",
    "linux.2xlarge": "CPU x86",
    "lf-l-bx86iamx-92-167": "CPU x86",
    # Legacy and hypothetical labels that pin the ordering rules.
    "windows.g4dn.xlarge": "T4",
    "windows.8xlarge.nvidia.gpu": "unknown",
    "linux.aws.h100": "H100",
    "linux.aws.a100": "A100",
    "linux.p5.48xlarge.nvidia.gpu": "H100",
    "linux.g6.4xlarge.experimental.nvidia.gpu": "L4",
    "linux.g6e.4xlarge.experimental.nvidia.gpu": "unknown",
    "linux.hpu.gaudi": "unknown",
    "linux.arm64.gpu.grace": "unknown",
    "linux.24_04.4x": "CPU x86",
    "ubuntu-24.04": "CPU x86",
    "linux.arm64.2xlarge": "CPU arm64",
    "mt-w-x86iavx512-16-64": "Windows",
    "mt-m-arm64-8-16": "macOS",
    "lf-w-x86iavx512-8-32-t4": "T4",
    "": "unknown",
}


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, got: object, want: object) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, expected {want!r}")

    for label, want in SELF_TEST_LABELS.items():
        check(f"hardware_class({label!r})", hardware_class(label), want)
    check("HW_PATTERNS classes", all(c in HW_CLASSES for _, c in HW_PATTERNS), True)
    check("HW_FAMILY covers HW_CLASSES", set(HW_FAMILY) == set(HW_CLASSES), True)
    test_dir = REPO_ROOT / "test"
    for name, want in (
        ("dynamo.test_deviceguard", "test/dynamo/test_deviceguard.py"),
        ("test_ops", "test/test_ops.py"),
        ("functorch.test_ops", "test/functorch/test_ops.py"),
        ("distributed/test_c10d_common", "test/distributed/test_c10d_common.py"),
        (".__w.pytorch.pytorch.test.test_ops", "test/test_ops.py"),
        ("test_cpp_extensions_aot_ninja", "test/test_cpp_extensions_aot.py"),
        ("test_cpp_extensions_aot_no_ninja", "test/test_cpp_extensions_aot.py"),
        ("test_custom_backend", "test/custom_backend/test_custom_backend.py"),
        ("test_libtorch", None),
        ("dynamo.test_file_that_only_exists_on_a_pr", None),
    ):
        check(
            f"invoking_file_to_path({name!r})",
            invoking_file_to_path(name, test_dir),
            want,
        )
    for rel, want in (
        ("test/dynamo/test_deviceguard.py", "dynamo"),
        ("test/test_ops.py", "unknown"),
        ("test/test_tensorexpr.py", "NNC"),
        ("test/dynamo/test_flat_apply.py", "dynamo"),
        ("test/test_numpy_interop.py", "numpy"),
        ("test/distributed/test_c10d_common.py", "distributed"),
    ):
        check(f"read_owner({rel!r})", read_owner(REPO_ROOT / rel), want)
    with tempfile.TemporaryDirectory() as tmp:
        for text, want in (
            ("# Owner(s): ['oncall: distributed']\n", "distributed"),
            ('﻿# Owner(s):  [ "module:dynamo" ]\n', "dynamo"),
            ('#!/usr/bin/env python3\n# comment\n# Owner(s): ["module: nn"]\n', "nn"),
            ("import torch\n", "no-header"),
            ("# Owner(s): not a list\n", "no-header"),
        ):
            path = Path(tmp) / "case.py"
            path.write_text(text, encoding="utf-8")
            check(f"read_owner({text!r})", read_owner(path), want)

    day1, day2 = date(2026, 9, 1), date(2026, 9, 2)
    cpu, a10g = "mt-l-x86iavx512-16-128", "mt-l-x86aavx2-29-113-a10g"
    ops, guard, lib = "test_ops", "dynamo.test_deviceguard", "test_libtorch"
    days = [
        DayData(
            day1,
            [JobRow(cpu, "pull", "pr", "success", 2, 7200)],
            [
                FileRow(cpu, "pull", "pr", ops, 1000.0, 3600.0, 50, 2, [ops]),
                FileRow(cpu, "pull", "pr", guard, 500.0, 3600.0, 20, 1, [guard]),
            ],
            settled=True,
            cached=True,
        ),
        DayData(
            day2,
            [JobRow(a10g, "trunk", "main", "failure", 1, 3600)],
            [
                FileRow(a10g, "trunk", "main", ops, 200.0, 1800.0, 10, 1, [ops]),
                FileRow(a10g, "trunk", "main", lib, 0.0, 1800.0, 1, 1, [lib]),
            ],
            settled=False,
            cached=False,
        ),
    ]
    meta = Meta(
        Window(day1, day2 + timedelta(days=1)),
        "selftest",
        "abc123",
        "python3 test_cost.py --self-test",
    )
    report = aggregate(days, REPO_ROOT, meta)
    check("wall hours", round(report.wall_h, 6), 3.0)
    check("attributed hours", round(report.attr_h, 6), 3.0)
    check("accelerator share", round(report.accelerator_share, 6), round(1 / 3, 6))
    check(
        "owner order", [o.key for o in report.owners], ["unknown", "dynamo", "unmapped"]
    )
    check("unknown owner hours", round(report.owners[0].hours, 6), 1.5)
    check("unknown owner pr hours", round(report.owners[0].by_trigger["pr"], 6), 1.0)
    check(
        "file order",
        [f.key for f in report.files],
        ["test/test_ops.py", "test/dynamo/test_deviceguard.py"],
    )
    check(
        "test_ops by class",
        {c: round(h, 6) for c, h in report.files[0].by_class.items() if h},
        {"CPU x86": 1.0, "A10G": 0.5},
    )
    check("test_ops jobs", report.files[0].jobs, 3)
    check(
        "unmapped",
        [(u.invoking_file, round(u.attr_h, 6)) for u in report.unmapped],
        [("test_libtorch", 0.5)],
    )
    check(
        "labels",
        [(lab.label, lab.hw_class, lab.jobs) for lab in report.labels],
        [(cpu, "CPU x86", 2), (a10g, "A10G", 1)],
    )
    check(
        "days",
        [(d.day, d.jobs, d.settled) for d in report.days],
        [(day1, 2, True), (day2, 1, False)],
    )
    check("total jobs", report.total_jobs, 3)
    check(
        "trigger wall",
        {t: h for t, h in report.trigger_wall_h.items() if h},
        {"main": 1.0, "pr": 2.0},
    )
    check("warnings", report.warnings, [])
    report.verify = [VerifyRow(1, 2, cpu, 3600, cpu, 3600, "OK")]
    report.verify_requested = 1
    html = render(report)
    check("render is deterministic", render(report) == html, True)
    check(
        "render has every section",
        [sid for sid in SECTION_IDS if f'id="{sid}"' not in html],
        [],
    )
    check(
        "render has no nan/None", [tok for tok in ("nan", ">None<") if tok in html], []
    )
    check("render closes the document", "</html>" in html, True)
    hostile = "<script>alert(1)</script>"
    runner = "<b>evil</b>"
    hostile_days = [
        DayData(
            day1,
            [JobRow(runner, "wf<1>", "pr", "success", 1, 3600)],
            [
                FileRow(runner, "wf<1>", "pr", hostile, 1.0, 1800.0, 1, 1, [hostile]),
                FileRow(runner, "wf<1>", "pr", "test_nn", 0.0, 0.0, 3, 1, ["test_nn"]),
            ],
            settled=True,
            cached=True,
        )
    ]
    hostile_report = aggregate(hostile_days, REPO_ROOT, meta)
    hostile_html = render(hostile_report)
    check(
        "render escapes",
        (
            hostile in hostile_html,
            "&lt;script&gt;alert(1)&lt;/script&gt;" in hostile_html,
            runner in hostile_html,
        ),
        (False, True, False),
    )
    check("zero-hour rows skipped", [f.key for f in hostile_report.files], [])

    if failures:
        print("self-test failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print(
        f"self-test passed ({len(SELF_TEST_LABELS)} labels, path, owner, aggregate and render checks)"
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="number of full UTC days to cover (default 14)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="exclusive end date in UTC (default today, so the window is the last N full days)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output HTML file (default agent_space/test-cost/<start>_<last>.html)",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=20,
        metavar="N",
        help="cross-check N jobs against the GitHub Actions API via gh (default 20, 0 disables)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="skip the Pixelcloud upload even when px is installed",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="refetch every day and write nothing to the cache",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run offline checks of the classifier, mapping, aggregation and rendering, then exit",
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")
    if args.verify_sample < 0:
        parser.error("--verify-sample must not be negative")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    today = datetime.now(timezone.utc).date()
    end = args.end or today
    if end > today:
        print(
            f"error: --end {end} is in the future (today is {today} UTC)",
            file=sys.stderr,
        )
        return 1
    window = Window(end - timedelta(days=args.days), end)
    out = args.out or OUT_DIR / f"{window.start}_{window.last}.html"
    script = str(SCRIPT_DIR.relative_to(REPO_ROOT) / "test_cost.py")
    command = ["python3", script, "--days", str(args.days), "--end", end.isoformat()]
    if args.verify_sample != 20:
        command += ["--verify-sample", str(args.verify_sample)]
    if args.no_cache:
        command.append("--no-cache")
    if args.no_upload:
        command.append("--no-upload")
    if args.out is not None:
        command += ["--out", str(args.out)]
    try:
        days = [load_day(day, today, args.no_cache) for day in window.days]
        meta = Meta(window, QUERY_HASH, git_checkout(), shlex.join(command))
        report = aggregate(days, REPO_ROOT, meta)
        for warning in report.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if report.total_jobs == 0:
            print(
                f"error: no test jobs found in {window.start}..{window.last}",
                file=sys.stderr,
            )
            return 2
        if args.verify_sample:
            report.verify = verify_sample(window, args.verify_sample)
            report.verify_requested = args.verify_sample
    except QueryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    html = render(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=out.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(html)
    os.chmod(tmp.name, 0o644)
    os.replace(tmp.name, out)
    title = f"CI test cost {window.start}..{window.last}: machine-hours by test file and owner"
    upload_result = "skipped (--no-upload)" if args.no_upload else upload(out, title)
    print_summary(report, out, upload_result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
