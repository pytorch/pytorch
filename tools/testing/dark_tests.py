#!/usr/bin/env python3
"""Find tests that exist but never run anywhere in CI.

A test only runs if some job selects it and no gate skips it. Neither is checked
against the other, so tests go dark -- and a skipped test is a green test. This
compares the tests a file defines against the tests trunk actually observed
running, and reports the difference.

Unlike gpu_coverage.py this asks nothing about *why* a test is dark. Any reason
counts: a wrong decorator, a gate no runner satisfies, a file no config selects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test"
KNOWN_DARK = REPO_ROOT / "tools" / "testing" / "known_dark_tests.json"

sys.path.insert(0, str(REPO_ROOT))

# Enumerate under the widest platform: device-generic tests are generated per
# device, so a narrower one under-counts what exists.
ENUMERATION_JOBS = ["linux-cuda-sm100/default", "linux-cpu/default"]

# Tests that ran on any runner in this window count as covered. Long enough to
# span the periodic h100/b200 schedules and to absorb sharding and flakiness.
DEFAULT_DAYS = 7

# Narrow to trunk jobs before touching test_run_s3: a week of every test run in
# PyTorch CI is far too much to scan and then filter.
RAN_QUERY = """
WITH runs AS (
    SELECT id
    FROM default.workflow_run w FINAL
    WHERE w.head_branch = 'main'
      AND w.repository.'full_name' = 'pytorch/pytorch'
      AND w.created_at > now() - INTERVAL {days: Int32} DAY
), jobs AS (
    SELECT id
    FROM default.workflow_job j FINAL
    WHERE j.run_id IN (SELECT id FROM runs)
)
SELECT DISTINCT
    test_run.classname AS classname,
    test_run.name AS name
FROM default.test_run_s3 test_run
WHERE test_run.job_id IN (SELECT id FROM jobs)
  AND test_run.classname != ''
  AND empty(test_run.skipped)
  AND test_run.time_inserted > now() - INTERVAL {days: Int32} DAY
"""


def existing(files: list[str]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """({file: {Class::method}}, {file: why it could not be enumerated}).

    A file that fails to enumerate must be reported, not treated as empty: zero
    known tests is indistinguishable from zero dark tests.
    """
    from tools.testing.introspection import collector, platforms

    out: dict[str, set[str]] = {f: set() for f in files}
    errors: dict[str, str] = {}
    for job in ENUMERATION_JOBS:
        for rel, payload in collector.collect(
            platforms.get_job(job), "enumerate", files
        ).items():
            if "error" in payload:
                errors.setdefault(rel, f"{job}: {payload['error']}")
                continue
            for cls, methods in payload["classes"].items():
                out[rel].update(f"{cls}::{m}" for m in methods)
    # Enumerating under any platform is enough; only a file no platform could
    # read is genuinely unmeasured.
    return out, {r: e for r, e in errors.items() if not out[r]}


def ran(days: int) -> set[str]:
    from tools.testing.clickhouse import query_clickhouse

    rows = query_clickhouse(RAN_QUERY, {"days": days})
    return {f"{r['classname']}::{r['name']}" for r in rows}


def load_known_dark() -> dict[str, str]:
    if not KNOWN_DARK.is_file():
        return {}
    return json.loads(KNOWN_DARK.read_text(encoding="utf-8"))


def disabled() -> set[str]:
    """Tests the disable bot has turned off; dark on purpose, not a gap."""
    from tools.stats.import_test_stats import get_disabled_tests

    try:
        raw = get_disabled_tests(dirpath=str(REPO_ROOT / ".additional_ci_files"))
    except Exception:
        return set()
    out = set()
    for entry in raw or {}:
        # "test_foo (__main__.TestBar)" -> "TestBar::test_foo"
        name, _, rest = entry.partition(" (")
        cls = rest.rstrip(")").rpartition(".")[2]
        if cls:
            out.add(f"{cls}::{name}")
    return out


def test_files(paths: list[str]) -> list[str]:
    if paths:
        return sorted({str(Path(p).resolve().relative_to(REPO_ROOT)) for p in paths})
    return sorted(
        str(p.relative_to(REPO_ROOT))
        for p in TEST_ROOT.rglob("test_*.py")
        if "/fb/" not in str(p) and "/cpython/" not in str(p)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument(
        "--ran-json",
        help="read the observed-ran set from a file instead of querying CI, so "
        "the join can be exercised without ClickHouse credentials",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    files = test_files(args.files)
    observed = (
        set(json.loads(Path(args.ran_json).read_text(encoding="utf-8")))
        if args.ran_json
        else ran(args.days)
    )
    if not observed:
        # An empty set would make every test look dark.
        raise SystemExit("dark_tests: no observed test runs; refusing to report")

    known = load_known_dark()
    off = disabled()
    defined, unreadable = existing(files)

    report: dict[str, list[str]] = {}
    for rel, tests in defined.items():
        gap = sorted(t for t in tests - observed if t not in off and rel not in known)
        if gap:
            report[rel] = gap

    if args.json:
        json.dump(
            {
                "dark": report,
                "unreadable": unreadable,
                "files_scanned": len(files),
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    total = sum(len(v) for v in report.values())
    for rel in sorted(report):
        print(f"\n{rel}: {len(report[rel])} test(s) never observed running")
        for t in report[rel][:20]:
            print(f"    {t}")
        if len(report[rel]) > 20:
            print(f"    ... {len(report[rel]) - 20} more")
    for rel in sorted(unreadable):
        print(f"\n{rel}: NOT MEASURED -- {unreadable[rel]}")
    print(
        f"\n{total} dark test(s) across {len(report)} of {len(files)} file(s); "
        f"{len(unreadable)} file(s) could not be enumerated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
