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

sys.path.insert(0, str(REPO_ROOT))

# Enumerate under the widest platform: device-generic tests are generated per
# device, so a narrower one under-counts what exists.
ENUMERATION_JOBS = ["linux-cuda-sm100/default", "linux-cpu/default"]

# Tests that ran on any runner in this window count as covered. Long enough to
# span the periodic h100/b200 schedules and to absorb sharding and flakiness.
DEFAULT_DAYS = 7

# Shaped after test-infra's testStats3d: bound test_run_s3 by time first, then join
# up to the job, rather than collecting job ids and filtering a week of every test
# run in PyTorch CI afterwards.
#
# trunk/<sha> refs are included alongside main because upload_test_stats.py's
# should_upload_full_test_run accepts both, so restricting to main alone would
# discard part of the observed set and make live tests look dark.
RAN_QUERY = """
SELECT DISTINCT
    t.classname AS classname,
    t.name AS name
FROM default.test_run_s3 t
INNER JOIN default.workflow_job j FINAL ON t.job_id = j.id
INNER JOIN default.workflow_run w FINAL ON j.run_id = w.id
WHERE t.time_inserted > now() - INTERVAL {days: Int32} DAY
  AND t.classname != ''
  AND empty(t.skipped)
  AND (w.head_branch = 'main' OR match(w.head_branch, '^trunk/[0-9a-fA-F]{40}$'))
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


def unscheduled(files: list[str]) -> set[str]:
    """Files no run_test.py config can select, derived rather than declared.

    discover_tests.py already decides this, so reading it keeps the answer current;
    a checked-in list of the same files would drift the moment that one changed.
    Files whose tests run under an aggregator (fx/ via test_fx.py, and so on) are
    not in TESTS but their tests still appear in the observed set, so they resolve
    through the join and are deliberately not treated as unscheduled here.
    """
    from tools.testing.discover_tests import TESTS

    known = set(TESTS)
    return {f for f in files if f[len("test/") : -len(".py")] not in known}


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
        out = set()
        for p in paths:
            try:
                rel = str(Path(p).resolve().relative_to(REPO_ROOT))
            except ValueError:
                raise SystemExit(f"dark_tests: {p} is outside {REPO_ROOT}") from None
            if not (rel.startswith("test/") and rel.endswith(".py")):
                raise SystemExit(f"dark_tests: {p} is not a test/**/*.py file")
            out.add(rel)
        return sorted(out)
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
    ap.add_argument("--out", help="also write the JSON report here")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    files = test_files(args.files)
    if args.ran_json:
        try:
            observed = set(json.loads(Path(args.ran_json).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as e:
            raise SystemExit(f"dark_tests: cannot read {args.ran_json}: {e}") from None
    else:
        observed = ran(args.days)
    if not observed:
        # An empty set would make every test look dark.
        raise SystemExit("dark_tests: no observed test runs; refusing to report")

    off = disabled()
    no_config = unscheduled(files)
    defined, unreadable = existing(files)

    report: dict[str, list[str]] = {}
    for rel, tests in defined.items():
        gap = sorted(t for t in tests - observed if t not in off)
        if gap:
            report[rel] = gap

    payload = {
        "dark": {k: v for k, v in report.items() if k not in no_config},
        "unscheduled": {k: v for k, v in report.items() if k in no_config},
        "unreadable": unreadable,
        "files_scanned": len(files),
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        print()
        return 0

    dark, no_cfg = payload["dark"], payload["unscheduled"]
    for section, title in (
        (dark, "never observed running"),
        (no_cfg, "no CI config selects this file"),
    ):
        for rel in sorted(section):
            print(f"\n{rel}: {len(section[rel])} test(s), {title}")
            for name in section[rel][:20]:
                print(f"    {name}")
            if len(section[rel]) > 20:
                print(f"    ... {len(section[rel]) - 20} more")
    for rel in sorted(unreadable):
        print(f"\n{rel}: NOT MEASURED -- {unreadable[rel]}")
    print(
        f"\n{sum(len(v) for v in dark.values())} dark test(s) across {len(dark)} file(s); "
        f"{sum(len(v) for v in no_cfg.values())} in {len(no_cfg)} unscheduled file(s); "
        f"{len(unreadable)} file(s) could not be enumerated; {len(files)} scanned"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
