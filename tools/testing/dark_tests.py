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
import contextlib
import json
import sys
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test"

sys.path.insert(0, str(REPO_ROOT))

# Enumerate under the widest platform: device-generic tests are generated per
# device, so a narrower one under-counts what exists.
ENUMERATION_JOBS = ["linux-cuda-sm100/default", "linux-cpu/default"]

# Tests that ran on any runner in this window count as covered. Long enough to
# span the periodic h100/b200 schedules and to absorb sharding and flakiness.
DEFAULT_DAYS = 7

# Shaped after test-infra's testStats3d. Every table is bounded by time: without a
# bound on the job and run sides ClickHouse builds the hash side of each join from
# their entire history under FINAL, which does not complete.
RAN_QUERY = """
SELECT DISTINCT
    t.classname AS classname,
    t.name AS name
FROM default.test_run_s3 t
INNER JOIN default.workflow_job j FINAL ON t.job_id = j.id
INNER JOIN default.workflow_run w FINAL ON j.run_id = w.id
WHERE t.time_inserted > now() - INTERVAL {days: Int32} DAY
  AND j.created_at > now() - INTERVAL {days: Int32} DAY
  AND w.created_at > now() - INTERVAL {days: Int32} DAY
  AND t.classname != ''
  AND empty(t.skipped)
  AND w.head_branch = 'main'
  AND w.head_repository.'full_name' = 'pytorch/pytorch'
"""


# Job names embed the runner label, so the observed capability of a run is
# readable from the job name. g4dn/g5/g6 are the sm75-sm89 runners every
# auto-discovered config lands on; h100/b200 are the scarce ones the smoke lists
# curate. Aggregated in ClickHouse rather than returned per job: a week of
# per-(test, job) rows is orders of magnitude larger than the answer.
BIG_RUNNERS = ("linux.aws.h100", "linux.dgx.b200")
SMALL_RUNNERS = ("linux.g4dn", "linux.g5.", "linux.g6.")

ARCH_QUERY = """
SELECT
    t.classname AS classname,
    t.name AS name,
    maxIf(1, empty(t.skipped) AND ({big})) AS ran_big,
    maxIf(1, notEmpty(t.skipped) AND ({small})) AS skipped_small
FROM default.test_run_s3 t
INNER JOIN default.workflow_job j FINAL ON t.job_id = j.id
INNER JOIN default.workflow_run w FINAL ON j.run_id = w.id
WHERE t.time_inserted > now() - INTERVAL {{days: Int32}} DAY
  AND j.created_at > now() - INTERVAL {{days: Int32}} DAY
  AND w.created_at > now() - INTERVAL {{days: Int32}} DAY
  AND t.classname != ''
  AND w.head_branch = 'main'
  AND w.head_repository.'full_name' = 'pytorch/pytorch'
GROUP BY t.classname, t.name
HAVING ran_big = 1 AND skipped_small = 1
""".format(
    big=" OR ".join(f"j.name LIKE '%{r}%'" for r in BIG_RUNNERS),
    small=" OR ".join(f"j.name LIKE '%{r}%'" for r in SMALL_RUNNERS),
)


def requires_big_gpu(days: int) -> set[str]:
    """Tests observed running on an h100/b200 runner and skipping on a small one.

    This is the same question gpu_coverage.py answers by simulation, but measured,
    so it sees requirements expressed anywhere -- setUp, a test body, a library
    probe -- not only those a decorator declares.
    """
    from tools.testing.clickhouse import query_clickhouse

    rows = query_clickhouse(ARCH_QUERY, {"days": days})
    return {f"{r['classname']}::{r['name']}" for r in rows}


class Enumerated(NamedTuple):
    tests: dict[str, set[str]]  # {file: {Class::method}}
    xfail: set[str]  # expected failures: run, but recorded as skipped
    unreadable: dict[str, str]  # no platform could import the file
    partial: dict[str, str]  # some platform could not, so tests may be missing


def existing(files: list[str]) -> Enumerated:
    """What each file defines, and what could not be read.

    A file that fails to enumerate must be reported rather than treated as empty:
    zero known tests is indistinguishable from zero dark tests. Failing under only
    some platforms is reported too, since the tests that platform would have
    contributed are missing from the comparison with no other trace.
    """
    from tools.testing.introspection import collector, platforms

    tests: dict[str, set[str]] = {f: set() for f in files}
    xfail: set[str] = set()
    errors: dict[str, str] = {}
    for job in ENUMERATION_JOBS:
        for rel, payload in collector.collect(
            platforms.get_job(job), "enumerate", files
        ).items():
            if "error" in payload:
                errors.setdefault(rel, f"{job}: {payload['error']}")
                continue
            for cls, methods in payload["classes"].items():
                tests[rel].update(f"{cls}::{m}" for m in methods)
            xfail.update(payload.get("xfail", ()))
    return Enumerated(
        tests=tests,
        xfail=xfail,
        unreadable={r: e for r, e in errors.items() if not tests[r]},
        partial={r: e for r, e in errors.items() if tests[r]},
    )


def ambiguous(tests: dict[str, set[str]]) -> set[str]:
    """Keys defined by more than one file.

    Observations are keyed Class::method with no file, so a live test in one file
    marks an identically named test elsewhere live too. AOTInductorLoggingTest::
    test_shape_env_reuse exists in both test_aot_inductor.py and
    test_aot_inductor_custom_ops.py, for instance. Rather than key on the file --
    which would mean trusting a format this has no way to check -- report the
    verdicts that could be masked.
    """
    seen: dict[str, int] = {}
    for owned in tests.values():
        for key in owned:
            seen[key] = seen.get(key, 0) + 1
    return {k for k, n in seen.items() if n > 1}


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


def unlisted(needs_big: set[str], defined: dict[str, set[str]]) -> dict[str, list[str]]:
    """Of the tests measured to need an h100/b200, those a smoke list omits.

    ciflow/h100 and ciflow/b200 are independent labels, so a test needing scarce
    hardware belongs in both lists; missing from either is a gap.
    """
    from tools.testing.gpu_coverage import selects, smoke_patterns, TARGETS, TEST_SH

    text = TEST_SH.read_text(encoding="utf-8")
    patterns = {fn: smoke_patterns(text, fn) for _, _, fn in TARGETS}
    owner = {t: rel for rel, tests in defined.items() for t in tests}

    out: dict[str, list[str]] = {}
    for test in sorted(needs_big):
        rel = owner.get(test)
        if rel is None:
            continue
        include = rel[len("test/") : -len(".py")]
        if any(
            not selects(patterns[fn].get(include, []), test) for _, _, fn in TARGETS
        ):
            out.setdefault(rel, []).append(test)
    return out


def disabled() -> set[str]:
    """Tests the disable bot turned off everywhere; dark on purpose, not a gap.

    Only unconditional disables are exempt. Most entries name the platforms they
    apply to, and a test disabled on rocm alone that has also gone dark on every
    other runner is a real gap, not a sanctioned one.
    """
    from tools.stats.import_test_stats import get_disabled_tests

    # get_disabled_tests narrates the download on stdout, which would land in the
    # middle of --json output.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            raw = get_disabled_tests(dirpath=str(REPO_ROOT / ".additional_ci_files"))
        except Exception:
            raw = None
    if not raw:
        print(
            "dark_tests: no disabled-test set; none will be exempted", file=sys.stderr
        )
        return set()

    out = set()
    for entry, value in raw.items():
        platforms = (
            value[1] if isinstance(value, (list, tuple)) and len(value) > 1 else None
        )
        if platforms:
            continue
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
    enum = existing(files)

    # Expected failures execute in full but JUnit records them as skipped, so no
    # observation can ever show them running; excluding them beats reporting a
    # gap nothing can close.
    exempt = off | enum.xfail
    masked = ambiguous(enum.tests)

    report: dict[str, list[str]] = {}
    for rel, tests in enum.tests.items():
        gap = sorted(t for t in tests - observed if t not in exempt)
        if gap:
            report[rel] = gap

    # Only available from a live query: --ran-json carries no per-job detail.
    needs_big = set() if args.ran_json else requires_big_gpu(args.days)

    payload = {
        "dark": {k: v for k, v in report.items() if k not in no_config},
        "unscheduled": {k: v for k, v in report.items() if k in no_config},
        "unlisted": unlisted(needs_big, enum.tests),
        "unreadable": enum.unreadable,
        "partially_enumerated": enum.partial,
        "ambiguous": sorted({t for tests in report.values() for t in tests} & masked),
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
        (
            payload["unlisted"],
            "measured to need an h100/b200, missing from a smoke list",
        ),
    ):
        for rel in sorted(section):
            print(f"\n{rel}: {len(section[rel])} test(s), {title}")
            for name in section[rel][:20]:
                print(f"    {name}")
            if len(section[rel]) > 20:
                print(f"    ... {len(section[rel]) - 20} more")
    for rel in sorted(enum.unreadable):
        print(f"\n{rel}: NOT MEASURED -- {enum.unreadable[rel]}")
    for rel in sorted(enum.partial):
        print(f"\n{rel}: PARTIALLY ENUMERATED -- {enum.partial[rel]}")
    if payload["ambiguous"]:
        print(
            f"\n{len(payload['ambiguous'])} verdict(s) may be masked: another file "
            f"defines the same Class::method, e.g. {payload['ambiguous'][0]}"
        )
    print(
        f"\n{sum(len(v) for v in dark.values())} dark test(s) across {len(dark)} file(s); "
        f"{sum(len(v) for v in no_cfg.values())} in {len(no_cfg)} unscheduled file(s); "
        f"{len(enum.unreadable)} unreadable, {len(enum.partial)} partial; "
        f"{len(files)} scanned"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
