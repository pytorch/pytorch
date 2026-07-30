#!/usr/bin/env python3
"""Scan CI log files (.txt) for test failures not captured in XML reports.

Tests that timeout (exit code 124), crash (SIGIOT, SIGSEGV, Fatal Python error),
or are killed (SIGKILL, OOM) never produce JUnit XML output. This script detects
those failures from the raw log files and outputs a CSV/summary.

Usage:
    python detect_log_failures.py --logs-dir <folder> [--output <path.csv>]
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


RE_RUNNING = re.compile(
    r"Running (?P<test_file>\S+) (?P<shard>\d+)/(?P<total>\d+) \.\.\."
)
RE_SUCCESS = re.compile(
    r"(?P<test_file>\S+) (?P<shard>\d+)/(?P<total>\d+) was successful"
)
RE_FAILED = re.compile(
    r"(?P<test_file>\S+) (?P<shard>\d+)/(?P<total>\d+) failed!(?P<reason>.*)"
)
RE_EXIT_CODE = re.compile(r"Got exit code (?P<code>\d+)")
RE_TIMEOUT = re.compile(r"Command took >(\d+)min, returning 124")
RE_FAILED_CONSISTENTLY = re.compile(
    r"FAILED CONSISTENTLY: (?P<test_path>\S+)"
)
RE_STEPCURRENT = re.compile(
    r"stepcurrent:.*Running only (?:test/)?(?P<test_path>\S+)"
)
RE_INDIVIDUAL_TEST = re.compile(
    r"(?P<test_path>\S+\.py::(?P<cls>\w+)::(?P<method>\w+))"
)
RE_INDIV_PASSED = re.compile(
    r"(?:test/)?(?P<file>\S+\.py)::(?P<cls>\w+)::(?P<method>\S+?)\s+PASSED"
)
RE_NEW_PROCESS_SUCCESS = re.compile(r"Test succeeded in new process")

CRASH_PATTERNS = [
    (re.compile(r"Segmentation fault", re.IGNORECASE), "SEGFAULT"),
    (re.compile(r"SIGSEGV"), "SIGSEGV"),
    (re.compile(r"SIGIOT"), "SIGIOT"),
    (re.compile(r"SIGABRT"), "SIGABRT"),
    (re.compile(r"SIGKILL"), "SIGKILL"),
    (re.compile(r"Fatal Python error", re.IGNORECASE), "FATAL_PYTHON"),
    (re.compile(r"core dumped", re.IGNORECASE), "CORE_DUMP"),
    (re.compile(r"Aborted \(core dumped\)", re.IGNORECASE), "ABORTED"),
    (re.compile(r"torch\.cuda\.OutOfMemoryError"), "CUDA_OOM"),
    (re.compile(r"std::bad_alloc"), "BAD_ALLOC"),
]

LOG_FILE_MAP = {
    "rocm": ("rocm", "default"),
    "rocm_dist": ("rocm", "distributed"),
    "rocm_inductor": ("rocm", "inductor"),
    "cuda": ("cuda", "default"),
    "cuda_dist": ("cuda", "distributed"),
    "cuda_inductor": ("cuda", "inductor"),
    "baseline": ("baseline", "default"),
}


def classify_log_file(filename):
    """Return (platform, test_config, shard_num) from a log filename like rocm3.txt.

    Commit-vs-commit parity prefixes log files with the short commit SHA
    (for example, 09e0c59b_rocm3.txt). In that mode the SHA label is the
    platform name used by generate_summary.py, so preserve it here.
    """
    stem = Path(filename).stem
    label = None
    m = re.match(r"(?P<label>[0-9a-f]{8,40})_(?P<stem>.+)", stem)
    if m:
        label = m.group("label")[:8]
        stem = m.group("stem")
    for prefix, (platform, test_config) in sorted(LOG_FILE_MAP.items(), key=lambda x: -len(x[0])):
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            if remainder.isdigit():
                return label or platform, test_config, int(remainder)
    return None, None, None


RE_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*")
RE_TS_CAPTURE = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+)Z")


def _parse_ts(s):
    """Parse a CI log ISO-8601 timestamp (without the trailing Z) to datetime.

    Returns None if it doesn't match the expected shapes (with or without
    fractional seconds), so callers can skip un-timestamped lines."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_log_file(filepath):
    """Parse a single log file and return test file results, consistent failures,
    and flaky tests.

    A flaky test is one that failed in its normal-process run but PASSED when the
    CI harness re-ran it alone in a new subprocess (indicated by a PASSED line
    for the specific test::class::method, followed by 'Test succeeded in new
    process, continuing with the rest of the tests').
    """
    results = {}
    current_test = None
    last_failed_test = None
    consistent_failures = []
    flaky_tests = []
    last_passed_individual = None
    first_ts = None
    last_ts = None

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            # First/last ISO timestamps give the job's wall-clock (the run time
            # for log-based failures, which have no XML). Must run before the
            # line filter below, since skipped lines still advance the clock.
            m_ts = RE_TS_CAPTURE.match(line)
            if m_ts:
                ts = _parse_ts(m_ts.group(1))
                if ts is not None:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

            # Lightweight tracking of individual pytest test lines.
            # These are very frequent (~37% of lines) so we extract the
            # test name directly without timestamp stripping.
            if ".py::" in line:
                m_ind = RE_INDIVIDUAL_TEST.search(line)
                if m_ind:
                    active = current_test or last_failed_test
                    if active and active in results:
                        # Only update if the pytest path belongs to this shard's test file,
                        # otherwise rerun output from earlier shards contaminates later ones.
                        shard_file = results[active]["test_file"]
                        if shard_file + ".py" in m_ind.group("test_path"):
                            results[active]["last_test"] = f"{m_ind.group('cls')}::{m_ind.group('method')}"

            if " ... [" not in line and "was successful" not in line \
               and "failed!" not in line and "Got exit code" not in line \
               and "returning 124" not in line and "FAILED CONSISTENTLY" not in line \
               and "Retrying" not in line \
               and "Segmentation fault" not in line and "SIGIOT" not in line \
               and "SIGSEGV" not in line and "SIGABRT" not in line \
               and "SIGKILL" not in line \
               and "Fatal Python error" not in line and "core dumped" not in line \
               and "Aborted (core dumped)" not in line \
               and "OutOfMemoryError" not in line \
               and "bad_alloc" not in line \
               and "stepcurrent" not in line \
               and "PASSED" not in line \
               and "new process" not in line:
                continue

            stripped = RE_TIMESTAMP.sub("", line).rstrip()

            m = RE_RUNNING.search(stripped)
            if m:
                key = f"{m.group('test_file')} {m.group('shard')}/{m.group('total')}"
                current_test = key
                if key not in results:
                    results[key] = {
                        "test_file": m.group("test_file"),
                        "shard": int(m.group("shard")),
                        "total": int(m.group("total")),
                        "status": "RUNNING",
                        "reason": "",
                        "exit_codes": [],
                        "crashes": [],
                        "crash_tests": [],
                        "last_test": "",
                    }
                continue

            m = RE_SUCCESS.search(stripped)
            if m:
                key = f"{m.group('test_file')} {m.group('shard')}/{m.group('total')}"
                if key in results:
                    results[key]["status"] = "PASSED"
                current_test = None
                last_failed_test = None
                continue

            m = RE_FAILED.search(stripped)
            if m:
                key = f"{m.group('test_file')} {m.group('shard')}/{m.group('total')}"
                reason = m.group("reason").strip()
                if key in results:
                    results[key]["status"] = "FAILED"
                    if reason:
                        results[key]["reason"] = reason
                last_failed_test = key
                current_test = key
                continue

            active = current_test or last_failed_test

            # Track stepcurrent rerun lines — identifies crash-causing test
            m = RE_STEPCURRENT.search(stripped)
            if m:
                test_path = m.group("test_path")
                parts = test_path.split("::")
                if len(parts) >= 3:
                    crash_id = f"{parts[1]}::{parts[2]}"
                elif len(parts) == 2:
                    crash_id = parts[1]
                else:
                    crash_id = None
                if crash_id and active and active in results:
                    shard_file = results[active]["test_file"]
                    if shard_file in test_path:
                        if crash_id not in results[active]["crash_tests"]:
                            results[active]["crash_tests"].append(crash_id)
                continue

            # Track individual pytest test lines for last-running-test context
            m_ind = RE_INDIVIDUAL_TEST.search(stripped)
            if m_ind and active and active in results:
                cls = m_ind.group("cls")
                method = m_ind.group("method")
                results[active]["last_test"] = f"{cls}::{method}"

            m = RE_EXIT_CODE.search(stripped)
            if m:
                code = int(m.group("code"))
                if active and active in results:
                    results[active]["exit_codes"].append(code)

            m = RE_TIMEOUT.search(stripped)
            if m and active and active in results:
                if "TIMEOUT" not in results[active]["crashes"]:
                    results[active]["crashes"].append("TIMEOUT")

            m = RE_FAILED_CONSISTENTLY.search(stripped)
            if m:
                shard_str = ""
                if active and active in results:
                    info = results[active]
                    shard_str = f"{info['shard']}/{info['total']}"
                consistent_failures.append((m.group("test_path"), shard_str))

            # Detect individual PASSED lines for flaky-rerun tracking.
            m = RE_INDIV_PASSED.search(stripped)
            if m:
                last_passed_individual = {
                    "file": m.group("file"),
                    "cls": m.group("cls"),
                    "method": m.group("method"),
                    "active": active,
                }

            # When we see 'Test succeeded in new process' after a PASSED
            # individual test, that test was originally failing in the main
            # process (CI only falls back to rerun-in-new-process for tests
            # that crashed or failed) but passed on retry -> flaky.
            if RE_NEW_PROCESS_SUCCESS.search(stripped) and last_passed_individual:
                lp = last_passed_individual
                lp_active = lp.get("active")
                test_shard = ""
                if lp_active and lp_active in results:
                    info = results[lp_active]
                    test_shard = f"{info['shard']}/{info['total']}"
                flaky_tests.append({
                    "file": lp["file"],
                    "cls": lp["cls"],
                    "method": lp["method"],
                    "test_shard": test_shard,
                })
                last_passed_individual = None

            if active and active in results:
                for pattern, label in CRASH_PATTERNS:
                    if pattern.search(stripped):
                        if label not in results[active]["crashes"]:
                            results[active]["crashes"].append(label)

    job_run_time = ""
    if first_ts is not None and last_ts is not None:
        job_run_time = round(max(0.0, (last_ts - first_ts).total_seconds()), 2)

    return results, consistent_failures, flaky_tests, job_run_time


def scan_logs(logs_dir):
    """Scan all log files and return non-passing test file results plus a
    test-level shard inventory.

    Returns (all_failures, shard_inventory) where shard_inventory is a list
    of dicts with one entry per (platform, test_config, job_shard, test_file)
    combination seen in the logs, plus a sorted comma-separated list of the
    test-level shards observed (e.g. "1/1" or "1/15,2/15,...,15/15"). This
    lets downstream consumers look up the test-level shard for any XML-based
    failure whose only shard info is the job-level shard."""
    all_failures = []
    all_flaky = []
    shard_map = defaultdict(set)

    # Pre-compute job-level shard totals per (platform, test_config) by
    # counting how many log files belong to each group. Log files are
    # 1-indexed (e.g. rocm1.txt..rocm6.txt for a 6-way sharded job), so
    # the count == total shards for that CI job.
    shard_totals = defaultdict(int)
    for fname in os.listdir(logs_dir):
        if not fname.endswith(".txt"):
            continue
        platform, test_config, shard_num = classify_log_file(fname)
        if platform is None:
            continue
        shard_totals[(platform, test_config)] += 1

    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".txt"):
            continue

        platform, test_config, shard_num = classify_log_file(fname)
        if platform is None:
            continue

        job_total = shard_totals.get((platform, test_config), 0)
        job_shard_str = f"{shard_num}/{job_total}" if job_total else str(shard_num)

        # If download_testlogs left a "<log>.job_url" file next to this log,
        # it contains the URL of the upstream pytorch CI job that produced
        # the log. We surface it in the LOG-BASED FAILURES table as a link
        # to that job's page. Empty for older runs that predate this.
        job_url_file = os.path.join(logs_dir, fname + ".job_url")
        job_url = ""
        if os.path.isfile(job_url_file):
            with open(job_url_file) as f:
                job_url = f.read().strip()

        filepath = os.path.join(logs_dir, fname)
        results, consistent_failures, flaky_tests, job_run_time = parse_log_file(filepath)

        for ft in flaky_tests:
            file_part = ft["file"].replace("test/", "").replace(".py", "")
            all_flaky.append({
                "log_file": fname,
                "platform": platform,
                "test_config": test_config,
                "test_file": file_part,
                "test_class": ft["cls"],
                "test_name": ft["method"],
                "job_shard": job_shard_str,
                "test_shard": ft["test_shard"],
                "run_time": job_run_time,
                "job_url": job_url,
            })

        # Record every (test_file, test_shard) observed in this log file,
        # including PASSED ones, so the inventory covers the full run.
        for info in results.values():
            shard_map[(platform, test_config, job_shard_str, info["test_file"])].add(
                f"{info['shard']}/{info['total']}"
            )

        for key, info in results.items():
            if info["status"] == "PASSED":
                continue

            categories = []
            if 124 in info["exit_codes"] or "TIMEOUT" in info["crashes"]:
                categories.append("TIMEOUT")
            for c in info["crashes"]:
                if c != "TIMEOUT":
                    categories.append(c)
            if info["status"] == "FAILED" and not categories:
                categories.append("FAILED")
            if info["status"] == "RUNNING" and not categories:
                categories.append("INCOMPLETE")

            if not categories:
                continue
            # Skip tests stuck in RUNNING with no evidence of failure —
            # these are typically from multi-shard logs where a different
            # shard's "Running ..." line appeared but the result was elsewhere.
            if info["status"] == "RUNNING" and categories == ["INCOMPLETE"]:
                continue

            reason = info["reason"]
            # Populate reason with identified crash/timeout test name
            crash_tests = info.get("crash_tests", [])
            last_test = info.get("last_test", "")
            identified_test = ""
            if crash_tests:
                identified_test = crash_tests[0]
            elif last_test:
                identified_test = last_test

            if identified_test and "::" in identified_test:
                if not reason:
                    reason = identified_test
                elif "::" not in reason:
                    reason = f"{identified_test} | {reason}"

            all_failures.append({
                "log_file": fname,
                "platform": platform,
                "test_config": test_config,
                "test_file": info["test_file"],
                "job_shard": job_shard_str,
                "test_shard": f"{info['shard']}/{info['total']}",
                "status": info["status"],
                "category": "+".join(categories),
                "reason": reason,
                "exit_codes": ",".join(str(c) for c in info["exit_codes"]),
                "run_time": job_run_time,
                "job_url": job_url,
            })

        for test_path, shard_str in consistent_failures:
            parts = test_path.split("::")
            file_part = parts[0].replace("test/", "").replace(".py", "")
            test_class = parts[1] if len(parts) > 1 else ""
            test_name = parts[2] if len(parts) > 2 else ""

            all_failures.append({
                "log_file": fname,
                "platform": platform,
                "test_config": test_config,
                "test_file": file_part,
                "job_shard": job_shard_str,
                "test_shard": shard_str,
                "status": "FAILED_CONSISTENTLY",
                "category": "CONSISTENT_FAILURE",
                "reason": f"{test_class}::{test_name}" if test_class else "",
                "exit_codes": "",
                "run_time": job_run_time,
                "job_url": job_url,
            })

    def _sort_shards(vals):
        def key(v):
            try:
                a, b = v.split("/", 1)
                return (int(b), int(a))
            except (ValueError, AttributeError):
                return (0, 0)
        return sorted(vals, key=key)

    shard_inventory = [
        {
            "platform": platform,
            "test_config": test_config,
            "job_shard": job_shard_str,
            "test_file": test_file,
            "test_shards": ",".join(_sort_shards(shards)),
        }
        for (platform, test_config, job_shard_str, test_file), shards in shard_map.items()
    ]
    shard_inventory.sort(key=lambda r: (r["platform"], r["test_config"],
                                        r["job_shard"], r["test_file"]))

    all_flaky.sort(key=lambda r: (r["platform"], r["test_config"],
                                  r["job_shard"], r["test_file"],
                                  r["test_class"], r["test_name"]))

    return all_failures, shard_inventory, all_flaky


def write_csv_report(failures, output_path):
    fieldnames = [
        "log_file", "platform", "test_config", "test_file",
        "job_shard", "test_shard",
        "status", "category", "reason", "exit_codes",
        "run_time",
        "job_url",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)
    print(f"Log failure report: {output_path} ({len(failures)} entries)")


def write_shards_report(inventory, output_path):
    fieldnames = ["platform", "test_config", "job_shard", "test_file", "test_shards"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)
    print(f"Log shard inventory: {output_path} ({len(inventory)} entries)")


def write_flaky_report(flaky, output_path):
    fieldnames = [
        "log_file", "platform", "test_config", "test_file",
        "test_class", "test_name", "job_shard", "test_shard",
        "run_time",
        "job_url",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flaky)
    print(f"Flaky test report: {output_path} ({len(flaky)} entries)")


def _derive_sibling_path(output_path, new_prefix):
    """Given an output path like '.../log_failures_mi355.csv' and
    new_prefix='log_shards', return '.../log_shards_mi355.csv'. Falls back to
    appending '.{new_prefix}.csv' if the expected prefix isn't present."""
    d, base = os.path.split(output_path)
    if base.startswith("log_failures"):
        return os.path.join(d, new_prefix + base[len("log_failures"):])
    stem, ext = os.path.splitext(base)
    return os.path.join(d, f"{stem}.{new_prefix}{ext or '.csv'}")


def _derive_shards_path(output_path):
    return _derive_sibling_path(output_path, "log_shards")


def _derive_flaky_path(output_path):
    return _derive_sibling_path(output_path, "flaky_tests")


def print_summary(failures):
    if not failures:
        print("No log-based failures detected.")
        return

    by_category = defaultdict(list)
    for f in failures:
        by_category[f["category"]].append(f)

    print(f"\n{'='*60}")
    print("LOG FAILURE DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total failures detected: {len(failures)}")
    print()

    for cat, items in sorted(by_category.items()):
        print(f"  {cat}: {len(items)}")
        for item in items:
            print(f"    - {item['test_file']} ({item['platform']}/{item['test_config']}) [{item['log_file']}]")
            if item["reason"]:
                print(f"      Reason: {item['reason'][:120]}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Detect test failures from CI log files not captured in XML reports"
    )
    parser.add_argument(
        "--logs-dir", required=True,
        help="Directory containing .txt log files"
    )
    parser.add_argument(
        "--output", default="log_failures.csv",
        help="Output CSV path (default: log_failures.csv)"
    )
    args = parser.parse_args()

    failures, shard_inventory, flaky_tests = scan_logs(args.logs_dir)
    print_summary(failures)
    write_csv_report(failures, args.output)
    write_shards_report(shard_inventory, _derive_shards_path(args.output))
    write_flaky_report(flaky_tests, _derive_flaky_path(args.output))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
