#!/usr/bin/env python3
"""Generate test-times.json and test-class-times.json from local JUnit XML reports.

For use by external CI pipelines (e.g. RISC-V) that cannot reach PyTorch's
internal S3/ClickHouse infrastructure.  Output schema is identical to
import_test_stats.get_test_times() / get_test_class_times() so no consumer
changes are required.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import DefaultDict
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def _parse_xml(report: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Parse a single JUnit XML report.

    Returns
    -------
    module_times : dict[str, float]
        Total elapsed seconds keyed by *invoking module* name.  The invoking
        module is the directory that contains the XML file, matching the
        convention used by upload_test_stats.py (``report.parent.name``).
    class_times : dict[str, float]
        Total elapsed seconds keyed by ``"<classname>::<invoking_module>"``.
        Entries with a missing or empty classname are skipped.
    """
    invoking_module = report.parent.name
    module_total: float = 0.0
    class_totals: DefaultDict[str, float] = defaultdict(float)

    try:
        tree = ET.parse(report)  # noqa: S314 – local files only, no network input
    except ET.ParseError:
        # Malformed XML is silently skipped; a warning is printed so the
        # operator is aware but the rest of the run is unaffected.
        print(f"WARNING: skipping malformed XML: {report}", file=sys.stderr)
        return {}, {}

    for testcase in tree.iter("testcase"):
        raw_time = testcase.get("time", "")
        try:
            elapsed = float(raw_time)
        except ValueError:
            # Missing or non-numeric time attribute – skip this testcase.
            continue

        module_total += elapsed

        classname = testcase.get("classname", "").strip()
        if classname:
            class_totals[f"{classname}::{invoking_module}"] += elapsed

    module_times = {invoking_module: module_total} if module_total > 0 else {}
    return module_times, dict(class_totals)


def collect_times(
    reports_dir: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Walk *reports_dir* recursively and aggregate all JUnit XML files.

    Returns
    -------
    module_times, class_times
        Aggregated dictionaries as described in ``_parse_xml``.
    """
    all_module_times: DefaultDict[str, float] = defaultdict(float)
    all_class_times: DefaultDict[str, float] = defaultdict(float)

    xml_files = list(reports_dir.rglob("*.xml"))
    if not xml_files:
        print(
            f"WARNING: no XML reports found under {reports_dir}",
            file=sys.stderr,
        )

    for xml_file in xml_files:
        mod_times, cls_times = _parse_xml(xml_file)
        for module, t in mod_times.items():
            all_module_times[module] += t
        for cls, t in cls_times.items():
            all_class_times[cls] += t

    return dict(all_module_times), dict(all_class_times)


# ---------------------------------------------------------------------------
# JSON construction
# ---------------------------------------------------------------------------


def _build_payload(
    times: dict[str, float],
    job_name: str,
    test_config: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Wrap *times* in the three-level lookup structure expected by run_test.py.

    The three keys correspond to the fallback hierarchy:
        [job_name][test_config]   – exact match
        ["default"][test_config]  – any job, specific config
        ["default"]["default"]    – ultimate fallback
    """
    inner = {test_config: times, "default": times}
    return {
        job_name: inner,
        "default": inner,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build test-times.json / test-class-times.json from local JUnit XML "
            "reports for use by run_test.py shard balancing on external CI."
        ),
    )
    parser.add_argument(
        "--reports-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root directory that contains JUnit XML reports (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help=(
            "Directory where test-times.json and test-class-times.json are written. "
            "Created if it does not exist."
        ),
    )
    parser.add_argument(
        "--job-name",
        required=True,
        metavar="NAME",
        help=(
            'Top-level key in the output JSON, e.g. "riscv64-linux-py3.12 / test". '
            "Use the same value that the CI job would report."
        ),
    )
    parser.add_argument(
        "--test-config",
        default="default",
        metavar="CONFIG",
        help=(
            'Second-level key in the output JSON, e.g. "default" or "slow". '
            "Matches the TEST_CONFIG environment variable used by run_test.py."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    reports_dir: Path = args.reports_dir
    output_dir: Path = args.output_dir

    if not reports_dir.is_dir():
        print(f"ERROR: reports-dir does not exist: {reports_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning XML reports under: {reports_dir}")
    module_times, class_times = collect_times(reports_dir)
    print(
        f"Aggregated {len(module_times)} module(s) "
        f"and {len(class_times)} class(es) from XML reports."
    )

    test_times_path = output_dir / "test-times.json"
    class_times_path = output_dir / "test-class-times.json"

    test_times_payload = _build_payload(module_times, args.job_name, args.test_config)
    class_times_payload = _build_payload(class_times, args.job_name, args.test_config)

    test_times_path.write_text(json.dumps(test_times_payload, indent=2), encoding="utf-8")
    class_times_path.write_text(json.dumps(class_times_payload, indent=2), encoding="utf-8")

    print(f"Wrote: {test_times_path}")
    print(f"Wrote: {class_times_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
