# Owner(s): ["module: inductor"]
"""Report strict pointwise coverage, including skips and unexecuted tests.

Use a fresh directory for each run:
    PYTORCH_POINTWISE_COVERAGE_DIR=/tmp/pointwise python test/inductor/test_strict_numerics.py
    python torch/testing/_internal/inductor_pointwise_coverage.py /tmp/pointwise

Workers write separate JSONL logs, archived by the existing CI log uploader.
The summary retains failed attempts across
reruns. A matched row describes the executed comparisons, with kernel generation
checked per test; it does not prove every signature used a generated kernel.
"""

import argparse
import contextlib
import json
import os
import unittest
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock


FAMILIES = (
    "test_pointwise_edges_backward",
    "test_pointwise_backward",
    "test_pointwise_bitwise",
    "test_pointwise_nonfloat",
    "test_pointwise_edges",
)
_WORKER_ID = uuid.uuid4().hex


def coverage_case(method):
    for family in FAMILIES:
        if method.startswith(family + "_"):
            op, separator, dtype = method[len(family) + 1 :].rpartition("_cuda_")
            if separator:
                return {"test": method, "family": family, "op": op, "dtype": dtype}
    return None


def write_event(directory, event):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"pointwise-{_WORKER_ID}-{os.getpid()}.log").open("a") as output:
        output.write(json.dumps(event) + "\n")


def record_class_fixtures(test_class, directory):
    def wrap(name):
        original = getattr(test_class, name)

        @classmethod
        def fixture(cls):
            try:
                return original()
            except BaseException as error:
                status = "skipped" if isinstance(error, unittest.SkipTest) else "failed"
                write_event(
                    directory,
                    {
                        "kind": "class_fixture",
                        "class": cls.__name__,
                        "fixture": name,
                        "status": status,
                        "reason": str(error),
                    },
                )
                raise

        return fixture

    test_class.setUpClass = wrap("setUpClass")
    test_class.tearDownClass = wrap("tearDownClass")


@dataclass
class CoverageRecord:
    case: dict
    expected_mismatch: bool = False
    compared: int = 0
    matched: int = 0
    generated_kernels: int = 0
    mismatches: Counter = field(default_factory=Counter)
    filters: Counter = field(default_factory=Counter)
    status: str = "not_run"
    reason: str = ""

    def event(self):
        return {
            "kind": "result",
            **self.case,
            "status": self.status,
            "reason": self.reason,
            "expected_mismatch": self.expected_mismatch,
            "compared": self.compared,
            "matched": self.matched,
            "generated_kernels": self.generated_kernels,
            "mismatches": dict(self.mismatches),
            "filters": dict(self.filters),
        }


@contextlib.contextmanager
def record_result(record, result, test):
    # Keep the original result object: common_utils dispatches on its type.
    with contextlib.ExitStack() as stack:

        def callback(name):
            original = getattr(result, name)

            def wrapped(item, *args):
                if name == "addSkip" and item is not test:
                    record.filters["skipped_subtest"] += 1
                if item is test:
                    if name == "addSubTest" and args[1] is not None:
                        record.status, record.reason = "failed", str(args[1][1])
                    elif name in ("addFailure", "addError", "addUnexpectedSuccess"):
                        record.status = "failed"
                        record.reason = (
                            str(args[0][1]) if args else "unexpected success"
                        )
                    elif record.status != "failed":
                        if name == "addSkip":
                            record.status, record.reason = "skipped", str(args[0])
                        elif name == "addExpectedFailure":
                            record.status, record.reason = (
                                "expected_failure",
                                str(args[0][1]),
                            )
                        elif name == "addSuccess":
                            if record.expected_mismatch and record.mismatches:
                                record.status = "known_mismatch"
                            elif (
                                not record.expected_mismatch
                                and record.compared > 0
                                and record.matched == record.compared
                                and record.generated_kernels > 0
                            ):
                                record.status = "matched"
                            else:
                                record.status = "unverified"
                return original(item, *args)

            return wrapped

        for name in (
            "addSuccess",
            "addFailure",
            "addError",
            "addSkip",
            "addExpectedFailure",
            "addUnexpectedSuccess",
        ):
            stack.enter_context(mock.patch.object(result, name, callback(name)))
        if hasattr(result, "addSubTest"):
            stack.enter_context(
                mock.patch.object(result, "addSubTest", callback("addSubTest"))
            )
        yield


def summarize(directory):
    cases = {}
    attempts = {}
    manifests = []
    class_fixtures = []
    for path in sorted(Path(directory).glob("pointwise-*.log")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["kind"] == "manifest":
                manifests.append(event["configuration"])
                cases.update((case["test"], case) for case in event["cases"])
            elif event["kind"] == "class_fixture":
                class_fixtures.append(event)
            else:
                attempts.setdefault(event["test"], []).append(event)
    if not manifests:
        raise ValueError("No pointwise coverage manifest found")
    if any(configuration != manifests[0] for configuration in manifests[1:]):
        raise ValueError(
            "Use separate coverage directories for different configurations"
        )
    if attempts.keys() - cases.keys():
        raise ValueError("Coverage results contain tests absent from the manifest")
    priority = {
        "failed": 0,
        "expected_failure": 1,
        "known_mismatch": 2,
        "unverified": 3,
        "skipped": 4,
        "matched": 5,
        "not_run": 6,
    }
    rows = []
    filters = Counter()
    for name, case in sorted(cases.items()):
        runs = attempts.get(name, [])
        row = (
            dict(min(runs, key=lambda run: priority[run["status"]]))
            if runs
            else {
                **case,
                "status": "not_run",
                "reason": "no completed result",
            }
        )
        row.pop("kind", None)
        row["attempts"] = len(runs)
        row["attempt_statuses"] = [run["status"] for run in runs]
        rows.append(row)
        for run in runs:
            filters.update(run["filters"])
    return {
        "configuration": manifests[0],
        "opinfos_without_generated_tests": sorted(
            set(manifests[0].get("selected_opinfo_ids", []))
            - {case["op"] for case in cases.values()}
        ),
        "kernel_check_scope": "per test, not per signature",
        "class_fixture_attempts": class_fixtures,
        "class_fixture_counts": dict(
            Counter(event["status"] for event in class_fixtures)
        ),
        "counts": dict(Counter(row["status"] for row in rows)),
        "filter_events_across_attempts": dict(filters),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = summarize(args.directory)
    output = args.directory / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("| Outcome | Tests |\n|---|---:|")
    for status, count in sorted(summary["counts"].items()):
        print(f"| {status} | {count} |")
    if summary["class_fixture_counts"]:
        print("\n| Class fixture outcome | Attempts |\n|---|---:|")
        for status, count in sorted(summary["class_fixture_counts"].items()):
            print(f"| {status} | {count} |")
    print(f"\nDetails: {output}")
    print(
        f"Selected OpInfos without generated tests: {len(summary['opinfos_without_generated_tests'])}"
    )
    return int(
        summary["counts"].get("failed", 0) > 0
        or summary["class_fixture_counts"].get("failed", 0) > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
