# Owner(s): ["module: ci"]

"""Compare historical PR test reports with TD tracer selections."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pr_tests


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing import select_tests_from_td_tracer


sys.path.remove(str(REPO_ROOT))


REPORT_SCHEMA_VERSION = 3
_PR_TRAILER = re.compile(
    r"^Pull Request resolved: https://github\.com/pytorch/pytorch/pull/([1-9]\d*)\s*$",
    re.MULTILINE,
)
_TEST_FILE_ALIASES = {
    "test_cpp_extensions_aot_ninja.py": "test_cpp_extensions_aot.py",
    "test_cpp_extensions_aot_no_ninja.py": "test_cpp_extensions_aot.py",
}
_CSV_FIELDS = (
    "pr_number",
    "url",
    "landing_commit",
    "committed_at",
    "subject",
    "status",
    "fully_covered_changes",
    "changed_file_count",
    "actual_unique",
    "selected_unique",
    "selected_comparable_unique",
    "selected_non_junit_unique",
    "intersection_unique",
    "actual_only_unique",
    "selected_only_unique",
    "selected_only_comparable_unique",
    "tests_reduced",
    "reduction_percent",
    "workflow_runs",
    "artifacts",
    "reports",
    "changed_files",
    "unsupported_files",
    "unmatched_files",
    "error",
)


class ComparisonError(RuntimeError):
    pass


@dataclass(frozen=True)
class LandedPR:
    number: int
    landing_commit: str
    committed_at: str
    subject: str
    changed_files: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"https://github.com/pytorch/pytorch/pull/{self.number}"


@dataclass(frozen=True)
class PredictionIndex:
    comparable_masks: Mapping[str, int]
    non_junit_masks: Mapping[str, int]
    comparable_counts: tuple[int, ...]
    non_junit_counts: tuple[int, ...]
    unsupported: tuple[tuple[str, ...], ...]
    unmatched: tuple[tuple[str, ...], ...]


def _git_output(args: Sequence[str], repo_root: Path = REPO_ROOT) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise ComparisonError(f"Git command failed{suffix}") from error
    except OSError as error:
        raise ComparisonError(f"Unable to run Git: {error}") from error
    return result.stdout


def _resolve_commit(revision: str, repo_root: Path = REPO_ROOT) -> str:
    if not revision or revision.startswith("-") or not revision.isprintable():
        raise ComparisonError("Main ref must be a printable Git revision")
    output = _git_output(["rev-parse", "--verify", f"{revision}^{{commit}}"], repo_root)
    try:
        commit = output.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ComparisonError("Git returned an invalid main commit") from error
    if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit) is None:
        raise ComparisonError("Git returned an invalid main commit")
    return commit.lower()


def _history(
    revision: str, max_count: int, repo_root: Path = REPO_ROOT
) -> list[tuple[str, str, str, str]]:
    output = _git_output(
        [
            "log",
            "-z",
            "--first-parent",
            f"--max-count={max_count}",
            "--format=%H%x00%cI%x00%s%x00%B",
            revision,
            "--",
        ],
        repo_root,
    )
    fields = output.split(b"\0")
    if fields[-1:] == [b""]:
        fields.pop()
    if len(fields) % 4:
        raise ComparisonError("Git returned malformed history")
    entries: list[tuple[str, str, str, str]] = []
    try:
        for index in range(0, len(fields), 4):
            decoded = [field.decode("utf-8") for field in fields[index : index + 4]]
            entries.append((decoded[0], decoded[1], decoded[2], decoded[3]))
    except UnicodeDecodeError as error:
        raise ComparisonError("Git history contains non-UTF-8 metadata") from error
    return entries


def enumerate_landed_prs(
    main_ref: str, limit: int, repo_root: Path = REPO_ROOT
) -> tuple[str, list[LandedPR]]:
    if limit < 1:
        raise ComparisonError("PR limit must be positive")
    main_tip = _resolve_commit(main_ref, repo_root)
    max_count = max(100, limit * 2)
    while True:
        entries = _history(main_tip, max_count, repo_root)
        selected: list[tuple[int, str, str, str]] = []
        seen: set[int] = set()
        for commit, committed_at, subject, body in entries:
            match = _PR_TRAILER.search(body)
            if match is None:
                continue
            number = int(match.group(1))
            if number in seen:
                continue
            seen.add(number)
            selected.append((number, commit, committed_at, subject))
            if len(selected) == limit:
                break
        if len(selected) == limit:
            break
        if len(entries) < max_count:
            raise ComparisonError(
                f"Found only {len(selected)} unique landed PRs in {main_ref}"
            )
        max_count *= 2

    prs = [
        LandedPR(
            number=number,
            landing_commit=commit,
            committed_at=committed_at,
            subject=subject,
            changed_files=tuple(
                select_tests_from_td_tracer.changed_files_for_commit(commit, repo_root)
            ),
        )
        for number, commit, committed_at, subject in selected
    ]
    return main_tip, prs


def canonical_test_id(test: str) -> str:
    source, separator, remainder = test.partition("::")
    parent, slash, basename = source.rpartition("/")
    canonical = _TEST_FILE_ALIASES.get(basename)
    if canonical is None:
        return test
    source = f"{parent}{slash}{canonical}"
    return f"{source}{separator}{remainder}" if separator else source


def _add_mask(masks: dict[str, int], test: str, mask: int) -> None:
    masks[test] = masks.get(test, 0) | mask


def _counts_for_masks(masks: Iterable[int], size: int) -> tuple[int, ...]:
    counts = [0] * size
    for original_mask in masks:
        mask = original_mask
        while mask:
            bit = mask & -mask
            counts[bit.bit_length() - 1] += 1
            mask ^= bit
    return tuple(counts)


def build_prediction_index(
    prs: Sequence[LandedPR], selection: select_tests_from_td_tracer.SelectionResult
) -> PredictionIndex:
    path_masks: dict[str, int] = {}
    for index, pr in enumerate(prs):
        bit = 1 << index
        for path in pr.changed_files:
            path_masks[path] = path_masks.get(path, 0) | bit

    comparable_masks: dict[str, int] = {}
    non_junit_masks: dict[str, int] = {}
    for test, paths in selection.matches_by_test.items():
        mask = 0
        for path in paths:
            mask |= path_masks.get(path, 0)
        if not mask:
            continue
        canonical = canonical_test_id(test)
        destination = (
            comparable_masks if canonical.startswith("test/") else non_junit_masks
        )
        _add_mask(destination, canonical, mask)

    unsupported_set = set(selection.unsupported)
    unmatched_set = set(selection.unmatched)
    unsupported = tuple(
        tuple(path for path in pr.changed_files if path in unsupported_set)
        for pr in prs
    )
    unmatched = tuple(
        tuple(path for path in pr.changed_files if path in unmatched_set) for pr in prs
    )
    return PredictionIndex(
        comparable_masks=comparable_masks,
        non_junit_masks=non_junit_masks,
        comparable_counts=_counts_for_masks(comparable_masks.values(), len(prs)),
        non_junit_counts=_counts_for_masks(non_junit_masks.values(), len(prs)),
        unsupported=unsupported,
        unmatched=unmatched,
    )


def comparison_metrics(
    actual_tests: Iterable[str], index: int, predictions: PredictionIndex
) -> dict[str, int | float | None]:
    actual = set(actual_tests)
    bit = 1 << index
    intersection = sum(
        1 for test in actual if predictions.comparable_masks.get(test, 0) & bit
    )
    comparable = predictions.comparable_counts[index]
    non_junit = predictions.non_junit_counts[index]
    selected = comparable + non_junit
    actual_count = len(actual)
    tests_reduced = actual_count - comparable
    reduction_percent = tests_reduced * 100.0 / actual_count if actual_count else None
    return {
        "actual_unique": actual_count,
        "selected_unique": selected,
        "selected_comparable_unique": comparable,
        "selected_non_junit_unique": non_junit,
        "intersection_unique": intersection,
        "actual_only_unique": actual_count - intersection,
        "selected_only_unique": selected - intersection,
        "selected_only_comparable_unique": comparable - intersection,
        "tests_reduced": tests_reduced,
        "reduction_percent": reduction_percent,
    }


def _base_record(
    pr: LandedPR, index: int, predictions: PredictionIndex
) -> dict[str, Any]:
    unsupported = list(predictions.unsupported[index])
    unmatched = list(predictions.unmatched[index])
    return {
        "pr_number": pr.number,
        "url": pr.url,
        "landing_commit": pr.landing_commit,
        "committed_at": pr.committed_at,
        "subject": pr.subject,
        "changed_files": list(pr.changed_files),
        "changed_file_count": len(pr.changed_files),
        "unsupported_files": unsupported,
        "unmatched_files": unmatched,
        "fully_covered_changes": not unsupported and not unmatched,
    }


def compare_pr(
    pr: LandedPR,
    index: int,
    predictions: PredictionIndex,
    job_filter: pr_tests.TestJobFilter | None = None,
) -> dict[str, Any]:
    base = _base_record(pr, index, predictions)
    try:
        collected = pr_tests.collect_pr_tests(str(pr.number), job_filter=job_filter)
    except pr_tests.PRTestsError as error:
        return {**base, "status": "error", "error": str(error)}
    return {
        **base,
        "status": "success",
        "error": None,
        "collection": {
            "workflow_runs": collected.workflow_runs,
            "artifacts": collected.artifacts,
            "reports": collected.reports,
        },
        "metrics": comparison_metrics(collected.tests, index, predictions),
    }


def _aggregate(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    metric_names = (
        "actual_unique",
        "selected_unique",
        "selected_comparable_unique",
        "selected_non_junit_unique",
        "intersection_unique",
        "actual_only_unique",
        "selected_only_unique",
        "selected_only_comparable_unique",
        "tests_reduced",
    )
    totals = {
        name: sum(int(record["metrics"][name]) for record in successful)
        for name in metric_names
    }
    actual = totals["actual_unique"]
    totals["reduction_percent"] = (
        totals["tests_reduced"] * 100.0 / actual if actual else None
    )
    return {"pr_count": len(successful), **totals}


def aggregate_results(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    fully_covered = [
        record for record in successful if record.get("fully_covered_changes") is True
    ]
    return {
        "requested_prs": len(records),
        "completed_prs": len(successful),
        "error_prs": len(records) - len(successful),
        "scope_incomplete_prs": sum(
            record.get("status") == "success"
            and record.get("fully_covered_changes") is not True
            for record in records
        ),
        "raw": _aggregate(successful),
        "fully_covered": _aggregate(fully_covered),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            while chunk := input_file.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ComparisonError(f"Unable to hash trace {path}: {error}") from error
    return digest.hexdigest()


def _configuration(
    trace_path: Path,
    trace_sha256: str,
    metadata: select_tests_from_td_tracer.TraceMetadata,
    main_ref: str,
    main_tip: str,
    limit: int,
    job_filter: pr_tests.TestJobFilter | None = None,
) -> dict[str, Any]:
    warnings = [
        "One trace snapshot is being applied to historical PR trees; results "
        "compare behavior and do not establish regression safety."
    ]
    if not metadata.usable:
        warnings.append(
            "TD tracer output is incomplete or unsuccessful; selections are observed lower bounds."
        )
    if len(metadata.revisions) != 1:
        warnings.append(
            "TD tracer output does not describe exactly one revision; "
            "historical results are provisional."
        )
    return {
        "trace": {
            "path": str(trace_path.resolve()),
            "sha256": trace_sha256,
            "schema_version": metadata.schema_version,
            "run_id": metadata.run_id,
            "complete": metadata.complete,
            "successful": metadata.successful,
            "usable": metadata.usable,
            "revisions": list(metadata.revisions),
        },
        "main_ref": main_ref,
        "main_tip": main_tip,
        "limit": limit,
        "collector_scope": (
            {"mode": "all-completed-actions-runs-for-pr-head"}
            if job_filter is None
            else {
                "mode": "filtered-test-jobs-for-pr-head",
                "build_environment": job_filter.build_environment,
                "test_configs": (
                    sorted(job_filter.test_configs)
                    if job_filter.test_configs is not None
                    else None
                ),
                "workflow_names": (
                    sorted(job_filter.workflow_names)
                    if job_filter.workflow_names is not None
                    else None
                ),
                "events": (
                    sorted(job_filter.events) if job_filter.events is not None else None
                ),
            }
        ),
        "provisional": True,
        "warnings": warnings,
    }


def _report(
    configuration: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    requested_prs: int,
    complete: bool,
) -> dict[str, Any]:
    summary = aggregate_results(records)
    summary["requested_prs"] = requested_prs
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "complete": complete,
        "configuration": dict(configuration),
        "summary": summary,
        "results": list(records),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    collection = record.get("collection", {})
    row = {field: record.get(field) for field in _CSV_FIELDS}
    row.update({field: metrics.get(field) for field in metrics})
    row.update({field: collection.get(field) for field in collection})
    for field in ("changed_files", "unsupported_files", "unmatched_files"):
        row[field] = json.dumps(record.get(field, []), separators=(",", ":"))
    return row


def _atomic_write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(_csv_row(record) for record in records)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_reusable_records(
    path: Path,
    configuration: Mapping[str, Any],
    prs: Sequence[LandedPR],
    resume: bool,
) -> dict[tuple[int, str], dict[str, Any]]:
    if not resume or not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as input_file:
            previous = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        print(
            f"warning: ignoring unreadable checkpoint {path}: {error}", file=sys.stderr
        )
        return {}
    if (
        not isinstance(previous, dict)
        or previous.get("schema_version") != REPORT_SCHEMA_VERSION
        or previous.get("configuration") != configuration
        or not isinstance(previous.get("results"), list)
    ):
        print(
            f"warning: checkpoint {path} is incompatible; starting over",
            file=sys.stderr,
        )
        return {}
    requested = {(pr.number, pr.landing_commit) for pr in prs}
    reusable: dict[tuple[int, str], dict[str, Any]] = {}
    for record in previous["results"]:
        if not isinstance(record, dict) or record.get("status") != "success":
            continue
        key = (record.get("pr_number"), record.get("landing_commit"))
        if key in requested:
            reusable[key] = record
    return reusable


def run_comparison(
    trace_path: Path,
    main_ref: str,
    limit: int,
    output_path: Path,
    csv_output: Path | None,
    workers: int,
    resume: bool,
    job_filter: pr_tests.TestJobFilter | None = None,
) -> tuple[dict[str, Any], int]:
    if workers < 1:
        raise ComparisonError("Worker count must be positive")
    trace_resolved = trace_path.resolve()
    output_resolved = output_path.resolve()
    csv_resolved = csv_output.resolve() if csv_output is not None else None
    if output_resolved == trace_resolved or csv_resolved == trace_resolved:
        raise ComparisonError("Report paths must not overwrite the TD tracer input")
    if csv_resolved is not None and csv_resolved == output_resolved:
        raise ComparisonError("JSON and CSV report paths must be different")
    main_tip, prs = enumerate_landed_prs(main_ref, limit)
    print(
        f"Selected {len(prs)} unique landed PRs from {main_ref} at {main_tip}.",
        file=sys.stderr,
    )
    affected = sorted({path for pr in prs for path in pr.changed_files})
    trace_sha256 = _sha256(trace_path)
    selection = select_tests_from_td_tracer.select_tests_with_details(
        trace_path, affected
    )
    predictions = build_prediction_index(prs, selection)
    configuration = _configuration(
        trace_path,
        trace_sha256,
        selection.metadata,
        main_ref,
        main_tip,
        limit,
        job_filter,
    )
    del selection
    for warning in configuration["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)

    reusable = _load_reusable_records(output_path, configuration, prs, resume)
    records_by_key = dict(reusable)
    pending = [
        (index, pr)
        for index, pr in enumerate(prs)
        if (pr.number, pr.landing_commit) not in reusable
    ]
    if reusable:
        print(f"Reusing {len(reusable)} completed PR comparisons.", file=sys.stderr)

    def ordered_records() -> list[dict[str, Any]]:
        return [
            records_by_key[(pr.number, pr.landing_commit)]
            for pr in prs
            if (pr.number, pr.landing_commit) in records_by_key
        ]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            executor.submit(compare_pr, pr, index, predictions, job_filter): (
                index,
                pr,
            )
            for index, pr in pending
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=len(reusable) + 1
        ):
            _, pr = futures[future]
            record = future.result()
            records_by_key[(pr.number, pr.landing_commit)] = record
            checkpoint_records = ordered_records()
            _atomic_write_json(
                output_path,
                _report(
                    configuration,
                    checkpoint_records,
                    requested_prs=len(prs),
                    complete=len(checkpoint_records) == len(prs),
                ),
            )
            detail = "ok" if record["status"] == "success" else record["error"]
            print(f"[{completed}/{len(prs)}] PR {pr.number}: {detail}", file=sys.stderr)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    records = ordered_records()
    report = _report(
        configuration,
        records,
        requested_prs=len(prs),
        complete=len(records) == len(prs),
    )
    _atomic_write_json(output_path, report)
    if csv_output is not None:
        _atomic_write_csv(csv_output, records)
    errors = report["summary"]["error_prs"]
    return report, 1 if errors else 0


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare historical PR tests with TD tracer selections."
    )
    parser.add_argument("trace", type=Path, help="TD tracer JSON file")
    parser.add_argument("--main-ref", default="origin/main")
    parser.add_argument("--limit", type=_positive_int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument(
        "--build-environment",
        help="Shell-style pattern for the test job build environment",
    )
    parser.add_argument("--test-config", dest="test_configs", action="append")
    parser.add_argument("--workflow-name", dest="workflow_names", action="append")
    parser.add_argument("--event", dest="events", action="append")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _job_filter_from_options(
    options: argparse.Namespace,
) -> pr_tests.TestJobFilter | None:
    qualifiers = options.test_configs or options.workflow_names or options.events
    if options.build_environment is None:
        if qualifiers:
            raise ComparisonError(
                "--build-environment is required when filtering test jobs"
            )
        return None
    return pr_tests.TestJobFilter(
        build_environment=options.build_environment,
        test_configs=(
            frozenset(options.test_configs)
            if options.test_configs is not None
            else None
        ),
        workflow_names=(
            frozenset(options.workflow_names)
            if options.workflow_names is not None
            else None
        ),
        events=frozenset(options.events) if options.events is not None else None,
    )


def _format_reduction(aggregate: Mapping[str, Any]) -> str:
    percent = aggregate["reduction_percent"]
    formatted = "n/a" if percent is None else f"{percent:.2f}%"
    return (
        f"{aggregate['pr_count']} PRs, {aggregate['actual_unique']} actual, "
        f"{aggregate['selected_comparable_unique']} comparable selected, "
        f"{aggregate['selected_non_junit_unique']} non-JUnit targets, "
        f"{formatted} reduction"
    )


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        job_filter = _job_filter_from_options(options)
        report, return_code = run_comparison(
            trace_path=options.trace,
            main_ref=options.main_ref,
            limit=options.limit,
            output_path=options.output,
            csv_output=options.csv_output,
            workers=options.workers,
            resume=not options.no_resume,
            job_filter=job_filter,
        )
    except (
        ComparisonError,
        OSError,
        select_tests_from_td_tracer.TDSelectionError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = report["summary"]
    print(f"Raw: {_format_reduction(summary['raw'])}")
    print(f"Fully covered: {_format_reduction(summary['fully_covered'])}")
    print(f"Report: {options.output}")
    if options.csv_output is not None:
        print(f"CSV: {options.csv_output}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
