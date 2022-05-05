#!/usr/bin/env python3

import argparse
import json
import os
import statistics
from collections import defaultdict
from tools.stats.s3_stat_parser import (
    get_previous_reports_for_branch,
    Report,
    Version2Report,
)
from typing import cast, DefaultDict, Dict, List, Any
from urllib.request import urlopen

SLOW_TESTS_FILE = ".pytorch-slow-tests.json"
SLOW_TEST_CASE_THRESHOLD_SEC = 60.0
RELATIVE_DIFFERENCE_THRESHOLD = 0.1
IGNORED_JOBS = ["asan", "periodic"]


def get_test_case_times() -> Dict[str, float]:
    reports: List[Report] = get_previous_reports_for_branch("origin/viable/strict", "")
    # an entry will be like ("test_doc_examples (__main__.TestTypeHints)" -> [values]))
    test_names_to_times: DefaultDict[str, List[float]] = defaultdict(list)
    for report in reports:
        if report.get("format_version", 1) != 2:  # type: ignore[misc]
            raise RuntimeError("S3 format currently handled is version 2 only")
        v2report = cast(Version2Report, report)

        if any(job_name in str(report["build_job"]) for job_name in IGNORED_JOBS):
            continue

        for test_file in v2report["files"].values():
            for suitename, test_suite in test_file["suites"].items():
                for casename, test_case in test_suite["cases"].items():
                    # The below attaches a __main__ as that matches the format of test.__class__ in
                    # common_utils.py (where this data will be used), and also matches what the output
                    # of a running test would look like.
                    name = f"{casename} (__main__.{suitename})"
                    succeeded: bool = test_case["status"] is None
                    if succeeded:
                        test_names_to_times[name].append(test_case["seconds"])
    return {
        test_case: statistics.mean(times)
        for test_case, times in test_names_to_times.items()
    }


def filter_slow_tests(test_cases_dict: Dict[str, float]) -> Dict[str, float]:
    return {
        test_case: time
        for test_case, time in test_cases_dict.items()
        if time >= SLOW_TEST_CASE_THRESHOLD_SEC
    }


def get_test_infra_slow_tests() -> Dict[str, float]:
    url = "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/slow-tests.json"
    contents = urlopen(url, timeout=1).read().decode("utf-8")
    return cast(Dict[str, float], json.loads(contents))


def too_similar(
    calculated_times: Dict[str, float], other_times: Dict[str, float], threshold: float
) -> bool:
    # check that their keys are the same
    if calculated_times.keys() != other_times.keys():
        return False

    for test_case, test_time in calculated_times.items():
        other_test_time = other_times[test_case]
        relative_difference = abs(
            (other_test_time - test_time) / max(other_test_time, test_time)
        )
        if relative_difference > threshold:
            return False
    return True


def export_slow_tests(options: Any) -> None:
    filename = options.filename
    if os.path.exists(filename):
        print(f"Overwriting existent file: {filename}")
    with open(filename, "w+") as file:
        slow_test_times: Dict[str, float] = filter_slow_tests(get_test_case_times())
        if options.ignore_small_diffs:
            test_infra_slow_tests_dict = get_test_infra_slow_tests()
            if too_similar(
                slow_test_times, test_infra_slow_tests_dict, options.ignore_small_diffs
            ):
                slow_test_times = test_infra_slow_tests_dict
        json.dump(
            slow_test_times, file, indent="    ", separators=(",", ": "), sort_keys=True
        )
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a JSON of slow test cases in PyTorch unit test suite"
    )
    parser.add_argument(
        "-f",
        "--filename",
        nargs="?",
        type=str,
        default=SLOW_TESTS_FILE,
        const=SLOW_TESTS_FILE,
        help="Specify a file path to dump slow test times from previous S3 stats. Default file path: .pytorch-slow-tests.json",
    )
    parser.add_argument(
        "--ignore-small-diffs",
        nargs="?",
        type=float,
        const=RELATIVE_DIFFERENCE_THRESHOLD,
        help="Compares generated results with stats/slow-tests.json in pytorch/test-infra. If the relative differences "
        "between test times for each test are smaller than the threshold and the set of test cases have not "
        "changed, we will export the stats already in stats/slow-tests.json. Else, we will export the calculated "
        "results. The default threshold is 10%.",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    export_slow_tests(options)


if __name__ == "__main__":
    main()
