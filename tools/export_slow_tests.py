#!/usr/bin/env python

import argparse
import json
import os
import statistics
from collections import defaultdict
from tools.stats_utils.s3_stat_parser import get_previous_reports_for_branch, Report, Version2Report
from typing import cast, DefaultDict, Dict, List

SLOW_TESTS_FILE = '.pytorch-slow-tests'
SLOW_TEST_CASE_THRESHOLD_SEC = 60.0


def get_test_case_times() -> Dict[str, float]:
    reports: List[Report] = get_previous_reports_for_branch('origin/viable/strict', "")
    # an entry will be like ("test_doc_examples (__main__.TestTypeHints)" -> [values]))
    test_names_to_times: DefaultDict[str, List[float]] = defaultdict(list)
    for report in reports:
        if report.get('format_version', 1) != 2:
            raise RuntimeError("S3 format currently handled is version 2 only")
        v2report = cast(Version2Report, report)
        for test_file in v2report['files'].values():
            for suitename, test_suite in test_file['suites'].items():
                for casename, test_case in test_suite['cases'].items():
                    # The below attaches a __main__ as that matches the format of test.__class__ in
                    # common_utils.py (where this data will be used), and also matches what the output
                    # of a running test would look like.
                    name = f'{casename} (__main__.{suitename})'
                    succeeded: bool = test_case['status'] is None
                    if succeeded:
                        test_names_to_times[name].append(test_case['seconds'])
    return {test_case: statistics.mean(times) for test_case, times in test_names_to_times.items()}


def filter_slow_tests(test_cases_dict: Dict[str, float]) -> Dict[str, float]:
    return {test_case: time for test_case, time in test_cases_dict.items() if time >= SLOW_TEST_CASE_THRESHOLD_SEC}


def export_slow_tests(filename: str) -> None:
    if os.path.exists(filename):
        print(f'Overwriting existent file: {filename}')
    with open(filename, 'w+') as file:
        slow_test_times: Dict[str, float] = filter_slow_tests(get_test_case_times())
        json.dump(slow_test_times, file, indent='    ', separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export a JSON of slow test cases in PyTorch unit test suite')
    parser.add_argument(
        '-f',
        '--filename',
        nargs='?',
        type=str,
        default=SLOW_TESTS_FILE,
        const=SLOW_TESTS_FILE,
        help='Specify a file path to dump slow test times from previous S3 stats. Default file path: .pytorch-slow-tests',
    )
    return parser.parse_args()


def main():
    options = parse_args()
    export_slow_tests(options.filename)


if __name__ == '__main__':
    main()
