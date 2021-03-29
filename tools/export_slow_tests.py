#!/usr/bin/env python

import argparse
import json
import os
from tools.stats_utils.s3_stat_parser import get_previous_reports_for_branch, Report, Version2Report
from typing import cast, Dict, Tuple, List

SLOW_TESTS_FILE = '.pytorch-slow-tests'
SLOW_TEST_CASE_THRESHOLD = 60.0


def get_test_case_times() -> Dict[str, float]:
    # an entry will be like ("test_doc_examples (__main__.TestTypeHints)" -> (current_avg, # values))
    reports: List[Report] = get_previous_reports_for_branch('origin/viable/strict', "")
    test_names_to_times: Dict[str, Tuple[float, int]] = dict()
    for report in reports:
        if 'format_version' not in report:  # version 1 implicitly
            raise RuntimeError("S3 format currently handled is version 2 only")
        else:
            v2report = cast(Version2Report, report)
            for _, test_file in v2report['files'].items():
                for suitename, test_suite in test_file['suites'].items():
                    for casename, test_case in test_suite['cases'].items():
                        name = f'{casename} (__main__.{suitename})'
                        succeeded: bool = test_case['status'] is None
                        if succeeded:
                            if name not in test_names_to_times:
                                test_names_to_times[name] = (test_case['seconds'], 1)
                            else:
                                curr_avg, curr_count = test_names_to_times[name]
                                new_count = curr_count + 1
                                new_avg = (curr_avg * curr_count + test_case['seconds']) / new_count
                                test_names_to_times[name] = (new_avg, new_count)
    return {test_case: time for test_case, (time, _) in test_names_to_times.items()}


def filter_slow_tests(test_cases_dict: Dict[str, float]) -> Dict[str, float]:
    return {test_case: time for test_case, time in test_cases_dict.items() if time >= SLOW_TEST_CASE_THRESHOLD}


def export_slow_tests(filename: str) -> None:
    if os.path.exists(filename):
        print(f'Overwriting existent file: {filename}')
    with open(filename, 'w+') as file:
        slow_test_times: Dict[str, float] = filter_slow_tests(get_test_case_times())
        json.dump(slow_test_times, file)


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
