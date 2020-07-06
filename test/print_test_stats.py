#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Read and print test results statistics
from xml.dom import minidom
from glob import glob
import os.path as osp
import datetime

class TestCase:
    def __init__(self, dom):
        self.class_name = str(dom.attributes['classname'].value)
        self.name = str(dom.attributes['name'].value)
        self.time = float(dom.attributes['time'].value)
        self.errored = len(dom.getElementsByTagName('error')) > 0
        self.failed = len(dom.getElementsByTagName('failure')) > 0
        self.skipped = len(dom.getElementsByTagName('skipped')) > 0


class TestSuite:
    def __init__(self, name):
        self.name = name
        self.test_cases = []
        self.failed_count = 0
        self.skipped_count = 0
        self.errored_count = 0
        self.total_time = 0.0

    def __repr__(self):
        rc = f'{self.name} run_time: {self.total_time:.2f} tests: {len(self.test_cases)}'
        if self.skipped_count > 0:
            rc += f' skipped: {self.skipped_count}'
        return f'TestSuite({rc})'

    def append(self, test_case):
        self.test_cases.append(test_case)
        self.total_time += test_case.time
        self.failed_count += 1 if test_case.failed else 0
        self.skipped_count += 1 if test_case.skipped else 0
        self.errored_count += 1 if test_case.errored else 0

    def print_report(self):
        sorted_tests = sorted(self.test_cases, key=lambda x: x.time)
        test_count = len(sorted_tests)
        print(f"class {self.name}:")
        print(f"    tests: {test_count} failed: {self.failed_count} skipped: {self.skipped_count} errored: {self.errored_count}")
        print(f"    run_time: {self.total_time:.2f} seconds")
        print(f"    avg_time: {self.total_time/test_count:.2f} seconds")
        if test_count > 2:
            print(f"    mean_time: {sorted_tests[test_count>>1].time:.2f} seconds")
            print("    Three longest tests:")
            for idx in [-1, -2, -3]:
                print(f"        {sorted_tests[idx].name} time: {sorted_tests[idx].time:.2f} seconds")
        elif test_count > 0:
            print("    Longest test:")
            print(f"        {sorted_tests[-1].name} time: {sorted_tests[-1].time:.2f} seconds")
        print("")



def parse_report(path):
    dom = minidom.parse(path)
    for test_case in dom.getElementsByTagName('testcase'):
        yield TestCase(test_case)

def parse_reports(folder):
    reports = glob(osp.join(folder, '**', '*.xml'), recursive=True)
    tests_by_class = dict()
    for report in reports:
        for test_case in parse_report(report):
            class_name = test_case.class_name
            if class_name not in tests_by_class:
                tests_by_class[class_name] = TestSuite(class_name)
            tests_by_class[class_name].append(test_case)
    return tests_by_class

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("Please specify test report folder")
        sys.exit(0)

    reports = parse_reports(sys.argv[1])
    if len(reports) == 0:
        print(f"No test reports found in {sys.argv[1]}")
        sys.exit(0)

    longest_tests = []
    total_time = 0
    for name in sorted(reports.keys()):
        test_suite = reports[name]
        test_suite.print_report()
        total_time += test_suite.total_time
        longest_tests.extend(test_suite.test_cases)
        if len(longest_tests) > 10:
            longest_tests = sorted(longest_tests, key=lambda x: x.time)[-10:]

    print(f"Total runtime is {datetime.timedelta(seconds=int(total_time))}")
    print("Ten longest tests of entire run:")
    for test_case in reversed(longest_tests):
        print(f"    {test_case.class_name}.{test_case.name}  time: {test_case.time:.2f} seconds")
