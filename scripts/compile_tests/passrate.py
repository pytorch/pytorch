import argparse

from common import (
    get_excluded_testcases,
    get_passed_testcases,
    get_testcases,
    key,
    open_test_results,
)

"""
Usage: passrate.py eager_test_reports_dir/ dynamo_test_reports_dir/

Parses test reports to measure the passrate. The passrate is defined as:

A) Take the number of tests that pass under eager mode, excluding
CUDA, OpInfo, and ModuleInfo tests
B) Of those tests, count the number of tests that pass under Dynamo
C) Take B/A.

Each directory should have the pytest test reports for their respective
configurations. You may find the test reports in the HUD:
- click on a commit
- find the desired job
- click on "show artifacts"
- get the "test report" zip
- unzip it into the right place

"""


def testcases_by_time(xmls):
    testcases = get_testcases(xmls)
    testcases.sort(reverse=True, key=lambda x: float(x.attrib["time"]))
    return testcases


def compute_pass_rate(eager_dir, dynamo_dir):
    print("parsing xmls")
    eager_xmls = open_test_results(eager_dir)
    dynamo_xmls = open_test_results(dynamo_dir)

    print("computing pass rate")
    eager_passed = get_passed_testcases(eager_xmls)
    dynamo_passed = get_passed_testcases(dynamo_xmls)
    dynamo_pass_keys = {key(testcase) for testcase in dynamo_passed}
    eager_pass_keys = {key(testcase) for testcase in eager_passed}
    excluded = [key(t) for t in get_excluded_testcases(dynamo_xmls)]
    eager_pass_keys = eager_pass_keys - set(excluded)

    subset = eager_pass_keys.intersection(dynamo_pass_keys)
    total_subset = len(subset)
    total_tests = len(eager_pass_keys)
    print("pass rate", total_subset / total_tests, total_subset, total_tests)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="passrate", description="Computes the Dynamo unittest pass rate"
    )
    # linux-focal-py3.11-clang10 (default) Test Reports (xml) directory
    parser.add_argument("eager_dir")
    # linux-focal-py3.8-clang10 (dynamo) Test Reports (xml) directory
    parser.add_argument("dynamo_dir")
    args = parser.parse_args()
    compute_pass_rate(args.eager_dir, args.dynamo_dir)
