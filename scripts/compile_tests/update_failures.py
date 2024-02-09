#!/usr/bin/env python3
import argparse
import csv
import pathlib

from common import (
    download_reports,
    get_testcases,
    is_failure,
    is_passing_skipped_test,
    is_unexpected_success,
    key,
    open_test_results,
)

"""
Usage: update_failures.py /path/to/dynamo_test_failures.csv commit_sha

Best-effort updates the xfail and skip lists in dynamo_test_failures.csv
by parsing test reports.

You'll need to provide the commit_sha for the latest commit on a PR
from which we will pull CI test results.

Instructions:
- On your PR, add the "keep-going" label to ensure that all the tests are
  failing (as opposed to CI stopping on the first failure). You may need to
  restart your test jobs by force-pushing to your branch for CI to pick
  up the "keep-going" label.
- Wait for all the tests to finish running.
- Find the full SHA of your commit and run this command.

This script requires the `gh` cli. You'll need to install it and then
authenticate with it via `gh auth login` before using this script.
https://docs.github.com/en/github-cli/github-cli/quickstart
"""


def patch_file(filename, unexpected_successes, new_xfails, new_skips, unexpected_skips):
    header = None
    test_failures = {}
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # There should be 3 lines in between everything in dynamo_test_failures.csv!
            if i % 4 != 0:
                assert len(row) == 0
                continue
            if i == 0:
                header = row
                continue
            assert len(row) >= 2
            test_failures[row[0]] = row[1:]

    def as_key(testcase):
        classname = testcase.attrib["classname"]
        name = testcase.attrib["name"]
        return f"{classname}.{name}"

    # remove unexpected_successes
    for test in unexpected_successes.values():
        key = as_key(test)
        if key not in test_failures:
            print(
                f"WARNING: we were unable to remove {test} from the expected failures list"
            )
        assert test_failures[key][0] == "xfail"
        del test_failures[key]

    # add in new_xfails
    for test in new_xfails.values():
        key = as_key(test)
        assert key not in test_failures
        test_failures[key] = ["xfail", test.attrib["file"]]

    # add in new_skips
    for test in new_skips.values():
        key = as_key(test)
        assert key not in test_failures
        test_failures[key] = ["skip", test.attrib["file"]]

    # remove unexpected_skips
    for test in unexpected_skips.values():
        key = as_key(test)
        assert test_failures[key][0] == "skip"
        del test_failures[key]

    # Write test_failures out to disk
    with open(filename, "w") as f:
        writer = csv.writer(f)
        sorted_keys = sorted(list(test_failures.keys()))
        empty_rows = 3

        # write header
        writer.writerow(header)
        for _ in range(empty_rows):
            writer.writerow([])

        # write everything else
        for k in sorted_keys:
            v = test_failures[k]
            writer.writerow([k, *v])
            if k != sorted_keys[-1]:
                for _ in range(empty_rows):
                    writer.writerow([])


def get_intersection_and_outside(a_dict, b_dict):
    a = set(a_dict.keys())
    b = set(b_dict.keys())
    intersection = a.intersection(b)
    outside = (a.union(b)) - intersection

    def build_dict(keys):
        result = {}
        for k in keys:
            if k in a_dict:
                result[k] = a_dict[k]
            else:
                result[k] = b_dict[k]
        return result

    return build_dict(intersection), build_dict(outside)


def update(filename, py38_dir, py311_dir, also_remove_skips):
    def read_test_results(directory):
        xmls = open_test_results(directory)
        testcases = get_testcases(xmls)
        unexpected_successes = {
            key(test): test for test in testcases if is_unexpected_success(test)
        }
        failures = {key(test): test for test in testcases if is_failure(test)}
        passing_skipped_tests = {
            key(test): test for test in testcases if is_passing_skipped_test(test)
        }
        return unexpected_successes, failures, passing_skipped_tests

    (
        py38_unexpected_successes,
        py38_failures,
        py38_passing_skipped_tests,
    ) = read_test_results(py38_dir)
    (
        py311_unexpected_successes,
        py311_failures,
        py311_passing_skipped_tests,
    ) = read_test_results(py311_dir)

    unexpected_successes = {**py38_unexpected_successes, **py311_unexpected_successes}
    _, skips = get_intersection_and_outside(
        py38_unexpected_successes, py311_unexpected_successes
    )
    xfails, more_skips = get_intersection_and_outside(py38_failures, py311_failures)
    if also_remove_skips:
        unexpected_skips, _ = get_intersection_and_outside(
            py38_passing_skipped_tests, py311_passing_skipped_tests
        )
    else:
        unexpected_skips = {}
    all_skips = {**skips, **more_skips}
    print(
        f"Discovered {len(unexpected_successes)} new unexpected successes, "
        f"{len(xfails)} new xfails, {len(all_skips)} new skips, {len(unexpected_skips)} new unexpected skips"
    )
    return patch_file(
        filename, unexpected_successes, xfails, all_skips, unexpected_skips
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="update_dynamo_test_failures",
        description="Read from logs and update the dynamo_test_failures file",
    )
    # dynamo_test_failures path
    parser.add_argument(
        "filename",
        nargs="?",
        default=str(
            pathlib.Path(__file__).absolute().parent.parent.parent
            / "torch/testing/_internal/dynamo_test_failures.csv"
        ),
        help="Optional path to dynamo_test_failures.csv",
    )
    parser.add_argument(
        "commit",
        help=(
            "The commit sha for the latest commit on a PR from which we will "
            "pull CI test results, e.g. 7e5f597aeeba30c390c05f7d316829b3798064a5"
        ),
    )
    parser.add_argument(
        "--also-remove-skips",
        help="Also attempt to remove skips. WARNING: does not guard against test flakiness",
        action="store_true",
    )
    args = parser.parse_args()
    assert pathlib.Path(args.filename).exists(), args.filename
    dynamo38, dynamo311 = download_reports(args.commit, ("dynamo38", "dynamo311"))
    update(args.filename, dynamo38, dynamo311, args.also_remove_skips)
    update(args.filename, None, None, args.also_remove_skips)
