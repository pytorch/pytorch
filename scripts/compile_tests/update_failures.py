#!/usr/bin/env python3
import argparse
import os
import pathlib
import subprocess

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
Usage: update_failures.py /path/to/dynamo_test_failures.py /path/to/test commit_sha

Best-effort updates the xfail and skip files under test directory
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


def patch_file(
    filename, test_dir, unexpected_successes, new_xfails, new_skips, unexpected_skips
):
    failures_directory = os.path.join(test_dir, "dynamo_expected_failures")
    skips_directory = os.path.join(test_dir, "dynamo_skips")

    dynamo_expected_failures = set(os.listdir(failures_directory))
    dynamo_skips = set(os.listdir(skips_directory))

    # These are hand written skips
    extra_dynamo_skips = set()
    with open(filename, "r") as f:
        start = False
        for text in f.readlines():
            text = text.strip()
            if start:
                if text == "}":
                    break
                extra_dynamo_skips.add(text.strip(',"'))
            else:
                if text == "extra_dynamo_skips = {":
                    start = True

    def format(testcase):
        classname = testcase.attrib["classname"]
        name = testcase.attrib["name"]
        return f"{classname}.{name}"

    formatted_unexpected_successes = {
        f"{format(test)}" for test in unexpected_successes.values()
    }
    formatted_unexpected_skips = {
        f"{format(test)}" for test in unexpected_skips.values()
    }
    formatted_new_xfails = [f"{format(test)}" for test in new_xfails.values()]
    formatted_new_skips = [f"{format(test)}" for test in new_skips.values()]

    def remove_file(path, name):
        file = os.path.join(path, name)
        cmd = ["git", "rm", file]
        subprocess.run(cmd)

    def add_file(path, name):
        file = os.path.join(path, name)
        with open(file, "w") as fp:
            pass
        cmd = ["git", "add", file]
        subprocess.run(cmd)

    covered_unexpected_successes = set()

    # dynamo_expected_failures
    for test in dynamo_expected_failures:
        if test in formatted_unexpected_successes:
            covered_unexpected_successes.add(test)
            remove_file(failures_directory, test)
    for test in formatted_new_xfails:
        add_file(failures_directory, test)

    leftover_unexpected_successes = (
        formatted_unexpected_successes - covered_unexpected_successes
    )
    if len(leftover_unexpected_successes) > 0:
        print(
            "WARNING: we were unable to remove these "
            f"{len(leftover_unexpected_successes)} expectedFailures:"
        )
        for stuff in leftover_unexpected_successes:
            print(stuff)

    # dynamo_skips
    for test in dynamo_skips:
        if test in formatted_unexpected_skips:
            remove_file(skips_directory, test)
    for test in extra_dynamo_skips:
        if test in formatted_unexpected_skips:
            print(
                f"WARNING: {test} in dynamo_test_failures.py needs to be removed manually"
            )
    for test in formatted_new_skips:
        add_file(skips_directory, test)


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


def update(filename, test_dir, py38_dir, py311_dir, also_remove_skips):
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
        filename, test_dir, unexpected_successes, xfails, all_skips, unexpected_skips
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
            / "torch/testing/_internal/dynamo_test_failures.py"
        ),
        help="Optional path to dynamo_test_failures.py",
    )
    # test path
    parser.add_argument(
        "test_dir",
        nargs="?",
        default=str(pathlib.Path(__file__).absolute().parent.parent.parent / "test"),
        help="Optional path to test folder",
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
    assert pathlib.Path(args.test_dir).exists(), args.test_dir
    dynamo38, dynamo311 = download_reports(args.commit, ("dynamo38", "dynamo311"))
    update(args.filename, args.test_dir, dynamo38, dynamo311, args.also_remove_skips)
