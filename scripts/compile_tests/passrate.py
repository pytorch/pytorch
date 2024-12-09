import argparse

from common import compute_pass_rate, get_testcases
from download_reports import download_reports


"""
Usage: passrate.py commit_sha

Parses test reports to measure the passrate. The passrate is defined as:

A) Take the number of tests that pass under eager mode, excluding
CUDA, OpInfo, and ModuleInfo tests
B) Of those tests, count the number of tests that pass under Dynamo
C) Take B/A.

You'll need to provide the commit_sha for a commit on the main branch,
from which we will pull CI test results.

This script requires the `gh` cli. You'll need to install it and then
authenticate with it via `gh auth login` before using this script.
https://docs.github.com/en/github-cli/github-cli/quickstart
"""


def testcases_by_time(xmls):
    testcases = get_testcases(xmls)
    testcases.sort(reverse=True, key=lambda x: float(x.attrib["time"]))
    return testcases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="passrate", description="Computes the Dynamo unittest pass rate"
    )
    parser.add_argument(
        "commit",
        help=(
            "The commit sha for the latest commit on a PR from which we will "
            "pull CI test results, e.g. 7e5f597aeeba30c390c05f7d316829b3798064a5"
        ),
    )
    args = parser.parse_args()
    dynamo311, eager311 = download_reports(args.commit, ("dynamo311", "eager311"))
    compute_pass_rate(eager311, dynamo311)
