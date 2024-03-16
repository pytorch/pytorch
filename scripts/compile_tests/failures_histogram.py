import argparse
import re

from common import download_reports, get_testcases, key, open_test_results, skipped_test

from passrate import compute_pass_rate


"""
python failures_histogram.py commit_sha

Analyzes skip reasons for Dynamo tests and prints a histogram with repro
commands. You'll need to provide the commit_sha for a commit on the main branch,
from which we will pull CI test results.

This script requires the `gh` cli. You'll need to install it and then
authenticate with it via `gh auth login` before using this script.
https://docs.github.com/en/github-cli/github-cli/quickstart
"""


def skip_reason(testcase):
    for child in testcase.iter():
        if child.tag != "skipped":
            continue
        return child.attrib["message"]
    raise AssertionError("no message?")


def skip_reason_normalized(testcase):
    for child in testcase.iter():
        if child.tag != "skipped":
            continue
        result = child.attrib["message"].split("\n")[0]
        result = result.split(">")[0]
        result = re.sub(r"0x\w+", "0xDEADBEEF", result)
        result = re.sub(r"MagicMock id='\d+'", "MagicMock id='0000000000'", result)
        result = re.sub(r"issues/\d+", "issues/XXX", result)
        result = re.sub(r"torch.Size\(\[.*\]\)", "torch.Size([...])", result)
        result = re.sub(
            r"Could not get qualified name for class '.*'",
            "Could not get qualified name for class",
            result,
        )
        return result
    raise AssertionError("no message?")


def get_failures(testcases):
    skipped = [t for t in testcases if skipped_test(t)]
    skipped_dict = {}
    for s in skipped:
        reason = skip_reason_normalized(s)
        if reason not in skipped_dict:
            skipped_dict[reason] = []
        skipped_dict[reason].append(s)
    result = []
    for s, v in skipped_dict.items():
        result.append((len(v), s, v))
    result.sort(reverse=True)
    return result


def repro(testcase):
    return f"PYTORCH_TEST_WITH_DYNAMO=1 pytest {testcase.attrib['file']} -v -k {testcase.attrib['name']}"


def all_tests(testcase):
    return f"{testcase.attrib['file']}::{testcase.attrib['classname']}.{testcase.attrib['name']}"


# e.g. "17c5f69852/eager", "17c5f69852/dynamo"
def failures_histogram(eager_dir, dynamo_dir, verbose=False, format_issues=False):
    fail_keys = compute_pass_rate(eager_dir, dynamo_dir)
    xmls = open_test_results(dynamo_dir)

    testcases = get_testcases(xmls)
    testcases = [t for t in testcases if key(t) in fail_keys]
    dct = get_failures(testcases)

    result = []
    for count, reason, testcases in dct:
        if verbose:
            row = (
                count,
                reason,
                repro(testcases[0]),
                [all_tests(t) for t in testcases],
            )
        else:
            row = (count, reason, repro(testcases[0]))
        result.append(row)

    header = (
        "(num_failed_tests, error_msg, sample_test, all_tests)"
        if verbose
        else "(num_failed_tests, error_msg, sample_test)"
    )
    print(header)
    sum_counts = sum([r[0] for r in result])
    for row in result:
        if format_issues:
            print(as_issue(*row))
        else:
            print(row)
    print("[counts]", sum_counts)


def as_issue(count, msg, repro, tests):
    tests = "\n".join(tests)
    result = f"""
{'-' * 50}
{count} Dynamo test are failing with \"{msg}\".

## Repro

`{repro}`

You will need to remove the skip or expectedFailure before running the repro command.
This may be just removing a sentinel file from in
[dynamo_expected_failures](https://github.com/pytorch/pytorch/blob/main/test/dynamo_expected_failures)
or [dynamo_skips](https://github.com/pytorch/pytorch/blob/main/test/dynamo_skips).


## Failing tests

Here's a comprehensive list of tests that fail (as of this issue) with the above message:
<details>
<summary>Click me</summary>
```
{tests}
```
</details>
"""
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="failures_histogram",
        description="See statistics about skipped Dynamo tests",
    )
    parser.add_argument(
        "commit",
        help=(
            "The commit sha for the latest commit on a PR from which we will "
            "pull CI test results, e.g. 7e5f597aeeba30c390c05f7d316829b3798064a5"
        ),
    )
    parser.add_argument(
        "-v", "--verbose", help="Prints all failing test names", action="store_true"
    )
    parser.add_argument(
        "--format-issues",
        help="Prints histogram in a way that they can be copy-pasted as a github issues",
        action="store_true",
    )
    args = parser.parse_args()

    # args.format_issues implies verbose=True
    verbose = args.verbose
    if args.format_issues:
        verbose = True

    dynamo311, eager311 = download_reports(args.commit, ("dynamo311", "eager311"))
    failures_histogram(eager311, dynamo311, verbose, args.format_issues)
