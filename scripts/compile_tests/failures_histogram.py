import argparse
import re

from common import get_testcases, key, open_test_results, skipped_test

from passrate import compute_pass_rate


"""
python failures_histogram.py eager_logs_for_py311/ dynamo_logs_for_py311/

Analyzes skip reasons for Dynamo tests and prints a histogram with repro
commands. You'll need to download the test reports for the Dynamo shards
and put them under the specified directory; ditto for the eager shards.
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
def failures_histogram(eager_dir, dynamo_dir, verbose=False):
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
        print(row)
    print("[counts]", sum_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="failures_histogram",
        description="See statistics about skipped Dynamo tests",
    )
    # linux-focal-py3.11-clang10 (default) Test Reports (xml) directory
    parser.add_argument("eager_dir")
    # linux-focal-py3.11-clang10 (dynamo) Test Reports (xml) directory
    parser.add_argument("dynamo_dir")
    parser.add_argument(
        "-v", "--verbose", help="Prints all failing test names", action="store_true"
    )
    args = parser.parse_args()
    failures_histogram(args.eager_dir, args.dynamo_dir, args.verbose)
