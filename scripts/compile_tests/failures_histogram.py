import argparse
import re

from common import get_testcases, open_test_results, skipped_test


"""
python failures_histogram.py dynamo_logs_for_py311/

Analyzes skip reasons for Dynamo tests and prints a histogram with repro
commands. You'll need to download the test reports for the Dynamo shards
and put them under the specified directory.
"""


def skip_reason(testcase):
    for child in testcase.iter():
        if child.tag != "skipped":
            continue
        return child.attrib["message"]
    raise AssertionError("no message?")


IGNORED_REASONS = {
    # We don't run OpInfo tests under Dynamo
    "Policy: we don't run OpInfo tests w/ Dynamo",
    # We don't run ModuleInfo tests under Dynamo
    "Policy: we don't run ModuleInfo tests w/ Dynamo",
    # We don't run CUDA tests in CI (yet)
    "Excluded from CUDA tests",
    # We don't run CUDA tests in CI (yet)
    "CUDA not found",
    # We don't run CUDA tests in CI (yet)
    "Only runs on cuda",
    # We don't run slow tests in CI
    "test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test",
}


def skip_reason_normalized(testcase):
    for child in testcase.iter():
        if child.tag != "skipped":
            continue
        result = child.attrib["message"].split("\n")[0]
        result = result.split(">")[0]
        result = re.sub(r"0x\w+", "0xDEADBEEF", result)
        result = re.sub(r"MagicMock id='\d+'", "MagicMock id='0000000000'", result)
        result = re.sub(r"issues/\d", "issues/XXX", result)
        return result
    raise AssertionError("no message?")


def get_failures(xmls):
    testcases = get_testcases(xmls)
    skipped = [t for t in testcases if skipped_test(t)]
    skipped_dict = {}
    for s in skipped:
        reason = skip_reason_normalized(s)
        if reason in IGNORED_REASONS:
            continue
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


# e.g. "17c5f69852/dynamo"
def failures_histogram(directory):
    xmls = open_test_results(directory)
    dct = get_failures(xmls)

    a = [(x, y, repro(z[0])) for x, y, z in dct]

    for row in a:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="failures_histogram",
        description="See statistics about skipped Dynamo tests",
    )
    parser.add_argument("directory")
    args = parser.parse_args()
    failures_histogram(args.directory)
