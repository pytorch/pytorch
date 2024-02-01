import argparse

from common import (
    get_testcases,
    is_failure,
    is_unexpected_success,
    key,
    open_test_results,
)

"""
Usage: update_failures.py /path/to/dynamo_test_failures.py py38_test_reports_dir/ py311_test_reports_dir/

Best-effort updates the xfail and skip lists in dynamo_test_failures.py
by parsing test reports.

Instructions:
- On your PR, add the "keep-going" label to ensure that all the tests are
  failing (as opposed to CI stopping on the first failure). You may need to
  restart your test jobs by force-pushing to your branch for CI to pick
  up the "keep-going" label.
- Create py38_test_reports_dir/ and py311_test_reports_dir/ directories.
- Now, download all test reports for the dynamo test jobs. Unzip the test
  reports for Dynamo py3.8 to py38_test_reports_dir/ and
  the reports for Dynamo py3.11 to py311_test_reports_dir/
- You can find the test reports in the HUD, by:
  1) clicking on a commit
     (e.g. https://hud.pytorch.org/pr/pytorch/pytorch/116071#20830570884),
  2) scrolling down to "pull - Job Status", finding the job
     (e.g. linux-focal-py3.11-clang10 / test (dynamo, 1, 3)),
  3) click on "show artifacts"
  4) downloading the test report zip

"""


def patch_file(filename, unexpected_successes, new_xfails, new_skips):
    with open(filename, "r") as f:
        text = f.readlines()

    new_text = []

    i = 0
    while True:
        line = text[i]
        if line.startswith("dynamo_expected_failures"):
            break
        new_text.append(line)
        i += 1

    def format(testcase):
        classname = testcase.attrib["classname"]
        name = testcase.attrib["name"]
        return f"{classname}.{name}"

    formatted_unexpected_successes = {
        f"{format(test)}" for test in unexpected_successes.values()
    }
    formatted_new_xfails = [
        f'    "{format(test)}",  # {test.attrib["file"]}\n'
        for test in new_xfails.values()
    ]
    formatted_new_skips = [
        f'    "{format(test)}",  # {test.attrib["file"]}\n'
        for test in new_skips.values()
    ]

    def in_unexpected_successes(line):
        splits = line.split('"')
        if len(splits) < 3:
            return None
        test_name = splits[1]
        if test_name in formatted_unexpected_successes:
            return test_name
        return None

    covered_unexpected_successes = set({})

    # dynamo_expected_failures
    while True:
        line = text[i]
        match = in_unexpected_successes(line)
        if match is not None:
            covered_unexpected_successes.add(match)
            i += 1
            continue
        if line == "}\n":
            new_text.extend(formatted_new_xfails)
            new_text.append(line)
            i += 1
            break
        new_text.append(line)
        i += 1

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
    while True:
        line = text[i]
        if line == "}\n":
            new_text.extend(formatted_new_skips)
            break
        if line == "dynamo_skips = {}\n":
            new_text.extend("dynamo_skips = {\n")
            new_text.extend(new_skips)
            new_text.extend("}\n")
            i += 1
            break
        new_text.append(line)
        i += 1

    for j in range(i, len(text)):
        new_text.append(text[j])

    with open(filename, "w") as f:
        f.writelines(new_text)


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


def update(filename, py38_dir, py311_dir):
    def read_test_results(directory):
        xmls = open_test_results(directory)
        testcases = get_testcases(xmls)
        unexpected_successes = {
            key(test): test for test in testcases if is_unexpected_success(test)
        }
        failures = {key(test): test for test in testcases if is_failure(test)}
        return unexpected_successes, failures

    py38_unexpected_successes, py38_failures = read_test_results(py38_dir)
    py311_unexpected_successes, py311_failures = read_test_results(py311_dir)

    unexpected_successes = {**py38_unexpected_successes, **py311_unexpected_successes}
    _, skips = get_intersection_and_outside(
        py38_unexpected_successes, py311_unexpected_successes
    )
    xfails, more_skips = get_intersection_and_outside(py38_failures, py311_failures)
    all_skips = {**skips, **more_skips}
    print(
        f"Discovered {len(unexpected_successes)} new unexpected successes, "
        f"{len(xfails)} new xfails, {len(all_skips)} new skips"
    )
    return patch_file(filename, unexpected_successes, xfails, all_skips)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="update_dynamo_test_failures",
        description="Read from logs and update the dynamo_test_failures file",
    )
    # dynamo_test_failures path
    parser.add_argument("filename")
    # linux-focal-py3.8-clang10 (dynamo) Test Reports (xml) directory
    parser.add_argument("py38_test_reports_dir")
    # linux-focal-py3.11-clang10 (dynamo) Test Reports (xml) directory
    parser.add_argument("py311_test_reports_dir")
    args = parser.parse_args()
    update(args.filename, args.py38_test_reports_dir, args.py311_test_reports_dir)
