import functools
import os
import warnings


try:
    import lxml.etree

    p = lxml.etree.XMLParser(huge_tree=True)
    parse = functools.partial(lxml.etree.parse, parser=p)
except ImportError:
    import xml.etree.ElementTree as ET

    parse = ET.parse
    warnings.warn(
        "lxml was not found. `pip install lxml` to make this script run much faster"
    )


def open_test_results(directory):
    xmls = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                tree = parse(f"{root}/{file}")
                xmls.append(tree)
    return xmls


def get_testcases(xmls):
    testcases = []
    for xml in xmls:
        root = xml.getroot()
        testcases.extend(list(root.iter("testcase")))
    return testcases


def find(testcase, condition):
    children = list(testcase.iter())
    assert children[0] is testcase
    children = children[1:]
    return condition(children)


def skipped_test(testcase):
    def condition(children):
        return "skipped" in {child.tag for child in children}

    return find(testcase, condition)


def passed_test(testcase):
    def condition(children):
        if len(children) == 0:
            return True
        tags = {child.tag for child in children}
        return "skipped" not in tags and "failed" not in tags

    return find(testcase, condition)


def key(testcase):
    file = testcase.attrib.get("file", "UNKNOWN")
    classname = testcase.attrib["classname"]
    name = testcase.attrib["name"]
    return "::".join([file, classname, name])


def get_passed_testcases(xmls):
    testcases = get_testcases(xmls)
    passed_testcases = [testcase for testcase in testcases if passed_test(testcase)]
    return passed_testcases


def get_excluded_testcases(xmls):
    testcases = get_testcases(xmls)
    excluded_testcases = [t for t in testcases if excluded_testcase(t)]
    return excluded_testcases


def excluded_testcase(testcase):
    def condition(children):
        for child in children:
            if child.tag == "skipped":
                if "Policy: we don't run" in child.attrib["message"]:
                    return True
        return False

    return find(testcase, condition)


def is_unexpected_success(testcase):
    def condition(children):
        for child in children:
            if child.tag != "failure":
                continue
            is_unexpected_success = (
                "unexpected success" in child.attrib["message"].lower()
            )
            if is_unexpected_success:
                return True
        return False

    return find(testcase, condition)


MSG = "This test passed, maybe we can remove the skip from dynamo_test_failures.py"


def is_passing_skipped_test(testcase):
    def condition(children):
        for child in children:
            if child.tag != "skipped":
                continue
            has_passing_skipped_test_msg = MSG in child.attrib["message"]
            if has_passing_skipped_test_msg:
                return True
        return False

    return find(testcase, condition)


# NB: not an unexpected success
def is_failure(testcase):
    def condition(children):
        for child in children:
            if child.tag != "failure":
                continue
            is_unexpected_success = (
                "unexpected success" in child.attrib["message"].lower()
            )
            if not is_unexpected_success:
                return True
        return False

    return find(testcase, condition)


def should_exclude(key):
    test_file = key.split("::")[0]
    # C++ tests
    if test_file == "UNKNOWN":
        return True
    # Policy: "pass rate" does not include inductor, export, or dynamo tests.
    return test_file.startswith(("inductor/", "export/", "dynamo/"))


def compute_pass_rate(eager_dir, dynamo_dir):
    print("parsing xmls")
    eager_xmls = open_test_results(eager_dir)
    dynamo_xmls = open_test_results(dynamo_dir)

    print("computing pass rate")
    eager_passed = get_passed_testcases(eager_xmls)
    dynamo_passed = get_passed_testcases(dynamo_xmls)
    dynamo_pass_keys = {key(testcase) for testcase in dynamo_passed}
    dynamo_pass_keys = {key_ for key_ in dynamo_pass_keys if not should_exclude(key_)}
    tmp_eager_pass_keys = {key(testcase) for testcase in eager_passed}
    tmp_eager_pass_keys = {
        key_ for key_ in tmp_eager_pass_keys if not should_exclude(key_)
    }
    excluded = [key(t) for t in get_excluded_testcases(dynamo_xmls)]
    eager_pass_keys = tmp_eager_pass_keys - set(excluded)

    subset = eager_pass_keys.intersection(dynamo_pass_keys)
    print(f"pass_subset:{subset}")
    print(f"base_subset:{eager_pass_keys}")
    total_subset = len(subset)
    total_tests = len(eager_pass_keys)
    print("pass rate", total_subset / total_tests, total_subset, total_tests)

    dynamo_testcases = get_testcases(dynamo_xmls)
    tc = {key(t): t for t in dynamo_testcases}

    # Useful for debugging
    not_there_keys = set()
    for key_ in eager_pass_keys:
        if key_ not in tc:
            not_there_keys.add(key_)

    fail_keys = eager_pass_keys - subset
    return fail_keys


def compute_pass_rate_aot_eager_subclasses(control_dir, test_dir):
    print("parsing xmls")
    control_xmls = open_test_results(control_dir)
    test_xmls = open_test_results(test_dir)

    print("computing pass rate")
    control_passed = get_passed_testcases(control_xmls)
    print(f"CONTROL passed:{control_passed}")
    test_passed = get_passed_testcases(test_xmls)
    print(f"TEST passed:{test_passed}")
    test_pass_keys = {key(testcase) for testcase in test_passed}
    print(f"TEST test_pass_keys:{test_pass_keys}")
    # test_pass_keys = {key_ for key_ in test_pass_keys if not should_exclude(key_)}
    tmp_control_pass_keys = {key(testcase) for testcase in control_passed}
    print(f"CONTROL tmp_control_pass_keys:{tmp_control_pass_keys}")
    # tmp_control_pass_keys = {
    #     key_ for key_ in tmp_control_pass_keys if not should_exclude(key_)
    # }
    excluded = [key(t) for t in get_excluded_testcases(test_xmls)]
    print(f"EXCLUDED testcases:{excluded}")
    control_pass_keys = tmp_control_pass_keys - set(excluded)
    print(f"CONTROL PASS KEYS(without EXCLUDED):{control_pass_keys}")

    subset = control_pass_keys.intersection(test_pass_keys)
    print(f"pass_subset:{subset}")
    print(f"base_subset:{control_pass_keys}")
    total_subset = len(subset)
    total_tests = len(control_pass_keys)
    print("pass rate", total_subset / total_tests, total_subset, total_tests)

    test_testcases = get_testcases(test_xmls)
    tc = {key(t): t for t in test_testcases}

    # Useful for debugging
    not_there_keys = set()
    for key_ in control_pass_keys:
        if key_ not in tc:
            not_there_keys.add(key_)

    fail_keys = control_pass_keys - subset
    return fail_keys
