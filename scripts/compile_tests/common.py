import functools
import os
import warnings
from dataclasses import dataclass


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

from enum import Enum, auto
class TestCaseStatus(Enum):
    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()

from typing import Any

@dataclass
class TestCaseResult:
    status: TestCaseStatus
    key: str
    testcase: Any

    def is_passed(self):
        return self.status == TestCaseStatus.PASSED

    def is_failed(self):
        return self.status == TestCaseStatus.FAILED

    def is_skipped(self):
        return self.status == TestCaseStatus.SKIPPED

def parse_testcase(testcase) -> TestCaseResult:
    children = list(testcase.iter())
    k = key(testcase)
    tags = {child.tag for child in children}
    #if len(tags) > 1:
    #    breakpoint()
    if tags <= {"testcase", "system-out", "system-err", "properties", "property"}:
        return TestCaseResult(TestCaseStatus.PASSED, k, testcase)

    if "skipped" in tags:
        return TestCaseResult(TestCaseStatus.SKIPPED, k, testcase)

    if "failure" in tags:
        return TestCaseResult(TestCaseStatus.FAILED, k, testcase)

    breakpoint()
    assert False


def get_testsuites(xmls):
    ret = []
    for xml in xmls:
        root = xml.getroot()
        ret.extend(list(root.iter("testsuite")))
    return ret


def parse_xmls(xmls):
    testsuites = get_testsuites(xmls)
    total_tests = 0
    total_skipped = 0
    total_failed = 0
    for ts in testsuites:
        total_tests += int(ts.attrib["tests"])
        total_failed += int(ts.attrib["failures"])
        total_skipped += int(ts.attrib["skipped"])
    total_passed = total_tests - total_skipped - total_failed

    # print("--- HEADER DATA ---")
    # print(f"total_tests:{total_tests}")
    # print(f"total_failed:{total_failed}")
    # print(f"total_skipped:{total_skipped}")
    # print(f"total_passed:{total_passed}")
    # print("=== === ===")

    testcases = get_testcases(xmls)

    tc_total_tests = 0
    tc_total_skipped = 0
    tc_total_failed = 0
    tc_total_passed = 0
    tcs = dict()
    for testcase in testcases:
        k = key(testcase)
        v = parse_testcase(testcase)
        tcs[k] = v
        tc_total_tests += 1
        if v.is_failed():
            tc_total_failed += 1
        elif v.is_skipped():
            tc_total_skipped += 1
        elif v.is_passed():
            tc_total_passed += 1
        else:
            assert False

    print(f"XXX   total_tests:{tc_total_tests:5} (h:{total_tests:5})")
    print(f"XXX  total_failed:{tc_total_failed:5} (h:{total_failed:5})")
    print(f"XXX total_skipped:{tc_total_skipped:5} (h:{total_skipped:5})")
    print(f"XXX  total_passed:{tc_total_passed:5} (h:{total_passed:5})")

    return tcs


def compute_pass_rate_tcs(control_tcs, test_tcs):
    # passed - passed
    # 0 - passed
    # passed - 0
    # passed - failed
    # failed - failed

    # number of tests, that passed or fail in both
    base_count = 0
    c_passed_t_failed = 0
    for k, tv in test_tcs.items():
        if tv.is_skipped():
            continue

        if (cv := control_tcs.get(k, None)) is None:
            continue

        if cv.is_skipped():
            continue

        if cv.is_passed() and tv.is_failed():
            base_count += 1
            c_passed_t_failed += 1
            print(f"CONTROL_PASSED_TEST_FAILED:{k}")
        elif cv.is_failed() and tv.is_passed():
            c_failed_t_passed +=1
            print(f"STRANGE! CONTROL_FAILED_TEST_PASSED:{k}")
        elif cv.is_failed() and tv.is_failed():
            c_failed_t_failed += 1
            print(f"BOTH FAILED:{k}")
        else:
            assert cv.is_passed() and tv.is_passed()
            base_count += 1

    print(f"XXX base_count:{base_count}")
    print(f"XXX passed_failed_count:{c_passed_t_failed}")

    print(f"Pass ratio:{c_passed_t_failed / base_count}")


def find_test_in_control_statuses(
    control_tcs,
    control_status: TestCaseStatus,
    test_tcs,
    test_status: TestCaseStatus
):
    for k, tv in test_tcs.items():
        if tv.status != test_status:
            continue

        if (cv := control_tcs.get(k, None)) is None:
            continue

        if cv.status != control_status:
            continue

        print(f"XXX TEST:{k} {tv.key} {tv.testcase} (control:{control_status} vs test_status:{test_status}")


def find_control_in_test_statuses(
    control_tcs,
    control_status: TestCaseStatus,
    test_tcs,
    test_status: TestCaseStatus
):
    for k, cv in control_tcs.items():
        if cv.status != control_status:
            continue

        if (tv := test_tcs.get(k, None)) is None:
            continue

        if tv.status != test_status:
            continue

        print(f"XXX TEST:{k} {tv.key} {tv.testcase} (control:{control_status} vs test_status:{test_status}")


def find_control_passed_missing_in_test(
    control_tcs,
    test_tcs,
):
    d = {}
    for k, cv in control_tcs.items():
        if not cv.is_passed():
            continue

        tv = test_tcs.get(k, None)

        if not (tv is None or tv.is_skipped()):
            continue

        dk = k.split(':')[0]
        d[dk] = d.get(dk, 0) + 1

    sd = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    for k, v in sd.items():
        print(f"XXX TEST_MISSING_IN_SC: test/{k} - {v}") 


def compute_pass_rate_aot_eager_subclasses(e_dir, dw_dir, ae_dir, sc_dir):
    print("parsing xmls")
    print(f"-- PARSE EAGER XMLS {e_dir}")
    e_xmls = open_test_results(e_dir)
    tcs_e = parse_xmls(e_xmls)
    
    print(f"-- PARSE DYNAMO_WRAPPED XMLS {dw_dir}")
    dw_xmls = open_test_results(dw_dir)
    tcs_dw = parse_xmls(dw_xmls)

    print(f"-- PARSE AOT_EAGER XMLS {ae_dir}")
    ae_xmls = open_test_results(ae_dir)
    tcs_ae = parse_xmls(ae_xmls)

    print(f"-- PARSE SC XMLS {sc_dir}")
    sc_xmls = open_test_results(sc_dir)
    tcs_sc = parse_xmls(sc_xmls)
    print("===")

    find_control_passed_missing_in_test(tcs_dw, tcs_sc)

    # find_test_in_control_statuses(tcs_dw, TestCaseStatus.PASSED, tcs_sc, TestCaseStatus.SKIPPED)
    # find_control_in_test_statuses(tcs_dw, TestCaseStatus.PASSED, tcs_sc, TestCaseStatus.SKIPPED)

    # print("computing pass rate EAGER vs AOT_EAGER")
    # compute_pass_rate_tcs(tcs_e, tcs_ae)

    # print("computing pass rate AOT_EAGER vs SC")
    # compute_pass_rate_tcs(tcs_ae, tcs_sc)


    # test_testcases = get_testcases(test_xmls)
    # tc = {key(t): t for t in test_testcases}

    # # Useful for debugging
    # not_there_keys = set()
    # for key_ in control_pass_keys:
    #     if key_ not in tc:
    #         not_there_keys.add(key_)

    # fail_keys = control_pass_keys - subset
    # return fail_keys
