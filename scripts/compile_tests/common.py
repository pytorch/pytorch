import functools
import os
import warnings
from dataclasses import dataclass
from typing import List


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


from enum import auto, Enum


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
    if tags <= {"testcase", "system-out", "system-err", "properties", "property"}:
        return TestCaseResult(TestCaseStatus.PASSED, k, testcase)

    if "skipped" in tags:
        return TestCaseResult(TestCaseStatus.SKIPPED, k, testcase)

    if "failure" in tags:
        return TestCaseResult(TestCaseStatus.FAILED, k, testcase)

    # TODO: handle rerun
    # print(f"XXX unmarked testcase tags:{tags}")


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
    total_errors = 0
    for ts in testsuites:
        total_tests += int(ts.attrib["tests"])
        total_failed += int(ts.attrib["failures"])
        total_skipped += int(ts.attrib["skipped"])
        total_errors += int(ts.attrib["errors"])
        unknown_attribs = set(ts.keys()) - {
            "tests",
            "failures",
            "skipped",
            "timestamp",
            "time",
            "name",
            "errors",
        }
        if unknown_attribs:
            # print(f"XXX testsuite {ts} unknown attribs:{unknown_attribs}")
            for attr in unknown_attribs:
                # print(f"XXX {attr}: {ts.attrib[attr]}")
                pass
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
        if v is None:
            continue
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

    print(f"     total_tests:{tc_total_tests:5} (h:{total_tests:5})")
    print(f"    total_failed:{tc_total_failed:5} (h:{total_failed:5})")
    print(f"   total_skipped:{tc_total_skipped:5} (h:{total_skipped:5})")
    print(f"    total_passed:{tc_total_passed:5} (h:{total_passed:5})")

    return tcs


def compute_pass_rate_tcs(control_tcs, test_tcs, name=""):
    c_passed_t_passed = 0
    c_passed_t_notpassed = 0

    for k, tv in test_tcs.items():
        if tv.is_skipped():
            continue

        if (cv := control_tcs.get(k, None)) is None:
            continue

        if cv.is_skipped():
            continue

        if cv.is_passed():
            if tv.is_passed():
                c_passed_t_passed += 1
            else:
                c_passed_t_notpassed += 1

    num = c_passed_t_passed
    den = c_passed_t_passed + c_passed_t_notpassed
    pass_ratio = num / den
    print(f"PASS_RATIO_{name}:{pass_ratio:.3} ({num} / {den})")


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

        dk = k.split(":")[0]
        d[dk] = d.get(dk, 0) + 1

    sd = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    for k, v in sd.items():
        print(f"{v:4} PYTORCH_TEST_WITH_SUBCLASSES=1 python test/{k} -v")


@dataclass
class Failure:
    testcase_result: TestCaseResult
    msg: str
    text: str
    sim: int = -1

    @property
    def key(self) -> str:
        return self.testcase_result.key


@dataclass
class Category:
    items: List[Failure]


from fuzzywuzzy import fuzz


def similarity(x: Failure, y: Failure) -> int:  # [0, 100]
    r = fuzz.ratio(x.msg, y.msg)
    return r


SIMILARITY_THRESHOLD: int = 78


def categorize(categories, f: Failure):
    if not categories:
        categories.append(Category([f]))
        return

    max_similarity = 0
    max_similarity_cat = None
    for c in categories:
        s = similarity(f, c.items[0])
        if s > max_similarity:
            max_similarity = s
            max_similarity_cat = c

    if max_similarity > SIMILARITY_THRESHOLD:
        f.sim = max_similarity
        max_similarity_cat.items.append(f)
        return

    categories.append(Category([f]))


def report_control_pass_test_notpass(control_tcs, test_tcs, dirname):
    import os

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    categories: List[List[Failure]] = []

    i = 0
    num_tcs = len(test_tcs.items())
    progress_n = 10
    print_step = int(num_tcs / 1000 / progress_n) * 1000
    import time

    prev_time_s = time.time()
    progress_i = 0
    with open(f"{dirname}/report", "w") as file:
        for _i, (k, tv) in enumerate(test_tcs.items()):
            if _i % print_step == 0:
                ts = time.time()
                print(
                    f"[{progress_i}/{progress_n}]...processing... {_i}/{num_tcs} dur_s:{int(ts - prev_time_s)}"
                )
                prev_time_s = ts
                progress_i += 1

            if tv.is_skipped():
                continue

            if (cv := control_tcs.get(k, None)) is None:
                continue

            if cv.is_skipped():
                continue

            if cv.is_passed() and not tv.is_passed():
                file.write("-" * 40)
                tc = tv.testcase
                file.write(f"{i:5}: {k}\n")
                try:
                    failure = tc.find("failure")
                    if failure is None:
                        failure = tc.find("rerun")
                    msg = failure.attrib["message"]
                    file.write(f"Failure_message:{msg}\n")
                    text = failure.text

                    f = Failure(tv, msg, text)
                    categorize(categories, f)
                except Exception as e:
                    print(f"Exception on report prep for key:{k} e:{e}")
                    import traceback

                    print(traceback.format_exc())
                i += 1
    total_failures = i
    num_cats = len(categories)
    for i, c in enumerate(sorted(categories, key=lambda c: len(c.items), reverse=True)):
        print(
            f"CATEGORY {i}/{num_cats} len(items):{len(c.items):5} of {total_failures}"
        )
        with open(f"{dirname}/cat_{i}_N{len(c.items)}", "w") as file:
            for j, f in enumerate(c.items):
                file.write(f"Failure {j} sim:{f.sim} key:{f.key}\n")
                file.write(f"msg: {f.msg}\n")
                file.write(f"text: {f.text}\n")


def compute_pass_rate_aot_eager_subclasses(e_dir, dw_dir, ae_dir, sc_dir, name=""):
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

    print("*** computing pass rate DYNAMO vs AOT_EAGER")
    compute_pass_rate_tcs(tcs_dw, tcs_ae, "DYNAMO_vs_AOT_EAGER")

    print("*** computing pass rate AOT_EAGER vs SC")
    compute_pass_rate_tcs(tcs_ae, tcs_sc, "AOT_EAGER_vs_SUBCLASSES")

    print("*** Find passed in dynamo, missing in SC")
    find_control_passed_missing_in_test(tcs_dw, tcs_sc)

    # report_control_pass_test_notpass(
    #   tcs_ae,
    #   tcs_sc,
    #   f"test_pass_ae_notpass_sc_{name}"
    # )

    # report_control_pass_test_notpass(
    #   tcs_dw,
    #   tcs_ae,
    #   f"test_pass_dw_notpass_ae_{name}"
    # )
