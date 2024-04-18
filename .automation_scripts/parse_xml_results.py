""" The Python PyTorch testing script.
##
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Tuple

# Backends list
BACKENDS_LIST = [
    "dist-gloo",
    "dist-nccl"
]

TARGET_WORKFLOW = "--rerun-disabled-tests"

def get_job_id(report: Path) -> int:
    # [Job id in artifacts]
    # Retrieve the job id from the report path. In our GHA workflows, we append
    # the job id to the end of the report name, so `report` looks like:
    #     unzipped-test-reports-foo_5596745227/test/test-reports/foo/TEST-foo.xml
    # and we want to get `5596745227` out of it.
    try:
        return int(report.parts[0].rpartition("_")[2])
    except ValueError:
        return -1

def is_rerun_disabled_tests(root: ET.ElementTree) -> bool:
    """
    Check if the test report is coming from rerun_disabled_tests workflow
    """
    skipped = root.find(".//*skipped")
    # Need to check against None here, if not skipped doesn't work as expected
    if skipped is None:
        return False

    message = skipped.attrib.get("message", "")
    return TARGET_WORKFLOW in message or "num_red" in message

def parse_xml_report(
    tag: str,
    report: Path,
    workflow_id: int,
    workflow_run_attempt: int,
    work_flow_name: str
) -> Dict[Tuple[str], Dict[str, Any]]:
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    print(f"Parsing {tag}s for test report: {report}")

    job_id = get_job_id(report)
    print(f"Found job id: {job_id}")

    test_cases: Dict[Tuple[str], Dict[str, Any]] = {}

    root = ET.parse(report)
    # TODO: unlike unittest, pytest-flakefinder used by rerun disabled tests for test_ops
    # includes skipped messages multiple times (50 times by default). This slows down
    # this script too much (O(n)) because it tries to gather all the stats. This should
    # be fixed later in the way we use pytest-flakefinder. A zipped test report from rerun
    # disabled test is only few MB, but will balloon up to a much bigger XML file after
    # extracting from a dozen to few hundred MB
    if is_rerun_disabled_tests(root):
        return test_cases

    for test_case in root.iter(tag):
        case = process_xml_element(test_case)
        if tag == 'testcase':
            case["workflow_id"] = workflow_id
            case["workflow_run_attempt"] = workflow_run_attempt
            case["job_id"] = job_id
            case["work_flow_name"] = work_flow_name

            # [invoking file]
            # The name of the file that the test is located in is not necessarily
            # the same as the name of the file that invoked the test.
            # For example, `test_jit.py` calls into multiple other test files (e.g.
            # jit/test_dce.py). For sharding/test selection purposes, we want to
            # record the file that invoked the test.
            #
            # To do this, we leverage an implementation detail of how we write out
            # tests (https://bit.ly/3ajEV1M), which is that reports are created
            # under a folder with the same name as the invoking file.
            case_name = report.parent.name
            for ind in range(len(BACKENDS_LIST)):
                if BACKENDS_LIST[ind] in report.parts:
                    case_name = case_name + "_" + BACKENDS_LIST[ind]
                    break
            case["invoking_file"] = case_name
            test_cases[ ( case["invoking_file"], case["classname"], case["name"], case["work_flow_name"] ) ] = case
        elif tag == 'testsuite':
            case["work_flow_name"] = work_flow_name
            case["invoking_xml"] = report.name
            case["running_time_xml"] = case["time"]
            case_name = report.parent.name
            for ind in range(len(BACKENDS_LIST)):
                if BACKENDS_LIST[ind] in report.parts:
                    case_name = case_name + "_" + BACKENDS_LIST[ind]
                    break
            case["invoking_file"] = case_name

            test_cases[ ( case["invoking_file"], case["invoking_xml"], case["work_flow_name"] ) ] = case

    return test_cases

def process_xml_element(element: ET.Element) -> Dict[str, Any]:
    """Convert a test suite element into a JSON-serializable dict."""
    ret: Dict[str, Any] = {}

    # Convert attributes directly into dict elements.
    # e.g.
    #     <testcase name="test_foo" classname="test_bar"></testcase>
    # becomes:
    #     {"name": "test_foo", "classname": "test_bar"}
    ret.update(element.attrib)

    # The XML format encodes all values as strings. Convert to ints/floats if
    # possible to make aggregation possible in Rockset.
    for k, v in ret.items():
        try:
            ret[k] = int(v)
        except ValueError:
            pass
        try:
            ret[k] = float(v)
        except ValueError:
            pass

    # Convert inner and outer text into special dict elements.
    # e.g.
    #     <testcase>my_inner_text</testcase> my_tail
    # becomes:
    #     {"text": "my_inner_text", "tail": " my_tail"}
    if element.text and element.text.strip():
        ret["text"] = element.text
    if element.tail and element.tail.strip():
        ret["tail"] = element.tail

    # Convert child elements recursively, placing them at a key:
    # e.g.
    #     <testcase>
    #       <foo>hello</foo>
    #       <foo>world</foo>
    #       <bar>another</bar>
    #     </testcase>
    # becomes
    #    {
    #       "foo": [{"text": "hello"}, {"text": "world"}],
    #       "bar": {"text": "another"}
    #    }
    for child in element:
        if child.tag not in ret:
            ret[child.tag] = process_xml_element(child)
        else:
            # If there are multiple tags with the same name, they should be
            # coalesced into a list.
            if not isinstance(ret[child.tag], list):
                ret[child.tag] = [ret[child.tag]]
            ret[child.tag].append(process_xml_element(child))
    return ret