#!/usr/bin/env python

import argparse
import os
import typing

try:
    import junitparser
except ImportError:
    raise ImportError(
        "junitparser not found, please install with 'pip install junitparser'"
    )

try:
    import rich
except ImportError:
    print("rich not found, for color output use 'pip install rich'")

def parse_junit_reports(path_to_reports: str) -> typing.List[junitparser.TestCase]:
    if not os.path.exists(path_to_reports):
        raise FileNotFoundError(f"Path '{path_to_reports}', not found")
    ret_xml = ""
    # Return early if the path provided is just a file
    if os.path.isfile(path_to_reports):
        ret_xml = junitparser.JUnitXml.fromfile(path_to_reports)
    elif os.path.isdir(path_to_reports):
        ret_xml = junitparser.JUnitXml()
        for root, _, files in os.walk(path_to_reports):
            for file in [f for f in files if f.endswith("xml")]:
                ret_xml += junitparser.JUnitXml.fromfile(os.path.join(root, file))
    return convert_junit_to_testcases(ret_xml)


def convert_junit_to_testcases(
    xml: typing.Union[junitparser.JUnitXml, junitparser.TestSuite, typing.Literal[""]]
) -> typing.List[junitparser.TestCase]:
    testcases = list()
    for item in xml:
        if isinstance(item, junitparser.TestSuite):
            testcases.extend(convert_junit_to_testcases(item))
        else:
            testcases.append(item)
    return testcases

def render_tests(
    testcases: typing.List[junitparser.TestCase],
) -> None:
    num_passed = 0
    num_skipped = 0
    num_failed = 0
    for testcase in testcases:
        if not testcase.result:
            num_passed += 1
            continue
        for result in testcase.result:
            if isinstance(result, junitparser.Error):
                icon = ":rotating_light: [white on red]ERROR[/white on red]:"
                num_failed += 1
            elif isinstance(result, junitparser.Failure):
                icon = ":x: [white on red]Failure[/white on red]:"
                num_failed += 1
            else:
                num_skipped += 1
                continue
            rich.print(f"{icon} [bold red]{testcase.classname}.{testcase.name}[/bold red]")
            print(f"{result.text}")
    rich.print(f":white_check_mark: {num_passed} [green]Passed[green]")
    rich.print(f":dash: {num_skipped} [grey]Skipped[grey]")
    rich.print(f":rotating_light: {num_failed} [grey]Failed[grey]")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Render xunit output for failed tests",
    )
    parser.add_argument(
        "report_path",
        help="Base xunit reports (single file or directory) to compare to",
    )
    return parser.parse_args()


def main():
    options = parse_args()
    testcases = parse_junit_reports(options.report_path)
    render_tests(testcases)


if __name__ == "__main__":
    main()
