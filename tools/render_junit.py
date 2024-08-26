#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from typing import Any


try:
    from junitparser import (  # type: ignore[import]
        Error,
        Failure,
        JUnitXml,
        TestCase,
        TestSuite,
    )
except ImportError as e:
    raise ImportError(
        "junitparser not found, please install with 'pip install junitparser'"
    ) from e

try:
    import rich
except ImportError:
    print("rich not found, for color output use 'pip install rich'")


def parse_junit_reports(path_to_reports: str) -> list[TestCase]:  # type: ignore[no-any-unimported]
    def parse_file(path: str) -> list[TestCase]:  # type: ignore[no-any-unimported]
        try:
            return convert_junit_to_testcases(JUnitXml.fromfile(path))
        except Exception as err:
            rich.print(
                f":Warning: [yellow]Warning[/yellow]: Failed to read {path}: {err}"
            )
            return []

    if not os.path.exists(path_to_reports):
        raise FileNotFoundError(f"Path '{path_to_reports}', not found")
    # Return early if the path provided is just a file
    if os.path.isfile(path_to_reports):
        return parse_file(path_to_reports)
    ret_xml = []
    if os.path.isdir(path_to_reports):
        for root, _, files in os.walk(path_to_reports):
            for fname in [f for f in files if f.endswith("xml")]:
                ret_xml += parse_file(os.path.join(root, fname))
    return ret_xml


def convert_junit_to_testcases(xml: JUnitXml | TestSuite) -> list[TestCase]:  # type: ignore[no-any-unimported]
    testcases = []
    for item in xml:
        if isinstance(item, TestSuite):
            testcases.extend(convert_junit_to_testcases(item))
        else:
            testcases.append(item)
    return testcases


def render_tests(testcases: list[TestCase]) -> None:  # type: ignore[no-any-unimported]
    num_passed = 0
    num_skipped = 0
    num_failed = 0
    for testcase in testcases:
        if not testcase.result:
            num_passed += 1
            continue
        for result in testcase.result:
            if isinstance(result, Error):
                icon = ":rotating_light: [white on red]ERROR[/white on red]:"
                num_failed += 1
            elif isinstance(result, Failure):
                icon = ":x: [white on red]Failure[/white on red]:"
                num_failed += 1
            else:
                num_skipped += 1
                continue
            rich.print(
                f"{icon} [bold red]{testcase.classname}.{testcase.name}[/bold red]"
            )
            print(f"{result.text}")
    rich.print(f":white_check_mark: {num_passed} [green]Passed[green]")
    rich.print(f":dash: {num_skipped} [grey]Skipped[grey]")
    rich.print(f":rotating_light: {num_failed} [grey]Failed[grey]")


def parse_args() -> Any:
    parser = argparse.ArgumentParser(
        description="Render xunit output for failed tests",
    )
    parser.add_argument(
        "report_path",
        help="Base xunit reports (single file or directory) to compare to",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    testcases = parse_junit_reports(options.report_path)
    render_tests(testcases)


if __name__ == "__main__":
    main()
