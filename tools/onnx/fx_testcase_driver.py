# Owner(s): ["module: onnx"]
"""Automate the updating of "skipFxTest" decorators for
"test_pytorch_onnx_onnxruntime.py" tests based on test results.

Usage: python tools/onnx/fx_testcase_decorator_helper.py -h
"""
from __future__ import annotations

import pytest
import io
import contextlib
import os
import logging
import re
from typing import Deque, Mapping, List, Optional
import collections
import difflib
import random
import argparse
import pathlib
import csv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_FX_TEST_CLASS_NAME = "TestONNXRuntime_opset_version_17_is_script_False_keep_initializers_as_inputs_False_is_fx_True"
_TEST_FILE_PATH = pathlib.Path("test", "onnx", "test_pytorch_onnx_onnxruntime.py")


class TestCaseNotFoundError(RuntimeError):
    pass


class TestCaseSegment:
    lines: Deque[str]
    _next: Optional[TestCaseSegment]
    _prev: Optional[TestCaseSegment]
    _test_case_name: str
    _indent: str

    def __init__(self, lines: List[str], prev: Optional[TestCaseSegment] = None):
        self.lines = collections.deque(lines)
        self._next = None
        self._prev = prev

        assert len(self.lines) > 0
        assert (match := re.match(r"(\s*)def (test_.*)\(", self.lines[0])) is not None
        self._indent, self._test_case_name = match.groups()
        log.debug("indent found: ", self._indent, self._test_case_name)

        if prev is not None:
            # FIXME: bug if the first test case has decorators.
            # Might consider making prefix lines a special TestCaseSegment.
            line_count_to_move = 0
            # Grab decorators from previous segment.
            for line in reversed(prev.lines):
                # Search til empty line.
                if line.strip():
                    self.lines.appendleft(line)
                    line_count_to_move += 1
                else:
                    break

            # Remove lines from previous segment.
            for _ in range(line_count_to_move):
                prev.lines.pop()

    def next(self) -> Optional[TestCaseSegment]:
        return self._next

    def prev(self) -> Optional[TestCaseSegment]:
        return self._prev

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"TestCaseSegment({self._test_case_name})",
                *self.lines,
                f"indent: ({self._indent})",
            ]
        )

    def __str__(self) -> str:
        return "\n".join(self.lines)

    @property
    def test_case_name(self) -> str:
        return self._test_case_name

    def xfail(self, reason: str):
        safe_reason = reason.replace('"', '\\"')
        # TODO(bowbao): how to not have flake8 and black complain about this line?
        # Can only apply one of these '# fmt: skip' or '# noqa: B950'
        self.lines.appendleft(
            f'{self._indent}@skipFxTest(reason="{safe_reason}")  # fmt: skip'
        )


class TestCasesSourceTree:
    prefix_lines: List[str]
    test_case_nodes: List[TestCaseSegment]
    test_case_index: Mapping[str, TestCaseSegment]

    def __init__(self, source_file_path: str):
        source_str = read_source_file(source_file_path)
        source_lines = source_str.splitlines()

        prev_segment: Optional[TestCaseSegment] = None
        current_segment: Optional[TestCaseSegment] = None

        test_case_func_indices = [
            i
            for i, line in enumerate(source_lines)
            if line.strip().startswith("def test_")
        ]

        if len(test_case_func_indices) == 0:
            raise RuntimeError("No test cases found in source file")

        self.prefix_lines = source_lines[: test_case_func_indices[0]]
        self.test_case_nodes = []

        for i, idx in enumerate(test_case_func_indices):
            if current_segment is not None:
                prev_segment = current_segment
            if i == len(test_case_func_indices) - 1:
                current_segment = TestCaseSegment(source_lines[idx:], prev_segment)
            else:
                current_segment = TestCaseSegment(
                    source_lines[idx : test_case_func_indices[i + 1]], prev_segment
                )
            if prev_segment is not None:
                prev_segment._next = current_segment

            self.test_case_nodes.append(current_segment)

        self.test_case_index = {
            node.test_case_name: node for node in self.test_case_nodes
        }

    def __str__(self) -> str:
        return (
            "\n".join(
                [*self.prefix_lines, *[str(node) for node in self.test_case_nodes]]
            )
            + "\n"
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Prefix: ",
                *self.prefix_lines,
                "Test Cases: ",
                *[repr(node) for node in self.test_case_nodes],
            ]
        )

    def xfail(self, test_case_name: str, reason: str) -> None:
        if (node := self.test_case_index.get(test_case_name, None)) is not None:
            node.xfail(reason)
        else:
            raise TestCaseNotFoundError(test_case_name)


class ErrorAggregator(object):
    """
    Collect and group error messages for report at the end

    Copied and modified from pytorch-jit-paritybench/paritybench/reporting.py
    """

    def __init__(self, context=None, log=None):
        super(ErrorAggregator, self).__init__()
        self.context = context or ""
        self.error_groups = []
        self.bigram_to_group_ids = collections.defaultdict(list)
        self.log = log or logging.getLogger(__name__)

    def record(self, e: str, module):
        # NOTE: Hack since we don't have actually `Exception` object, just the message.
        error_msg = e
        full_msg = e
        return self._add(error_msg, [(error_msg, f"{self.context}:{module}", full_msg)])

    def update(self, other):
        for errors in other.error_groups:
            self._add(errors[0][0], errors)

    def _add(self, error_msg: str, errors: List):
        msg_words = list(re.findall(r"[a-zA-Z]+", error_msg))
        if "NameError" in error_msg or "call_function" in error_msg:
            msg_bigrams = [error_msg]  # need exact match
        else:
            msg_bigrams = [
                f"{a}_{b}" for a, b in zip(msg_words, msg_words[1:])
            ] or msg_words

        shared_bigrams = collections.Counter()
        for bigram in msg_bigrams:
            shared_bigrams.update(self.bigram_to_group_ids[bigram])

        if shared_bigrams:
            best_match, count = shared_bigrams.most_common(1)[0]
            if count > len(msg_bigrams) // 2:
                self.error_groups[best_match].extend(errors)
                return False

        # No match, create a new error group
        group_id = len(self.error_groups)
        self.error_groups.append(errors)
        for bigram in msg_bigrams:
            self.bigram_to_group_ids[bigram].append(group_id)

        return True

    @staticmethod
    def format_error_group(errors):
        contexts = [
            context
            for context, _ in random.choices(
                list(
                    collections.Counter(context for msg, context, _ in errors).items()
                ),
                k=3,
            )
        ]
        return f"  - {len(errors)} errors like: {errors[0][0]} (examples {', '.join(contexts)})"

    def __str__(self):
        errors = sorted(self.error_groups, key=len, reverse=True)
        return "\n".join(map(self.format_error_group, errors[:20]))

    def __len__(self):
        return sum(map(len, self.error_groups))

    csv_headers = [
        "phase",
        "count",
        "example_short",
        "example_long",
        "example_from1",
        "example_from2",
    ]

    def write_csv(self, phase, out: csv.writer):
        for errors in sorted(self.error_groups, key=len, reverse=True)[:20]:
            short, context, long = random.choice(errors)
            if "#" in context:
                context1, _, context2 = context.partition(" # ")
            else:
                context1 = context
                context2 = context

            out.writerow([phase, len(errors), short, long, context1, context2])


class ErrorAggregatorDict(object):
    """
    Collect and group error messages for a debug report at the end

    Copied and modified from pytorch-jit-paritybench/paritybench/reporting.py
    """

    @classmethod
    def single(cls, name: str, e: Exception, context=None):
        errors = cls(context)
        errors.record(name, e, "global")
        return errors

    def __init__(self, context=None):
        super(ErrorAggregatorDict, self).__init__()
        self.aggregator = dict()
        self.context = context
        if context:
            self.name = re.sub(r"[.]zip$", "", os.path.basename(context))
        else:
            self.name = __name__

    def __getitem__(self, item):
        if item not in self.aggregator:
            self.aggregator[item] = ErrorAggregator(
                self.context, logging.getLogger(f"{item}.{self.name}")
            )
        return self.aggregator[item]

    def update(self, other):
        for key, value in other.aggregator.items():
            self[key].update(other=value)

    def print_report(self):
        for name in sorted(list(self.aggregator.keys())):
            self[name].log.info(
                f"\nTop errors in {name} ({len(self[name])} total):\n{self[name]}\n"
            )

    def write_csv(self, csv_file_path: pathlib.Path):
        with open(csv_file_path, "w") as fd:
            out = csv.writer(fd)
            out.writerow(ErrorAggregator.csv_headers)
            for name in sorted(list(self.aggregator.keys())):
                self[name].write_csv(name, out)

    def record(self, error_type, error, module=None):
        module = str(getattr(module, "__name__", module))
        if self[error_type].record(error, module):
            log.exception(f"{error_type} error from {self.context}:{module}")


class Report:
    pytest_summary_line: str
    error_aggregator: ErrorAggregatorDict

    def __init__(self, pytest_summary: str) -> None:
        self.error_aggregator = ErrorAggregatorDict()
        self.pytest_summary_line = pytest_summary.splitlines()[-1]

    def print_report(self) -> None:
        # Aggregated error report of new errors and new passes.
        self.error_aggregator.print_report()

        # Only grab the last line of the pytest summary
        # E.g., === x failed, y passed, z skipped ===
        log.info(self.pytest_summary_line)

    def write_csv(self, csv_file_path: pathlib.Path) -> None:
        self.error_aggregator.write_csv(csv_file_path)

    def record_error(self, error_type: str, error: str, test_case: str) -> None:
        self.error_aggregator.record(error_type, error, test_case)


def read_source_file(source_file_path: str) -> str:
    with open(source_file_path, "r") as f:
        return f.read()


def run(
    save_diff_path: pathlib.Path,
    save_report_csv_path: pathlib.Path,
    full_summary: bool,
    auto_apply: bool,
) -> None:

    source_tree = TestCasesSourceTree(str(_TEST_FILE_PATH))

    pytest_options = [str(_TEST_FILE_PATH), "-k", _FX_TEST_CLASS_NAME, "-n", "auto"]
    if full_summary:
        # TODO: ignore skips. This needs conftest.py.
        # pytest_options.append("--runxfail")
        pass

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        retcode = pytest.main(pytest_options)
        log.debug(f"Pytest returned {retcode}")
    test_output_str = f.getvalue()
    log.debug(test_output_str)

    summary_str = test_output_str[test_output_str.rfind("short test summary info") :]
    report = Report(summary_str)

    failure_summary_lines = [
        line for line in summary_str.splitlines() if line.startswith("FAILED")
    ]
    log.debug("Parsed failures")
    log.debug(failure_summary_lines)

    for line in failure_summary_lines:
        # TODO: support the other way around, xfail that should pass.
        matched = re.match(
            f"FAILED {_TEST_FILE_PATH}::{_FX_TEST_CLASS_NAME}::(test_[a-zA-Z0-9_]+) - (.*)",
            line,
        )

        if matched is None:
            report.record_error(
                "TOOL_ERROR", f"Failed to parse failure line: {line}", "global"
            )
            continue
        test_name, failure_reason = matched.groups()

        try:
            source_tree.xfail(test_name, failure_reason)
            report.record_error("TestFailure", failure_reason, test_name)
        except TestCaseNotFoundError:
            log.debug(
                f"Test case not found (possibly parameterized): {test_name} - {failure_reason}"
            )
            report.record_error(
                "TOOL_ERROR",
                f"Test case not found (possibly parameterized): {test_name} - {failure_reason}",
                test_name,
            )

    diff = "".join(
        difflib.unified_diff(
            read_source_file(_TEST_FILE_PATH).splitlines(keepends=True),
            str(source_tree).splitlines(keepends=True),
            fromfile=str(pathlib.Path("a", _TEST_FILE_PATH)),
            tofile=str(pathlib.Path("b", _TEST_FILE_PATH)),
        )
    )

    with open(f"{pathlib.Path(save_diff_path, _TEST_FILE_PATH.name)}.diff", "w") as f:
        f.write(diff)

    report.print_report()

    if not full_summary and auto_apply:
        with open(_TEST_FILE_PATH, "w") as f:
            f.write(str(source_tree))

    report.write_csv(pathlib.Path(save_report_csv_path, "report.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Driver tool to provide test report summary, "
        "as well as automatically mark failing tests as xfail by generating a diff file."
    )
    parser.add_argument(
        "--save-diff-path",
        type=str,
        help=f"Path to save the diff file. default: {_TEST_FILE_PATH.parent}",
        default=_TEST_FILE_PATH.parent,
    )
    parser.add_argument(
        "--save-report-csv-path",
        type=str,
        help=f"Path to save the report csv file. default: {_TEST_FILE_PATH.parent}",
        default=_TEST_FILE_PATH.parent,
    )
    parser.add_argument(
        "--full-summary",
        action="store_true",
        help="Report the results of all tests. Ignore skip decorators as if they were not applied. No diff will be generated.",
    )
    parser.add_argument(
        "--auto-apply",
        "-a",
        action="store_true",
        help="Auto apply the diff. This will overwrite the test file.",
    )

    args = parser.parse_args()

    run(
        args.save_diff_path,
        args.save_report_csv_path,
        args.full_summary,
        args.auto_apply,
    )
