# mypy: ignore-errors

import json
import sys
from pathlib import Path

from tools.linter.adapters.docstring_linter import (
    DocstringLinter,
    file_summary,
    make_recursive,
    make_terse,
)


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if _PARENT in _PATH:
    from linter_test_case import LinterTestCase
else:
    from .linter_test_case import LinterTestCase

TEST_FILE = Path("tools/test/docstring_linter_testdata/python_code.py.txt")
TEST_FILE2 = Path("tools/test/docstring_linter_testdata/more_python_code.py.txt")
ARGS = "--max-class=3", "--max-def=4", "--min-docstring=16"


class TestDocstringLinter(LinterTestCase):
    LinterClass = DocstringLinter
    maxDiff = 10_240

    def test_python_code(self):
        self.lint_test(TEST_FILE, ARGS)

    def test_report(self):
        actual = _dumps(_data())
        self.assertExpected(TEST_FILE, actual, "report.json")

    def NO_test_recursive(self):
        recursive = make_recursive(_data())
        actual = _dumps(recursive)
        self.assertExpected(TEST_FILE, actual, "recursive.json")

    def test_terse(self):
        terse = make_terse(_data(), index_by_line=False)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "terse.json")

    def NO_test_terse_recursive(self):
        recursive = make_recursive(_data())
        terse = make_terse(recursive, index_by_line=False)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "recursive.terse.json")

    def test_terse_line(self):
        terse = make_terse(_data(), index_by_line=True)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "terse.line.json")

    def NO_test_terse_line_recursive(self):
        recursive = make_recursive(_data())
        terse = make_terse(recursive, index_by_line=True)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "recursive.terse.line.json")

    def test_file_summary(self):
        actual = _dumps(file_summary(_data(), report_all=True))
        self.assertExpected(TEST_FILE, actual, "single.line.json")


def _dumps(d: dict) -> str:
    return json.dumps(d, sort_keys=True, indent=2) + "\n"


def _data():
    docstring_file = DocstringLinter.make_file(TEST_FILE)
    return [b.as_data() for b in docstring_file.blocks]


def _next_stdout(mock_stdout):
    length = 0
    while True:
        s = mock_stdout.getvalue()
        yield s[length:]
        length = len(s)
