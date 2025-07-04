# mypy: ignore-errors

import io
import itertools
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

from tools.linter.adapters._linter.block import _get_decorators
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
TEST_BLOCK_NAMES = Path("tools/test/docstring_linter_testdata/block_names.py.txt")
ARGS = "--max-class=3", "--max-def=4", "--min-docstring=16"


class TestDocstringLinter(LinterTestCase):
    LinterClass = DocstringLinter
    maxDiff = 10_240

    def test_python_code(self):
        self.lint_test(TEST_FILE, ARGS)

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_end_to_end(self, mock_stdout):
        argv_base = *ARGS, str(TEST_FILE), str(TEST_FILE2)
        report = "--report"
        write = "--write-grandfather"

        out = _next_stdout(mock_stdout)

        def run(name, *argv):
            DocstringLinter(argv_base + argv).lint_all()
            self.assertExpected(TEST_FILE2, next(out), name)

        with tempfile.TemporaryDirectory() as td:
            grandfather_file = f"{td}/grandfather.json"
            grandfather = f"--grandfather={grandfather_file}"

            # Find some failures
            run("before.txt", grandfather)

            # Rewrite grandfather file
            run("before.json", grandfather, report, write)
            actual = Path(grandfather_file).read_text()
            self.assertExpected(TEST_FILE2, actual, "grandfather.json")

            # Now there are no failures
            run("after.txt", grandfather)
            run("after.json", grandfather, report)

    def test_report(self):
        actual = _dumps(_data())
        self.assertExpected(TEST_FILE, actual, "report.json")

    def test_terse(self):
        terse = make_terse(_data(), index_by_line=False)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "terse.json")

    def test_terse_line(self):
        terse = make_terse(_data(), index_by_line=True)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "terse.line.json")

    def test_recursive(self):
        recursive = make_recursive(_data())
        actual = _dumps(recursive)
        self.assertExpected(TEST_FILE, actual, "recursive.json")

    def test_terse_recursive(self):
        recursive = make_recursive(_data())
        terse = make_terse(recursive, index_by_line=False)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "recursive.terse.json")

    def test_terse_line_recursive(self):
        recursive = make_recursive(_data())
        terse = make_terse(recursive, index_by_line=True)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "recursive.terse.line.json")

    def test_file_summary(self):
        actual = _dumps(file_summary(_data(), report_all=True))
        self.assertExpected(TEST_FILE, actual, "single.line.json")

    def test_file_names(self):
        f = DocstringLinter.make_file(TEST_BLOCK_NAMES)
        actual = [b.full_name for b in f.blocks]
        expected = [
            "top",
            "top.fun[1]",
            "top.fun[1].sab",
            "top.fun[1].sub",
            "top.fun[2]",
            "top.fun[2].sub[1]",
            "top.fun[2].sub[2]",
            "top.fun[3]",
            "top.fun[3].sub",
            "top.fun[3].sab",
            "top.run",
            "top.run.sub[1]",
            "top.run.sub[2]",
        ]
        self.assertEqual(actual, expected)

    def test_decorators(self):
        tests = itertools.product(INDENTS, DECORATORS.items())
        for indent, (name, (expected, test_inputs)) in tests:
            ind = indent * " "
            for data in test_inputs:
                prog = "".join(ind + d + "\n" for d in data)
                pf = DocstringLinter.make_file(prog)
                it = (i for i, t in enumerate(pf.tokens) if t.string == "def")
                def_t = next(it, 0)
                with self.subTest("Decorator", indent=indent, name=name, data=data):
                    actual = list(_get_decorators(pf.tokens, def_t))
                    self.assertEqual(actual, expected)


def _dumps(d: dict) -> str:
    return json.dumps(d, sort_keys=True, indent=2) + "\n"


def _data(file=TEST_FILE):
    docstring_file = DocstringLinter.make_file(file)
    return [b.as_data() for b in docstring_file.blocks]


def _next_stdout(mock_stdout):
    length = 0
    while True:
        s = mock_stdout.getvalue()
        yield s[length:]
        length = len(s)


CONSTANT = "A = 10"
COMMENT = "# a simple function"
OVER = "@override"
WRAPS = "@functools.wraps(fn)"
MASSIVE = (
    "@some.long.path.very_long_function_name(",
    "    adjust_something_fiddly=1231232,",
    "    disable_something_critical=True,)",
)
MASSIVE_FLAT = (
    "@some.long.path.very_long_function_name("
    "adjust_something_fiddly=1231232,"
    "disable_something_critical=True,)"
)
DEF = "def function():", "    pass"

INDENTS = 0, 4, 8
DECORATORS = {
    "none": (
        [],
        (
            [],
            [*DEF],
            [COMMENT, *DEF],
            [CONSTANT, "", COMMENT, *DEF],
            [OVER, CONSTANT, *DEF],  # Probably not even Python. :-)
        ),
    ),
    "one": (
        [OVER],
        (
            [OVER, *DEF],
            [OVER, COMMENT, *DEF],
            [OVER, COMMENT, "", *DEF],
            [COMMENT, OVER, "", COMMENT, "", *DEF],
        ),
    ),
    "two": (
        [OVER, WRAPS],
        (
            [OVER, WRAPS, *DEF],
            [COMMENT, OVER, COMMENT, WRAPS, COMMENT, *DEF],
        ),
    ),
    "massive": (
        [MASSIVE_FLAT, OVER],
        ([*MASSIVE, OVER, *DEF],),
    ),
}
