# mypy: ignore-errors

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

from tools.linter.adapters._linter.file_summary import _make_terse, file_summary
from tools.linter.adapters.docstring_linter import DocstringLinter


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

            # Find some faiures
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
        terse = _make_terse(_data(), index_by_line=False)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "terse.json")

    def test_terse_line(self):
        terse = _make_terse(_data(), index_by_line=True)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "terse.line.json")

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
