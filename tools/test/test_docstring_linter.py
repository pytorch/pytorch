# mypy: ignore-errors
from __future__ import annotations

from pathlib import Path

from tools.linter.adapters.docstring_linter import DocstringLinter
from tools.test.test_set_linter import LinterTestCase


TEST_FILE = Path("tools/test/docstring_linter_testdata/python_code.py.txt")


class TestDocstringLinter(LinterTestCase):
    LinterClass = DocstringLinter

    def test_python_code(self):
        args = f"--max-class=3 --max-def=4".split()
        self.lint_test(TEST_FILE, args)
