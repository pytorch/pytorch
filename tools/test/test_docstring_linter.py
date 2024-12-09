# mypy: ignore-errors
from __future__ import annotations

import sys
from pathlib import Path

from tools.linter.adapters.docstring_linter import DocstringLinter


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if _PARENT in _PATH:
    from linter_test_case import LinterTestCase
else:
    from .linter_test_case import LinterTestCase

TEST_FILE = Path("tools/test/docstring_linter_testdata/python_code.py.txt")


class TestDocstringLinter(LinterTestCase):
    LinterClass = DocstringLinter

    def test_python_code(self):
        args = "--max-class=3 --max-def=4".split()
        self.lint_test(TEST_FILE, args)
