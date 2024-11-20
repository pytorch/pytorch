# mypy: ignore-errors
from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

from tools.linter.adapters.docstring_linter import DocstringLinter
from tools.test.test_set_linter import assert_expected


TEST_FILE = Path("tools/test/docstring_linter_testdata/python_code.py.txt")


class TestDocstringLinter(TestCase):
    assertExpected = assert_expected

    maxDiff = 100_000

    def test_python_code(self):
        path = TEST_FILE

        args = f"--max-class=3 --max-def=4 --lintrunner {path}".split()
        linter = DocstringLinter(args)
        res = []

        linter.lint_all(print=res.append)
        messages = [json.loads(r) for r in res]
        actual = json.dumps(messages, indent=2, sort_keys=True) + "\n"
        self.assertExpected(path, actual, "json")
