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

    def test_python_code(self):
        path = TEST_FILE

        linter = DocstringLinter(["-c3", "-f4", str(path)])
        messages = [i.asdict() for i in linter.lint_all()]
        actual = json.dumps(messages, indent=2) + "\n"
        self.assertExpected(path, actual, "json")
