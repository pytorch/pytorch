# mypy: ignore-errors
import io
import json
import os
from pathlib import Path
from unittest import mock, TestCase

from tools.linter.adapters._linter import PythonFile


class LinterTestCase(TestCase):
    LinterClass = None
    rewrite_expected = "REWRITE_EXPECTED" in os.environ

    def assertExpected(self, path: Path, actual: str, suffix: str) -> None:
        expected_file = Path(f"{path}.{suffix}")
        if not self.rewrite_expected and expected_file.exists():
            self.assertEqual(actual, expected_file.read_text())
        else:
            expected_file.write_text(actual)

    def replace(self, s: str):
        linter = self.LinterClass("dummy")
        pf = PythonFile(linter.linter_name, contents=s)
        replacement, _results = linter._replace(pf)
        return replacement

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    def lint_test(self, path, args, mock_stdout):
        return self._lint_test(path, args, mock_stdout)[:2]

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    def lint_fix_test(self, path, args, mock_stdout):
        rep, results, linter = self._lint_test(path, args, mock_stdout)
        r = results[-1]
        path = linter.paths[0]
        self.assertEqual(r.original, path.read_text())
        self.assertEqual(rep, r.replacement)
        self.assertExpected(path, r.replacement, "python")
        return r

    def _lint_test(self, path, args, mock_stdout):
        with self.subTest("from-command-line"):
            linter = self.LinterClass([str(path), *args])
            linter.lint_all()
            self.assertExpected(path, mock_stdout.getvalue(), "lintrunner")

        with self.subTest("from-lintrunner"):
            linter = self.LinterClass(["--lintrunner", str(path), *args])
            pf = PythonFile(linter.linter_name, path)
            replacement, results = linter._replace(pf)

            actual = [json.loads(d) for d in linter._display(pf, results)]
            actual = json.dumps(actual, indent=2, sort_keys=True) + "\n"
            self.assertExpected(path, actual, "json")

        return replacement, results, linter
