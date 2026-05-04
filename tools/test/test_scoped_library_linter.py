# mypy: ignore-errors
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.linter.adapters.scoped_library_linter import (
    check_file,
    LINTER_CODE,
    LintMessage,
    LintSeverity,
)


EXPECTED_DESCRIPTION = f"In tests, use torch.library._scoped_library, or skip linting with '# noqa: {LINTER_CODE}'"


class TestScopedLibraryLinter(unittest.TestCase):
    """Regression tests for ``scoped_library_linter.check_file``."""

    def _write(self, directory: Path, rel: str, content: str) -> Path:
        path = directory / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_reports_torch_dot_library_dot_library(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self._write(
                root,
                "sample.py",
                'import torch\n\nlib = torch.library.Library("x", "DEF")\n',
            )
            msgs = check_file(str(path))
            self.assertEqual(len(msgs), 1)
            self.assertEqual(
                msgs[0],
                LintMessage(
                    path=str(path),
                    line=3,
                    char=6,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="direct-torch-Library",
                    original=None,
                    replacement=None,
                    description=EXPECTED_DESCRIPTION,
                ),
            )

    def test_reports_imported_library_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self._write(
                root,
                "sample.py",
                'from torch.library import Library\n\nlib = Library("y", "FRAGMENT")\n',
            )
            msgs = check_file(str(path))
            self.assertEqual(len(msgs), 1)
            self.assertEqual(msgs[0].line, 3)

    def test_allows_inline_noqa(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self._write(
                root,
                "sample.py",
                "import torch\n\n"
                'lib = torch.library.Library("x", "DEF")  # noqa: SCOPED_LIBRARY\n',
            )
            self.assertEqual(check_file(str(path)), [])

    def test_allows_noqa_alone_on_previous_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self._write(
                root,
                "sample.py",
                "import torch\n\n"
                "# noqa: SCOPED_LIBRARY\n"
                'lib = torch.library.Library("x", "DEF")\n',
            )
            self.assertEqual(check_file(str(path)), [])

    def test_allows_multiline_call_if_any_line_has_noqa(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self._write(
                root,
                "sample.py",
                "import torch\n\n"
                "lib = torch.library.Library(\n"
                '    "x",\n'
                '    "DEF",  # noqa: SCOPED_LIBRARY\n'
                ")\n",
            )
            self.assertEqual(check_file(str(path)), [])

    def test_reports_multiline_call_without_noqa(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self._write(
                root,
                "sample.py",
                "import torch\n\n"
                "lib = torch.library.Library(\n"
                '    "x",\n'
                '    "DEF",\n'
                ")\n",
            )
            msgs = check_file(str(path))
            self.assertEqual(len(msgs), 1)
            self.assertEqual(msgs[0].line, 3)

    def test_empty_on_syntax_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self._write(root, "broken.py", "def incomplete(\n")
            self.assertEqual(check_file(str(path)), [])

    def test_no_match_for_non_library_call(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self._write(
                root,
                "sample.py",
                "class NotLibrary:\n"
                "    def __call__(self, *a, **k):\n"
                "        pass\n\n"
                "lib = NotLibrary()('x', 'DEF')\n",
            )
            self.assertEqual(check_file(str(path)), [])


if __name__ == "__main__":
    unittest.main()
