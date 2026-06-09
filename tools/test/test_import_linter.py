from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.linter.adapters.import_linter import check_file


class TestImportLinter(unittest.TestCase):
    def check_contents(self, contents: str):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            path = Path(tmpdir) / "example.py"
            path.write_text(textwrap.dedent(contents))
            return check_file(str(path))

    def test_disallows_unknown_import(self) -> None:
        messages = self.check_contents(
            """
            import definitely_not_allowed
            """
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "Disallowed import")
        self.assertEqual(messages[0].line, 2)

    def test_disallows_optional_import_at_import_time(self) -> None:
        messages = self.check_contents(
            """
            if True:
                import triton
            """
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "Disallowed import-time import")
        self.assertEqual(messages[0].line, 3)

    def test_disallows_import_time_pytest_import(self) -> None:
        messages = self.check_contents(
            """
            from _pytest.raises import RaisesExc
            """
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "Disallowed import-time import")
        self.assertEqual(messages[0].line, 2)

    def test_disallows_import_time_has_triton_package_call(self) -> None:
        messages = self.check_contents(
            """
            from torch.utils._triton import has_triton_package

            HAS_TRITON_PACKAGE = has_triton_package()
            """
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "Disallowed import-time call")
        self.assertEqual(messages[0].line, 4)

    def test_allows_local_optional_import(self) -> None:
        messages = self.check_contents(
            """
            def fn():
                import triton
            """
        )

        self.assertEqual(messages, [])

    def test_allows_local_has_triton_package_call(self) -> None:
        messages = self.check_contents(
            """
            from torch.utils._triton import has_triton_package

            def fn():
                return has_triton_package()
            """
        )

        self.assertEqual(messages, [])

    def test_allows_type_checking_optional_import(self) -> None:
        messages = self.check_contents(
            """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                import triton
            """
        )

        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
