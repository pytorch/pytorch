# mypy: ignore-errors
import tempfile
import unittest
from pathlib import Path

from tools.linter.adapters.unspecified_backend_linter import (
    check_file,
    LINTER_CODE,
    LintSeverity,
)


class TestUnspecifiedBackendLinter(unittest.TestCase):
    """Test the torch.compile explicit-backend linter."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = Path(self.tmpdir.name) / "callsite.py"

    def _check(self, source: str):
        self.path.write_text(source)
        return check_file(str(self.path))

    def test_bare_call_without_backend_is_flagged(self):
        messages = self._check(
            "import torch\ndef f(x):\n    return x\ng = torch.compile(f)\n"
        )
        self.assertEqual(len(messages), 1)
        msg = messages[0]
        self.assertEqual(msg.path, str(self.path))
        self.assertEqual(msg.line, 4)
        self.assertEqual(msg.char, 5)
        self.assertEqual(msg.code, LINTER_CODE)
        self.assertEqual(msg.severity, LintSeverity.ERROR)
        self.assertEqual(msg.name, "implicit-inductor-backend")

    def test_call_with_backend_is_ok(self):
        messages = self._check(
            "import torch\n"
            "def f(x):\n"
            "    return x\n"
            'g = torch.compile(f, backend="eager")\n'
        )
        self.assertEqual(messages, [])

    def test_bare_decorator_is_flagged(self):
        messages = self._check(
            "import torch\n@torch.compile\ndef f(x):\n    return x\n"
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].line, 2)
        self.assertEqual(messages[0].name, "implicit-inductor-backend")

    def test_call_decorator_without_backend_is_flagged(self):
        messages = self._check(
            "import torch\n@torch.compile()\ndef f(x):\n    return x\n"
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].line, 2)

    def test_call_decorator_with_backend_is_ok(self):
        messages = self._check(
            "import torch\n"
            '@torch.compile(backend="aot_eager")\n'
            "def f(x):\n"
            "    return x\n"
        )
        self.assertEqual(messages, [])

    def test_noqa_same_line_suppresses(self):
        messages = self._check(
            "import torch\n"
            "def f(x):\n"
            "    return x\n"
            "g = torch.compile(f)  # noqa: UNSPECIFIED_BACKEND\n"
        )
        self.assertEqual(messages, [])

    def test_noqa_on_any_line_of_multiline_call_suppresses(self):
        messages = self._check(
            "import torch\n"
            "def f(x):\n"
            "    return x\n"
            "g = torch.compile(\n"
            "    f,  # noqa: UNSPECIFIED_BACKEND\n"
            ")\n"
        )
        self.assertEqual(messages, [])

    def test_unrelated_compile_is_ignored(self):
        messages = self._check('import re\np = re.compile("x")\n')
        self.assertEqual(messages, [])

    def test_multiple_offenders(self):
        messages = self._check(
            "import torch\n"
            "def f(x):\n"
            "    return x\n"
            "a = torch.compile(f)\n"
            "b = torch.compile(f)\n"
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual([m.line for m in messages], [4, 5])

    def test_syntax_error_reports_message(self):
        messages = self._check("def f(:\n    pass\n")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "syntax-error")
        self.assertEqual(messages[0].severity, LintSeverity.ERROR)


if __name__ == "__main__":
    unittest.main()
