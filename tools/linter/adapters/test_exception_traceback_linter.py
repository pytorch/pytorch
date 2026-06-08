"""Unit tests for exception_traceback_linter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exception_traceback_linter import check_file, LINTER_CODE


def _lint(source: str) -> list[dict]:
    """Write source to a temp file, lint it, return messages as dicts."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        f.flush()
        msgs = check_file(f.name)
    return [msg._asdict() for msg in msgs]


class TestExceptionTracebackLinter(unittest.TestCase):
    # --- cases that SHOULD be flagged ---

    def test_basic_store(self):
        source = """\
try:
    pass
except Exception as e:
    saved = e
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["code"], LINTER_CODE)
        self.assertEqual(msgs[0]["line"], 4)

    def test_store_without_clear(self):
        source = """\
try:
    x = do_thing()
except ValueError as e:
    self.last_error = e
    result = e
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["line"], 5)

    # --- cases that should NOT be flagged ---

    def test_cleared_with_traceback_none(self):
        source = """\
try:
    pass
except Exception as e:
    saved = e
    e.__traceback__ = None
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_cleared_via_alias(self):
        source = """\
try:
    pass
except Exception as e:
    exc = e
    exc.__traceback__ = None
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_cleared_with_clear_frames(self):
        source = """\
import traceback
try:
    pass
except Exception as e:
    saved = e
    traceback.clear_frames(e.__traceback__)
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_cleared_with_clear_frames_on_alias(self):
        source = """\
import traceback
try:
    pass
except Exception as e:
    exc = e
    traceback.clear_frames(exc.__traceback__)
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_reraised_bare(self):
        source = """\
try:
    pass
except Exception as e:
    saved = e
    raise
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_reraised_explicit(self):
        source = """\
try:
    pass
except Exception as e:
    saved = e
    raise e
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_cause_chain_not_flagged(self):
        source = """\
try:
    pass
except AttributeError as e:
    new_err = RuntimeError("wrapped")
    new_err.__cause__ = e
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_noqa_suppresses(self):
        source = """\
try:
    pass
except Exception as e:
    saved = e  # noqa: EXCEPTION_TRACEBACK
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_noqa_bare_suppresses(self):
        source = "try:\n    pass\nexcept Exception as e:\n    saved = e  #" + " noqa\n"
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_noqa_wrong_code_does_not_suppress(self):
        source = """\
try:
    pass
except Exception as e:
    saved = e  # noqa: E501
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 1)

    def test_no_store_no_warning(self):
        source = """\
try:
    pass
except Exception as e:
    print(e)
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_conditional_raise(self):
        """If the exception is re-raised on any path, don't flag."""
        source = """\
try:
    pass
except Exception as e:
    saved = e
    if condition:
        raise e
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 0)

    def test_nested_except_raise_does_not_suppress(self):
        """A bare raise in a nested except handler should not suppress
        the warning for the outer exception."""
        source = """\
try:
    pass
except Exception as e:
    saved = e
    try:
        other()
    except ValueError:
        raise
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 1)

    def test_nested_function_clear_does_not_suppress(self):
        """clear_frames inside a nested function does not clear the outer
        exception."""
        source = """\
import traceback
try:
    pass
except Exception as e:
    saved = e
    def cleanup(exc):
        traceback.clear_frames(exc.__traceback__)
"""
        msgs = _lint(source)
        self.assertEqual(len(msgs), 1)


if __name__ == "__main__":
    unittest.main()
