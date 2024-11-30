# mypy: ignore-errors
from __future__ import annotations

import sys
from pathlib import Path

from tools.linter.adapters._linter import is_name, is_op, PythonFile
from tools.linter.adapters.annotations_linter import (
    _close_bracket,
    _split_on_commas,
    AnnotationsLinter,
    FROM_FUTURE,
)


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if _PARENT in _PATH:
    from linter_test_case import LinterTestCase
else:
    from .linter_test_case import LinterTestCase


TEST_FILE = Path("tools/test/annotations_linter_testdata/python_code.py.txt")


def run_on(fn, list_of_tests):
    def run(x):
        try:
            return fn(x)
        except Exception as e:
            return e

    args, expected = zip(*list_of_tests)
    actual = (run(a) for a in args)
    errors = [rea for rea in zip(args, expected, actual) if rea[1] != rea[2]]
    return [{"args": r, "expected": e, "actual": a} for r, e, a in errors]


class TestAnnotationsLinter(LinterTestCase):
    LinterClass = AnnotationsLinter

    def test_broken_case(self):
        s = "b: Union[int | None, str]"
        r = self.replace(s).replace(FROM_FUTURE, "").strip()
        self.assertEqual(r, "b: int | None | str")

    def test_close_bracket(self):
        def cb(s):
            pf = PythonFile("annotation_linter", contents=s)
            return _close_bracket(pf.tokens, 1)

        errors = run_on(cb, BRACKET_TESTS)
        self.assertEqual(errors, [])

    def test_commas(self):
        def cb(s):
            pf = PythonFile("annotation_linter", contents=s)
            tokens = list(pf.tokens)
            self.assertTrue(is_name(tokens[0], "Union"))
            self.assertTrue(is_op(tokens[1], "["))
            while not is_op(tokens.pop(), "]"):
                pass
            return [i.strip() for i in _split_on_commas(tokens[2:])]

        errors = run_on(cb, COMMA_TESTS)
        self.assertEqual(errors, [])

    def test_python_json(self):
        self.lint_fix_test(TEST_FILE, [])


BRACKET_TESTS = (
    ("Set", None),
    ("List[]", 2),
    ("tuple[Set[str]]", 6),
    ("tuple[Optional[str]]", 6),
    ("Sequence[Dict[str, Map[Any]]]", 11),
    ("Iterator[Dict[str, Map[Union[set, dict]]]]", 16),
)

COMMA_TESTS = (
    ("Union[str, int]", ["str", "int"]),
    ("Union[str]", ["str"]),
    ("Union[str, int]", ["str", "int"]),
    ("Union[str, dict[str, int]]", ["str", "dict[str, int]"]),
)
