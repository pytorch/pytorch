# mypy: ignore-errors
from __future__ import annotations

import sys
from pathlib import Path
from token import NAME
from tokenize import TokenInfo

from tools.linter.adapters.set_linter import SetLinter


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if _PARENT in _PATH:
    from linter_test_case import LinterTestCase
else:
    from .linter_test_case import LinterTestCase


TESTDATA = Path("tools/test/set_linter_testdata")

TESTFILE = TESTDATA / "python_code.py.txt"
INCLUDES_FILE = TESTDATA / "includes.py.txt"
INCLUDES_FILE2 = TESTDATA / "includes_doesnt_change.py.txt"
FILES = TESTFILE, INCLUDES_FILE, INCLUDES_FILE2


class TestSetLinter(LinterTestCase):
    maxDiff = 10000000
    LinterClass = SetLinter

    def test_get_all_tokens(self) -> None:
        self.assertEqual(EXPECTED_SETS, SetLinter.make_file(TESTFILE).sets)

    def test_omitted_lines(self) -> None:
        actual = sorted(SetLinter.make_file(TESTFILE).omitted.omitted)
        expected = [6, 16]
        self.assertEqual(expected, actual)

    def test_linting(self) -> None:
        for path in (TESTFILE, INCLUDES_FILE, INCLUDES_FILE2):
            with self.subTest(path):
                r = self.lint_fix_test(path, [])
                self.assertEqual(r.name, "Suggested fixes for set_linter")

    def test_bracket_pairs(self) -> None:
        TESTS: tuple[tuple[str, dict[int, int]], ...] = (
            ("", {}),
            ("{}", {0: 1}),
            ("{1}", {0: 2}),
            ("{1, 2}", {0: 4}),
            ("{1: 2}", {0: 4}),
            ("{One()}", {0: 4, 2: 3}),
            (
                "{One({1: [2], 2: {3}, 3: {4: 5}})}",
                {0: 25, 2: 24, 3: 23, 6: 8, 12: 14, 18: 22},
            ),
            ("f'{a}'", {}),
        )
        for s, expected in TESTS:
            pf = SetLinter.make_file(s)
            if s:
                actual = pf._lines_with_sets[0].bracket_pairs
            else:
                self.assertEqual(pf._lines_with_sets, [])
                actual = {}
            self.assertEqual(actual, expected)

    def test_match_braced_sets(self) -> None:
        TESTS: tuple[tuple[str, int], ...] = (
            ("{cast(int, inst.offset): inst for inst in instructions}", 0),
            ("", 0),
            ("{}", 0),
            ("{1: 0}", 0),
            ("{1}", 1),
            ("{i for i in range(2, 3)}", 1),
            ("{1, 2}", 1),
            ("{One({'a': 1}), Two([{}, {2}, {1, 2}])}", 3),
            ('f" {h:{w}} "', 0),
        )
        for s, expected in TESTS:
            pf = SetLinter.make_file(s)
            actual = pf._lines_with_sets and pf._lines_with_sets[0].braced_sets
            self.assertEqual(len(actual), expected)


EXPECTED_SETS = [
    TokenInfo(NAME, "set", (7, 4), (7, 7), "a = set()\n"),
    TokenInfo(NAME, "set", (9, 4), (9, 7), "c = set\n"),
    TokenInfo(NAME, "set", (12, 3), (12, 6), "   set(\n"),
]
