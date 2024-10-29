from __future__ import annotations

from itertools import product
from pathlib import Path
from token import NAME
from tokenize import TokenInfo
from unittest import TestCase

from tools.linter.adapters.set_linter import fix_set_tokens, PythonLines


TESTDATA = Path(__file__).parent / "set_linter_testdata"

TESTFILE = TESTDATA / "sample.py.txt"
TESTFILE_OMITTED = TESTDATA / "sample-omitted.py.txt"
INCLUDES_FILE = TESTDATA / "includes.py.txt"
INCLUDES_FILE2 = TESTDATA / "includes2.py.txt"
PATHS = TESTFILE, TESTFILE_OMITTED, INCLUDES_FILE, INCLUDES_FILE2
FIX_TESTS = product(PATHS, (False, True))


class TestSetLinter(TestCase):
    def test_get_all_tokens(self) -> None:
        self.assertEqual(EXPECTED_SETS, PythonLines(TESTFILE).sets)

    def test_all_sets_omitted(self) -> None:
        self.assertEqual(EXPECTED_SETS_OMITTED, PythonLines(TESTFILE_OMITTED).sets)

    def test_omitted_lines(self) -> None:
        actual = sorted(PythonLines(TESTFILE_OMITTED).omitted.omitted)
        expected = [1, 5, 12]
        self.assertEqual(expected, actual)

    # TODO(rec): how to get parametrize to work with unittest?
    # @pytest.mark.parametrize(("path", "add_any"),
    def test_fix_set_token(self) -> None:
        for path, add_any in FIX_TESTS:
            actual, expected = _fix_set_tokens(path, add_any)
            self.assertEqual(actual, expected)

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
        )
        for i, (s, expected) in enumerate(TESTS):
            pl = PythonLines(s)
            if s:
                actual = pl.token_lines[0].bracket_pairs
            else:
                self.assertEqual(pl.token_lines, [])
                actual = {}
            self.assertEqual(actual, expected)

    def test_match_braced_sets(self) -> None:
        TESTS: tuple[tuple[str, int], ...] = (
            ("", 0),
            ("{}", 0),
            ("{1: 0}", 0),
            ("{1}", 1),
            ("{i for i in range(2, 3)}", 1),
            ("{1, 2}", 1),
            ("{One({'a': 1}), Two([{}, {2}, {1, 2}])}", 3),
        )
        for i, (s, expected) in enumerate(TESTS):
            pl = PythonLines(s)
            actual = pl.token_lines and pl.token_lines[0].braced_sets
            self.assertEqual(len(actual), expected)


def _fix_set_tokens(path: Path, add_any: bool = False) -> tuple[list[str], list[str]]:
    pl = PythonLines(path)
    fix_set_tokens(pl, add_any)
    expected_file = Path(f"{path}{'.add_any' * add_any}.expected")
    if expected_file.exists():
        with expected_file.open() as fp:
            expected = fp.readlines()
    else:
        expected_file.write_text("".join(pl.lines))
        expected = pl.lines

    return pl.lines, expected


EXPECTED_SETS = [
    TokenInfo(NAME, "set", (1, 4), (1, 7), "a = set()\n"),
    TokenInfo(NAME, "set", (3, 4), (3, 7), "c = set\n"),
    TokenInfo(NAME, "set", (6, 3), (6, 6), "   set(\n"),
]
EXPECTED_SETS_OMITTED = [
    TokenInfo(NAME, "set", (2, 4), (2, 7), "a = set()\n"),
    TokenInfo(NAME, "set", (4, 4), (4, 7), "c = set\n"),
    TokenInfo(NAME, "set", (8, 3), (8, 6), "   set(\n"),
]
