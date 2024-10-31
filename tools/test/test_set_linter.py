from __future__ import annotations

from pathlib import Path
from token import NAME
from tokenize import TokenInfo
from typing import TYPE_CHECKING
from unittest import TestCase

from tools.linter.adapters.set_linter import get_args, PythonLines


if TYPE_CHECKING:
    from argparse import Namespace


TESTDATA = Path(__file__).parent / "set_linter_testdata"

TESTFILE = TESTDATA / "python_code.py.txt"
INCLUDES_FILE = TESTDATA / "includes.py.txt"
INCLUDES_FILE2 = TESTDATA / "includes_doesnt_change.py.txt"
INCLUDES = INCLUDES_FILE, INCLUDES_FILE2

ARGS_FIX_ALL = get_args(["--fix"])
ARGS_FIX_ANY = get_args(["--add-any", "--set-fix"])
ARGS_FIX_BRACE = get_args(["--brace-fix"])
ARGS_FIX_SET = get_args(["--set-fix"])

ARGS = ARGS_FIX_ALL, ARGS_FIX_ANY, ARGS_FIX_BRACE, ARGS_FIX_SET

FIX_TESTS = [(TESTFILE, a) for a in ARGS] + [(f, ARGS_FIX_ALL) for f in INCLUDES]


class TestSetLinter(TestCase):
    maxDiff = 1_000_000

    def test_get_all_tokens(self) -> None:
        self.assertEqual(EXPECTED_SETS, PythonLines(TESTFILE).sets)

    def test_omitted_lines(self) -> None:
        actual = sorted(PythonLines(TESTFILE).omitted.omitted)
        expected = [1, 5, 12]
        self.assertEqual(expected, actual)

    # TODO(rec): how to get parametrize to work with unittest?
    def test_fix_set_token(self) -> None:
        for path, args in FIX_TESTS:
            expected, actual = _fix_set_tokens(path, args)
            if expected != actual:
                print("FAILING", path)
            self.assertEqual(expected, actual)

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
            ("{cast(int, inst.offset): inst for inst in instructions}", 0),
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


def _fix_set_tokens(path: Path, args: Namespace) -> tuple[list[str], list[str]]:
    pl = PythonLines(path)
    pl.fix_all_tokens(args)
    assert args.brace_fix or args.set_fix, "No fix requested"
    flags = "add_any", "brace_fix", "set_fix"
    flags_suffix = "".join(f".{f}" for f in flags if getattr(args, f))
    expected_file = Path(f"{path}{flags_suffix}")
    if expected_file.exists():
        with expected_file.open() as fp:
            expected = fp.readlines()
    else:
        expected_file.write_text("".join(pl.lines))
        expected = pl.lines

    return expected, pl.lines


EXPECTED_SETS = [
    TokenInfo(NAME, "set", (2, 4), (2, 7), "a = set()\n"),
    TokenInfo(NAME, "set", (4, 4), (4, 7), "c = set\n"),
    TokenInfo(NAME, "set", (8, 3), (8, 6), "   set(\n"),
]
