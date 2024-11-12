# mypy: ignore-errors
from __future__ import annotations

import json
from pathlib import Path
from token import NAME
from tokenize import TokenInfo
from unittest import TestCase

from tools.linter.adapters.set_linter import get_args, lint_file, PythonLines


TESTDATA = Path("tools/test/set_linter_testdata")

TESTFILE = TESTDATA / "python_code.py.txt"
INCLUDES_FILE = TESTDATA / "includes.py.txt"
INCLUDES_FILE2 = TESTDATA / "includes_doesnt_change.py.txt"
FILES = TESTFILE, INCLUDES_FILE, INCLUDES_FILE2

ARGS = get_args([])


class TestSetLinter(TestCase):
    maxDiff = 1_000_000

    def test_get_all_tokens(self) -> None:
        self.assertEqual(EXPECTED_SETS, PythonLines(TESTFILE).sets)

    def test_omitted_lines(self) -> None:
        actual = sorted(PythonLines(TESTFILE).omitted.omitted)
        expected = [3, 13]
        self.assertEqual(expected, actual)

    # TODO(rec): how to get parametrize to work with unittest?
    def test_python(self) -> None:
        self._test_linting(TESTFILE)

    def test_includes(self) -> None:
        self._test_linting(INCLUDES_FILE)

    def test_includes2(self) -> None:
        self._test_linting(INCLUDES_FILE2)

    def _test_linting(self, path: Path) -> None:
        def assert_expected(actual: str, suffix: str) -> None:
            expected_file = Path(f"{path}.{suffix}")
            if expected_file.exists():
                self.assertEqual(actual, expected_file.read_text())
            else:
                expected_file.write_text(actual)

        all_messages = [m.asdict() for m in lint_file(str(path), ARGS)]
        replace = all_messages[-1]
        self.assertEqual(replace["original"], path.read_text())
        assert_expected(replace["replacement"], "python")

        msgs = json.dumps(all_messages, indent=2) + "\n"
        assert_expected(msgs, "json")

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


EXPECTED_SETS = [
    TokenInfo(NAME, "set", (4, 4), (4, 7), "a = set()\n"),
    TokenInfo(NAME, "set", (6, 4), (6, 7), "c = set\n"),
    TokenInfo(NAME, "set", (9, 3), (9, 6), "   set(\n"),
]
