from __future__ import annotations

from itertools import product
from pathlib import Path
from token import NAME
from tokenize import TokenInfo

import pytest
from tools.linter.adapters.set_linter.fix_set_tokens import fix_set_tokens
from tools.linter.adapters.set_linter.python_file import PythonFile


TESTDATA = Path(__file__).parent / "set_linter_testdata"

TESTFILE = TESTDATA / "sample.py.txt"
TESTFILE_OMITTED = TESTDATA / "sample-omitted.py.txt"
INCLUDES_FILE = TESTDATA / "includes.py.txt"
INCLUDES_FILE2 = TESTDATA / "includes2.py.txt"
FILENAMES = TESTFILE, TESTFILE_OMITTED, INCLUDES_FILE, INCLUDES_FILE2


def test_get_all_tokens() -> None:
    assert EXPECTED_SETS == PythonFile.create(TESTFILE).set_tokens


def test_all_sets_omitted() -> None:
    assert EXPECTED_SETS_OMITTED == PythonFile.create(TESTFILE_OMITTED).set_tokens


def test_omitted_lines() -> None:
    actual = sorted(PythonFile.create(TESTFILE_OMITTED).omitted.omitted)
    expected = [1, 5, 12]
    assert expected == actual


@pytest.mark.parametrize(("filename", "add_any"), product(FILENAMES, (False, True)))
def test_fix_set_token(filename: Path, add_any: bool) -> None:
    actual, expected = _fix_set_tokens(filename, add_any)
    assert actual == expected


def _fix_set_tokens(
    filename: Path, add_any: bool = False
) -> tuple[list[str], list[str]]:
    pf = PythonFile.create(filename)
    fix_set_tokens(pf, add_any)
    expected_file = Path(f"{filename}{'.add_any' * add_any}.expected")
    if expected_file.exists():
        with expected_file.open() as fp:
            expected = fp.readlines()
    else:
        expected_file.write_text("".join(pf.lines))
        expected = pf.lines

    return pf.lines, expected


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
