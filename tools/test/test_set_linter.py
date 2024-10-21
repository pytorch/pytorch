import token
from pathlib import Path
from tokenize import TokenInfo

import pytest
from tools.linter.adapters.set_linter.fix_set_tokens import fix_set_tokens
from tools.linter.adapters.set_linter.python_file import OmittedLines, PythonFile


TESTDATA = Path(__file__).parent / "set_linter_testdata"

TESTFILE = TESTDATA / "sample.py.txt"
TESTFILE_OMITTED = TESTDATA / "sample-omitted.py.txt"
INCLUDES_FILE = TESTDATA / "includes.py.txt"
INCLUDES_FILE2 = TESTDATA / "includes2.py.txt"


def test_get_all_tokens():
    assert EXPECTED_SETS == PythonFile(TESTFILE).set_tokens


def test_omitted_lines():
    actual = sorted(OmittedLines(TESTFILE_OMITTED).lines)
    expected = [1, 5, 12]
    assert expected == actual


def test_all_sets_omitted():
    assert EXPECTED_SETS_OMITTED == PythonFile(TESTFILE_OMITTED).set_tokens


def _token_info(start, end, line):
    return TokenInfo(type=token.NAME, string="set", start=start, end=end, line=line)


def _fix_set_tokens(filename):
    pf = PythonFile(filename)
    fix_set_tokens.fix_set_tokens(pf)
    expected_file = Path(filename + ".expected")
    if expected_file.exists():
        with expected_file.open() as fp:
            expected = fp.readlines()
    else:
        expected_file.write_text("".join(pf.lines))
        expected = pf.lines

    return pf.lines, expected


@pytest.mark.parametrize(
    "filename",
    (TESTFILE, TESTFILE_OMITTED, INCLUDES_FILE, INCLUDES_FILE2),
)
def test_fix_set_token(filename):
    actual, expected = _fix_set_tokens(filename)
    assert actual == expected


EXPECTED_SETS = [
    _token_info((1, 4), (1, 7), "a = set()\n"),
    _token_info((3, 4), (3, 7), "c = set\n"),
    _token_info((6, 3), (6, 6), "   set(\n"),
]

EXPECTED_SETS_OMITTED = [
    _token_info((2, 4), (2, 7), "a = set()\n"),
    _token_info((4, 4), (4, 7), "c = set\n"),
    _token_info((8, 3), (8, 6), "   set(\n"),
]
