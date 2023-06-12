# Owner(s): ["module: typing"]
# based on NumPy numpy/typing/tests/test_typing.py

import itertools
import os
import re
import shutil
from collections import defaultdict
from typing import IO, Dict, List, Optional

import pytest

from torch.testing._internal.common_utils import run_tests

try:
    from mypy import api
except ImportError:
    NO_MYPY = True
else:
    NO_MYPY = False


DATA_DIR = os.path.join(os.path.dirname(__file__), "typing")
REVEAL_DIR = os.path.join(DATA_DIR, "reveal")
PASS_DIR = os.path.join(DATA_DIR, "pass")
FAIL_DIR = os.path.join(DATA_DIR, "fail")
MYPY_INI = os.path.join(DATA_DIR, os.pardir, os.pardir, "mypy.ini")
CACHE_DIR = os.path.join(DATA_DIR, ".mypy_cache")

#: A dictionary with file names as keys and lists of the mypy stdout as values.
#: To-be populated by `run_mypy`.
OUTPUT_MYPY: Dict[str, List[str]] = {}


def _key_func(key: str) -> str:
    """Split at the first occurance of the ``:`` character.

    Windows drive-letters (*e.g.* ``C:``) are ignored herein.
    """
    drive, tail = os.path.splitdrive(key)
    return os.path.join(drive, tail.split(":", 1)[0])


def _strip_filename(msg: str) -> str:
    """Strip the filename from a mypy message."""
    _, tail = os.path.splitdrive(msg)
    return tail.split(":", 1)[-1]


@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.fixture(scope="module", autouse=True)
def run_mypy() -> None:
    """Clears the cache and run mypy before running any of the typing tests.

    The mypy results are cached in `OUTPUT_MYPY` for further use.

    """
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    for directory in (REVEAL_DIR, PASS_DIR, FAIL_DIR):
        # Run mypy
        stdout, stderr, _ = api.run(
            [
                "--show-absolute-path",
                "--config-file",
                MYPY_INI,
                "--cache-dir",
                CACHE_DIR,
                directory,
            ]
        )
        assert not stderr, directory
        stdout = stdout.replace("*", "")

        # Parse the output
        iterator = itertools.groupby(stdout.split("\n"), key=_key_func)
        OUTPUT_MYPY.update((k, list(v)) for k, v in iterator if k)


def get_test_cases(directory):
    for root, _, files in os.walk(directory):
        for fname in files:
            if os.path.splitext(fname)[-1] == ".py":
                fullpath = os.path.join(root, fname)
                # Use relative path for nice py.test name
                relpath = os.path.relpath(fullpath, start=directory)

                yield pytest.param(
                    fullpath,
                    # Manually specify a name for the test
                    id=relpath,
                )


@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR))
def test_success(path):
    # Alias `OUTPUT_MYPY` so that it appears in the local namespace
    output_mypy = OUTPUT_MYPY
    if path in output_mypy:
        msg = "Unexpected mypy output\n\n"
        msg += "\n".join(_strip_filename(v) for v in output_mypy[path])
        raise AssertionError(msg)


@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(FAIL_DIR))
def test_fail(path):
    __tracebackhide__ = True

    with open(path) as fin:
        lines = fin.readlines()

    errors = defaultdict(lambda: "")

    output_mypy = OUTPUT_MYPY
    assert path in output_mypy
    for error_line in output_mypy[path]:
        error_line = _strip_filename(error_line)
        match = re.match(
            r"(?P<lineno>\d+): (error|note): .+$",
            error_line,
        )
        if match is None:
            raise ValueError(f"Unexpected error line format: {error_line}")
        lineno = int(match.group('lineno'))
        errors[lineno] += f'{error_line}\n'

    for i, line in enumerate(lines):
        lineno = i + 1
        if line.startswith('#') or (" E:" not in line and lineno not in errors):
            continue

        target_line = lines[lineno - 1]
        if "# E:" in target_line:
            marker = target_line.split("# E:")[-1].strip()
            expected_error = errors.get(lineno)
            _test_fail(path, marker, expected_error, lineno)
        else:
            pytest.fail(f"Unexpected mypy output\n\n{errors[lineno]}")


_FAIL_MSG1 = """Extra error at line {}
Extra error: {!r}
"""

_FAIL_MSG2 = """Error mismatch at line {}
Expected error: {!r}
Observed error: {!r}
"""


def _test_fail(path: str, error: str, expected_error: Optional[str], lineno: int) -> None:
    if expected_error is None:
        raise AssertionError(_FAIL_MSG1.format(lineno, error))
    elif error not in expected_error:
        raise AssertionError(_FAIL_MSG2.format(lineno, expected_error, error))


def _construct_format_dict():
    dct = {
        'ModuleList': 'torch.nn.modules.container.ModuleList',
        'AdaptiveAvgPool2d': 'torch.nn.modules.pooling.AdaptiveAvgPool2d',
        'AdaptiveMaxPool2d': 'torch.nn.modules.pooling.AdaptiveMaxPool2d',
        'Tensor': 'torch._tensor.Tensor',
        'Adagrad': 'torch.optim.adagrad.Adagrad',
        'Adam': 'torch.optim.adam.Adam',
    }
    return dct


#: A dictionary with all supported format keys (as keys)
#: and matching values
FORMAT_DICT: Dict[str, str] = _construct_format_dict()


def _parse_reveals(file: IO[str]) -> List[str]:
    """Extract and parse all ``"  # E: "`` comments from the passed file-like object.

    All format keys will be substituted for their respective value from `FORMAT_DICT`,
    *e.g.* ``"{Tensor}"`` becomes ``"torch.tensor.Tensor"``.
    """
    string = file.read().replace("*", "")

    # Grab all `# E:`-based comments
    comments_array = [str.partition("  # E: ")[2] for str in string.split("\n")]
    comments = "/n".join(comments_array)

    # Only search for the `{*}` pattern within comments,
    # otherwise there is the risk of accidently grabbing dictionaries and sets
    key_set = set(re.findall(r"\{(.*?)\}", comments))
    kwargs = {
        k: FORMAT_DICT.get(k, f"<UNRECOGNIZED FORMAT KEY {k!r}>") for k in key_set
    }
    fmt_str = comments.format(**kwargs)

    return fmt_str.split("/n")


@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(REVEAL_DIR))
def test_reveal(path):
    __tracebackhide__ = True

    with open(path) as fin:
        lines = _parse_reveals(fin)

    output_mypy = OUTPUT_MYPY
    assert path in output_mypy
    for error_line in output_mypy[path]:
        match = re.match(
            r"^.+\.py:(?P<lineno>\d+): note: .+$",
            error_line,
        )
        if match is None:
            raise ValueError(f"Unexpected reveal line format: {error_line}")
        lineno = int(match.group("lineno")) - 1
        assert "Revealed type is" in error_line

        marker = lines[lineno]
        _test_reveal(path, marker, error_line, 1 + lineno)


_REVEAL_MSG = """Reveal mismatch at line {}

Expected reveal: {!r}
Observed reveal: {!r}
"""


def _test_reveal(path: str, reveal: str, expected_reveal: str, lineno: int) -> None:
    if reveal not in expected_reveal:
        raise AssertionError(_REVEAL_MSG.format(lineno, expected_reveal, reveal))


if __name__ == "__main__":
    run_tests()
