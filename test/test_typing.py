# based on NumPy numpy/typing/tests/test_typing.py

import itertools
import os
import re
import shutil
from typing import IO, Dict, List

import pytest

try:
    from mypy import api
except ImportError:
    NO_MYPY = True
else:
    NO_MYPY = False


DATA_DIR = os.path.join(os.path.dirname(__file__), "typing")
REVEAL_DIR = os.path.join(DATA_DIR, "reveal")
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


@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.fixture(scope="module", autouse=True)
def run_mypy() -> None:
    """Clears the cache and run mypy before running any of the typing tests.

    The mypy results are cached in `OUTPUT_MYPY` for further use.

    """
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    for directory in (REVEAL_DIR,):
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

def _construct_format_dict():
    dct = {
        'ModuleList': 'torch.nn.modules.container.ModuleList',
        'AdaptiveAvgPool2d': 'torch.nn.modules.pooling.AdaptiveAvgPool2d',
        'AdaptiveMaxPool2d': 'torch.nn.modules.pooling.AdaptiveMaxPool2d',
        'Tensor': 'torch.tensor.Tensor',
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
    comments_array = list(map(lambda str: str.partition("  # E: ")[2], string.split("\n")))
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


if __name__ == '__main__':
    pytest.main([__file__])
