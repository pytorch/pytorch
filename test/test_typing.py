# Owner(s): ["module: typing"]
# based on NumPy numpy/typing/tests/test_typing.py

import itertools
import os
import re
import shutil

import unittest
from collections import defaultdict
from threading import Lock
from typing import Dict, IO, List, Optional

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

try:
    from mypy import api
except ImportError:
    NO_MYPY = True
else:
    NO_MYPY = False


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "typing"))
REVEAL_DIR = os.path.join(DATA_DIR, "reveal")
PASS_DIR = os.path.join(DATA_DIR, "pass")
FAIL_DIR = os.path.join(DATA_DIR, "fail")
MYPY_INI = os.path.join(DATA_DIR, os.pardir, os.pardir, "mypy.ini")
CACHE_DIR = os.path.join(DATA_DIR, ".mypy_cache")


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


def _run_mypy() -> Dict[str, List[str]]:
    """Clears the cache and run mypy before running any of the typing tests."""
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    rc: Dict[str, List[str]] = {}
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
        assert not stderr, stderr
        stdout = stdout.replace("*", "")

        # Parse the output
        iterator = itertools.groupby(stdout.split("\n"), key=_key_func)
        rc.update((k, list(v)) for k, v in iterator if k)
    return rc


def get_test_cases(directory):
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.startswith("disabled_"):
                continue
            if os.path.splitext(fname)[-1] == ".py":
                fullpath = os.path.join(root, fname)
                yield fullpath


_FAIL_MSG1 = """Extra error at line {}
Extra error: {!r}
"""

_FAIL_MSG2 = """Error mismatch at line {}
Expected error: {!r}
Observed error: {!r}
"""


def _test_fail(
    path: str, error: str, expected_error: Optional[str], lineno: int
) -> None:
    if expected_error is None:
        raise AssertionError(_FAIL_MSG1.format(lineno, error))
    elif error not in expected_error:
        raise AssertionError(_FAIL_MSG2.format(lineno, expected_error, error))


def _construct_format_dict():
    dct = {
        "ModuleList": "torch.nn.modules.container.ModuleList",
        "AdaptiveAvgPool2d": "torch.nn.modules.pooling.AdaptiveAvgPool2d",
        "AdaptiveMaxPool2d": "torch.nn.modules.pooling.AdaptiveMaxPool2d",
        "Tensor": "torch._tensor.Tensor",
        "Adagrad": "torch.optim.adagrad.Adagrad",
        "Adam": "torch.optim.adam.Adam",
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


_REVEAL_MSG = """Reveal mismatch at line {}

Expected reveal: {!r}
Observed reveal: {!r}
"""


def _test_reveal(path: str, reveal: str, expected_reveal: str, lineno: int) -> None:
    if reveal not in expected_reveal:
        raise AssertionError(_REVEAL_MSG.format(lineno, expected_reveal, reveal))


@unittest.skipIf(NO_MYPY, reason="Mypy is not installed")
class TestTyping(TestCase):
    _lock = Lock()
    _cached_output: Optional[Dict[str, List[str]]] = None

    @classmethod
    def get_mypy_output(cls) -> Dict[str, List[str]]:
        with cls._lock:
            if cls._cached_output is None:
                cls._cached_output = _run_mypy()
            return cls._cached_output

    @parametrize(
        "path",
        get_test_cases(PASS_DIR),
        name_fn=lambda b: os.path.relpath(b, start=PASS_DIR),
    )
    def test_success(self, path) -> None:
        output_mypy = self.get_mypy_output()
        if path in output_mypy:
            msg = "Unexpected mypy output\n\n"
            msg += "\n".join(_strip_filename(v) for v in output_mypy[path])
            raise AssertionError(msg)

    @parametrize(
        "path",
        get_test_cases(FAIL_DIR),
        name_fn=lambda b: os.path.relpath(b, start=FAIL_DIR),
    )
    def test_fail(self, path):
        __tracebackhide__ = True

        with open(path) as fin:
            lines = fin.readlines()

        errors = defaultdict(lambda: "")

        output_mypy = self.get_mypy_output()
        self.assertIn(path, output_mypy)
        for error_line in output_mypy[path]:
            error_line = _strip_filename(error_line)
            match = re.match(
                r"(?P<lineno>\d+):(?P<colno>\d+): (error|note): .+$",
                error_line,
            )
            if match is None:
                raise ValueError(f"Unexpected error line format: {error_line}")
            lineno = int(match.group("lineno"))
            errors[lineno] += f"{error_line}\n"

        for i, line in enumerate(lines):
            lineno = i + 1
            if line.startswith("#") or (" E:" not in line and lineno not in errors):
                continue

            target_line = lines[lineno - 1]
            self.assertIn(
                "# E:", target_line, f"Unexpected mypy output\n\n{errors[lineno]}"
            )
            marker = target_line.split("# E:")[-1].strip()
            expected_error = errors.get(lineno)
            _test_fail(path, marker, expected_error, lineno)

    @parametrize(
        "path",
        get_test_cases(REVEAL_DIR),
        name_fn=lambda b: os.path.relpath(b, start=REVEAL_DIR),
    )
    def test_reveal(self, path):
        __tracebackhide__ = True

        with open(path) as fin:
            lines = _parse_reveals(fin)

        output_mypy = self.get_mypy_output()
        assert path in output_mypy
        for error_line in output_mypy[path]:
            match = re.match(
                r"^.+\.py:(?P<lineno>\d+):(?P<colno>\d+): note: .+$",
                error_line,
            )
            if match is None:
                raise ValueError(f"Unexpected reveal line format: {error_line}")
            lineno = int(match.group("lineno")) - 1
            assert "Revealed type is" in error_line

            marker = lines[lineno]
            _test_reveal(path, marker, error_line, 1 + lineno)


instantiate_parametrized_tests(TestTyping)

if __name__ == "__main__":
    run_tests()
