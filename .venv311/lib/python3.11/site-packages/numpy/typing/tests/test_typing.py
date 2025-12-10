import importlib.util
import os
import re
import shutil
import textwrap
from collections import defaultdict
from typing import TYPE_CHECKING

import pytest

# Only trigger a full `mypy` run if this environment variable is set
# Note that these tests tend to take over a minute even on a macOS M1 CPU,
# and more than that in CI.
RUN_MYPY = "NPY_RUN_MYPY_IN_TESTSUITE" in os.environ
if RUN_MYPY and RUN_MYPY not in ('0', '', 'false'):
    RUN_MYPY = True

# Skips all functions in this file
pytestmark = pytest.mark.skipif(
    not RUN_MYPY,
    reason="`NPY_RUN_MYPY_IN_TESTSUITE` not set"
)


try:
    from mypy import api
except ImportError:
    NO_MYPY = True
else:
    NO_MYPY = False

if TYPE_CHECKING:
    from collections.abc import Iterator

    # We need this as annotation, but it's located in a private namespace.
    # As a compromise, do *not* import it during runtime
    from _pytest.mark.structures import ParameterSet

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PASS_DIR = os.path.join(DATA_DIR, "pass")
FAIL_DIR = os.path.join(DATA_DIR, "fail")
REVEAL_DIR = os.path.join(DATA_DIR, "reveal")
MISC_DIR = os.path.join(DATA_DIR, "misc")
MYPY_INI = os.path.join(DATA_DIR, "mypy.ini")
CACHE_DIR = os.path.join(DATA_DIR, ".mypy_cache")

#: A dictionary with file names as keys and lists of the mypy stdout as values.
#: To-be populated by `run_mypy`.
OUTPUT_MYPY: defaultdict[str, list[str]] = defaultdict(list)


def _key_func(key: str) -> str:
    """Split at the first occurrence of the ``:`` character.

    Windows drive-letters (*e.g.* ``C:``) are ignored herein.
    """
    drive, tail = os.path.splitdrive(key)
    return os.path.join(drive, tail.split(":", 1)[0])


def _strip_filename(msg: str) -> tuple[int, str]:
    """Strip the filename and line number from a mypy message."""
    _, tail = os.path.splitdrive(msg)
    _, lineno, msg = tail.split(":", 2)
    return int(lineno), msg.strip()


def strip_func(match: re.Match[str]) -> str:
    """`re.sub` helper function for stripping module names."""
    return match.groups()[1]


@pytest.fixture(scope="module", autouse=True)
def run_mypy() -> None:
    """Clears the cache and run mypy before running any of the typing tests.

    The mypy results are cached in `OUTPUT_MYPY` for further use.

    The cache refresh can be skipped using

    NUMPY_TYPING_TEST_CLEAR_CACHE=0 pytest numpy/typing/tests
    """
    if (
        os.path.isdir(CACHE_DIR)
        and bool(os.environ.get("NUMPY_TYPING_TEST_CLEAR_CACHE", True))  # noqa: PLW1508
    ):
        shutil.rmtree(CACHE_DIR)

    split_pattern = re.compile(r"(\s+)?\^(\~+)?")
    for directory in (PASS_DIR, REVEAL_DIR, FAIL_DIR, MISC_DIR):
        # Run mypy
        stdout, stderr, exit_code = api.run([
            "--config-file",
            MYPY_INI,
            "--cache-dir",
            CACHE_DIR,
            directory,
        ])
        if stderr:
            pytest.fail(f"Unexpected mypy standard error\n\n{stderr}", False)
        elif exit_code not in {0, 1}:
            pytest.fail(f"Unexpected mypy exit code: {exit_code}\n\n{stdout}", False)

        str_concat = ""
        filename: str | None = None
        for i in stdout.split("\n"):
            if "note:" in i:
                continue
            if filename is None:
                filename = _key_func(i)

            str_concat += f"{i}\n"
            if split_pattern.match(i) is not None:
                OUTPUT_MYPY[filename].append(str_concat)
                str_concat = ""
                filename = None


def get_test_cases(*directories: str) -> "Iterator[ParameterSet]":
    for directory in directories:
        for root, _, files in os.walk(directory):
            for fname in files:
                short_fname, ext = os.path.splitext(fname)
                if ext not in (".pyi", ".py"):
                    continue

                fullpath = os.path.join(root, fname)
                yield pytest.param(fullpath, id=short_fname)


_FAIL_INDENT = " " * 4
_FAIL_SEP = "\n" + "_" * 79 + "\n\n"

_FAIL_MSG_REVEAL = """{}:{} - reveal mismatch:

{}"""


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR, FAIL_DIR))
def test_pass(path) -> None:
    # Alias `OUTPUT_MYPY` so that it appears in the local namespace
    output_mypy = OUTPUT_MYPY

    if path not in output_mypy:
        return

    relpath = os.path.relpath(path)

    # collect any reported errors, and clean up the output
    messages = []
    for message in output_mypy[path]:
        lineno, content = _strip_filename(message)
        content = content.removeprefix("error:").lstrip()
        messages.append(f"{relpath}:{lineno} - {content}")

    if messages:
        pytest.fail("\n".join(messages), pytrace=False)


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(REVEAL_DIR))
def test_reveal(path: str) -> None:
    """Validate that mypy correctly infers the return-types of
    the expressions in `path`.
    """
    __tracebackhide__ = True

    output_mypy = OUTPUT_MYPY
    if path not in output_mypy:
        return

    relpath = os.path.relpath(path)

    # collect any reported errors, and clean up the output
    failures = []
    for error_line in output_mypy[path]:
        lineno, error_msg = _strip_filename(error_line)
        error_msg = textwrap.indent(error_msg, _FAIL_INDENT)
        reason = _FAIL_MSG_REVEAL.format(relpath, lineno, error_msg)
        failures.append(reason)

    if failures:
        reasons = _FAIL_SEP.join(failures)
        pytest.fail(reasons, pytrace=False)


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR))
def test_code_runs(path: str) -> None:
    """Validate that the code in `path` properly during runtime."""
    path_without_extension, _ = os.path.splitext(path)
    dirname, filename = path.split(os.sep)[-2:]

    spec = importlib.util.spec_from_file_location(
        f"{dirname}.{filename}", path
    )
    assert spec is not None
    assert spec.loader is not None

    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
