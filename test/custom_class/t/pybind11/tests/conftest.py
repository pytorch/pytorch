"""pytest configuration

Extends output capture as needed by pybind11: ignore constructors, optional unordered lines.
Adds docstring and exceptions message sanitizers: ignore Python 2 vs 3 differences.
"""

import pytest
import textwrap
import difflib
import re
import sys
import contextlib
import platform
import gc

_unicode_marker = re.compile(r'u(\'[^\']*\')')
_long_marker = re.compile(r'([0-9])L')
_hexadecimal = re.compile(r'0x[0-9a-fA-F]+')


def _strip_and_dedent(s):
    """For triple-quote strings"""
    return textwrap.dedent(s.lstrip('\n').rstrip())


def _split_and_sort(s):
    """For output which does not require specific line order"""
    return sorted(_strip_and_dedent(s).splitlines())


def _make_explanation(a, b):
    """Explanation for a failed assert -- the a and b arguments are List[str]"""
    return ["--- actual / +++ expected"] + [line.strip('\n') for line in difflib.ndiff(a, b)]


class Output(object):
    """Basic output post-processing and comparison"""
    def __init__(self, string):
        self.string = string
        self.explanation = []

    def __str__(self):
        return self.string

    def __eq__(self, other):
        # Ignore constructor/destructor output which is prefixed with "###"
        a = [line for line in self.string.strip().splitlines() if not line.startswith("###")]
        b = _strip_and_dedent(other).splitlines()
        if a == b:
            return True
        else:
            self.explanation = _make_explanation(a, b)
            return False


class Unordered(Output):
    """Custom comparison for output without strict line ordering"""
    def __eq__(self, other):
        a = _split_and_sort(self.string)
        b = _split_and_sort(other)
        if a == b:
            return True
        else:
            self.explanation = _make_explanation(a, b)
            return False


class Capture(object):
    def __init__(self, capfd):
        self.capfd = capfd
        self.out = ""
        self.err = ""

    def __enter__(self):
        self.capfd.readouterr()
        return self

    def __exit__(self, *args):
        self.out, self.err = self.capfd.readouterr()

    def __eq__(self, other):
        a = Output(self.out)
        b = other
        if a == b:
            return True
        else:
            self.explanation = a.explanation
            return False

    def __str__(self):
        return self.out

    def __contains__(self, item):
        return item in self.out

    @property
    def unordered(self):
        return Unordered(self.out)

    @property
    def stderr(self):
        return Output(self.err)


@pytest.fixture
def capture(capsys):
    """Extended `capsys` with context manager and custom equality operators"""
    return Capture(capsys)


class SanitizedString(object):
    def __init__(self, sanitizer):
        self.sanitizer = sanitizer
        self.string = ""
        self.explanation = []

    def __call__(self, thing):
        self.string = self.sanitizer(thing)
        return self

    def __eq__(self, other):
        a = self.string
        b = _strip_and_dedent(other)
        if a == b:
            return True
        else:
            self.explanation = _make_explanation(a.splitlines(), b.splitlines())
            return False


def _sanitize_general(s):
    s = s.strip()
    s = s.replace("pybind11_tests.", "m.")
    s = s.replace("unicode", "str")
    s = _long_marker.sub(r"\1", s)
    s = _unicode_marker.sub(r"\1", s)
    return s


def _sanitize_docstring(thing):
    s = thing.__doc__
    s = _sanitize_general(s)
    return s


@pytest.fixture
def doc():
    """Sanitize docstrings and add custom failure explanation"""
    return SanitizedString(_sanitize_docstring)


def _sanitize_message(thing):
    s = str(thing)
    s = _sanitize_general(s)
    s = _hexadecimal.sub("0", s)
    return s


@pytest.fixture
def msg():
    """Sanitize messages and add custom failure explanation"""
    return SanitizedString(_sanitize_message)


# noinspection PyUnusedLocal
def pytest_assertrepr_compare(op, left, right):
    """Hook to insert custom failure explanation"""
    if hasattr(left, 'explanation'):
        return left.explanation


@contextlib.contextmanager
def suppress(exception):
    """Suppress the desired exception"""
    try:
        yield
    except exception:
        pass


def gc_collect():
    ''' Run the garbage collector twice (needed when running
    reference counting tests with PyPy) '''
    gc.collect()
    gc.collect()


def pytest_configure():
    """Add import suppression and test requirements to `pytest` namespace"""
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        import scipy
    except ImportError:
        scipy = None
    try:
        from pybind11_tests.eigen import have_eigen
    except ImportError:
        have_eigen = False
    pypy = platform.python_implementation() == "PyPy"

    skipif = pytest.mark.skipif
    pytest.suppress = suppress
    pytest.requires_numpy = skipif(not np, reason="numpy is not installed")
    pytest.requires_scipy = skipif(not np, reason="scipy is not installed")
    pytest.requires_eigen_and_numpy = skipif(not have_eigen or not np,
                                             reason="eigen and/or numpy are not installed")
    pytest.requires_eigen_and_scipy = skipif(
        not have_eigen or not scipy, reason="eigen and/or scipy are not installed")
    pytest.unsupported_on_pypy = skipif(pypy, reason="unsupported on PyPy")
    pytest.unsupported_on_py2 = skipif(sys.version_info.major < 3,
                                       reason="unsupported on Python 2.x")
    pytest.gc_collect = gc_collect


def _test_import_pybind11():
    """Early diagnostic for test module initialization errors

    When there is an error during initialization, the first import will report the
    real error while all subsequent imports will report nonsense. This import test
    is done early (in the pytest configuration file, before any tests) in order to
    avoid the noise of having all tests fail with identical error messages.

    Any possible exception is caught here and reported manually *without* the stack
    trace. This further reduces noise since the trace would only show pytest internals
    which are not useful for debugging pybind11 module issues.
    """
    # noinspection PyBroadException
    try:
        import pybind11_tests  # noqa: F401 imported but unused
    except Exception as e:
        print("Failed to import pybind11_tests from pytest:")
        print("  {}: {}".format(type(e).__name__, e))
        sys.exit(1)


_test_import_pybind11()
