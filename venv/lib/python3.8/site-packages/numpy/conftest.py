"""
Pytest configuration and fixtures for the Numpy test suite.
"""
import os
import tempfile

import hypothesis
import pytest
import numpy

from numpy.core._multiarray_tests import get_fpu_mode


_old_fpu_mode = None
_collect_results = {}

# Use a known and persistent tmpdir for hypothesis' caches, which
# can be automatically cleared by the OS or user.
hypothesis.configuration.set_hypothesis_home_dir(
    os.path.join(tempfile.gettempdir(), ".hypothesis")
)
# See https://hypothesis.readthedocs.io/en/latest/settings.html
hypothesis.settings.register_profile(
    name="numpy-profile", deadline=None, print_blob=True,
)
# We try loading the profile defined by np.test(), which disables the
# database and forces determinism, and fall back to the profile defined
# above if we're running pytest directly.  The odd dance is required
# because np.test() executes this file *after* its own setup code.
try:
    hypothesis.settings.load_profile("np.test() profile")
except hypothesis.errors.InvalidArgument:  # profile not found
    hypothesis.settings.load_profile("numpy-profile")


def pytest_configure(config):
    config.addinivalue_line("markers",
        "valgrind_error: Tests that are known to error under valgrind.")
    config.addinivalue_line("markers",
        "leaks_references: Tests that are known to leak references.")
    config.addinivalue_line("markers",
        "slow: Tests that are very slow.")
    config.addinivalue_line("markers",
        "slow_pypy: Tests that are very slow on pypy.")


def pytest_addoption(parser):
    parser.addoption("--available-memory", action="store", default=None,
                     help=("Set amount of memory available for running the "
                           "test suite. This can result to tests requiring "
                           "especially large amounts of memory to be skipped. "
                           "Equivalent to setting environment variable "
                           "NPY_AVAILABLE_MEM. Default: determined"
                           "automatically."))


def pytest_sessionstart(session):
    available_mem = session.config.getoption('available_memory')
    if available_mem is not None:
        os.environ['NPY_AVAILABLE_MEM'] = available_mem


#FIXME when yield tests are gone.
@pytest.hookimpl()
def pytest_itemcollected(item):
    """
    Check FPU precision mode was not changed during test collection.

    The clumsy way we do it here is mainly necessary because numpy
    still uses yield tests, which can execute code at test collection
    time.
    """
    global _old_fpu_mode

    mode = get_fpu_mode()

    if _old_fpu_mode is None:
        _old_fpu_mode = mode
    elif mode != _old_fpu_mode:
        _collect_results[item] = (_old_fpu_mode, mode)
        _old_fpu_mode = mode


@pytest.fixture(scope="function", autouse=True)
def check_fpu_mode(request):
    """
    Check FPU precision mode was not changed during the test.
    """
    old_mode = get_fpu_mode()
    yield
    new_mode = get_fpu_mode()

    if old_mode != new_mode:
        raise AssertionError("FPU precision mode changed from {0:#x} to {1:#x}"
                             " during the test".format(old_mode, new_mode))

    collect_result = _collect_results.get(request.node)
    if collect_result is not None:
        old_mode, new_mode = collect_result
        raise AssertionError("FPU precision mode changed from {0:#x} to {1:#x}"
                             " when collecting the test".format(old_mode,
                                                                new_mode))


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy
