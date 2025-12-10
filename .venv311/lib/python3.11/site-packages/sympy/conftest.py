import sys

sys._running_pytest = True  # type: ignore
from sympy.external.importtools import version_tuple

import pytest
from sympy.core.cache import clear_cache, USE_CACHE
from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.misc import ARCH
import re

try:
    import hypothesis

    hypothesis.settings.register_profile("sympy_hypothesis_profile", deadline=None)
    hypothesis.settings.load_profile("sympy_hypothesis_profile")
except ImportError:
    raise ImportError(
        "hypothesis is a required dependency to run the SymPy test suite. "
        "Install it with 'pip install hypothesis' or 'conda install -c conda-forge hypothesis'"
    )


sp = re.compile(r"([0-9]+)/([1-9][0-9]*)")


def process_split(config, items):
    split = config.getoption("--split")
    if not split:
        return
    m = sp.match(split)
    if not m:
        raise ValueError(
            "split must be a string of the form a/b " "where a and b are ints."
        )
    i, t = map(int, m.groups())
    start, end = (i - 1) * len(items) // t, i * len(items) // t

    if i < t:
        # remove elements from end of list first
        del items[end:]
    del items[:start]


def pytest_report_header(config):
    s = "architecture: %s\n" % ARCH
    s += "cache:        %s\n" % USE_CACHE
    version = ""
    if GROUND_TYPES == "gmpy":
        import gmpy2

        version = gmpy2.version()
    elif GROUND_TYPES == "flint":
        try:
            from flint import __version__
        except ImportError:
            version = "unknown"
        else:
            version = f'(python-flint=={__version__})'
    s += "ground types: %s %s\n" % (GROUND_TYPES, version)
    return s


def pytest_terminal_summary(terminalreporter):
    if terminalreporter.stats.get("error", None) or terminalreporter.stats.get(
        "failed", None
    ):
        terminalreporter.write_sep(" ", "DO *NOT* COMMIT!", red=True, bold=True)


def pytest_addoption(parser):
    parser.addoption("--split", action="store", default="", help="split tests")


def pytest_collection_modifyitems(config, items):
    """pytest hook."""
    # handle splits
    process_split(config, items)


@pytest.fixture(autouse=True, scope="module")
def file_clear_cache():
    clear_cache()


@pytest.fixture(autouse=True, scope="module")
def check_disabled(request):
    if getattr(request.module, "disabled", False):
        pytest.skip("test requirements not met.")
    elif getattr(request.module, "ipython", False):
        # need to check version and options for ipython tests
        if (
            version_tuple(pytest.__version__) < version_tuple("2.6.3")
            and pytest.config.getvalue("-s") != "no"
        ):
            pytest.skip("run py.test with -s or upgrade to newer version.")
