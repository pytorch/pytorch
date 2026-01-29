# Owner(s): ["module: dynamo"]

import sys

import pytest

import torch._numpy as tnp


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: very slow tests")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--nonp", action="store_true", help="error when NumPy is accessed")


class Inaccessible:
    def __getattribute__(self, attr):
        raise RuntimeError(f"Using --nonp but accessed np.{attr}")


def pytest_sessionstart(session):
    # Use default=False to handle case when option isn't registered
    # (e.g., when pytest is run from the root directory)
    if session.config.getoption("--nonp", default=False):
        sys.modules["numpy"] = Inaccessible()


def pytest_generate_tests(metafunc):
    """
    Hook to parametrize test cases
    See https://docs.pytest.org/en/6.2.x/parametrize.html#pytest-generate-tests

    The logic here allows us to test with both NumPy-proper and torch._numpy.
    Normally we'd just test torch._numpy, e.g.

        import torch._numpy as np
        ...
        def test_foo():
            np.array([42])
            ...

    but this hook allows us to test NumPy-proper as well, e.g.

        def test_foo(np):
            np.array([42])
            ...

    np is a pytest parameter, which is either NumPy-proper or torch._numpy. This
    allows us to sanity check our own tests, so that tested behaviour is
    consistent with NumPy-proper.

    pytest will have test names respective to the library being tested, e.g.

        $ pytest --collect-only
        test_foo[torch._numpy]
        test_foo[numpy]

    """
    np_params = [tnp]

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if not isinstance(np, Inaccessible):  # i.e. --nonp was used
            np_params.append(np)

    if "np" in metafunc.fixturenames:
        metafunc.parametrize("np", np_params)


def pytest_collection_modifyitems(config, items):
    # Use default=False to handle case when option isn't registered
    # (e.g., when pytest is run from the root directory)
    # See: https://github.com/pytorch/pytorch/issues/171563
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="slow test, use --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
