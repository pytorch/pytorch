# Owner(s): ["module: dynamo"]

"""Light smoke test switching between numpy to pytorch random streams.
"""
from contextlib import contextmanager
import pytest

import torch._numpy as tnp
from torch._numpy.testing import assert_equal
import numpy as _np


@contextmanager
def control_stream(use_numpy=False):
    oldstate = tnp.random.USE_NUMPY_RANDOM
    tnp.random.USE_NUMPY_RANDOM = use_numpy
    try:
        yield
    finally:
        tnp.random.USE_NUMPY_RANDOM = oldstate


def test_uniform():
    r = tnp.random.uniform(0, 1, size=10)


def test_shuffle():
    x = tnp.arange(10)
    tnp.random.shuffle(x)



def test_numpy_global():
    with control_stream(use_numpy=True):
        tnp.random.seed(12345)
        x = tnp.random.uniform(0, 1, size=11)

    # check that the stream is identical to numpy's
    _np.random.seed(12345)
    x_np = _np.random.uniform(0, 1, size=11)
    assert_equal(x, tnp.asarray(x_np))

    # switch to the pytorch stream, variates differ
    with control_stream(use_numpy=False):
        tnp.random.seed(12345)
        x_1 = tnp.random.uniform(0, 1, size=11)

    assert not (x_1 == x).all()


def test_wrong_global():
    with control_stream("oops"):
        with pytest.raises(ValueError):
            tnp.random.rand()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
