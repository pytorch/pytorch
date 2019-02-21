from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


"""Performance microbenchmarks's utils.

This module contains utilities for writing microbenchmark tests.
"""


def shape_to_string(shape):
    return '_'.join([str(x) for x in shape])


def numpy_random_fp32(*shape):
    """Return a random numpy tensor of float32 type.
    The dynamic range is [-0.5, 0.5).
    """
    # TODO: consider more complex/custom dynamic ranges for
    # comprehensive test coverage.
    return np.random.rand(*shape).astype(np.float32) - 0.5
