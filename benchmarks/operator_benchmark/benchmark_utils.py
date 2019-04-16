from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import itertools
import random


"""Performance microbenchmarks's utils.

This module contains utilities for writing microbenchmark tests.
"""


def shape_to_string(shape):
    return ', '.join([str(x) for x in shape])


def numpy_random_fp32(*shape):
    """Return a random numpy tensor of float32 type.
    """
    # TODO: consider more complex/custom dynamic ranges for
    # comprehensive test coverage.
    return np.random.rand(*shape).astype(np.float32)


def cross_product(*inputs):
    return (list(itertools.product(*inputs)))


def get_n_rand_nums(min_val, max_val, n):
    random.seed((1 << 32) - 1)
    return random.sample(range(min_val, max_val), n)
