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
    """
    Return a list of cartesian product of input iterables.
    For example, cross_product(A, B) returns ((x,y) for x in A for y in B).
    """
    return (list(itertools.product(*inputs)))


def get_n_rand_nums(min_val, max_val, n):
    random.seed((1 << 32) - 1)
    return random.sample(range(min_val, max_val), n)


def generate_configs(**configs):
    """
    Given configs from users, we want to generate different combinations of
    those configs
    For example, given M = ((1, 2), N = (4, 5)) and sample_func being cross_product,
    we will generate (({'M': 1}, {'N' : 4}),
                      ({'M': 1}, {'N' : 5}),
                      ({'M': 2}, {'N' : 4}),
                      ({'M': 2}, {'N' : 5}))
    """
    assert 'sample_func' in configs, "Missing sample_func to generat configs"
    result = []
    for key, values in configs.items():
        if key == 'sample_func':
            continue
        tmp_result = []
        for value in values:
            tmp_result.append({key : value})
        result.append(tmp_result)

    results = configs['sample_func'](*result)
    return results

def is_caffe2_enabled(framework_arg):
    return 'Caffe2' in framework_arg

def is_pytorch_enabled(framework_arg):
    return 'PyTorch' in framework_arg

def get_requested_frameworks(framework_arg):
    return [fr.strip() for fr in framework_arg.split(',') if len(fr.strip()) > 0]
