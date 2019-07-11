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


def cross_product_configs(**configs):
    """
    Given configs from users, we want to generate different combinations of
    those configs
    For example, given M = ((1, 2), N = (4, 5)),
    we will generate (({'M': 1}, {'N' : 4}),
                      ({'M': 1}, {'N' : 5}),
                      ({'M': 2}, {'N' : 4}),
                      ({'M': 2}, {'N' : 5}))
    """
    configs_attrs_list = []
    for key, values in configs.items():
        tmp_results = [{key : value} for value in values]
        configs_attrs_list.append(tmp_results)

    # TODO(mingzhe0908) remove the conversion to list. 
    # itertools.product produces an iterator that produces element on the fly
    # while converting to a list produces everything at the same time.
    generated_configs = list(itertools.product(*configs_attrs_list))
    return generated_configs


def config_list(**configs):
    """
    Take specific inputs from users
    For example, given 
    attrs = [
        [1, 2],
        [4, 5],
    ]
    attr_names = ["M", "N"]
    we will generate (({'M': 1}, {'N' : 2}),
                      ({'M': 4}, {'N' : 5}))
    """
    generated_configs = []
    if "attrs" not in configs:
        raise ValueError("Missing attrs in configs")
    for inputs in configs["attrs"]:
        tmp_result = [{configs["attr_names"][i] : input_value} 
                      for i, input_value in enumerate(inputs)]
        # TODO(mingzhe0908): 
        # If multiple "tags" were provided, do they get concat?
        # If a config has both ["short", "medium"], it should match 
        # both "short" and "medium" tag-filter?
        tmp_result.append({"tags" : '_'.join(configs["tags"])})
        generated_configs.append(tmp_result)
    return generated_configs


def is_caffe2_enabled(framework_arg):
    return 'Caffe2' in framework_arg


def is_pytorch_enabled(framework_arg):
    return 'PyTorch' in framework_arg


def get_requested_frameworks(framework_arg):
    return [fr.strip() for fr in framework_arg.split(',') if len(fr.strip()) > 0]
