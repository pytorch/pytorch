import argparse
import bisect
import itertools
import os
import random

import numpy as np


"""Performance microbenchmarks's utils.

This module contains utilities for writing microbenchmark tests.
"""

# Here are the reserved keywords in the benchmark suite
_reserved_keywords = {"probs", "total_samples", "tags"}
_supported_devices = {"cpu", "cuda"}


def shape_to_string(shape):
    return ", ".join([str(x) for x in shape])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def numpy_random(dtype, *shapes):
    """Return a random numpy tensor of the provided dtype.
    Args:
        shapes: int or a sequence of ints to defining the shapes of the tensor
        dtype: use the dtypes from numpy
            (https://docs.scipy.org/doc/numpy/user/basics.types.html)
    Return:
        numpy tensor of dtype
    """
    # TODO: consider more complex/custom dynamic ranges for
    # comprehensive test coverage.
    return np.random.rand(*shapes).astype(dtype)


def set_omp_threads(num_threads):
    existing_value = os.environ.get("OMP_NUM_THREADS", "")
    if existing_value != "":
        print(
            f"Overwriting existing OMP_NUM_THREADS value: {existing_value}; Setting it to {num_threads}."
        )
    os.environ["OMP_NUM_THREADS"] = str(num_threads)


def set_mkl_threads(num_threads):
    existing_value = os.environ.get("MKL_NUM_THREADS", "")
    if existing_value != "":
        print(
            f"Overwriting existing MKL_NUM_THREADS value: {existing_value}; Setting it to {num_threads}."
        )
    os.environ["MKL_NUM_THREADS"] = str(num_threads)


def cross_product(*inputs):
    """
    Return a list of cartesian product of input iterables.
    For example, cross_product(A, B) returns ((x,y) for x in A for y in B).
    """
    return list(itertools.product(*inputs))


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
    assert "sample_func" in configs, "Missing sample_func to generate configs"
    result = []
    for key, values in configs.items():
        if key == "sample_func":
            continue
        tmp_result = []
        for value in values:
            tmp_result.append({key: value})
        result.append(tmp_result)

    results = configs["sample_func"](*result)
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
    _validate(configs)
    configs_attrs_list = []
    for key, values in configs.items():
        tmp_results = [{key: value} for value in values]
        configs_attrs_list.append(tmp_results)

    # TODO(mingzhe0908) remove the conversion to list.
    # itertools.product produces an iterator that produces element on the fly
    # while converting to a list produces everything at the same time.
    generated_configs = list(itertools.product(*configs_attrs_list))
    return generated_configs


def _validate(configs):
    """Validate inputs from users."""
    if "device" in configs:
        for v in configs["device"]:
            assert v in _supported_devices, "Device needs to be a string."


def config_list(**configs):
    """Generate configs based on the list of input shapes.
    This function will take input shapes specified in a list from user. Besides
    that, all other parameters will be cross producted first and each of the
    generated list will be merged with the input shapes list.

    Reserved Args:
        attr_names(reserved): a list of names for input shapes.
        attrs(reserved): a list of values for each input shape.
        corss_product: a dictionary of attributes which will be
                       cross producted with the input shapes.
        tags(reserved): a tag used to filter inputs.

    Here is an example:
    attrs = [
        [1, 2],
        [4, 5],
    ],
    attr_names = ['M', 'N'],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },

    we will generate [[{'M': 1}, {'N' : 2}, {'device' : 'cpu'}],
                      [{'M': 1}, {'N' : 2}, {'device' : 'cuda'}],
                      [{'M': 4}, {'N' : 5}, {'device' : 'cpu'}],
                      [{'M': 4}, {'N' : 5}, {'device' : 'cuda'}]]
    """
    generated_configs = []
    reserved_names = ["attrs", "attr_names", "tags"]
    if any(attr not in configs for attr in reserved_names):
        raise ValueError("Missing attrs in configs")

    _validate(configs)

    cross_configs = None
    if "cross_product_configs" in configs:
        cross_configs = cross_product_configs(**configs["cross_product_configs"])

    for inputs in configs["attrs"]:
        tmp_result = [
            {configs["attr_names"][i]: input_value}
            for i, input_value in enumerate(inputs)
        ]
        # TODO(mingzhe0908):
        # If multiple 'tags' were provided, do they get concat?
        # If a config has both ['short', 'medium'], it should match
        # both 'short' and 'medium' tag-filter?
        tmp_result.append({"tags": "_".join(configs["tags"])})
        if cross_configs:
            generated_configs += [tmp_result + list(config) for config in cross_configs]
        else:
            generated_configs.append(tmp_result)

    return generated_configs


def attr_probs(**probs):
    """return the inputs in a dictionary"""
    return probs


class RandomSample:
    def __init__(self, configs):
        self.saved_cum_distribution = {}
        self.configs = configs

    def _distribution_func(self, key, weights):
        """this is a cumulative distribution function used for random sampling inputs"""
        if key in self.saved_cum_distribution:
            return self.saved_cum_distribution[key]

        total = sum(weights)
        result = []
        cumsum = 0
        for w in weights:
            cumsum += w
            result.append(cumsum / total)
        self.saved_cum_distribution[key] = result
        return result

    def _random_sample(self, key, values, weights):
        """given values and weights, this function randomly sample values based their weights"""
        # TODO(mingzhe09088): cache the results to avoid recalculation overhead
        assert len(values) == len(weights)
        _distribution_func_vals = self._distribution_func(key, weights)
        x = random.random()
        idx = bisect.bisect(_distribution_func_vals, x)

        assert idx <= len(values), "Wrong index value is returned"
        # Due to numerical property, the last value in cumsum could be slightly
        # smaller than 1, and lead to the (index == len(values)).
        if idx == len(values):
            idx -= 1
        return values[idx]

    def get_one_set_of_inputs(self):
        tmp_attr_list = []
        for key, values in self.configs.items():
            if key in _reserved_keywords:
                continue
            value = self._random_sample(key, values, self.configs["probs"][str(key)])
            tmp_results = {key: value}
            tmp_attr_list.append(tmp_results)
        return tmp_attr_list


def random_sample_configs(**configs):
    """
    This function randomly sample <total_samples> values from the given inputs based on
    their weights.
    Here is an example showing what are the expected inputs and outputs from this function:
    M = [1, 2],
    N = [4, 5],
    K = [7, 8],
    probs = attr_probs(
        M = [0.7, 0.2],
        N = [0.5, 0.2],
        K = [0.6, 0.2],
    ),
    total_samples=10,
    this function will generate
    [
        [{'K': 7}, {'M': 1}, {'N': 4}],
        [{'K': 7}, {'M': 2}, {'N': 5}],
        [{'K': 8}, {'M': 2}, {'N': 4}],
        ...
    ]
    Note:
    The probs is optional. Without them, it implies everything is 1. The probs doesn't
    have to reflect the actual normalized probability, the implementation will
    normalize it.
    TODO (mingzhe09088):
    (1):  a lambda that accepts or rejects a config as a sample. For example: for matmul
    with M, N, and K, this function could get rid of (M * N * K > 1e8) to filter out
    very slow benchmarks.
    (2): Make sure each sample is unique. If the number of samples are larger than the
    total combinations, just return the cross product. Otherwise, if the number of samples
    is close to the number of cross-products, it is numerical safer to generate the list
    that you don't want, and remove them.
    """
    if "probs" not in configs:
        raise ValueError(
            "probs is missing. Consider adding probs or using other config functions"
        )

    configs_attrs_list = []
    randomsample = RandomSample(configs)
    for i in range(configs["total_samples"]):
        tmp_attr_list = randomsample.get_one_set_of_inputs()
        tmp_attr_list.append({"tags": "_".join(configs["tags"])})
        configs_attrs_list.append(tmp_attr_list)
    return configs_attrs_list


def op_list(**configs):
    """Generate a list of ops organized in a specific format.
    It takes two parameters which are "attr_names" and "attr".
    attrs stores the name and function of operators.
    Args:
        configs: key-value pairs including the name and function of
        operators. attrs and attr_names must be present in configs.
    Return:
        a sequence of dictionaries which stores the name and function
        of ops in a specifal format
    Example:
    attrs = [
        ["abs", torch.abs],
        ["abs_", torch.abs_],
    ]
    attr_names = ["op_name", "op"].

    With those two examples,
    we will generate (({"op_name": "abs"}, {"op" : torch.abs}),
                      ({"op_name": "abs_"}, {"op" : torch.abs_}))
    """
    generated_configs = []
    if "attrs" not in configs:
        raise ValueError("Missing attrs in configs")
    for inputs in configs["attrs"]:
        tmp_result = {
            configs["attr_names"][i]: input_value
            for i, input_value in enumerate(inputs)
        }
        generated_configs.append(tmp_result)
    return generated_configs


def is_caffe2_enabled(framework_arg):
    return "Caffe2" in framework_arg


def is_pytorch_enabled(framework_arg):
    return "PyTorch" in framework_arg


def get_operator_range(chars_range):
    """Generates the characters from chars_range inclusive."""
    if chars_range == "None" or chars_range is None:
        return None

    if all(item not in chars_range for item in [",", "-"]):
        raise ValueError(
            "The correct format for operator_range is "
            "<start>-<end>, or <point>, <start>-<end>"
        )

    ops_start_chars_set = set()
    ranges = chars_range.split(",")
    for item in ranges:
        if len(item) == 1:
            ops_start_chars_set.add(item.lower())
            continue
        start, end = item.split("-")
        for c in range(ord(start), ord(end) + 1):
            ops_start_chars_set.add(chr(c).lower())
    return ops_start_chars_set


def process_arg_list(arg_list):
    if arg_list == "None":
        return None

    return [fr.strip() for fr in arg_list.split(",") if len(fr.strip()) > 0]
