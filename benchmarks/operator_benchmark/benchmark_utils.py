from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import itertools
import random
import os


"""Performance microbenchmarks's utils.

This module contains utilities for writing microbenchmark tests.
"""


def shape_to_string(shape):
    return ', '.join([str(x) for x in shape])


def numpy_random(dtype, *shapes):
    """ Return a random numpy tensor of the provided dtype.
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
    existing_value = os.environ.get('OMP_NUM_THREADS', '')
    if existing_value != '': 
        print("Overwriting existing OMP_NUM_THREADS value: {}; Setting it to {}.".format(
            existing_value, num_threads))
    os.environ["OMP_NUM_THREADS"] = str(num_threads)


def set_mkl_threads(num_threads):
    existing_value = os.environ.get('MKL_NUM_THREADS', '')
    if existing_value != '': 
        print("Overwriting existing MKL_NUM_THREADS value: {}; Setting it to {}.".format(
            existing_value, num_threads))
    os.environ["MKL_NUM_THREADS"] = str(num_threads)


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
    """ Generate configs based on the list of input shapes.
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
    reserved_names = ['attrs', 'attr_names', 'tags']
    if any(attr not in configs for attr in reserved_names): 
        raise ValueError("Missing attrs in configs")

    cross_configs = None
    if 'cross_product_configs' in configs: 
        cross_configs = cross_product_configs(**configs['cross_product_configs'])

    for inputs in configs['attrs']:
        tmp_result = [{configs['attr_names'][i] : input_value} 
                      for i, input_value in enumerate(inputs)]
        # TODO(mingzhe0908): 
        # If multiple 'tags' were provided, do they get concat?
        # If a config has both ['short', 'medium'], it should match 
        # both 'short' and 'medium' tag-filter?
        tmp_result.append({'tags' : '_'.join(configs['tags'])})
        if cross_configs: 
            generated_configs += [tmp_result + list(config) for config in cross_configs]
        else: 
            generated_configs.append(tmp_result)

    return generated_configs


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
        tmp_result = {configs["attr_names"][i] : input_value
                      for i, input_value in enumerate(inputs)}
        generated_configs.append(tmp_result)
    return generated_configs


def is_caffe2_enabled(framework_arg):
    return 'Caffe2' in framework_arg


def is_pytorch_enabled(framework_arg):
    return 'PyTorch' in framework_arg


def process_arg_list(arg_list):
    if arg_list == 'None':
        return None 

    return [fr.strip() for fr in arg_list.split(',') if len(fr.strip()) > 0]


class SkipInputShape(Exception):
    """Used when a test case should be skipped"""
    pass
