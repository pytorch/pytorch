from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from benchmarks.operator_benchmark.benchmark_caffe2 import Caffe2OperatorTestCase
from benchmarks.operator_benchmark.benchmark_pytorch import PyTorchOperatorTestCase
from benchmarks.operator_benchmark.benchmark_utils import * # noqa


def generate_test(configs, map_config, ops, OperatorTestCase):
    """
    This function is used to create PyTorch/Caffe2 operators based on configs.
    configs usually include both long_config and short_config and they will be
    mapped to input_shapes and args which are ready to be digested by an operator.
    OperatorTestCase is used to create an operator with inputs/outputs and args.
    """
    for config in configs:
        for case in config:
            shapes = {}
            for item in case:
                if 'mode' in item:
                    run_mode = item['mode']
                    continue
                shapes.update(item)
            assert run_mode is not None, "Missing mode in configs"
            shapes_args = map_config(**shapes)
            if shapes_args is not None:
                for op in ops:
                    OperatorTestCase(
                        test_name=op[0],
                        op_type=op[1],
                        input_shapes=shapes_args[0],
                        op_args=shapes_args[1],
                        run_mode=run_mode)


def generate_pt_test(configs, pt_map_func, pt_ops):
    """
    This function creates PyTorch operators which will be benchmarked.
    """
    generate_test(configs, pt_map_func, pt_ops, PyTorchOperatorTestCase)


def generate_c2_test(configs, c2_map_func, c2_ops):
    """
    This function creates Caffe2 operators which will be benchmarked.
    """
    generate_test(configs, c2_map_func, c2_ops, Caffe2OperatorTestCase)


def map_c2_config_add(M, N, K):
    input_one = (M, N, K)
    input_two = (M, N, K)
    input_shapes = [input_one, input_two]
    args = {}
    return (input_shapes, args)

map_pt_config_add = map_c2_config_add


def map_c2_config_matmul(M, N, K, trans_a, trans_b):
    input_one = (N, M) if trans_a else (M, N)
    input_two = (K, N) if trans_b else (N, K)
    input_shapes = [input_one, input_two]
    args = {'trans_a': trans_a, 'trans_b': trans_b}
    return (input_shapes, args)


def map_pt_config_matmul(M, N, K, trans_a, trans_b):
    input_one = (N, M) if trans_a else (M, N)
    input_two = (K, N) if trans_b else (N, K)
    input_shapes = [input_one, input_two]
    args = {}
    if not trans_a and not trans_b:
        return (input_shapes, args)
    return None
