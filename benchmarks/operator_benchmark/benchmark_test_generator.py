from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from benchmark_core import TestConfig
from benchmark_caffe2 import register_caffe2_op_test_case
from benchmark_pytorch import register_pytorch_op_test_case


def generate_test(configs, bench_op, OperatorTestCase, run_backward):
    """
    This function is used to generate PyTorch/Caffe2 tests based on 
    the configs and operators.
    TODO(mingzhe0908): introduce device and add it to the benchmark name 
    """
    for config in configs:
        test_attrs = {}
        tags = None
        for attr in config:
            # tags is only used in our benchmark backend to filter tests and
            # it will be removed from config which is then passed to the init function 
            # an example of config and atrr is: 
            # config: [{'M': 16}, {'N': 16}, {'K': 64}, {'tags': 'short'}]
            # attr: {'tags': 'short'} 
            if "tags" in attr:
                tags = attr["tags"]
                continue
            test_attrs.update(attr)
        if tags is None:
            raise ValueError("Missing tags in configs")
        op = bench_op()
        op.init(**test_attrs)
        test_name = op.test_name(**test_attrs)
        input_config = str(test_attrs)[1:-1].replace('\'', '')
        test_config = TestConfig(test_name, input_config, tags, run_backward)
        if op is not None:
            OperatorTestCase(
                op,
                test_config)


def generate_pt_test(configs, pt_bench_op):
    """ This function creates PyTorch op test based on the given operator
    """
    generate_test(configs, pt_bench_op, register_pytorch_op_test_case, run_backward=False)


def generate_c2_test(configs, c2_bench_op):
    """ This function creates Caffe2 op test based on the given operator 
    """
    generate_test(configs, c2_bench_op, register_caffe2_op_test_case, run_backward=False)


def generate_pt_gradient_test(configs, pt_bench_op):
    """ This function creates PyTorch op test based on the given operator
    """
    generate_test(configs, pt_bench_op, register_pytorch_op_test_case, run_backward=True)


def generate_c2_gradient_test(configs, c2_bench_op):
    """ This function creates Caffe2 op test based on the given operator 
    """
    generate_test(configs, c2_bench_op, register_caffe2_op_test_case, run_backward=True)
