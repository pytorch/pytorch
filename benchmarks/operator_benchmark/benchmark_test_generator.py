from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import copy 
from benchmark_core import TestConfig
from benchmark_caffe2 import register_caffe2_op_test_case
from benchmark_pytorch import register_pytorch_op_test_case
from benchmark_utils import SkipInputShape


def _generate_test(configs, bench_op, OperatorTestCase, run_backward, op_name_function=None):
    """Generate PyTorch/Caffe2 tests of operators with different inputs.
       Args: 
           configs: a dictionary that has the input shapes
           bench_op: a subclass of Caffe2BenchmarkBase/TorchBenchmarkBase which includes tensor
               creation and operator execution
           OperatorTestCase: a named tuple to save the metadata of an test 
           run_backward: a bool parameter indicating backward path 
           op_name_function: a dictionary includes operator name and function
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
        input_config = str(test_attrs)[1:-1].replace('\'', '')
        op = bench_op()
        tensor_error_info = None
        # op_name_function is a dictionary which has op_name and op_function.
        # an example of op_name_function is:
        # {'op_name' : 'abs', 'op_function' : torch.abs}
        # op_function is concatenated with the input dict then passed to the init function 
        # op_name is passed to the set_module_name function
        init_dict = copy.deepcopy(test_attrs)
        if op_name_function is not None:
            op_name = op_name_function['op_name']
            init_dict.update({'op_func' : op_name_function['op_func']})
            op.set_module_name(op_name)
        try:
            op.init(**init_dict)
        except SkipInputShape:
            print("Skipping: Config<{}> is not valid for op<{}>".format(input_config, op.module_name()))
            continue
        test_name = op.test_name(**test_attrs)
        input_config = str(test_attrs)[1:-1].replace('\'', '')
        test_config = TestConfig(test_name, input_config, tags, run_backward)
        if op is not None:
            OperatorTestCase(
                op,
                test_config)


OpMeta = namedtuple("OpMeta", "op_type num_inputs input_dims input_types \
                    output_dims num_outputs args")

def generate_c2_test_from_ops(ops_metadata, bench_op, tags):
    """
    This function is used to generate Caffe2 tests based on the meatdata
    of operators. The metadata includes seven fields which are 1) op_type: 
    the name of the operator. 2) num_inputs: the number of input blobs. 
    3) input_dims: a dictionary which includes the shapes of the input blobs.
    4) input_types: a list which includes the types of input blobs. 5) 
    output_dims: a dictionary which includes the shapes of output blobs.
    6) num_oupts: the number of output blobs. 7) args: a dictionary which
    includes the args for th operator. 
    Here is an example to show the metadata for the WeighedSum operator
    op_type : WeightedSum
    num_inputs: 4 
    input_dims: {'0': [256], '1': [1], '2': [256], '3': [1]}
    input_types: ['float', 'float', 'float', 'float']
    output_dims:  {'0': [256]} 
    num_outputs: 4 
    args: {}
    TODO(mingzhe0908): introduce device and add it to the benchmark name 
    """
    for op_metadata in ops_metadata:
        tmp_attrs = OpMeta(op_metadata.op_type, 
                           op_metadata.num_inputs, 
                           op_metadata.input_dims,
                           op_metadata.input_types, 
                           op_metadata.output_dims, 
                           op_metadata.num_outputs, 
                           op_metadata.args)
        test_attrs = tmp_attrs._asdict()
        op = bench_op()
        op.init(**test_attrs)
        test_name = op.test_name("short")
        input_config = "Shapes: {}, Type: {}, Args: {}".format(
            op_metadata.input_dims, 
            op_metadata.input_types,
            str(op_metadata.args))
        test_config = TestConfig(test_name, input_config, tags, run_backward=False)
        if op is not None:
            register_caffe2_op_test_case(
                op,
                test_config)


def generate_pt_test(configs, pt_bench_op):
    """ This function creates PyTorch op test based on the given operator
    """
    _generate_test(configs, pt_bench_op, register_pytorch_op_test_case, 
                   run_backward=False)


def generate_c2_test(configs, c2_bench_op):
    """ This function creates Caffe2 op test based on the given operator 
    """
    _generate_test(configs, c2_bench_op, register_caffe2_op_test_case, 
                   run_backward=False)


def generate_pt_gradient_test(configs, pt_bench_op):
    """ This function creates PyTorch op test based on the given operator
    """
    _generate_test(configs, pt_bench_op, register_pytorch_op_test_case, 
                   run_backward=True)


def generate_c2_gradient_test(configs, c2_bench_op):
    """ This function creates Caffe2 op test based on the given operator 
    """
    _generate_test(configs, c2_bench_op, register_caffe2_op_test_case, 
                   run_backward=True)


def generate_pt_tests_from_op_list(ops_list, configs, pt_bench_op):
    """ This function creates pt op tests one by one from a list of dictionaries.
        ops_list is a list of dictionary. Each dictionary includes 
        the name of the operator and the math operation. Here is an example of using this API:
        unary_ops_configs = op_bench.config_list(
            attrs=[...],
            attr_names=["M", "N"],
        )
        unary_ops_list = op_bench.op_list(
            attr_names=["op_name", "op_func"],
            attrs=[
                ["abs", torch.abs],
            ],
        )
        class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
            def init(self, M, N, op_name, op_func): 
                ... 
            def forward(self):
                ...
        op_bench.generate_pt_tests_from_op_list(unary_ops_list, unary_ops_configs, UnaryOpBenchmark)
    """
    for op in ops_list:
        _generate_test(configs, pt_bench_op, register_pytorch_op_test_case, 
                       run_backward=False, 
                       op_name_function=op)
