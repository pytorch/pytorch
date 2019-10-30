from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy 
import ast
import json
from benchmark_core import TestConfig
from benchmark_pytorch import register_pytorch_op_test_case
from benchmark_utils import SkipInputShape


def _register_test(bench_op_obj, orig_test_attrs, tags, OperatorTestCase, run_backward, bwd_input): 
    """ Register tests with the benchmark backend. 
        Args: 
            bench_op_obj: an object which instantiated from a subclass of 
                Caffe2BenchmarkBase/TorchBenchmarkBase which includes tensor
                creation and operator execution.
            test_attrs: a dictionary includes test configs. 
            tags: a attribute in test config to filter inputs 
            OperatorTestCase: a named tuple to save the metadata of an test
            run_backward: a bool parameter indicating backward path
    """
    test_attrs = copy.deepcopy(orig_test_attrs)
    test_attrs = {k: str(v) for k, v in test_attrs.items()}
    ascii_test_attrs = ast.literal_eval(json.dumps(test_attrs))
    input_config = str(ascii_test_attrs)[1:-1].replace('\'', '')
    if bwd_input: 
        # When auto_set is used, the test name needs to include input.  
        test_attrs.update({'bwd': bwd_input})
    test_name = bench_op_obj.test_name(**test_attrs)
    test_config = TestConfig(test_name, input_config, tags, run_backward)
    OperatorTestCase(bench_op_obj, test_config)

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
        assert op is not None, "Can't create test"
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

        op._set_backward_test(run_backward)
        try:
            op.init(**init_dict)
        except SkipInputShape:
            print("Skipping: Config<{}> is not valid for op<{}>".format(input_config, op.module_name()))
            continue

        input_name = None

        # _num_inputs_require_grads is used to track the number of tensors 
        # which use auto_set().
        if op._num_inputs_require_grads > 0: 
            input_name = 'all'
        _register_test(op, test_attrs, tags, OperatorTestCase, run_backward, input_name)

        # This for loop is only used when auto_set is used. 
        # _pass_count counts how many times init has been called. 
        # _auto_set_counter is reset after init is called. 
        for i in range(op._num_inputs_require_grads):
            op._pass_count += 1
            op._auto_set_counter = 0

            # TODO(mingzhe09088): remove this deepcopy when we encounter 
            # performance issue. 
            new_op = copy.deepcopy(op)
            new_op.init(**init_dict)
            # Input name index will start from input1
            input_name = i + 1
            _register_test(new_op, test_attrs, tags, OperatorTestCase, run_backward, input_name)


def generate_pt_test(configs, pt_bench_op):
    """ This function creates PyTorch op test based on the given operator
    """
    _generate_test(configs, pt_bench_op, register_pytorch_op_test_case,
                   run_backward=False)


def generate_pt_gradient_test(configs, pt_bench_op):
    """ This function creates PyTorch op test based on the given operator
    """
    _generate_test(configs, pt_bench_op, register_pytorch_op_test_case,
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
