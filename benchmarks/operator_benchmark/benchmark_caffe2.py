from caffe2.python import workspace
from caffe2.python import core
from caffe2.proto import caffe2_pb2
import benchmark_utils
from collections import namedtuple
from benchmark_test_generator import _register_test

"""Caffe2 performance microbenchmarks.

This module contains Caffe2-specific functionalities for performance
microbenchmarks.
"""


class Caffe2BenchmarkBase:
    """ This is a base class used to create Caffe2 operator benchmark
    """
    tensor_index = 0
    test_index = 0

    def __init__(self):
        self.args = {}
        self.user_provided_name = None
        self._num_inputs_require_grads = 0
        self._pass_count = 0

    def _set_backward_test(self, is_backward):
        pass

    def _device_option(self, device):
        """ This method is used to set device option.
        """
        if device not in ['cuda', 'cpu']:
            raise ValueError("Missing attrs in configs")

        if 'cuda' in device:
            self.dev = core.DeviceOption(caffe2_pb2.CUDA, 0)
        else:
            self.dev = core.DeviceOption(caffe2_pb2.CPU)
        return self.dev

    def tensor(self, shapes, dtype='float32', device='cpu'):
        """ A wapper function to create C2 tensor filled with random data.
            The name/label of the tensor is returned and it is available
            throughout the benchmark execution phase.
            Args:
                shapes: int or a sequence of ints to defining the shapes of the tensor
                dtype: use the dtypes from numpy
                    (https://docs.scipy.org/doc/numpy/user/basics.types.html)
            Return:
                C2 tensor of dtype
        """
        return self.feed_tensor(benchmark_utils.numpy_random(dtype, *shapes), device)

    def feed_tensor(self, tensor, device='cpu'):
        """ Similar to tensor, but can supply any data compatible with FeedBlob
        """
        blob_name = 'blob_' + str(Caffe2BenchmarkBase.tensor_index)
        dev = self._device_option(device)
        with core.DeviceScope(dev):
            workspace.FeedBlob(blob_name, tensor)
        Caffe2BenchmarkBase.tensor_index += 1
        return blob_name

    def module_name(self):
        """ this is used to label the operator being benchmarked
        """
        if self.user_provided_name:
            return self.user_provided_name
        return self.__class__.__name__

    def set_module_name(self, name):
        self.user_provided_name = name

    def _value_to_str(self, value):
        """ if value is bool, we will convert it to 0 and 1
        """
        ret = value
        if type(value) == bool:
            ret = int(value)
        return str(ret)

    def test_name(self, name_type="long", **kargs):
        """ this is a globally unique name which can be used to
            label a specific test
        """
        if name_type == "long":
            test_name_str = []
            for key in kargs:
                value = kargs[key]
                test_name_str.append(
                    key + self._value_to_str(value))
            name = (self.module_name() + '_' +
                    '_'.join(test_name_str)).replace(" ", "")
        elif name_type == "short":
            # this is used to generate test name based on unique index
            name = '_'.join([self.module_name(), 'test', str(Caffe2BenchmarkBase.test_index)])
            Caffe2BenchmarkBase.test_index += 1
        return name

    def extract_inputs_tuple(self):
        # add a dummy function here to match the interface of TorchBenchmarkBase
        pass


class Caffe2OperatorTestCase:
    """ This class includes all the information needed to benchmark an operator.
        op_bench: it's a user-defined class (child of Caffe2BenchmarkBase)
        which includes input and operator, .etc
        test_config: a namedtuple includes test_name, input_shape, tag, run_backward.
        When run_backward is false, the run_forward method will be executed, otherwise
        run_backward method will be executed.
    """
    def __init__(self, op_bench, test_config):
        self.op_bench = op_bench
        self.test_config = test_config
        self.framework = "Caffe2"

    def run_forward(self, num_runs, print_per_iter=False, cuda_sync=False):
        """ Run the forward path of an operator in a loop
        """
        with core.DeviceScope(self.op_bench.dev):
            op = self.op_bench.forward()
        if not workspace.RunOperatorMultiple(op, num_runs):
            raise ValueError(f"Unable to run operator test case: {self.test_name}")

    def run_backward(self, num_runs, print_per_iter=False):
        """ Run the backward path of an operator in a loop
        """
        with core.DeviceScope(self.op_bench.dev):
            op = self.op_bench.backward()
        if not workspace.RunOperatorMultiple(op, num_runs):
            raise ValueError(f"Unable to run operator gradient test case: {self.test_name}")

    def _print_per_iter(self):
        pass


def create_caffe2_op_test_case(op_bench, test_config):
    test_case = Caffe2OperatorTestCase(op_bench, test_config)
    test_config = test_case.test_config
    op = test_case.op_bench
    func_name = f"{op.module_name()}{test_case.framework}{str(test_config)}"
    return (func_name, test_case)


OpMeta = namedtuple("OpMeta", "op_type num_inputs input_dims input_types \
                    output_dims num_outputs args device")

def generate_c2_test_from_ops(ops_metadata, bench_op, tags):
    """
    This function is used to generate Caffe2 tests based on the metadata
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
                           op_metadata.args,
                           op_metadata.device)
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
            create_caffe2_op_test_case(
                op,
                test_config)


def generate_c2_test(configs, c2_bench_op):
    """ This function creates Caffe2 op test based on the given operator
    """
    return _register_test(configs, c2_bench_op, create_caffe2_op_test_case,
                          False)


def generate_c2_gradient_test(configs, c2_bench_op):
    """ This function creates Caffe2 op test based on the given operator
    """
    return _register_test(configs, c2_bench_op, create_caffe2_op_test_case,
                          True)
