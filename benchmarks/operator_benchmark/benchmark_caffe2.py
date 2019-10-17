from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace
from caffe2.python import core
from caffe2.proto import caffe2_pb2
import benchmark_core
import benchmark_utils

"""Caffe2 performance microbenchmarks.

This module contains Caffe2-specific functionalities for performance
microbenchmarks.
"""


class Caffe2BenchmarkBase(object):
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
        blob_name = 'blob_' + str(Caffe2BenchmarkBase.tensor_index)
        dev = self._device_option(device)
        with core.DeviceScope(dev):
            workspace.FeedBlob(blob_name, benchmark_utils.numpy_random(dtype, *shapes))
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


class Caffe2OperatorTestCase(object):
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

    def run_forward(self, num_runs, print_per_iter=False):
        """ Run the forward path of an operator in a loop
        """
        with core.DeviceScope(self.op_bench.dev):
            op = self.op_bench.forward()
        if not workspace.RunOperatorMultiple(op, num_runs):
            raise ValueError("Unable to run operator test case: {}".format(self.test_name))

    def run_backward(self, num_runs):
        """ Run the backward path of an operator in a loop
        """
        with core.DeviceScope(self.op_bench.dev):
            op = self.op_bench.backward()
        if not workspace.RunOperatorMultiple(op, num_runs):
            raise ValueError("Unable to run operator gradient test case: {}".format(self.test_name))

    def _print_per_iter(self):
        pass


def register_caffe2_op_test_case(op_bench, test_config):
    test_case = Caffe2OperatorTestCase(op_bench, test_config)
    benchmark_core._register_test(test_case)
