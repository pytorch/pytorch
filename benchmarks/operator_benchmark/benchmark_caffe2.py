from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace
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

    def __init__(self):
        self.args = {}
        self.user_provided_name = None

    # TODO: Add other dtype support
    def tensor(self, *shapes):
        """ A wapper function to create tensor (blob in caffe2) filled with random
            value. The name/label of the tensor is returned and it is available 
            throughout the benchmark execution phase. 
        """
        blob_name = 'blob_' + str(Caffe2BenchmarkBase.tensor_index)
        workspace.FeedBlob(blob_name, benchmark_utils.numpy_random_fp32(*shapes))
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

    def test_name(self, **kargs):
        """ FIXME(mingzhe0908):
            this is a globally unique name which can be used to
            label a specific test
        """
        test_name_str = []
        for key in kargs:
            value = kargs[key]
            test_name_str.append(
                key + self._value_to_str(value))
        return '_'.join(test_name_str)


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

    def run_forward(self, num_runs):
        """ Run the forward path of an operator in a loop
        """
        if not workspace.RunOperatorMultiple(self.op_bench.forward(), num_runs):
            raise ValueError("Unable to run operator test case: {}".format(self.test_name))

    def run_backward(self, num_runs):
        """ Run the backward path of an operator in a loop
        """
        if not workspace.RunOperatorMultiple(self.op_bench.backward(), num_runs):
            raise ValueError("Unable to run operator gradient test case: {}".format(self.test_name))


def register_caffe2_op_test_case(op_bench, test_config):
    test_case = Caffe2OperatorTestCase(op_bench, test_config)
    benchmark_core._register_test(test_case)
