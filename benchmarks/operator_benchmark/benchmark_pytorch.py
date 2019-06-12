from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import benchmark_core


"""PyTorch performance microbenchmarks.

This module contains PyTorch-specific functionalities for performance
microbenchmarks.
"""

class TorchBenchmarkBase(object):
    """ This is a base class used to create Pytorch operator benchmark.
        module_name is the name of the operator being benchmarked. 
        test_name is the name (it's created by concatenating all the 
        inputs) of a specific test
    """

    def __init__(self):
        self.user_given_name = None

    def forward(self):
        pass 

    def module_name(self):
        """ this is used to label the operator being benchmarked
        """
        if self.user_given_name:
            return self.user_given_name 
        return self.__class__.__name__

    def set_module_name(self, name): 
        self.user_given_name = name

    def test_name(self, **kargs):
        """ this is a globally unique name which can be used to 
            label a specific test 
        """
        test_name_str = []
        for key in kargs:
            value = kargs[key]
            test_name_str.append(
                key + str(value if type(value) != bool else int(value)))
        name = (self.module_name() + '_' +
                '_'.join(test_name_str)).replace(" ", "")
        return name


class PyTorchOperatorTestCase(object):
    """ This class includes all the information needed to benchmark an operator. 
        op_bench: it's a user-defined class (child of TorchBenchmarkBase)
        which includes input and operator, .etc
        test_config: a namedtuple includes test_name, input_shape, tag, run_backward.
        When run_backward is false, the run_forward method will be executed, 
        When run_backward is true, run_forward_eager and _output_mean will be 
        executed to generate output. Then, run_backward will be executed.
    """
    def __init__(self, op_bench, test_config):
        self.test_config = test_config
        self.op_bench = op_bench
        self.framework = "PyTorch"

    def run_jit_forward(self, num_runs):
        """ This is a temp solution and will be removed later 
            Run the forward op with JIT 
        """
        self.op_bench.jit_forward(num_runs)

    def run_forward(self, num_runs):
        """ TODO (mingzhe): when JIT is ready, switch this to JIT 
            Run the forward path of an op in many iterations
        """
        for _ in range(num_runs):
            self.op_bench.forward()

    def _output_mean(self):
        """ TODO (mingzhe): it is not necessary to sum up everything by myself, 
            torch.autograd.backward do take a gradient tensor. By default, it 
            is the same shape as your output tensor, with all 1s. 
            Mathematically, it is the same as if the output is summed together. 
            So we should be able to get ride of this method. 
            dummy function for gradient calculation
        """
        self.mean = self.output.mean()

    def run_backward(self, num_runs):
        """ Run the backward path of an op in many iterations
        """
        # TODO: can we use JIT here to reduce python overhead?
        for _ in range(num_runs):
            self.mean.backward(retain_graph=True)


def register_pytorch_op_test_case(op_bench, test_config):
    test_case = PyTorchOperatorTestCase(op_bench, test_config)
    benchmark_core._register_test(test_case)
