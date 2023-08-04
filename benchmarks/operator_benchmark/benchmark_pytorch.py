import json
import time

import benchmark_cpp_extension  # noqa: F401
import torch


"""PyTorch performance microbenchmarks.

This module contains PyTorch-specific functionalities for performance
microbenchmarks.
"""


class TorchBenchmarkBase(torch.nn.Module):
    """This is a base class used to create Pytorch operator benchmark.
    module_name is the name of the operator being benchmarked.
    test_name is the name (it's created by concatenating all the
    inputs) of a specific test
    """

    def __init__(self):
        super().__init__()
        self.user_given_name = None
        self._pass_count = 0
        self._num_inputs_require_grads = 0

    def _set_backward_test(self, is_backward):
        self._is_backward = is_backward

    def auto_set(self):
        """This is used to automatically set the require_grad for the backward patch.
        It is implemented based on two counters. One counter to save the number of
        times init has been called. The other counter to save the number of times
        this function itself has been called. In the very first time init is called,
        this function counts how many inputs require gradient. In each of the
        following init calls, this function will return only one true value.
        Here is an example:
            ...
            self.v1 = torch.rand(M, N, K, requires_grad=self.auto_set())
            self.v2 = torch.rand(M, N, K, requires_grad=self.auto_set())
            ...
        """
        if not self._is_backward:
            return False

        if self._pass_count == 0:
            self._num_inputs_require_grads += 1
            return True
        else:
            self._auto_set_counter += 1
            return self._pass_count == self._auto_set_counter

    def extract_inputs_tuple(self):
        self.inputs_tuple = tuple(self.inputs.values())

    @torch.jit.export
    def get_inputs(self):
        # Need to convert the inputs to tuple outside of JIT so that
        # JIT can infer the size of the inputs.
        return self.inputs_tuple

    @torch.jit.export
    def forward_impl(self):
        # This is to supply the inputs to the forward function which
        # will be called in both the eager and JIT mode of local runs
        return self.forward(*self.get_inputs())

    @torch.jit.export
    def forward_consume(self, iters: int):
        #  _consume is used to avoid the dead-code-elimination optimization
        for _ in range(iters):
            torch.ops.operator_benchmark._consume(self.forward_impl())

    def module_name(self):
        """this is used to label the operator being benchmarked"""
        if self.user_given_name:
            return self.user_given_name
        return self.__class__.__name__

    def set_module_name(self, name):
        self.user_given_name = name

    def test_name(self, **kargs):
        """this is a globally unique name which can be used to
        label a specific test
        """

        # This is a list of attributes which will not be included
        # in the test name.
        skip_key_list = ["device"]

        test_name_str = []
        for key in kargs:
            value = kargs[key]
            test_name_str.append(
                ("" if key in skip_key_list else key)
                + str(value if type(value) != bool else int(value))
            )
        name = (self.module_name() + "_" + "_".join(test_name_str)).replace(" ", "")
        return name


class PyTorchOperatorTestCase:
    """This class includes all the information needed to benchmark an operator.
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
        self.place_holder_tensor = torch.ones(1)
        self.framework = "PyTorch"
        self.time_series = []
        self._jit_forward_graph = None

    def _generate_jit_forward_graph(self):
        """generate a graph for the forward function via scripting"""
        scripted_op_bench = torch.jit.script(self.op_bench)
        return scripted_op_bench.forward_consume

    def run_jit_forward(self, num_runs, print_per_iter=False, cuda_sync=False):
        """Run the forward path of an op with JIT mode"""
        if self._jit_forward_graph is None:
            self._jit_forward_graph = self._generate_jit_forward_graph()
        self._jit_forward_graph(num_runs)

    def _print_per_iter(self):
        # print last 50 values
        length = min(len(self.time_series), 50)
        for i in range(length):
            print(
                "PyTorchObserver "
                + json.dumps(
                    {
                        "type": self.test_config.test_name,
                        "metric": "latency",
                        "unit": "ms",
                        "value": str(self.time_series[length - i - 1]),
                    }
                )
            )

    def run_forward(self, num_runs, print_per_iter, cuda_sync):
        """Run the forward path of an op with eager mode"""
        if print_per_iter:
            for _ in range(num_runs):
                start_time = time.time()
                self.output = self.op_bench.forward_impl()
                if cuda_sync:
                    torch.cuda.synchronize(torch.cuda.current_device())
                end_time = time.time()
                self.time_series.append((end_time - start_time) * 1e3)
        else:
            for _ in range(num_runs):
                self.output = self.op_bench.forward_impl()
            if cuda_sync:
                torch.cuda.synchronize(torch.cuda.current_device())

    def _output_mean(self):
        """TODO (mingzhe): it is not necessary to sum up everything by myself,
        torch.autograd.backward do take a gradient tensor. By default, it
        is the same shape as your output tensor, with all 1s.
        Mathematically, it is the same as if the output is summed together.
        So we should be able to get ride of this method.
        dummy function for gradient calculation
        """
        self.mean = self.output.mean()

    def run_backward(self, num_runs, print_per_iter=False):
        """Run the backward path of an op in many iterations"""
        # TODO: can we use JIT here to reduce python overhead?
        for _ in range(num_runs):
            self.mean.backward(retain_graph=True)


def create_pytorch_op_test_case(op_bench, test_config):
    """This method is used to generate est. func_name is a global unique
    string. For PyTorch add operator with M=8, N=2, K=1, tag = long, here
    are the values for the members in test_case:
    op.module_name: add
    framework: PyTorch
    test_config: TestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
        tag='long', run_backward=False)
    func_name: addPyTorchTestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
                                    tag='long', run_backward=False)
    """
    test_case = PyTorchOperatorTestCase(op_bench, test_config)
    test_config = test_case.test_config
    op = test_case.op_bench
    func_name = f"{op.module_name()}{test_case.framework}{str(test_config)}"
    return (func_name, test_case)
