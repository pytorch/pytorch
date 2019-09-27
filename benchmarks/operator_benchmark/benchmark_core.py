from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import numpy as np
import timeit
import json
import torch

# needs to be imported after torch
import cpp_extension # noqa

import benchmark_utils
from collections import namedtuple

"""Performance microbenchmarks.

This module contains core functionalities for performance microbenchmark tests.
"""

"""
This is used to store configs of tests
An example input is:
TestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
    tag='long', run_backward=False)
"""
TestConfig = namedtuple("TestConfig", "test_name input_config tag run_backward")


BENCHMARK_TESTER = {}


def _register_test(test_case):
    """ This method is used to register test. func_name is a global unique
    string. For PyTorch add operator with M=8, N=2, K=1, tag = long, here
    are the values for the members in test_case:
    op.module_name: add
    framework: PyTorch
    test_config: TestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
        tag='long', run_backward=False)
    func_name: addPyTorchTestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
                                    tag='long', run_backward=False)
    """
    test_config = test_case.test_config
    op = test_case.op_bench
    func_name = "{}{}{}".format(op.module_name(), test_case.framework, str(test_config))
    BENCHMARK_TESTER[func_name] = test_case


class BenchmarkRunner(object):
    """BenchmarkRunner is responsible for benchmarking all the registered
    benchmark test groups.

    Attributes:
        tag_filter (str): control the benchmarks which matches the tag.
        operator (str): only run benchmark test cases that contains
    this filter string in the test case's id.
        test_name (str): only run benchmark test cases that matches this filter,
        this is a case-sensitive substring match and it happens in
        the _keep_test method.
    """
    def __init__(self, args):
        # TODO: consider time-bound constraints as well.
        self.args = args
        self.iters = 200
        self.has_explicit_iteration_count = False
        self.multiplier = 2
        self.predefined_minimum_secs = 4
        self.max_iters = 1e6
        self.use_jit = args.use_jit
        self.num_runs = args.num_runs
        if self.args.iterations:
            self.has_explicit_iteration_count = True
            self.iters = self.args.iterations
        # when a specific test is selected by a user, we don't need
        # to match the tag anymore
        if self.args.test_name is not None:
            self.args.tag_filter = None

    def _print_header(self):
        DASH_LINE = '-' * 40
        print("# {}\n"
              "# PyTorch/Caffe2 Operator Micro-benchmarks\n"
              "# {}\n"
              "# Tag : {}\n".format(DASH_LINE, DASH_LINE, self.args.tag_filter))
        if self.args.list_tests:
            print("# List of tests:")
            for _, test_case in BENCHMARK_TESTER.items():
                print("# {}".format(test_case.test_config.test_name))
        elif self.args.list_ops:
            print("# List of Operators to run:")
            if self.args.operators is None:
                ops = set(test_case.op_bench.module_name()
                          for _, test_case in BENCHMARK_TESTER.items())
                for op in ops:
                    print("# {}".format(op))
            else:
                print("# {}".format(self.args.operators))

    def _print_perf_result(self, reported_run_time_us, test_case):
        if self.args.ai_pep_format:
            # Output for AI-PEP
            test_name = '_'.join([test_case.framework, test_case.test_config.test_name])
            for run in range(self.num_runs):
                print("{}Observer ".format(test_case.framework) + json.dumps(
                    {
                        "type": test_name,
                        "metric": "latency",
                        "unit": "us",
                        "value": str(reported_run_time_us[run]),
                    }
                ))
        else:
            if test_case.framework == "PyTorch":
                print("# Mode: {}".format("JIT" if self.use_jit else "Eager"))

            print("# Name: {}\n"
                  "# Input: {}".format(
                      test_case.test_config.test_name,
                      test_case.test_config.input_config))

            mode = "Backward" if test_case.test_config.run_backward else "Forward"
            if self.num_runs > 1:
                for run in range(self.num_runs):
                    print("Run: {}, {} Execution Time (us) : {:.3f}".format(
                        run,
                        mode, reported_run_time_us[run]))
                print()
            else:
                print("{} Execution Time (us) : {:.3f}\n".format(
                    mode, reported_run_time_us[0]))

    def _predict_num_iter_needed(self, i):
        return (i * self.multiplier)

    def _iteration_result_is_significant(self, iters, run_time_sec, curr_test_total_time, has_explicit_iteration_count):
        """ This function decides whether the measured time can be reported based on the
        following conditions: 1) the number of iterations is larger than the max_iters.
        2) the execution time is larger than the predefined minimum_time
        3) the execution time is larger than user defined minimum_time
        """
        return ((iters > self.max_iters or
                run_time_sec > self.predefined_minimum_secs or
                has_explicit_iteration_count) and
                curr_test_total_time > self.args.min_time_per_test)

    def _launch_forward(self, test_case, iters):
        """ Use Python's timeit module to measure execution time (unit: second).
        """
        func = test_case.run_forward
        if self.use_jit:
            func = test_case.run_jit_forward
        forward_time = timeit.timeit(functools.partial(func, iters), number=1)
        return forward_time

    def _launch_backward(self, test_case, iters):
        """ This function runs forward path of an op to get an output. Then the backward path is executed
        and the execution time is reported
        """
        test_case.run_forward(num_runs=1)
        if test_case.framework == "PyTorch":
            test_case._output_mean()
        backward_time = timeit.timeit(functools.partial(test_case.run_backward, iters), number=1)
        return backward_time

    def _measure_time(self, launch_test, test_case, iters):
        """
        This function execute the operator for <iters> iterations then look at the time.
        If it's not significant, the number of iterations will be increased before rerun.
        The execution stops when the time becomes significant.
        """
        curr_test_total_time = 0
        while True:
            # Wipe cache
            if self.args.wipe_cache:
                torch.ops.operator_benchmark._clear_cache()

            run_time_sec = launch_test(test_case, iters)
            curr_test_total_time += run_time_sec
            # Analyze time after each run to decide if the result is stable
            results_are_significant = self._iteration_result_is_significant(
                iters, run_time_sec, curr_test_total_time, self.has_explicit_iteration_count)

            if results_are_significant:
                break

            # Re-estimate the hopefully-sufficient
            # iteration count, and run the benchmark again...
            iters = self._predict_num_iter_needed(iters)

        reported_run_time_us = (1e6 * run_time_sec / iters)
        return reported_run_time_us

    def _check_keep(self, test_flag, cmd_flag):
        return (cmd_flag is None or test_flag == cmd_flag)

    def _check_keep_list(self, test_flag, cmd_flag_list):
        if (cmd_flag_list is None or
                any(test_flag == cmd_flag for cmd_flag in cmd_flag_list)):
            return True
        return False

    def _keep_test(self, test_case):
        # TODO: consider regex matching for test filtering.
        # Currently, this is a sub-string matching.
        op_test_config = test_case.test_config

        if self.args.framework:
            frameworks = benchmark_utils.process_arg_list(self.args.framework)

        operators = benchmark_utils.process_arg_list(self.args.operators) if self.args.operators else None

        # Filter framework, operator, test_name, tag, forward_only
        if (self._check_keep(op_test_config.test_name, self.args.test_name) and
            self._check_keep(op_test_config.tag, self.args.tag_filter) and
            self._check_keep_list(test_case.op_bench.module_name(), operators) and
            self._check_keep_list(test_case.framework, frameworks) and
                (not self.args.forward_only or op_test_config.run_backward != self.args.forward_only)):
            return True

        return False

    def run(self):
        self._print_header()

        if self.args.list_ops or self.args.list_tests:
            return

        for full_test_id, test_case in BENCHMARK_TESTER.items():
            op_test_config = test_case.test_config

            if not self._keep_test(test_case):
                continue

            # To reduce variance, fix a numpy randseed to the test case,
            # so that the randomly generated input tensors remain the
            # same for each test case.
            # The random seed is limited to 32-bit because of numpy
            # requirement.
            np.random.seed(seed=hash(full_test_id) & ((1 << 32) - 1))

            print("# Benchmarking {}: {}".format(
                test_case.framework,
                test_case.op_bench.module_name()))

            if op_test_config.run_backward:
                launch_func = self._launch_backward
            else:
                launch_func = self._launch_forward

            # Warmup
            launch_func(test_case, self.args.warmup_iterations)
            # Actual Execution
            reported_time = [self._measure_time(launch_func, test_case, self.iters)
                             for _ in range(self.num_runs)]

            self._print_perf_result(reported_time, test_case)
