from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import numpy as np
import timeit
import json

from operator_benchmark import benchmark_utils

"""Performance microbenchmarks.

This module contains core functionalities for performance microbenchmark tests.
"""


# List of run modes we support.
# Each benchmark test case is associated with a run mode.
# If the value of the test case's run mode is less than the value of the
# benchmark binary's run mode, the test case will be executed, e.g. a short-mode
# test case will be executed when the binary is on either long and short
# modes; while a long-mode test case will only be executed when the binary is
# on long-mode.
RUN_MODES = {'short': 0, 'long': 1}
BENCHMARK_TESTER = [{} for _ in range(len(RUN_MODES))]
BENCHMARK_TEST_GROUP = {}


def add_benchmark_tester(framework, op_name, input_shapes, op_args, run_mode, func):
    func_name = "__".join([framework, op_name, benchmark_utils.shape_to_string(input_shapes)
                          , str(op_args), run_mode])
    run_mode = RUN_MODES[run_mode]
    for mode in RUN_MODES.values():
        # short mode runs with some of the input shapes for an op
        # long mode runs with all the input shapes for an op
        if (mode < run_mode):
            continue
        BENCHMARK_TESTER[mode][func_name] = func


def register_test(func):
    """Decorator to register a benchmark test group.
    A benchmark test group is a function that returns a list of benchmark test
    case objects to be run.
    """
    BENCHMARK_TEST_GROUP[__name__ + "." + func.__name__] = func
    return func


HEADER_LINE = """
# {}
# PyTorch/Caffe2 Operator Micro-benchmarks
# {}
# Run_mode : {}
"""


class BenchmarkRunner(object):
    """BenchmarkRunner is responsible for benchmarking all the registered
    benchmark test groups.

    Attributes:
        run_mode (str): Must of one of 'short', 'long'. For long mode, the
    benchmark runner takes a longer time to run since it repeats each benchmark
    test case more times to reduce measured variance, and it also executes
    longer running test cases that is marked as long mode.
        operator (str): Only run benchmark test cases that contains
    this filter string in the test case's id.
    """
    def __init__(self, args):
        # Depend on the run mode, set the execution contrains based of number of
        # runs per measure, and number of measures.
        # TODO: consider time-bound constraints as well.
        self.args = args
        self.iters = 100
        self.has_explicit_iteration_count = False
        self.multiplier = 2
        self.min_time = 0.8
        self.max_iters = 1e6
        for test_group in BENCHMARK_TEST_GROUP.items():
            test_group_func = test_group[1]
            test_group_func()
        if self.args.iterations:
            self.has_explicit_iteration_count = True
            self.iters = self.args.iterations

    def _print_header(self, run_mode):
        DASH_LINE = '-' * 40
        print(HEADER_LINE.format(DASH_LINE, DASH_LINE, self.args.run_mode, self.iters))
        print("# List of Operators to run:")
        if self.args.operator is None:
            ops = set()
            for tester in BENCHMARK_TESTER[run_mode].items():
                full_test_id = tester[0]
                framework, op_name, input_shapes, args, run_mode = full_test_id.split("__")
                if op_name not in ops:
                    print("# {}".format(op_name))
                    ops.add(op_name)
        else:
            print("# {}".format(self.args.operator))
        print("\n")

    def _print_perf_result(self, full_test_id, input_shapes, args, reported_run_time):
        if self.args.ai_pep_format:
            # Output for AI-PEP
            print("Caffe2Observer " + json.dumps(
                {
                    "type": "NET",
                    "metric": full_test_id,
                    "unit": "ms",
                    "value": str(reported_run_time),
                }
            ))
        else:
            print("# Input Shape: {}\n"
                  "Execution Time (us) : {:.3f} \n"
                  .format(input_shapes, reported_run_time))

    def _predict_num_iter_needed(self, i):
        return (i * self.multiplier)

    def _report_iteration_result(self, iters, run_time):
        return (iters > self.max_iters or
                run_time > 5 * self.min_time)

    def run(self):
        run_mode = RUN_MODES[self.args.run_mode]
        self._print_header(run_mode)

        if self.args.list_tests:
            return

        for tester in BENCHMARK_TESTER[run_mode].items():
            full_test_id = tester[0]
            benchmark_func = tester[1]
            framework, op_name, input_shapes, args, run_mode = full_test_id.split("__")
            # TODO: consider regex matching for test filtering.
            # Currently, this is a sub-string matching.
            if self.args.operator and (self.args.operator not in full_test_id):
                continue
            if self.args.framework:
                frameworks = benchmark_utils.get_requested_frameworks(self.args.framework)
                if all([fr not in full_test_id for fr in frameworks]):
                    continue

            # To reduce variance, fix a numpy randseed to the test case,
            # so that the randomly generated input tensors remain the
            # same for each test case.
            # The random seed is limited to 32-bit because of numpy
            # requirement.
            np.random.seed(seed=hash(full_test_id) & ((1 << 32) - 1))

            print("# Benchmarking {} {}".format(
                framework,
                op_name))
            # Warmup
            functools.partial(benchmark_func, self.args.warmup_iterations)

            # Actual Execution
            run_time = 0
            iters = self.iters
            while True:
                # Use Python's timeit module to measure execution time (unit: second).
                # Each experiment consists of repeated execution of
                # the benchmark_func a number of times (self.iters)
                # because otherwise the duration is too short to get
                # an accurate measure. The benchmark loop is pushed
                # to C++ to minimize Python overhead.
                # The experiment is also repeated a number of times
                # (num_repeats) and we then take the minimum execution
                # time as the final measurement result (this is also
                # recommended by timeit's doc).
                run_time = min(timeit.repeat(functools.partial(benchmark_func, iters),
                               repeat=1, number=1))
                # Analyze time after each run to decide if the result is stable
                results_are_significant = self.has_explicit_iteration_count or \
                    self._report_iteration_result(iters, run_time)

                if results_are_significant:
                    break

                # Re-estimate the hopefully-sufficient
                # iteration count, and run the benchmark again...
                iters = self._predict_num_iter_needed(iters)

            reported_run_time = (1e6 * run_time / iters)
            self._print_perf_result(full_test_id, input_shapes, args, reported_run_time)
