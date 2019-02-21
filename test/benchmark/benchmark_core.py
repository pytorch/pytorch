from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import timeit
import json

"""Performance microbenchmarks.

This module contains core functionalities for performance microbenchmark tests.
"""


BENCHMARK_TESTERS = {}


def benchmark_tester(func):
    """Decorator to register a benchmark tester function.
    A benchmark tester function is reponsible for taking in a test case object,
    running necessary prepraration steps (such as creating input data), and
    defining the microbenchmark function (what is to be benchmarked).

    """
    BENCHMARK_TESTERS[func.__name__] = func
    return func

BENCHMARK_TESTS = {}


def benchmark_test(func):
    """Decorator to register a benchmark test group.
    A benchmark test group is a function that returns a list of benchmark test
    case objects to be run.
    """
    BENCHMARK_TESTS[func.__name__] = func
    return func


# List of run modes we support.
# Each benchmark test case is associated with a run mode.
# If the value of the test case's run mode is less than the value of the
# benchmark binary's run mode, the test case will be executed, e.g. a short-mode
# test case will be executed when the binary is on either long and short
# modes; while a long-mode test case will only be executed when the binary is
# on long-mode.
RUN_MODES = {'short': 0, 'long': 1}


class BenchmarkRunner(object):
    """BenchmarkRunner is responsible for benchmarking all the registered
    benchmark test groups.

    Attributes:
        run_mode (str): Must of one of 'short', 'long'. For long mode, the
    benchmark runner takes a longer time to run since it repeats each benchmark
    test case more times to reduce measured variance, and it also executes
    longer running test cases that is marked as long mode.
        test_case_filter (str): Only run benchmark test cases that contains
    this filter string in the test case's id.
    """
    def __init__(self, run_mode='short', test_case_filter=None):
        self.run_mode = run_mode
        self.test_case_filter = test_case_filter
        # Depend on the run mode, set the execution contrains based of number of
        # runs per measure, and number of measures.
        # TODO: consider time-bound constraints as well.
        if run_mode == 'short':
            self.num_repeats = 50
            self.num_runs = 100
        else:
            self.num_repeats = 5
            self.num_runs = 100
        print("Initialize benchmark runner with run_mode = %s, num_repeats = %d, num_runs = %d" %
              (self.run_mode, self.num_repeats, self.num_runs))

    def run(self):
        for test in BENCHMARK_TESTS.items():
            test_group_name = test[0]
            print("Running benchmark test group %s" % test_group_name)
            test_cases = test[1]()
            for test_case in test_cases:
                full_test_id = test_group_name + "." + type(test_case).__name__ + "." + test_case.test_name
                # TODO: consider regex matching for test filtering.
                # Currently, this is a sub-string matching.
                if self.test_case_filter and (self.test_case_filter not in full_test_id):
                    print("Skipping benchmark test case %s" % full_test_id)
                    continue
                if RUN_MODES[self.run_mode] < RUN_MODES[test_case.run_mode]:
                    print("Skipping benchmark test %s with run mode = %s since the current run mode = %s" %
                          (full_test_id, test_case.run_mode, self.run_mode))
                    continue
                for tester in BENCHMARK_TESTERS.items():
                    # To reduce variance, fix a numpy randseed to the test case,
                    # so that the randomly generated input tensors remain the
                    # same for each test case.
                    # The random seed is limited to 32-bit because of numpy
                    # requirement.
                    np.random.seed(seed=hash(full_test_id) & ((1 << 32) - 1))

                    benchmark_func = tester[1](test_case)
                    if callable(benchmark_func):
                        # Use Python's timeit module to measure execution time.
                        # Each experiment consists of repeated execution of
                        # the benchmark_func a number of times (self.num_runs)
                        # because otherwise the duration is too short to get
                        # an accurate measure.
                        # The experiment is also repeated a number of times
                        # (num_repeats) and we then take the minimum execution
                        # time as the final measurement result (this is also
                        # recommended by timeit's doc).
                        run_time = min(timeit.repeat(benchmark_func, repeat=self.num_repeats, number=self.num_runs))
                        reported_run_time_ms = (1000 * run_time / self.num_runs)
                        print('\tExecution time in milliseconds: %.5f' % reported_run_time_ms)
                        print("Caffe2Observer " + json.dumps(
                            {
                                "type": "NET",
                                "metric": full_test_id,
                                "unit": "ms",
                                "value": str(reported_run_time_ms),
                            }
                        ))
