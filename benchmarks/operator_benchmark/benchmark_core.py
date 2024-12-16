import ast
import copy
import functools
import json
import timeit
from collections import namedtuple

import benchmark_utils

import numpy as np

import torch

# needs to be imported after torch
import torch.utils.cpp_extension as cpp_extension  # noqa: F401


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


BENCHMARK_TESTER = []


def _register_test(*test_metainfo):
    """save the metainfo needed to create a test. Currently test_metainfo
    takes two different inputs:
    1) This input when adds single op to the benchmark
     _register_test(configs, pt_bench_op, create_pytorch_op_test_case,
                      run_backward=True)
    2) This input when adds a list of ops to the benchmark
    _register_test(configs, pt_bench_op, create_pytorch_op_test_case,
                      run_backward=False,
                      op_name_function=op)
    """
    BENCHMARK_TESTER.append(test_metainfo)


def _create_test(
    bench_op_obj, orig_test_attrs, tags, OperatorTestCase, run_backward, bwd_input
):
    """Create tests with the benchmark backend.
    Args:
        bench_op_obj: an object which instantiated from a subclass of
            TorchBenchmarkBase which includes tensor
            creation and operator execution.
        orig_test_attrs: a dictionary includes test configs.
        tags: a attribute in test config to filter inputs
        OperatorTestCase: a named tuple to save the metadata of an test
        run_backward: a bool parameter indicating backward path
    """
    test_attrs = copy.deepcopy(orig_test_attrs)
    test_attrs = {k: str(v) for k, v in test_attrs.items()}
    ascii_test_attrs = ast.literal_eval(json.dumps(test_attrs))
    input_config = str(ascii_test_attrs)[1:-1].replace("'", "")
    if bwd_input:
        # When auto_set is used, the test name needs to include input.
        test_attrs.update({"bwd": bwd_input})
    test_name = bench_op_obj.test_name(**test_attrs)
    test_config = TestConfig(test_name, input_config, tags, run_backward)
    return OperatorTestCase(bench_op_obj, test_config)


def _build_test(
    configs, bench_op, OperatorTestCase, run_backward, op_name_function=None
):
    """Generate PyTorch/Caffe2 tests of operators with different inputs.
    Args:
        configs: a dictionary that has the input shapes
        bench_op: a subclass of TorchBenchmarkBase which includes tensor
            creation and operator execution
        OperatorTestCase: a named tuple to save the metadata of an test
        run_backward: a bool parameter indicating backward path
        op_name_function: a dictionary includes operator name and function
    """
    for config in configs:
        test_attrs = {}
        tags = None
        keep_config = True
        for attr in config:
            # tags is only used in our benchmark backend to filter tests and
            # it will be removed from config which is then passed to the init function
            # an example of config and atrr is:
            # config: [{'M': 16}, {'N': 16}, {'K': 64}, {'tags': 'short'}]
            # attr: {'tags': 'short'}
            if "tags" in attr:
                tags = attr["tags"]
                continue

            # if 'cuda' is specified in input shape but the testing machines doesn't
            # support, we will skip this input
            if "cuda" in attr.values():
                if not torch.cuda.is_available():
                    keep_config = False
                    break

            test_attrs.update(attr)

        if not keep_config:
            continue

        if tags is None:
            raise ValueError("Missing tags in configs")
        input_config = str(test_attrs)[1:-1].replace("'", "")
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
            op_name = op_name_function["op_name"]
            init_dict.update({"op_func": op_name_function["op_func"]})
            op.set_module_name(op_name)

        op._set_backward_test(run_backward)
        op.init(**init_dict)
        op.extract_inputs_tuple()

        if not run_backward:
            for attr in vars(op).values():
                if isinstance(attr, torch.nn.Module):
                    for param in attr.parameters():
                        param.requires_grad = False

        input_name = None

        # _num_inputs_require_grads is used to track the number of tensors
        # which use auto_set().
        if op._num_inputs_require_grads > 0:
            input_name = "all"
        yield _create_test(
            op, test_attrs, tags, OperatorTestCase, run_backward, input_name
        )

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
            yield _create_test(
                new_op, test_attrs, tags, OperatorTestCase, run_backward, input_name
            )


class BenchmarkRunner:
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
        self.iters = 100
        self.has_explicit_iteration_count = False
        self.multiplier = 2
        self.predefined_minimum_secs = 1
        self.max_iters = 1e6
        self.use_jit = args.use_jit
        self.num_runs = args.num_runs
        self.print_per_iter = False
        self.operator_range = benchmark_utils.get_operator_range(args.operator_range)
        # 100 is the default warmup iterations
        if self.args.warmup_iterations == -1:
            self.args.warmup_iterations = 100
        if self.args.iterations and self.args.iterations != -1:
            self.has_explicit_iteration_count = True
            self.iters = self.args.iterations
        # when a specific test is selected by a user, we don't need
        # to match the tag anymore
        if self.args.test_name is not None:
            self.args.tag_filter = None

    def _print_header(self):
        DASH_LINE = "-" * 40
        print(
            f"# {DASH_LINE}\n"
            "# PyTorch/Caffe2 Operator Micro-benchmarks\n"
            f"# {DASH_LINE}\n"
            f"# Tag : {self.args.tag_filter}\n"
        )
        if self.args.list_tests:
            print("# List of tests:")
        elif self.args.list_ops:
            print("# List of Operators to run:")
            self.printed_ops_list = set()
            if self.args.operators:
                print(f"# {self.args.operators}")

    def _print_perf_result(self, reported_run_time_us, test_case):
        if self.args.report_aibench:
            # Output for AIBench
            # Print out per iteration execution time instead of avg time
            return
            test_name = "_".join([test_case.framework, test_case.test_config.test_name])
            for run in range(self.num_runs):
                print(
                    f"{test_case.framework}Observer "
                    + json.dumps(
                        {
                            "type": test_name,
                            "metric": "latency",
                            "unit": "us",
                            "value": str(reported_run_time_us[run]),
                        }
                    )
                )
        else:
            print(f"# Mode: {'JIT' if self.use_jit else 'Eager'}")
            print(
                f"# Name: {test_case.test_config.test_name}\n# Input: {test_case.test_config.input_config}"
            )

            mode = "Backward" if test_case.test_config.run_backward else "Forward"
            if self.num_runs > 1:
                for run in range(self.num_runs):
                    print(
                        f"Run: {run}, {mode} Execution Time (us) : {reported_run_time_us[run]:.3f}"
                    )
                print()
            else:
                print(f"{mode} Execution Time (us) : {reported_run_time_us[0]:.3f}\n")

    def _predict_num_iter_needed(self, i):
        return i * self.multiplier

    def _iteration_result_is_significant(
        self, iters, run_time_sec, curr_test_total_time, has_explicit_iteration_count
    ):
        """This function decides whether the measured time can be reported based on the
        following conditions: 1) the number of iterations is larger than the max_iters.
        2) the execution time is larger than the predefined minimum_time
        3) the execution time is larger than user defined minimum_time
        """
        return (
            iters > self.max_iters
            or run_time_sec > self.predefined_minimum_secs
            or has_explicit_iteration_count
        ) and curr_test_total_time > self.args.min_time_per_test

    def _launch_forward(self, test_case, iters, print_per_iter):
        """Use Python's timeit module to measure execution time (unit: second)."""
        cuda_sync = "cuda" in test_case.test_config.test_name
        func = test_case.run_forward
        if self.use_jit:
            func = test_case.run_jit_forward
        forward_time = timeit.timeit(
            functools.partial(func, iters, print_per_iter, cuda_sync), number=1
        )
        return forward_time

    def _launch_backward(self, test_case, iters, print_per_iter=False):
        """This function runs forward path of an op to get an output. Then the backward path is executed
        and the execution time is reported
        """
        test_case.run_forward(num_runs=1, print_per_iter=False, cuda_sync=False)
        test_case._output_mean()
        backward_time = timeit.timeit(
            functools.partial(test_case.run_backward, iters, print_per_iter), number=1
        )
        return backward_time

    def _measure_time(self, launch_test, test_case, iters, print_per_iter):
        """
        This function execute the operator for <iters> iterations then look at the time.
        If it's not significant, the number of iterations will be increased before rerun.
        The execution stops when the time becomes significant.
        """
        curr_test_total_time = 0
        time_trace = []
        while True:
            run_time_sec = launch_test(test_case, iters, print_per_iter)
            curr_test_total_time += run_time_sec
            # Analyze time after each run to decide if the result is stable
            results_are_significant = self._iteration_result_is_significant(
                iters,
                run_time_sec,
                curr_test_total_time,
                self.has_explicit_iteration_count,
            )

            report_run_time = 1e6 * run_time_sec / iters
            time_trace.append(report_run_time)
            # Print out the time spent in each epoch in ms
            if self.args.report_aibench:
                mode = "JIT" if self.use_jit else "Eager"
                test_name = "_".join(
                    [test_case.framework, test_case.test_config.test_name, mode]
                )
                print(
                    "PyTorchObserver "
                    + json.dumps(
                        {
                            "type": test_name,
                            "metric": "latency",
                            "unit": "ms",
                            "value": str(report_run_time / 1e3),
                        }
                    )
                )
            if results_are_significant:
                break

            # Re-estimate the hopefully-sufficient
            # iteration count, and run the benchmark again...
            iters = self._predict_num_iter_needed(iters)
        reported_run_time_us = np.percentile(np.array(time_trace), 50)
        return reported_run_time_us

    def _check_keep(self, test_flag, cmd_flag):
        return cmd_flag is None or test_flag == cmd_flag

    def _check_operator_first_char(self, test_flag, cmd_flag):
        return cmd_flag is None or test_flag[:1].lower() in cmd_flag

    def _check_keep_list(self, test_flag, cmd_flag_list):
        return cmd_flag_list is None or any(
            test_flag == cmd_flag for cmd_flag in cmd_flag_list
        )

    def _keep_test(self, test_case):
        # TODO: consider regex matching for test filtering.
        # Currently, this is a sub-string matching.
        op_test_config = test_case.test_config

        operators = (
            benchmark_utils.process_arg_list(self.args.operators)
            if self.args.operators
            else None
        )

        # Filter framework, operator, test_name, tag, forward_only
        return (
            self._check_keep(op_test_config.test_name, self.args.test_name)
            and self._check_keep_list(test_case.op_bench.module_name(), operators)
            and self._check_operator_first_char(
                test_case.op_bench.module_name(), self.operator_range
            )
            and (
                self.args.tag_filter == "all"
                or self._check_keep(op_test_config.tag, self.args.tag_filter)
            )
            and (
                not self.args.forward_only
                or op_test_config.run_backward != self.args.forward_only
            )
            and (
                self.args.device == "None"
                or "device" not in test_case.test_config.input_config
                or self.args.device in op_test_config.test_name
            )
        )

    def _print_test_case_info(self, test_case):
        # Print out the test name and skip the real execution
        if self.args.list_tests:
            print(f"# {test_case.test_config.test_name}")
            return True
        elif self.args.list_ops:
            if self.args.operators is None:
                op_name = test_case.op_bench.module_name()

                if op_name not in self.printed_ops_list:
                    print(f"# {op_name}")
                    self.printed_ops_list.add(op_name)
            return True

        return False

    def run(self):
        self._print_header()

        for test_metainfo in BENCHMARK_TESTER:
            for test in _build_test(*test_metainfo):
                full_test_id, test_case = test
                op_test_config = test_case.test_config

                if self._print_test_case_info(test_case):
                    continue

                if not self._keep_test(test_case):
                    continue

                # To reduce variance, fix a numpy randseed to the test case,
                # so that the randomly generated input tensors remain the
                # same for each test case.
                # The random seed is limited to 32-bit because of numpy
                # requirement.
                np.random.seed(seed=hash(full_test_id) & ((1 << 32) - 1))

                print(
                    f"# Benchmarking {test_case.framework}: {test_case.op_bench.module_name()}"
                )

                if op_test_config.run_backward:
                    launch_func = self._launch_backward
                else:
                    launch_func = self._launch_forward

                # Warmup
                launch_func(
                    test_case, self.args.warmup_iterations, print_per_iter=False
                )
                # Actual Execution
                reported_time = [
                    self._measure_time(
                        launch_func, test_case, self.iters, self.print_per_iter
                    )
                    for _ in range(self.num_runs)
                ]

                self._print_perf_result(reported_time, test_case)
