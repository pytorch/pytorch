import ast
import copy
import csv
import functools
import json
import os
import platform
import timeit
from collections import namedtuple
from dataclasses import asdict, dataclass
from typing import Any, Optional

import benchmark_utils

import numpy as np

import torch

# needs to be imported after torch
import torch.utils.cpp_extension as cpp_extension  # noqa: F401
from torch.utils.benchmark import Timer


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

SKIP_OP_LISTS = ["weight_norm_sparsifier_step"]


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

        op = bench_op()
        assert op is not None, "Can't create test"
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
        self.use_compile = args.use_compile
        if self.use_jit and self.use_compile:
            raise ValueError(
                "use_jit and use_compile are mutually exclusive, please specify one."
            )
        self.num_runs = args.num_runs
        self.print_per_iter = False
        self.output_csv = args.output_csv
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

    def _print_perf_result(self, results, test_case):
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
                            "value": str(results["reported_run_time_us"[run]]),
                        }
                    )
                )
        else:
            print(
                f"# Mode: {'JIT' if self.use_jit else 'Compile' if self.use_compile else 'Eager'}"
            )
            print(
                f"# Name: {test_case.test_config.test_name}\n# Input: {test_case.test_config.input_config}"
            )

            mode = "Backward" if test_case.test_config.run_backward else "Forward"
            if self.num_runs > 1:
                for run in range(self.num_runs):
                    print(
                        f"Run: {run}, {mode} Execution Time (us) : {results['reported_run_time_us'][run]:.3f}"
                    )
                print()
            else:
                print(
                    f"{mode} Execution Time (us) : {results['reported_run_time_us'][0]:.3f}"
                )
                print(f"Peak Memory (KB) : {results['peak_memory']}\n")

    def _perf_result_to_dict(self, results, test_case):
        """This function is the parallel of _print_perf_result, which instead of
        writing information to terminal, returns a dictionary.
        """
        if self.args.report_aibench:
            return {}

        out = {
            "test_name": test_case.test_config.test_name,
            "input_config": test_case.test_config.input_config,
            "runtime": (
                "JIT" if self.use_jit else "Compile" if self.use_compile else "Eager"
            ),
            "run": "Backward" if test_case.test_config.run_backward else "Forward",
            "latency": round(results["reported_run_time_us"][0], 3),
            "latency unit": "us",
            "peak memory": results["peak_memory"],
            "memory unit": "KB",
        }

        # parsing test_case.test_config.input_config, adding it as entries to the 'out' dictionary
        # input: 'M: 1, N: 1, K: 1, device: cpu'
        # output: {'M':'1', 'N':'1', 'K':'1', 'device': 'cpu'}
        # splitting the string on unnested commas
        def split(s):
            open_to_close = {"{": "}", "(": ")", "[": "]"}
            break_idxs = [-1]
            curr_brackets = []
            for i, c in enumerate(s):
                if c in open_to_close:
                    curr_brackets.append(c)
                elif c in open_to_close.values():
                    assert curr_brackets and open_to_close[curr_brackets[-1]] == c, (
                        "ERROR: not able to parse the string!"
                    )
                    curr_brackets.pop()
                elif c == "," and (not curr_brackets):
                    break_idxs.append(i)
            break_idxs.append(len(s))
            out = []
            for i in range(len(break_idxs) - 1):
                start, end = break_idxs[i], break_idxs[i + 1]
                out.append(s[start + 1 : end])
            return out

        key_vals = split(
            test_case.test_config.input_config
        )  # 'M: [(32, 16), (64, 32)], ZPB: 2' -> ['M: [(32, 16), (64, 32)]', 'ZPB: 2']
        key_vals = [
            (key.strip(), value.strip())
            for key, value in map(lambda str: str.split(":"), key_vals)  # noqa: C417
        ]  # ['M: (32, 16)', 'ZPB: 2'] -> [('M', '(32, 16)'), ('ZPB', '2')]
        out.update(key_vals)

        return out

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
        if self.use_compile:
            func = test_case.run_compile_forward

        if not cuda_sync:
            forward_time = timeit.timeit(
                functools.partial(func, iters, print_per_iter, cuda_sync), number=1
            )
            return forward_time
        # Stable timing with Timer
        timer = Timer(
            stmt="func(iters, print_per_iter, cuda_sync)",
            globals={
                "func": func,
                "iters": iters,
                "print_per_iter": print_per_iter,
                "cuda_sync": cuda_sync,
            },
        )
        result = timer.adaptive_autorange(min_run_time=0.0001)
        return result.median * iters

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

    def _measure_metrics(self, launch_test, test_case, iters, print_per_iter):
        """
        This function execute the operator for <iters> iterations then look at the time.
        If it's not significant, the number of iterations will be increased before rerun.
        The execution stops when the time becomes significant.
        """
        curr_test_total_time = 0
        time_trace = []
        peak_memory = 0
        input_values = test_case.op_bench.inputs.values()
        device, device_module = None, None
        if input_values and isinstance(next(iter(input_values)), torch.Tensor):
            # The device and device module information are crucial for memory metric calculation,
            # In case of ops where inputs are integers (not tensor), memory metrics need not be calculated.
            sample_input = next(iter(input_values))
            device = sample_input.device
            device_module = torch.get_device_module(device.type)
        # TODO: add support for cpu memory measurement
        while True:
            if hasattr(device_module, "reset_peak_memory_stats"):
                device_module.reset_peak_memory_stats(device)
            run_time_sec = launch_test(test_case, iters, print_per_iter)
            if hasattr(device_module, "synchronize"):
                device_module.synchronize(device)
            # Memory measurement process
            if hasattr(device_module, "max_memory_allocated"):
                peak_memory = device_module.max_memory_allocated(device)
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
                mode = (
                    "JIT"
                    if self.use_jit
                    else "Compile"
                    if self.use_compile
                    else "Eager"
                )
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
                        },
                    )
                )
            if results_are_significant:
                break

            # Re-estimate the hopefully-sufficient
            # iteration count, and run the benchmark again...
            iters = self._predict_num_iter_needed(iters)
        reported_run_time_us = np.percentile(np.array(time_trace), 50)
        return reported_run_time_us, peak_memory / 1024

    def _check_keep(self, test_flag, cmd_flag):
        return cmd_flag is None or test_flag == cmd_flag

    def _check_operator_first_char(self, test_flag, cmd_flag):
        return cmd_flag is None or test_flag[:1].lower() in cmd_flag

    def _check_keep_list(self, test_flag, cmd_flag_list):
        return cmd_flag_list is None or any(
            test_flag == cmd_flag for cmd_flag in cmd_flag_list
        )

    def _check_skip(self, test_module, cmd_flag):
        return cmd_flag is None or (test_module not in cmd_flag)

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
            and self._check_skip(test_case.op_bench.module_name(), SKIP_OP_LISTS)
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

    def _output_csv(self, filename, headers, row):
        if os.path.exists(filename):
            with open(filename) as fd:
                lines = list(csv.reader(fd)) or [[]]
                if headers and len(headers) > len(lines[0]):
                    # if prior results failed the header might not be filled in yet
                    lines[0] = headers
                else:
                    headers = lines[0]
        else:
            lines = [headers]
        lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
        with open(filename, "w") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            for line in lines:
                writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))

    def _output_json(
        self,
        perf_list,
        output_file,
        benchmark_name="PyTorch operator benchmark",
    ):
        """
        Write the result into JSON format, so that it can be uploaded to the benchmark database
        to be displayed on OSS dashboard. The JSON format is defined at
        https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
        """
        if not perf_list:
            return

        # Prepare headers and records for JSON output
        records = []
        for perf_item in perf_list:
            # Extract data from perf_item
            test_name = perf_item.get("test_name", "unknown")
            input_config = perf_item.get("input_config", "")
            run_type = perf_item.get("run")
            latency = perf_item.get("latency", 0)
            peak_memory = perf_item.get("peak memory", 0)
            device = perf_item.get("device", "unknown")
            dtype = perf_item.get("dtype", "torch.float").split(".")[1]
            runtime = perf_item.get("runtime", None)

            # Extract mode based on run_type
            mode = None
            if run_type == "Forward":
                mode = "inference"
            elif run_type == "Backward":
                mode = "training"

            # Extract use_compile from it
            if runtime == "Compile":
                use_compile = True
            elif runtime == "Eager":
                use_compile = False
            else:
                use_compile = None

            device_arch = (
                torch.cuda.get_device_name(0)
                if device == "cuda"
                else platform.processor()
                if device == "cpu"
                else "unknown"
            )

            # Extract operator name from test_name
            operator_name = test_name.split("_")[0]

            # Create the record
            @dataclass
            class BenchmarkInfo:
                name: str
                mode: Optional[str]
                dtype: str
                extra_info: dict[str, Any]

            @dataclass
            class ModelInfo:
                name: str
                type: str
                origins: list[str]
                extra_info: dict[str, Any]

            @dataclass
            class MetricInfo:
                name: str
                unit: str
                benchmark_values: list[float]
                target_value: Optional[float]

            @dataclass
            class BenchmarkRecord:
                benchmark: BenchmarkInfo
                model: ModelInfo
                metric: MetricInfo

            # Add record for latency
            record_latency = BenchmarkRecord(
                benchmark=BenchmarkInfo(
                    name=benchmark_name,
                    mode=mode,
                    dtype=dtype,
                    extra_info={
                        "input_config": input_config,
                        "device": device,
                        "arch": device_arch,
                        "use_compile": use_compile,
                        "operator_name": operator_name,
                    },
                ),
                model=ModelInfo(
                    name=test_name,
                    type="micro-benchmark",
                    origins=["pytorch"],
                    extra_info={"operator_name": operator_name},
                ),
                metric=MetricInfo(
                    name="latency",
                    unit="us",
                    benchmark_values=[latency],
                    target_value=None,
                ),
            )
            records.append(asdict(record_latency))

            # Add record for peak memory
            record_memory = copy.deepcopy(record_latency)
            record_memory.metric = MetricInfo(
                name="peak memory",
                unit="KB",
                benchmark_values=[peak_memory],
                target_value=None,
            )
            records.append(asdict(record_memory))

        # Write all records to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def run(self):
        self._print_header()
        output_csv_filename = self.args.output_csv
        headers = [
            "Benchmarking Framework",
            "Benchmarking Module Name",
            "Case Name",
            "tag",
            "run_backward",
            "Execution Time",
            "Peak Memory (KB)",
        ]

        if self.args.output_json or self.args.output_json_for_dashboard:
            perf_list = []

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
                results = [
                    self._measure_metrics(
                        launch_func, test_case, self.iters, self.print_per_iter
                    )
                    for _ in range(self.num_runs)
                ]
                result_dict = dict()
                result_dict["reported_run_time_us"] = [r[0] for r in results]
                result_dict["peak_memory"] = results[0][1]
                self._print_perf_result(results=result_dict, test_case=test_case)

                # output results to csv
                self._output_csv(
                    output_csv_filename,
                    headers,
                    [
                        test_case.framework,
                        test_case.op_bench.module_name(),
                        (
                            test_case.test_config.test_name + "_BACKWARD"
                            if test_case.test_config.run_backward is True
                            else test_case.test_config.test_name
                        ),
                        test_case.test_config.tag,
                        test_case.test_config.run_backward,
                        result_dict["reported_run_time_us"][0],
                        result_dict["peak_memory"],
                    ],
                )
                if self.args.output_json or self.args.output_json_for_dashboard:
                    perf_list.append(self._perf_result_to_dict(result_dict, test_case))

        if self.args.output_json_for_dashboard:
            self._output_json(
                perf_list, self.args.output_json_for_dashboard, self.args.benchmark_name
            )

        if self.args.output_json:
            with open(self.args.output_json, "w") as f:
                json.dump(perf_list, f)
