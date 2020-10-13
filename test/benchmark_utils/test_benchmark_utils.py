import json
import os
import re
import sys
import textwrap
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union
import unittest

import torch
import torch.utils.benchmark as benchmark_utils
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS, slowTest
import numpy as np


CALLGRIND_ARTIFACTS = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    "callgrind_artifacts.json"
)


def generate_callgrind_artifacts():
    print("Regenerating callgrind artifact.")

    stats_no_data = benchmark_utils.Timer(
        "y = torch.ones(())"
    ).collect_callgrind(number=1000)

    stats_with_data = benchmark_utils.Timer(
        "y = torch.ones((1,))"
    ).collect_callgrind(number=1000)

    user = os.getenv("USER")
    def to_entry(fn_counts):
        return [f"{c} {fn.replace(f'/{user}/', '/test_user/')}" for c, fn in fn_counts]

    artifacts = {
        "baseline_inclusive": to_entry(stats_no_data.baseline_inclusive_stats),
        "baseline_exclusive": to_entry(stats_no_data.baseline_exclusive_stats),
        "ones_no_data_inclusive": to_entry(stats_no_data.stmt_inclusive_stats),
        "ones_no_data_exclusive": to_entry(stats_no_data.stmt_exclusive_stats),
        "ones_with_data_inclusive": to_entry(stats_with_data.stmt_inclusive_stats),
        "ones_with_data_exclusive": to_entry(stats_with_data.stmt_exclusive_stats),
    }

    with open(CALLGRIND_ARTIFACTS, "wt") as f:
        json.dump(artifacts, f, indent=4)


def load_callgrind_artifacts() -> Tuple[benchmark_utils.CallgrindStats, benchmark_utils.CallgrindStats]:
    """Hermetic artifact to unit test Callgrind wrapper.

    In addition to collecting counts, this wrapper provides some facilities for
    manipulating and displaying the collected counts. The results of several
    measurements are stored in callgrind_artifacts.json.

    While FunctionCounts and CallgrindStats are pickleable, the artifacts for
    testing are stored in raw string form for easier inspection and to avoid
    baking any implementation details into the artifact itself.
    """
    with open(CALLGRIND_ARTIFACTS, "rt") as f:
        artifacts = json.load(f)

    pattern = re.compile(r"^\s*([0-9]+)\s(.+)$")

    def to_function_counts(
        count_strings: List[str],
        inclusive: bool
    ) -> benchmark_utils.FunctionCounts:
        data: List[benchmark_utils.FunctionCount] = []
        for cs in count_strings:
            # Storing entries as f"{c} {fn}" rather than [c, fn] adds some work
            # reviving the artifact, but it makes the json much easier to read.
            match = pattern.search(cs)
            assert match is not None
            c, fn = match.groups()
            data.append(benchmark_utils.FunctionCount(count=int(c), function=fn))

        return benchmark_utils.FunctionCounts(
            tuple(sorted(data, reverse=True)),
            inclusive=inclusive)

    baseline_inclusive = to_function_counts(artifacts["baseline_inclusive"], True)
    baseline_exclusive = to_function_counts(artifacts["baseline_exclusive"], False)

    stats_no_data = benchmark_utils.CallgrindStats(
        benchmark_utils.TaskSpec("y = torch.ones(())", "pass"),
        number_per_run=1000,
        built_with_debug_symbols=True,
        baseline_inclusive_stats=baseline_inclusive,
        baseline_exclusive_stats=baseline_exclusive,
        stmt_inclusive_stats=to_function_counts(artifacts["ones_no_data_inclusive"], True),
        stmt_exclusive_stats=to_function_counts(artifacts["ones_no_data_exclusive"], False),
    )

    stats_with_data = benchmark_utils.CallgrindStats(
        benchmark_utils.TaskSpec("y = torch.ones((1,))", "pass"),
        number_per_run=1000,
        built_with_debug_symbols=True,
        baseline_inclusive_stats=baseline_inclusive,
        baseline_exclusive_stats=baseline_exclusive,
        stmt_inclusive_stats=to_function_counts(artifacts["ones_with_data_inclusive"], True),
        stmt_exclusive_stats=to_function_counts(artifacts["ones_with_data_exclusive"], False),
    )

    return stats_no_data, stats_with_data


class _ExpectedBase:
    def __init__(self, test_name: str):
        self._results: Dict[str, Dict[str, Any]] = {}
        self._path = os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            f"{test_name}.json"
        )

    def __call__(self, label: str, values: Tuple[str, Any]):
        pass

    def finalize(self):
        pass


class StoreExpected(_ExpectedBase):
    def __init__(self, test_name: str):
        super().__init__(test_name)
        print(f"Regenerating artifacts for test_{test_name}")
        self._plaintext = [
            "=" * 80,
            f"== Plaintext cases for: test_{test_name} ".ljust(80, "="),
            "=" * 80,
        ]

    def __call__(self, label: str, values: Tuple[str, Any]):
        # Check for invalid tests.
        assert label not in self._results
        assert len(values) == len({k: v for k, v in values})

        self._results[label] = {k: v for k, v in values}

        self._plaintext.append(label)
        for sub_label, value in values:
            if "\\n" in repr(value):
                self._plaintext.append(f"  {sub_label}:\n{textwrap.indent(value, ' ' * 4)}\n")
            else:
                self._plaintext.append(f"  {sub_label}: {value}")
        self._plaintext.append("\n")

    def finalize(self):
        with open(self._path, "wt") as f:
            json.dump(self._results, f, indent=4)

        with open(re.sub(r"\.json$", ".txt", self._path), "wt") as f:
            f.write("\n".join(self._plaintext))


class TestExpected(_ExpectedBase):
    def __init__(self, test_name: str, assert_equal_fn: Callable[[Any, Any], NoReturn]):
        super().__init__(test_name)
        with open(self._path, "rt") as f:
            self._results = json.load(f)
        self._assert_equal_fn = assert_equal_fn
        self._count = 0

    def __call__(self, label: str, values: Tuple[str, Any]):
        for sub_label, value in values:
            self._assert_equal_fn(
                value,
                self._results[label][sub_label],
            )
            self._count += 1

    def finalize(self):
        expected_count = 0
        for i in self._results.values():
            expected_count += len(i)

        if self._count != expected_count:
            raise ValueError(
                f"Expected {expected_count} tests, but got {self._count}. "
                "Please regenerate tests."
            )


class TestBenchmarkUtils(TestCase):
    def test_timer(self):
        timer = benchmark_utils.Timer(
            stmt="torch.ones(())",
        )
        sample = timer.timeit(5).median
        self.assertIsInstance(sample, float)

        median = timer.blocked_autorange(min_run_time=0.01).median
        self.assertIsInstance(median, float)

        # We set a very high threshold to avoid flakiness in CI.
        # The internal algorithm is tested in `test_adaptive_timer`
        median = timer.adaptive_autorange(threshold=0.5).median

        # Test that multi-line statements work properly.
        median = benchmark_utils.Timer(
            stmt="""
                with torch.no_grad():
                    y = x + 1""",
            setup="""
                x = torch.ones((1,), requires_grad=True)
                for _ in range(5):
                    x = x + 1.0""",
        ).timeit(5).median
        self.assertIsInstance(sample, float)

    class _MockTimer:
        _seed = 0

        _timer_noise_level = 0.05
        _timer_cost = 100e-9  # 100 ns

        _function_noise_level = 0.05
        _function_costs = (
            ("pass", 8e-9),
            ("cheap_fn()", 4e-6),
            ("expensive_fn()", 20e-6),
            ("with torch.no_grad():\n    y = x + 1", 10e-6),
        )

        def __init__(self, stmt, setup, timer, globals):
            self._random_state = np.random.RandomState(seed=self._seed)
            self._mean_cost = {k: v for k, v in self._function_costs}[stmt]

        def sample(self, mean, noise_level):
            return max(self._random_state.normal(mean, mean * noise_level), 5e-9)

        def timeit(self, number):
            return sum([
                # First timer invocation
                self.sample(self._timer_cost, self._timer_noise_level),

                # Stmt body
                self.sample(self._mean_cost * number, self._function_noise_level),

                # Second timer invocation
                self.sample(self._timer_cost, self._timer_noise_level),
            ])

    def _test_or_regen_adaptive_timer(self, test_or_record: _ExpectedBase):
        class MockTimer(benchmark_utils.Timer):
            _timer_cls = self._MockTimer

        class _MockCudaTimer(self._MockTimer):
            # torch.cuda.synchronize is much more expensive than
            # just timeit.default_timer
            _timer_cost = 10e-6

            _function_costs = (
                self._MockTimer._function_costs[0],
                self._MockTimer._function_costs[1],

                # GPU should be faster once there is enough work.
                ("expensive_fn()", 5e-6),
            )

        class MockCudaTimer(benchmark_utils.Timer):
            _timer_cls = _MockCudaTimer

        def strip_address(m: benchmark_utils.Measurement) -> str:
            return re.sub(
                "object at 0x[0-9a-fA-F]+>",
                "object at 0xXXXXXXXXXXXX>",
                repr(m)
            )

        timers = (
            MockTimer("pass"),
            MockTimer("cheap_fn()"),
            MockTimer("expensive_fn()"),

            MockCudaTimer("pass"),
            MockCudaTimer("cheap_fn()"),
            MockCudaTimer("expensive_fn()"),
        )

        # Make sure __repr__ is reasonable for
        # multi-line / label / sub_label / description, but we don't need to
        # check numerics.
        multi_line_stmt = """
        with torch.no_grad():
            y = x + 1
        """
        repr_only_timers = (
            MockTimer(multi_line_stmt),
            MockTimer(multi_line_stmt, sub_label="scalar_add"),
            MockTimer(
                multi_line_stmt,
                label="x + 1",
                sub_label="scalar_add",
            ),
            MockTimer(
                multi_line_stmt,
                setup="setup_fn()",
                sub_label="scalar_add",
            ),
            MockTimer(
                multi_line_stmt,
                setup="""
                    x = torch.ones((1,), requires_grad=True)
                    for _ in range(5):
                        x = x + 1.0""",
                sub_label="scalar_add",
                description="Multi-threaded scalar math!",
                num_threads=16,
            ),
        )

        for timer in timers:
            m0 = timer.blocked_autorange(min_run_time=10)
            m1 = timer.adaptive_autorange()
            test_or_record(
                f"{timer.__class__.__name__}("
                f"stmt={repr(timer._task_spec.stmt)}, "
                f"setup={repr(timer._task_spec.setup)})",
                (
                    ("blocked_autorange", strip_address(m0)),
                    ("adaptive_autorange", strip_address(m1)),
                    ("mean", m0.mean),
                    ("median", m0.median),
                    ("repeats", len(m0.times)),
                    ("number_per_run", m0.number_per_run),
                )
            )

        for i, timer in enumerate(repr_only_timers):
            test_or_record(
                # We're testing the repr, so simply indexing will still produce
                # a readable `adaptive_timer.txt`
                f"{timer.__class__.__name__}: [{i}]",
                ((
                    "__repr__",
                    strip_address(timer.blocked_autorange(min_run_time=10))
                ),)
            )

        test_or_record.finalize()

    def test_adaptive_timer(self):
        self._test_or_regen_adaptive_timer(
            TestExpected("adaptive_timer", self.assertEqual)
        )

    @slowTest
    @unittest.skipIf(IS_WINDOWS, "Valgrind is not supported on Windows.")
    def test_collect_callgrind(self):
        @torch.jit.script
        def add_one(x):
            return x + 1

        timer = benchmark_utils.Timer(
            "y = add_one(x) + k",
            setup="x = torch.ones((1,))",
            globals={"add_one": add_one, "k": 5}
        )

        # Don't collect baseline to speed up unit test by ~30 seconds.
        stats = timer.collect_callgrind(number=1000, collect_baseline=False)
        counts = stats.counts(denoise=False)

        self.assertIsInstance(counts, int)
        self.assertGreater(counts, 0)

        from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import wrapper_singleton
        self.assertIsNone(
            wrapper_singleton()._bindings_module,
            "JIT'd bindings are only for back testing."
        )

    def _test_or_regen_callgrind_stats(self, test_or_record: _ExpectedBase):
        stats_no_data, stats_with_data = load_callgrind_artifacts()

        def strip_address(m: Union[benchmark_utils.FunctionCounts, benchmark_utils.CallgrindStats]) -> str:
            return re.sub(
                "object at 0x[0-9a-fA-F]+>",
                "object at 0xXXXXXXXXXXXX>",
                repr(m)
            )

        # Mock `torch.set_printoptions(linewidth=160)`
        wide_linewidth = benchmark_utils.FunctionCounts(
            stats_no_data.stats(inclusive=False)._data, False, 160)

        for l in repr(wide_linewidth).splitlines(keepends=False):
            self.assertLessEqual(len(l), 160)

        self.assertEqual(
            # `delta` is just a convenience method.
            stats_with_data.delta(stats_no_data)._data,
            (stats_with_data.stats() - stats_no_data.stats())._data
        )

        deltas = stats_with_data.as_standardized().delta(stats_no_data.as_standardized())
        def custom_transforms(fn: str):
            fn = re.sub(re.escape("/usr/include/c++/8/bits/"), "", fn)
            fn = re.sub(r"build/../", "", fn)
            fn = re.sub(".+" + re.escape("libsupc++"), "libsupc++", fn)
            return fn

        test_or_record(
            "stats_no_data",
            (
                ("CallgrindStats __repr__", strip_address(stats_no_data)),
                ("Total", stats_no_data.counts()),
                ("Total (denoised)", strip_address(stats_no_data.counts(denoise=True))),
                ("stats (exclusive)", strip_address(stats_no_data.stats())),
                ("stats (inclusive)", strip_address(stats_no_data.stats(inclusive=True))),
                ("stats (linewidth=160)", strip_address(wide_linewidth)),
                ("stats (as_standardized)", strip_address(stats_no_data.as_standardized().stats())),
                ("deltas", strip_address(deltas)),
                ("deltas (len)", len(deltas)),
                ("deltas (user transform)", strip_address(deltas.transform(custom_transforms))),
                ("deltas (user filter: ???)", strip_address(deltas.filter(lambda fn: fn.startswith("???")))),
                ("deltas (slice)", strip_address(deltas[:5]))
            ),
        )

        test_or_record.finalize()

    def test_manipulate_callgrind_stats(self):
        self._test_or_regen_callgrind_stats(
            TestExpected("manipulate_callgrind_stats", self.assertEqual)
        )

    def _test_or_regen_compare(self, test_or_record: _ExpectedBase):
        # Simulate several approaches.
        costs = (
            # overhead_optimized_fn()
            (1e-6, 1e-9),

            # compute_optimized_fn()
            (3e-6, 5e-10),

            # special_case_fn()  [square inputs only]
            (1e-6, 4e-10),
        )

        sizes = (
            (16, 16),
            (16, 128),
            (128, 128),
            (4096, 1024),
            (2048, 2048),
        )

        # overhead_optimized_fn()
        class _MockTimer_0(self._MockTimer):
            _function_costs = tuple(
                (f"fn({i}, {j})", costs[0][0] + costs[0][1] * i * j)
                for i, j in sizes
            )

        class MockTimer_0(benchmark_utils.Timer):
            _timer_cls = _MockTimer_0

        # compute_optimized_fn()
        class _MockTimer_1(self._MockTimer):
            _function_costs = tuple(
                (f"fn({i}, {j})", costs[1][0] + costs[1][1] * i * j)
                for i, j in sizes
            )

        class MockTimer_1(benchmark_utils.Timer):
            _timer_cls = _MockTimer_1

        # special_case_fn()
        class _MockTimer_2(self._MockTimer):
            _function_costs = tuple(
                (f"fn({i}, {j})", costs[2][0] + costs[2][1] * i * j)
                for i, j in sizes if i == j
            )

        class MockTimer_2(benchmark_utils.Timer):
            _timer_cls = _MockTimer_2

        results = []
        for i, j in sizes:
            results.append(
                MockTimer_0(
                    f"fn({i}, {j})",
                    label="fn",
                    description=f"({i}, {j})",
                    sub_label="overhead_optimized",
                ).blocked_autorange(min_run_time=10)
            )

            results.append(
                MockTimer_1(
                    f"fn({i}, {j})",
                    label="fn",
                    description=f"({i}, {j})",
                    sub_label="compute_optimized",
                ).blocked_autorange(min_run_time=10)
            )

            if i == j:
                results.append(
                    MockTimer_2(
                        f"fn({i}, {j})",
                        label="fn",
                        description=f"({i}, {j})",
                        sub_label="special_case (square)",
                    ).blocked_autorange(min_run_time=10)
                )

        compare = benchmark_utils.Compare(results)
        cases = [("Default", str(compare))]

        compare.trim_significant_figures()
        cases.append(("Trim significant figures", str(compare)))

        compare.colorize()
        cases.append(("Colorize (columnwise)", str(compare)))

        # Use `less -R test/benchmark_utils/compare.txt` to view colors.
        compare.colorize(rowwise=True)
        cases.append(("Colorize (rowwise)", str(compare)))

        test_or_record("Compare", tuple(cases))
        test_or_record.finalize()

    def test_compare(self):
        self._test_or_regen_compare(
            TestExpected("compare", self.assertEqual)
        )

    @unittest.skipIf(IS_WINDOWS and os.getenv("VC_YEAR") == "2019", "Random seed only accepts int32")
    def test_fuzzer(self):
        fuzzer = benchmark_utils.Fuzzer(
            parameters=[
                benchmark_utils.FuzzedParameter(
                    "n", minval=1, maxval=16, distribution="loguniform")],
            tensors=[benchmark_utils.FuzzedTensor("x", size=("n",))],
            seed=0,
        )

        expected_results = [
            (0.7821, 0.0536, 0.9888, 0.1949, 0.5242, 0.1987, 0.5094),
            (0.7166, 0.5961, 0.8303, 0.005),
        ]

        for i, (tensors, _, _) in enumerate(fuzzer.take(2)):
            x = tensors["x"]
            self.assertEqual(
                x, torch.Tensor(expected_results[i]), rtol=1e-3, atol=1e-3)

    def regenerate_all(self):
        generate_callgrind_artifacts()
        self._test_or_regen_adaptive_timer(StoreExpected("adaptive_timer"))
        self._test_or_regen_callgrind_stats(StoreExpected("manipulate_callgrind_stats"))
        self._test_or_regen_compare(StoreExpected("compare"))


if __name__ == '__main__':
    if "--regenerate" in sys.argv:
        print("Regenerating test artifacts:")
        TestBenchmarkUtils().regenerate_all()

    else:
        run_tests()
