# Owner(s): ["module: unknown"]

import collections
import json
import os
import re
import textwrap
import timeit
from typing import Any, List, Tuple
import unittest

import torch
import torch.utils.benchmark as benchmark_utils
from torch.testing._internal.common_utils import TestCase, run_tests, IS_SANDCASTLE, IS_WINDOWS, slowTest, TEST_WITH_ASAN
import expecttest
import numpy as np


CALLGRIND_ARTIFACTS: str = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    "callgrind_artifacts.json"
)


def generate_callgrind_artifacts() -> None:
    """Regenerate `callgrind_artifacts.json`

    Unlike the expect tests, regenerating callgrind counts will produce a
    large diff since build directories and conda/pip directories are included
    in the instruction string. It is also not 100% deterministic (due to jitter
    from Python) and takes over a minute to run. As a result, running this
    function is manual.
    """
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
        stmt_callgrind_out=None,
    )

    stats_with_data = benchmark_utils.CallgrindStats(
        benchmark_utils.TaskSpec("y = torch.ones((1,))", "pass"),
        number_per_run=1000,
        built_with_debug_symbols=True,
        baseline_inclusive_stats=baseline_inclusive,
        baseline_exclusive_stats=baseline_exclusive,
        stmt_inclusive_stats=to_function_counts(artifacts["ones_with_data_inclusive"], True),
        stmt_exclusive_stats=to_function_counts(artifacts["ones_with_data_exclusive"], False),
        stmt_callgrind_out=None,
    )

    return stats_no_data, stats_with_data


class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


class TestBenchmarkUtils(TestCase):
    def regularizeAndAssertExpectedInline(
        self, x: Any,
        expect: str,
        indent: int = 12
    ) -> None:
        x_str: str = re.sub(
            "object at 0x[0-9a-fA-F]+>",
            "object at 0xXXXXXXXXXXXX>",
            x if isinstance(x, str) else repr(x)
        )
        if "\n" in x_str:
            # Indent makes the reference align at the call site.
            x_str = textwrap.indent(x_str, " " * indent)

        self.assertExpectedInline(x_str, expect, skip=1)

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

    @slowTest
    @unittest.skipIf(IS_SANDCASTLE, "C++ timing is OSS only.")
    @unittest.skipIf(True, "Failing on clang, see 74398")
    def test_timer_tiny_fast_snippet(self):
        timer = benchmark_utils.Timer(
            'auto x = 1;(void)x;',
            timer=timeit.default_timer,
            language=benchmark_utils.Language.CPP,
        )
        median = timer.blocked_autorange().median
        self.assertIsInstance(median, float)

    @slowTest
    @unittest.skipIf(IS_SANDCASTLE, "C++ timing is OSS only.")
    @unittest.skipIf(True, "Failing on clang, see 74398")
    def test_cpp_timer(self):
        timer = benchmark_utils.Timer(
            """
                #ifndef TIMER_GLOBAL_CHECK
                static_assert(false);
                #endif

                torch::Tensor y = x + 1;
            """,
            setup="torch::Tensor x = torch::empty({1});",
            global_setup="#define TIMER_GLOBAL_CHECK",
            timer=timeit.default_timer,
            language=benchmark_utils.Language.CPP,
        )
        t = timer.timeit(10)
        self.assertIsInstance(t.median, float)

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
            self._mean_cost = dict(self._function_costs)[stmt]

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

    def test_adaptive_timer(self):
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

        m = MockTimer("pass").blocked_autorange(min_run_time=10)
        self.regularizeAndAssertExpectedInline(
            m,
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            pass
              Median: 7.98 ns
              IQR:    0.52 ns (7.74 to 8.26)
              125 measurements, 10000000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer("pass").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            pass
              Median: 7.86 ns
              IQR:    0.71 ns (7.63 to 8.34)
              6 measurements, 1000000 runs per measurement, 1 thread"""
        )

        # Check against strings so we can reuse expect infra.
        self.regularizeAndAssertExpectedInline(m.mean, """8.0013658357956e-09""")
        self.regularizeAndAssertExpectedInline(m.median, """7.983151323215967e-09""")
        self.regularizeAndAssertExpectedInline(len(m.times), """125""")
        self.regularizeAndAssertExpectedInline(m.number_per_run, """10000000""")

        self.regularizeAndAssertExpectedInline(
            MockTimer("cheap_fn()").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            cheap_fn()
              Median: 3.98 us
              IQR:    0.27 us (3.85 to 4.12)
              252 measurements, 10000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer("cheap_fn()").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            cheap_fn()
              Median: 4.16 us
              IQR:    0.22 us (4.04 to 4.26)
              4 measurements, 1000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer("expensive_fn()").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            expensive_fn()
              Median: 19.97 us
              IQR:    1.35 us (19.31 to 20.65)
              501 measurements, 1000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer("expensive_fn()").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            expensive_fn()
              Median: 20.79 us
              IQR:    1.09 us (20.20 to 21.29)
              4 measurements, 1000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("pass").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            pass
              Median: 7.92 ns
              IQR:    0.43 ns (7.75 to 8.17)
              13 measurements, 100000000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("pass").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            pass
              Median: 7.75 ns
              IQR:    0.57 ns (7.56 to 8.13)
              4 measurements, 10000000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("cheap_fn()").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            cheap_fn()
              Median: 4.04 us
              IQR:    0.30 us (3.90 to 4.19)
              25 measurements, 100000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("cheap_fn()").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            cheap_fn()
              Median: 4.09 us
              IQR:    0.38 us (3.90 to 4.28)
              4 measurements, 100000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("expensive_fn()").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            expensive_fn()
              Median: 4.98 us
              IQR:    0.31 us (4.83 to 5.13)
              20 measurements, 100000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("expensive_fn()").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            expensive_fn()
              Median: 5.01 us
              IQR:    0.28 us (4.87 to 5.15)
              4 measurements, 10000 runs per measurement, 1 thread"""
        )

        # Make sure __repr__ is reasonable for
        # multi-line / label / sub_label / description, but we don't need to
        # check numerics.
        multi_line_stmt = """
        with torch.no_grad():
            y = x + 1
        """

        self.regularizeAndAssertExpectedInline(
            MockTimer(multi_line_stmt).blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            stmt:
              with torch.no_grad():
                  y = x + 1

              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer(multi_line_stmt, sub_label="scalar_add").blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            stmt: (scalar_add)
              with torch.no_grad():
                  y = x + 1

              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer(
                multi_line_stmt,
                label="x + 1 (no grad)",
                sub_label="scalar_add",
            ).blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            x + 1 (no grad): scalar_add
              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer(
                multi_line_stmt,
                setup="setup_fn()",
                sub_label="scalar_add",
            ).blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            stmt: (scalar_add)
              with torch.no_grad():
                  y = x + 1

            setup: setup_fn()
              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 1 thread"""
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer(
                multi_line_stmt,
                setup="""
                    x = torch.ones((1,), requires_grad=True)
                    for _ in range(5):
                        x = x + 1.0""",
                sub_label="scalar_add",
                description="Multi-threaded scalar math!",
                num_threads=16,
            ).blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            stmt: (scalar_add)
              with torch.no_grad():
                  y = x + 1

            Multi-threaded scalar math!
            setup:
              x = torch.ones((1,), requires_grad=True)
              for _ in range(5):
                  x = x + 1.0

              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 16 threads"""
        )

    @slowTest
    @unittest.skipIf(IS_WINDOWS, "Valgrind is not supported on Windows.")
    @unittest.skipIf(IS_SANDCASTLE, "Valgrind is OSS only.")
    @unittest.skipIf(TEST_WITH_ASAN, "fails on asan")
    def test_collect_callgrind(self):
        with self.assertRaisesRegex(
            ValueError,
            r"`collect_callgrind` requires that globals be wrapped "
            r"in `CopyIfCallgrind` so that serialization is explicit."
        ):
            benchmark_utils.Timer(
                "pass",
                globals={"x": 1}
            ).collect_callgrind(collect_baseline=False)

        with self.assertRaisesRegex(
            # Subprocess raises AttributeError (from pickle),
            # _ValgrindWrapper re-raises as generic OSError.
            OSError, "AttributeError: Can't get attribute 'MyModule'"
        ):
            benchmark_utils.Timer(
                "model(1)",
                globals={"model": benchmark_utils.CopyIfCallgrind(MyModule())}
            ).collect_callgrind(collect_baseline=False)


        @torch.jit.script
        def add_one(x):
            return x + 1

        timer = benchmark_utils.Timer(
            "y = add_one(x) + k",
            setup="x = torch.ones((1,))",
            globals={
                "add_one": benchmark_utils.CopyIfCallgrind(add_one),
                "k": benchmark_utils.CopyIfCallgrind(5),
                "model": benchmark_utils.CopyIfCallgrind(
                    MyModule(),
                    setup=f"""\
                    import sys
                    sys.path.append({repr(os.path.split(os.path.abspath(__file__))[0])})
                    from test_benchmark_utils import MyModule
                    """
                )
            }
        )

        stats = timer.collect_callgrind(number=1000)
        counts = stats.counts(denoise=False)

        self.assertIsInstance(counts, int)
        self.assertGreater(counts, 0)

        # There is some jitter with the allocator, so we use a simpler task to
        # test reproducibility.
        timer = benchmark_utils.Timer(
            "x += 1",
            setup="x = torch.ones((1,))",
        )

        stats = timer.collect_callgrind(number=1000, repeats=20)
        assert isinstance(stats, tuple)

        # Check that the repeats are at least somewhat repeatable. (within 10 instructions per iter)
        counts = collections.Counter([s.counts(denoise=True) // 10_000 * 10_000 for s in stats])
        self.assertGreater(max(counts.values()), 1, f"Every instruction count total was unique: {counts}")

        from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import wrapper_singleton
        self.assertIsNone(
            wrapper_singleton()._bindings_module,
            "JIT'd bindings are only for back testing."
        )

    @slowTest
    @unittest.skipIf(IS_WINDOWS, "Valgrind is not supported on Windows.")
    @unittest.skipIf(IS_SANDCASTLE, "Valgrind is OSS only.")
    @unittest.skipIf(True, "Failing on clang, see 74398")
    def test_collect_cpp_callgrind(self):
        timer = benchmark_utils.Timer(
            "x += 1;",
            setup="torch::Tensor x = torch::ones({1});",
            timer=timeit.default_timer,
            language="c++",
        )
        stats = [
            timer.collect_callgrind()
            for _ in range(3)
        ]
        counts = [s.counts() for s in stats]

        self.assertGreater(
            min(counts), 0, "No stats were collected")
        self.assertEqual(
            min(counts), max(counts), "C++ Callgrind should be deterministic")

        for s in stats:
            self.assertEqual(
                s.counts(denoise=True), s.counts(denoise=False),
                "De-noising should not apply to C++.")

        stats = timer.collect_callgrind(number=1000, repeats=20)
        assert isinstance(stats, tuple)

        # NB: Unlike the example above, there is no expectation that all
        #     repeats will be identical.
        counts = collections.Counter([s.counts(denoise=True) // 10_000 * 10_000 for s in stats])
        self.assertGreater(max(counts.values()), 1, repr(counts))

    def test_manipulate_callgrind_stats(self):
        stats_no_data, stats_with_data = load_callgrind_artifacts()

        # Mock `torch.set_printoptions(linewidth=160)`
        wide_linewidth = benchmark_utils.FunctionCounts(
            stats_no_data.stats(inclusive=False)._data, False, _linewidth=160)

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

        self.regularizeAndAssertExpectedInline(
            stats_no_data,
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0xXXXXXXXXXXXX>
            y = torch.ones(())
                                       All          Noisy symbols removed
                Instructions:      8869966                    8728096
                Baseline:             6682                       5766
            1000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            stats_no_data.counts(),
            """8869966""",
        )

        self.regularizeAndAssertExpectedInline(
            stats_no_data.counts(denoise=True),
            """8728096""",
        )

        self.regularizeAndAssertExpectedInline(
            stats_no_data.stats(),
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0xXXXXXXXXXXXX>
              408000  ???:__tls_get_addr [/usr/lib64/ld-2.28.so]
              388193  ???:_int_free [/usr/lib64/libc-2.28.so]
              274000  build/../torch/csrc/utils/python ... rch/torch/lib/libtorch_python.so]
              264000  build/../aten/src/ATen/record_fu ... ytorch/torch/lib/libtorch_cpu.so]
              192000  build/../c10/core/Device.h:c10:: ... epos/pytorch/torch/lib/libc10.so]
              169855  ???:_int_malloc [/usr/lib64/libc-2.28.so]
              154000  build/../c10/core/TensorOptions. ... ytorch/torch/lib/libtorch_cpu.so]
              148561  /tmp/build/80754af9/python_15996 ... da3/envs/throwaway/bin/python3.6]
              135000  ???:malloc [/usr/lib64/libc-2.28.so]
                 ...
                2000  /usr/include/c++/8/ext/new_allocator.h:torch::PythonArgs::intlist(int)
                2000  /usr/include/c++/8/bits/stl_vect ... *, _object*, _object*, _object**)
                2000  /usr/include/c++/8/bits/stl_vect ... rningHandler::~PyWarningHandler()
                2000  /usr/include/c++/8/bits/stl_vect ... ject*, _object*, _object**, bool)
                2000  /usr/include/c++/8/bits/stl_algobase.h:torch::PythonArgs::intlist(int)
                2000  /usr/include/c++/8/bits/shared_p ... ad_accumulator(at::Tensor const&)
                2000  /usr/include/c++/8/bits/move.h:c ... te<c10::AutogradMetaInterface> >)
                2000  /usr/include/c++/8/bits/atomic_b ... DispatchKey&&, caffe2::TypeMeta&)
                2000  /usr/include/c++/8/array:at::Ten ... , at::Tensor&, c10::Scalar) const

            Total: 8869966""",
        )

        self.regularizeAndAssertExpectedInline(
            stats_no_data.stats(inclusive=True),
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0xXXXXXXXXXXXX>
              8959166  ???:0x0000000000001050 [/usr/lib64/ld-2.28.so]
              8959166  ???:(below main) [/usr/lib64/libc-2.28.so]
              8959166  /tmp/build/80754af9/python_15996 ... a3/envs/throwaway/bin/python3.6]
              8959166  /tmp/build/80754af9/python_15996 ... a3/envs/throwaway/bin/python3.6]
              8959166  /tmp/build/80754af9/python_15996 ... a3/envs/throwaway/bin/python3.6]
              8959166  /tmp/build/80754af9/python_15996 ... a3/envs/throwaway/bin/python3.6]
              8959166  /tmp/build/80754af9/python_15996 ... a3/envs/throwaway/bin/python3.6]
              8959166  /tmp/build/80754af9/python_15996 ... a3/envs/throwaway/bin/python3.6]
              8959166  /tmp/build/80754af9/python_15996 ... a3/envs/throwaway/bin/python3.6]
                  ...
                92821  /tmp/build/80754af9/python_15996 ... a3/envs/throwaway/bin/python3.6]
                91000  build/../torch/csrc/tensor/pytho ... ch/torch/lib/libtorch_python.so]
                91000  /data/users/test_user/repos/pyto ... nsors::get_default_scalar_type()
                90090  ???:pthread_mutex_lock [/usr/lib64/libpthread-2.28.so]
                90000  build/../c10/core/TensorImpl.h:c ... ch/torch/lib/libtorch_python.so]
                90000  build/../aten/src/ATen/record_fu ... torch/torch/lib/libtorch_cpu.so]
                90000  /data/users/test_user/repos/pyto ... uard(c10::optional<c10::Device>)
                90000  /data/users/test_user/repos/pyto ... ersionCounter::~VersionCounter()
                88000  /data/users/test_user/repos/pyto ... ratorKernel*, at::Tensor const&)""",
        )

        self.regularizeAndAssertExpectedInline(
            wide_linewidth,
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0xXXXXXXXXXXXX>
              408000  ???:__tls_get_addr [/usr/lib64/ld-2.28.so]
              388193  ???:_int_free [/usr/lib64/libc-2.28.so]
              274000  build/../torch/csrc/utils/python_arg_parser.cpp:torch::FunctionSignature ...  bool) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]
              264000  build/../aten/src/ATen/record_function.cpp:at::RecordFunction::RecordFun ... ordScope) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]
              192000  build/../c10/core/Device.h:c10::Device::validate() [/data/users/test_user/repos/pytorch/torch/lib/libc10.so]
              169855  ???:_int_malloc [/usr/lib64/libc-2.28.so]
              154000  build/../c10/core/TensorOptions.h:c10::TensorOptions::merge_in(c10::Tens ... ns) const [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]
              148561  /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:_PyEval_EvalFrameDefault [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]
              135000  ???:malloc [/usr/lib64/libc-2.28.so]
                 ...
                2000  /usr/include/c++/8/ext/new_allocator.h:torch::PythonArgs::intlist(int)
                2000  /usr/include/c++/8/bits/stl_vector.h:torch::PythonArgParser::raw_parse(_object*, _object*, _object*, _object**)
                2000  /usr/include/c++/8/bits/stl_vector.h:torch::PyWarningHandler::~PyWarningHandler()
                2000  /usr/include/c++/8/bits/stl_vector.h:torch::FunctionSignature::parse(_object*, _object*, _object*, _object**, bool)
                2000  /usr/include/c++/8/bits/stl_algobase.h:torch::PythonArgs::intlist(int)
                2000  /usr/include/c++/8/bits/shared_ptr_base.h:torch::autograd::impl::try_get_grad_accumulator(at::Tensor const&)
                2000  /usr/include/c++/8/bits/move.h:c10::TensorImpl::set_autograd_meta(std::u ... AutogradMetaInterface, std::default_delete<c10::AutogradMetaInterface> >)
                2000  /usr/include/c++/8/bits/atomic_base.h:at::Tensor at::detail::make_tensor ... t_null_type<c10::StorageImpl> >&&, c10::DispatchKey&&, caffe2::TypeMeta&)
                2000  /usr/include/c++/8/array:at::Tensor& c10::Dispatcher::callWithDispatchKe ... , c10::Scalar)> const&, c10::DispatchKey, at::Tensor&, c10::Scalar) const

            Total: 8869966"""  # noqa: B950
        )

        self.regularizeAndAssertExpectedInline(
            stats_no_data.as_standardized().stats(),
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0xXXXXXXXXXXXX>
              408000  ???:__tls_get_addr
              388193  ???:_int_free
              274000  build/../torch/csrc/utils/python ... ject*, _object*, _object**, bool)
              264000  build/../aten/src/ATen/record_fu ... ::RecordFunction(at::RecordScope)
              192000  build/../c10/core/Device.h:c10::Device::validate()
              169855  ???:_int_malloc
              154000  build/../c10/core/TensorOptions. ... erge_in(c10::TensorOptions) const
              148561  Python/ceval.c:_PyEval_EvalFrameDefault
              135000  ???:malloc
                 ...
                2000  /usr/include/c++/8/ext/new_allocator.h:torch::PythonArgs::intlist(int)
                2000  /usr/include/c++/8/bits/stl_vect ... *, _object*, _object*, _object**)
                2000  /usr/include/c++/8/bits/stl_vect ... rningHandler::~PyWarningHandler()
                2000  /usr/include/c++/8/bits/stl_vect ... ject*, _object*, _object**, bool)
                2000  /usr/include/c++/8/bits/stl_algobase.h:torch::PythonArgs::intlist(int)
                2000  /usr/include/c++/8/bits/shared_p ... ad_accumulator(at::Tensor const&)
                2000  /usr/include/c++/8/bits/move.h:c ... te<c10::AutogradMetaInterface> >)
                2000  /usr/include/c++/8/bits/atomic_b ... DispatchKey&&, caffe2::TypeMeta&)
                2000  /usr/include/c++/8/array:at::Ten ... , at::Tensor&, c10::Scalar) const

            Total: 8869966""",
        )

        self.regularizeAndAssertExpectedInline(
            deltas,
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0xXXXXXXXXXXXX>
                85000  Objects/dictobject.c:lookdict_unicode
                59089  ???:_int_free
                43000  ???:malloc
                25000  build/../torch/csrc/utils/python ... :torch::PythonArgs::intlist(int)
                24000  ???:__tls_get_addr
                23000  ???:free
                21067  Objects/dictobject.c:lookdict_unicode_nodummy
                20000  build/../torch/csrc/utils/python ... :torch::PythonArgs::intlist(int)
                18000  Objects/longobject.c:PyLong_AsLongLongAndOverflow
                  ...
                 2000  /home/nwani/m3/conda-bld/compile ... del_op.cc:operator delete(void*)
                 1000  /usr/include/c++/8/bits/stl_vector.h:torch::PythonArgs::intlist(int)
                  193  ???:_int_malloc
                   75  ???:_int_memalign
                -1000  build/../c10/util/SmallVector.h: ... _contiguous(c10::ArrayRef<long>)
                -1000  build/../c10/util/SmallVector.h: ... nsor_restride(c10::MemoryFormat)
                -1000  /usr/include/c++/8/bits/stl_vect ... es(_object*, _object*, _object*)
                -8000  Python/ceval.c:_PyEval_EvalFrameDefault
               -16000  Objects/tupleobject.c:PyTuple_New

            Total: 432917""",
        )

        self.regularizeAndAssertExpectedInline(len(deltas), """35""")

        self.regularizeAndAssertExpectedInline(
            deltas.transform(custom_transforms),
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0xXXXXXXXXXXXX>
                85000  Objects/dictobject.c:lookdict_unicode
                59089  ???:_int_free
                43000  ???:malloc
                25000  torch/csrc/utils/python_numbers.h:torch::PythonArgs::intlist(int)
                24000  ???:__tls_get_addr
                23000  ???:free
                21067  Objects/dictobject.c:lookdict_unicode_nodummy
                20000  torch/csrc/utils/python_arg_parser.h:torch::PythonArgs::intlist(int)
                18000  Objects/longobject.c:PyLong_AsLongLongAndOverflow
                  ...
                 2000  c10/util/SmallVector.h:c10::TensorImpl::compute_contiguous() const
                 1000  stl_vector.h:torch::PythonArgs::intlist(int)
                  193  ???:_int_malloc
                   75  ???:_int_memalign
                -1000  stl_vector.h:torch::autograd::TH ... es(_object*, _object*, _object*)
                -1000  c10/util/SmallVector.h:c10::Tens ... _contiguous(c10::ArrayRef<long>)
                -1000  c10/util/SmallVector.h:c10::Tens ... nsor_restride(c10::MemoryFormat)
                -8000  Python/ceval.c:_PyEval_EvalFrameDefault
               -16000  Objects/tupleobject.c:PyTuple_New

            Total: 432917""",
        )

        self.regularizeAndAssertExpectedInline(
            deltas.filter(lambda fn: fn.startswith("???")),
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0xXXXXXXXXXXXX>
              59089  ???:_int_free
              43000  ???:malloc
              24000  ???:__tls_get_addr
              23000  ???:free
                193  ???:_int_malloc
                 75  ???:_int_memalign

            Total: 149357""",
        )

        self.regularizeAndAssertExpectedInline(
            deltas[:5],
            """\
            <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0xXXXXXXXXXXXX>
              85000  Objects/dictobject.c:lookdict_unicode
              59089  ???:_int_free
              43000  ???:malloc
              25000  build/../torch/csrc/utils/python_ ... h:torch::PythonArgs::intlist(int)
              24000  ???:__tls_get_addr

            Total: 236089""",
        )

    def test_compare(self):
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

        def rstrip_lines(s: str) -> str:
            # VSCode will rstrip the `expected` string literal whether you like
            # it or not. So we have to rstrip the compare table as well.
            return "\n".join([i.rstrip() for i in s.splitlines(keepends=False)])

        compare = benchmark_utils.Compare(results)
        self.regularizeAndAssertExpectedInline(
            rstrip_lines(str(compare).strip()),
            """\
            [------------------------------------------------- fn ------------------------------------------------]
                                         |  (16, 16)  |  (16, 128)  |  (128, 128)  |  (4096, 1024)  |  (2048, 2048)
            1 threads: --------------------------------------------------------------------------------------------
                  overhead_optimized     |    1.3     |     3.0     |     17.4     |     4174.4     |     4174.4
                  compute_optimized      |    3.1     |     4.0     |     11.2     |     2099.3     |     2099.3
                  special_case (square)  |    1.1     |             |      7.5     |                |     1674.7

            Times are in microseconds (us)."""
        )

        compare.trim_significant_figures()
        self.regularizeAndAssertExpectedInline(
            rstrip_lines(str(compare).strip()),
            """\
            [------------------------------------------------- fn ------------------------------------------------]
                                         |  (16, 16)  |  (16, 128)  |  (128, 128)  |  (4096, 1024)  |  (2048, 2048)
            1 threads: --------------------------------------------------------------------------------------------
                  overhead_optimized     |     1      |     3.0     |      17      |      4200      |      4200
                  compute_optimized      |     3      |     4.0     |      11      |      2100      |      2100
                  special_case (square)  |     1      |             |       8      |                |      1700

            Times are in microseconds (us)."""
        )

        compare.colorize()
        columnwise_colored_actual = rstrip_lines(str(compare).strip())
        columnwise_colored_expected = textwrap.dedent(
            """\
            [------------------------------------------------- fn ------------------------------------------------]
                                         |  (16, 16)  |  (16, 128)  |  (128, 128)  |  (4096, 1024)  |  (2048, 2048)
            1 threads: --------------------------------------------------------------------------------------------
                  overhead_optimized     |     1      |  \x1b[92m\x1b[1m   3.0   \x1b[0m\x1b[0m  |  \x1b[2m\x1b[91m    17    \x1b[0m\x1b[0m  |      4200      |  \x1b[2m\x1b[91m    4200    \x1b[0m\x1b[0m
                  compute_optimized      |  \x1b[2m\x1b[91m   3    \x1b[0m\x1b[0m  |     4.0     |      11      |  \x1b[92m\x1b[1m    2100    \x1b[0m\x1b[0m  |      2100
                  special_case (square)  |  \x1b[92m\x1b[1m   1    \x1b[0m\x1b[0m  |             |  \x1b[92m\x1b[1m     8    \x1b[0m\x1b[0m  |                |  \x1b[92m\x1b[1m    1700    \x1b[0m\x1b[0m

            Times are in microseconds (us)."""  # noqa: B950
        )

        compare.colorize(rowwise=True)
        rowwise_colored_actual = rstrip_lines(str(compare).strip())
        rowwise_colored_expected = textwrap.dedent(
            """\
            [------------------------------------------------- fn ------------------------------------------------]
                                         |  (16, 16)  |  (16, 128)  |  (128, 128)  |  (4096, 1024)  |  (2048, 2048)
            1 threads: --------------------------------------------------------------------------------------------
                  overhead_optimized     |  \x1b[92m\x1b[1m   1    \x1b[0m\x1b[0m  |  \x1b[2m\x1b[91m   3.0   \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    17    \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    4200    \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    4200    \x1b[0m\x1b[0m
                  compute_optimized      |  \x1b[92m\x1b[1m   3    \x1b[0m\x1b[0m  |     4.0     |  \x1b[2m\x1b[91m    11    \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    2100    \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    2100    \x1b[0m\x1b[0m
                  special_case (square)  |  \x1b[92m\x1b[1m   1    \x1b[0m\x1b[0m  |             |  \x1b[31m\x1b[1m     8    \x1b[0m\x1b[0m  |                |  \x1b[31m\x1b[1m    1700    \x1b[0m\x1b[0m

            Times are in microseconds (us)."""  # noqa: B950
        )

        def print_new_expected(s: str) -> None:
            print(f'{"":>12}"""\\', end="")
            for l in s.splitlines(keepends=False):
                print("\n" + textwrap.indent(repr(l)[1:-1], " " * 12), end="")
            print('"""\n')

        if expecttest.ACCEPT:
            # expecttest does not currently support non-printable characters,
            # so these two entries have to be updated manually.
            if columnwise_colored_actual != columnwise_colored_expected:
                print("New columnwise coloring:\n")
                print_new_expected(columnwise_colored_actual)

            if rowwise_colored_actual != rowwise_colored_expected:
                print("New rowwise coloring:\n")
                print_new_expected(rowwise_colored_actual)

        self.assertEqual(columnwise_colored_actual, columnwise_colored_expected)
        self.assertEqual(rowwise_colored_actual, rowwise_colored_expected)

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
                x, torch.tensor(expected_results[i]), rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    run_tests()
