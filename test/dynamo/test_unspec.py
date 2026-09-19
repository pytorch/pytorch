# Owner(s): ["module: dynamo"]
import contextlib
import itertools
import math
import operator
import random
import subprocess
import sys
import time
import unittest
import warnings
from unittest import mock

import numpy as np

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config as functorch_config
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.comptime import comptime
from torch._dynamo.exc import Unsupported, UserError
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend, same
from torch._dynamo.utils import counters
from torch._dynamo.variables.functions import _TIME_FUNCTION_NAMES
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    skipIfWindows,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU
from torch.testing._internal.logging_utils import logs_to_string


# The intention of this test file is you should put test cases specifically
# for assume_static_by_default=False, aka you want to YOLO make everything as
# dynamic as possible.  If you want to test the more normal situation where
# you assume static by default, put it in a regular test file and
# test_dynamic_shapes will cover both the YOLO and non-YOLO cases.

_EXPECTED_TIME_FUNCTION_NAMES = (
    "clock_gettime",
    "clock_gettime_ns",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "process_time_ns",
    "thread_time",
    "thread_time_ns",
    "time",
    "time_ns",
)

_TIME_FUNCTION_TEST_CASES = tuple(
    (
        name,
        (time.CLOCK_MONOTONIC,) if name.startswith("clock_gettime") else (),
    )
    for name in _EXPECTED_TIME_FUNCTION_NAMES
    if hasattr(time, name)
)


_RANDOM_TEST_POPULATION = [3, 1, 2]
_RANDOM_TEST_RNG = random.Random(0)


@torch._dynamo.config.patch(assume_static_by_default=False)
@instantiate_parametrized_tests
class UnspecTests(torch._dynamo.test_case.TestCase):
    def _assert_module_random_parity(
        self,
        fn,
        *args,
        repeats=3,
        seed=0,
        fullgraph=False,
        expect_graph_break=None,
        backend="eager",
    ):
        # Compiled calls must match eager call by call, and leave the
        # generators in the same state.
        torch._dynamo.reset()
        counters.clear()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=fullgraph)

        def run(f):
            random.seed(seed)
            _RANDOM_TEST_RNG.seed(seed)
            results = [f(*args) for _ in range(repeats)]
            return results, (random.getstate(), _RANDOM_TEST_RNG.getstate())

        expected, expected_state = run(fn)
        # Compile before seeding in case compilation itself draws from random.
        opt_fn(*args)
        actual, actual_state = run(opt_fn)
        self.assertEqual(actual, expected)
        # 1, 1.0 and True compare equal but must keep their types.
        self.assertEqual(repr(actual), repr(expected))
        self.assertEqual(actual_state, expected_state)
        if expect_graph_break is not None:
            graph_broke = bool(counters["graph_break"] or counters["unimplemented"])
            self.assertEqual(graph_broke, expect_graph_break)
        return expected, actual

    def _assert_fullgraph_unsupported(self, fn, *args, regex, exc=Unsupported):
        torch._dynamo.reset()
        with self.assertRaisesRegex(exc, regex):
            torch.compile(fn, backend="eager", fullgraph=True)(*args)

    def test_time_function_names(self):
        self.assertEqual(_TIME_FUNCTION_NAMES, _EXPECTED_TIME_FUNCTION_NAMES)

    def test_numpy_correctness(self):
        def fn(x, y, z):
            xy = [x + y, y, False]
            np_x = x.numpy()
            np_y = y.numpy()
            return {
                "x": x,
                "z": z,
                "a": np_y.sum(),
                "b": xy,
                "c": np_y[0][0] / 68,
                "d": np_x.sum(),
                "e": np_x + np_y,
            }, x + np_y.sum() + z

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        y = torch.ones([2, 2], dtype=torch.int64)
        z = np.int64(12)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x, y, z)
        self.assertEqual(res1, res2)

    def test_no_recompilations(self):
        # no recompilations if passing on different numpy int values
        def fn(x, y):
            return {"a": x + 1, "b": y / 2}

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for i in range(10):
            opt_fn(x, np.int64(i))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    @unittest.expectedFailure  # array scalars decay to 0D arrays
    def test_builtin_max_min(self):
        # test unspecialized primitive max/min
        def fn(x, y, z):
            return z + 1, max(x, y), min(x - 4, y)

        x = np.int64(12)
        y = 10
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x, y, z)
        self.assertTrue(same(res1, res2, relax_numpy_equality=True))

    def test_feed_random_values_into_graph_only(self):
        def fn(shape):
            torch.manual_seed(123)
            x = torch.randn(shape) * random.randint(30, 100)
            return x

        shape = [2, 3]
        random.seed(1)
        res1 = fn(shape)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(shape)
        random.seed(1)
        res2 = opt_fn(shape)

        self.assertTrue(same(res1, res2))

    def test_random_values_with_graph_break(self):
        def fn(x):
            r1 = random.random()
            y = x + random.uniform(10, 20)
            y.sum().item()
            r2 = random.randint(2, 18)  # no graph output in this frame
            y.sum().item()
            return y + r1, r2

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(x)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_random_seed_takes_effect_on_first_call(self):
        # An in-function random.seed() must be visible to a scalar draw
        # traced right after it, including on the very first (compiling)
        # call - unlike test_feed_random_values_into_graph_only and
        # test_random_values_with_graph_break above, this deliberately does
        # NOT "shake out" the compile before comparing, since that's exactly
        # the call this is testing.
        def fn(x):
            random.seed(0)
            return x + random.random(), random.random(), random.randint(0, 100)

        x = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(3):
            res1 = fn(x)
            eager_state = random.getstate()
            res2 = opt_fn(x)
            compiled_state = random.getstate()
            self.assertEqual(res1, res2)
            self.assertEqual(eager_state, compiled_state)
        self.assertEqual(cnts.frame_count, 1)

    # Really annoying intersection of specialization and RandomValueSource
    # If we get a RandomValueSource with a single element tensor, we should return a ConstantVariable like other
    # unspects... but if we do, we break the bytecode assumptions and guards will not work as we will be referring
    # to a name from a source that is not there. If we call .item() and take the wrapped_value out, where we do
    # wrapped_value = wrapped_value.item() where we send unspec down to wrap_fx_proxy, this test passes and then
    # some models fail on missing codegen.tx.output.random_values_var. If we let the tensor value go into wrap as
    # it is, this test fails.
    # The real solution here is to rewrite RandomValueSource and all the codegen it does from the ground up.
    def test_multiple_consecutive_random_calls_before_graph(self):
        def fn(x):
            dim1 = random.randrange(start=0, stop=5)
            dim2 = random.randrange(start=0, stop=5)
            dim3 = random.randrange(start=0, stop=5)
            y = torch.rand(dim1, dim2, dim3)
            return x + 2, y

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(x)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_compiled_random_calls_are_random(self):
        # For compiled functions with random calls,
        # it should return different values for every iteration.
        # https://github.com/pytorch/pytorch/issues/95425
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return (x + 1) * random.uniform(0, 1)

        res = []
        for _ in range(5):
            res.append(fn(torch.ones(2)))
        for i in range(1, 5):
            self.assertFalse(same(res[i - 1], res[i]))

    def test_module_random_random_fullgraph(self):
        # random.random is a C builtin method bound to the module-global
        # Random instance (unlike randint/randrange/uniform, which are Python
        # methods); it must still route through the RandomValueSource path
        # rather than graph-breaking on a skipped builtin.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return (x + 1) * random.random()

        res = []
        for _ in range(5):
            res.append(fn(torch.ones(2)))
        for i in range(1, 5):
            self.assertFalse(same(res[i - 1], res[i]))

    @parametrize("clock_name,clock_args", _TIME_FUNCTION_TEST_CASES)
    def test_time_function_unused_no_warning(self, clock_name, clock_args):
        torch._dynamo.reset()
        clock_fn = getattr(time, clock_name)

        def fn():
            clock_fn(*clock_args)

        opt_fn = torch.compile(fn, backend="eager")
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self.assertIsNone(opt_fn())

        self.assertFalse(
            any(
                f"time.{clock_name}" in str(warning.message)
                for warning in caught_warnings
            )
        )

    def test_time_functions_preserve_call_order(self):
        # Guard against hoisting clock reads into pregraph runtime inputs.
        graph_times = []

        def backend(gm, _):
            def run(*args):
                graph_times.append(time.perf_counter())
                return gm(*args)

            return run

        def fn(x):
            before = time.perf_counter()
            y = x + 1
            after = time.perf_counter()
            return y, before, after

        opt_fn = torch.compile(fn, backend=backend)
        x = torch.zeros(())

        opt_fn(x)
        graph_times.clear()
        result, before, after = opt_fn(x)

        self.assertEqual(result, x + 1)
        self.assertEqual(len(graph_times), 1)
        self.assertLessEqual(before, graph_times[0])
        self.assertLessEqual(graph_times[0], after)

    def test_time_time_does_not_bypass_disable(self):
        # A disabled monkey patch should retain the usual disable graph break.
        @torch.compiler.disable
        def disabled_time():
            return 0.0

        with mock.patch.object(time, "time", disabled_time):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "Skip calling `torch.compiler.disable\\(\\)`d function",
            ):
                torch.compile(lambda: time.time(), backend="eager", fullgraph=True)()

    @unittest.skipIf(IS_FBCODE, "Subprocess spawning doesn't work in fbcode")
    def test_time_function_patch_before_dynamo_import(self):
        script = r"""
import importlib
import os
import sys
import time

import numpy
import torch

if "torch._dynamo.variables.functions" in sys.modules:
    raise AssertionError("Dynamo functions imported before the time patch")

original_process_time = time.process_time
original_perf_counter = time.perf_counter


class UnhashableClock:
    __hash__ = None

    def __call__(self):
        return original_perf_counter()


time.process_time = os.getpid
time.perf_counter = UnhashableClock()
try:
    importlib.import_module("torch._dynamo.variables.functions")
finally:
    time.perf_counter = original_perf_counter

try:
    torch.compile(lambda: time.process_time(), backend="eager", fullgraph=True)()
except torch._dynamo.exc.Unsupported as exc:
    if "Attempted to call function marked as skipped" not in str(exc):
        raise
else:
    raise AssertionError("The monkey-patched time.process_time was treated as a clock")

time.process_time = original_process_time
torch._dynamo.reset()
try:
    torch.compile(lambda: time.process_time(), backend="eager", fullgraph=True)()
except torch._dynamo.exc.Unsupported as exc:
    if "Call to a time function" not in str(exc):
        raise
else:
    raise AssertionError("The restored time.process_time was not treated as a clock")
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=lambda msg: f"{msg}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_random_call_with_while_loop(self):
        def fn(x):
            dim1 = random.randrange(start=0, stop=3)
            dim2 = dim1
            while dim1 == dim2:
                dim2 = random.randrange(start=0, stop=3)
            return x * 2

        x = torch.randn(4)
        random.seed(1)
        res1 = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(x)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

        random.seed(10)
        res1 = fn(x)
        random.seed(10)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_random_object(self):
        # test argument passing, mutation, reconstruction, state correctness
        def fn(x, rand2):
            r1 = random.randint(1, 9)
            r2 = rand2.randint(1, 9)
            rand3 = random.Random(42)
            r3 = rand3.randint(1, 9)

            y = x + r1 + r2 + r3
            return y, rand2, rand3

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        random.seed(0)
        y_1, rand2_1, rand3_1 = fn(inp, random.Random(12))
        state_1 = random.getstate()
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(inp, random.Random(12))
        random.seed(0)
        y_2, rand2_2, rand3_2 = opt_fn(inp, random.Random(12))
        state_2 = random.getstate()
        self.assertEqual(y_1, y_2)
        self.assertEqual(state_1, state_2)
        self.assertEqual(rand2_1.getstate(), rand2_2.getstate())
        self.assertEqual(rand3_1.getstate(), rand3_2.getstate())

    def test_random_object_methods(self):
        def fn(x, rand1, rand2, rand3):
            rand1.seed(42)
            rand4 = random.Random(9002)
            rand2.setstate(rand4.getstate())
            r1 = rand1.random()
            r2 = rand2.randint(1, 10)
            r3 = rand3.randrange(10)
            r4 = rand4.uniform(0, 1)
            return x + r1 + r2 + r3 + r4

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        rand1_1 = random.Random(1)
        rand2_1 = random.Random(2)
        rand3_1 = random.Random(3)
        rand1_2 = random.Random(1)
        rand2_2 = random.Random(2)
        rand3_2 = random.Random(3)
        y1 = fn(inp, rand1_1, rand2_1, rand3_1)
        y2 = opt_fn(inp, rand1_2, rand2_2, rand3_2)
        self.assertEqual(y1, y2)
        self.assertEqual(rand1_1.getstate(), rand1_2.getstate())
        self.assertEqual(rand2_1.getstate(), rand2_2.getstate())
        self.assertEqual(rand3_1.getstate(), rand3_2.getstate())

    def test_random_object_shuffle(self):
        # shuffle on an explicit Random object is an exact, reproducible,
        # index-only permutation, so it also works for non-constant (tensor)
        # elements. The trailing draw checks the RNG state advanced correctly.
        def fn(x):
            r = random.Random(42)
            items = list(range(10))
            r.shuffle(items)
            tensors = [x + i for i in range(5)]
            r.shuffle(tensors)
            return items, tensors, r.random()

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref_items, ref_tensors, ref_r = fn(x)
        res_items, res_tensors, res_r = opt_fn(x)
        self.assertEqual(ref_items, res_items)
        self.assertEqual(ref_tensors, res_tensors)
        self.assertEqual(ref_r, res_r)

    def test_random_object_sample(self):
        # sample selects index positions, so it is exact/reproducible for an
        # explicit Random object and works over sequence populations (ranges,
        # strings) as well as non-constant (tensor) elements.
        def fn(x):
            r = random.Random(42)
            from_range = r.sample(range(100), 5)
            from_str = r.sample("abcdefghij", 3)
            tensors = [x + i for i in range(8)]
            from_tensors = r.sample(tensors, 3)
            return from_range, from_str, from_tensors, r.random()

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])
        self.assertEqual(ref[2], res[2])
        self.assertEqual(ref[3], res[3])

    def test_random_object_sample_raises(self):
        # A sample larger than the population raises ValueError like eager.
        def fn():
            r = random.Random(0)
            return r.sample([1, 2, 3], 5)

        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaises(ValueError):
            opt_fn()

    def test_random_module_shuffle_sample(self):
        def fn(x):
            items = list(range(10))
            random.shuffle(items)
            picks = random.sample(tuple(range(10)), 4)
            empty = []
            random.shuffle(empty)
            no_picks = random.sample(tuple(range(3)), 0)
            # Constants read from globals are guarded, so they replay too.
            from_global = list(_RANDOM_TEST_POPULATION)
            random.shuffle(from_global)
            # Up to 256 results are replayed.
            limit = list(range(256))
            random.shuffle(limit)
            floats = [0.0, -0.5, 1.0, -2.5]
            random.shuffle(floats)
            return (
                items,
                picks,
                empty,
                no_picks,
                from_global,
                random.sample(_RANDOM_TEST_POPULATION, 2),
                limit,
                floats,
                random.sample(list(range(1000)), 256),
                random.sample(range(10**6), 3),
                random.random(),
                x + 1,
            )

        _, results = self._assert_module_random_parity(
            fn, torch.zeros(2), fullgraph=True
        )
        self.assertNotEqual(results[0][:2], results[1][:2])

    @parametrize("seeded", [False, True])
    def test_random_module_aliases(self, seeded):
        # Imported and module-level names are bound to one generator; their
        # calls, including seed, are replayed in order on it.
        from random import randint, sample, seed, shuffle

        shuffle_alias = random.shuffle

        def fn(x):
            if seeded:
                seed(1)
            first, second, third = [1, 2, 3], [4, 5, 6], [7, 8, 9]
            shuffle(first)
            picks = sample(range(10), 3)
            shuffle_alias(second)
            random.shuffle(third)
            return first, picks, second, third, x + random.random() + randint(0, 9)

        self._assert_module_random_parity(fn, torch.zeros(1), fullgraph=True)

    def test_random_instance_replay(self):
        # A random.Random read from outside the frame (an argument or a global)
        # is replayed like the module-level generator.
        def fn(x, rng):
            items, more = [1, 2, 3, 4], [5, 6, 7]
            rng.shuffle(items)
            _RANDOM_TEST_RNG.shuffle(more)
            picks = rng.sample(range(10), 2)
            return items, more, picks, x + rng.random(), _RANDOM_TEST_RNG.randint(0, 9)

        def run(f):
            rng = random.Random(0)
            _RANDOM_TEST_RNG.seed(0)
            results = [f(torch.zeros(1), rng) for _ in range(3)]
            return results, rng.getstate(), _RANDOM_TEST_RNG.getstate()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        opt_fn(torch.zeros(1), random.Random(0))
        self.assertEqual(run(opt_fn), run(fn))

    def test_random_module_bound_method_guards(self):
        # Another receiver or method recompiles.
        def fn(op):
            items = [1, 2, 3]
            try:
                op(items)
            except TypeError:
                pass
            return items, random.random()

        def run(f):
            random.seed(0)
            rng = random.Random(1)
            ops = [random.shuffle, rng.shuffle, random.seed, random.shuffle]
            return [f(op) for op in ops], random.getstate(), rng.getstate()

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        self.assertEqual(run(opt_fn), run(fn))
        self.assertEqual(counter.frame_count, 3)

    @parametrize("seeded", [False, True])
    def test_random_module_alias_of_global(self, seeded):
        # An argument that is (or is not) the global generator at trace time
        # must not be treated as one later; after a seed, the global results
        # are also constant-folded.
        def fn(rng):
            if seeded:
                random.seed(0)
            items, more = [1, 2, 3], [4, 5, 6]
            rng.shuffle(more)
            random.shuffle(items)
            return items, more

        def run(f):
            random.seed(0)
            other = random.Random(5)
            global_rng = random.shuffle.__self__
            results = [f(global_rng), f(other), f(global_rng), f(other)]
            return results, random.getstate(), other.getstate()

        self.assertEqual(run(torch.compile(fn, backend="eager")), run(fn))

    @parametrize("form", ["method", "function"])
    @parametrize("op", ["permute", "transpose", "sum", "select", "reshape", "repeat"])
    def test_random_module_static_metadata(self, op, form):
        # Random sizes are traced. Random dims are data-dependent: fullgraph
        # fails and ordinary compilation falls back to eager.
        def fn(x):
            dims = list(range(x.ndim))
            random.shuffle(dims)
            sizes = [1, 2, 4]
            random.shuffle(sizes)
            args, kwargs = {
                "permute": ((dims,), {}),
                "transpose": ((dims[0], dims[1]), {}),
                "sum": ((), {"dim": dims[0]}),
                "select": ((dims[0], 0), {}),
                "reshape": ((sizes,), {}),
                "repeat": ((sizes,), {}),
            }[op]
            if form == "method":
                return getattr(x, op)(*args, **kwargs)
            func = torch.Tensor.repeat if op == "repeat" else getattr(torch, op)
            return func(x, *args, **kwargs)

        x = torch.randn(2, 2, 2)
        traced = op in ("reshape", "repeat")
        self._assert_module_random_parity(
            fn, x, fullgraph=traced, expect_graph_break=not traced
        )
        if not traced:
            self._assert_fullgraph_unsupported(
                fn, x, regex="data-dependent", exc=UserError
            )

    @parametrize(
        "kind",
        ["setitem", "getitem", "operator_getitem", "draw", "tuple_get", "tuple_set"],
    )
    def test_random_module_list_index(self, kind):
        # A list of runtime ints indexes like the list of Python ints.
        def fn(mask):
            prune = random.sample(list(range(mask.shape[0])), 2)
            if kind == "getitem":
                return mask[prune], prune
            if kind == "operator_getitem":
                return operator.getitem(mask, prune), prune
            if kind == "tuple_get":
                return mask.view(1, -1)[:, prune], prune
            if kind == "tuple_set":
                result = mask.clone().view(1, -1)
                result[:, prune] = False
                return result, prune
            if kind == "draw":
                prune = [random.randint(0, 3), prune[0]]
            result = mask.clone()
            result[prune] = False
            return result, prune

        self._assert_module_random_parity(
            fn, torch.ones(4, dtype=torch.bool), fullgraph=True
        )

    def test_random_module_sample_scalar_tensor_index(self):
        def fn(x):
            index = random.sample(range(x.shape[0]), 1)[0]
            result = x.clone()
            result[index] = 7
            return result, x[index, :]

        self._assert_module_random_parity(
            fn, torch.arange(8).reshape(4, 2), fullgraph=True
        )

    @parametrize("kind", ["shuffle_int", "shuffle_float", "draw"])
    def test_random_module_scalar_type_queries(self, kind):
        def fn(x):
            if kind == "shuffle_int":
                items = [1, 2, 3]
                random.shuffle(items)
                item = items[0]
            elif kind == "shuffle_float":
                items = [0.5, 1.5, 2.5]
                random.shuffle(items)
                item = items[0]
            else:
                item = random.randint(1, 3)
            checks = (
                isinstance(item, int),
                isinstance(item, float),
                isinstance(item, (int, float)),
                isinstance(item, torch.Tensor),
                type(item) is int,
                type(item) is float,
                hasattr(item, "is_integer"),
                hasattr(item, "bit_length"),
                hasattr(item, "__name__"),
            )
            return [x + (1 if check else 2) for check in checks]

        self._assert_module_random_parity(fn, torch.zeros(1), fullgraph=True)

    def test_symint_hasattr(self):
        def fn(x, n):
            return x + (1 if hasattr(n, "__name__") else 2) + hasattr(n, "bit_length")

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, fullgraph=True)
        self.assertEqual(opt_fn(torch.zeros(1), 5), fn(torch.zeros(1), 5))
        self.assertEqual(opt_fn(torch.zeros(1), 6), fn(torch.zeros(1), 6))
        self.assertEqual(counter.frame_count, 1)

    @parametrize(
        "kind",
        [
            "strings",
            "mixed",
            "bool",
            "nan",
            "negative_zero",
            "big_int",
            "tensors",
            "shuffled_twice",
            "shuffle_257",
            "sample_257",
            "range_sample_257",
            "range_big",
            "shuffle_non_list",
            "sample_counts",
            "draw_dynamic_arguments",
            "list_index_mixed",
            "list_index_numpy",
            "hash_shuffle",
            "hash_draw",
            "hash_tuple_key",
            "int_add",
            "int_sum",
            "int_floordiv",
            "int_float",
            "custom_instancecheck",
            "tuple_index",
            "getstate",
        ],
    )
    def test_random_module_falls_back(self, kind):
        # Cases that cannot be represented faithfully run eagerly and, where a
        # dedicated graph break exists, raise it under fullgraph.
        class Meta(type):
            def __instancecheck__(cls, value):
                return value > 1

        class Big(metaclass=Meta):
            pass

        populations = {
            "strings": ["a", "b", "c"],
            "mixed": [True, 1, 2.0],
            "bool": [True, True, False],
            "nan": [1.0, float("nan"), 2.0],
            "negative_zero": [0.0, -0.0, 1.0],
            "big_int": [2**70, 1],
            "shuffle_257": list(range(257)),
        }

        def fn(x, n):
            items = [3, 1, 2]
            random.shuffle(items)
            if kind in populations:
                values = list(populations[kind])
                random.shuffle(values)
                return values, x + 1
            if kind == "tensors":
                values = [x + i for i in range(3)]
                random.shuffle(values)
                return values
            if kind == "shuffled_twice":
                random.shuffle(items)
                return items
            if kind == "sample_257":
                return random.sample(list(range(300)), 257)
            if kind == "range_sample_257":
                return random.sample(range(300), 257)
            if kind == "range_big":
                return random.sample(range(2**63, 2**63 + 10), 3)
            if kind == "shuffle_non_list":
                try:
                    random.shuffle((1, 2, 3))
                except TypeError as e:
                    return str(e), x + 1
            if kind == "sample_counts":
                return random.sample(["a", "b"], counts=[2, 1], k=2), x + 1
            if kind == "draw_dynamic_arguments":
                return x + _RANDOM_TEST_RNG.randint(0, n)
            if kind == "list_index_mixed":
                return torch.arange(4)[[items[0], 1]]
            if kind == "list_index_numpy":
                return np.arange(4)[items[:2]]
            if kind.startswith("hash_"):
                key = {"hash_shuffle": items[0], "hash_draw": random.randint(1, 3)}
                if kind == "hash_tuple_key":
                    return x * len({(items[0], items[1]), (items[1], items[2])})
                mapping = {1: "a", 2: "b", 3: "c", 4: "d"}
                return mapping.get(key[kind], "MISS"), key[kind] in {1, 2}
            if kind.startswith("int_"):
                # The traced 0-d int64 tensor could overflow, or give a float32
                # result with a float.
                return {
                    "int_add": lambda: items[0] + 1,
                    "int_sum": lambda: sum(items),
                    "int_floordiv": lambda: items[0] // 2,
                    "int_float": lambda: items[0] + 0.1,
                }[kind]()
            if kind == "custom_instancecheck":
                return x + (1 if isinstance(items[0], Big) else 2)
            if kind == "tuple_index":
                order = [0] * n + [1] * n
                random.shuffle(order)
                lists = ([], [])
                its = itertools.tee(iter(range(n)))
                for i in order:
                    lists[i].append(next(its[i]))
                return lists
            state = random.getstate()
            first = random.random()
            random.setstate(state)
            return first, random.random()

        regex = {
            "shuffle_non_list": "random.shuffle on a non-list sequence",
            "sample_counts": "random.sample with counts",
            "draw_dynamic_arguments": "Random draw with non-constant arguments",
            "list_index_mixed": "list index of unspecialized scalars",
            "list_index_numpy": "list index of unspecialized scalars",
            "hash_shuffle": "Unspecialized Python scalar used as hash key",
            "hash_draw": "Unspecialized Python scalar used as hash key",
            "hash_tuple_key": "Unspecialized Python scalar used as hash key",
            "getstate": "Random state read while unknown",
        }.get(kind, "Random sequence requires eager execution")
        args = (torch.ones(1), 5)
        self._assert_module_random_parity(
            fn, *args, repeats=10, expect_graph_break=True
        )
        if not kind.startswith(("int_", "custom", "tuple")):
            self._assert_fullgraph_unsupported(fn, *args, regex=regex)

    def test_random_module_python_errors(self):
        # Invalid calls raise the same exceptions as eager.
        def fn(x):
            items = [1, 2, 3]
            random.shuffle(items)
            errors = []
            calls = [
                lambda: random.randint(3, 1),
                lambda: random.seed([1]),
                lambda: random.setstate((99, *random.Random(0).getstate()[1:])),
            ]
            state = random.Random(0).getstate()
            calls.append(
                lambda: random.setstate((state[0], (2**64,) + state[1][1:], state[2]))
            )
            for population, k in [
                (range(3), -1),
                (range(3), 4),
                ((), 1),
                (range(3), 1.5),
                (range(3), 4.5),
            ]:
                calls.append(lambda p=population, k=k: random.sample(p, k))
            if sys.version_info >= (3, 11):
                calls.append(lambda: random.sample({1, 2, 3}, 2))
            for call in calls:
                try:
                    call()
                    errors.append(None)
                except (TypeError, ValueError, OverflowError) as e:
                    errors.append(type(e).__name__)
            return (
                items,
                errors,
                random.sample(range(10), k=3),
                random.sample([1, 2, 3], k=1),
                x + random.random(),
            )

        self._assert_module_random_parity(fn, torch.zeros(1), fullgraph=True)

    def test_random_module_large_sequence_is_cached(self):
        def fn(x):
            x = x + 1
            items = list(range(5000))
            random.shuffle(items)
            return x + items[0]

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        x = torch.zeros(1)
        random.seed(0)
        expected = [fn(x) for _ in range(6)]
        random.seed(0)
        actual = [opt_fn(x) for _ in range(3)]
        frame_count = counter.frame_count
        actual += [opt_fn(x) for _ in range(3)]
        self.assertEqual(actual, expected)
        self.assertEqual(counter.frame_count, frame_count)

    def test_random_module_sample_symbolic_size(self):
        # UnspecTests makes int inputs dynamic, so k arrives as a SymInt and is
        # specialized: each k compiles once.
        def fn(x, k):
            return random.sample(range(10), k), x + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, fullgraph=True)
        for k in (2, 3, 4, 2):
            random.seed(0)
            expected = fn(torch.zeros(1), k)
            random.seed(0)
            self.assertEqual(opt_fn(torch.zeros(1), k), expected)
        self.assertEqual(counter.frame_count, 3)

    @parametrize("kind", ["shuffle", "sample"])
    def test_random_module_export_sequence(self, kind):
        class Model(torch.nn.Module):
            def forward(self, x):
                items = list(range(5))
                if kind == "shuffle":
                    random.shuffle(items)
                    chosen = items[0]
                else:
                    chosen = random.sample(items, 1)[0]
                return x + chosen

        exported = torch.export.export(Model(), (torch.ones(1),), strict=True)
        # Export records the trace-time result as a constant.
        results = [exported.module()(torch.ones(1)).item() for _ in range(5)]
        self.assertEqual(len(set(results)), 1)
        self.assertIn(results[0], {1.0, 2.0, 3.0, 4.0, 5.0})

    def test_random_module_sort_full_shuffle(self):
        def fn(x):
            ints, floats = list(range(20)), [i / 4 for i in range(1, 21)]
            int_copy, float_copy = list(ints), list(floats)
            random.shuffle(ints)
            ints.sort(reverse=True)
            random.shuffle(float_copy)
            random.shuffle(int_copy)
            return ints, sorted(int_copy), sorted(float_copy), x + random.random()

        self._assert_module_random_parity(fn, torch.zeros(1), fullgraph=True)

    @parametrize("kind", ["partial", "key", "two_shuffles", "sample", "duplicate"])
    def test_random_module_sort_not_folded(self, kind):
        def fn():
            data = list(range(10))
            random.shuffle(data)
            if kind == "partial":
                return sorted(data[:5])
            if kind == "key":
                return sorted(data, key=lambda v: -v)
            if kind == "sample":
                return sorted(random.sample(range(10), 10))
            if kind == "duplicate":
                return sorted(data + data[:1])
            other = list(range(10))
            random.shuffle(other)
            return sorted(data[:5] + other[5:])

        self._assert_module_random_parity(fn, expect_graph_break=True)

    def test_random_module_seeded_sequence(self):
        # After an in-frame constant seed the results are constants, so any
        # population can be traced.
        def fn(x):
            random.seed(0)
            first = random.random(), random.randint(0, 9)
            items = ["a", "b", "c", "d"]
            random.shuffle(items)
            random.seed(a=1)
            items += random.sample(["p", "q", "r"], 2)
            random.setstate(random.Random(2).getstate())
            state = random.getstate()
            draw = random.random()
            random.setstate(state)
            return (
                first,
                items,
                random.sample(range(1000), 3),
                draw,
                x + random.random(),
            )

        _, results = self._assert_module_random_parity(
            fn, torch.zeros(1), fullgraph=True
        )
        self.assertEqual(results[0], results[1])

    @parametrize("seed_call", ["no_args", "none", "none_keyword"])
    def test_random_module_entropy_seed(self, seed_call):
        # seed() and seed(None) draw from OS entropy, so the state stays unknown.
        def seed():
            if seed_call == "no_args":
                random.seed()
            elif seed_call == "none":
                random.seed(None)
            else:
                random.seed(a=None)

        def draw():
            seed()
            return random.random()

        opt_draw = torch.compile(draw, backend="eager", fullgraph=True)
        self.assertNotEqual(opt_draw(), opt_draw())

        def shuffle_after_seed():
            seed()
            items = ["a", "b", "c"]
            random.shuffle(items)
            return items

        self._assert_fullgraph_unsupported(
            shuffle_after_seed, regex="Random sequence requires eager execution"
        )

    def test_random_module_seed_does_not_survive_graph_break(self):
        def fn():
            random.seed(0)
            torch._dynamo.graph_break()
            items = ["a", "b", "c"]
            random.shuffle(items)
            return items

        self._assert_module_random_parity(fn, expect_graph_break=True)
        self.assertTrue(
            any(
                "Random sequence requires eager execution" in reason
                for reason in counters["graph_break"]
            )
        )

    @parametrize("keyword", [False, True])
    def test_random_module_setstate_from_input(self, keyword):
        def fn(x, state):
            if keyword:
                random.setstate(state=state)
            else:
                random.setstate(state)
            return x + random.random()

        opt_fn = torch.compile(fn, backend="eager")
        for seed in range(4):
            state = random.Random(seed).getstate()
            self.assertEqual(opt_fn(torch.zeros(1), state), fn(torch.zeros(1), state))
        self._assert_fullgraph_unsupported(
            fn,
            torch.zeros(1),
            random.Random(0).getstate(),
            regex="Random state set from a non-constant value",
        )

    def test_random_object_overridden_methods(self):
        # these will result in graph breaks, but we shouldn't crash
        def get_rng():
            rand1 = random.Random(1)
            rand2 = random.Random(2)

            orig_random = rand1.random

            def custom_random():
                return orig_random()

            orig_getstate = rand2.getstate

            def custom_getstate():
                return orig_getstate()

            rand1.random = custom_random
            rand2.getstate = custom_getstate
            return rand1, rand2

        def fn(x, rand1, rand2):
            r1 = rand1.random()
            rand3 = random.Random()
            rand3.setstate(rand2.getstate())
            r2 = rand3.random()
            return x + r1 + r2

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        y1 = fn(inp, *get_rng())
        y2 = opt_fn(inp, *get_rng())
        self.assertEqual(y1, y2)

    def test_random_in_dynamo(self):
        # test that system random calls still work even
        # if Dynamo calls random methods.

        exit_stack = contextlib.ExitStack()

        def patch_fn_with_rng_burn(name):
            orig_fn = eval(name)

            def bad(*args, **kwargs):
                # burn random call within dynamo
                random.random()
                return orig_fn(*args, **kwargs)

            exit_stack.enter_context(unittest.mock.patch(name, bad))

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # we don't guard against random calls in eval_frame.py today
        # patch_fn_with_rng_burn("torch._dynamo.eval_frame._maybe_set_eval_frame")
        patch_fn_with_rng_burn("torch._dynamo.convert_frame._compile")
        patch_fn_with_rng_burn(
            "torch._dynamo.symbolic_convert.InstructionTranslator.run"
        )

        def f1(x):
            # simple test
            r1 = random.randint(1, 9)
            y = x + random.uniform(10, 20)
            r2 = random.randrange(0, 10)
            return y + r1, r2

        random.seed(1)
        ref1 = f1(x)
        opt_f1 = torch.compile(f1, backend="eager", fullgraph=True)
        random.seed(1)
        res1 = opt_f1(x)
        self.assertEqual(ref1, res1)

        def f2(x):
            # test with graph breaks
            r1 = random.randint(1, 9)
            x = x + r1
            torch._dynamo.graph_break()
            r2 = random.randint(10, 19)
            x = x + r2
            return x, r1, r2

        random.seed(2)
        ref2 = f2(x)
        opt_f2 = torch.compile(f2, backend="eager")
        random.seed(2)
        res2 = opt_f2(x)
        self.assertEqual(ref2, res2)

        def f3(x):
            # test consecutive calls
            return x + random.randint(1, 10)

        random.seed(3)
        ref3 = f3(x)
        ref3_ = f3(x)
        opt_f3 = torch.compile(f3, backend="eager", fullgraph=True)
        random.seed(3)
        res3 = opt_f3(x)
        res3_ = opt_f3(x)
        self.assertEqual(ref3, res3)
        self.assertEqual(ref3_, res3_)

    def test_builtin_getitem(self):
        # builtin getitem args[0] is python list and args[1] is unspec
        def fn(x, idx):
            return (torch.zeros(idx), x[idx], x[idx:])

        x = list(range(50))
        ref = fn(x, 48)  # 48 is unspecialized
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, 48)
        self.assertTrue(same(ref, res))

    def test_use_and_specialize(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x, y):
            x = x + y
            if y == 2:
                return x - 1
            else:
                return x + 1

        self.assertTrue(same(fn(torch.tensor([5]), 2), 6))
        self.assertTrue(same(fn(torch.tensor([6]), 2), 7))
        self.assertTrue(same(fn(torch.tensor([5]), 3), 9))
        self.assertTrue(same(fn(torch.tensor([4]), 3), 8))
        self.assertEqual(cnt.frame_count, 2)

    def test_no_recompiles(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x, y):
            return x + y

        self.assertTrue(same(fn(torch.tensor([5]), 100), 105))
        self.assertTrue(same(fn(torch.tensor([4]), 200), 204))
        self.assertTrue(same(fn(torch.tensor([3]), 300), 303))
        self.assertTrue(same(fn(torch.tensor([2]), 400), 402))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_no_recompiles_prod_backward(self):
        # https://github.com/pytorch/pytorch/issues/120608
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(t):
            return torch.prod(t, 3, keepdim=True)

        input_shapes = [(8, 10, 3, 2), (8, 3, 5, 2), (8, 4, 8, 2)]
        for s in input_shapes:
            t1 = torch.randn(s, requires_grad=True)
            h_result = fn(t1)
            grad = torch.ones_like(h_result)
            h_result.backward(grad)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_unspec_float_precision(self):
        def fn(image, scale_factor):
            image = torch.nn.functional.interpolate(
                image[None],
                size=None,
                scale_factor=scale_factor,
                mode="bilinear",
                recompute_scale_factor=True,
                align_corners=False,
            )[0]

            return image.shape

        x = torch.rand([3, 427, 640])
        scale_factor = 1.873536229133606
        ref = fn(x, scale_factor)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, scale_factor)
        self.assertTrue(same(ref, res))

    @unittest.expectedFailure  # fails as long as numpy scalars are 0D arrays
    def test_specializing_numpy_float_in_control_flow(self):
        # np.float64 is unspecialized by default,
        # but it should be specialized when used in control flow.
        def fn(x, y):
            if y > 1.0:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        for t in [np.float16, np.float32, np.float64]:
            y = t(1.23)
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, res))

    def test_mark_static_inside(self):
        def fn(x):
            torch._dynamo.mark_static(x, 0)
            comptime.assert_static(x.size(0))
            return x + 1

        opt_fn = torch.compile(fn, dynamic=True, fullgraph=True, backend="eager")
        opt_fn(torch.randn(12, 23))

    def test_shape_graph_break(self):
        from torch._dynamo.comptime import comptime

        def fn(x):
            x_shape = x.size()
            comptime.graph_break()
            return x + torch.randn(x_shape)

        x = torch.randn(20)
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(x)

    def test_isinstance_symint(self):
        def fn(x):
            assert isinstance(x.size(0), int)  # noqa: S101
            return x * 2

        x = torch.randn(20)
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(x)
        y = torch.randn(30)
        torch._dynamo.mark_dynamic(y, 0)
        opt_fn(y)

    def test_mark_01_dynamic(self):
        def fn(x):
            return x * 2

        x = torch.randn(1)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager")
        # This will fail to compile a generic kernel, but we should not
        # complain about it (mark dynamic will try its best but 0/1
        # specialization is allowed)
        opt_fn(x)

    def test_conv1d_symint_padding(self):
        kernel = torch.randn(1, 1, 4)

        def func(x):
            padding = math.ceil((kernel.shape[-1] + x.shape[-1] % 2) / 2) - 1
            out = F.conv1d(x, kernel, padding=padding, stride=2)
            return out

        opt_func = torch.compile(func, backend="eager")

        x = torch.randn(1, 1, 175)
        opt_func(x)  # passes
        x = torch.randn(1, 1, 249)
        opt_func(x)  # crashes

    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_propagate_dynamic_dim(self):
        x = torch.randn(20)
        torch._dynamo.mark_dynamic(x, 0)

        @torch.compile()  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            y = x * 2
            comptime.graph_break()
            z = y * 2
            return z

        z = fn(x)
        self.assertEqual(z._dynamo_propagated_dynamic_indices, {0})

    def test_rshift_dynamic(self):
        def shift_right(tensor: torch.Tensor) -> torch.Tensor:
            return (tensor >> 2).to(torch.long)

        opt_fn = torch.compile(
            shift_right, fullgraph=True, dynamic=True, backend="eager"
        )
        sample_input = torch.tensor([4, 4, 16, 32], dtype=torch.uint8)
        opt_fn(sample_input)

    def test_lshift_scalar_dynamic(self):
        def shift_left(x: int) -> int:
            return 1 << x

        opt_fn = torch.compile(
            shift_left, fullgraph=True, dynamic=True, backend="eager"
        )
        self.assertEqual(opt_fn(1), 2)
        self.assertEqual(opt_fn(5), 32)

    def test_rshift_scalar_dynamic(self):
        def shift_right(x: int) -> int:
            return 64 >> x

        opt_fn = torch.compile(
            shift_right, fullgraph=True, dynamic=True, backend="eager"
        )
        self.assertEqual(opt_fn(2), 16)
        self.assertEqual(opt_fn(4), 4)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symfloat_to_tensor(self):
        def f1(v):
            return torch.tensor([v.item()])

        def f2(v):
            return torch.tensor([[v.item()], [2.0]])

        def f3(v):
            return torch.tensor(v.item())

        def f4(v):
            return torch.tensor((v.item(),))

        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        r = torch.randn(1)

        self.assertEqual(f1(r), optimize(f1)(r))
        self.assertEqual(f2(r), optimize(f2)(r))
        self.assertEqual(f3(r), optimize(f3)(r))
        self.assertEqual(f4(r), optimize(f4)(r))

    @skipIfWindows(
        msg="AssertionError: The values for attribute 'dtype' do not match: torch.int32 != torch.int64."
    )
    def test_to_tensor(self):
        def f1():
            a = np.random.uniform(low=-1, high=1, size=(20, 1))
            return torch.tensor([a, a, a, a], dtype=torch.float64)

        def f2():
            a = torch.tensor([[[123]]])
            return torch.tensor([a, a])

        def f3():
            a = torch.tensor(123)
            return torch.tensor([a, a])

        def f4():
            a = torch.tensor(123)
            b = torch.tensor([[[456]]])
            return torch.tensor([a, b])

        def f5():
            a = np.array([1, 2])
            return torch.tensor([a, a])

        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        self.assertEqual(f1().shape, optimize(f1)().shape)
        self.assertEqual(f2(), optimize(f2)())
        self.assertEqual(f3(), optimize(f3)())
        self.assertEqual(f4(), optimize(f4)())
        self.assertEqual(f5(), optimize(f5)())

    def test_sym_int_conversion(self):
        def f(x):
            y = x.size(0)
            return x * int(y == 0)

        opt_fn = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(2, 3)
        opt_fn(x)

    def test_sum_dimlist_spec(self):
        def fn(inputs, dim):
            return torch.sum(inputs, dim)

        inputs = torch.randn(128, 5, 24, 24)
        dim = (-1, 1, 0, 2)
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(inputs, dim), fn(inputs, dim))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_max(self):
        def fn(x):
            return torch.ones(max(x.item(), 1024))

        x = torch.tensor([1000])
        y = torch.tensor([2000])
        compl_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compl_fn(x))
        self.assertEqual(fn(y), compl_fn(y))

    # https://github.com/pytorch/pytorch/issues/104812
    def test_argmin_coerces_symint_to_intlist_spec(self):
        def fn(x, dim):
            # the python arg parser coerces dim into a vector<int>
            return torch.amin(x, dim=dim, keepdim=True)

        x = torch.randn(4, 4, 4)
        dim = 2
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(x, dim), fn(x, dim))

    def test_exponential(self):
        def fn(inputs, op_inputs_dict):
            res = inputs.exponential_(**op_inputs_dict)
            return res

        inputs = torch.randn(2, 3, 4)
        op_inputs_dict = {"lambd": 10, "generator": None}
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(inputs, op_inputs_dict), fn(inputs, op_inputs_dict))

    def test_symbol_guard_limit_before_specialize(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, dynamic=True)
        def fn(x):
            torch._check(x.size(0) != 3)
            torch._check(x.size(0) != 4)
            torch._check(x.size(0) != 5)
            torch._check(x.size(0) != 6)
            return x + 2

        # Control test
        fn(torch.randn(12))
        fn(torch.randn(13))
        fn(torch.randn(14))

        self.assertExpectedInline(cnts.frame_count, """1""")
        cnts.frame_count = 0

        torch._dynamo.reset()

        with torch.fx.experimental._config.patch(
            symbol_guard_limit_before_specialize=3
        ):
            fn(torch.randn(12))
            fn(torch.randn(13))
            fn(torch.randn(14))

            self.assertExpectedInline(cnts.frame_count, """3""")

    def test_defaults(self):
        def g(x, i=8):
            comptime.assert_static(i)
            return x * i

        def fn(x):
            return g(x)

        inputs = torch.randn(2, 3, 4)
        compl_fn = torch.compile(fn, dynamic=True, backend="eager")
        self.assertEqual(compl_fn(inputs), fn(inputs))

    @torch._dynamo.config.patch(specialize_float=False)
    def test_symfloat_no_replacement(self):
        # See https://github.com/pytorch/pytorch/pull/139250 for more context
        # The high level idea is if we don't want to set a replacement where a
        # symbol is on both the right and left side, otherwise we'll end up
        # in an infinite self._find recursion.
        def fn(t, m):
            return 2 * t if m.is_integer() else t

        t = torch.tensor([1])
        compl_fn = torch.compile(fn, dynamic=True, backend="eager")
        self.assertEqual(fn(t, 1.0), compl_fn(t, 1.0))

    def test_symint_number_methods(self):
        def fn(x):
            n = x.size(0)
            bit_length = n.bit_length()
            conjugate = n.conjugate()
            ratio = n.as_integer_ratio()[0]
            return bit_length + conjugate + ratio + n.__int__()

        x = torch.randn(8, 3)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn(x))

    def test_symint_bit_length_wrong_arity(self):
        def fn(x):
            return x.size(0).bit_length(1)

        x = torch.randn(8, 3)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "takes no arguments"
        ):
            compiled(x)

    @torch._dynamo.config.patch(specialize_float=False)
    def test_symfloat_number_methods(self):
        def fn(t, m):
            return (2 * t if m.is_integer() else t) + m.conjugate()

        t = torch.tensor([1.0])
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(t, 1.0), fn(t, 1.0))
        self.assertEqual(compiled(t, 1.5), fn(t, 1.5))

    @torch._dynamo.config.patch(specialize_float=False)
    def test_unspec_roundtrip_float_input(self):
        def f(x, y):
            if y == 5.0:
                return x + 2
            else:
                return x + y
            return (x, y)

        cf = torch.compile(backend="eager", fullgraph=True)(f)
        x = 1.1234567891234568
        y = 1.1234567891234569
        self.assertAlmostEqual(f(x, y), cf(x, y))

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=True)
    def test_unspec_float_input(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            if y == 5.0:
                return x + 2
            else:
                return x + y

        cf = torch.compile(backend=cnts, fullgraph=True)(f)

        x = torch.randn(3)
        self.assertEqual(f(x, 2.0), cf(x, 2.0))
        self.assertEqual(f(x, 3.0), cf(x, 3.0))  # automatic dynamic kicks in here
        self.assertEqual(f(x, 4.0), cf(x, 4.0))
        self.assertExpectedInline(cnts.frame_count, """2""")  # no recompile
        self.assertEqual(f(x, 5.0), cf(x, 5.0))
        self.assertExpectedInline(cnts.frame_count, """3""")  # guard worked
        self.assertEqual(f(x, math.nan), cf(x, math.nan))
        self.assertExpectedInline(cnts.frame_count, """4""")  # nan always recompiles

    @torch._dynamo.config.patch(specialize_float=False, capture_scalar_outputs=True)
    def test_unspecialized_float_multiply_precision(self):
        dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
        for dtype in dtypes:

            def fn(x, y):
                return x * y

            cnt = CompileCounterWithBackend("aot_eager")
            fn_opt = torch.compile(fn, backend=cnt)
            x = torch.randn(5, dtype=dtype, requires_grad=True)
            y1 = 1.00048828125
            y2 = 1.00048828126
            y3 = 1.00048828127

            self.assertEqual(fn_opt(x, y1), fn(x, y1))
            self.assertEqual(fn_opt(x, y2), fn(x, y2))
            self.assertEqual(fn_opt(x, y3), fn(x, y3))
            self.assertEqual(cnt.frame_count, 1)

    # assume_static_by_default=False is needed for frame_count == 1 even when
    # this test is rerun without the class-level patch (see
    # test_nested_graph_breaks_wrapped.py); otherwise the first call compiles a
    # static graph and the second recompiles dynamically.
    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=False)
    def test_unspecialized_float_clamp_tensorify(self):
        # https://github.com/pytorch/pytorch/issues/194976
        def fn(x, limit):
            gate, up = torch.chunk(x, 2, dim=-1)
            gate = F.silu(gate).clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            return gate * up

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch.compile(fn, backend=cnt)
        for limit in [0.5, 1.0, 0.25]:
            x = torch.full((1, 4), 0.75, requires_grad=True)
            x_ref = x.detach().clone().requires_grad_(True)
            actual = fn_opt(x, limit)
            expected = fn(x_ref, limit)
            self.assertEqual(actual, expected)
            actual.sum().backward()
            expected.sum().backward()
            self.assertEqual(x.grad, x_ref.grad)
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(specialize_float=False)
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("force_disable_caches", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_unspecialized_float_untensorifiable_use_specializes(self):
        # https://github.com/pytorch/pytorch/issues/194976
        # torch.full's fill value cannot be tensorified while x * scale can.
        # The surviving use must force Dynamo to specialize and guard on the
        # float; specializing only in the joint graph poisons the
        # AOTAutogradCache with a baked-in value that is silently reused for
        # other values of scale.
        def fn(x, scale):
            return torch.full((3,), scale, device=x.device) + x * scale

        fn_opt = torch.compile(fn, backend="inductor")
        with fresh_cache():
            x = torch.randn(3)
            for scale in [0.5, 0.75, 1.0]:
                self.assertEqual(fn_opt(x, scale), fn(x, scale))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorfiy_python_scalars_1(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            y = x.sum()
            return x + y.item()

        dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
        for dtype in dtypes:
            x = torch.ones(3, 3, dtype=dtype)
            self.assertEqual(f(x), x + x.sum().item())

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorfiy_python_scalars_2(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.item() * x.item() * torch.ones((), dtype=torch.float64)

        x = torch.tensor(1e20, dtype=torch.float32)
        self.assertEqual(
            f(x), x.item() * x.item() * torch.ones((), dtype=torch.float64)
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorfiy_python_scalars_3(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            y = x.item() * 101
            return y * torch.tensor([1], dtype=torch.float32)

        finfo_float16 = torch.finfo(torch.float16)
        x = torch.tensor([finfo_float16.max], dtype=torch.float16)
        self.assertEqual(f(x), x.item() * 101 * torch.tensor([1], dtype=torch.float32))

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=False)
    def test_unspec_float_input_f64(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        cf = torch.compile(backend=cnts, fullgraph=True)(f)

        x = torch.zeros(3, dtype=torch.float64)
        # 17 digits of precision so unrepresentable in float32
        flt = 1.2345678901234567
        self.assertEqual(f(x, flt), cf(x, flt))

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=True)
    def test_unspec_float_output(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + 1, y * 2

        cf = torch.compile(backend=cnts, fullgraph=True)(f)
        x = torch.randn(3)

        self.assertEqual(f(x, 3.0), cf(x, 3.0))
        self.assertEqual(f(x, 4.0), cf(x, 4.0))
        self.assertEqual(f(x, 5.0), cf(x, 5.0))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_data_dependent_evaluate_expr_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        # To ensure that the continuation frame is compiled,
        # have to write the test function in this funny way.
        # See https://github.com/pytorch/pytorch/issues/111918
        def test(y):
            if y > 2:
                return True
            else:
                return False

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1
            y = x.item()
            if test(y):
                return x * 2
            else:
                return x * 3

        x = torch.tensor([3.0])
        fn(x)

        self.assertExpectedInline(cnts.frame_count, """2""")
        self.assertExpectedInline(cnts.op_count, """4""")

    def test_prune_torch_check(self):
        log_stream, ctx = logs_to_string("torch._dynamo.output_graph", "graph_code")

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def f(x, y):
            torch._check(y + 5 == 85)
            torch._check(x.size(0) == 80)

        with ctx():
            f(torch.randn(80, 100), 80)

        out = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()
        self.assertEqual(out.count("torch.ops.aten._assert_scalar.default"), 2)
        self.assertRegex(out, r"l_(y|args_1)_ \+ 5")
        self.assertRegex(out, r"l_(x|args_0)_\.size\(0\)")

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_aot_autograd(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, i):
            y, z = i.tolist()
            return torch.split(x, [y, z])

        print(f(torch.randn(10, requires_grad=True), torch.tensor([7, 3])))

    def test_bool_tensor_ctor(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, dynamic=True, fullgraph=True)
        def f(x):
            y = torch.empty((x.size(0) // 13) * 13)
            return torch.tensor(y.numel() == 0)

        self.assertTrue(f(torch.empty(8)).item())
        self.assertFalse(f(torch.empty(13)).item())

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked(self):
        class TestModel(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                return x * 2

        main_model = TestModel()
        opt_model = torch.compile(main_model, mode="max-autotune", dynamic=True)  # noqa: UNSPECIFIED_BACKEND

        x1 = torch.rand(3, 5, 4, 8)
        x2 = torch.rand(1, 5, 4, 8)

        torch._dynamo.decorators.mark_unbacked(x1, 0)

        o1_ref = main_model(x1, 2)
        o1 = opt_model(x1, 2)
        self.assertEqual(o1_ref, o1)

        o1_2_ref = main_model(x2, 2)
        o1_2 = opt_model(x2, 2)
        self.assertEqual(o1_2_ref, o1_2)

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked_hint_consistency(self):
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        x = torch.randn(1)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        @torch.compile(backend="eager")
        def f(x):
            if guard_size_oblivious(x.size(0) != 1):
                return x + 3
            else:
                return x + 4

        self.assertEqual(f(x), x + 3)

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked_channels_last(self):
        class TestModel(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                return x * 2

        main_model = TestModel()
        opt_model = torch.compile(main_model, mode="max-autotune", dynamic=True)  # noqa: UNSPECIFIED_BACKEND

        x1 = torch.rand(3, 5, 4, 8).to(memory_format=torch.channels_last)
        x2 = torch.rand(1, 5, 4, 8).to(memory_format=torch.channels_last)

        torch._dynamo.decorators.mark_unbacked(x1, 0)

        o1_ref = main_model(x1, 2)
        o1 = opt_model(x1, 2)
        self.assertEqual(o1_ref, o1)

        o1_2_ref = main_model(x2, 2)
        o1_2 = opt_model(x2, 2)
        self.assertEqual(o1_2_ref, o1_2)

    def test_float_guard_source_on_recompile(self):
        # Regression test: when a float attribute triggers recompilation and
        # becomes dynamic, the guard produced by produce_guards_verbose should
        # have a proper source annotation, not "(unknown source)".
        cache = {}

        class Module(torch.nn.Module):
            def __init__(self, key: float):
                super().__init__()
                self.key = key
                cache[key] = torch.randn(16)

            def forward(self, x):
                return x + cache[self.key]

        x = torch.randn(16)
        log_stream, ctx = logs_to_string("torch._dynamo.guards", "guards")
        with ctx():
            for key in [1.0, 2.0, 3.0]:
                model = torch.compile(Module(key))  # noqa: UNSPECIFIED_BACKEND
                model(x)

        guard_log = log_stream.getvalue()
        self.assertNotIn("unknown source", guard_log)


class UnspecTestsDevice(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(assume_static_by_default=False)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Platform does not support efficient attention",
    )
    def test_no_recompilations_with_efficient_attention(self, device):
        if self.device_type == "cpu":
            raise unittest.SkipTest("EFFICIENT_ATTENTION requires a non-CPU device")

        def fn(q, k, v, attn_mask):
            from torch.nn.attention import sdpa_kernel, SDPBackend
            from torch.nn.functional import scaled_dot_product_attention

            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                return scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, scale=1.0
                )

        def make_q_k_v_mask(batch, num_heads, head_dim, seq_len_kv):
            from collections import namedtuple
            from functools import partial

            dtype = torch.float16
            make_tensor = partial(
                torch.rand, device=device, dtype=dtype, requires_grad=True
            )
            seq_len_q = 64
            SdpaShape = namedtuple(
                "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
            )
            query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
            kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
            key, value = make_tensor(kv_shape), make_tensor(kv_shape)
            mask = torch.randn(
                (batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype
            )

            return query, key, value, mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        q, k, v, mask = make_q_k_v_mask(16, 16, 64, 15)
        opt_fn(q, k, v, mask)

        q, k, v, mask = make_q_k_v_mask(16, 16, 64, 16)
        opt_fn(q, k, v, mask)

        self.assertEqual(cnts.frame_count, 1)

    def test_builtin_functions_on_device(self, device):
        def fn(x, scaler):
            m = torch.nn.ReLU()
            m.to(device)
            y = m(x) * scaler
            return y

        x = torch.randn([3, 6], device=device)
        scaler = 0.23  # 0.23 is unspecialized
        ref = fn(x, scaler)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, scaler)
        self.assertTrue(same(ref, res))
        self.assertEqual(ref.device, res.device)

    @dtypes(torch.float32, torch.float64, torch.int64)
    def test_random_float_precision(self, device, dtype):
        # Random floats keep binary64 precision and promote like Python floats
        # (int64 tensor + float is float32). Inductor misreads an input whose
        # dtype differs from the traced one.
        if not (HAS_CPU if self.device_type == "cpu" else HAS_GPU):
            raise unittest.SkipTest("requires inductor")
        from torch._inductor.utils import device_supports_fp64

        if not device_supports_fp64(torch.device(device)):
            raise unittest.SkipTest("Inductor passes Python floats as fp32")

        def fn(x):
            values = [0.12345678901234567, 1.2345678901234567, 2.345678901234567]
            shuffled = list(values)
            random.shuffle(shuffled)
            picked = random.sample(values, 1)[0]
            return x + picked, x + shuffled[0], x + random.random(), shuffled

        x = torch.ones(2, device=device, dtype=dtype)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        opt_fn(x)
        for seed in range(3):
            random.seed(seed)
            expected = fn(x)
            random.seed(seed)
            self.assertEqual(opt_fn(x), expected, atol=0, rtol=0)


instantiate_device_type_tests(UnspecTestsDevice, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
