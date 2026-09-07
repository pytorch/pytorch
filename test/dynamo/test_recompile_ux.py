# Owner(s): ["module: dynamo"]
import gc
import operator
import queue
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import weakref
from functools import cache

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging
from torch._C._dynamo.eval_frame import _clear_cache_entries_for_region
from torch._dynamo.eval_frame import (
    _get_cache_entries_for_region,
    _get_total_cache_entry_count,
)
from torch._dynamo.exc import FailOnRecompileLimitHit
from torch._dynamo.types import FrameAction, FrameExecStrategy
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.logging_utils import kwargs_to_settings, log_settings


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class RecompileUxTests(torch._dynamo.test_case.TestCase):
    # TODO(whc) dynamo actually recompiles one more time than the cache limit
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch("recompile_limit", cls.cache_limit)
        )

    def test_drop_cache_on_skip(self):
        def model(x, i):
            return x + i

        attached = False
        triggered = False

        def trigger():
            nonlocal triggered
            triggered = True

        def compiler(gm, input):
            nonlocal attached
            f = gm.forward
            if attached:
                raise AssertionError("Expected not attached")
            # NB: making this a weakref.ref causes the cycle to no
            # longer be promptly GC'ed
            weakref.finalize(f, trigger)
            attached = True
            return f

        x = torch.randn(2)
        for i in range(2):
            opt_model = torch.compile(model, backend=compiler)
            opt_model(x, i)

        self.assertTrue(triggered)

    def test_loop_torture(self):
        def loop_torture(input, iters):
            out = input
            # randint itself causes one graph break
            for _ in range(iters):
                out += input
            return out

        compile_counter = torch._dynamo.testing.CompileCounter()
        for _ in range(10):
            x = torch.randn(3)
            iters = torch.randint(low=0, high=1000, size=())
            opt_loop_torture = torch.compile(loop_torture, backend=compile_counter)
            opt_loop_torture(x, iters)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        # TODO(whc) these checks fail on py37.  Why?
        # self.assertEqual(counters["frames"]["total"], 2 + self.cache_limit)
        # self.assertEqual(counters["frames"]["ok"], 1 + self.cache_limit)

        # compile_counter only sees frames that were fed to the backend compiler,
        # which is a subset of counters["frames"]["ok"] -- probably because
        # counters["frames"]["ok"] includes frames not containing torch ops?
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_input(self):
        def model(input):
            return input + input

        expected_recompiles = 2
        compile_counter = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch("recompile_limit", expected_recompiles):
            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                for _ in range(10):
                    bsz = torch.randint(low=0, high=1000, size=())
                    x = torch.randn((bsz, 3, 4))
                    opt_model = torch.compile(model, backend=compile_counter)
                    opt_model(x)

        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        self.assertEqual(len(logs.records), 1)
        print(logs.records[0])
        self.assertTrue(
            logs.records[0]
            .getMessage()
            .startswith("torch._dynamo hit config.recompile_limit")
        )

    @unittest.skipIf(
        not torch.cuda.is_available() and not torch.xpu.is_available(),
        "requires cuda or xpu",
    )
    def test_nvfuser_guards(self):
        # we may want to model dynamo's guards sufficiently after nvfuser's ProfilingExecutor guards
        # such that we ensure dynamo is in charge of all the recompilations at the top level,
        # and we could thus simplify the underlying torchscript executor
        def func(a, b, c):
            return a + b * c

        a = torch.rand(3, 4, 5, device=device_type)
        b = torch.rand(3, 4, 5, device=device_type)
        b_v = torch.rand(3, 5, 4, device=device_type).view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device=device_type).permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device=device_type)
        compile_counter = torch._dynamo.testing.CompileCounter()

        with torch._dynamo.config.patch("recompile_limit", 2):
            opt_func = torch.compile(func, backend=compile_counter)
            opt_func(a, b, c)  # warmup
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b, c)  # no guard fail or recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_v, c)  # a view should not cause nvfuser recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_p, c)  # a permutation should cause recompile
            self.assertEqual(compile_counter.frame_count, 2)

    def assert_single_log_contains(self, logs, contains_str):
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(
            logs.records[0].getMessage().find(contains_str) > 0,
            msg=lambda msg: f'{msg}\nExpected to find "{contains_str}" in log "{logs.records[0].getMessage()}"',
        )

    def test_verbose_tensor_check(self):
        def func(a):
            # Warning: choose a function here whose meta implementation lives
            # entirely in C++.  If you do a Python one, Dynamo will dive into
            # torch._refs which is OK but it will muddy up the warnings
            return torch.add(a, 4)

        def cache_fail_test(cached_input, missed_input, expected_failure):
            # TODO(whc) maybe its hacky to have a 'test within a test' but this seemed convenient
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            opt_func = torch.compile(func, backend="eager")
            # warmup
            opt_func(cached_input)

            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                opt_func = torch.compile(func, backend="eager")
                opt_func(missed_input)
            self.assert_single_log_contains(logs, expected_failure)

        a = torch.rand(3, 4, 5)
        cache_fail_test(
            a,
            a[0:2, :, :],
            "tensor 'a' size mismatch at index 0. expected 3, actual 2",
        )
        cache_fail_test(
            a,
            a.clone().as_strided((3, 4, 5), stride=(1, 3, 12)),
            "tensor 'a' stride mismatch at index 0. expected 20, actual 1",
        )
        cache_fail_test(a, a[0, :, :], "tensor 'a' rank mismatch. expected 3, actual 2")
        cache_fail_test(a, a.to("meta"), "tensor 'a' dispatch key set mismatch.")
        cache_fail_test(
            a,
            a.to(torch.float16),
            "tensor 'a' dtype mismatch. expected Float, actual Half",
        )
        a_grad = a.clone()
        a_grad.requires_grad = True
        cache_fail_test(
            a,
            a_grad,
            "tensor 'a' requires_grad mismatch. expected requires_grad=0",
        )

    def test_mismatched_type(self):
        a = torch.rand(3, 4, 5)
        b = torch.rand(3, 4, 5)

        def func(a, b):
            return a + b

        opt_func = torch.compile(func, backend="eager")
        # warmup
        opt_func(a, b)

        with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
            opt_func = torch.compile(func, backend="eager")
            opt_func(a, 1)
        self.assert_single_log_contains(
            logs,
            "expected type of 'b' to be <class 'torch.Tensor'>, but found <class 'int'>",
        )

    def test_mismatched_tensor_type(self):
        a = torch.rand(3, 4, 5)
        b_parameter = torch.nn.Parameter(torch.rand(3, 4, 5))
        b_tensor = torch.rand(3, 4, 5)

        def func(a, b):
            return a + b

        opt_func = torch.compile(func, backend="eager")
        # warmup
        opt_func(a, b_parameter)

        with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
            opt_func = torch.compile(func, backend="eager")
            opt_func(a, b_tensor)
        self.assert_single_log_contains(
            logs,
            "expected type of 'b' to be <class 'torch.nn.parameter.Parameter'>, but found <class 'torch.Tensor'>",
        )

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_fail_on_recompile_limit_hit(self):
        @torch.compile(backend="eager")
        def func(b, a):
            if a:
                return b * 2
            else:
                return b + 1

        func(torch.randn(5), True)
        with self.assertRaises(FailOnRecompileLimitHit):
            func(torch.randn(5), False)

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails(self):
        failure_reasons = []

        def guard_fail_fn(failure):
            failure_reasons.append(failure[0])

        def f(x):
            return torch.relu(x)

        opt_f = torch._dynamo.optimize(
            backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
        )(f)

        for i in range(5):
            failure_reasons.clear()
            opt_f(torch.randn(8 + i))

        failure_str = "\n".join(failure_reasons)
        for line in [
            "tensor 'x' size mismatch at index 0. expected 11, actual 12",
            "tensor 'x' size mismatch at index 0. expected 10, actual 12",
            "tensor 'x' size mismatch at index 0. expected 9, actual 12",
            "tensor 'x' size mismatch at index 0. expected 8, actual 12",
        ]:
            self.assertIn(
                line,
                failure_str,
            )

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails_report_all(self):
        with log_settings(kwargs_to_settings(recompiles_verbose=True)):
            failure_reasons = []

            def guard_fail_fn(failure):
                failure_reasons.append(failure[0])

            def f(x):
                return torch.ones(len(x), x[-1])

            opt_f = torch._dynamo.optimize(
                backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
            )(f)

            opt_f([4, 5, 6])

            def filter_reasons():
                return "\n".join(
                    [
                        line
                        for line in "\n".join(failure_reasons).splitlines()
                        if not line.startswith("___check_type_id")
                    ]
                )

            failure_reasons.clear()
            opt_f([7, 8])

            for line in ["len(x) == 3"]:
                self.assertIn(line, filter_reasons())

            failure_reasons.clear()
            opt_f([9])

            for line in ["len(x) == 2", "len(x) == 3"]:
                self.assertIn(line, filter_reasons())

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_recompile_child_run_only(self):
        def f(x, n):
            if torch.compiler.is_compiling():
                x = x + 1
            x = g(x)
            return h(x) + n

        def g(x):
            if torch.compiler.is_compiling():
                return x + 2
            return x

        def h(x):
            if torch.compiler.is_compiling():
                return x + 4
            return x

        torch.compile(g, backend="eager")(torch.randn(3))
        inp = torch.randn(3)
        opt_f = torch.compile(f, backend="eager")
        opt_f(inp, 0)

        # expect f to run eager, g compiled (from previous invocatino), h eager
        res = opt_f(inp, 1)

        self.assertEqual(res, inp + 3)


class RecompileLimitKwargTests(torch._dynamo.test_case.TestCase):
    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    def test_recompile_limit_basic(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        opt_f = torch.compile(f, backend=cnt, recompile_limit=2)

        opt_f(torch.randn(3), torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 1)

        opt_f(torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64))
        self.assertEqual(self._num_cache_entries(f), 2)

        # Third dtype should NOT trigger recompilation (recompile_limit=2)
        opt_f(torch.randn(3, dtype=torch.float16), torch.randn(3, dtype=torch.float16))
        self.assertEqual(self._num_cache_entries(f), 2)

    def test_recompile_limit_none_uses_global(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        # Without recompile_limit kwarg, uses global config (default 8)
        opt_f = torch.compile(f, backend=cnt)

        for i in range(10):
            dtype = [
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
                torch.int32,
                torch.int64,
                torch.int16,
                torch.int8,
                torch.uint8,
                torch.complex64,
            ][i]
            opt_f(torch.ones(3, dtype=dtype), torch.ones(3, dtype=dtype))

        self.assertEqual(
            self._num_cache_entries(f), torch._dynamo.config.recompile_limit
        )

    def test_recompile_limit_fullgraph_raises(self):
        """With fullgraph=True, hitting the recompile_limit kwarg raises
        FailOnRecompileLimitHit, consistent with the fullgraph contract."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_f = torch.compile(f, backend=cnt, fullgraph=True, recompile_limit=1)

        opt_f(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_f(torch.randn(3, dtype=torch.float64))

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_recompile_limit_resume_function_auto_dynamic(self):
        """With automatic dynamic shapes and recompile_limit=2, the resume
        function recompiles via dimension changes on a global tensor while
        the main function gets cache hits. The resume function should stop
        at 2 entries and fall back to eager."""
        cnt = torch._dynamo.testing.CompileCounter()

        y_holder = {"tensor": torch.randn(4, 8, 2)}

        def f(x):
            x.sin()
            print("graph break")
            return y_holder["tensor"].cos()

        opt_f = torch.compile(f, backend=cnt, recompile_limit=2)

        # Call 1: static compile
        y_holder["tensor"] = torch.randn(4, 8, 2)
        opt_f(torch.randn(4, 8, 2))

        # Call 2: y dim0 changes -> f cache hit, resume recompiles
        y_holder["tensor"] = torch.randn(5, 8, 2)
        opt_f(torch.randn(4, 8, 2))
        frame_count_after_2 = cnt.frame_count

        # Call 3: y dim1 changes -> resume should NOT recompile
        # (resume already has 2 entries = recompile_limit)
        y_holder["tensor"] = torch.randn(5, 9, 2)
        opt_f(torch.randn(4, 8, 2))
        self.assertEqual(cnt.frame_count, frame_count_after_2)

        # Verify f has 1 entry, resume has 2
        num_f_entries = len(torch._dynamo.eval_frame._debug_get_cache_entry_list(f))
        self.assertEqual(num_f_entries, 1)

        from torch._dynamo.resume_execution import ContinueExecutionCache

        resume_codes = list(ContinueExecutionCache.cache[f.__code__].values())
        self.assertTrue(len(resume_codes) > 0, "No resume functions found")
        for resume_code in resume_codes:
            num_resume_entries = len(
                torch._dynamo.eval_frame._debug_get_cache_entry_list(resume_code)
            )
            self.assertEqual(num_resume_entries, 2)


def _count_graphs(graphs, node_op, target):
    return sum(
        any(n.op == node_op and n.target == target for n in gm.graph.nodes)
        for gm in graphs
    )


def _count_sin_graphs(graphs):
    # Graphs with a `call_method` "sin" node (`x.sin()`, the main frame).
    return _count_graphs(graphs, "call_method", "sin")


def _count_add_graphs(graphs):
    # Graphs with an `operator.add` call_function node (`a + 1`, the resume frame).
    return _count_graphs(graphs, "call_function", operator.add)


def _reraise_worker_error(raised):
    # For a concurrency failure the worker traceback is the finding, and a bare
    # "a call wedged" would mask it. Surface the first with its traceback, but
    # name the rest: two threads failing differently is the diagnostic, not one.
    if not raised:
        return
    first = raised[0]
    extra = raised[1:]
    if extra and hasattr(first, "add_note"):
        first.add_note(
            f"+{len(extra)} more worker error(s): " + "; ".join(repr(e) for e in extra)
        )
    raise first


class IsolateRecompilesTests(torch._dynamo.test_case.TestCase):
    """Tests for isolate_recompiles=True on torch.compile().

    Each torch.compile() call with isolate_recompiles=True gets its own
    isolated cache bucket via the per-compile cache map in ExtraState.
    Without isolation, all compile calls on the same code object share a
    single cache — entries from one call interfere with another's lookup,
    recompile limit, and FrameExecStrategy.
    """

    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    # ===== Basic isolation: independent caches per compile call =====

    def test_concurrent_calls_do_not_deadlock_on_the_cache_lock(self):
        """lookup() takes the ExtraState cache lock only to snapshot the
        cache entries -- brief, but it touches Python objects, so the GIL can
        drop under it -- then releases the lock before evaluating guards. A
        thread that blocks on that lock while HOLDING the GIL wedges the
        owner, who needs the GIL to finish. The lock therefore has to release
        the GIL before it waits. A short switch interval makes the handoff
        frequent.

        Stress test; not a deterministic reproduction. A wedge fails this test:
        the joins are bounded by a shared deadline and any thread still alive
        after it trips the assertion below.
        """

        def f(x):
            return x.sin() + x.cos()

        opt = torch.compile(f, backend="eager", dynamic=False)
        args = [torch.randn(n) for n in (3, 4, 5)]
        for arg in args:
            opt(arg)

        errors = queue.SimpleQueue()

        def hammer():
            try:
                for _ in range(200):
                    for arg in args:
                        opt(arg)
            except Exception as e:
                errors.put(e)

        threads = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
        prior_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for thread in threads:
                thread.start()
            deadline = time.monotonic() + 120
            for thread in threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        finally:
            sys.setswitchinterval(prior_interval)
        raised = []
        while not errors.empty():
            raised.append(errors.get_nowait())
        _reraise_worker_error(raised)
        self.assertFalse(any(t.is_alive() for t in threads), "a call wedged")

    def test_reset_code_racing_lookup_does_not_destroy_the_cache_state(self):
        """reset_code can run while other threads are parked on the same
        ExtraState's cache lock -- the lock releases the GIL while it waits --
        so destroying the state there deletes the very mutex the waiter is
        blocked on. reset_code must empty the state in place instead. This
        drives resets against concurrent lookups and the recompiles they
        force; every call must either serve the cache or recompile cleanly,
        and the emptied state must serve fresh compiles like a new one.
        Stress test; not a deterministic reproduction.
        """

        def f(x):
            return x.sin() + x.cos()

        opt = torch.compile(f, backend="eager", dynamic=False)
        args = [torch.randn(n) for n in (3, 4)]
        for arg in args:
            opt(arg)

        errors = queue.SimpleQueue()
        stop = threading.Event()

        def caller():
            try:
                while not stop.is_set():
                    for arg in args:
                        opt(arg)
            except Exception as e:
                errors.put(e)

        def resetter():
            try:
                for _ in range(30):
                    # The public reset path: it holds compile_lock, so it
                    # races the LOOKUPS here (which take no compile lock) but
                    # not an in-flight compile's cache-entry snapshot.
                    torch._dynamo.eval_frame.remove_from_cache(f.__code__)
            except Exception as e:
                errors.put(e)
            finally:
                stop.set()

        threads = [threading.Thread(target=caller, daemon=True) for _ in range(4)]
        threads.append(threading.Thread(target=resetter, daemon=True))
        prior_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for thread in threads:
                thread.start()
            deadline = time.monotonic() + 120
            for thread in threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        finally:
            stop.set()
            sys.setswitchinterval(prior_interval)
        raised = []
        while not errors.empty():
            raised.append(errors.get_nowait())
        _reraise_worker_error(raised)
        self.assertFalse(any(t.is_alive() for t in threads), "a call wedged")
        self.assertEqual(opt(args[0]), f(args[0]))

    def test_concurrent_install_and_reset_against_lookups(self):
        """Eight threads look f up while two install and reset precompile
        entries on its code object. lookup() snapshots precompile_entries
        under the cache lock and raises cache_python_depth, then releases the
        lock and runs their guards -- Python, so the GIL can drop. An
        installer takes the same lock to append to or splice the list; the
        raised depth parks any destroy until the readers holding the snapshot
        finish, so a reader never touches a freed node. The threads are joined
        under a shared deadline and asserted not alive, so a wedge fails this
        test. Stress test; not a deterministic reproduction, and it passes on
        the lock-free parent as well: it guards the locking against
        regressions.
        """
        from torch._C._dynamo.eval_frame import (
            _debug_get_cache_entry_list,
            _debug_get_precompile_entries,
            _load_precompile_entry,
            _reset_precompile_entries_for_owner,
        )

        def f(x):
            return x.sin() + x.cos()

        code = f.__code__
        opt = torch.compile(f, backend="eager", dynamic=False)
        args = [torch.randn(n) for n in (3, 4)]
        expected = [f(arg) for arg in args]
        for arg in args:
            opt(arg)
        # Each installer re-installs the compiled variants' own guard managers
        # and code as precompile entries, so a hit serves the same graph.
        installables = [
            (e.guard_manager, e.code) for e in _debug_get_cache_entry_list(code)
        ]
        self.assertEqual(len(installables), 2)

        errors = queue.SimpleQueue()
        stop = threading.Event()
        owners = [object(), object()]

        def caller():
            try:
                while not stop.is_set():
                    for arg, want in zip(args, expected):
                        if not torch.equal(opt(arg), want):
                            raise AssertionError("lookup served the wrong result")
            except Exception as e:
                errors.put(e)

        def installer(owner):
            try:
                for _ in range(300):
                    for guard_manager, dynamo_code in installables:
                        _load_precompile_entry(
                            code, guard_manager, dynamo_code, -1, owner
                        )
                    _reset_precompile_entries_for_owner(code, -1, owner)
            except Exception as e:
                errors.put(e)

        callers = [threading.Thread(target=caller, daemon=True) for _ in range(8)]
        installers = [
            threading.Thread(target=installer, args=(owner,), daemon=True)
            for owner in owners
        ]
        prior_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for thread in callers + installers:
                thread.start()
            deadline = time.monotonic() + 120
            for thread in installers:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            stop.set()
            # Callers cannot begin exiting until stop.set() above, so a slow
            # installer phase must not spend their grace: give them their own
            # window rather than the already-drawn-down shared deadline.
            deadline = max(deadline, time.monotonic() + 30)
            for thread in callers:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        finally:
            stop.set()
            sys.setswitchinterval(prior_interval)
        raised = []
        while not errors.empty():
            raised.append(errors.get_nowait())
        _reraise_worker_error(raised)
        self.assertFalse(
            any(t.is_alive() for t in callers + installers), "a call wedged"
        )
        # A reset that arrived while a lookup held the lock was parked; the
        # entry reader applies whatever is still parked, and nothing survives.
        for owner in owners:
            _reset_precompile_entries_for_owner(code, -1, owner)
        self.assertEqual(len(_debug_get_precompile_entries(code)), 0)
        self.assertEqual(opt(args[0]), expected[0])

    def test_isolate_recompiles_id_is_thread_local(self):
        """The current region is a property of the call in flight, so it is
        per thread: a worker spawned from inside a region reads the default
        (and compiles into the default bucket) unless it enters a region
        itself, and a region the worker enters is invisible to its parent."""
        from torch._C._dynamo.eval_frame import (
            get_eval_frame_isolate_recompiles_id,
            set_eval_frame_isolate_recompiles_id,
        )

        seen = []

        def worker():
            seen.append(get_eval_frame_isolate_recompiles_id())
            set_eval_frame_isolate_recompiles_id(9)
            seen.append(get_eval_frame_isolate_recompiles_id())

        prior = set_eval_frame_isolate_recompiles_id(7)
        try:
            self.assertEqual(prior, -1)
            self.assertEqual(get_eval_frame_isolate_recompiles_id(), 7)
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertEqual(get_eval_frame_isolate_recompiles_id(), 7)
        finally:
            set_eval_frame_isolate_recompiles_id(prior)
        self.assertEqual(seen, [-1, 9])
        self.assertEqual(get_eval_frame_isolate_recompiles_id(), -1)

    def test_reset_code_from_python_run_by_lookup_is_safe(self):
        """try_lookup_without_guard_eval() holds the recursive cache lock
        across the backend comparison it runs -- Python, which can call
        torch._dynamo back in on the SAME thread. reset_code arriving there
        used to free the very list nodes the interrupted lookup was walking
        (a same-thread use-after-free); it must instead land as if it ran
        just after that lookup."""
        from torch._C._dynamo.eval_frame import reset_code

        def f(x):
            return x.sin() + x.cos()

        code = f.__code__
        resets = []

        class ResettingBackend:
            def __call__(self, gm, example_inputs):
                return gm.forward

            def __eq__(self, other):
                if isinstance(other, ResettingBackend):
                    resets.append(True)
                    reset_code(code)
                    return True
                return NotImplemented

            def __hash__(self):
                return 0

        x = torch.randn(8)
        first, second = ResettingBackend(), ResettingBackend()
        opt1 = torch._dynamo.optimize(backend=first, dynamic=False)(f)
        self.assertEqual(opt1(x), f(x))
        self.assertEqual(_get_total_cache_entry_count(code), 1)
        # A second-but-equal backend makes try_lookup_without_guard_eval
        # compare it against the saved one under the fast-path lock; that
        # __eq__ resets this code object, parking the eviction.
        opt2 = torch._dynamo.optimize(backend=second, dynamic=False)(f)
        # The fast path bails (the entry is guarded), so the fallback lookup
        # runs at depth 0, drains the parked reset, and compiles fresh -- all
        # on this one call.
        self.assertEqual(opt2(x), f(x))
        self.assertGreater(len(resets), 0)
        # The backend is now pointer-identical, so this is a plain cache hit.
        self.assertEqual(opt2(x), f(x))
        self.assertEqual(_get_total_cache_entry_count(code), 1)

    def test_invalidation_racing_a_held_cache_lock_parks_and_drains(self):
        """invalidate() reached from weakref.finalize must never block on
        cache_mutex (GC can fire it while ANOTHER state's lock is held; two
        threads doing that against each other's states deadlock ABBA-style).
        This pins the contended path itself: the very call finalize runs,
        arriving while a lookup holds the lock, must return promptly
        (parked), and the parked invalidation must be applied by a later
        lock holder rather than serve forever."""
        from torch._dynamo.guards import DeletedGuardManagerWrapper

        def f(x):
            return x.sin() + x.cos()

        code = f.__code__
        in_eq = threading.Event()
        release_eq = threading.Event()

        class BlockingBackend:
            def __call__(self, gm, example_inputs):
                return gm.forward

            def __eq__(self, other):
                if isinstance(other, BlockingBackend):
                    in_eq.set()
                    release_eq.wait(timeout=120)
                    return True
                return NotImplemented

            def __hash__(self):
                return 0

        first, second = BlockingBackend(), BlockingBackend()
        x = torch.randn(8)
        opt1 = torch._dynamo.optimize(backend=first, dynamic=False)(f)
        self.assertEqual(opt1(x), f(x))
        wrapper = _get_cache_entries_for_region(code, -1)[0].guard_manager

        errors = queue.SimpleQueue()

        def caller():
            try:
                opt2 = torch._dynamo.optimize(backend=second, dynamic=False)(f)
                opt2(x)
            except Exception as e:
                errors.put(e)

        thread = threading.Thread(target=caller, daemon=True)
        thread.start()
        try:
            # If the caller raised before reaching __eq__, in_eq never fires;
            # surface that exception instead of waiting out the full timeout.
            deadline = time.monotonic() + 120
            while not in_eq.wait(timeout=1):
                if not errors.empty():
                    raise errors.get_nowait()
                self.assertTrue(thread.is_alive(), "caller exited before __eq__")
                now = time.monotonic()
                self.assertLess(now, deadline, "caller never reached __eq__")
            # The caller thread is inside lookup, holding the cache lock.
            # This is exactly what a guarded object's weakref.finalize runs;
            # it must park rather than block behind that lock.
            invalidator_done = threading.Event()

            def invalidator():
                try:
                    wrapper.extra_state.invalidate(
                        DeletedGuardManagerWrapper("test object"),
                        wrapper,
                    )
                    invalidator_done.set()
                except Exception as e:
                    errors.put(e)

            inv_thread = threading.Thread(target=invalidator, daemon=True)
            inv_thread.start()
            self.assertTrue(invalidator_done.wait(timeout=60))
        finally:
            # Join inside finally: an assertion above must not leave the caller
            # running a compile into the next test, holding compile_lock.
            release_eq.set()
            thread.join(timeout=120)
        raised = []
        while not errors.empty():
            raised.append(errors.get_nowait())
        _reraise_worker_error(raised)
        self.assertFalse(thread.is_alive())
        # A later lock holder drains the parked request: the entry reports
        # itself invalidated and a fresh compile serves the next call.
        self.assertEqual(opt1(x), f(x))
        entries = _get_cache_entries_for_region(code, -1)
        self.assertTrue(any(e.trace_annotation == "Invalidated" for e in entries))

    def _install_eager_package(self, fn, region):
        # A DiskDynamoStore round trip is the only way to get a package whose
        # install() loads precompile entries for fn.__code__.
        from torch._dynamo.package import CompilePackage, DiskDynamoStore

        store = DiskDynamoStore()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name
        package = CompilePackage(fn)
        torch._dynamo.optimize(backend="eager", package=package)(fn)(torch.randn(8))
        for backend_id, backend in package.cached_backends.items():
            store.record_eager_backend(backend_id, backend)
        store.save_package(package, path)
        torch._dynamo.reset()
        package, backends = store.load_package(fn, path)
        package.install(backends, isolate_recompiles_id=region)
        return package, backends

    def test_reinstall_survives_an_eviction_parked_by_uninstall(self):
        # uninstall() from Python run BY an in-flight lookup (here a backend
        # __eq__) cannot splice the list that lookup is walking, so its owner
        # eviction is parked. A reinstall that reused the same owner token
        # then lost its fresh entries to that parked eviction at the next
        # depth-zero lock holder; each install now mints its own token.
        from torch._C._dynamo.eval_frame import _debug_get_precompile_entries

        def f(x):
            return x.sin() + x.cos()

        code = f.__code__
        region = 7
        package, backends = self._install_eager_package(f, region)
        self.assertEqual(len(_debug_get_precompile_entries(code)), 1)
        hook = []

        class Backend:
            def __call__(self, gm, example_inputs):
                return gm.forward

            def __hash__(self):
                return 0

            def __eq__(self, other):
                if isinstance(other, Backend):
                    if hook:
                        hook.pop()()
                    return True
                return NotImplemented

        x = torch.randn(8)
        torch._dynamo.optimize(backend=Backend(), dynamic=False)(f)(x)

        def reinstall():
            package.uninstall()
            package.install(backends, isolate_recompiles_id=region)

        hook.append(reinstall)
        torch._dynamo.optimize(backend=Backend(), dynamic=False)(f)(x)
        # The parked eviction has been applied by now; the reinstall's entries
        # must have survived it.
        self.assertEqual(len(_debug_get_precompile_entries(code)), 1)
        package.uninstall()
        self.assertEqual(len(_debug_get_precompile_entries(code)), 0)

    def test_region_clear_from_inside_a_lookup_is_parked(self):
        # _clear_cache_entries_for_region run by a backend __eq__ inside
        # lookup() used to splice and destroy the very list lookup was
        # walking (a use-after-free that segfaulted); it now parks like
        # reset_code does, and the next depth-zero holder applies it.
        def f(x):
            return x.sin() + x.cos()

        code = f.__code__
        hook = []
        compiles = []

        class Backend:
            def __call__(self, gm, example_inputs):
                compiles.append(gm)
                return gm.forward

            def __hash__(self):
                return 0

            def __eq__(self, other):
                if isinstance(other, Backend):
                    if hook:
                        hook.pop()()
                    return True
                return NotImplemented

        x = torch.randn(8)
        opt1 = torch._dynamo.optimize(
            backend=Backend(), dynamic=False, isolate_recompiles=True
        )(f)
        self.assertEqual(opt1(x), f(x))
        region = opt1._isolate_recompiles_id
        self.assertEqual(len(_get_cache_entries_for_region(code, region)), 1)
        ctx = torch._dynamo.optimize(
            backend=Backend(), dynamic=False, isolate_recompiles=True
        )
        ctx._isolate_recompiles_id = region

        seen = []

        def clear():
            for _ in range(3):
                _clear_cache_entries_for_region(code, region)
            # Recorded, not asserted: an exception raised inside a backend
            # __eq__ is swallowed by the lookup as a mismatch.
            seen.append(len(_get_cache_entries_for_region(code, region)))

        hook.append(clear)
        self.assertEqual(ctx(f)(x), f(x))
        # Still walked by the interrupted lookup, so nothing was gone yet.
        self.assertEqual(seen, [1])
        # The next depth-zero lookup applies the parked clear before scanning
        # candidates, so the bucket is already empty: it misses with no guard to
        # evaluate and recompiles into the region.
        self.assertEqual(len(compiles), 2)
        self.assertEqual(len(_get_cache_entries_for_region(code, region)), 1)

    def test_precompile_reset_from_inside_a_lookup_is_parked(self):
        # Same hazard for _reset_precompile_entries: run by a backend __eq__
        # inside lookup() it now parks, and the next depth-zero holder (the
        # precompile-entry reader here) applies it.
        from torch._C._dynamo.eval_frame import (
            _debug_get_precompile_entries,
            _reset_precompile_entries,
        )

        def f(x):
            return x.sin() + x.cos()

        code = f.__code__
        # Kept alive: a dead package's finalizer would uninstall the entries.
        package, _ = self._install_eager_package(f, 7)
        self.assertEqual(len(_debug_get_precompile_entries(code)), 1)
        hook = []

        class Backend:
            def __call__(self, gm, example_inputs):
                return gm.forward

            def __hash__(self):
                return 0

            def __eq__(self, other):
                if isinstance(other, Backend):
                    if hook:
                        hook.pop()()
                    return True
                return NotImplemented

        x = torch.randn(8)
        torch._dynamo.optimize(backend=Backend(), dynamic=False)(f)(x)

        seen = []

        def reset():
            _reset_precompile_entries(code)
            # Recorded, not asserted; see test_region_clear_from_inside_a_lookup.
            seen.append(len(_debug_get_precompile_entries(code)))

        hook.append(reset)
        self.assertEqual(
            torch._dynamo.optimize(backend=Backend(), dynamic=False)(f)(x), f(x)
        )
        self.assertEqual(len(_debug_get_precompile_entries(code)), 0)
        # The parked reset left the entry in place until the lookup finished.
        self.assertEqual(seen, [1])
        package.uninstall()

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_basic(self):
        """Each isolated region has its own recompile limit: region A hitting
        its limit (recompile_limit=1) does not consume region B's budget."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        self.assertNotEqual(opt_a._isolate_recompiles_id, opt_b._isolate_recompiles_id)

        opt_a(torch.randn(3))
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_a(torch.randn(4))  # region A exhausts its own limit

        # Region B is unaffected: its own budget still allows one compile.
        opt_b(torch.randn(5))
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_a._isolate_recompiles_id)), 1
        )
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_b._isolate_recompiles_id)), 1
        )
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 0)

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_same_function_different_regions(self):
        """Two compile calls on the same function get independent caches.
        Each can compile once without the other's entry causing a limit hit."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))

        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_factory_pattern(self):
        """Factory creates multiple torch.compile wrappers around the same
        inner function. Each gets its own isolated cache bucket."""

        def core(x):
            return x.sum()

        @cache
        def factory(key):
            @torch.compile(fullgraph=True, dynamic=False, isolate_recompiles=True)  # noqa: UNSPECIFIED_BACKEND
            def frontend(x, n):
                return core(x) + n

            return frontend

        factory("foo")(torch.ones(3), 3)
        factory("bar")(torch.ones(4), 3)
        factory("baz")(torch.ones(5), 3)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_same_backend_different_regions(self):
        """Two isolated regions sharing the SAME CompileCounter backend.
        Without per-region bucketing, the second region would get a cache
        hit from the first (same backend, same guards). Verifies the
        per-region cache map routes entries to the correct bucket, that each
        region recompiles independently for a new shape, and that both produce
        correct outputs."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)

        x3 = torch.randn(3)
        x4 = torch.randn(4)

        self.assertEqual(opt_a(x3), f(x3))
        self.assertEqual(cnt.frame_count, 1)

        # Must compile again — different region, even though same backend + input
        self.assertEqual(opt_b(x3), f(x3))
        self.assertEqual(cnt.frame_count, 2)

        # Cache hits within each region for the same shape
        opt_a(x3)
        opt_b(x3)
        self.assertEqual(cnt.frame_count, 2)

        # A new shape recompiles per-region, independently in each bucket
        self.assertEqual(opt_a(x4), f(x4))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(opt_b(x4), f(x4))
        self.assertEqual(cnt.frame_count, 4)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_isolate_recompiles_string_backends(self, backend):
        """Two isolated regions with the same string backend compile
        independently — verified by total cache entry count."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=backend, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=backend, isolate_recompiles=True)

        opt_a(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 1)

        opt_b(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 2)

        opt_a(torch.randn(3))
        opt_b(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 2)

    # ===== Static vs dynamic: independent compilation strategies =====

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_static_and_dynamic(self):
        """Two regions on the same function: one static, one dynamic.
        isolate_recompiles keeps their cache entries separate so static
        recompiles don't count against the dynamic region."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_static = torch.compile(
            f, backend=cnt, dynamic=False, isolate_recompiles=True
        )
        opt_dynamic = torch.compile(
            f, backend=cnt, dynamic=True, isolate_recompiles=True
        )

        opt_static(torch.randn(4, 8))
        self.assertEqual(cnt.frame_count, 1)

        opt_dynamic(torch.randn(5, 9))
        self.assertEqual(cnt.frame_count, 2)

        # Static cache hit
        opt_static(torch.randn(4, 8))
        self.assertEqual(cnt.frame_count, 2)

        # Dynamic cache hit with different shape
        opt_dynamic(torch.randn(6, 10))
        self.assertEqual(cnt.frame_count, 2)

        # Static recompile with new shape
        opt_static(torch.randn(5, 9))
        self.assertEqual(cnt.frame_count, 3)

        # Entries are bucketed per region, not pooled: the static region holds
        # its 2 shapes, the dynamic region its 1, and nothing leaked to the
        # default (-1) bucket.
        self.assertNotEqual(
            opt_static._isolate_recompiles_id, opt_dynamic._isolate_recompiles_id
        )
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_static._isolate_recompiles_id)), 2
        )
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_dynamic._isolate_recompiles_id)),
            1,
        )
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 0)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_mark_dynamic_vs_static(self):
        """Two regions: one with mark_static, one with mark_dynamic.
        Their guards don't interfere across regions."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_static = torch.compile(f, backend=cnt_static, isolate_recompiles=True)
        opt_dynamic = torch.compile(f, backend=cnt_dynamic, isolate_recompiles=True)

        x_static = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static, 0)
        opt_static(x_static)
        self.assertEqual(cnt_static.frame_count, 1)

        x_dynamic = torch.randn(4, 8)
        torch._dynamo.mark_dynamic(x_dynamic, 0)
        opt_dynamic(x_dynamic)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static cache hit — same shape
        x_static2 = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static2, 0)
        opt_static(x_static2)
        self.assertEqual(cnt_static.frame_count, 1)

        # Dynamic cache hit — different shape, same dynamic dim
        x_dynamic2 = torch.randn(7, 8)
        opt_dynamic(x_dynamic2)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static recompile — different shape
        x_static3 = torch.randn(7, 8)
        torch._dynamo.mark_static(x_static3, 0)
        opt_static(x_static3)
        self.assertEqual(cnt_static.frame_count, 2)

        # Per-region buckets: static holds its 2 shapes, dynamic its 1, and the
        # default (-1) bucket stays empty.
        self.assertNotEqual(
            opt_static._isolate_recompiles_id, opt_dynamic._isolate_recompiles_id
        )
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_static._isolate_recompiles_id)), 2
        )
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_dynamic._isolate_recompiles_id)),
            1,
        )
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 0)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_isolate_recompiles_auto_dynamic_shared_pgo(self):
        """PGO (frame_state) is shared across isolated regions. Region B
        benefits from region A's shape observations — compiles with dynamic
        shapes immediately without redundant static-then-dynamic recompilation."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt_a, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, isolate_recompiles=True)

        opt_a(torch.randn(3, 4))
        opt_a(torch.randn(5, 4))
        self.assertEqual(cnt_a.frame_count, 2)

        # Region B benefits from A's PGO — compiles dynamic immediately
        opt_b(torch.randn(7, 4))
        self.assertEqual(cnt_b.frame_count, 1)

        opt_b(torch.randn(9, 4))
        self.assertEqual(cnt_b.frame_count, 1)

        # PGO is shared, but cache entries are bucketed per region: region B's
        # single entry lives in its own bucket, not the default and not A's.
        self.assertNotEqual(opt_a._isolate_recompiles_id, opt_b._isolate_recompiles_id)
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_b._isolate_recompiles_id)), 1
        )
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 0)

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_same_backend_different_dynamic_independent_limits(self):
        """Two regions with the same backend, one static and one dynamic.
        Each exhausts its recompile_limit independently."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_static = torch.compile(
            f, backend=cnt, dynamic=False, isolate_recompiles=True
        )
        opt_dynamic = torch.compile(
            f, backend=cnt, dynamic=True, isolate_recompiles=True
        )

        # Static region: two shapes fill recompile_limit=2
        opt_static(torch.randn(3))
        opt_static(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        # Dynamic region: compiles once, different shapes are cache hits
        opt_dynamic(torch.randn(5))
        self.assertEqual(cnt.frame_count, 3)
        opt_dynamic(torch.randn(6))
        self.assertEqual(cnt.frame_count, 3)

        # Static region hits its limit
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_static(torch.randn(5))

        # Dynamic region still works — independent limit
        opt_dynamic(torch.randn(7))
        self.assertEqual(cnt.frame_count, 3)

    # ===== Recompile limits: per-region and accumulated =====

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_isolate_recompiles_fullgraph_raises(self):
        """With fullgraph=True, hitting a region's recompile limit raises
        FailOnRecompileLimitHit regardless of fail_on_recompile_limit_hit. The
        limit is per-region: region A raising does not stop region B from
        compiling its first graph."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", fullgraph=True, dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", fullgraph=True, dynamic=False, isolate_recompiles=True
        )
        self.assertNotEqual(opt_a._isolate_recompiles_id, opt_b._isolate_recompiles_id)

        opt_a(torch.randn(3))
        with self.assertRaisesRegex(FailOnRecompileLimitHit, "fullgraph=True"):
            opt_a(torch.randn(4))

        # Region B has its own budget: its first compile must succeed, not raise.
        opt_b(torch.randn(5))
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_b._isolate_recompiles_id)), 1
        )
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 0)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_graph_break_independent_regions(self):
        """fullgraph=False with isolate_recompiles=True and a graph break:
        each region keeps independent buckets for both the main frame and
        the resume frame."""
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        def f(x):
            a = x.sin()
            torch._dynamo.graph_break()
            return a + 1

        opt_a = torch.compile(
            f, backend=backend, dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend=backend, dynamic=False, isolate_recompiles=True
        )

        x3 = torch.randn(3)
        x4 = torch.randn(4)
        expected3 = x3.sin() + 1
        expected4 = x4.sin() + 1

        self.assertEqual(opt_a(x3), expected3)
        self.assertEqual(opt_b(x3), expected3)
        self.assertEqual(opt_a(x4), expected4)
        self.assertEqual(opt_b(x4), expected4)

        # 8 graphs: 2 regions x 2 shapes x 2 frames (main + resume). Static
        # shapes (automatic_dynamic_shapes=False) make each new shape recompile.
        # Main frames trace `x.sin()`, resume frames trace `a + 1`.
        self.assertEqual(len(backend.graphs), 8)
        self.assertEqual(_count_sin_graphs(backend.graphs), 4)
        self.assertEqual(_count_add_graphs(backend.graphs), 4)

    @torch._dynamo.config.patch(recompile_limit=2, automatic_dynamic_shapes=False)
    def test_isolate_recompiles_graph_break_per_region_limit(self):
        """Graph-break function with two regions: each region independently
        tracks and limits both its main frame (sin) and its resume frame (add).
        Region a exhausts its per-region recompile_limit; region b continues
        compiling unaffected. Static shapes (automatic_dynamic_shapes=False)
        make each new shape recompile until the limit is hit."""
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        def f(x):
            a = x.sin()
            torch._dynamo.graph_break()
            return a + 1

        opt_a = torch.compile(
            f, backend=backend, dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend=backend, dynamic=False, isolate_recompiles=True
        )

        # Fill region a up to its limit with 2 distinct shapes. Both frames
        # reach the limit: 2 main (sin) + 2 resume (add).
        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        self.assertEqual(len(backend.graphs), 4)
        self.assertEqual(_count_sin_graphs(backend.graphs), 2)
        self.assertEqual(_count_add_graphs(backend.graphs), 2)

        # Third shape in region a hits the per-region limit -> the main frame
        # goes RUN_ONLY and runs eagerly. The post-break code still executes
        # eagerly, but no compiled resume function is invoked, so neither frame
        # produces a new graph.
        opt_a(torch.randn(5))
        self.assertEqual(len(backend.graphs), 4)
        self.assertEqual(_count_sin_graphs(backend.graphs), 2)
        self.assertEqual(_count_add_graphs(backend.graphs), 2)

        # Region b is independent: both its main and resume frames compile.
        opt_b(torch.randn(3))
        self.assertEqual(len(backend.graphs), 6)
        self.assertEqual(_count_sin_graphs(backend.graphs), 3)
        self.assertEqual(_count_add_graphs(backend.graphs), 3)

    @torch._dynamo.config.patch(
        accumulated_recompile_limit=6,
        recompile_limit=4,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_accumulated_limit(self):
        """accumulated_recompile_limit is a global safety cap across all
        regions on the same code object. Three regions collectively contribute
        6 entries (2 each), hitting the global cap even though each region
        is below its per-region recompile_limit of 4. New shapes fall back
        to eager via RUN_ONLY."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_c = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )

        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id
        id_c = opt_c._isolate_recompiles_id

        # Region A: 2 compilations (total 2)
        opt_a(torch.randn(1))
        opt_a(torch.randn(2))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 0)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_c)), 0)
        self.assertEqual(_get_total_cache_entry_count(f), 2)

        # Region B: 2 compilations (total 4)
        opt_b(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_c)), 0)
        self.assertEqual(_get_total_cache_entry_count(f), 4)

        # Region C: 2 compilations (total 6 = accumulated_recompile_limit)
        opt_c(torch.randn(5))
        opt_c(torch.randn(6))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_c)), 2)
        self.assertEqual(_get_total_cache_entry_count(f), 6)

        # All three regions blocked — new shapes fall back to eager
        x7 = torch.randn(7)
        self.assertEqual(opt_a(x7), f(x7))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)

        x8 = torch.randn(8)
        self.assertEqual(opt_b(x8), f(x8))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)

        x9 = torch.randn(9)
        self.assertEqual(opt_c(x9), f(x9))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_c)), 2)
        self.assertEqual(_get_total_cache_entry_count(f), 6)

        # Existing cached shapes still hit cache
        x1 = torch.randn(1)
        self.assertEqual(opt_a(x1), f(x1))

    @torch._dynamo.config.patch(
        accumulated_recompile_limit=4,
        recompile_limit=8,
        automatic_dynamic_shapes=False,
        fail_on_recompile_limit_hit=True,
    )
    def test_isolate_recompiles_accumulated_limit_hard_fail(self):
        """With fail_on_recompile_limit_hit=True, exceeding accumulated_recompile_limit
        across isolated regions raises FailOnRecompileLimitHit."""

        def f(x):
            return x.cos()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )

        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id

        # 2 entries each = 4 total = accumulated_recompile_limit
        opt_a(torch.randn(1))
        opt_a(torch.randn(2))
        opt_b(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_a(torch.randn(5))
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_b(torch.randn(6))

    # ===== RUN_ONLY strategy: per-region persistence after limit hit =====

    @torch._dynamo.config.patch(
        recompile_limit=1,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_limit_does_not_skip_other_regions(self):
        """When one region hits its recompile limit and goes RUN_ONLY,
        other regions (both isolated and non-isolated) can still compile."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_default = torch.compile(f, backend="eager", dynamic=False)

        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id
        # The two regions and the default bucket are genuinely distinct.
        self.assertNotEqual(id_a, id_b)
        self.assertNotEqual(id_a, -1)

        # Region A compiles once, then hits limit
        opt_a(torch.randn(3))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)
        opt_a(torch.randn(4))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)

        # Region B still compiles
        opt_b(torch.randn(5))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 1)

        # Default (non-isolated) region still compiles
        opt_default(torch.randn(6))
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 1)

    @torch._dynamo.config.patch(
        recompile_limit=1,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_region_run_only_persists(self):
        """After hitting the recompile limit, RUN_ONLY is persisted per-region
        in ExtraState.region_strategy_map. Subsequent calls skip the callback
        entirely (no repeated limit-hit warnings). Cached shapes still work."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id

        # Region A compiles shape 3
        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)

        # Region A hits limit — RUN_ONLY set for this region
        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)

        # RUN_ONLY persists — callback not re-entered on new shapes
        opt_a(torch.randn(5))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)

        # Cached shape 3 still produces correct result
        x3 = torch.randn(3)
        self.assertEqual(opt_a(x3), f(x3))
        self.assertEqual(cnt.frame_count, 1)

        # Region B is unaffected
        opt_b(torch.randn(6))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 1)

    # ===== Default strategy × region: SKIP inherited, RUN_ONLY not =====

    def test_isolate_recompiles_inherits_default_skip(self):
        """Global SKIP (from skip_code / @torch._dynamo.skip / FX plumbing /
        TorchScript __init__ / etc.) is a correctness decision — the code
        must not be traced. Isolated regions inherit this SKIP, so neither
        the default nor isolated wrapper compiles a skip_code-marked code
        object. Only the automatic RUN_ONLY (from a prior non-isolated
        recompile-limit hit) is prevented from bleeding into regions."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        torch._dynamo.eval_frame.skip_code(f.__code__)

        opt_default = torch.compile(f, backend=cnt)
        opt_default(torch.randn(3))
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 0)

        opt_iso = torch.compile(f, backend=cnt, isolate_recompiles=True)
        id_iso = opt_iso._isolate_recompiles_id
        # A genuine isolated region (distinct from the default bucket) that
        # nonetheless inherits the global SKIP.
        self.assertNotEqual(id_iso, -1)

        x = torch.randn(3)
        self.assertEqual(opt_iso(x), f(x))
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_iso)), 0)

    def test_isolate_recompiles_ignores_default_run_only(self):
        """Regression for the RUN_ONLY-bleed case: a prior non-isolated
        recompile-limit hit sets RUN_ONLY on extra->strategy. A later
        isolated region on the same code object must not inherit that
        RUN_ONLY and must compile normally."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        # Trip the non-isolated recompile limit so RUN_ONLY is persisted
        # to extra->strategy (non-isolated bucket).
        with torch._dynamo.config.patch(
            recompile_limit=1, automatic_dynamic_shapes=False
        ):
            opt_default = torch.compile(f, backend=cnt, dynamic=False)
            opt_default(torch.randn(3))
            opt_default(torch.randn(4))  # hits limit → RUN_ONLY persisted
            opt_default(torch.randn(5))  # RUN_ONLY path, no compile
        self.assertEqual(cnt.frame_count, 1)

        # Isolated region must ignore the persisted RUN_ONLY.
        opt_iso = torch.compile(f, backend=cnt, isolate_recompiles=True)
        id_iso = opt_iso._isolate_recompiles_id
        opt_iso(torch.randn(6))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_iso)), 1)

    # ===== Cache internals: insertion order, fallback, shared bucket =====

    @torch._dynamo.config.patch(recompile_limit=2, automatic_dynamic_shapes=False)
    def test_isolate_recompiles_insertion_order_per_region(self):
        """New entries are added at the front of their region's list.
        Interleaved compilations across regions don't mix ordering.
        After hitting recompile_limit, entries are frozen."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id
        self.assertNotEqual(id_a, id_b)

        # Interleave compilations
        opt_a(torch.randn(3))
        opt_b(torch.randn(10))
        opt_a(torch.randn(4))
        opt_b(torch.randn(11))

        # 4 entries total across the two regions (2 each), not pooled into one.
        self.assertEqual(_get_total_cache_entry_count(f), 4)

        # Newest at front in each region
        entries_a = _get_cache_entries_for_region(f, id_a)
        self.assertEqual(len(entries_a), 2)
        self.assertGreater(
            entries_a[0].compile_id.frame_compile_id,
            entries_a[1].compile_id.frame_compile_id,
        )

        entries_b = _get_cache_entries_for_region(f, id_b)
        self.assertEqual(len(entries_b), 2)
        self.assertGreater(
            entries_b[0].compile_id.frame_compile_id,
            entries_b[1].compile_id.frame_compile_id,
        )

        # Both at limit — no new entries
        opt_a(torch.randn(5))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        opt_b(torch.randn(12))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_lru_move_to_front(self):
        """On a cache hit, the matched entry moves to the front of its
        region's list (LRU). Verify by inspecting compile_id ordering
        before and after the hit. Also verify cross-region independence."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id

        # Region A: compile shapes 3, 4, 5.
        # Insertion order (newest at front): [5, 4, 3]
        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        opt_a(torch.randn(5))

        entries_a = _get_cache_entries_for_region(f, id_a)
        self.assertEqual(len(entries_a), 3)
        ids_before = [e.compile_id for e in entries_a]

        # Region B: compile shapes 6, 7.
        opt_b(torch.randn(6))
        opt_b(torch.randn(7))
        entries_b = _get_cache_entries_for_region(f, id_b)

        # Hit region A with shape 3 (oldest entry, at back) — LRU moves to front
        opt_a(torch.randn(3))
        entries_a_after = _get_cache_entries_for_region(f, id_a)
        ids_after = [e.compile_id for e in entries_a_after]

        # shape-3 entry was last, now first
        self.assertEqual(ids_after[0], ids_before[-1])
        self.assertEqual(ids_after[1], ids_before[0])
        self.assertEqual(ids_after[2], ids_before[1])

        # No new entries — cache hit, not recompilation
        self.assertEqual(len(entries_a_after), 3)

        # Region B order unchanged — LRU on A doesn't affect B
        entries_b_after = _get_cache_entries_for_region(f, id_b)
        self.assertEqual(
            [e.compile_id for e in entries_b_after],
            [e.compile_id for e in entries_b],
        )

    def test_non_isolated_entries_visible_to_isolated(self):
        """Non-isolated entries (bucket -1) are visible to isolated regions
        via read-only fallback when the backend matches. BC friendly —
        isolated compiles reuse existing non-isolated compilations."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.exp()

        opt_global = torch.compile(f, backend=cnt)
        opt_global(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        opt_isolated = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_isolated(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # The reuse is genuine cross-bucket fallback: opt_isolated is a real
        # region (not the default), it added no entry of its own, and the reused
        # entry lives in the default (-1) bucket.
        self.assertNotEqual(opt_isolated._isolate_recompiles_id, -1)
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_isolated._isolate_recompiles_id)),
            0,
        )
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 1)

    @torch._dynamo.config.patch(recompile_limit=8)
    def test_isolate_recompiles_reasons_include_default_bucket(self):
        """Recompile-reason logging for an isolated region must also walk
        the default (-1) bucket. lookup() checks default entries as a
        fallback for isolated regions, so their guard failures are real
        and must not be dropped from the recompile-reason log."""
        default_fails: list[str] = []
        region_fails: list[str] = []

        def record_default(failure):
            default_fails.append(failure.reason)

        def record_region(failure):
            region_fails.append(failure.reason)

        def f(x):
            return x.sum()

        opt_default = torch._dynamo.optimize(
            "eager", guard_fail_fn=record_default, dynamic=False
        )(f)
        opt_isolated = torch._dynamo.optimize(
            "eager",
            guard_fail_fn=record_region,
            dynamic=False,
            isolate_recompiles=True,
        )(f)

        # Populate default bucket with a shape-3 entry.
        opt_default(torch.randn(3))
        # Populate region bucket with a shape-4 entry.
        opt_isolated(torch.randn(4))
        # Recompile in the region: shape-5 misses both buckets.
        # The logging path must report guard failures for BOTH the
        # region's shape-4 entry and the default bucket's shape-3 entry.
        opt_isolated(torch.randn(5))

        self.assertTrue(
            region_fails,
            lambda msg: f"{msg}\nregion entries' guard failures missing: {region_fails}",
        )
        self.assertTrue(
            default_fails,
            lambda msg: f"{msg}\ndefault-bucket entries' guard failures dropped from "
            f"recompile reasons (bug): {default_fails}",
        )
        # Region and default are separate buckets: only shape-3 is in the
        # default bucket; the region holds its own shape-4/5 entries.
        self.assertNotEqual(opt_isolated._isolate_recompiles_id, -1)
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 1)
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_isolated._isolate_recompiles_id)),
            2,
        )

    @torch._dynamo.config.patch(recompile_limit=8)
    def test_non_isolated_reasons_unchanged(self):
        """Regression: recompile-reason logging for non-isolated compiles
        (id=-1) must still work. The split between cache_entries and
        cache_entries_for_reasons should not have changed this path."""
        fails: list[str] = []

        def f(x):
            return x.sum()

        opt = torch._dynamo.optimize(
            "eager",
            guard_fail_fn=lambda failure: fails.append(failure.reason),
            dynamic=False,
        )(f)

        opt(torch.randn(3))
        opt(torch.randn(4))
        self.assertTrue(
            fails, lambda msg: f"{msg}\nno recompile reasons logged: {fails}"
        )

    @torch._dynamo.config.patch(recompile_limit=8)
    def test_isolate_recompiles_reasons_include_all_default_entries(self):
        """When the default bucket has multiple entries, recompile-reason
        logging from an isolated region must report guard failures from
        each of them, not just one."""
        default_fails: list[str] = []

        def f(x):
            return x.sum()

        opt_default = torch._dynamo.optimize(
            "eager",
            guard_fail_fn=lambda failure: default_fails.append(failure.reason),
            dynamic=False,
        )(f)
        opt_default(torch.randn(3))
        opt_default(torch.randn(4))

        # Prime the region with one entry — recompile-reason logging only
        # fires on subsequent calls (is_recompilation requires the region
        # to already have ≥1 entry).
        opt_iso = torch._dynamo.optimize(
            "eager", dynamic=False, isolate_recompiles=True
        )(f)
        opt_iso(torch.randn(5))
        default_fails.clear()

        # Recompile in the region — shape 6 misses the region entry and
        # both default entries. All three should contribute guard-failure
        # reasons; default_fails must receive both default entries' fails.
        opt_iso(torch.randn(6))

        self.assertGreaterEqual(
            len(default_fails),
            2,
            lambda msg: f"{msg}\nexpected guard failures for both default entries, got {default_fails}",
        )
        # The two default entries (shapes 3, 4) stay in the default bucket; the
        # region's shape-5/6 entries live in its own bucket.
        self.assertNotEqual(opt_iso._isolate_recompiles_id, -1)
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 2)

    def test_isolate_recompiles_reset_clears_region_strategy(self):
        """torch._dynamo.reset() must clear region_strategy_map on
        ExtraState. Otherwise a RUN_ONLY persisted by a prior region
        would survive reset and prevent the new region from compiling."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        with torch._dynamo.config.patch(
            recompile_limit=1, automatic_dynamic_shapes=False
        ):
            opt_a = torch.compile(
                f, backend=cnt, dynamic=False, isolate_recompiles=True
            )
            opt_a(torch.randn(3))
            opt_a(torch.randn(4))  # hits limit → region RUN_ONLY persisted
            self.assertEqual(cnt.frame_count, 1)

        torch._dynamo.reset()

        opt_b = torch.compile(f, backend=cnt, isolate_recompiles=True)
        # Both are genuine isolated regions (distinct from the default bucket);
        # the RUN_ONLY persisted for opt_a's region must not survive reset.
        self.assertNotEqual(opt_a._isolate_recompiles_id, -1)
        self.assertNotEqual(opt_b._isolate_recompiles_id, -1)
        opt_b(torch.randn(5))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_non_isolated_compiles_share_cache(self):
        """Without isolate_recompiles, two compile calls share bucket -1.
        They share cache hits AND recompile limits."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.exp()

        opt_a = torch.compile(f, backend=cnt, dynamic=False)
        opt_b = torch.compile(f, backend=cnt, dynamic=False)

        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Cache hit from opt_a's entry
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # New shape from opt_b counts toward shared limit
        opt_b(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        # Shared limit exceeded
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_a(torch.randn(5))

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_different_backends_shared_cache_without_isolate(self):
        """Baseline: without isolate_recompiles, different backends share the
        cache. Entries from backend A count against backend B's limit."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_a = torch.compile(f, backend=cnt_a, dynamic=False)
        opt_b = torch.compile(f, backend=cnt_b, dynamic=False)

        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_b(torch.randn(5))

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_different_backends_independent_with_isolate(self):
        """With isolate_recompiles, different backends get independent buckets."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_a = torch.compile(f, backend=cnt_a, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, dynamic=False, isolate_recompiles=True)

        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)

        # B compiles independently
        opt_b(torch.randn(5))
        self.assertEqual(cnt_b.frame_count, 1)
        opt_b(torch.randn(6))
        self.assertEqual(cnt_b.frame_count, 2)

    # ===== Lifecycle: reset, resume functions, GC =====

    def test_isolate_recompiles_reset(self):
        """torch._dynamo.reset() clears all regions."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.cos()

        opt_a = torch.compile(f, backend=cnt_a, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, isolate_recompiles=True)

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 1)
        self.assertEqual(cnt_b.frame_count, 1)
        # Each region owns one entry; nothing leaked to the default bucket.
        self.assertNotEqual(opt_a._isolate_recompiles_id, opt_b._isolate_recompiles_id)
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_a._isolate_recompiles_id)), 1
        )
        self.assertEqual(
            len(_get_cache_entries_for_region(f, opt_b._isolate_recompiles_id)), 1
        )
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 0)

        torch._dynamo.reset()

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)
        self.assertEqual(cnt_b.frame_count, 2)

    @torch._dynamo.config.patch(recompile_limit=3)
    def test_isolate_recompiles_resume_function(self):
        """Resume functions from a graph break are bucketed by their region's
        isolate_recompiles_id, both for cache lookup and for the per-region
        recompile limit. Region A exhausting its resume-frame limit leaves
        region B free to compile the same main/resume frames independently
        (they share the backend, so without isolation B would cache-hit A)."""
        cnt = torch._dynamo.testing.CompileCounter()

        mode = {"value": "a"}

        def f(x):
            a = x.sin()
            torch._dynamo.graph_break()
            if mode["value"] == "a":
                return a.cos()
            elif mode["value"] == "b":
                return a.tan()
            elif mode["value"] == "c":
                return a.exp()
            else:
                return a + 1

        opt_a = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, isolate_recompiles=True)
        self.assertNotEqual(opt_a._isolate_recompiles_id, opt_b._isolate_recompiles_id)

        # Region A, first call: main frame (sin) + resume frame (cos) = 2.
        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        # Each new mode recompiles only region A's resume frame (main cached).
        for m, expected in (("b", 3), ("c", 4)):
            mode["value"] = m
            opt_a(torch.randn(4))
            self.assertEqual(cnt.frame_count, expected)

        # Region A's resume frame has 3 entries = recompile_limit. A fourth
        # mode is blocked (runs eager), so no new compile.
        mode["value"] = "d"
        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 4)

        # Region B is independent: mode "a" would cache-hit region A's main and
        # resume frames without isolation, but here both recompile (+2).
        mode["value"] = "a"
        opt_b(torch.randn(4))
        self.assertEqual(cnt.frame_count, 6)

    def test_isolate_recompiles_gc_wrapper(self):
        """When an isolated compile wrapper is GC'd, orphaned cache entries
        remain. A new torch.compile gets a fresh region and compiles
        independently. reset() clears everything including orphans."""

        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(self._num_cache_entries(f), 1)

        del opt_a
        gc.collect()

        # Orphaned entry persists
        self.assertEqual(self._num_cache_entries(f), 1)

        # Fresh region compiles independently
        opt_b = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(self._num_cache_entries(f), 2)

        torch._dynamo.reset()
        self.assertEqual(self._num_cache_entries(f), 0)

    # ===== Debug / introspection =====

    def test_cache_key_lookup_is_off_until_a_cache_key_backend_exists(self):
        # get_backend walks the callback chain on every intercepted frame, and
        # looking for _torchdynamo_cache_key there is a MISS at every level for
        # anyone who never precompiles -- a raising attribute lookup per level
        # per frame, measured at ~1 us on a steady-state compiled call. The
        # lookup is therefore gated on a flag that only a backend carrying such
        # a key turns on. Nothing else in the tree sets that attribute, so the
        # gate is invisible; this pins that the switch exists and is one-way.
        # The gate is process-global and one-way, and anything that imports the
        # precompile backend flips it, so the OFF half can only be observed in a
        # fresh interpreter. Two wrappers over ONE shared backend are the
        # discriminator: with the gate off get_backend follows
        # _torchdynamo_orig_backend to that shared object and both compilations
        # share one cache identity; with it on, their distinct cache keys are
        # two identities.
        script = textwrap.dedent(
            """
            import torch
            from torch._C._dynamo.eval_frame import (
                _debug_get_cache_entry_list,
                _enable_precompile_cache_keys,
            )

            def fn(x):
                return x.sin()

            class Backend:
                # The shape _PrecompileBackend has, minus the __init__ that
                # would flip the gate before the OFF half is measured.
                def __init__(self, inner):
                    self._torchdynamo_orig_backend = inner
                    self._torchdynamo_cache_key = object()

                def __call__(self, gm, inputs):
                    return self._torchdynamo_orig_backend(gm, inputs)

            inner = torch._dynamo.lookup_backend("eager")

            def entries(same_key):
                torch._dynamo.reset()
                x = torch.randn(4)
                a, b = Backend(inner), Backend(inner)
                if same_key:
                    b._torchdynamo_cache_key = a._torchdynamo_cache_key
                # optimize(), not compile(backend=), because compile() wraps the
                # backend in a _TorchCompileWrapper that is not in the chain
                # get_backend walks.
                torch._dynamo.optimize(a)(fn)(x)
                torch._dynamo.optimize(b)(fn)(x)
                return len(_debug_get_cache_entry_list(fn.__code__))

            print("off", entries(False))
            _enable_precompile_cache_keys()
            _enable_precompile_cache_keys()  # idempotent
            print("on", entries(False))
            print("on_same_key", entries(True))
            """
        )
        stdout, stderr = self.run_process_no_exception(script)
        out = stdout.decode()
        self.assertIn("off 1\n", out, stderr.decode())
        self.assertIn("on 2\n", out, stderr.decode())
        # The key IS the identity, so two backends sharing one key still share
        # one cache entry -- the gate must not simply split every backend.
        self.assertIn("on_same_key 1\n", out, stderr.decode())

    def test_has_precompile_entries_is_region_exact(self):
        """_has_precompile_entries answers for one region only. lookup() never
        serves a precompile entry from another region, so an entry belonging to
        a second artifact installed on the same code object is not coverage for
        the first. It exists so that a caller can ask that question without
        building the list of wrappers _debug_get_precompile_entries returns."""
        from torch._C._dynamo.eval_frame import (
            _debug_get_cache_entry_list,
            _has_precompile_entries,
            _load_precompile_entry,
            _reset_precompile_entries_for_region,
        )

        def never_compiled(x):
            return x + 1

        self.assertFalse(_has_precompile_entries(never_compiled.__code__, -1))
        with self.assertRaisesRegex(TypeError, "expected a code object"):
            _has_precompile_entries(never_compiled, -1)

        def f(x):
            return x.sin()

        torch.compile(f, backend="eager", dynamic=False)(torch.randn(3))
        code = f.__code__
        self.assertFalse(_has_precompile_entries(code, 7))

        entry = _debug_get_cache_entry_list(code)[0]
        _load_precompile_entry(code, entry.guard_manager, entry.code, 7)
        try:
            self.assertTrue(_has_precompile_entries(code, 7))
            self.assertFalse(_has_precompile_entries(code, 9))
            self.assertFalse(_has_precompile_entries(code, -1))
        finally:
            _reset_precompile_entries_for_region(code, 7)
        self.assertFalse(_has_precompile_entries(code, 7))

    def test_precompile_entries_are_removed_by_owner_not_by_region(self):
        """Several packages may legitimately hold entries for one code object in
        one region -- a library frame two loaded models both reach -- and lookup
        picks between them by evaluating guards. Teardown must therefore remove
        what one installer put there and leave the neighbour's alone; clearing
        the whole region evicts a live artifact that, because lookup is
        region-exact, nothing else can serve."""
        from torch._C._dynamo.eval_frame import (
            _debug_get_cache_entry_list,
            _debug_get_precompile_entries,
            _load_precompile_entry,
            _reset_precompile_entries_for_owner,
            _reset_precompile_entries_for_region,
        )

        def f(x):
            return x.sin()

        torch.compile(f, backend="eager", dynamic=False)(torch.randn(3))
        code = f.__code__
        entry = _debug_get_cache_entry_list(code)[0]

        first, second = object(), object()
        _load_precompile_entry(code, entry.guard_manager, entry.code, -1, first)
        _load_precompile_entry(code, entry.guard_manager, entry.code, -1, second)
        try:
            self.assertEqual(len(_debug_get_precompile_entries(code)), 2)
            _reset_precompile_entries_for_owner(code, -1, first)
            # The neighbour survives.
            self.assertEqual(len(_debug_get_precompile_entries(code)), 1)
            # Removing an owner that holds nothing here is a no-op.
            _reset_precompile_entries_for_owner(code, -1, first)
            self.assertEqual(len(_debug_get_precompile_entries(code)), 1)
            # Same owner, different region: also a no-op.
            _reset_precompile_entries_for_owner(code, 7, second)
            self.assertEqual(len(_debug_get_precompile_entries(code)), 1)
            _reset_precompile_entries_for_owner(code, -1, second)
            self.assertEqual(len(_debug_get_precompile_entries(code)), 0)
        finally:
            _reset_precompile_entries_for_region(code, -1)

    # ===== Exec strategy / region API =====

    def test_exec_strategy_token_and_compare_and_set(self):
        """The token API is the concurrency contract installers rely on: a
        strategy read returns a generation, and a later compare-and-set with
        that generation succeeds only if nothing -- including a reset -- wrote
        the strategy in between."""
        from torch._C._dynamo.eval_frame import (
            compare_and_set_code_exec_strategy,
            get_code_exec_strategy,
            get_code_exec_strategy_token,
            reset_code,
            set_code_exec_strategy_with_token,
        )

        def f(x):
            return x + 1

        # A fresh, unattached code object each run: f is a constant of this
        # method, so f.__code__ is the same object across in-process reruns
        # (--repeat, --flake-runs) and its ExtraState -- and thus its nonzero
        # strategy_generation -- would survive to fail the generation-0 asserts
        # below. replace() mints a distinct object; nothing here runs it.
        code = f.__code__.replace()
        # Never-touched code object: DEFAULT strategy, generation 0, and a
        # compare-and-set against it fails outright (no state to write).
        self.assertEqual(get_code_exec_strategy(code).cur_action, FrameAction.DEFAULT)
        strategy, generation = get_code_exec_strategy_token(code)
        self.assertEqual(strategy.cur_action, FrameAction.DEFAULT)
        self.assertEqual(generation, 0)
        skip = FrameExecStrategy(FrameAction.SKIP, FrameAction.SKIP)
        self.assertFalse(compare_and_set_code_exec_strategy(code, generation, skip))

        prior, generation = set_code_exec_strategy_with_token(
            code, FrameExecStrategy(FrameAction.RUN_ONLY, FrameAction.DEFAULT)
        )
        self.assertEqual(prior.cur_action, FrameAction.DEFAULT)
        self.assertGreater(generation, 0)
        strategy, token = get_code_exec_strategy_token(code)
        self.assertEqual(strategy.cur_action, FrameAction.RUN_ONLY)
        self.assertEqual(token, generation)

        # A compare-and-set with the current generation wins and bumps it...
        self.assertTrue(compare_and_set_code_exec_strategy(code, token, skip))
        strategy, new_token = get_code_exec_strategy_token(code)
        self.assertEqual(strategy.cur_action, FrameAction.SKIP)
        self.assertNotEqual(new_token, token)
        # ...and the stale token now loses, leaving the strategy alone.
        default = FrameExecStrategy(FrameAction.DEFAULT, FrameAction.DEFAULT)
        self.assertFalse(compare_and_set_code_exec_strategy(code, token, default))
        self.assertEqual(get_code_exec_strategy(code).cur_action, FrameAction.SKIP)

        # A reset invalidates every outstanding token.
        reset_code(code)
        self.assertEqual(get_code_exec_strategy(code).cur_action, FrameAction.DEFAULT)
        self.assertFalse(compare_and_set_code_exec_strategy(code, new_token, skip))
        self.assertEqual(get_code_exec_strategy(code).cur_action, FrameAction.DEFAULT)

        # Zero must not be a resurrection token either. A state created by a
        # region write hands out generation 0 before any global write; that
        # token must lose after a global write plus a reset, so the reset may
        # not put the generation back to 0.
        from torch._C._dynamo.eval_frame import set_code_region_exec_strategy

        def g(x):
            return x + 2

        # Fresh object per run, same reason as `code` above.
        code2 = g.__code__.replace()
        set_code_region_exec_strategy(
            code2, 3, FrameExecStrategy(FrameAction.RUN_ONLY, FrameAction.DEFAULT)
        )
        strategy, zero_token = get_code_exec_strategy_token(code2)
        self.assertEqual(zero_token, 0)
        set_code_exec_strategy_with_token(code2, skip)
        reset_code(code2)
        self.assertFalse(compare_and_set_code_exec_strategy(code2, zero_token, skip))
        self.assertEqual(get_code_exec_strategy(code2).cur_action, FrameAction.DEFAULT)

    def test_region_exec_strategy_inherits_skip_but_not_run_only(self):
        from torch._C._dynamo.eval_frame import (
            get_code_exec_strategy,
            get_code_region_exec_strategy,
            reset_code,
            set_code_region_exec_strategy,
        )

        def f(x):
            return x + 1

        code = f.__code__
        try:
            # A region write is region-local: neither its siblings nor the
            # global strategy see it.
            set_code_region_exec_strategy(
                code, 7, FrameExecStrategy(FrameAction.RUN_ONLY, FrameAction.DEFAULT)
            )
            region7 = get_code_region_exec_strategy(code, 7)
            self.assertEqual(region7.cur_action, FrameAction.RUN_ONLY)
            region9 = get_code_region_exec_strategy(code, 9)
            self.assertEqual(region9.cur_action, FrameAction.DEFAULT)
            self.assertEqual(
                get_code_exec_strategy(code).cur_action, FrameAction.DEFAULT
            )
            # A global RUN_ONLY (a recompile-limit hit) must not poison fresh
            # regions...
            set_code_region_exec_strategy(
                code, -1, FrameExecStrategy(FrameAction.RUN_ONLY, FrameAction.RUN_ONLY)
            )
            region9 = get_code_region_exec_strategy(code, 9)
            self.assertEqual(region9.cur_action, FrameAction.DEFAULT)
            self.assertEqual(region9.recursive_action, FrameAction.DEFAULT)
            # ...but a global SKIP (a deliberate do-not-trace mark) applies
            # everywhere, except where a region's own strategy wins.
            set_code_region_exec_strategy(
                code, -1, FrameExecStrategy(FrameAction.SKIP, FrameAction.SKIP)
            )
            region9 = get_code_region_exec_strategy(code, 9)
            self.assertEqual(region9.cur_action, FrameAction.SKIP)
            self.assertEqual(region9.recursive_action, FrameAction.SKIP)
            region7 = get_code_region_exec_strategy(code, 7)
            self.assertEqual(region7.cur_action, FrameAction.RUN_ONLY)
        finally:
            reset_code(code)

    def test_clear_cache_entries_for_region_is_region_exact(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_a(torch.randn(3))
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)
        code = f.__code__
        region_a = opt_a._isolate_recompiles_id
        region_b = opt_b._isolate_recompiles_id

        with self.assertRaisesRegex(TypeError, "expected a code object"):
            _clear_cache_entries_for_region(f, region_a)
        with self.assertRaisesRegex(ValueError, "default cache region"):
            _clear_cache_entries_for_region(code, -1)

        _clear_cache_entries_for_region(code, region_a)
        self.assertEqual(len(_get_cache_entries_for_region(code, region_a)), 0)
        # The neighbour region is untouched and still serves its entry.
        self.assertEqual(len(_get_cache_entries_for_region(code, region_b)), 1)
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)
        # Clearing an already-empty region is a no-op.
        _clear_cache_entries_for_region(code, region_a)

    def test_force_callback_on_cache_miss_marker_overrides_run_only(self):
        """Contract of the `_torchdynamo_force_callback_on_cache_miss` marker
        (read by eval_frame_cpp.cpp): a RUN_ONLY frame whose installed callback
        carries it still reaches the callback on a cache miss. A precompile
        serving callback is the intended setter, so a miss errors or recaptures
        instead of silently running eager."""
        from torch._C._dynamo.eval_frame import reset_code
        from torch._dynamo.eval_frame import set_code_exec_strategy

        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        ctx = torch._dynamo.optimize(cnt, dynamic=False)
        opt = ctx(f)
        opt(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)
        set_code_exec_strategy(
            f.__code__, FrameExecStrategy(FrameAction.RUN_ONLY, FrameAction.RUN_ONLY)
        )
        # A miss without the marker runs eager: no new compile.
        opt(torch.randn(4, 4))
        self.assertEqual(cnt.frame_count, 1)
        ctx.callback._torchdynamo_force_callback_on_cache_miss = True
        try:
            opt(torch.randn(5, 5))
            self.assertEqual(cnt.frame_count, 2)
        finally:
            del ctx.callback._torchdynamo_force_callback_on_cache_miss
            # reset_code drops the RUN_ONLY strategy this test set on f.__code__
            # so it does not leak into the next test.
            reset_code(f.__code__)

    def test_isolate_recompiles_debug_cache_entry_list_deterministic_order(self):
        """_debug_get_cache_entry_list returns entries sorted by
        isolate_recompiles_id for deterministic output."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opts = [
            torch.compile(f, backend=cnt, isolate_recompiles=True) for _ in range(6)
        ]
        for opt in reversed(opts):
            opt(torch.randn(3))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(f)
        self.assertEqual(len(entries), 6)

        ids = [e.isolate_recompiles_id for e in entries]
        self.assertEqual(ids, sorted(ids))


instantiate_parametrized_tests(IsolateRecompilesTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
