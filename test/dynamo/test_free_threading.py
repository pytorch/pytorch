# Owner(s): ["module: dynamo"]

"""Concurrent use of torch.compile from several threads.

Each test stresses one piece of state that Dynamo shares between threads.  They
are aimed at free-threaded builds, but they are valid on the default build too,
where guard evaluation still drops the GIL in the middle of a cache lookup.
"""

import contextvars
import gc
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._inductor.config
from torch._C._dynamo.eval_frame import (
    _debug_get_cache_entry_list,
    get_eval_frame_isolate_recompiles_id,
    set_eval_frame,
    set_eval_frame_isolate_recompiles_id,
    set_skip_guard_eval_unsafe,
)


NTHREADS = 8
ITERS = 100
# Under the runner's 10 minute timeout, so a deadlock fails with stacks.
TIMEOUT = 300


def run_threads(body: Callable[[int], None], nthreads: int = NTHREADS) -> None:
    """Runs ``body(tid)`` on ``nthreads`` threads at once and re-raises the
    first failure, or fails with every thread's stack if they are still running
    after ``TIMEOUT`` seconds.

    Each thread runs in a copy of the caller's context, so the config a test
    patches reaches its threads on every build, not only on free-threaded ones,
    where new threads inherit the context.
    """
    barrier = threading.Barrier(nthreads)
    contexts = [contextvars.copy_context() for _ in range(nthreads)]
    failures: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            contexts[tid].run(body, tid)
        except BaseException as e:
            failures.append(e)

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True) for i in range(nthreads)
    ]
    for t in threads:
        t.start()
    deadline = time.monotonic() + TIMEOUT
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in threads):
        stacks = "\n".join(
            f"Thread {ident}:\n{''.join(traceback.format_stack(frame))}"
            for ident, frame in sys._current_frames().items()
        )
        raise AssertionError(f"threads still running after {TIMEOUT}s\n{stacks}")
    if failures:
        raise failures[0]


def hot_fn(x: torch.Tensor) -> torch.Tensor:
    return (x * 2 + 1).relu().sum()


def add_attr(x: torch.Tensor, obj: Any) -> torch.Tensor:
    return x + obj.v


def compile_hot_fn(backend: Any = "eager") -> Callable[[torch.Tensor], torch.Tensor]:
    return torch.compile(hot_fn, backend=backend, dynamic=False)


# In fbcode, even checking for a remote cache loads its client, and with it
# boost.python, whose destructors crash a free-threaded interpreter at exit.
@torch._inductor.config.patch(fx_graph_remote_cache=False)
@torch._dynamo.config.patch(automatic_dynamic_remote_pgo=False)
class FreeThreadingTests(torch._dynamo.test_case.TestCase):
    def assert_no_recompiles(self, cnt: torch._dynamo.testing.CompileCounter) -> None:
        self.assertEqual(cnt.frame_count, 0, "a lookup missed an entry that was cached")

    def test_isolate_recompiles_id_is_per_thread(self) -> None:
        entered = threading.Event()
        checked = threading.Event()

        def other() -> None:
            entered.wait(timeout=30)
            set_eval_frame_isolate_recompiles_id(222)
            checked.set()

        prior = get_eval_frame_isolate_recompiles_id()
        t = threading.Thread(target=other)
        t.start()
        try:
            set_eval_frame_isolate_recompiles_id(111)
            entered.set()
            self.assertTrue(checked.wait(timeout=30))
            self.assertEqual(get_eval_frame_isolate_recompiles_id(), 111)
        finally:
            t.join()
            set_eval_frame_isolate_recompiles_id(prior)

    def test_skip_guard_eval_unsafe_is_per_thread(self) -> None:
        entered = threading.Event()
        checked = threading.Event()

        def other() -> None:
            entered.wait(timeout=30)
            set_skip_guard_eval_unsafe(True)
            checked.set()

        t = threading.Thread(target=other)
        t.start()
        try:
            set_skip_guard_eval_unsafe(False)
            entered.set()
            self.assertTrue(checked.wait(timeout=30))
            # Returns the previous value, which is this thread's view.
            self.assertFalse(set_skip_guard_eval_unsafe(False))
        finally:
            t.join()

    def test_shared_cache_entry(self) -> None:
        cnt = torch._dynamo.testing.CompileCounter()
        cf = compile_hot_fn(cnt)
        x = torch.randn(32, 32)
        expected = cf(x)
        cnt.clear()

        def body(tid: int) -> None:
            for _ in range(ITERS):
                self.assertEqual(cf(x), expected)

        run_threads(body)
        self.assert_no_recompiles(cnt)

    @torch._dynamo.testing.lru_cache_reordering(True)
    @torch._dynamo.config.patch(recompile_limit=64)
    def test_lru_reordering(self) -> None:
        cnt = torch._dynamo.testing.CompileCounter()
        cf = compile_hot_fn(cnt)
        xs = [torch.randn(n, n) for n in (8, 16, 32, 64, 128)]
        for x in xs:
            cf(x)
        cnt.clear()

        def body(tid: int) -> None:
            x = xs[tid % len(xs)]
            for _ in range(ITERS):
                cf(x)

        run_threads(body)
        self.assert_no_recompiles(cnt)

    @torch._dynamo.testing.lru_cache_reordering(True)
    @torch._dynamo.config.patch(recompile_limit=64)
    def test_contended_lookup(self) -> None:
        cnt = torch._dynamo.testing.CompileCounter()
        cf = compile_hot_fn(cnt)
        xs = [torch.randn(n) for n in range(1, 9)]
        for x in xs:
            cf(x)
        cnt.clear()

        def body(tid: int) -> None:
            for i in range(ITERS):
                cf(xs[(i + tid) % len(xs)])

        run_threads(body)
        self.assert_no_recompiles(cnt)

    @torch._dynamo.config.patch(
        recompile_limit=1024, accumulated_recompile_limit=4096
    )
    def test_concurrent_recompile(self) -> None:
        per_thread = 3

        def body(tid: int) -> None:
            cf = compile_hot_fn()
            for i in range(per_thread):
                n = 4 + tid * per_thread + i
                x = torch.randn(n, n)
                self.assertEqual(cf(x), hot_fn(x))

        run_threads(body)
        self.assertEqual(
            len(_debug_get_cache_entry_list(hot_fn.__code__)),
            NTHREADS * per_thread,
        )

    def test_reset_race(self) -> None:
        cf = compile_hot_fn()
        x = torch.randn(32, 32)
        expected = cf(x)
        stop = threading.Event()

        def resetter() -> None:
            # Unthrottled, every reset drops the workers' entries before they
            # can use them, and they spend the test recompiling.
            while not stop.is_set():
                torch._dynamo.reset()
                time.sleep(0.01)

        def body(tid: int) -> None:
            for _ in range(4 * ITERS):
                self.assertEqual(cf(x), expected)

        r = threading.Thread(target=resetter)
        r.start()
        try:
            run_threads(body)
        finally:
            stop.set()
            r.join()

    @torch._dynamo.config.patch(
        recompile_limit=1024, accumulated_recompile_limit=4096
    )
    def test_invalidate_races_reset(self) -> None:
        # Each call guards on the type of an instance of a new class, and
        # collecting the class invalidates that entry from this thread.
        cf = torch.compile(add_attr, backend="eager", dynamic=False)
        x = torch.randn(4)
        stop = threading.Event()

        def resetter() -> None:
            while not stop.is_set():
                torch._dynamo.reset()
                time.sleep(0.01)

        def body(tid: int) -> None:
            for _ in range(ITERS // 10):
                cls = type("C", (), {"v": 1})
                self.assertEqual(cf(x, cls()), x + 1)
                del cls
                gc.collect()

        r = threading.Thread(target=resetter)
        r.start()
        try:
            run_threads(body)
        finally:
            stop.set()
            r.join()

    def test_eval_frame_churn(self) -> None:
        cf = compile_hot_fn()
        x = torch.randn(16, 16)
        expected = cf(x)

        def body(tid: int) -> None:
            for _ in range(ITERS):
                if tid % 2 == 0:
                    self.assertEqual(cf(x), expected)
                else:
                    # Run-only mode is not None, so it installs the shim.
                    set_eval_frame(set_eval_frame(False))

        run_threads(body)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
