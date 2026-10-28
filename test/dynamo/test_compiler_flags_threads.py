# Owner(s): ["module: dynamo"]

# Tests that the compile-session / export tracing flags
# (torch.compiler.is_compiling / torch._dynamo.is_compiling / is_exporting)
# are per-thread: other threads must not observe a compiling thread's session,
# and concurrent torch.compile + non-strict torch.export sessions must not
# corrupt each other's save/restore state (pytorch/pytorch#202042). CUDA
# backward nodes run on autograd engine worker threads, which only inherit
# at::ThreadLocalState, so the session flags must be visible there too.

import threading
import unittest

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._export.utils import _compiling_state_context
from torch.testing._internal.common_utils import HardwareClassification, TEST_CUDA


flags_during_session = {}
flags_from_export_thread = {}
thread_failures = []


class ProbeFn(torch.autograd.Function):
    # Python backward of a CUDA node runs on an autograd engine worker thread;
    # it reads the session flags through the TLS slot the session stashed for
    # at::ThreadLocalState inheritance.
    recorded: list = []

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        ProbeFn.recorded.append(
            (
                threading.current_thread() is threading.main_thread(),
                torch.compiler.is_compiling(),
                torch.compiler.is_exporting(),
                torch.compiler._is_non_strict_tracing(),
            )
        )
        return g


class CompilerFlagsThreadTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _join_thread(self, thread: threading.Thread) -> None:
        thread.join(60)
        self.assertFalse(
            thread.is_alive(), f"thread {thread.name} did not finish in time"
        )

    def test_is_compiling_not_observed_across_threads(self):
        flags_during_session.clear()
        thread_failures.clear()
        backend_started, release_backend = (threading.Event() for _ in range(2))

        def blocking_backend(gm, example_inputs):
            backend_started.set()
            # Custom backends run eagerly inside the compile session and must
            # observe the session flags in the compiling thread.
            flags_during_session["backend_eager"] = (
                torch.compiler.is_compiling(),
                torch._dynamo.is_compiling(),
                torch.compiler.is_exporting(),
                torch.compiler.is_dynamo_compiling(),
            )
            if not release_backend.wait(60):
                raise AssertionError("release_backend event not set in time")
            return gm.forward

        def compile_job():
            torch.compile(lambda x: x * 2, backend=blocking_backend)(torch.ones(3))

        compile_thread = threading.Thread(target=compile_job, name="compiling-thread")
        compile_thread.start()
        self.assertTrue(backend_started.wait(60))

        # Main thread runs no session and must not observe the compiling
        # thread's session (pytorch/pytorch#202042 repro 1).
        self.assertFalse(torch.compiler.is_compiling())
        self.assertFalse(torch._dynamo.is_compiling())
        self.assertFalse(torch.compiler.is_dynamo_compiling())
        self.assertFalse(torch.compiler.is_exporting())

        release_backend.set()
        self._join_thread(compile_thread)
        self.assertEqual(
            flags_during_session["backend_eager"], (True, True, False, False)
        )
        # Session exit restores the compiling thread's own flag.
        self.assertFalse(torch.compiler.is_compiling())

    def test_compile_export_interleave_no_stuck_flags(self):
        flags_from_export_thread.clear()
        thread_failures.clear()
        b_entered, release_b, a_entered, release_a = (
            threading.Event() for _ in range(4)
        )

        class M(torch.nn.Module):
            def forward(self, x):
                a_entered.set()
                # Read the flags from inside the exporting thread's session;
                # the eager execution happens in the exporting thread.
                flags_from_export_thread["is_exporting"] = torch.compiler.is_exporting()
                flags_from_export_thread["is_compiling"] = torch.compiler.is_compiling()
                release_a.wait(60)
                return x + 1

        def blocking_backend(gm, example_inputs):
            b_entered.set()
            release_b.wait(60)
            return gm.forward

        def compile_job():
            try:
                torch.compile(lambda x: x * 2, backend=blocking_backend)(torch.ones(3))
            except AssertionError as e:
                if "Guard failed on the same frame it was created" not in str(e):
                    thread_failures.append(e)
                # Else: known separate bug - the process-global fx annotation
                # state set by the concurrent export makes the compile's
                # FX_ANNOTATION ambient guard fail right here. Out of scope
                # for the per-thread flag fix (pytorch/pytorch#202042); the
                # session context still unwinds on this exception.

        def export_job():
            torch.export.export(M(), (torch.ones(3),), strict=False)

        compile_thread = threading.Thread(target=compile_job, name="compiling-thread")
        compile_thread.start()
        # The compile session must be entered before the export session so
        # that a process-global save/restore can be corrupted by their exit
        # order.
        self.assertTrue(b_entered.wait(60))

        export_thread = threading.Thread(target=export_job, name="exporting-thread")
        export_thread.start()
        self.assertTrue(a_entered.wait(60))

        # Neither session may leak into the main thread (the export side of
        # pytorch/pytorch#202042).
        self.assertFalse(torch.compiler.is_compiling())
        self.assertFalse(torch.compiler.is_exporting())

        release_b.set()
        self._join_thread(compile_thread)
        self.assertTrue(flags_from_export_thread["is_exporting"])
        self.assertTrue(flags_from_export_thread["is_compiling"])

        release_a.set()
        self._join_thread(export_thread)
        self.assertEqual(thread_failures, [])
        # No flag may be stuck after both sessions exit.
        self.assertFalse(torch.compiler.is_compiling())
        self.assertFalse(torch.compiler.is_exporting())

    def test_non_strict_tracing_not_observed_across_threads(self):
        started, release = (threading.Event() for _ in range(2))
        read = {}

        def session_job():
            with torch.compiler._non_strict_tracing_context():
                read["in_session"] = torch.compiler._is_non_strict_tracing()
                started.set()
                release.wait(60)
            read["after_session"] = torch.compiler._is_non_strict_tracing()

        session_thread = threading.Thread(target=session_job, name="tracing-thread")
        session_thread.start()
        self.assertTrue(started.wait(60))
        self.assertFalse(torch.compiler._is_non_strict_tracing())
        release.set()
        self._join_thread(session_thread)
        self.assertTrue(read["in_session"])
        self.assertFalse(read["after_session"])

    @unittest.skipIf(not TEST_CUDA, "cuda needed")
    def test_session_flags_reach_autograd_backward_worker_thread(self):
        def run(session_ctx):
            ProbeFn.recorded.clear()
            with session_ctx():
                x = torch.randn(4, device="cuda", requires_grad=True)
                y = ProbeFn.apply(x.sin())
                torch.autograd.grad(y.sum(), (x,))
            return ProbeFn.recorded

        # A compile session holds the compiling flag for eager backward
        # tracing on the engine's CUDA worker thread (AOTAutograd traces the
        # joint graph with torch.autograd.grad; see #184547).
        records = run(torch.compiler._compile_session_context)
        self.assertTrue(records)
        self.assertTrue(any(not is_main for is_main, _, _, _ in records))
        self.assertTrue(all(compiling for _, compiling, _, _ in records))
        self.assertTrue(all(not exporting for _, _, exporting, _ in records))

        # Export holds both flags; backward tracing observes them on the
        # worker thread too.
        records = run(_compiling_state_context)
        self.assertTrue(records)
        self.assertTrue(any(not is_main for is_main, _, _, _ in records))
        self.assertTrue(
            all(compiling and exporting for _, compiling, exporting, _ in records)
        )

        self.assertFalse(torch.compiler.is_compiling())
        self.assertFalse(torch.compiler.is_exporting())
        self.assertFalse(torch.compiler._is_non_strict_tracing())

    @unittest.skipIf(not TEST_CUDA, "cuda needed")
    def test_non_strict_tracing_reaches_autograd_backward_worker_thread(self):
        # SAC recompute runs on the engine's CUDA worker thread during
        # non-strict tracing; torch.utils.checkpoint gates on this flag
        # (checkpoint.py _is_compiling).
        ProbeFn.recorded.clear()
        with torch.compiler._non_strict_tracing_context():
            x = torch.randn(4, device="cuda", requires_grad=True)
            y = ProbeFn.apply(x.sin())
            torch.autograd.grad(y.sum(), (x,))
        self.assertTrue(ProbeFn.recorded)
        self.assertTrue(all(not is_main for is_main, _, _, _ in ProbeFn.recorded))
        self.assertTrue(all(non_strict for _, _, _, non_strict in ProbeFn.recorded))
        self.assertFalse(torch.compiler._is_non_strict_tracing())

    def test_nested_session_restore_same_thread(self):
        with torch.compiler._compile_session_context():
            self.assertTrue(torch.compiler.is_compiling())
            with torch.compiler._compile_session_context():
                self.assertTrue(torch.compiler.is_compiling())
            self.assertTrue(torch.compiler.is_compiling())
        self.assertFalse(torch.compiler.is_compiling())

        with _compiling_state_context():
            self.assertTrue(torch.compiler.is_exporting())
            self.assertTrue(torch.compiler.is_compiling())
            with torch.compiler._compile_session_context():
                self.assertTrue(torch.compiler.is_compiling())
            self.assertTrue(torch.compiler.is_exporting())
            self.assertTrue(torch.compiler.is_compiling())
        self.assertFalse(torch.compiler.is_exporting())
        self.assertFalse(torch.compiler.is_compiling())


if __name__ == "__main__":
    run_tests()
