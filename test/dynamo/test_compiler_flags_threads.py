# Owner(s): ["module: dynamo"]

# Tests that the compile-session / export tracing flags
# (torch.compiler.is_compiling / torch._dynamo.is_compiling / is_exporting)
# are per-thread: other threads must not observe a compiling thread's session,
# and concurrent torch.compile + non-strict torch.export sessions must not
# corrupt each other's save/restore state (pytorch/pytorch#202042).

import threading

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._export.utils import _compiling_state_context
from torch.testing._internal.common_utils import HardwareClassification


flags_during_session = {}
flags_from_export_thread = {}
thread_failures = []


class WrapperTensor(torch.Tensor):
    """Traceable wrapper subclass for the #184547 __torch_dispatch__ scenario."""

    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.shape,
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )

    def __init__(self, elem):
        self._elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            return t._elem if isinstance(t, cls) else t

        flags_during_session.setdefault("dispatch", []).append(
            torch.compiler.is_compiling()
        )
        return func(
            *torch.utils._pytree.tree_map(unwrap, args),
            **torch.utils._pytree.tree_map(unwrap, kwargs or {}),
        )


def recording_backend(gm, example_inputs):
    flags_during_session.setdefault("backend", []).append(torch.compiler.is_compiling())
    return gm.forward


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

    def test_wrapper_subclass_torch_dispatch_sees_session_flags(self):
        flags_during_session.clear()
        thread_failures.clear()
        # Plain-input compile first: with the subclass input alone, dynamo
        # does not reach the backend on the process's very first compile.
        torch.compile(lambda x: x + 1, backend=recording_backend)(torch.ones(3))
        torch.compile(lambda x: x + 1, backend=recording_backend)(
            WrapperTensor(torch.ones(3))
        )
        self.assertTrue(flags_during_session["backend"])
        self.assertTrue(all(flags_during_session["backend"]))
        self.assertTrue(flags_during_session["dispatch"])
        self.assertTrue(all(flags_during_session["dispatch"]))
        self.assertFalse(torch.compiler.is_compiling())
        self.assertFalse(torch.compiler.is_exporting())

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
                # dynamo folds these calls at trace time to the values its
                # own (exporting) thread observes.
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
