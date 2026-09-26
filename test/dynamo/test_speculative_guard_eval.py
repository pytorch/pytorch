# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch._C._dynamo import guards
from torch._C._dynamo.eval_frame import set_eval_frame, set_guard_complete_hook
from torch._dynamo.guards import DeletedGuardManagerWrapper, GuardManagerWrapper
from torch._dynamo.types import ConvertFrameReturn, GuardedCode, wrap_guarded_code
from torch._guards import CompileId
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


events = []
bytecode_pending = None


def target(x, mode):
    return x


def compiled_true(x, mode):
    events.append(("code", True))
    return x + 1


def compiled_false(x, mode):
    events.append(("code", False))
    return x - 1


def compiled_from_ticket(x, mode):
    global bytecode_pending
    events.append(("code", mode))
    if bytecode_pending is not None:
        result = bytecode_pending
        bytecode_pending = None
        return result
    return x + 1 if mode else x - 1


class Ticket:
    def __init__(self, mode, result):
        self.mode = mode
        self.result = result

    def commit(self):
        events.append(("commit", self.mode))
        return self.result

    def abort(self):
        events.append(("abort", self.mode))


class Descriptor:
    def __init__(self, mode, admit=True):
        self.mode = mode
        self.admit = admit

    def launch(self, frame_locals):
        events.append(("launch", self.mode, frame_locals["mode"]))
        if not self.admit:
            return None
        x = frame_locals["x"]
        result = x + 1 if self.mode else x - 1
        return Ticket(self.mode, result)


class UnaryDescriptor:
    def launch(self, frame_locals):
        events.append(("launch",))
        return Ticket("unary", frame_locals["x"] + 1)


class StructuredDescriptor:
    def launch(self, frame_locals):
        events.append(("launch",))
        x = frame_locals["x"]
        return Ticket("structured", {"result": (x + 1, x + 2)})


class BytecodeTicket:
    def __init__(self, result):
        self.result = result

    def commit_to_cached_code(self):
        global bytecode_pending
        events.append(("commit_to_cached_code",))
        bytecode_pending = self.result

    def finish(self):
        events.append(("finish",))
        if bytecode_pending is not None:
            raise RuntimeError("speculative result was not consumed")

    def abort(self):
        global bytecode_pending
        events.append(("abort_bytecode",))
        bytecode_pending = None


class BytecodeDescriptor:
    def launch(self, frame_locals):
        events.append(("launch_bytecode",))
        return BytecodeTicket(frame_locals["x"] + 1)


class MutatingDescriptor(Descriptor):
    def launch(self, frame_locals):
        actual_mode = frame_locals["mode"]
        events.append(("launch", self.mode, actual_mode))
        frame_locals["mode"] = self.mode
        x = frame_locals["x"]
        result = x + 1 if self.mode else x - 1
        return Ticket(self.mode, result)


class ResetDescriptor(Descriptor):
    def launch(self, frame_locals):
        events.append(("launch", self.mode, frame_locals["mode"]))
        torch._C._dynamo.eval_frame.reset_code(target.__code__)
        return Ticket(self.mode, frame_locals["x"] + 1)


class ErrorTicket(Ticket):
    def commit(self):
        events.append(("commit_error", self.mode))
        raise RuntimeError("commit failed")


class ErrorDescriptor(Descriptor):
    def __init__(self, phase):
        super().__init__(True)
        self.phase = phase

    def launch(self, frame_locals):
        events.append(("launch", self.mode, frame_locals["mode"]))
        if self.phase == "launch":
            raise RuntimeError("launch failed")
        return ErrorTicket(self.mode, frame_locals["x"] + 1)


def make_guard_manager(expected):
    root = guards.RootGuardManager()
    root.add_lambda_guard(
        lambda frame_locals: frame_locals["mode"] is expected,
        [f"mode is {expected}"],
        None,
    )
    return GuardManagerWrapper(root)


@instantiate_parametrized_tests
class SpeculativeGuardEvalTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        global bytecode_pending
        super().setUp()
        events.clear()
        bytecode_pending = None

    def test_hit_commits_and_wrong_prediction_aborts(self):
        compile_id = 0

        def callback(frame, cache_entry, frame_state):
            nonlocal compile_id
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            mode = frame.f_locals["mode"]
            code = compiled_true.__code__ if mode else compiled_false.__code__
            result = wrap_guarded_code(
                GuardedCode(
                    code,
                    make_guard_manager(mode),
                    CompileId(frame_id=None, frame_compile_id=compile_id),
                    speculation_descriptor=Descriptor(mode),
                )
            )
            compile_id += 1
            return result

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            self.assertEqual(target(10, False), 9)
            events.clear()

            self.assertEqual(target(10, True), 11)
            self.assertEqual(
                events,
                [("launch", False, True), ("abort", False), ("code", True)],
            )

            events.clear()
            self.assertEqual(target(10, True), 11)
            self.assertEqual(events, [("launch", True, True), ("commit", True)])

            entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(target)
            self.assertIsNotNone(entries[0].speculation_descriptor)
        finally:
            set_eval_frame(previous)

    def test_hit_can_resume_cached_bytecode(self):
        descriptor = BytecodeDescriptor()

        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_from_ticket.__code__,
                    make_guard_manager(True),
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=descriptor,
                )
            )

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            events.clear()

            self.assertEqual(target(10, True), 11)
            self.assertEqual(
                events,
                [
                    ("launch_bytecode",),
                    ("commit_to_cached_code",),
                    ("code", True),
                    ("finish",),
                ],
            )
        finally:
            set_eval_frame(previous)

    def test_rejected_launch_uses_normal_cached_code(self):
        descriptor = Descriptor(True, admit=False)

        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__,
                    make_guard_manager(True),
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=descriptor,
                )
            )

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            events.clear()
            self.assertEqual(target(10, True), 11)
            self.assertEqual(events, [("launch", True, True), ("code", True)])
        finally:
            set_eval_frame(previous)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    def test_backend_descriptor_is_attached_to_cache_entry(self):
        descriptor = UnaryDescriptor()

        def fn(x):
            return x + 1

        def backend(gm, example_inputs):
            class Run:
                def __call__(self, *args):
                    events.append(("backend",))
                    return gm(*args)

            run = Run()
            run._torchdynamo_speculation_descriptor = descriptor
            return run

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.tensor(4)
        self.assertEqual(compiled(x), 5)
        events.clear()

        self.assertEqual(compiled(x), 5)
        self.assertEqual(events, [("launch",), ("commit", "unary")])

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIs(entries[0].speculation_descriptor, descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    def test_backend_descriptor_with_backend_specialization(self):
        descriptor = UnaryDescriptor()

        def fn(x):
            return x + 1

        def backend(gm, example_inputs):
            def run(*args):
                events.append(("backend",))
                return gm(*args)

            run._torchdynamo_speculation_descriptor = descriptor
            return run

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(
            x,
            0,
            specialize_on=[lambda size: size == 4],
        )
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        self.assertEqual(compiled(x), x + 1)
        events.clear()

        self.assertEqual(compiled(x), x + 1)
        self.assertEqual(events, [("launch",), ("commit", "unary")])

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIs(entries[0].speculation_descriptor, descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    def test_exhausted_local_generator_is_eligible(self):
        descriptor = UnaryDescriptor()

        def fn(x):
            return sum(value for value in (x, 1))

        def backend(gm, example_inputs):
            def run(*args):
                return gm(*args)

            run._torchdynamo_speculation_descriptor = descriptor
            return run

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.tensor(4)
        self.assertEqual(compiled(x), 5)
        events.clear()

        self.assertEqual(compiled(x), 5)
        self.assertEqual(events, [("launch",), ("commit", "unary")])

    @torch._dynamo.config.patch(speculative_guard_eval=False)
    def test_backend_descriptor_requires_opt_in(self):
        descriptor = UnaryDescriptor()

        def fn(x):
            return x + 1

        def backend(gm, example_inputs):
            def run(*args):
                events.append(("backend",))
                return gm(*args)

            run._torchdynamo_speculation_descriptor = descriptor
            return run

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.tensor(4)
        self.assertEqual(compiled(x), 5)
        events.clear()

        self.assertEqual(compiled(x), 5)
        self.assertEqual(events, [("backend",)])

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNone(entries[0].speculation_descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    def test_backend_descriptor_requires_fullgraph(self):
        descriptor = UnaryDescriptor()

        def fn(x):
            return x + 1

        def backend(gm, example_inputs):
            def run(*args):
                events.append(("backend",))
                return gm(*args)

            run._torchdynamo_speculation_descriptor = descriptor
            return run

        compiled = torch.compile(fn, backend=backend)
        x = torch.tensor(4)
        self.assertEqual(compiled(x), 5)
        events.clear()

        self.assertEqual(compiled(x), 5)
        self.assertEqual(events, [("backend",)])

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNone(entries[0].speculation_descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    def test_backend_descriptor_requires_no_python_side_effects(self):
        descriptor = UnaryDescriptor()
        side_effects = []

        def fn(x):
            side_effects.append(x)
            return x + 1

        def backend(gm, example_inputs):
            def run(*args):
                return gm(*args)

            run._torchdynamo_speculation_descriptor = descriptor
            return run

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        self.assertEqual(compiled(torch.tensor(4)), 5)
        self.assertEqual(len(side_effects), 1)

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNone(entries[0].speculation_descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    def test_descriptor_returns_full_frame_result(self):
        descriptor = StructuredDescriptor()

        def fn(x):
            return {"result": (x + 1, x + 2)}

        def backend(gm, example_inputs):
            def run(*args):
                return gm(*args)

            run._torchdynamo_speculation_descriptor = descriptor
            return run

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.tensor(4)
        self.assertEqual(compiled(x), {"result": (x + 1, x + 2)})
        events.clear()

        self.assertEqual(compiled(x), {"result": (x + 1, x + 2)})
        self.assertEqual(events, [("launch",), ("commit", "structured")])

    def test_guard_complete_rejection_aborts(self):
        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__,
                    make_guard_manager(True),
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=Descriptor(True),
                )
            )

        previous = set_eval_frame(callback)
        previous_hook = set_guard_complete_hook(None)
        try:
            self.assertEqual(target(10, True), 11)
            events.clear()
            set_guard_complete_hook(lambda guard_passed: False)

            self.assertEqual(target(10, True), 11)
            self.assertEqual(
                events,
                [("launch", True, True), ("abort", True), ("code", True)],
            )
        finally:
            set_guard_complete_hook(previous_hook)
            set_eval_frame(previous)

    def test_guard_complete_error_aborts(self):
        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__,
                    make_guard_manager(True),
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=Descriptor(True),
                )
            )

        def guard_complete_hook(guard_passed):
            events.append(("hook", guard_passed))
            raise RuntimeError("guard collective failed")

        previous = set_eval_frame(callback)
        previous_hook = set_guard_complete_hook(None)
        try:
            self.assertEqual(target(10, True), 11)
            events.clear()
            set_guard_complete_hook(guard_complete_hook)

            with self.assertRaisesRegex(RuntimeError, "guard collective failed"):
                target(10, True)
            self.assertEqual(
                events,
                [("launch", True, True), ("hook", True), ("abort", True)],
            )
            current = set_eval_frame(None)
            self.assertIs(current, callback)
            set_eval_frame(current)
        finally:
            set_guard_complete_hook(previous_hook)
            set_eval_frame(previous)

    def test_guard_exception_aborts(self):
        root = guards.RootGuardManager()

        def fail_guard(frame_locals):
            events.append(("guard", frame_locals["mode"]))
            raise RuntimeError("guard failed")

        root.add_lambda_guard(fail_guard, ["fail_guard"], None)
        guard_manager = GuardManagerWrapper(root)

        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__,
                    guard_manager,
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=Descriptor(True),
                )
            )

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            events.clear()

            self.assertEqual(target(10, True), 11)
            self.assertEqual(
                events,
                [
                    ("launch", True, True),
                    ("guard", True),
                    ("abort", True),
                    ("code", True),
                ],
            )
            current = set_eval_frame(None)
            self.assertIs(current, callback)
            set_eval_frame(current)
        finally:
            set_eval_frame(previous)

    def test_cache_miss_aborts(self):
        root = guards.RootGuardManager()
        root.add_lambda_guard(lambda frame_locals: False, ["False"], None)
        guard_manager = GuardManagerWrapper(root)

        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__,
                    guard_manager,
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=Descriptor(True),
                )
            )

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            events.clear()

            self.assertEqual(target(10, True), 11)
            self.assertEqual(
                events,
                [("launch", True, True), ("abort", True), ("code", True)],
            )
        finally:
            set_eval_frame(previous)

    def test_invalidation_clears_descriptor(self):
        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__,
                    make_guard_manager(True),
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=Descriptor(True),
                )
            )

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            entry = torch._dynamo.eval_frame._debug_get_cache_entry_list(target)[0]
            extra_state = entry.guard_manager.extra_state
            extra_state.invalidate(entry, DeletedGuardManagerWrapper("test"))
            self.assertIsNone(entry.speculation_descriptor)
        finally:
            set_eval_frame(previous)

    def test_descriptor_cannot_change_guard_inputs(self):
        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            mode = frame.f_locals["mode"]
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__ if mode else compiled_false.__code__,
                    make_guard_manager(mode),
                    CompileId(frame_id=None, frame_compile_id=int(mode)),
                    speculation_descriptor=MutatingDescriptor(mode),
                )
            )

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            events.clear()

            self.assertEqual(target(10, False), 9)
            self.assertEqual(
                events,
                [("launch", True, False), ("abort", True), ("code", False)],
            )
        finally:
            set_eval_frame(previous)

    def test_descriptor_cache_reset_does_not_use_stale_entry(self):
        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__,
                    make_guard_manager(True),
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=ResetDescriptor(True),
                )
            )

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            events.clear()

            self.assertEqual(target(10, True), 11)
            self.assertEqual(
                events,
                [("launch", True, True), ("abort", True), ("code", True)],
            )
        finally:
            set_eval_frame(previous)

    @parametrize("phase", ("launch", "commit"))
    def test_descriptor_error_restores_eval_frame(self, phase):
        descriptor = ErrorDescriptor(phase)

        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not target.__code__:
                return ConvertFrameReturn()
            return wrap_guarded_code(
                GuardedCode(
                    compiled_true.__code__,
                    make_guard_manager(True),
                    CompileId(frame_id=None, frame_compile_id=0),
                    speculation_descriptor=descriptor,
                )
            )

        previous = set_eval_frame(callback)
        try:
            self.assertEqual(target(10, True), 11)
            with self.assertRaisesRegex(RuntimeError, f"{phase} failed"):
                target(10, True)

            current = set_eval_frame(None)
            self.assertIs(current, callback)
            set_eval_frame(current)
        finally:
            set_eval_frame(previous)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
