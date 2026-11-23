# Owner(s): ["module: dynamo"]

import gc
import sys

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._C._dynamo.eval_frame import (
    _invalidate_cache_entry,
    reset_code,
    set_eval_frame,
)
from torch._dynamo.guards import DeletedGuardManagerWrapper
from torch._dynamo.types import ConvertFrameReturn, GuardedCode, wrap_guarded_code
from torch._guards import CompileId
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    skipIfFreeThreaded,
)


def target_with_varkwargs(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    local = 1
    return {
        "local": local,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def varkwargs_code1(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # remove a local variable: local = 1
    return {
        "local": 1,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def varkwargs_code2(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # introduce a local variable
    local1 = 0
    local2 = 1
    return {
        "local": local1 + local2,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def target_with_varargs(arg1, /, positional_only_arg, *varargs, **kwargs):
    local = 1
    return {
        "local": local,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


def varargs_code1(arg1, /, positional_only_arg, *varargs, **kwargs):
    # remove a local variable: local = 1
    return {
        "local": 1,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


def varargs_code2(arg1, /, positional_only_arg, *varargs, **kwargs):
    # introduce a local variable
    local1 = 0
    local2 = 1
    return {
        "local": local1 + local2,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


def invalidate(guard_manager):
    return _invalidate_cache_entry(guard_manager, DeletedGuardManagerWrapper("test"))


@instantiate_parametrized_tests
class FrameInitTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("guardless", (False, True))
    @torch._dynamo.testing.lru_cache_reordering(True)
    def test_lookup_continues_after_candidate_invalidation(self, guardless):
        def fn(x):
            return x + 1

        def replacement(x):
            return x + 2

        class Callback:
            def __init__(self):
                self.guard_manager = torch._dynamo.guards.GuardManagerWrapper()
                if not guardless:
                    self.guard_manager.root.add_lambda_guard(
                        lambda locals: True, ["always passes"], None
                    )
                self.compile_count = 0
                self.matches = False
                self.invalidate_on_compare = False
                self.invalidated = False

            def __call__(self, frame, cache_entry, frame_state):
                if frame.f_code is not fn.__code__:
                    return ConvertFrameReturn()
                self.compile_count += 1
                return wrap_guarded_code(
                    GuardedCode(
                        replacement.__code__,
                        self.guard_manager,
                        CompileId(frame_id=0, frame_compile_id=self.compile_count),
                    )
                )

            def __eq__(self, other):
                if self.invalidate_on_compare:
                    self.invalidate_on_compare = False
                    self.invalidated = invalidate(self.guard_manager)
                return self is other or self.matches

        fallback = Callback()
        first = Callback()
        original = set_eval_frame(fallback)
        try:
            fn(1)
            set_eval_frame(first)
            fn(1)
            first.matches = True
            first.invalidate_on_compare = True
            set_eval_frame(fallback)
            result = fn(1)
        finally:
            set_eval_frame(original)

        self.assertEqual(result, 3)
        self.assertTrue(first.invalidated)
        self.assertEqual((first.compile_count, fallback.compile_count), (1, 1))

    @parametrize("guardless", (False, True))
    @torch._dynamo.testing.lru_cache_reordering(True)
    def test_entries_sharing_a_guard_manager(self, guardless):
        def fn(x):
            return x + 1

        def replacement_a(x):
            return x + 2

        def replacement_b(x):
            return x + 3

        guard_manager = torch._dynamo.guards.GuardManagerWrapper()
        if not guardless:
            guard_manager.root.add_lambda_guard(
                lambda locals: True, ["always passes"], None
            )

        class Callback:
            def __init__(self, code):
                self.code = code
                self.compile_count = 0

            def __call__(self, frame, cache_entry, frame_state):
                if frame.f_code is not fn.__code__:
                    return ConvertFrameReturn()
                self.compile_count += 1
                compile_id = CompileId(frame_id=0, frame_compile_id=self.compile_count)
                return wrap_guarded_code(
                    GuardedCode(self.code, guard_manager, compile_id)
                )

        a = Callback(replacement_a.__code__)
        b = Callback(replacement_b.__code__)
        original = set_eval_frame(a)
        try:
            fn(1)
            set_eval_frame(b)
            fn(1)
            # b's entry is now in front of a's, with the same guard manager.
            set_eval_frame(a)
            result = fn(1)
            invalidated = invalidate(guard_manager)
            fn(1)
        finally:
            set_eval_frame(original)

        self.assertEqual(result, 3)
        self.assertTrue(invalidated)
        # Invalidation reaches both entries, so a recompiles.
        self.assertEqual((a.compile_count, b.compile_count), (2, 1))

    def test_callback_cache_entry_survives_reset(self):
        def fn(x):
            return x + 1

        def replacement(x):
            return x + 2

        guard_manager = torch._dynamo.guards.GuardManagerWrapper()
        guard_manager.root.add_lambda_guard(
            lambda locals: False, ["always fails"], None
        )
        seen = []

        def callback(frame, cache_entry, frame_state):
            if frame.f_code is not fn.__code__:
                return ConvertFrameReturn()
            if cache_entry is not None:
                # Frees the cache the entry came from.
                reset_code(fn.__code__)
                seen.append(cache_entry.code)
            compile_id = CompileId(frame_id=0, frame_compile_id=len(seen))
            return wrap_guarded_code(
                GuardedCode(replacement.__code__, guard_manager, compile_id)
            )

        original = set_eval_frame(callback)
        try:
            fn(1)
            fn(1)
        finally:
            set_eval_frame(original)

        self.assertEqual(seen, [replacement.__code__])

    def test_frame_init(self):
        code_map1 = {
            target_with_varargs.__code__: varargs_code1.__code__,
            target_with_varkwargs.__code__: varkwargs_code1.__code__,
        }
        code_map2 = {
            target_with_varargs.__code__: varargs_code2.__code__,
            target_with_varkwargs.__code__: varkwargs_code2.__code__,
        }

        empty_guard_manager = torch._dynamo.guards.GuardManagerWrapper()

        def callback1(frame, cache_entry, frame_state):
            if frame.f_code in code_map1:
                transformed_code = code_map1[frame.f_code]
                return wrap_guarded_code(
                    GuardedCode(
                        transformed_code,
                        empty_guard_manager,
                        CompileId(
                            frame_id=None, frame_compile_id=0, compiled_autograd_id=0
                        ),
                    )
                )
            return ConvertFrameReturn()

        def callback2(frame, cache_entry, frame_state):
            if frame.f_code in code_map2:
                transformed_code = code_map2[frame.f_code]
                return wrap_guarded_code(
                    GuardedCode(
                        transformed_code,
                        empty_guard_manager,
                        CompileId(
                            frame_id=None, frame_compile_id=0, compiled_autograd_id=0
                        ),
                    )
                )
            return ConvertFrameReturn()

        for _ in [callback1, callback2]:
            torch._dynamo.reset()
            expected_varargs_output = target_with_varargs(
                1, 2, 3, 4, name1=1, name2=2, name3=3
            )
            expected_kwargs_output = target_with_varkwargs(
                1, 2, keyword_only_arg=1, name2=2, name3=3
            )
            original = set_eval_frame(callback1)
            real_varargs_output = target_with_varargs(
                1, 2, 3, 4, name1=1, name2=2, name3=3
            )
            real_kwargs_output = target_with_varkwargs(
                1, 2, keyword_only_arg=1, name2=2, name3=3
            )
            self.assertEqual(real_varargs_output, expected_varargs_output)
            self.assertEqual(real_kwargs_output, expected_kwargs_output)
            set_eval_frame(original)


class FrameClearTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    # Free-threaded builds immortalize code objects, so refcounts there are
    # constant and this check cannot observe a leak.
    @skipIfFreeThreaded("code objects are immortal, refcounts are not observable")
    def test_intercepted_frame_releases_code_object(self):
        # clear_old_frame_if_python_312_plus stands in for CPython's
        # _PyEval_FrameClearAndPop, which releases frame->f_executable.
        def fn(x):
            return x + 1

        code = fn.__code__
        cnts = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnts, dynamic=False)
        x = torch.ones(3)
        compiled(x)
        compiled(x)  # settle into the cached path

        gc.collect()
        before = sys.getrefcount(code)
        for _ in range(50):
            compiled(x)
        gc.collect()
        # A recompile would also churn refcounts, so pin it down.
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(sys.getrefcount(code), before)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
