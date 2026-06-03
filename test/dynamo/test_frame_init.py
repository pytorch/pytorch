# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.guards
import torch._dynamo.test_case
from torch._C._dynamo.eval_frame import get_eval_frame_callback, set_eval_frame
from torch._dynamo.types import ConvertFrameReturn, GuardedCode, wrap_guarded_code
from torch._guards import CompileId


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


class FrameInitTests(torch._dynamo.test_case.TestCase):
    def _compile_id(self):
        return CompileId(frame_id=None, frame_compile_id=0, compiled_autograd_id=0)

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
                        self._compile_id(),
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
                        self._compile_id(),
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

    def test_guard_eval_recursion_uses_default_eval_frame(self):
        guard_calls = []
        helper_intercepts = []

        def target(x):
            return x + 1

        def target_code(x):
            return x + 1

        def helper():
            guard_calls.append("helper")
            return True

        def helper_code():
            return False

        target_guard_manager = torch._dynamo.guards.GuardManagerWrapper()
        target_guard_manager.root.add_lambda_guard(
            lambda _f_locals: helper(), ["helper()"], None
        )
        empty_guard_manager = torch._dynamo.guards.GuardManagerWrapper()

        def callback(frame, cache_entry, frame_state):
            if frame.f_code is target.__code__:
                return wrap_guarded_code(
                    GuardedCode(
                        target_code.__code__,
                        target_guard_manager,
                        self._compile_id(),
                    )
                )
            if frame.f_code is helper.__code__:
                helper_intercepts.append(frame.f_code.co_name)
                return wrap_guarded_code(
                    GuardedCode(
                        helper_code.__code__,
                        empty_guard_manager,
                        self._compile_id(),
                    )
                )
            return ConvertFrameReturn()

        torch._dynamo.reset()
        original = set_eval_frame(callback)
        try:
            self.assertEqual(target(1), 2)
            self.assertEqual(target(1), 2)
        finally:
            set_eval_frame(original)

        self.assertEqual(guard_calls, ["helper"])
        self.assertEqual(helper_intercepts, [])

    def test_run_only_guard_eval_uses_default_eval_frame_for_nested_frames(self):
        guard_callbacks = []
        helper_calls = []

        def target(x):
            return x + 1

        def target_code(x):
            return x + 1

        def helper():
            helper_calls.append("eager")
            return True

        def helper_code():
            helper_calls.append("cached")
            return False

        target_guard_manager = torch._dynamo.guards.GuardManagerWrapper()
        target_guard_manager.root.add_lambda_guard(
            lambda _f_locals: (
                guard_callbacks.append(get_eval_frame_callback()) or helper()
            ),
            ["helper()"],
            None,
        )
        empty_guard_manager = torch._dynamo.guards.GuardManagerWrapper()

        def callback(frame, cache_entry, frame_state):
            if frame.f_code is target.__code__:
                return wrap_guarded_code(
                    GuardedCode(
                        target_code.__code__,
                        target_guard_manager,
                        self._compile_id(),
                    )
                )
            if frame.f_code is helper.__code__:
                return wrap_guarded_code(
                    GuardedCode(
                        helper_code.__code__,
                        empty_guard_manager,
                        self._compile_id(),
                    )
                )
            return ConvertFrameReturn()

        torch._dynamo.reset()
        original = set_eval_frame(callback)
        try:
            self.assertFalse(helper())
            self.assertEqual(target(1), 2)
            helper_calls.clear()
        finally:
            set_eval_frame(original)

        self.assertEqual(torch._dynamo.run(target)(1), 2)

        self.assertEqual(guard_callbacks, [False])
        self.assertEqual(helper_calls, ["eager"])


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
