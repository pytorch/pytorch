# Owner(s): ["module: dynamo"]
import importlib
import os
import sys
import types
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounter, optimize_assert, reset


g_tensor_export = torch.ones(10)


tensor_for_import_testing = torch.ones(10, 10)


def inner_func():
    return torch.is_grad_enabled()


def outer_func(func):
    def wrapped(*args):
        a = func(*args)
        torch._dynamo.graph_break()
        return torch.sin(a + 1), inner_func()

    return wrapped


# Create a dummy python module and function to test skipfiles rules.
module_code = """
def add(x):
    return x + 1
"""


def add(x):
    return x + 1


def create_dummy_module_and_function():
    module = types.ModuleType("dummy_module")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "dummy_module", None, origin=os.path.abspath(__file__)
    )
    exec(module_code, module.__dict__)
    sys.modules["dummy_module"] = module
    # Need to override the original function since its __code__.co_filename is not a regular python file name,
    # and the skipfiles rules use filename when checking SKIP_DIRS.
    module.add = add
    return module, module.add


@contextmanager
def install_guard_manager_testing_hook(hook_fn):
    old_value = torch._dynamo.guards.guard_manager_testing_hook_fn
    try:
        torch._dynamo.guards.guard_manager_testing_hook_fn = hook_fn
        yield
    finally:
        torch._dynamo.guards.guard_manager_testing_hook_fn = old_value


def make_dynamo_test(fn=None, expected_frame_count=1):
    if fn is None:
        return lambda fn: make_dynamo_test(
            fn, expected_frame_count=expected_frame_count
        )

    def standard_test(
        self: Any,
        fn: Callable[..., Any],
        expected_frame_count: int = 1,
    ) -> None:
        def dummy(self_, fn, t):
            fn(self_)
            return t.sin()

        actual = CompileCounter()

        t = torch.randn(2)
        correct = dummy(self, fn, t)
        reset()
        opt_fn = optimize_assert(actual)(dummy)
        val = opt_fn(self, fn, t)
        reset()
        self.assertEqual(correct, val)
        self.assertEqual(actual.frame_count, expected_frame_count)

    def test_fn(self):
        return standard_test(
            self,
            fn=fn,
            expected_frame_count=expected_frame_count,
        )

    return test_fn
