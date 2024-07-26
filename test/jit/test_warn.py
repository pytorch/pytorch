# Owner(s): ["oncall: jit"]

import io
import os
import sys
import warnings
from contextlib import redirect_stderr

import torch
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestWarn(JitTestCase):
    def test_warn(self):
        @torch.jit.script
        def fn():
            warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_only_once(self):
        @torch.jit.script
        def fn():
            for _ in range(10):
                warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_only_once_in_loop_func(self):
        def w():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            for _ in range(10):
                w()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_once_per_func(self):
        def w1():
            warnings.warn("I am warning you")

        def w2():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            w1()
            w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_once_per_func_in_loop(self):
        def w1():
            warnings.warn("I am warning you")

        def w2():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            for _ in range(10):
                w1()
                w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_multiple_calls_multiple_warnings(self):
        @torch.jit.script
        def fn():
            warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_multiple_calls_same_func_diff_stack(self):
        def warn(caller: str):
            warnings.warn("I am warning you from " + caller)

        @torch.jit.script
        def foo():
            warn("foo")

        @torch.jit.script
        def bar():
            warn("bar")

        f = io.StringIO()
        with redirect_stderr(f):
            foo()
            bar()

        FileCheck().check_count(
            str="UserWarning: I am warning you from foo", count=1, exactly=True
        ).check_count(
            str="UserWarning: I am warning you from bar", count=1, exactly=True
        ).run(
            f.getvalue()
        )
