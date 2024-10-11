# Owner(s): ["oncall: jit"]

import os
import sys
from textwrap import dedent

import torch
from torch.testing._internal import jit_utils


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


# Tests various JIT-related utility functions.
class TestJitUtils(JitTestCase):
    # Tests that POSITIONAL_OR_KEYWORD arguments are captured.
    def test_get_callable_argument_names_positional_or_keyword(self):
        def fn_positional_or_keyword_args_only(x, y):
            return x + y

        self.assertEqual(
            ["x", "y"],
            torch._jit_internal.get_callable_argument_names(
                fn_positional_or_keyword_args_only
            ),
        )

    # Tests that POSITIONAL_ONLY arguments are ignored.
    def test_get_callable_argument_names_positional_only(self):
        code = dedent(
            """
            def fn_positional_only_arg(x, /, y):
                return x + y
        """
        )

        fn_positional_only_arg = jit_utils._get_py3_code(code, "fn_positional_only_arg")
        self.assertEqual(
            ["y"],
            torch._jit_internal.get_callable_argument_names(fn_positional_only_arg),
        )

    # Tests that VAR_POSITIONAL arguments are ignored.
    def test_get_callable_argument_names_var_positional(self):
        # Tests that VAR_POSITIONAL arguments are ignored.
        def fn_var_positional_arg(x, *arg):
            return x + arg[0]

        self.assertEqual(
            ["x"],
            torch._jit_internal.get_callable_argument_names(fn_var_positional_arg),
        )

    # Tests that KEYWORD_ONLY arguments are ignored.
    def test_get_callable_argument_names_keyword_only(self):
        def fn_keyword_only_arg(x, *, y):
            return x + y

        self.assertEqual(
            ["x"], torch._jit_internal.get_callable_argument_names(fn_keyword_only_arg)
        )

    # Tests that VAR_KEYWORD arguments are ignored.
    def test_get_callable_argument_names_var_keyword(self):
        def fn_var_keyword_arg(**args):
            return args["x"] + args["y"]

        self.assertEqual(
            [], torch._jit_internal.get_callable_argument_names(fn_var_keyword_arg)
        )

    # Tests that a function signature containing various different types of
    # arguments are ignored.
    def test_get_callable_argument_names_hybrid(self):
        code = dedent(
            """
            def fn_hybrid_args(x, /, y, *args, **kwargs):
                return x + y + args[0] + kwargs['z']
        """
        )
        fn_hybrid_args = jit_utils._get_py3_code(code, "fn_hybrid_args")
        self.assertEqual(
            ["y"], torch._jit_internal.get_callable_argument_names(fn_hybrid_args)
        )

    def test_checkscriptassertraisesregex(self):
        def fn():
            tup = (1, 2)
            return tup[2]

        self.checkScriptRaisesRegex(fn, (), Exception, "range", name="fn")

        s = dedent(
            """
        def fn():
            tup = (1, 2)
            return tup[2]
        """
        )

        self.checkScriptRaisesRegex(s, (), Exception, "range", name="fn")

    def test_no_tracer_warn_context_manager(self):
        torch._C._jit_set_tracer_state_warn(True)
        with jit_utils.NoTracerWarnContextManager() as no_warn:
            self.assertEqual(False, torch._C._jit_get_tracer_state_warn())
        self.assertEqual(True, torch._C._jit_get_tracer_state_warn())
