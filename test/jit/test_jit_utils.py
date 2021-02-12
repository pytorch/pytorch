import os
import sys
from textwrap import dedent
import unittest

import torch

from torch.testing._internal import jit_utils

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Tests various JIT-related utility functions.
class TestJitUtils(JitTestCase):
    # Tests that POSITIONAL_OR_KEYWORD arguments are captured.
    def test_get_callable_argument_names_positional_or_keyword(self):
        def fn_positional_or_keyword_args_only(x, y):
            return x + y
        self.assertEqual(
            ["x", "y"],
            torch._jit_internal.get_callable_argument_names(fn_positional_or_keyword_args_only))

    # Tests that POSITIONAL_ONLY arguments are ignored.
    @unittest.skipIf(sys.version_info < (3, 8), 'POSITIONAL_ONLY arguments are not supported before 3.8')
    def test_get_callable_argument_names_positional_only(self):
        code = dedent('''
            def fn_positional_only_arg(x, /, y):
                return x + y
        ''')

        fn_positional_only_arg = jit_utils._get_py3_code(code, 'fn_positional_only_arg')
        self.assertEqual(
            [],
            torch._jit_internal.get_callable_argument_names(fn_positional_only_arg))

    # Tests that VAR_POSITIONAL arguments are ignored.
    def test_get_callable_argument_names_var_positional(self):
        # Tests that VAR_POSITIONAL arguments are ignored.
        def fn_var_positional_arg(x, *arg):
            return x + arg[0]
        self.assertEqual(
            [],
            torch._jit_internal.get_callable_argument_names(fn_var_positional_arg))

    # Tests that KEYWORD_ONLY arguments are ignored.
    def test_get_callable_argument_names_keyword_only(self):
        def fn_keyword_only_arg(x, *, y):
            return x + y
        self.assertEqual(
            [],
            torch._jit_internal.get_callable_argument_names(fn_keyword_only_arg))

    # Tests that VAR_KEYWORD arguments are ignored.
    def test_get_callable_argument_names_var_keyword(self):
        def fn_var_keyword_arg(**args):
            return args['x'] + args['y']
        self.assertEqual(
            [],
            torch._jit_internal.get_callable_argument_names(fn_var_keyword_arg))

    # Tests that a function signature containing various different types of
    # arguments are ignored.
    @unittest.skipIf(sys.version_info < (3, 8), 'POSITIONAL_ONLY arguments are not supported before 3.8')
    def test_get_callable_argument_names_hybrid(self):
        code = dedent('''
            def fn_hybrid_args(x, /, y, *args, **kwargs):
                return x + y + args[0] + kwargs['z']
        ''')
        fn_hybrid_args = jit_utils._get_py3_code(code, 'fn_hybrid_args')
        self.assertEqual(
            [],
            torch._jit_internal.get_callable_argument_names(fn_hybrid_args))
