"""Test cases for traceback module"""

from collections import namedtuple
from io import StringIO
import linecache
import sys
import types
import inspect
import builtins
import unittest
import unittest.mock
import re
import tempfile
import random
import string
from test import support
import shutil
from test.support import (Error, captured_output, cpython_only, ALWAYS_EQ,
                          requires_debug_ranges, has_no_debug_ranges,
                          requires_subprocess)
from test.support.os_helper import TESTFN, unlink
from test.support.script_helper import assert_python_ok, assert_python_failure
from test.support.import_helper import forget
from test.support import force_not_colorized, force_not_colorized_test_class

import json
import textwrap
import traceback
from functools import partial
from pathlib import Path
import _colorize

# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_exceptions.py

import torch
import torch._dynamo.test_case
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    xfailIfTorchDynamo,
)

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

MODULE_PREFIX = f'{__name__}.' if __name__ == '__main__' else ''

test_code = namedtuple('code', ['co_filename', 'co_name'])
test_code.co_positions = lambda _: iter([(6, 6, 0, 0)])
test_frame = namedtuple('frame', ['f_code', 'f_globals', 'f_locals'])
test_tb = namedtuple('tb', ['tb_frame', 'tb_lineno', 'tb_next', 'tb_lasti'])


LEVENSHTEIN_DATA_FILE = Path(__file__).parent / 'levenshtein_examples.json'


class TracebackCases(__TestCase):
    # For now, a very minimal set of tests.  I want to be sure that
    # formatting of SyntaxErrors works based on changes for 2.1.
    def setUp(self):
        super().setUp()
        self.colorize = _colorize.COLORIZE
        _colorize.COLORIZE = False

    def tearDown(self):
        super().tearDown()
        _colorize.COLORIZE = self.colorize

    def get_exception_format(self, func, exc):
        try:
            func()
        except exc as value:
            return traceback.format_exception_only(exc, value)
        else:
            raise ValueError("call did not raise exception")

    def syntax_error_with_caret(self):
        compile("def fact(x):\n\treturn x!\n", "?", "exec")

    def syntax_error_with_caret_2(self):
        compile("1 +\n", "?", "exec")

    def syntax_error_with_caret_range(self):
        compile("f(x, y for y in range(30), z)", "?", "exec")

    def syntax_error_bad_indentation(self):
        compile("def spam():\n  print(1)\n print(2)", "?", "exec")

    def syntax_error_with_caret_non_ascii(self):
        compile('Python = "\u1e54\xfd\u0163\u0125\xf2\xf1" +', "?", "exec")

    def syntax_error_bad_indentation2(self):
        compile(" print(2)", "?", "exec")

    def tokenizer_error_with_caret_range(self):
        compile("blech  (  ", "?", "exec")

    def test_caret(self):
        err = self.get_exception_format(self.syntax_error_with_caret,
                                        SyntaxError)
        self.assertEqual(len(err), 4)
        self.assertTrue(err[1].strip() == "return x!")
        self.assertIn("^", err[2]) # third line has caret
        self.assertEqual(err[1].find("!"), err[2].find("^")) # in the right place
        self.assertEqual(err[2].count("^"), 1)

        err = self.get_exception_format(self.syntax_error_with_caret_2,
                                        SyntaxError)
        self.assertIn("^", err[2]) # third line has caret
        self.assertEqual(err[2].count('\n'), 1)   # and no additional newline
        self.assertEqual(err[1].find("+") + 1, err[2].find("^"))  # in the right place
        self.assertEqual(err[2].count("^"), 1)

        err = self.get_exception_format(self.syntax_error_with_caret_non_ascii,
                                        SyntaxError)
        self.assertIn("^", err[2]) # third line has caret
        self.assertEqual(err[2].count('\n'), 1)   # and no additional newline
        self.assertEqual(err[1].find("+") + 1, err[2].find("^"))  # in the right place
        self.assertEqual(err[2].count("^"), 1)

        err = self.get_exception_format(self.syntax_error_with_caret_range,
                                        SyntaxError)
        self.assertIn("^", err[2]) # third line has caret
        self.assertEqual(err[2].count('\n'), 1)   # and no additional newline
        self.assertEqual(err[1].find("y"), err[2].find("^"))  # in the right place
        self.assertEqual(err[2].count("^"), len("y for y in range(30)"))

        err = self.get_exception_format(self.tokenizer_error_with_caret_range,
                                        SyntaxError)
        self.assertIn("^", err[2]) # third line has caret
        self.assertEqual(err[2].count('\n'), 1)   # and no additional newline
        self.assertEqual(err[1].find("("), err[2].find("^"))  # in the right place
        self.assertEqual(err[2].count("^"), 1)

    def test_nocaret(self):
        exc = SyntaxError("error", ("x.py", 23, None, "bad syntax"))
        err = traceback.format_exception_only(SyntaxError, exc)
        self.assertEqual(len(err), 3)
        self.assertEqual(err[1].strip(), "bad syntax")

    @force_not_colorized
    def test_no_caret_with_no_debug_ranges_flag(self):
        # Make sure that if `-X no_debug_ranges` is used, there are no carets
        # in the traceback.
        try:
            with open(TESTFN, 'w') as f:
                f.write("x = 1 / 0\n")

            _, _, stderr = assert_python_failure(
                '-X', 'no_debug_ranges', TESTFN)

            lines = stderr.splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], b'Traceback (most recent call last):')
            self.assertIn(b'line 1, in <module>', lines[1])
            self.assertEqual(lines[2], b'    x = 1 / 0')
            self.assertEqual(lines[3], b'ZeroDivisionError: division by zero')
        finally:
            unlink(TESTFN)

    def test_no_caret_with_no_debug_ranges_flag_python_traceback(self):
        code = textwrap.dedent("""
            import traceback
            try:
                x = 1 / 0
            except ZeroDivisionError:
                traceback.print_exc()
            """)
        try:
            with open(TESTFN, 'w') as f:
                f.write(code)

            _, _, stderr = assert_python_ok(
                '-X', 'no_debug_ranges', TESTFN)

            lines = stderr.splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], b'Traceback (most recent call last):')
            self.assertIn(b'line 4, in <module>', lines[1])
            self.assertEqual(lines[2], b'    x = 1 / 0')
            self.assertEqual(lines[3], b'ZeroDivisionError: division by zero')
        finally:
            unlink(TESTFN)

    def test_recursion_error_during_traceback(self):
        code = textwrap.dedent("""
                import sys
                from weakref import ref

                sys.setrecursionlimit(15)

                def f():
                    ref(lambda: 0, [])
                    f()

                try:
                    f()
                except RecursionError:
                    pass
        """)
        try:
            with open(TESTFN, 'w') as f:
                f.write(code)

            rc, _, _ = assert_python_ok(TESTFN)
            self.assertEqual(rc, 0)
        finally:
            unlink(TESTFN)

    def test_bad_indentation(self):
        err = self.get_exception_format(self.syntax_error_bad_indentation,
                                        IndentationError)
        self.assertEqual(len(err), 4)
        self.assertEqual(err[1].strip(), "print(2)")
        self.assertIn("^", err[2])
        self.assertEqual(err[1].find(")") + 1, err[2].find("^"))

        # No caret for "unexpected indent"
        err = self.get_exception_format(self.syntax_error_bad_indentation2,
                                        IndentationError)
        self.assertEqual(len(err), 3)
        self.assertEqual(err[1].strip(), "print(2)")

    def test_base_exception(self):
        # Test that exceptions derived from BaseException are formatted right
        e = KeyboardInterrupt()
        lst = traceback.format_exception_only(e.__class__, e)
        self.assertEqual(lst, ['KeyboardInterrupt\n'])

    def test_format_exception_only_bad__str__(self):
        with torch._dynamo.error_on_graph_break(False):
            class X(Exception):
                def __str__(self):
                    1/0
        err = traceback.format_exception_only(X, X())
        self.assertEqual(len(err), 1)
        str_value = '<exception str() failed>'
        if X.__module__ in ('__main__', 'builtins'):
            str_name = X.__qualname__
        else:
            str_name = '.'.join([X.__module__, X.__qualname__])
        self.assertEqual(err[0], "%s: %s\n" % (str_name, str_value))

    def test_format_exception_group_without_show_group(self):
        eg = ExceptionGroup('A', [ValueError('B')])
        err = traceback.format_exception_only(eg)
        self.assertEqual(err, ['ExceptionGroup: A (1 sub-exception)\n'])

    def test_format_exception_group(self):
        eg = ExceptionGroup('A', [ValueError('B')])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A (1 sub-exception)\n',
            '   ValueError: B\n',
        ])

    def test_format_base_exception_group(self):
        eg = BaseExceptionGroup('A', [BaseException('B')])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'BaseExceptionGroup: A (1 sub-exception)\n',
            '   BaseException: B\n',
        ])

    def test_format_exception_group_with_note(self):
        exc = ValueError('B')
        exc.add_note('Note')
        eg = ExceptionGroup('A', [exc])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A (1 sub-exception)\n',
            '   ValueError: B\n',
            '   Note\n',
        ])

    def test_format_exception_group_explicit_class(self):
        eg = ExceptionGroup('A', [ValueError('B')])
        err = traceback.format_exception_only(ExceptionGroup, eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A (1 sub-exception)\n',
            '   ValueError: B\n',
        ])

    def test_format_exception_group_multiple_exceptions(self):
        eg = ExceptionGroup('A', [ValueError('B'), TypeError('C')])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A (2 sub-exceptions)\n',
            '   ValueError: B\n',
            '   TypeError: C\n',
        ])

    def test_format_exception_group_multiline_messages(self):
        eg = ExceptionGroup('A\n1', [ValueError('B\n2')])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A\n1 (1 sub-exception)\n',
            '   ValueError: B\n',
            '   2\n',
        ])

    def test_format_exception_group_multiline2_messages(self):
        exc = ValueError('B\n\n2\n')
        exc.add_note('\nC\n\n3')
        eg = ExceptionGroup('A\n\n1\n', [exc, IndexError('D')])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A\n\n1\n (2 sub-exceptions)\n',
            '   ValueError: B\n',
            '   \n',
            '   2\n',
            '   \n',
            '   \n',  # first char of `note`
            '   C\n',
            '   \n',
            '   3\n', # note ends
            '   IndexError: D\n',
        ])

    def test_format_exception_group_syntax_error(self):
        exc = SyntaxError("error", ("x.py", 23, None, "bad syntax"))
        eg = ExceptionGroup('A\n1', [exc])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A\n1 (1 sub-exception)\n',
            '     File "x.py", line 23\n',
            '       bad syntax\n',
            '   SyntaxError: error\n',
        ])

    def test_format_exception_group_nested_with_notes(self):
        exc = IndexError('D')
        exc.add_note('Note\nmultiline')
        eg = ExceptionGroup('A', [
            ValueError('B'),
            ExceptionGroup('C', [exc, LookupError('E')]),
            TypeError('F'),
        ])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A (3 sub-exceptions)\n',
            '   ValueError: B\n',
            '   ExceptionGroup: C (2 sub-exceptions)\n',
            '      IndexError: D\n',
            '      Note\n',
            '      multiline\n',
            '      LookupError: E\n',
            '   TypeError: F\n',
        ])

    def test_format_exception_group_with_tracebacks(self):
        def f():
            try:
                1 / 0
            except ZeroDivisionError as e:
                return e

        def g():
            try:
                raise TypeError('g')
            except TypeError as e:
                return e

        eg = ExceptionGroup('A', [
            f(),
            ExceptionGroup('B', [g()]),
        ])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A (2 sub-exceptions)\n',
            '   ZeroDivisionError: division by zero\n',
            '   ExceptionGroup: B (1 sub-exception)\n',
            '      TypeError: g\n',
        ])

    def test_format_exception_group_with_cause(self):
        def f():
            try:
                try:
                    1 / 0
                except ZeroDivisionError:
                    raise ValueError(0)
            except Exception as e:
                return e

        eg = ExceptionGroup('A', [f()])
        err = traceback.format_exception_only(eg, show_group=True)
        self.assertEqual(err, [
            'ExceptionGroup: A (1 sub-exception)\n',
            '   ValueError: 0\n',
        ])

    def test_format_exception_group_syntax_error_with_custom_values(self):
        # See https://github.com/python/cpython/issues/128894
        for exc in [
            SyntaxError('error', 'abcd'),
            SyntaxError('error', [None] * 4),
            SyntaxError('error', (1, 2, 3, 4)),
            SyntaxError('error', (1, 2, 3, 4)),
            SyntaxError('error', (1, 'a', 'b', 2)),
            # with end_lineno and end_offset:
            SyntaxError('error', 'abcdef'),
            SyntaxError('error', [None] * 6),
            SyntaxError('error', (1, 2, 3, 4, 5, 6)),
            SyntaxError('error', (1, 'a', 'b', 2, 'c', 'd')),
        ]:
            with self.subTest(exc=exc):
                err = traceback.format_exception_only(exc, show_group=True)
                # Should not raise an exception:
                if exc.lineno is not None:
                    self.assertEqual(len(err), 2)
                    self.assertTrue(err[0].startswith('  File'))
                else:
                    self.assertEqual(len(err), 1)
                self.assertEqual(err[-1], 'SyntaxError: error\n')

    @requires_subprocess()
    @force_not_colorized
    def test_encoded_file(self):
        # Test that tracebacks are correctly printed for encoded source files:
        # - correct line number (Issue2384)
        # - respect file encoding (Issue3975)
        import sys, subprocess

        # The spawned subprocess has its stdout redirected to a PIPE, and its
        # encoding may be different from the current interpreter, on Windows
        # at least.
        process = subprocess.Popen([sys.executable, "-c",
                                    "import sys; print(sys.stdout.encoding)"],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        output_encoding = str(stdout, 'ascii').splitlines()[0]

        def do_test(firstlines, message, charset, lineno):
            # Raise the message in a subprocess, and catch the output
            try:
                with open(TESTFN, "w", encoding=charset) as output:
                    output.write("""{0}if 1:
                        import traceback;
                        raise RuntimeError('{1}')
                        """.format(firstlines, message))

                process = subprocess.Popen([sys.executable, TESTFN],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = process.communicate()
                stdout = stdout.decode(output_encoding).splitlines()
            finally:
                unlink(TESTFN)

            # The source lines are encoded with the 'backslashreplace' handler
            encoded_message = message.encode(output_encoding,
                                             'backslashreplace')
            # and we just decoded them with the output_encoding.
            message_ascii = encoded_message.decode(output_encoding)

            err_line = "raise RuntimeError('{0}')".format(message_ascii)
            err_msg = "RuntimeError: {0}".format(message_ascii)

            self.assertIn(("line %s" % lineno), stdout[1],
                "Invalid line number: {0!r} instead of {1}".format(
                    stdout[1], lineno))
            self.assertTrue(stdout[2].endswith(err_line),
                "Invalid traceback line: {0!r} instead of {1!r}".format(
                    stdout[2], err_line))
            actual_err_msg = stdout[3]
            self.assertTrue(actual_err_msg == err_msg,
                "Invalid error message: {0!r} instead of {1!r}".format(
                    actual_err_msg, err_msg))

        do_test("", "foo", "ascii", 3)
        for charset in ("ascii", "iso-8859-1", "utf-8", "GBK"):
            if charset == "ascii":
                text = "foo"
            elif charset == "GBK":
                text = "\u4E02\u5100"
            else:
                text = "h\xe9 ho"
            do_test("# coding: {0}\n".format(charset),
                    text, charset, 4)
            do_test("#!shebang\n# coding: {0}\n".format(charset),
                    text, charset, 5)
            do_test(" \t\f\n# coding: {0}\n".format(charset),
                    text, charset, 5)
        # Issue #18960: coding spec should have no effect
        do_test("x=0\n# coding: GBK\n", "h\xe9 ho", 'utf-8', 5)

    def test_print_traceback_at_exit(self):
        # Issue #22599: Ensure that it is possible to use the traceback module
        # to display an exception at Python exit
        code = textwrap.dedent("""
            import sys
            import traceback

            class PrintExceptionAtExit(object):
                def __init__(self):
                    try:
                        x = 1 / 0
                    except Exception as e:
                        self.exc = e
                        # self.exc.__traceback__ contains frames:
                        # explicitly clear the reference to self in the current
                        # frame to break a reference cycle
                        self = None

                def __del__(self):
                    traceback.print_exception(self.exc)

            # Keep a reference in the module namespace to call the destructor
            # when the module is unloaded
            obj = PrintExceptionAtExit()
        """)
        rc, stdout, stderr = assert_python_ok('-c', code)
        expected = [b'Traceback (most recent call last):',
                    b'  File "<string>", line 8, in __init__',
                    b'    x = 1 / 0',
                    b'        ^^^^^',
                    b'ZeroDivisionError: division by zero']
        self.assertEqual(stderr.splitlines(), expected)

    def test_print_exception(self):
        output = StringIO()
        traceback.print_exception(
            Exception, Exception("projector"), None, file=output
        )
        self.assertEqual(output.getvalue(), "Exception: projector\n")

    def test_print_exception_exc(self):
        output = StringIO()
        traceback.print_exception(Exception("projector"), file=output)
        self.assertEqual(output.getvalue(), "Exception: projector\n")

    def test_print_last(self):
        with support.swap_attr(sys, 'last_exc', ValueError(42)):
            output = StringIO()
            traceback.print_last(file=output)
            self.assertEqual(output.getvalue(), "ValueError: 42\n")

    def test_format_exception_exc(self):
        e = Exception("projector")
        output = traceback.format_exception(e)
        self.assertEqual(output, ["Exception: projector\n"])
        with self.assertRaisesRegex(ValueError, 'Both or neither'):
            traceback.format_exception(e.__class__, e)
        with self.assertRaisesRegex(ValueError, 'Both or neither'):
            traceback.format_exception(e.__class__, tb=e.__traceback__)
        with self.assertRaisesRegex(TypeError, 'required positional argument'):
            traceback.format_exception(exc=e)

    def test_format_exception_only_exc(self):
        output = traceback.format_exception_only(Exception("projector"))
        self.assertEqual(output, ["Exception: projector\n"])

    def test_exception_is_None(self):
        NONE_EXC_STRING = 'NoneType: None\n'
        excfile = StringIO()
        traceback.print_exception(None, file=excfile)
        self.assertEqual(excfile.getvalue(), NONE_EXC_STRING)

        excfile = StringIO()
        traceback.print_exception(None, None, None, file=excfile)
        self.assertEqual(excfile.getvalue(), NONE_EXC_STRING)

        excfile = StringIO()
        traceback.print_exc(None, file=excfile)
        self.assertEqual(excfile.getvalue(), NONE_EXC_STRING)

        self.assertEqual(traceback.format_exc(None), NONE_EXC_STRING)
        self.assertEqual(traceback.format_exception(None), [NONE_EXC_STRING])
        self.assertEqual(
            traceback.format_exception(None, None, None), [NONE_EXC_STRING])
        self.assertEqual(
            traceback.format_exception_only(None), [NONE_EXC_STRING])
        self.assertEqual(
            traceback.format_exception_only(None, None), [NONE_EXC_STRING])

    def test_signatures(self):
        self.assertEqual(
            str(inspect.signature(traceback.print_exception)),
            ('(exc, /, value=<implicit>, tb=<implicit>, '
             'limit=None, file=None, chain=True, **kwargs)'))

        self.assertEqual(
            str(inspect.signature(traceback.format_exception)),
            ('(exc, /, value=<implicit>, tb=<implicit>, limit=None, '
             'chain=True, **kwargs)'))

        self.assertEqual(
            str(inspect.signature(traceback.format_exception_only)),
            '(exc, /, value=<implicit>, *, show_group=False, **kwargs)')


class PurePythonExceptionFormattingMixin:
    def get_exception(self, callable, slice_start=0, slice_end=-1):
        try:
            callable()
        except BaseException:
            return traceback.format_exc().splitlines()[slice_start:slice_end]
        else:
            self.fail("No exception thrown.")

    callable_line = get_exception.__code__.co_firstlineno + 2


class CAPIExceptionFormattingMixin:
    LEGACY = 0

    def get_exception(self, callable, slice_start=0, slice_end=-1):
        from _testcapi import exception_print
        try:
            callable()
            self.fail("No exception thrown.")
        except Exception as e:
            with captured_output("stderr") as tbstderr:
                exception_print(e, self.LEGACY)
            return tbstderr.getvalue().splitlines()[slice_start:slice_end]

    callable_line = get_exception.__code__.co_firstlineno + 3

class CAPIExceptionFormattingLegacyMixin(CAPIExceptionFormattingMixin):
    LEGACY = 1

@requires_debug_ranges()
class TracebackErrorLocationCaretTestBase:
    """
    Tests for printing code error expressions as part of PEP 657
    """
    def test_basic_caret(self):
        # NOTE: In caret tests, "if True:" is used as a way to force indicator
        #   display, since the raising expression spans only part of the line.
        def f():
            if True: raise ValueError("basic caret tests")

        lineno_f = f.__code__.co_firstlineno
        expected_f = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+1}, in f\n'
            '    if True: raise ValueError("basic caret tests")\n'
            '             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
        )
        result_lines = self.get_exception(f)
        self.assertEqual(result_lines, expected_f.splitlines())

    def test_line_with_unicode(self):
        # Make sure that even if a line contains multi-byte unicode characters
        # the correct carets are printed.
        def f_with_unicode():
            if True: raise ValueError("Ĥellö Wörld")

        lineno_f = f_with_unicode.__code__.co_firstlineno
        expected_f = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+1}, in f_with_unicode\n'
            '    if True: raise ValueError("Ĥellö Wörld")\n'
            '             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
        )
        result_lines = self.get_exception(f_with_unicode)
        self.assertEqual(result_lines, expected_f.splitlines())

    def test_caret_in_type_annotation(self):
        def f_with_type():
            def foo(a: THIS_DOES_NOT_EXIST ) -> int:
                return 0

        lineno_f = f_with_type.__code__.co_firstlineno
        expected_f = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+1}, in f_with_type\n'
            '    def foo(a: THIS_DOES_NOT_EXIST ) -> int:\n'
            '               ^^^^^^^^^^^^^^^^^^^\n'
        )
        result_lines = self.get_exception(f_with_type)
        self.assertEqual(result_lines, expected_f.splitlines())

    def test_caret_multiline_expression(self):
        # Make sure no carets are printed for expressions spanning multiple
        # lines.
        def f_with_multiline():
            if True: raise ValueError(
                "error over multiple lines"
            )

        lineno_f = f_with_multiline.__code__.co_firstlineno
        expected_f = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+1}, in f_with_multiline\n'
            '    if True: raise ValueError(\n'
            '             ^^^^^^^^^^^^^^^^^\n'
            '        "error over multiple lines"\n'
            '        ^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '    )\n'
            '    ^'
        )
        result_lines = self.get_exception(f_with_multiline)
        self.assertEqual(result_lines, expected_f.splitlines())

    def test_caret_multiline_expression_syntax_error(self):
        # Make sure an expression spanning multiple lines that has
        # a syntax error is correctly marked with carets.
        code = textwrap.dedent("""
        def foo(*args, **kwargs):
            pass

        a, b, c = 1, 2, 3

        foo(a, z
                for z in
                    range(10), b, c)
        """)

        def f_with_multiline():
            # Need to defer the compilation until in self.get_exception(..)
            return compile(code, "?", "exec")

        lineno_f = f_with_multiline.__code__.co_firstlineno

        expected_f = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_multiline\n'
            '    return compile(code, "?", "exec")\n'
            '  File "?", line 7\n'
            '    foo(a, z\n'
            '           ^'
            )

        result_lines = self.get_exception(f_with_multiline)
        self.assertEqual(result_lines, expected_f.splitlines())

        # Check custom error messages covering multiple lines
        code = textwrap.dedent("""
        dummy_call(
            "dummy value"
            foo="bar",
        )
        """)

        def f_with_multiline():
            # Need to defer the compilation until in self.get_exception(..)
            return compile(code, "?", "exec")

        lineno_f = f_with_multiline.__code__.co_firstlineno

        expected_f = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_multiline\n'
            '    return compile(code, "?", "exec")\n'
            '  File "?", line 3\n'
            '    "dummy value"\n'
            '    ^^^^^^^^^^^^^'
            )

        result_lines = self.get_exception(f_with_multiline)
        self.assertEqual(result_lines, expected_f.splitlines())

    def test_caret_multiline_expression_bin_op(self):
        # Make sure no carets are printed for expressions spanning multiple
        # lines.
        def f_with_multiline():
            return (
                2 + 1 /
                0
            )

        lineno_f = f_with_multiline.__code__.co_firstlineno
        expected_f = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_multiline\n'
            '    2 + 1 /\n'
            '        ~~^\n'
            '    0\n'
            '    ~'
        )
        result_lines = self.get_exception(f_with_multiline)
        self.assertEqual(result_lines, expected_f.splitlines())

    def test_caret_for_binary_operators(self):
        def f_with_binary_operator():
            divisor = 20
            return 10 + divisor / 0 + 30

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_binary_operator\n'
            '    return 10 + divisor / 0 + 30\n'
            '                ~~~~~~~~^~~\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_binary_operators_with_unicode(self):
        def f_with_binary_operator():
            áóí = 20
            return 10 + áóí / 0 + 30

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_binary_operator\n'
            '    return 10 + áóí / 0 + 30\n'
            '                ~~~~^~~\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_binary_operators_two_char(self):
        def f_with_binary_operator():
            divisor = 20
            return 10 + divisor // 0 + 30

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_binary_operator\n'
            '    return 10 + divisor // 0 + 30\n'
            '                ~~~~~~~~^^~~\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_binary_operators_with_spaces_and_parenthesis(self):
        def f_with_binary_operator():
            a = 1
            b = c = ""
            return ( a   )   +b + c

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+3}, in f_with_binary_operator\n'
            '    return ( a   )   +b + c\n'
            '           ~~~~~~~~~~^~\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_binary_operators_multiline(self):
        def f_with_binary_operator():
            b = 1
            c = ""
            a = b    \
         +\
               c  # test
            return a

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+3}, in f_with_binary_operator\n'
            '       a = b    \\\n'
            '           ~~~~~~\n'
            '    +\\\n'
            '    ^~\n'
            '          c  # test\n'
            '          ~\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_binary_operators_multiline_two_char(self):
        def f_with_binary_operator():
            b = 1
            c = ""
            a = (
                (b  # test +
                    )  \
                # +
            << (c  # test
                \
            )  # test
            )
            return a

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+4}, in f_with_binary_operator\n'
            '        (b  # test +\n'
            '        ~~~~~~~~~~~~\n'
            '            )  \\\n'
            '            ~~~~\n'
            '        # +\n'
            '        ~~~\n'
            '    << (c  # test\n'
            '    ^^~~~~~~~~~~~\n'
            '        \\\n'
            '        ~\n'
            '    )  # test\n'
            '    ~\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_binary_operators_multiline_with_unicode(self):
        def f_with_binary_operator():
            b = 1
            a = ("ááá" +
                "áá") + b
            return a

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_binary_operator\n'
            '    a = ("ááá" +\n'
            '        ~~~~~~~~\n'
            '        "áá") + b\n'
            '        ~~~~~~^~~\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_subscript(self):
        def f_with_subscript():
            some_dict = {'x': {'y': None}}
            return some_dict['x']['y']['z']

        lineno_f = f_with_subscript.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_subscript\n'
            "    return some_dict['x']['y']['z']\n"
            '           ~~~~~~~~~~~~~~~~~~~^^^^^\n'
        )
        result_lines = self.get_exception(f_with_subscript)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_subscript_unicode(self):
        def f_with_subscript():
            some_dict = {'ó': {'á': {'í': {'theta': 1}}}}
            return some_dict['ó']['á']['í']['beta']

        lineno_f = f_with_subscript.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_subscript\n'
            "    return some_dict['ó']['á']['í']['beta']\n"
            '           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^\n'
        )
        result_lines = self.get_exception(f_with_subscript)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_subscript_with_spaces_and_parenthesis(self):
        def f_with_binary_operator():
            a = []
            b = c = 1
            return b     [    a  ] + c

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+3}, in f_with_binary_operator\n'
            '    return b     [    a  ] + c\n'
            '           ~~~~~~^^^^^^^^^\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_subscript_multiline(self):
        def f_with_subscript():
            bbbbb = {}
            ccc = 1
            ddd = 2
            b = bbbbb \
                [  ccc # test

                 + ddd  \

                ] # test
            return b

        lineno_f = f_with_subscript.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+4}, in f_with_subscript\n'
            '    b = bbbbb \\\n'
            '        ~~~~~~~\n'
            '        [  ccc # test\n'
            '        ^^^^^^^^^^^^^\n'
            '    \n'
            '    \n'
            '         + ddd  \\\n'
            '         ^^^^^^^^\n'
            '    \n'
            '    \n'
            '        ] # test\n'
            '        ^\n'
        )
        result_lines = self.get_exception(f_with_subscript)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_call(self):
        def f_with_call():
            def f1(a):
                def f2(b):
                    raise RuntimeError("fail")
                return f2
            return f1("x")("y")("z")

        lineno_f = f_with_call.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+5}, in f_with_call\n'
            '    return f1("x")("y")("z")\n'
            '           ~~~~~~~^^^^^\n'
            f'  File "{__file__}", line {lineno_f+3}, in f2\n'
            '    raise RuntimeError("fail")\n'
        )
        result_lines = self.get_exception(f_with_call)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_call_unicode(self):
        def f_with_call():
            def f1(a):
                def f2(b):
                    raise RuntimeError("fail")
                return f2
            return f1("ó")("á")

        lineno_f = f_with_call.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+5}, in f_with_call\n'
            '    return f1("ó")("á")\n'
            '           ~~~~~~~^^^^^\n'
            f'  File "{__file__}", line {lineno_f+3}, in f2\n'
            '    raise RuntimeError("fail")\n'
        )
        result_lines = self.get_exception(f_with_call)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_call_with_spaces_and_parenthesis(self):
        def f_with_binary_operator():
            def f(a):
                raise RuntimeError("fail")
            return f     (    "x"  ) + 2

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+3}, in f_with_binary_operator\n'
            '    return f     (    "x"  ) + 2\n'
            '           ~~~~~~^^^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f\n'
            '    raise RuntimeError("fail")\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_for_call_multiline(self):
        def f_with_call():
            class C:
                def y(self, a):
                    def f(b):
                        raise RuntimeError("fail")
                    return f
            def g(x):
                return C()
            a = (g(1).y)(
                2
            )(3)(4)
            return a

        lineno_f = f_with_call.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+8}, in f_with_call\n'
            '    a = (g(1).y)(\n'
            '        ~~~~~~~~~\n'
            '        2\n'
            '        ~\n'
            '    )(3)(4)\n'
            '    ~^^^\n'
            f'  File "{__file__}", line {lineno_f+4}, in f\n'
            '    raise RuntimeError("fail")\n'
        )
        result_lines = self.get_exception(f_with_call)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_many_lines(self):
        def f():
            x = 1
            if True: x += (
                "a" +
                "a"
            )  # test

        lineno_f = f.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f\n'
            '    if True: x += (\n'
            '             ^^^^^^\n'
            '    ...<2 lines>...\n'
            '    )  # test\n'
            '    ^\n'
        )
        result_lines = self.get_exception(f)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_many_lines_no_caret(self):
        def f():
            x = 1
            x += (
                "a" +
                "a"
            )

        lineno_f = f.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f\n'
            '    x += (\n'
            '    ...<2 lines>...\n'
            '    )\n'
        )
        result_lines = self.get_exception(f)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_many_lines_binary_op(self):
        def f_with_binary_operator():
            b = 1
            c = "a"
            a = (
                b +
                b
            ) + (
                c +
                c +
                c
            )
            return a

        lineno_f = f_with_binary_operator.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+3}, in f_with_binary_operator\n'
            '    a = (\n'
            '        ~\n'
            '        b +\n'
            '        ~~~\n'
            '        b\n'
            '        ~\n'
            '    ) + (\n'
            '    ~~^~~\n'
            '        c +\n'
            '        ~~~\n'
            '    ...<2 lines>...\n'
            '    )\n'
            '    ~\n'
        )
        result_lines = self.get_exception(f_with_binary_operator)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_traceback_specialization_with_syntax_error(self):
        bytecode = compile("1 / 0 / 1 / 2\n", TESTFN, "exec")

        with open(TESTFN, "w") as file:
            # make the file's contents invalid
            file.write("1 $ 0 / 1 / 2\n")
        self.addCleanup(unlink, TESTFN)

        func = partial(exec, bytecode)
        result_lines = self.get_exception(func)

        lineno_f = bytecode.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{TESTFN}", line {lineno_f}, in <module>\n'
            "    1 $ 0 / 1 / 2\n"
            '    ^^^^^\n'
        )
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_traceback_very_long_line(self):
        source = "if True: " + "a" * 256
        bytecode = compile(source, TESTFN, "exec")

        with open(TESTFN, "w") as file:
            file.write(source)
        self.addCleanup(unlink, TESTFN)

        func = partial(exec, bytecode)
        result_lines = self.get_exception(func)

        lineno_f = bytecode.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{TESTFN}", line {lineno_f}, in <module>\n'
            f'    {source}\n'
            f'    {" "*len("if True: ") + "^"*256}\n'
        )
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_secondary_caret_not_elided(self):
        # Always show a line's indicators if they include the secondary character.
        def f_with_subscript():
            some_dict = {'x': {'y': None}}
            some_dict['x']['y']['z']

        lineno_f = f_with_subscript.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_f+2}, in f_with_subscript\n'
            "    some_dict['x']['y']['z']\n"
            '    ~~~~~~~~~~~~~~~~~~~^^^^^\n'
        )
        result_lines = self.get_exception(f_with_subscript)
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_caret_exception_group(self):
        # Notably, this covers whether indicators handle margin strings correctly.
        # (Exception groups use margin strings to display vertical indicators.)
        # The implementation must account for both "indent" and "margin" offsets.

        def exc():
            if True: raise ExceptionGroup("eg", [ValueError(1), TypeError(2)])

        expected_error = (
             f'  + Exception Group Traceback (most recent call last):\n'
             f'  |   File "{__file__}", line {self.callable_line}, in get_exception\n'
             f'  |     callable()\n'
             f'  |     ~~~~~~~~^^\n'
             f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 1}, in exc\n'
             f'  |     if True: raise ExceptionGroup("eg", [ValueError(1), TypeError(2)])\n'
             f'  |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
             f'  | ExceptionGroup: eg (2 sub-exceptions)\n'
             f'  +-+---------------- 1 ----------------\n'
             f'    | ValueError: 1\n'
             f'    +---------------- 2 ----------------\n'
             f'    | TypeError: 2\n')

        result_lines = self.get_exception(exc)
        self.assertEqual(result_lines, expected_error.splitlines())

    def assertSpecialized(self, func, expected_specialization):
        result_lines = self.get_exception(func)
        specialization_line = result_lines[-1]
        self.assertEqual(specialization_line.lstrip(), expected_specialization)

    def test_specialization_variations(self):
        self.assertSpecialized(lambda: 1/0,
                                      "~^~")
        self.assertSpecialized(lambda: 1/0/3,
                                      "~^~")
        self.assertSpecialized(lambda: 1 / 0,
                                      "~~^~~")
        self.assertSpecialized(lambda: 1 / 0 / 3,
                                      "~~^~~")
        self.assertSpecialized(lambda: 1/ 0,
                                      "~^~~")
        self.assertSpecialized(lambda: 1/ 0/3,
                                      "~^~~")
        self.assertSpecialized(lambda: 1    /  0,
                                      "~~~~~^~~~")
        self.assertSpecialized(lambda: 1    /  0   / 5,
                                      "~~~~~^~~~")
        self.assertSpecialized(lambda: 1 /0,
                                      "~~^~")
        self.assertSpecialized(lambda: 1//0,
                                      "~^^~")
        self.assertSpecialized(lambda: 1//0//4,
                                      "~^^~")
        self.assertSpecialized(lambda: 1 // 0,
                                      "~~^^~~")
        self.assertSpecialized(lambda: 1 // 0 // 4,
                                      "~~^^~~")
        self.assertSpecialized(lambda: 1 //0,
                                      "~~^^~")
        self.assertSpecialized(lambda: 1// 0,
                                      "~^^~~")

    def test_decorator_application_lineno_correct(self):
        def dec_error(func):
            raise TypeError
        def dec_fine(func):
            return func
        def applydecs():
            @dec_error
            @dec_fine
            def g(): pass
        result_lines = self.get_exception(applydecs)
        lineno_applydescs = applydecs.__code__.co_firstlineno
        lineno_dec_error = dec_error.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_applydescs + 1}, in applydecs\n'
            '    @dec_error\n'
            '     ^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_dec_error + 1}, in dec_error\n'
            '    raise TypeError\n'
        )
        self.assertEqual(result_lines, expected_error.splitlines())

        def applydecs_class():
            @dec_error
            @dec_fine
            class A: pass
        result_lines = self.get_exception(applydecs_class)
        lineno_applydescs_class = applydecs_class.__code__.co_firstlineno
        expected_error = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
            '    callable()\n'
            '    ~~~~~~~~^^\n'
            f'  File "{__file__}", line {lineno_applydescs_class + 1}, in applydecs_class\n'
            '    @dec_error\n'
            '     ^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_dec_error + 1}, in dec_error\n'
            '    raise TypeError\n'
        )
        self.assertEqual(result_lines, expected_error.splitlines())

    def test_multiline_method_call_a(self):
        def f():
            (None
                .method
            )()
        actual = self.get_exception(f)
        expected = [
            "Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 2}, in f",
            "    .method",
            "     ^^^^^^",
        ]
        self.assertEqual(actual, expected)

    def test_multiline_method_call_b(self):
        def f():
            (None.
                method
            )()
        actual = self.get_exception(f)
        expected = [
            "Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 2}, in f",
            "    method",
        ]
        self.assertEqual(actual, expected)

    def test_multiline_method_call_c(self):
        def f():
            (None
                . method
            )()
        actual = self.get_exception(f)
        expected = [
            "Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 2}, in f",
            "    . method",
            "      ^^^^^^",
        ]
        self.assertEqual(actual, expected)

    def test_wide_characters_unicode_with_problematic_byte_offset(self):
        def f():
            ｗｉｄｔｈ

        actual = self.get_exception(f)
        expected = [
            "Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    ｗｉｄｔｈ",
        ]
        self.assertEqual(actual, expected)


    def test_byte_offset_with_wide_characters_middle(self):
        def f():
            ｗｉｄｔｈ = 1
            raise ValueError(ｗｉｄｔｈ)

        actual = self.get_exception(f)
        expected = [
            "Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 2}, in f",
            "    raise ValueError(ｗｉｄｔｈ)",
        ]
        self.assertEqual(actual, expected)

    def test_byte_offset_multiline(self):
        def f():
            ｗｗｗ = 1
            ｔｈ = 0

            print(1, ｗｗｗ(
                    ｔｈ))

        actual = self.get_exception(f)
        expected = [
            "Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 4}, in f",
            f"    print(1, ｗｗｗ(",
            f"             ~~~~~~^",
            f"            ｔｈ))",
            f"            ^^^^^",
        ]
        self.assertEqual(actual, expected)

    def test_byte_offset_with_wide_characters_term_highlight(self):
        def f():
            说明说明 = 1
            şçöğıĤellö = 0 # not wide but still non-ascii
            return 说明说明 / şçöğıĤellö

        actual = self.get_exception(f)
        expected = [
            f"Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            f"    callable()",
            f"    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 3}, in f",
            f"    return 说明说明 / şçöğıĤellö",
            f"           ~~~~~~~~~^~~~~~~~~~~~",
        ]
        self.assertEqual(actual, expected)

    def test_byte_offset_with_emojis_term_highlight(self):
        def f():
            return "✨🐍" + func_说明说明("📗🚛",
                "📗🚛") + "🐍"

        actual = self.get_exception(f)
        expected = [
            f"Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            f"    callable()",
            f"    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            f'    return "✨🐍" + func_说明说明("📗🚛",',
            f"                    ^^^^^^^^^^^^^",
        ]
        self.assertEqual(actual, expected)

    def test_byte_offset_wide_chars_subscript(self):
        def f():
            my_dct = {
                "✨🚛✨": {
                    "说明": {
                        "🐍🐍🐍": None
                    }
                }
            }
            return my_dct["✨🚛✨"]["说明"]["🐍"]["说明"]["🐍🐍"]

        actual = self.get_exception(f)
        expected = [
            f"Traceback (most recent call last):",
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            f"    callable()",
            f"    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 8}, in f",
            f'    return my_dct["✨🚛✨"]["说明"]["🐍"]["说明"]["🐍🐍"]',
            f"           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^",
        ]
        self.assertEqual(actual, expected)

    def test_memory_error(self):
        def f():
            raise MemoryError()

        actual = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f'  File "{__file__}", line {self.callable_line}, in get_exception',
            '    callable()',
            '    ~~~~~~~~^^',
            f'  File "{__file__}", line {f.__code__.co_firstlineno + 1}, in f',
            '    raise MemoryError()']
        self.assertEqual(actual, expected)

    def test_anchors_for_simple_return_statements_are_elided(self):
        def g():
            1/0

        def f():
            return g()

        result_lines = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    return g()",
            f"  File \"{__file__}\", line {g.__code__.co_firstlineno + 1}, in g",
            "    1/0",
            "    ~^~"
        ]
        self.assertEqual(result_lines, expected)

        def g():
            1/0

        def f():
            return g() + 1

        result_lines = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    return g() + 1",
            "           ~^^",
            f"  File \"{__file__}\", line {g.__code__.co_firstlineno + 1}, in g",
            "    1/0",
            "    ~^~"
        ]
        self.assertEqual(result_lines, expected)

        def g(*args):
            1/0

        def f():
            return g(1,
                     2, 4,
                     5)

        result_lines = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    return g(1,",
            "             2, 4,",
            "             5)",
            f"  File \"{__file__}\", line {g.__code__.co_firstlineno + 1}, in g",
            "    1/0",
            "    ~^~"
        ]
        self.assertEqual(result_lines, expected)

        def g(*args):
            1/0

        def f():
            return g(1,
                     2, 4,
                     5) + 1

        result_lines = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    return g(1,",
            "           ~^^^",
            "             2, 4,",
            "             ^^^^^",
            "             5) + 1",
            "             ^^",
            f"  File \"{__file__}\", line {g.__code__.co_firstlineno + 1}, in g",
            "    1/0",
            "    ~^~"
        ]
        self.assertEqual(result_lines, expected)

    def test_anchors_for_simple_assign_statements_are_elided(self):
        def g():
            1/0

        def f():
            x = g()

        result_lines = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    x = g()",
            f"  File \"{__file__}\", line {g.__code__.co_firstlineno + 1}, in g",
            "    1/0",
            "    ~^~"
        ]
        self.assertEqual(result_lines, expected)

        def g(*args):
            1/0

        def f():
            x = g(1,
                  2, 3,
                  4)

        result_lines = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    x = g(1,",
            "          2, 3,",
            "          4)",
            f"  File \"{__file__}\", line {g.__code__.co_firstlineno + 1}, in g",
            "    1/0",
            "    ~^~"
        ]
        self.assertEqual(result_lines, expected)

        def g():
            1/0

        def f():
            x = y = g()

        result_lines = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    x = y = g()",
            "            ~^^",
            f"  File \"{__file__}\", line {g.__code__.co_firstlineno + 1}, in g",
            "    1/0",
            "    ~^~"
        ]
        self.assertEqual(result_lines, expected)

        def g(*args):
            1/0

        def f():
            x = y = g(1,
                      2, 3,
                      4)

        result_lines = self.get_exception(f)
        expected = ['Traceback (most recent call last):',
            f"  File \"{__file__}\", line {self.callable_line}, in get_exception",
            "    callable()",
            "    ~~~~~~~~^^",
            f"  File \"{__file__}\", line {f.__code__.co_firstlineno + 1}, in f",
            "    x = y = g(1,",
            "            ~^^^",
            "              2, 3,",
            "              ^^^^^",
            "              4)",
            "              ^^",
            f"  File \"{__file__}\", line {g.__code__.co_firstlineno + 1}, in g",
            "    1/0",
            "    ~^~"
        ]
        self.assertEqual(result_lines, expected)


@requires_debug_ranges()
@force_not_colorized_test_class
class PurePythonTracebackErrorCaretTests(
    PurePythonExceptionFormattingMixin,
    TracebackErrorLocationCaretTestBase,
    __TestCase,
):
    """
    Same set of tests as above using the pure Python implementation of
    traceback printing in traceback.py.
    """


@cpython_only
@requires_debug_ranges()
@force_not_colorized_test_class
class CPythonTracebackErrorCaretTests(
    CAPIExceptionFormattingMixin,
    TracebackErrorLocationCaretTestBase,
    __TestCase,
):
    """
    Same set of tests as above but with Python's internal traceback printing.
    """

@cpython_only
@requires_debug_ranges()
@force_not_colorized_test_class
class CPythonTracebackLegacyErrorCaretTests(
    CAPIExceptionFormattingLegacyMixin,
    TracebackErrorLocationCaretTestBase,
    __TestCase,
):
    """
    Same set of tests as above but with Python's legacy internal traceback printing.
    """


class TracebackFormatMixin:
    DEBUG_RANGES = True

    def some_exception(self):
        raise KeyError('blah')

    def _filter_debug_ranges(self, expected):
        return [line for line in expected if not set(line.strip()) <= set("^~")]

    def _maybe_filter_debug_ranges(self, expected):
        if not self.DEBUG_RANGES:
            return self._filter_debug_ranges(expected)
        return expected

    @cpython_only
    def check_traceback_format(self, cleanup_func=None):
        from _testcapi import traceback_print
        try:
            self.some_exception()
        except KeyError as e:
            tb = e.__traceback__
            if cleanup_func is not None:
                # Clear the inner frames, not this one
                cleanup_func(tb.tb_next)
            traceback_fmt = 'Traceback (most recent call last):\n' + \
                            ''.join(traceback.format_tb(tb))
            # clear caret lines from traceback_fmt since internal API does
            # not emit them
            traceback_fmt = "\n".join(
                self._filter_debug_ranges(traceback_fmt.splitlines())
            ) + "\n"
            file_ = StringIO()
            traceback_print(tb, file_)
            python_fmt  = file_.getvalue()
            # Call all _tb and _exc functions
            with captured_output("stderr") as tbstderr:
                traceback.print_tb(tb)
            tbfile = StringIO()
            traceback.print_tb(tb, file=tbfile)
            with captured_output("stderr") as excstderr:
                traceback.print_exc()
            excfmt = traceback.format_exc()
            excfile = StringIO()
            traceback.print_exc(file=excfile)
        else:
            raise Error("unable to create test traceback string")

        # Make sure that Python and the traceback module format the same thing
        self.assertEqual(traceback_fmt, python_fmt)
        # Now verify the _tb func output
        self.assertEqual(tbstderr.getvalue(), tbfile.getvalue())
        # Now verify the _exc func output
        self.assertEqual(excstderr.getvalue(), excfile.getvalue())
        self.assertEqual(excfmt, excfile.getvalue())

        # Make sure that the traceback is properly indented.
        tb_lines = python_fmt.splitlines()
        banner = tb_lines[0]
        self.assertEqual(len(tb_lines), 5)
        location, source_line = tb_lines[-2], tb_lines[-1]
        self.assertTrue(banner.startswith('Traceback'))
        self.assertTrue(location.startswith('  File'))
        self.assertTrue(source_line.startswith('    raise'))

    def test_traceback_format(self):
        self.check_traceback_format()

    def test_traceback_format_with_cleared_frames(self):
        # Check that traceback formatting also works with a clear()ed frame
        def cleanup_tb(tb):
            tb.tb_frame.clear()
        self.check_traceback_format(cleanup_tb)

    def test_stack_format(self):
        # Verify _stack functions. Note we have to use _getframe(1) to
        # compare them without this frame appearing in the output
        with captured_output("stderr") as ststderr:
            traceback.print_stack(sys._getframe(1))
        stfile = StringIO()
        traceback.print_stack(sys._getframe(1), file=stfile)
        self.assertEqual(ststderr.getvalue(), stfile.getvalue())

        stfmt = traceback.format_stack(sys._getframe(1))

        self.assertEqual(ststderr.getvalue(), "".join(stfmt))

    def test_print_stack(self):
        def prn():
            traceback.print_stack()
        with captured_output("stderr") as stderr:
            prn()
        lineno = prn.__code__.co_firstlineno
        self.assertEqual(stderr.getvalue().splitlines()[-4:], [
            '  File "%s", line %d, in test_print_stack' % (__file__, lineno+3),
            '    prn()',
            '  File "%s", line %d, in prn' % (__file__, lineno+1),
            '    traceback.print_stack()',
        ])

    # issue 26823 - Shrink recursive tracebacks
    def _check_recursive_traceback_display(self, render_exc):
        # Always show full diffs when this test fails
        # Note that rearranging things may require adjusting
        # the relative line numbers in the expected tracebacks
        self.maxDiff = None

        # Check hitting the recursion limit
        def f():
            f()

        with captured_output("stderr") as stderr_f:
            try:
                f()
            except RecursionError:
                render_exc()
            else:
                self.fail("no recursion occurred")

        lineno_f = f.__code__.co_firstlineno
        result_f = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {lineno_f+5}, in _check_recursive_traceback_display\n'
            '    f()\n'
            '    ~^^\n'
            f'  File "{__file__}", line {lineno_f+1}, in f\n'
            '    f()\n'
            '    ~^^\n'
            f'  File "{__file__}", line {lineno_f+1}, in f\n'
            '    f()\n'
            '    ~^^\n'
            f'  File "{__file__}", line {lineno_f+1}, in f\n'
            '    f()\n'
            '    ~^^\n'
            # XXX: The following line changes depending on whether the tests
            # are run through the interactive interpreter or with -m
            # It also varies depending on the platform (stack size)
            # Fortunately, we don't care about exactness here, so we use regex
            r'  \[Previous line repeated (\d+) more times\]' '\n'
            'RecursionError: maximum recursion depth exceeded\n'
        )

        expected = self._maybe_filter_debug_ranges(result_f.splitlines())
        actual = stderr_f.getvalue().splitlines()

        # Check the output text matches expectations
        # 2nd last line contains the repetition count
        self.assertEqual(actual[:-2], expected[:-2])
        self.assertRegex(actual[-2], expected[-2])
        # last line can have additional text appended
        self.assertIn(expected[-1], actual[-1])

        # Check the recursion count is roughly as expected
        rec_limit = sys.getrecursionlimit()
        self.assertIn(int(re.search(r"\d+", actual[-2]).group()), range(rec_limit-60, rec_limit))

        # Check a known (limited) number of recursive invocations
        def g(count=10):
            if count:
                return g(count-1) + 1
            raise ValueError

        with captured_output("stderr") as stderr_g:
            try:
                g()
            except ValueError:
                render_exc()
            else:
                self.fail("no value error was raised")

        lineno_g = g.__code__.co_firstlineno
        result_g = (
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            '  [Previous line repeated 7 more times]\n'
            f'  File "{__file__}", line {lineno_g+3}, in g\n'
            '    raise ValueError\n'
            'ValueError\n'
        )
        tb_line = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {lineno_g+7}, in _check_recursive_traceback_display\n'
            '    g()\n'
            '    ~^^\n'
        )
        expected = self._maybe_filter_debug_ranges((tb_line + result_g).splitlines())
        actual = stderr_g.getvalue().splitlines()
        self.assertEqual(actual, expected)

        # Check 2 different repetitive sections
        def h(count=10):
            if count:
                return h(count-1)
            g()

        with captured_output("stderr") as stderr_h:
            try:
                h()
            except ValueError:
                render_exc()
            else:
                self.fail("no value error was raised")

        lineno_h = h.__code__.co_firstlineno
        result_h = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {lineno_h+7}, in _check_recursive_traceback_display\n'
            '    h()\n'
            '    ~^^\n'
            f'  File "{__file__}", line {lineno_h+2}, in h\n'
            '    return h(count-1)\n'
            f'  File "{__file__}", line {lineno_h+2}, in h\n'
            '    return h(count-1)\n'
            f'  File "{__file__}", line {lineno_h+2}, in h\n'
            '    return h(count-1)\n'
            '  [Previous line repeated 7 more times]\n'
            f'  File "{__file__}", line {lineno_h+3}, in h\n'
            '    g()\n'
            '    ~^^\n'
        )
        expected = self._maybe_filter_debug_ranges((result_h + result_g).splitlines())
        actual = stderr_h.getvalue().splitlines()
        self.assertEqual(actual, expected)

        # Check the boundary conditions. First, test just below the cutoff.
        with captured_output("stderr") as stderr_g:
            try:
                g(traceback._RECURSIVE_CUTOFF)
            except ValueError:
                render_exc()
            else:
                self.fail("no error raised")
        result_g = (
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_g+3}, in g\n'
            '    raise ValueError\n'
            'ValueError\n'
        )
        tb_line = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {lineno_g+77}, in _check_recursive_traceback_display\n'
            '    g(traceback._RECURSIVE_CUTOFF)\n'
            '    ~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
        )
        expected = self._maybe_filter_debug_ranges((tb_line + result_g).splitlines())
        actual = stderr_g.getvalue().splitlines()
        self.assertEqual(actual, expected)

        # Second, test just above the cutoff.
        with captured_output("stderr") as stderr_g:
            try:
                g(traceback._RECURSIVE_CUTOFF + 1)
            except ValueError:
                render_exc()
            else:
                self.fail("no error raised")
        result_g = (
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            f'  File "{__file__}", line {lineno_g+2}, in g\n'
            '    return g(count-1) + 1\n'
            '           ~^^^^^^^^^\n'
            '  [Previous line repeated 1 more time]\n'
            f'  File "{__file__}", line {lineno_g+3}, in g\n'
            '    raise ValueError\n'
            'ValueError\n'
        )
        tb_line = (
            'Traceback (most recent call last):\n'
            f'  File "{__file__}", line {lineno_g+109}, in _check_recursive_traceback_display\n'
            '    g(traceback._RECURSIVE_CUTOFF + 1)\n'
            '    ~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
        )
        expected = self._maybe_filter_debug_ranges((tb_line + result_g).splitlines())
        actual = stderr_g.getvalue().splitlines()
        self.assertEqual(actual, expected)

    @requires_debug_ranges()
    def test_recursive_traceback(self):
        if self.DEBUG_RANGES:
            self._check_recursive_traceback_display(traceback.print_exc)
        else:
            from _testcapi import exception_print
            def render_exc():
                exception_print(sys.exception())
            self._check_recursive_traceback_display(render_exc)

    def test_format_stack(self):
        def fmt():
            return traceback.format_stack()
        result = fmt()
        lineno = fmt.__code__.co_firstlineno
        self.assertEqual(result[-2:], [
            '  File "%s", line %d, in test_format_stack\n'
            '    result = fmt()\n' % (__file__, lineno+2),
            '  File "%s", line %d, in fmt\n'
            '    return traceback.format_stack()\n' % (__file__, lineno+1),
        ])

    @cpython_only
    def test_unhashable(self):
        from _testcapi import exception_print

        with torch._dynamo.error_on_graph_break(False):
            class UnhashableException(Exception):
                def __eq__(self, other):
                    return True

        ex1 = UnhashableException('ex1')
        ex2 = UnhashableException('ex2')
        try:
            raise ex2 from ex1
        except UnhashableException:
            try:
                raise ex1
            except UnhashableException as e:
                exc_val = e

        with captured_output("stderr") as stderr_f:
            exception_print(exc_val)

        tb = stderr_f.getvalue().strip().splitlines()
        self.assertEqual(11, len(tb))
        self.assertEqual(context_message.strip(), tb[5])
        self.assertIn('UnhashableException: ex2', tb[3])
        self.assertIn('UnhashableException: ex1', tb[10])

    def deep_eg(self):
        e = TypeError(1)
        for i in range(2000):
            e = ExceptionGroup('eg', [e])
        return e

    @cpython_only
    def test_exception_group_deep_recursion_capi(self):
        from _testcapi import exception_print
        LIMIT = 75
        eg = self.deep_eg()
        with captured_output("stderr") as stderr_f:
            with support.infinite_recursion(max_depth=LIMIT):
                exception_print(eg)
        output = stderr_f.getvalue()
        self.assertIn('ExceptionGroup', output)
        self.assertLessEqual(output.count('ExceptionGroup'), LIMIT)

    def test_exception_group_deep_recursion_traceback(self):
        LIMIT = 75
        eg = self.deep_eg()
        with captured_output("stderr") as stderr_f:
            with support.infinite_recursion(max_depth=LIMIT):
                traceback.print_exception(type(eg), eg, eg.__traceback__)
        output = stderr_f.getvalue()
        self.assertIn('ExceptionGroup', output)
        self.assertLessEqual(output.count('ExceptionGroup'), LIMIT)

    @cpython_only
    def test_print_exception_bad_type_capi(self):
        from _testcapi import exception_print
        with captured_output("stderr") as stderr:
            with support.catch_unraisable_exception():
                exception_print(42)
        self.assertEqual(
            stderr.getvalue(),
            ('TypeError: print_exception(): '
             'Exception expected for value, int found\n')
        )

    def test_print_exception_bad_type_python(self):
        msg = "Exception expected for value, int found"
        with self.assertRaisesRegex(TypeError, msg):
            traceback.print_exception(42)


cause_message = (
    "\nThe above exception was the direct cause "
    "of the following exception:\n\n")

context_message = (
    "\nDuring handling of the above exception, "
    "another exception occurred:\n\n")

boundaries = re.compile(
    '(%s|%s)' % (re.escape(cause_message), re.escape(context_message)))

@force_not_colorized_test_class
class TestTracebackFormat(__TestCase, TracebackFormatMixin):
    pass

@cpython_only
@force_not_colorized_test_class
class TestFallbackTracebackFormat(__TestCase, TracebackFormatMixin):
    DEBUG_RANGES = False
    def setUp(self) -> None:
        self.original_unraisable_hook = sys.unraisablehook
        sys.unraisablehook = lambda *args: None
        self.original_hook = traceback._print_exception_bltin
        traceback._print_exception_bltin = lambda *args: 1/0
        return super().setUp()

    def tearDown(self) -> None:
        traceback._print_exception_bltin = self.original_hook
        sys.unraisablehook = self.original_unraisable_hook
        return super().tearDown()

class BaseExceptionReportingTests:

    def get_exception(self, exception_or_callable):
        if isinstance(exception_or_callable, BaseException):
            return exception_or_callable
        try:
            exception_or_callable()
        except Exception as e:
            return e

    callable_line = get_exception.__code__.co_firstlineno + 4

    def zero_div(self):
        1/0 # In zero_div

    def check_zero_div(self, msg):
        lines = msg.splitlines()
        if has_no_debug_ranges():
            self.assertTrue(lines[-3].startswith('  File'))
            self.assertIn('1/0 # In zero_div', lines[-2])
        else:
            self.assertTrue(lines[-4].startswith('  File'))
            self.assertIn('1/0 # In zero_div', lines[-3])
        self.assertTrue(lines[-1].startswith('ZeroDivisionError'), lines[-1])

    def test_simple(self):
        try:
            1/0 # Marker
        except ZeroDivisionError as _:
            e = _
        lines = self.get_report(e).splitlines()
        if has_no_debug_ranges():
            self.assertEqual(len(lines), 4)
            self.assertTrue(lines[3].startswith('ZeroDivisionError'))
        else:
            self.assertEqual(len(lines), 5)
            self.assertTrue(lines[4].startswith('ZeroDivisionError'))
        self.assertTrue(lines[0].startswith('Traceback'))
        self.assertTrue(lines[1].startswith('  File'))
        self.assertIn('1/0 # Marker', lines[2])

    def test_cause(self):
        def inner_raise():
            try:
                self.zero_div()
            except ZeroDivisionError as e:
                raise KeyError from e
        def outer_raise():
            inner_raise() # Marker
        blocks = boundaries.split(self.get_report(outer_raise))
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[1], cause_message)
        self.check_zero_div(blocks[0])
        self.assertIn('inner_raise() # Marker', blocks[2])

    def test_context(self):
        def inner_raise():
            try:
                self.zero_div()
            except ZeroDivisionError:
                raise KeyError
        def outer_raise():
            inner_raise() # Marker
        blocks = boundaries.split(self.get_report(outer_raise))
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[1], context_message)
        self.check_zero_div(blocks[0])
        self.assertIn('inner_raise() # Marker', blocks[2])

    def test_context_suppression(self):
        try:
            try:
                raise Exception
            except Exception:
                raise ZeroDivisionError from None
        except ZeroDivisionError as _:
            e = _
        lines = self.get_report(e).splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[3].startswith('ZeroDivisionError'))
        self.assertTrue(lines[0].startswith('Traceback'))
        self.assertTrue(lines[1].startswith('  File'))
        self.assertIn('ZeroDivisionError from None', lines[2])

    def test_cause_and_context(self):
        # When both a cause and a context are set, only the cause should be
        # displayed and the context should be muted.
        def inner_raise():
            try:
                self.zero_div()
            except ZeroDivisionError as _e:
                e = _e
            try:
                xyzzy
            except NameError:
                raise KeyError from e
        def outer_raise():
            inner_raise() # Marker
        blocks = boundaries.split(self.get_report(outer_raise))
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[1], cause_message)
        self.check_zero_div(blocks[0])
        self.assertIn('inner_raise() # Marker', blocks[2])

    def test_cause_recursive(self):
        def inner_raise():
            try:
                try:
                    self.zero_div()
                except ZeroDivisionError as e:
                    z = e
                    raise KeyError from e
            except KeyError as e:
                raise z from e
        def outer_raise():
            inner_raise() # Marker
        blocks = boundaries.split(self.get_report(outer_raise))
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[1], cause_message)
        # The first block is the KeyError raised from the ZeroDivisionError
        self.assertIn('raise KeyError from e', blocks[0])
        self.assertNotIn('1/0', blocks[0])
        # The second block (apart from the boundary) is the ZeroDivisionError
        # re-raised from the KeyError
        self.assertIn('inner_raise() # Marker', blocks[2])
        self.check_zero_div(blocks[2])

    def test_syntax_error_offset_at_eol(self):
        # See #10186.
        def e():
            raise SyntaxError('', ('', 0, 5, 'hello'))
        msg = self.get_report(e).splitlines()
        self.assertEqual(msg[-2], "        ^")
        def e():
            exec("x = 5 | 4 |")
        msg = self.get_report(e).splitlines()
        self.assertEqual(msg[-2], '               ^')

    def test_syntax_error_no_lineno(self):
        # See #34463.

        # Without filename
        e = SyntaxError('bad syntax')
        msg = self.get_report(e).splitlines()
        self.assertEqual(msg,
            ['SyntaxError: bad syntax'])
        e.lineno = 100
        msg = self.get_report(e).splitlines()
        self.assertEqual(msg,
            ['  File "<string>", line 100', 'SyntaxError: bad syntax'])

        # With filename
        e = SyntaxError('bad syntax')
        e.filename = 'myfile.py'

        msg = self.get_report(e).splitlines()
        self.assertEqual(msg,
            ['SyntaxError: bad syntax (myfile.py)'])
        e.lineno = 100
        msg = self.get_report(e).splitlines()
        self.assertEqual(msg,
            ['  File "myfile.py", line 100', 'SyntaxError: bad syntax'])

    def test_message_none(self):
        # A message that looks like "None" should not be treated specially
        err = self.get_report(Exception(None))
        self.assertIn('Exception: None\n', err)
        err = self.get_report(Exception('None'))
        self.assertIn('Exception: None\n', err)
        err = self.get_report(Exception())
        self.assertIn('Exception\n', err)
        err = self.get_report(Exception(''))
        self.assertIn('Exception\n', err)

    def test_syntax_error_various_offsets(self):
        for offset in range(-5, 10):
            for add in [0, 2]:
                text = " " * add + "text%d" % offset
                expected = ['  File "file.py", line 1']
                if offset < 1:
                    expected.append("    %s" % text.lstrip())
                elif offset <= 6:
                    expected.append("    %s" % text.lstrip())
                    # Set the caret length to match the length of the text minus the offset.
                    caret_length = max(1, len(text.lstrip()) - offset + 1)
                    expected.append("    %s%s" % (" " * (offset - 1), "^" * caret_length))
                else:
                    caret_length = max(1, len(text.lstrip()) - 4)
                    expected.append("    %s" % text.lstrip())
                    expected.append("    %s%s" % (" " * 5, "^" * caret_length))
                expected.append("SyntaxError: msg")
                expected.append("")
                err = self.get_report(SyntaxError("msg", ("file.py", 1, offset + add, text)))
                exp = "\n".join(expected)
                self.assertEqual(exp, err)

    def test_exception_with_note(self):
        e = ValueError(123)
        vanilla = self.get_report(e)

        e.add_note('My Note')
        self.assertEqual(self.get_report(e), vanilla + 'My Note\n')

        del e.__notes__
        e.add_note('')
        self.assertEqual(self.get_report(e), vanilla + '\n')

        del e.__notes__
        e.add_note('Your Note')
        self.assertEqual(self.get_report(e), vanilla + 'Your Note\n')

        del e.__notes__
        self.assertEqual(self.get_report(e), vanilla)

    def test_exception_with_invalid_notes(self):
        e = ValueError(123)
        vanilla = self.get_report(e)

        with torch._dynamo.error_on_graph_break(False):
            # non-sequence __notes__
            class BadThing:
                def __str__(self):
                    return 'bad str'

                def __repr__(self):
                    return 'bad repr'

            # unprintable, non-sequence __notes__
            class Unprintable:
                def __repr__(self):
                    raise ValueError('bad value')

        e.__notes__ = BadThing()
        notes_repr = 'bad repr'
        self.assertEqual(self.get_report(e), vanilla + notes_repr + '\n')

        e.__notes__ = Unprintable()
        err_msg = '<__notes__ repr() failed>'
        self.assertEqual(self.get_report(e), vanilla + err_msg + '\n')

        # non-string item in the __notes__ sequence
        e.__notes__  = [BadThing(), 'Final Note']
        bad_note = 'bad str'
        self.assertEqual(self.get_report(e), vanilla + bad_note + '\nFinal Note\n')

        # unprintable, non-string item in the __notes__ sequence
        e.__notes__  = [Unprintable(), 'Final Note']
        err_msg = '<note str() failed>'
        self.assertEqual(self.get_report(e), vanilla + err_msg + '\nFinal Note\n')

        e.__notes__  = "please do not explode me"
        err_msg = "'please do not explode me'"
        self.assertEqual(self.get_report(e), vanilla + err_msg + '\n')

        e.__notes__  = b"please do not show me as numbers"
        err_msg = "b'please do not show me as numbers'"
        self.assertEqual(self.get_report(e), vanilla + err_msg + '\n')

        with torch._dynamo.error_on_graph_break(False):
            # an exception with a broken __getattr__ raising a non expected error
            class BrokenException(Exception):
                broken = False
                def __getattr__(self, name):
                    if self.broken:
                        raise ValueError(f'no {name}')

        e = BrokenException(123)
        vanilla = self.get_report(e)
        e.broken = True
        self.assertEqual(
            self.get_report(e),
            vanilla + "Ignored error getting __notes__: ValueError('no __notes__')\n")

    def test_exception_with_multiple_notes(self):
        for e in [ValueError(42), SyntaxError('bad syntax')]:
            with self.subTest(e=e):
                vanilla = self.get_report(e)

                e.add_note('Note 1')
                e.add_note('Note 2')
                e.add_note('Note 3')

                self.assertEqual(
                    self.get_report(e),
                    vanilla + 'Note 1\n' + 'Note 2\n' + 'Note 3\n')

                del e.__notes__
                e.add_note('Note 4')
                del e.__notes__
                e.add_note('Note 5')
                e.add_note('Note 6')

                self.assertEqual(
                    self.get_report(e),
                    vanilla + 'Note 5\n' + 'Note 6\n')

    def test_exception_qualname(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                class B:
                    class X(Exception):
                        def __str__(self):
                            return "I am X"

        err = self.get_report(A.B.X())
        str_value = 'I am X'
        str_name = '.'.join([A.B.X.__module__, A.B.X.__qualname__])
        exp = "%s: %s\n" % (str_name, str_value)
        self.assertEqual(exp, MODULE_PREFIX + err)

    def test_exception_modulename(self):
        with torch._dynamo.error_on_graph_break(False):
            class X(Exception):
                def __str__(self):
                    return "I am X"

        for modulename in '__main__', 'builtins', 'some_module':
            X.__module__ = modulename
            with self.subTest(modulename=modulename):
                err = self.get_report(X())
                str_value = 'I am X'
                if modulename in ['builtins', '__main__']:
                    str_name = X.__qualname__
                else:
                    str_name = '.'.join([X.__module__, X.__qualname__])
                exp = "%s: %s\n" % (str_name, str_value)
                self.assertEqual(exp, err)

    def test_exception_angle_bracketed_filename(self):
        src = textwrap.dedent("""
            try:
                raise ValueError(42)
            except Exception as e:
                exc = e
            """)

        code = compile(src, "<does not exist>", "exec")
        g, l = {}, {}
        exec(code, g, l)
        err = self.get_report(l['exc'])
        exp = '  File "<does not exist>", line 3, in <module>\nValueError: 42\n'
        self.assertIn(exp, err)

    def test_exception_modulename_not_unicode(self):
        with torch._dynamo.error_on_graph_break(False):
            class X(Exception):
                def __str__(self):
                    return "I am X"

        X.__module__ = 42

        err = self.get_report(X())
        exp = f'<unknown>.{X.__qualname__}: I am X\n'
        self.assertEqual(exp, err)

    def test_exception_bad__str__(self):
        with torch._dynamo.error_on_graph_break(False):
            class X(Exception):
                def __str__(self):
                    1/0
        err = self.get_report(X())
        str_value = '<exception str() failed>'
        str_name = '.'.join([X.__module__, X.__qualname__])
        self.assertEqual(MODULE_PREFIX + err, f"{str_name}: {str_value}\n")


    # #### Exception Groups ####

    def test_exception_group_basic(self):
        def exc():
            raise ExceptionGroup("eg", [ValueError(1), TypeError(2)])

        expected = (
             f'  + Exception Group Traceback (most recent call last):\n'
             f'  |   File "{__file__}", line {self.callable_line}, in get_exception\n'
             f'  |     exception_or_callable()\n'
             f'  |     ~~~~~~~~~~~~~~~~~~~~~^^\n'
             f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 1}, in exc\n'
             f'  |     raise ExceptionGroup("eg", [ValueError(1), TypeError(2)])\n'
             f'  | ExceptionGroup: eg (2 sub-exceptions)\n'
             f'  +-+---------------- 1 ----------------\n'
             f'    | ValueError: 1\n'
             f'    +---------------- 2 ----------------\n'
             f'    | TypeError: 2\n'
             f'    +------------------------------------\n')

        report = self.get_report(exc)
        self.assertEqual(report, expected)

    def test_exception_group_cause(self):
        def exc():
            EG = ExceptionGroup
            try:
                raise EG("eg1", [ValueError(1), TypeError(2)])
            except Exception as e:
                raise EG("eg2", [ValueError(3), TypeError(4)]) from e

        expected = (f'  + Exception Group Traceback (most recent call last):\n'
                    f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 3}, in exc\n'
                    f'  |     raise EG("eg1", [ValueError(1), TypeError(2)])\n'
                    f'  | ExceptionGroup: eg1 (2 sub-exceptions)\n'
                    f'  +-+---------------- 1 ----------------\n'
                    f'    | ValueError: 1\n'
                    f'    +---------------- 2 ----------------\n'
                    f'    | TypeError: 2\n'
                    f'    +------------------------------------\n'
                    f'\n'
                    f'The above exception was the direct cause of the following exception:\n'
                    f'\n'
                    f'  + Exception Group Traceback (most recent call last):\n'
                    f'  |   File "{__file__}", line {self.callable_line}, in get_exception\n'
                    f'  |     exception_or_callable()\n'
                    f'  |     ~~~~~~~~~~~~~~~~~~~~~^^\n'
                    f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 5}, in exc\n'
                    f'  |     raise EG("eg2", [ValueError(3), TypeError(4)]) from e\n'
                    f'  | ExceptionGroup: eg2 (2 sub-exceptions)\n'
                    f'  +-+---------------- 1 ----------------\n'
                    f'    | ValueError: 3\n'
                    f'    +---------------- 2 ----------------\n'
                    f'    | TypeError: 4\n'
                    f'    +------------------------------------\n')

        report = self.get_report(exc)
        self.assertEqual(report, expected)

    def test_exception_group_context_with_context(self):
        def exc():
            EG = ExceptionGroup
            try:
                try:
                    raise EG("eg1", [ValueError(1), TypeError(2)])
                except EG:
                    raise EG("eg2", [ValueError(3), TypeError(4)])
            except EG:
                raise ImportError(5)

        expected = (
             f'  + Exception Group Traceback (most recent call last):\n'
             f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 4}, in exc\n'
             f'  |     raise EG("eg1", [ValueError(1), TypeError(2)])\n'
             f'  | ExceptionGroup: eg1 (2 sub-exceptions)\n'
             f'  +-+---------------- 1 ----------------\n'
             f'    | ValueError: 1\n'
             f'    +---------------- 2 ----------------\n'
             f'    | TypeError: 2\n'
             f'    +------------------------------------\n'
             f'\n'
             f'During handling of the above exception, another exception occurred:\n'
             f'\n'
             f'  + Exception Group Traceback (most recent call last):\n'
             f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 6}, in exc\n'
             f'  |     raise EG("eg2", [ValueError(3), TypeError(4)])\n'
             f'  | ExceptionGroup: eg2 (2 sub-exceptions)\n'
             f'  +-+---------------- 1 ----------------\n'
             f'    | ValueError: 3\n'
             f'    +---------------- 2 ----------------\n'
             f'    | TypeError: 4\n'
             f'    +------------------------------------\n'
             f'\n'
             f'During handling of the above exception, another exception occurred:\n'
             f'\n'
             f'Traceback (most recent call last):\n'
             f'  File "{__file__}", line {self.callable_line}, in get_exception\n'
             f'    exception_or_callable()\n'
             f'    ~~~~~~~~~~~~~~~~~~~~~^^\n'
             f'  File "{__file__}", line {exc.__code__.co_firstlineno + 8}, in exc\n'
             f'    raise ImportError(5)\n'
             f'ImportError: 5\n')

        report = self.get_report(exc)
        self.assertEqual(report, expected)

    def test_exception_group_nested(self):
        def exc():
            EG = ExceptionGroup
            VE = ValueError
            TE = TypeError
            try:
                try:
                    raise EG("nested", [TE(2), TE(3)])
                except Exception as e:
                    exc = e
                raise EG("eg", [VE(1), exc, VE(4)])
            except EG:
                raise EG("top", [VE(5)])

        expected = (f'  + Exception Group Traceback (most recent call last):\n'
                    f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 9}, in exc\n'
                    f'  |     raise EG("eg", [VE(1), exc, VE(4)])\n'
                    f'  | ExceptionGroup: eg (3 sub-exceptions)\n'
                    f'  +-+---------------- 1 ----------------\n'
                    f'    | ValueError: 1\n'
                    f'    +---------------- 2 ----------------\n'
                    f'    | Exception Group Traceback (most recent call last):\n'
                    f'    |   File "{__file__}", line {exc.__code__.co_firstlineno + 6}, in exc\n'
                    f'    |     raise EG("nested", [TE(2), TE(3)])\n'
                    f'    | ExceptionGroup: nested (2 sub-exceptions)\n'
                    f'    +-+---------------- 1 ----------------\n'
                    f'      | TypeError: 2\n'
                    f'      +---------------- 2 ----------------\n'
                    f'      | TypeError: 3\n'
                    f'      +------------------------------------\n'
                    f'    +---------------- 3 ----------------\n'
                    f'    | ValueError: 4\n'
                    f'    +------------------------------------\n'
                    f'\n'
                    f'During handling of the above exception, another exception occurred:\n'
                    f'\n'
                    f'  + Exception Group Traceback (most recent call last):\n'
                    f'  |   File "{__file__}", line {self.callable_line}, in get_exception\n'
                    f'  |     exception_or_callable()\n'
                    f'  |     ~~~~~~~~~~~~~~~~~~~~~^^\n'
                    f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 11}, in exc\n'
                    f'  |     raise EG("top", [VE(5)])\n'
                    f'  | ExceptionGroup: top (1 sub-exception)\n'
                    f'  +-+---------------- 1 ----------------\n'
                    f'    | ValueError: 5\n'
                    f'    +------------------------------------\n')

        report = self.get_report(exc)
        self.assertEqual(report, expected)

    def test_exception_group_width_limit(self):
        excs = []
        for i in range(1000):
            excs.append(ValueError(i))
        eg = ExceptionGroup('eg', excs)

        expected = ('  | ExceptionGroup: eg (1000 sub-exceptions)\n'
                    '  +-+---------------- 1 ----------------\n'
                    '    | ValueError: 0\n'
                    '    +---------------- 2 ----------------\n'
                    '    | ValueError: 1\n'
                    '    +---------------- 3 ----------------\n'
                    '    | ValueError: 2\n'
                    '    +---------------- 4 ----------------\n'
                    '    | ValueError: 3\n'
                    '    +---------------- 5 ----------------\n'
                    '    | ValueError: 4\n'
                    '    +---------------- 6 ----------------\n'
                    '    | ValueError: 5\n'
                    '    +---------------- 7 ----------------\n'
                    '    | ValueError: 6\n'
                    '    +---------------- 8 ----------------\n'
                    '    | ValueError: 7\n'
                    '    +---------------- 9 ----------------\n'
                    '    | ValueError: 8\n'
                    '    +---------------- 10 ----------------\n'
                    '    | ValueError: 9\n'
                    '    +---------------- 11 ----------------\n'
                    '    | ValueError: 10\n'
                    '    +---------------- 12 ----------------\n'
                    '    | ValueError: 11\n'
                    '    +---------------- 13 ----------------\n'
                    '    | ValueError: 12\n'
                    '    +---------------- 14 ----------------\n'
                    '    | ValueError: 13\n'
                    '    +---------------- 15 ----------------\n'
                    '    | ValueError: 14\n'
                    '    +---------------- ... ----------------\n'
                    '    | and 985 more exceptions\n'
                    '    +------------------------------------\n')

        report = self.get_report(eg)
        self.assertEqual(report, expected)

    def test_exception_group_depth_limit(self):
        exc = TypeError('bad type')
        for i in range(1000):
            exc = ExceptionGroup(
                f'eg{i}',
                [ValueError(i), exc, ValueError(-i)])

        expected = ('  | ExceptionGroup: eg999 (3 sub-exceptions)\n'
                    '  +-+---------------- 1 ----------------\n'
                    '    | ValueError: 999\n'
                    '    +---------------- 2 ----------------\n'
                    '    | ExceptionGroup: eg998 (3 sub-exceptions)\n'
                    '    +-+---------------- 1 ----------------\n'
                    '      | ValueError: 998\n'
                    '      +---------------- 2 ----------------\n'
                    '      | ExceptionGroup: eg997 (3 sub-exceptions)\n'
                    '      +-+---------------- 1 ----------------\n'
                    '        | ValueError: 997\n'
                    '        +---------------- 2 ----------------\n'
                    '        | ExceptionGroup: eg996 (3 sub-exceptions)\n'
                    '        +-+---------------- 1 ----------------\n'
                    '          | ValueError: 996\n'
                    '          +---------------- 2 ----------------\n'
                    '          | ExceptionGroup: eg995 (3 sub-exceptions)\n'
                    '          +-+---------------- 1 ----------------\n'
                    '            | ValueError: 995\n'
                    '            +---------------- 2 ----------------\n'
                    '            | ExceptionGroup: eg994 (3 sub-exceptions)\n'
                    '            +-+---------------- 1 ----------------\n'
                    '              | ValueError: 994\n'
                    '              +---------------- 2 ----------------\n'
                    '              | ExceptionGroup: eg993 (3 sub-exceptions)\n'
                    '              +-+---------------- 1 ----------------\n'
                    '                | ValueError: 993\n'
                    '                +---------------- 2 ----------------\n'
                    '                | ExceptionGroup: eg992 (3 sub-exceptions)\n'
                    '                +-+---------------- 1 ----------------\n'
                    '                  | ValueError: 992\n'
                    '                  +---------------- 2 ----------------\n'
                    '                  | ExceptionGroup: eg991 (3 sub-exceptions)\n'
                    '                  +-+---------------- 1 ----------------\n'
                    '                    | ValueError: 991\n'
                    '                    +---------------- 2 ----------------\n'
                    '                    | ExceptionGroup: eg990 (3 sub-exceptions)\n'
                    '                    +-+---------------- 1 ----------------\n'
                    '                      | ValueError: 990\n'
                    '                      +---------------- 2 ----------------\n'
                    '                      | ... (max_group_depth is 10)\n'
                    '                      +---------------- 3 ----------------\n'
                    '                      | ValueError: -990\n'
                    '                      +------------------------------------\n'
                    '                    +---------------- 3 ----------------\n'
                    '                    | ValueError: -991\n'
                    '                    +------------------------------------\n'
                    '                  +---------------- 3 ----------------\n'
                    '                  | ValueError: -992\n'
                    '                  +------------------------------------\n'
                    '                +---------------- 3 ----------------\n'
                    '                | ValueError: -993\n'
                    '                +------------------------------------\n'
                    '              +---------------- 3 ----------------\n'
                    '              | ValueError: -994\n'
                    '              +------------------------------------\n'
                    '            +---------------- 3 ----------------\n'
                    '            | ValueError: -995\n'
                    '            +------------------------------------\n'
                    '          +---------------- 3 ----------------\n'
                    '          | ValueError: -996\n'
                    '          +------------------------------------\n'
                    '        +---------------- 3 ----------------\n'
                    '        | ValueError: -997\n'
                    '        +------------------------------------\n'
                    '      +---------------- 3 ----------------\n'
                    '      | ValueError: -998\n'
                    '      +------------------------------------\n'
                    '    +---------------- 3 ----------------\n'
                    '    | ValueError: -999\n'
                    '    +------------------------------------\n')

        report = self.get_report(exc)
        self.assertEqual(report, expected)

    def test_exception_group_with_notes(self):
        def exc():
            try:
                excs = []
                for msg in ['bad value', 'terrible value']:
                    try:
                        raise ValueError(msg)
                    except ValueError as e:
                        e.add_note(f'the {msg}')
                        excs.append(e)
                raise ExceptionGroup("nested", excs)
            except ExceptionGroup as e:
                e.add_note(('>> Multi line note\n'
                            '>> Because I am such\n'
                            '>> an important exception.\n'
                            '>> empty lines work too\n'
                            '\n'
                            '(that was an empty line)'))
                raise

        expected = (f'  + Exception Group Traceback (most recent call last):\n'
                    f'  |   File "{__file__}", line {self.callable_line}, in get_exception\n'
                    f'  |     exception_or_callable()\n'
                    f'  |     ~~~~~~~~~~~~~~~~~~~~~^^\n'
                    f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 9}, in exc\n'
                    f'  |     raise ExceptionGroup("nested", excs)\n'
                    f'  | ExceptionGroup: nested (2 sub-exceptions)\n'
                    f'  | >> Multi line note\n'
                    f'  | >> Because I am such\n'
                    f'  | >> an important exception.\n'
                    f'  | >> empty lines work too\n'
                    f'  | \n'
                    f'  | (that was an empty line)\n'
                    f'  +-+---------------- 1 ----------------\n'
                    f'    | Traceback (most recent call last):\n'
                    f'    |   File "{__file__}", line {exc.__code__.co_firstlineno + 5}, in exc\n'
                    f'    |     raise ValueError(msg)\n'
                    f'    | ValueError: bad value\n'
                    f'    | the bad value\n'
                    f'    +---------------- 2 ----------------\n'
                    f'    | Traceback (most recent call last):\n'
                    f'    |   File "{__file__}", line {exc.__code__.co_firstlineno + 5}, in exc\n'
                    f'    |     raise ValueError(msg)\n'
                    f'    | ValueError: terrible value\n'
                    f'    | the terrible value\n'
                    f'    +------------------------------------\n')

        report = self.get_report(exc)
        self.assertEqual(report, expected)

    def test_exception_group_with_multiple_notes(self):
        def exc():
            try:
                excs = []
                for msg in ['bad value', 'terrible value']:
                    try:
                        raise ValueError(msg)
                    except ValueError as e:
                        e.add_note(f'the {msg}')
                        e.add_note(f'Goodbye {msg}')
                        excs.append(e)
                raise ExceptionGroup("nested", excs)
            except ExceptionGroup as e:
                e.add_note(('>> Multi line note\n'
                            '>> Because I am such\n'
                            '>> an important exception.\n'
                            '>> empty lines work too\n'
                            '\n'
                            '(that was an empty line)'))
                e.add_note('Goodbye!')
                raise

        expected = (f'  + Exception Group Traceback (most recent call last):\n'
                    f'  |   File "{__file__}", line {self.callable_line}, in get_exception\n'
                    f'  |     exception_or_callable()\n'
                    f'  |     ~~~~~~~~~~~~~~~~~~~~~^^\n'
                    f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 10}, in exc\n'
                    f'  |     raise ExceptionGroup("nested", excs)\n'
                    f'  | ExceptionGroup: nested (2 sub-exceptions)\n'
                    f'  | >> Multi line note\n'
                    f'  | >> Because I am such\n'
                    f'  | >> an important exception.\n'
                    f'  | >> empty lines work too\n'
                    f'  | \n'
                    f'  | (that was an empty line)\n'
                    f'  | Goodbye!\n'
                    f'  +-+---------------- 1 ----------------\n'
                    f'    | Traceback (most recent call last):\n'
                    f'    |   File "{__file__}", line {exc.__code__.co_firstlineno + 5}, in exc\n'
                    f'    |     raise ValueError(msg)\n'
                    f'    | ValueError: bad value\n'
                    f'    | the bad value\n'
                    f'    | Goodbye bad value\n'
                    f'    +---------------- 2 ----------------\n'
                    f'    | Traceback (most recent call last):\n'
                    f'    |   File "{__file__}", line {exc.__code__.co_firstlineno + 5}, in exc\n'
                    f'    |     raise ValueError(msg)\n'
                    f'    | ValueError: terrible value\n'
                    f'    | the terrible value\n'
                    f'    | Goodbye terrible value\n'
                    f'    +------------------------------------\n')

        report = self.get_report(exc)
        self.assertEqual(report, expected)

    def test_exception_group_wrapped_naked(self):
        # See gh-128799

        def exc():
            try:
                raise Exception(42)
            except* Exception as e:
                raise

        expected = (f'  + Exception Group Traceback (most recent call last):\n'
                    f'  |   File "{__file__}", line {self.callable_line}, in get_exception\n'
                    f'  |     exception_or_callable()\n'
                    f'  |     ~~~~~~~~~~~~~~~~~~~~~^^\n'
                    f'  |   File "{__file__}", line {exc.__code__.co_firstlineno + 3}, in exc\n'
                    f'  |     except* Exception as e:\n'
                    f'  |         raise\n'
                    f'  | ExceptionGroup:  (1 sub-exception)\n'
                    f'  +-+---------------- 1 ----------------\n'
                    f'    | Traceback (most recent call last):\n'
                    f'    |   File "{__file__}", line {exc.__code__.co_firstlineno + 2}, in exc\n'
                    f'    |     raise Exception(42)\n'
                    f'    | Exception: 42\n'
                    f'    +------------------------------------\n')

        report = self.get_report(exc)
        self.assertEqual(report, expected)

    def test_KeyboardInterrupt_at_first_line_of_frame(self):
        # see GH-93249
        def f():
            return sys._getframe()

        tb_next = None
        frame = f()
        lasti = 0
        lineno = f.__code__.co_firstlineno
        tb = types.TracebackType(tb_next, frame, lasti, lineno)

        exc = KeyboardInterrupt()
        exc.__traceback__ = tb

        expected = (f'Traceback (most recent call last):\n'
                    f'  File "{__file__}", line {lineno}, in f\n'
                    f'    def f():\n'
                    f'\n'
                    f'KeyboardInterrupt\n')

        report = self.get_report(exc)
        # remove trailing writespace:
        report = '\n'.join([l.rstrip() for l in report.split('\n')])
        self.assertEqual(report, expected)


@force_not_colorized_test_class
class PyExcReportingTests(BaseExceptionReportingTests, __TestCase):
    #
    # This checks reporting through the 'traceback' module, with both
    # format_exception() and print_exception().
    #

    def get_report(self, e):
        e = self.get_exception(e)
        s = ''.join(
            traceback.format_exception(type(e), e, e.__traceback__))
        with captured_output("stderr") as sio:
            traceback.print_exception(type(e), e, e.__traceback__)
        self.assertEqual(sio.getvalue(), s)
        return s


@force_not_colorized_test_class
class CExcReportingTests(BaseExceptionReportingTests, __TestCase):
    #
    # This checks built-in reporting by the interpreter.
    #

    @cpython_only
    def get_report(self, e):
        from _testcapi import exception_print
        e = self.get_exception(e)
        with captured_output("stderr") as s:
            exception_print(e)
        return s.getvalue()


class LimitTests(__TestCase):

    ''' Tests for limit argument.
        It's enough to test extact_tb, extract_stack and format_exception '''

    def last_raises1(self):
        raise Exception('Last raised')

    def last_raises2(self):
        self.last_raises1()

    def last_raises3(self):
        self.last_raises2()

    def last_raises4(self):
        self.last_raises3()

    def last_raises5(self):
        self.last_raises4()

    def last_returns_frame1(self):
        return sys._getframe()

    def last_returns_frame2(self):
        return self.last_returns_frame1()

    def last_returns_frame3(self):
        return self.last_returns_frame2()

    def last_returns_frame4(self):
        return self.last_returns_frame3()

    def last_returns_frame5(self):
        return self.last_returns_frame4()

    def test_extract_stack(self):
        frame = self.last_returns_frame5()
        def extract(**kwargs):
            return traceback.extract_stack(frame, **kwargs)
        def assertEqualExcept(actual, expected, ignore):
            self.assertEqual(actual[:ignore], expected[:ignore])
            self.assertEqual(actual[ignore+1:], expected[ignore+1:])
            self.assertEqual(len(actual), len(expected))

        with support.swap_attr(sys, 'tracebacklimit', 1000):
            nolim = extract()
            self.assertGreater(len(nolim), 5)
            self.assertEqual(extract(limit=2), nolim[-2:])
            assertEqualExcept(extract(limit=100), nolim[-100:], -5-1)
            self.assertEqual(extract(limit=-2), nolim[:2])
            assertEqualExcept(extract(limit=-100), nolim[:100], len(nolim)-5-1)
            self.assertEqual(extract(limit=0), [])
            del sys.tracebacklimit
            assertEqualExcept(extract(), nolim, -5-1)
            sys.tracebacklimit = 2
            self.assertEqual(extract(), nolim[-2:])
            self.assertEqual(extract(limit=3), nolim[-3:])
            self.assertEqual(extract(limit=-3), nolim[:3])
            sys.tracebacklimit = 0
            self.assertEqual(extract(), [])
            sys.tracebacklimit = -1
            self.assertEqual(extract(), [])

    def test_extract_tb(self):
        try:
            self.last_raises5()
        except Exception as e:
            tb = e.__traceback__
        def extract(**kwargs):
            return traceback.extract_tb(tb, **kwargs)

        with support.swap_attr(sys, 'tracebacklimit', 1000):
            nolim = extract()
            self.assertEqual(len(nolim), 5+1)
            self.assertEqual(extract(limit=2), nolim[:2])
            self.assertEqual(extract(limit=10), nolim)
            self.assertEqual(extract(limit=-2), nolim[-2:])
            self.assertEqual(extract(limit=-10), nolim)
            self.assertEqual(extract(limit=0), [])
            del sys.tracebacklimit
            self.assertEqual(extract(), nolim)
            sys.tracebacklimit = 2
            self.assertEqual(extract(), nolim[:2])
            self.assertEqual(extract(limit=3), nolim[:3])
            self.assertEqual(extract(limit=-3), nolim[-3:])
            sys.tracebacklimit = 0
            self.assertEqual(extract(), [])
            sys.tracebacklimit = -1
            self.assertEqual(extract(), [])

    def test_format_exception(self):
        try:
            self.last_raises5()
        except Exception as e:
            exc = e
        # [1:-1] to exclude "Traceback (...)" header and
        # exception type and value
        def extract(**kwargs):
            return traceback.format_exception(exc, **kwargs)[1:-1]

        with support.swap_attr(sys, 'tracebacklimit', 1000):
            nolim = extract()
            self.assertEqual(len(nolim), 5+1)
            self.assertEqual(extract(limit=2), nolim[:2])
            self.assertEqual(extract(limit=10), nolim)
            self.assertEqual(extract(limit=-2), nolim[-2:])
            self.assertEqual(extract(limit=-10), nolim)
            self.assertEqual(extract(limit=0), [])
            del sys.tracebacklimit
            self.assertEqual(extract(), nolim)
            sys.tracebacklimit = 2
            self.assertEqual(extract(), nolim[:2])
            self.assertEqual(extract(limit=3), nolim[:3])
            self.assertEqual(extract(limit=-3), nolim[-3:])
            sys.tracebacklimit = 0
            self.assertEqual(extract(), [])
            sys.tracebacklimit = -1
            self.assertEqual(extract(), [])


class MiscTracebackCases(__TestCase):
    #
    # Check non-printing functions in traceback module
    #

    def test_clear(self):
        def outer():
            middle()
        def middle():
            inner()
        def inner():
            i = 1
            1/0

        try:
            outer()
        except BaseException as e:
            tb = e.__traceback__

        # Initial assertion: there's one local in the inner frame.
        inner_frame = tb.tb_next.tb_next.tb_next.tb_frame
        self.assertEqual(len(inner_frame.f_locals), 1)

        # Clear traceback frames
        traceback.clear_frames(tb)

        # Local variable dict should now be empty.
        self.assertEqual(len(inner_frame.f_locals), 0)

    def test_extract_stack(self):
        def extract():
            return traceback.extract_stack()
        result = extract()
        lineno = extract.__code__.co_firstlineno
        self.assertEqual(result[-2:], [
            (__file__, lineno+2, 'test_extract_stack', 'result = extract()'),
            (__file__, lineno+1, 'extract', 'return traceback.extract_stack()'),
            ])
        self.assertEqual(len(result[0]), 4)


class TestFrame(__TestCase):

    def test_basics(self):
        linecache.clearcache()
        linecache.lazycache("f", globals())
        f = traceback.FrameSummary("f", 1, "dummy")
        self.assertEqual(f,
            ("f", 1, "dummy", '"""Test cases for traceback module"""'))
        self.assertEqual(tuple(f),
            ("f", 1, "dummy", '"""Test cases for traceback module"""'))
        self.assertEqual(f, traceback.FrameSummary("f", 1, "dummy"))
        self.assertEqual(f, tuple(f))
        # Since tuple.__eq__ doesn't support FrameSummary, the equality
        # operator fallbacks to FrameSummary.__eq__.
        self.assertEqual(tuple(f), f)
        self.assertIsNone(f.locals)
        self.assertNotEqual(f, object())
        self.assertEqual(f, ALWAYS_EQ)

    def test_lazy_lines(self):
        linecache.clearcache()
        f = traceback.FrameSummary("f", 1, "dummy", lookup_line=False)
        self.assertEqual(None, f._lines)
        linecache.lazycache("f", globals())
        self.assertEqual(
            '"""Test cases for traceback module"""',
            f.line)

    def test_no_line(self):
        f = traceback.FrameSummary("f", None, "dummy")
        self.assertEqual(f.line, None)

    def test_explicit_line(self):
        f = traceback.FrameSummary("f", 1, "dummy", line="line")
        self.assertEqual("line", f.line)

    def test_len(self):
        f = traceback.FrameSummary("f", 1, "dummy", line="line")
        self.assertEqual(len(f), 4)


class TestStack(__TestCase):

    def test_walk_stack(self):
        def deeper():
            return list(traceback.walk_stack(None))
        s1 = list(traceback.walk_stack(None))
        s2 = deeper()
        self.assertEqual(len(s2) - len(s1), 1)
        self.assertEqual(s2[1:], s1)

    def test_walk_tb(self):
        try:
            1/0
        except Exception as e:
            tb = e.__traceback__
        s = list(traceback.walk_tb(tb))
        self.assertEqual(len(s), 1)

    def test_extract_stack(self):
        s = traceback.StackSummary.extract(traceback.walk_stack(None))
        self.assertIsInstance(s, traceback.StackSummary)

    def test_extract_stack_limit(self):
        s = traceback.StackSummary.extract(traceback.walk_stack(None), limit=5)
        self.assertEqual(len(s), 5)

    def test_extract_stack_lookup_lines(self):
        linecache.clearcache()
        linecache.updatecache('/foo.py', globals())
        c = test_code('/foo.py', 'method')
        f = test_frame(c, None, None)
        s = traceback.StackSummary.extract(iter([(f, 6)]), lookup_lines=True)
        linecache.clearcache()
        self.assertEqual(s[0].line, "import sys")

    def test_extract_stackup_deferred_lookup_lines(self):
        linecache.clearcache()
        c = test_code('/foo.py', 'method')
        f = test_frame(c, None, None)
        s = traceback.StackSummary.extract(iter([(f, 6)]), lookup_lines=False)
        self.assertEqual({}, linecache.cache)
        linecache.updatecache('/foo.py', globals())
        self.assertEqual(s[0].line, "import sys")

    def test_from_list(self):
        s = traceback.StackSummary.from_list([('foo.py', 1, 'fred', 'line')])
        self.assertEqual(
            ['  File "foo.py", line 1, in fred\n    line\n'],
            s.format())

    def test_from_list_edited_stack(self):
        s = traceback.StackSummary.from_list([('foo.py', 1, 'fred', 'line')])
        s[0] = ('foo.py', 2, 'fred', 'line')
        s2 = traceback.StackSummary.from_list(s)
        self.assertEqual(
            ['  File "foo.py", line 2, in fred\n    line\n'],
            s2.format())

    def test_format_smoke(self):
        # For detailed tests see the format_list tests, which consume the same
        # code.
        s = traceback.StackSummary.from_list([('foo.py', 1, 'fred', 'line')])
        self.assertEqual(
            ['  File "foo.py", line 1, in fred\n    line\n'],
            s.format())

    def test_locals(self):
        linecache.updatecache('/foo.py', globals())
        c = test_code('/foo.py', 'method')
        f = test_frame(c, globals(), {'something': 1})
        s = traceback.StackSummary.extract(iter([(f, 6)]), capture_locals=True)
        self.assertEqual(s[0].locals, {'something': '1'})

    def test_no_locals(self):
        linecache.updatecache('/foo.py', globals())
        c = test_code('/foo.py', 'method')
        f = test_frame(c, globals(), {'something': 1})
        s = traceback.StackSummary.extract(iter([(f, 6)]))
        self.assertEqual(s[0].locals, None)

    def test_format_locals(self):
        def some_inner(k, v):
            a = 1
            b = 2
            return traceback.StackSummary.extract(
                traceback.walk_stack(None), capture_locals=True, limit=1)
        s = some_inner(3, 4)
        self.assertEqual(
            ['  File "%s", line %d, in some_inner\n'
             '    return traceback.StackSummary.extract(\n'
             '    a = 1\n'
             '    b = 2\n'
             '    k = 3\n'
             '    v = 4\n' % (__file__, some_inner.__code__.co_firstlineno + 3)
            ], s.format())

    def test_custom_format_frame(self):
        with torch._dynamo.error_on_graph_break(False):
            class CustomStackSummary(traceback.StackSummary):
                def format_frame_summary(self, frame_summary, colorize=False):
                    return f'{frame_summary.filename}:{frame_summary.lineno}'

        def some_inner():
            return CustomStackSummary.extract(
                traceback.walk_stack(None), limit=1)

        s = some_inner()
        self.assertEqual(
            s.format(),
            [f'{__file__}:{some_inner.__code__.co_firstlineno + 1}'])

    def test_dropping_frames(self):
        def f():
            1/0

        def g():
            try:
                f()
            except Exception as e:
                return e.__traceback__

        tb = g()

        with torch._dynamo.error_on_graph_break(False):
            class Skip_G(traceback.StackSummary):
                def format_frame_summary(self, frame_summary, colorize=False):
                    if frame_summary.name == 'g':
                        return None
                    return super().format_frame_summary(frame_summary)

        stack = Skip_G.extract(
            traceback.walk_tb(tb)).format()

        self.assertEqual(len(stack), 1)
        lno = f.__code__.co_firstlineno + 1
        self.assertEqual(
            stack[0],
            f'  File "{__file__}", line {lno}, in f\n    1/0\n'
        )

    def test_summary_should_show_carets(self):
        # See: https://github.com/python/cpython/issues/122353

        # statement to execute and to get a ZeroDivisionError for a traceback
        statement = "abcdef = 1 / 0 and 2.0"
        colno = statement.index('1 / 0')
        end_colno = colno + len('1 / 0')

        # Actual line to use when rendering the traceback
        # and whose AST will be extracted (it will be empty).
        cached_line = '# this line will be used during rendering'
        self.addCleanup(unlink, TESTFN)
        with open(TESTFN, "w") as file:
            file.write(cached_line)
        linecache.updatecache(TESTFN, {})

        try:
            exec(compile(statement, TESTFN, "exec"))
        except ZeroDivisionError as exc:
            # This is the simplest way to create a StackSummary
            # whose FrameSummary items have their column offsets.
            s = traceback.TracebackException.from_exception(exc).stack
            self.assertIsInstance(s, traceback.StackSummary)
            with unittest.mock.patch.object(s, '_should_show_carets',
                                            wraps=s._should_show_carets) as ff:
                self.assertEqual(len(s), 2)
                self.assertListEqual(
                    s.format_frame_summary(s[1]).splitlines(),
                    [
                        f'  File "{TESTFN}", line 1, in <module>',
                        f'    {cached_line}'
                     ]
                )
                ff.assert_called_with(colno, end_colno, [cached_line], None)

class Unrepresentable:
    def __repr__(self) -> str:
        raise Exception("Unrepresentable")


# Used in test_dont_swallow_cause_or_context_of_falsey_exception and
# test_dont_swallow_subexceptions_of_falsey_exceptiongroup.
class FalseyException(Exception):
    def __bool__(self):
        return False


class FalseyExceptionGroup(ExceptionGroup):
    def __bool__(self):
        return False


class TestTracebackException(__TestCase):
    def do_test_smoke(self, exc, expected_type_str):
        try:
            raise exc
        except Exception as e:
            exc_obj = e
            exc = traceback.TracebackException.from_exception(e)
            expected_stack = traceback.StackSummary.extract(
                traceback.walk_tb(e.__traceback__))
        self.assertEqual(None, exc.__cause__)
        self.assertEqual(None, exc.__context__)
        self.assertEqual(False, exc.__suppress_context__)
        self.assertEqual(expected_stack, exc.stack)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(type(exc_obj), exc.exc_type)
        self.assertEqual(expected_type_str, exc.exc_type_str)
        self.assertEqual(str(exc_obj), str(exc))

    def test_smoke_builtin(self):
        self.do_test_smoke(ValueError(42), 'ValueError')

    def test_smoke_user_exception(self):
        with torch._dynamo.error_on_graph_break(False):
            class MyException(Exception):
                pass

        if __name__ == '__main__':
            expected = ('TestTracebackException.'
                        'test_smoke_user_exception.<locals>.MyException')
        else:
            expected = ('test.test_traceback.TestTracebackException.'
                        'test_smoke_user_exception.<locals>.MyException')
        self.do_test_smoke(MyException('bad things happened'), expected)

    def test_from_exception(self):
        # Check all the parameters are accepted.
        def foo():
            1/0
        try:
            foo()
        except Exception as e:
            exc_obj = e
            tb = e.__traceback__
            self.expected_stack = traceback.StackSummary.extract(
                traceback.walk_tb(tb), limit=1, lookup_lines=False,
                capture_locals=True)
            self.exc = traceback.TracebackException.from_exception(
                e, limit=1, lookup_lines=False, capture_locals=True)
        expected_stack = self.expected_stack
        exc = self.exc
        self.assertEqual(None, exc.__cause__)
        self.assertEqual(None, exc.__context__)
        self.assertEqual(False, exc.__suppress_context__)
        self.assertEqual(expected_stack, exc.stack)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(type(exc_obj), exc.exc_type)
        self.assertEqual(type(exc_obj).__name__, exc.exc_type_str)
        self.assertEqual(str(exc_obj), str(exc))

    def test_cause(self):
        try:
            try:
                1/0
            finally:
                exc = sys.exception()
                exc_context = traceback.TracebackException.from_exception(exc)
                cause = Exception("cause")
                raise Exception("uh oh") from cause
        except Exception as e:
            exc_obj = e
            exc = traceback.TracebackException.from_exception(e)
            expected_stack = traceback.StackSummary.extract(
                traceback.walk_tb(e.__traceback__))
        exc_cause = traceback.TracebackException(Exception, cause, None)
        self.assertEqual(exc_cause, exc.__cause__)
        self.assertEqual(exc_context, exc.__context__)
        self.assertEqual(True, exc.__suppress_context__)
        self.assertEqual(expected_stack, exc.stack)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(type(exc_obj), exc.exc_type)
        self.assertEqual(type(exc_obj).__name__, exc.exc_type_str)
        self.assertEqual(str(exc_obj), str(exc))

    def test_context(self):
        try:
            try:
                1/0
            finally:
                exc = sys.exception()
                exc_context = traceback.TracebackException.from_exception(exc)
                raise Exception("uh oh")
        except Exception as e:
            exc_obj = e
            exc = traceback.TracebackException.from_exception(e)
            expected_stack = traceback.StackSummary.extract(
                traceback.walk_tb(e.__traceback__))
        self.assertEqual(None, exc.__cause__)
        self.assertEqual(exc_context, exc.__context__)
        self.assertEqual(False, exc.__suppress_context__)
        self.assertEqual(expected_stack, exc.stack)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(type(exc_obj), exc.exc_type)
        self.assertEqual(type(exc_obj).__name__, exc.exc_type_str)
        self.assertEqual(str(exc_obj), str(exc))

    def test_long_context_chain(self):
        def f():
            try:
                1/0
            except ZeroDivisionError:
                f()

        try:
            f()
        except RecursionError as e:
            exc_obj = e
        else:
            self.fail("Exception not raised")

        te = traceback.TracebackException.from_exception(exc_obj)
        res = list(te.format())

        # many ZeroDiv errors followed by the RecursionError
        self.assertGreater(len(res), sys.getrecursionlimit())
        self.assertGreater(
            len([l for l in res if 'ZeroDivisionError:' in l]),
            sys.getrecursionlimit() * 0.5)
        self.assertIn(
            "RecursionError: maximum recursion depth exceeded", res[-1])

    def test_compact_with_cause(self):
        try:
            try:
                1/0
            finally:
                cause = Exception("cause")
                raise Exception("uh oh") from cause
        except Exception as e:
            exc_obj = e
            exc = traceback.TracebackException.from_exception(exc_obj, compact=True)
            expected_stack = traceback.StackSummary.extract(
                traceback.walk_tb(exc_obj.__traceback__))
        exc_cause = traceback.TracebackException(Exception, cause, None)
        self.assertEqual(exc_cause, exc.__cause__)
        self.assertEqual(None, exc.__context__)
        self.assertEqual(True, exc.__suppress_context__)
        self.assertEqual(expected_stack, exc.stack)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(type(exc_obj), exc.exc_type)
        self.assertEqual(type(exc_obj).__name__, exc.exc_type_str)
        self.assertEqual(str(exc_obj), str(exc))

    def test_compact_no_cause(self):
        try:
            try:
                1/0
            finally:
                exc = sys.exception()
                exc_context = traceback.TracebackException.from_exception(exc)
                raise Exception("uh oh")
        except Exception as e:
            exc_obj = e
            exc = traceback.TracebackException.from_exception(e, compact=True)
            expected_stack = traceback.StackSummary.extract(
                traceback.walk_tb(exc_obj.__traceback__))
        self.assertEqual(None, exc.__cause__)
        self.assertEqual(exc_context, exc.__context__)
        self.assertEqual(False, exc.__suppress_context__)
        self.assertEqual(expected_stack, exc.stack)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(type(exc_obj), exc.exc_type)
        self.assertEqual(type(exc_obj).__name__, exc.exc_type_str)
        self.assertEqual(str(exc_obj), str(exc))

    def test_no_save_exc_type(self):
        try:
            1/0
        except Exception as e:
            exc = e

        te = traceback.TracebackException.from_exception(
                 exc, save_exc_type=False)
        with self.assertWarns(DeprecationWarning):
            self.assertIsNone(te.exc_type)

    def test_no_refs_to_exception_and_traceback_objects(self):
        try:
            1/0
        except Exception as e:
            exc_obj = e

        refcnt1 = sys.getrefcount(exc_obj)
        refcnt2 = sys.getrefcount(exc_obj.__traceback__)
        exc = traceback.TracebackException.from_exception(exc_obj)
        self.assertEqual(sys.getrefcount(exc_obj), refcnt1)
        self.assertEqual(sys.getrefcount(exc_obj.__traceback__), refcnt2)

    def test_comparison_basic(self):
        try:
            1/0
        except Exception as e:
            exc_obj = e
            exc = traceback.TracebackException.from_exception(exc_obj)
            exc2 = traceback.TracebackException.from_exception(exc_obj)
        self.assertIsNot(exc, exc2)
        self.assertEqual(exc, exc2)
        self.assertNotEqual(exc, object())
        self.assertEqual(exc, ALWAYS_EQ)

    def test_comparison_params_variations(self):
        def raise_exc():
            try:
                raise ValueError('bad value')
            except ValueError:
                raise

        def raise_with_locals():
            x, y = 1, 2
            raise_exc()

        try:
            raise_with_locals()
        except Exception as e:
            exc_obj = e

        exc = traceback.TracebackException.from_exception(exc_obj)
        exc1 = traceback.TracebackException.from_exception(exc_obj, limit=10)
        exc2 = traceback.TracebackException.from_exception(exc_obj, limit=2)

        self.assertEqual(exc, exc1)      # limit=10 gets all frames
        self.assertNotEqual(exc, exc2)   # limit=2 truncates the output

        # locals change the output
        exc3 = traceback.TracebackException.from_exception(exc_obj, capture_locals=True)
        self.assertNotEqual(exc, exc3)

        # there are no locals in the innermost frame
        exc4 = traceback.TracebackException.from_exception(exc_obj, limit=-1)
        exc5 = traceback.TracebackException.from_exception(exc_obj, limit=-1, capture_locals=True)
        self.assertEqual(exc4, exc5)

        # there are locals in the next-to-innermost frame
        exc6 = traceback.TracebackException.from_exception(exc_obj, limit=-2)
        exc7 = traceback.TracebackException.from_exception(exc_obj, limit=-2, capture_locals=True)
        self.assertNotEqual(exc6, exc7)

    def test_comparison_equivalent_exceptions_are_equal(self):
        excs = []
        for _ in range(2):
            try:
                1/0
            except Exception as e:
                excs.append(traceback.TracebackException.from_exception(e))
        self.assertEqual(excs[0], excs[1])
        self.assertEqual(list(excs[0].format()), list(excs[1].format()))

    def test_unhashable(self):
        with torch._dynamo.error_on_graph_break(False):
            class UnhashableException(Exception):
                def __eq__(self, other):
                    return True

        ex1 = UnhashableException('ex1')
        ex2 = UnhashableException('ex2')
        try:
            raise ex2 from ex1
        except UnhashableException:
            try:
                raise ex1
            except UnhashableException as e:
                exc_obj = e
        exc = traceback.TracebackException.from_exception(exc_obj)
        formatted = list(exc.format())
        self.assertIn('UnhashableException: ex2\n', formatted[2])
        self.assertIn('UnhashableException: ex1\n', formatted[6])

    def test_limit(self):
        def recurse(n):
            if n:
                recurse(n-1)
            else:
                1/0
        try:
            recurse(10)
        except Exception as e:
            exc = traceback.TracebackException.from_exception(e, limit=5)
            expected_stack = traceback.StackSummary.extract(
                traceback.walk_tb(e.__traceback__), limit=5)
        self.assertEqual(expected_stack, exc.stack)

    def test_lookup_lines(self):
        linecache.clearcache()
        e = Exception("uh oh")
        c = test_code('/foo.py', 'method')
        f = test_frame(c, None, None)
        tb = test_tb(f, 6, None, 0)
        exc = traceback.TracebackException(Exception, e, tb, lookup_lines=False)
        self.assertEqual(linecache.cache, {})
        linecache.updatecache('/foo.py', globals())
        self.assertEqual(exc.stack[0].line, "import sys")

    def test_locals(self):
        linecache.updatecache('/foo.py', globals())
        e = Exception("uh oh")
        c = test_code('/foo.py', 'method')
        f = test_frame(c, globals(), {'something': 1, 'other': 'string', 'unrepresentable': Unrepresentable()})
        tb = test_tb(f, 6, None, 0)
        exc = traceback.TracebackException(
            Exception, e, tb, capture_locals=True)
        self.assertEqual(
            exc.stack[0].locals,
            {'something': '1', 'other': "'string'", 'unrepresentable': '<local repr() failed>'})

    def test_no_locals(self):
        linecache.updatecache('/foo.py', globals())
        e = Exception("uh oh")
        c = test_code('/foo.py', 'method')
        f = test_frame(c, globals(), {'something': 1})
        tb = test_tb(f, 6, None, 0)
        exc = traceback.TracebackException(Exception, e, tb)
        self.assertEqual(exc.stack[0].locals, None)

    def test_traceback_header(self):
        # do not print a traceback header if exc_traceback is None
        # see issue #24695
        exc = traceback.TracebackException(Exception, Exception("haven"), None)
        self.assertEqual(list(exc.format()), ["Exception: haven\n"])

    @requires_debug_ranges()
    def test_print(self):
        def f():
            x = 12
            try:
                x/0
            except Exception as e:
                return e
        exc = traceback.TracebackException.from_exception(f(), capture_locals=True)
        output = StringIO()
        exc.print(file=output)
        self.assertEqual(
            output.getvalue().split('\n')[-5:],
            ['    x/0',
             '    ~^~',
             '    x = 12',
             'ZeroDivisionError: division by zero',
             ''])

    def test_dont_swallow_cause_or_context_of_falsey_exception(self):
        # see gh-132308: Ensure that __cause__ or __context__ attributes of exceptions
        # that evaluate as falsey are included in the output. For falsey term,
        # see https://docs.python.org/3/library/stdtypes.html#truth-value-testing.

        try:
            raise FalseyException from KeyError
        except FalseyException as e:
            self.assertIn(cause_message, traceback.format_exception(e))

        try:
            try:
                1/0
            except ZeroDivisionError:
                raise FalseyException
        except FalseyException as e:
            self.assertIn(context_message, traceback.format_exception(e))


class TestTracebackException_ExceptionGroups(__TestCase):
    def setUp(self):
        super().setUp()
        self.eg = self._get_exception_group()

    def _get_exception_group(self):
        def f():
            1/0

        def g(v):
            raise ValueError(v)

        self.lno_f = f.__code__.co_firstlineno
        self.lno_g = g.__code__.co_firstlineno

        try:
            try:
                try:
                    f()
                except Exception as e:
                    exc1 = e
                try:
                    g(42)
                except Exception as e:
                    exc2 = e
                raise ExceptionGroup("eg1", [exc1, exc2])
            except ExceptionGroup as e:
                exc3 = e
            try:
                g(24)
            except Exception as e:
                exc4 = e
            raise ExceptionGroup("eg2", [exc3, exc4])
        except ExceptionGroup as eg:
            return eg
        self.fail('Exception Not Raised')

    def test_exception_group_construction(self):
        eg = self.eg
        teg1 = traceback.TracebackException(type(eg), eg, eg.__traceback__)
        teg2 = traceback.TracebackException.from_exception(eg)
        self.assertIsNot(teg1, teg2)
        self.assertEqual(teg1, teg2)

    def test_exception_group_format_exception_only(self):
        teg = traceback.TracebackException.from_exception(self.eg)
        formatted = ''.join(teg.format_exception_only()).split('\n')
        expected = "ExceptionGroup: eg2 (2 sub-exceptions)\n".split('\n')

        self.assertEqual(formatted, expected)

    def test_exception_group_format_exception_onlyi_recursive(self):
        teg = traceback.TracebackException.from_exception(self.eg)
        formatted = ''.join(teg.format_exception_only(show_group=True)).split('\n')
        expected = [
                     'ExceptionGroup: eg2 (2 sub-exceptions)',
                     '   ExceptionGroup: eg1 (2 sub-exceptions)',
                     '      ZeroDivisionError: division by zero',
                     '      ValueError: 42',
                     '   ValueError: 24',
                     ''
                   ]

        self.assertEqual(formatted, expected)

    def test_exception_group_format(self):
        teg = traceback.TracebackException.from_exception(self.eg)

        formatted = ''.join(teg.format()).split('\n')
        lno_f = self.lno_f
        lno_g = self.lno_g

        expected = [
                    f'  + Exception Group Traceback (most recent call last):',
                    f'  |   File "{__file__}", line {lno_g+23}, in _get_exception_group',
                    f'  |     raise ExceptionGroup("eg2", [exc3, exc4])',
                    f'  | ExceptionGroup: eg2 (2 sub-exceptions)',
                    f'  +-+---------------- 1 ----------------',
                    f'    | Exception Group Traceback (most recent call last):',
                    f'    |   File "{__file__}", line {lno_g+16}, in _get_exception_group',
                    f'    |     raise ExceptionGroup("eg1", [exc1, exc2])',
                    f'    | ExceptionGroup: eg1 (2 sub-exceptions)',
                    f'    +-+---------------- 1 ----------------',
                    f'      | Traceback (most recent call last):',
                    f'      |   File "{__file__}", line {lno_g+9}, in _get_exception_group',
                    f'      |     f()',
                    f'      |     ~^^',
                    f'      |   File "{__file__}", line {lno_f+1}, in f',
                    f'      |     1/0',
                    f'      |     ~^~',
                    f'      | ZeroDivisionError: division by zero',
                    f'      +---------------- 2 ----------------',
                    f'      | Traceback (most recent call last):',
                    f'      |   File "{__file__}", line {lno_g+13}, in _get_exception_group',
                    f'      |     g(42)',
                    f'      |     ~^^^^',
                    f'      |   File "{__file__}", line {lno_g+1}, in g',
                    f'      |     raise ValueError(v)',
                    f'      | ValueError: 42',
                    f'      +------------------------------------',
                    f'    +---------------- 2 ----------------',
                    f'    | Traceback (most recent call last):',
                    f'    |   File "{__file__}", line {lno_g+20}, in _get_exception_group',
                    f'    |     g(24)',
                    f'    |     ~^^^^',
                    f'    |   File "{__file__}", line {lno_g+1}, in g',
                    f'    |     raise ValueError(v)',
                    f'    | ValueError: 24',
                    f'    +------------------------------------',
                    f'']

        self.assertEqual(formatted, expected)

    def test_max_group_width(self):
        excs1 = []
        excs2 = []
        for i in range(3):
            excs1.append(ValueError(i))
        for i in range(10):
            excs2.append(TypeError(i))

        EG = ExceptionGroup
        eg = EG('eg', [EG('eg1', excs1), EG('eg2', excs2)])

        teg = traceback.TracebackException.from_exception(eg, max_group_width=2)
        formatted = ''.join(teg.format()).split('\n')

        expected = [
                    '  | ExceptionGroup: eg (2 sub-exceptions)',
                    '  +-+---------------- 1 ----------------',
                    '    | ExceptionGroup: eg1 (3 sub-exceptions)',
                    '    +-+---------------- 1 ----------------',
                    '      | ValueError: 0',
                    '      +---------------- 2 ----------------',
                    '      | ValueError: 1',
                    '      +---------------- ... ----------------',
                    '      | and 1 more exception',
                    '      +------------------------------------',
                    '    +---------------- 2 ----------------',
                    '    | ExceptionGroup: eg2 (10 sub-exceptions)',
                    '    +-+---------------- 1 ----------------',
                    '      | TypeError: 0',
                    '      +---------------- 2 ----------------',
                    '      | TypeError: 1',
                    '      +---------------- ... ----------------',
                    '      | and 8 more exceptions',
                    '      +------------------------------------',
                    '']

        self.assertEqual(formatted, expected)

    def test_max_group_depth(self):
        exc = TypeError('bad type')
        for i in range(3):
            exc = ExceptionGroup('exc', [ValueError(-i), exc, ValueError(i)])

        teg = traceback.TracebackException.from_exception(exc, max_group_depth=2)
        formatted = ''.join(teg.format()).split('\n')

        expected = [
                    '  | ExceptionGroup: exc (3 sub-exceptions)',
                    '  +-+---------------- 1 ----------------',
                    '    | ValueError: -2',
                    '    +---------------- 2 ----------------',
                    '    | ExceptionGroup: exc (3 sub-exceptions)',
                    '    +-+---------------- 1 ----------------',
                    '      | ValueError: -1',
                    '      +---------------- 2 ----------------',
                    '      | ... (max_group_depth is 2)',
                    '      +---------------- 3 ----------------',
                    '      | ValueError: 1',
                    '      +------------------------------------',
                    '    +---------------- 3 ----------------',
                    '    | ValueError: 2',
                    '    +------------------------------------',
                    '']

        self.assertEqual(formatted, expected)

    def test_comparison(self):
        try:
            raise self.eg
        except ExceptionGroup as e:
            exc = e
        for _ in range(5):
            try:
                raise exc
            except Exception as e:
                exc_obj = e
        exc = traceback.TracebackException.from_exception(exc_obj)
        exc2 = traceback.TracebackException.from_exception(exc_obj)
        exc3 = traceback.TracebackException.from_exception(exc_obj, limit=300)
        ne = traceback.TracebackException.from_exception(exc_obj, limit=3)
        self.assertIsNot(exc, exc2)
        self.assertEqual(exc, exc2)
        self.assertEqual(exc, exc3)
        self.assertNotEqual(exc, ne)
        self.assertNotEqual(exc, object())
        self.assertEqual(exc, ALWAYS_EQ)

    def test_dont_swallow_subexceptions_of_falsey_exceptiongroup(self):
        # see gh-132308: Ensure that subexceptions of exception groups
        # that evaluate as falsey are displayed in the output. For falsey term,
        # see https://docs.python.org/3/library/stdtypes.html#truth-value-testing.

        try:
            raise FalseyExceptionGroup("Gih", (KeyError(), NameError()))
        except Exception as ee:
            str_exc = ''.join(traceback.format_exception(ee))
            self.assertIn('+---------------- 1 ----------------', str_exc)
            self.assertIn('+---------------- 2 ----------------', str_exc)

        # Test with a falsey exception, in last position, as sub-exceptions.
        msg = 'bool'
        try:
            raise FalseyExceptionGroup("Gah", (KeyError(), FalseyException(msg)))
        except Exception as ee:
            str_exc = traceback.format_exception(ee)
            self.assertIn(f'{FalseyException.__name__}: {msg}', str_exc[-2])


global_for_suggestions = None


class SuggestionFormattingTestBase:
    def get_suggestion(self, obj, attr_name=None):
        if attr_name is not None:
            def callable():
                getattr(obj, attr_name)
        else:
            callable = obj

        result_lines = self.get_exception(
            callable, slice_start=-1, slice_end=None
        )
        return result_lines[0]

    def test_getattr_suggestions(self):
        with torch._dynamo.error_on_graph_break(False):
            class Substitution:
                noise = more_noise = a = bc = None
                blech = None

            class Elimination:
                noise = more_noise = a = bc = None
                blch = None

            class Addition:
                noise = more_noise = a = bc = None
                bluchin = None

            class SubstitutionOverElimination:
                blach = None
                bluc = None

            class SubstitutionOverAddition:
                blach = None
                bluchi = None

            class EliminationOverAddition:
                blucha = None
                bluc = None

            class CaseChangeOverSubstitution:
                Luch = None
                fluch = None
                BLuch = None

        for cls, suggestion in [
            (Addition, "'bluchin'?"),
            (Substitution, "'blech'?"),
            (Elimination, "'blch'?"),
            (Addition, "'bluchin'?"),
            (SubstitutionOverElimination, "'blach'?"),
            (SubstitutionOverAddition, "'blach'?"),
            (EliminationOverAddition, "'bluc'?"),
            (CaseChangeOverSubstitution, "'BLuch'?"),
        ]:
            actual = self.get_suggestion(cls(), 'bluch')
            self.assertIn(suggestion, actual)

    def test_getattr_suggestions_underscored(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                bluch = None

        self.assertIn("'bluch'", self.get_suggestion(A(), 'blach'))
        self.assertIn("'bluch'", self.get_suggestion(A(), '_luch'))
        self.assertIn("'bluch'", self.get_suggestion(A(), '_bluch'))

        with torch._dynamo.error_on_graph_break(False):
            class B:
                _bluch = None
                def method(self, name):
                    getattr(self, name)

        self.assertIn("'_bluch'", self.get_suggestion(B(), '_blach'))
        self.assertIn("'_bluch'", self.get_suggestion(B(), '_luch'))
        self.assertNotIn("'_bluch'", self.get_suggestion(B(), 'bluch'))

        self.assertIn("'_bluch'", self.get_suggestion(partial(B().method, '_blach')))
        self.assertIn("'_bluch'", self.get_suggestion(partial(B().method, '_luch')))
        self.assertIn("'_bluch'", self.get_suggestion(partial(B().method, 'bluch')))

    def test_getattr_suggestions_do_not_trigger_for_long_attributes(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                blech = None

        actual = self.get_suggestion(A(), 'somethingverywrong')
        self.assertNotIn("blech", actual)

    def test_getattr_error_bad_suggestions_do_not_trigger_for_small_names(self):
        with torch._dynamo.error_on_graph_break(False):
            class MyClass:
                vvv = mom = w = id = pytho = None

        for name in ("b", "v", "m", "py"):
            with self.subTest(name=name):
                actual = self.get_suggestion(MyClass, name)
                self.assertNotIn("Did you mean", actual)
                self.assertNotIn("'vvv", actual)
                self.assertNotIn("'mom'", actual)
                self.assertNotIn("'id'", actual)
                self.assertNotIn("'w'", actual)
                self.assertNotIn("'pytho'", actual)

    def test_getattr_suggestions_do_not_trigger_for_big_dicts(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                blech = None
        # A class with a very big __dict__ will not be considered
        # for suggestions.
        for index in range(2000):
            setattr(A, f"index_{index}", None)

        actual = self.get_suggestion(A(), 'bluch')
        self.assertNotIn("blech", actual)

    def test_getattr_suggestions_no_args(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                blech = None
                def __getattr__(self, attr):
                    raise AttributeError()

        actual = self.get_suggestion(A(), 'bluch')
        self.assertIn("blech", actual)

        with torch._dynamo.error_on_graph_break(False):
            class A:
                blech = None
                def __getattr__(self, attr):
                    raise AttributeError

        actual = self.get_suggestion(A(), 'bluch')
        self.assertIn("blech", actual)

    def test_getattr_suggestions_invalid_args(self):
        with torch._dynamo.error_on_graph_break(False):
            class NonStringifyClass:
                __str__ = None
                __repr__ = None

            class A:
                blech = None
                def __getattr__(self, attr):
                    raise AttributeError(NonStringifyClass())

            class B:
                blech = None
                def __getattr__(self, attr):
                    raise AttributeError("Error", 23)

            class C:
                blech = None
                def __getattr__(self, attr):
                    raise AttributeError(23)

        for cls in [A, B, C]:
            actual = self.get_suggestion(cls(), 'bluch')
            self.assertIn("blech", actual)

    def test_getattr_suggestions_for_same_name(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __dir__(self):
                    return ['blech']
        actual = self.get_suggestion(A(), 'blech')
        self.assertNotIn("Did you mean", actual)

    def test_attribute_error_with_failing_dict(self):
        with torch._dynamo.error_on_graph_break(False):
            class T:
                bluch = 1
                def __dir__(self):
                    raise AttributeError("oh no!")

        actual = self.get_suggestion(T(), 'blich')
        self.assertNotIn("blech", actual)
        self.assertNotIn("oh no!", actual)

    def test_attribute_error_with_non_string_candidates(self):
        with torch._dynamo.error_on_graph_break(False):
            class T:
                bluch = 1

        instance = T()
        instance.__dict__[0] = 1
        actual = self.get_suggestion(instance, 'blich')
        self.assertIn("bluch", actual)

    def test_attribute_error_with_bad_name(self):
        def raise_attribute_error_with_bad_name():
            raise AttributeError(name=12, obj=23)

        result_lines = self.get_exception(
            raise_attribute_error_with_bad_name, slice_start=-1, slice_end=None
        )
        self.assertNotIn("?", result_lines[-1])

    def test_attribute_error_inside_nested_getattr(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                bluch = 1

            class B:
                def __getattribute__(self, attr):
                    a = A()
                    return a.blich

        actual = self.get_suggestion(B(), 'something')
        self.assertIn("Did you mean", actual)
        self.assertIn("bluch", actual)

    def make_module(self, code):
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir)

        sys.path.append(str(tmpdir))
        self.addCleanup(sys.path.pop)

        mod_name = ''.join(random.choices(string.ascii_letters, k=16))
        module = tmpdir / (mod_name + ".py")
        module.write_text(code)

        return mod_name

    def get_import_from_suggestion(self, code, name):
        modname = self.make_module(code)

        def callable():
            try:
                exec(f"from {modname} import {name}")
            except ImportError as e:
                raise e from None
            except Exception as e:
                self.fail(f"Expected ImportError but got {type(e)}")
        self.addCleanup(forget, modname)

        result_lines = self.get_exception(
            callable, slice_start=-1, slice_end=None
        )
        return result_lines[0]

    def test_import_from_suggestions(self):
        substitution = textwrap.dedent("""\
            noise = more_noise = a = bc = None
            blech = None
        """)

        elimination = textwrap.dedent("""
            noise = more_noise = a = bc = None
            blch = None
        """)

        addition = textwrap.dedent("""
            noise = more_noise = a = bc = None
            bluchin = None
        """)

        substitutionOverElimination = textwrap.dedent("""
            blach = None
            bluc = None
        """)

        substitutionOverAddition = textwrap.dedent("""
            blach = None
            bluchi = None
        """)

        eliminationOverAddition = textwrap.dedent("""
            blucha = None
            bluc = None
        """)

        caseChangeOverSubstitution = textwrap.dedent("""
            Luch = None
            fluch = None
            BLuch = None
        """)

        for code, suggestion in [
            (addition, "'bluchin'?"),
            (substitution, "'blech'?"),
            (elimination, "'blch'?"),
            (addition, "'bluchin'?"),
            (substitutionOverElimination, "'blach'?"),
            (substitutionOverAddition, "'blach'?"),
            (eliminationOverAddition, "'bluc'?"),
            (caseChangeOverSubstitution, "'BLuch'?"),
        ]:
            actual = self.get_import_from_suggestion(code, 'bluch')
            self.assertIn(suggestion, actual)

    def test_import_from_suggestions_underscored(self):
        code = "bluch = None"
        self.assertIn("'bluch'", self.get_import_from_suggestion(code, 'blach'))
        self.assertIn("'bluch'", self.get_import_from_suggestion(code, '_luch'))
        self.assertIn("'bluch'", self.get_import_from_suggestion(code, '_bluch'))

        code = "_bluch = None"
        self.assertIn("'_bluch'", self.get_import_from_suggestion(code, '_blach'))
        self.assertIn("'_bluch'", self.get_import_from_suggestion(code, '_luch'))
        self.assertNotIn("'_bluch'", self.get_import_from_suggestion(code, 'bluch'))

    def test_import_from_suggestions_non_string(self):
        modWithNonStringAttr = textwrap.dedent("""\
            globals()[0] = 1
            bluch = 1
        """)
        self.assertIn("'bluch'", self.get_import_from_suggestion(modWithNonStringAttr, 'blech'))

    def test_import_from_suggestions_do_not_trigger_for_long_attributes(self):
        code = "blech = None"

        actual = self.get_suggestion(code, 'somethingverywrong')
        self.assertNotIn("blech", actual)

    def test_import_from_error_bad_suggestions_do_not_trigger_for_small_names(self):
        code = "vvv = mom = w = id = pytho = None"

        for name in ("b", "v", "m", "py"):
            with self.subTest(name=name):
                actual = self.get_import_from_suggestion(code, name)
                self.assertNotIn("Did you mean", actual)
                self.assertNotIn("'vvv'", actual)
                self.assertNotIn("'mom'", actual)
                self.assertNotIn("'id'", actual)
                self.assertNotIn("'w'", actual)
                self.assertNotIn("'pytho'", actual)

    def test_import_from_suggestions_do_not_trigger_for_big_namespaces(self):
        # A module with lots of names will not be considered for suggestions.
        chunks = [f"index_{index} = " for index in range(200)]
        chunks.append(" None")
        code = " ".join(chunks)
        actual = self.get_import_from_suggestion(code, 'bluch')
        self.assertNotIn("blech", actual)

    def test_import_from_error_with_bad_name(self):
        def raise_attribute_error_with_bad_name():
            raise ImportError(name=12, obj=23, name_from=11)

        result_lines = self.get_exception(
            raise_attribute_error_with_bad_name, slice_start=-1, slice_end=None
        )
        self.assertNotIn("?", result_lines[-1])

    def test_name_error_suggestions(self):
        def Substitution():
            noise = more_noise = a = bc = None
            blech = None
            print(bluch)

        def Elimination():
            noise = more_noise = a = bc = None
            blch = None
            print(bluch)

        def Addition():
            noise = more_noise = a = bc = None
            bluchin = None
            print(bluch)

        def SubstitutionOverElimination():
            blach = None
            bluc = None
            print(bluch)

        def SubstitutionOverAddition():
            blach = None
            bluchi = None
            print(bluch)

        def EliminationOverAddition():
            blucha = None
            bluc = None
            print(bluch)

        for func, suggestion in [(Substitution, "'blech'?"),
                                (Elimination, "'blch'?"),
                                (Addition, "'bluchin'?"),
                                (EliminationOverAddition, "'blucha'?"),
                                (SubstitutionOverElimination, "'blach'?"),
                                (SubstitutionOverAddition, "'blach'?")]:
            actual = self.get_suggestion(func)
            self.assertIn(suggestion, actual)

    def test_name_error_suggestions_from_globals(self):
        def func():
            print(global_for_suggestio)
        actual = self.get_suggestion(func)
        self.assertIn("'global_for_suggestions'?", actual)

    def test_name_error_suggestions_from_builtins(self):
        def func():
            print(ZeroDivisionErrrrr)
        actual = self.get_suggestion(func)
        self.assertIn("'ZeroDivisionError'?", actual)

    def test_name_error_suggestions_from_builtins_when_builtins_is_module(self):
        def func():
            custom_globals = globals().copy()
            custom_globals["__builtins__"] = builtins
            print(eval("ZeroDivisionErrrrr", custom_globals))
        actual = self.get_suggestion(func)
        self.assertIn("'ZeroDivisionError'?", actual)

    def test_name_error_suggestions_with_non_string_candidates(self):
        def func():
            abc = 1
            custom_globals = globals().copy()
            custom_globals[0] = 1
            print(eval("abv", custom_globals, locals()))
        actual = self.get_suggestion(func)
        self.assertIn("abc", actual)

    def test_name_error_suggestions_do_not_trigger_for_long_names(self):
        def func():
            somethingverywronghehehehehehe = None
            print(somethingverywronghe)
        actual = self.get_suggestion(func)
        self.assertNotIn("somethingverywronghehe", actual)

    def test_name_error_bad_suggestions_do_not_trigger_for_small_names(self):

        def f_b():
            vvv = mom = w = id = pytho = None
            b

        def f_v():
            vvv = mom = w = id = pytho = None
            v

        def f_m():
            vvv = mom = w = id = pytho = None
            m

        def f_py():
            vvv = mom = w = id = pytho = None
            py

        for name, func in (("b", f_b), ("v", f_v), ("m", f_m), ("py", f_py)):
            with self.subTest(name=name):
                actual = self.get_suggestion(func)
                self.assertNotIn("you mean", actual)
                self.assertNotIn("vvv", actual)
                self.assertNotIn("mom", actual)
                self.assertNotIn("'id'", actual)
                self.assertNotIn("'w'", actual)
                self.assertNotIn("'pytho'", actual)

    def test_name_error_suggestions_do_not_trigger_for_too_many_locals(self):
        def func():
            # Mutating locals() is unreliable, so we need to do it by hand
            a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = a10 = \
            a11 = a12 = a13 = a14 = a15 = a16 = a17 = a18 = a19 = a20 = \
            a21 = a22 = a23 = a24 = a25 = a26 = a27 = a28 = a29 = a30 = \
            a31 = a32 = a33 = a34 = a35 = a36 = a37 = a38 = a39 = a40 = \
            a41 = a42 = a43 = a44 = a45 = a46 = a47 = a48 = a49 = a50 = \
            a51 = a52 = a53 = a54 = a55 = a56 = a57 = a58 = a59 = a60 = \
            a61 = a62 = a63 = a64 = a65 = a66 = a67 = a68 = a69 = a70 = \
            a71 = a72 = a73 = a74 = a75 = a76 = a77 = a78 = a79 = a80 = \
            a81 = a82 = a83 = a84 = a85 = a86 = a87 = a88 = a89 = a90 = \
            a91 = a92 = a93 = a94 = a95 = a96 = a97 = a98 = a99 = a100 = \
            a101 = a102 = a103 = a104 = a105 = a106 = a107 = a108 = a109 = a110 = \
            a111 = a112 = a113 = a114 = a115 = a116 = a117 = a118 = a119 = a120 = \
            a121 = a122 = a123 = a124 = a125 = a126 = a127 = a128 = a129 = a130 = \
            a131 = a132 = a133 = a134 = a135 = a136 = a137 = a138 = a139 = a140 = \
            a141 = a142 = a143 = a144 = a145 = a146 = a147 = a148 = a149 = a150 = \
            a151 = a152 = a153 = a154 = a155 = a156 = a157 = a158 = a159 = a160 = \
            a161 = a162 = a163 = a164 = a165 = a166 = a167 = a168 = a169 = a170 = \
            a171 = a172 = a173 = a174 = a175 = a176 = a177 = a178 = a179 = a180 = \
            a181 = a182 = a183 = a184 = a185 = a186 = a187 = a188 = a189 = a190 = \
            a191 = a192 = a193 = a194 = a195 = a196 = a197 = a198 = a199 = a200 = \
            a201 = a202 = a203 = a204 = a205 = a206 = a207 = a208 = a209 = a210 = \
            a211 = a212 = a213 = a214 = a215 = a216 = a217 = a218 = a219 = a220 = \
            a221 = a222 = a223 = a224 = a225 = a226 = a227 = a228 = a229 = a230 = \
            a231 = a232 = a233 = a234 = a235 = a236 = a237 = a238 = a239 = a240 = \
            a241 = a242 = a243 = a244 = a245 = a246 = a247 = a248 = a249 = a250 = \
            a251 = a252 = a253 = a254 = a255 = a256 = a257 = a258 = a259 = a260 = \
            a261 = a262 = a263 = a264 = a265 = a266 = a267 = a268 = a269 = a270 = \
            a271 = a272 = a273 = a274 = a275 = a276 = a277 = a278 = a279 = a280 = \
            a281 = a282 = a283 = a284 = a285 = a286 = a287 = a288 = a289 = a290 = \
            a291 = a292 = a293 = a294 = a295 = a296 = a297 = a298 = a299 = a300 = \
            a301 = a302 = a303 = a304 = a305 = a306 = a307 = a308 = a309 = a310 = \
            a311 = a312 = a313 = a314 = a315 = a316 = a317 = a318 = a319 = a320 = \
            a321 = a322 = a323 = a324 = a325 = a326 = a327 = a328 = a329 = a330 = \
            a331 = a332 = a333 = a334 = a335 = a336 = a337 = a338 = a339 = a340 = \
            a341 = a342 = a343 = a344 = a345 = a346 = a347 = a348 = a349 = a350 = \
            a351 = a352 = a353 = a354 = a355 = a356 = a357 = a358 = a359 = a360 = \
            a361 = a362 = a363 = a364 = a365 = a366 = a367 = a368 = a369 = a370 = \
            a371 = a372 = a373 = a374 = a375 = a376 = a377 = a378 = a379 = a380 = \
            a381 = a382 = a383 = a384 = a385 = a386 = a387 = a388 = a389 = a390 = \
            a391 = a392 = a393 = a394 = a395 = a396 = a397 = a398 = a399 = a400 = \
            a401 = a402 = a403 = a404 = a405 = a406 = a407 = a408 = a409 = a410 = \
            a411 = a412 = a413 = a414 = a415 = a416 = a417 = a418 = a419 = a420 = \
            a421 = a422 = a423 = a424 = a425 = a426 = a427 = a428 = a429 = a430 = \
            a431 = a432 = a433 = a434 = a435 = a436 = a437 = a438 = a439 = a440 = \
            a441 = a442 = a443 = a444 = a445 = a446 = a447 = a448 = a449 = a450 = \
            a451 = a452 = a453 = a454 = a455 = a456 = a457 = a458 = a459 = a460 = \
            a461 = a462 = a463 = a464 = a465 = a466 = a467 = a468 = a469 = a470 = \
            a471 = a472 = a473 = a474 = a475 = a476 = a477 = a478 = a479 = a480 = \
            a481 = a482 = a483 = a484 = a485 = a486 = a487 = a488 = a489 = a490 = \
            a491 = a492 = a493 = a494 = a495 = a496 = a497 = a498 = a499 = a500 = \
            a501 = a502 = a503 = a504 = a505 = a506 = a507 = a508 = a509 = a510 = \
            a511 = a512 = a513 = a514 = a515 = a516 = a517 = a518 = a519 = a520 = \
            a521 = a522 = a523 = a524 = a525 = a526 = a527 = a528 = a529 = a530 = \
            a531 = a532 = a533 = a534 = a535 = a536 = a537 = a538 = a539 = a540 = \
            a541 = a542 = a543 = a544 = a545 = a546 = a547 = a548 = a549 = a550 = \
            a551 = a552 = a553 = a554 = a555 = a556 = a557 = a558 = a559 = a560 = \
            a561 = a562 = a563 = a564 = a565 = a566 = a567 = a568 = a569 = a570 = \
            a571 = a572 = a573 = a574 = a575 = a576 = a577 = a578 = a579 = a580 = \
            a581 = a582 = a583 = a584 = a585 = a586 = a587 = a588 = a589 = a590 = \
            a591 = a592 = a593 = a594 = a595 = a596 = a597 = a598 = a599 = a600 = \
            a601 = a602 = a603 = a604 = a605 = a606 = a607 = a608 = a609 = a610 = \
            a611 = a612 = a613 = a614 = a615 = a616 = a617 = a618 = a619 = a620 = \
            a621 = a622 = a623 = a624 = a625 = a626 = a627 = a628 = a629 = a630 = \
            a631 = a632 = a633 = a634 = a635 = a636 = a637 = a638 = a639 = a640 = \
            a641 = a642 = a643 = a644 = a645 = a646 = a647 = a648 = a649 = a650 = \
            a651 = a652 = a653 = a654 = a655 = a656 = a657 = a658 = a659 = a660 = \
            a661 = a662 = a663 = a664 = a665 = a666 = a667 = a668 = a669 = a670 = \
            a671 = a672 = a673 = a674 = a675 = a676 = a677 = a678 = a679 = a680 = \
            a681 = a682 = a683 = a684 = a685 = a686 = a687 = a688 = a689 = a690 = \
            a691 = a692 = a693 = a694 = a695 = a696 = a697 = a698 = a699 = a700 = \
            a701 = a702 = a703 = a704 = a705 = a706 = a707 = a708 = a709 = a710 = \
            a711 = a712 = a713 = a714 = a715 = a716 = a717 = a718 = a719 = a720 = \
            a721 = a722 = a723 = a724 = a725 = a726 = a727 = a728 = a729 = a730 = \
            a731 = a732 = a733 = a734 = a735 = a736 = a737 = a738 = a739 = a740 = \
            a741 = a742 = a743 = a744 = a745 = a746 = a747 = a748 = a749 = a750 = \
            a751 = a752 = a753 = a754 = a755 = a756 = a757 = a758 = a759 = a760 = \
            a761 = a762 = a763 = a764 = a765 = a766 = a767 = a768 = a769 = a770 = \
            a771 = a772 = a773 = a774 = a775 = a776 = a777 = a778 = a779 = a780 = \
            a781 = a782 = a783 = a784 = a785 = a786 = a787 = a788 = a789 = a790 = \
            a791 = a792 = a793 = a794 = a795 = a796 = a797 = a798 = a799 = a800 \
                = None
            print(a0)

        actual = self.get_suggestion(func)
        self.assertNotRegex(actual, r"NameError.*a1")

    def test_name_error_with_custom_exceptions(self):
        def func():
            blech = None
            raise NameError()

        actual = self.get_suggestion(func)
        self.assertNotIn("blech", actual)

        def func():
            blech = None
            raise NameError

        actual = self.get_suggestion(func)
        self.assertNotIn("blech", actual)

    def test_name_error_with_instance(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __init__(self):
                    self.blech = None
                def foo(self):
                    blich = 1
                    x = blech

        instance = A()
        actual = self.get_suggestion(instance.foo)
        self.assertIn("self.blech", actual)

    def test_unbound_local_error_with_instance(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __init__(self):
                    self.blech = None
                def foo(self):
                    blich = 1
                    x = blech
                    blech = 1

        instance = A()
        actual = self.get_suggestion(instance.foo)
        self.assertNotIn("self.blech", actual)

    def test_unbound_local_error_with_side_effect(self):
        with torch._dynamo.error_on_graph_break(False):
            # gh-132385
            class A:
                def __getattr__(self, key):
                    if key == 'foo':
                        raise AttributeError('foo')
                    if key == 'spam':
                        raise ValueError('spam')

                def bar(self):
                    foo
                def baz(self):
                    spam

        suggestion = self.get_suggestion(A().bar)
        self.assertNotIn('self.', suggestion)
        self.assertIn("'foo'", suggestion)

        suggestion = self.get_suggestion(A().baz)
        self.assertNotIn('self.', suggestion)
        self.assertIn("'spam'", suggestion)

    def test_unbound_local_error_does_not_match(self):
        def func():
            something = 3
            print(somethong)
            somethong = 3

        actual = self.get_suggestion(func)
        self.assertNotIn("something", actual)

    def test_name_error_for_stdlib_modules(self):
        def func():
            stream = io.StringIO()

        actual = self.get_suggestion(func)
        self.assertIn("forget to import 'io'", actual)

    def test_name_error_for_private_stdlib_modules(self):
        def func():
            stream = _io.StringIO()

        actual = self.get_suggestion(func)
        self.assertIn("forget to import '_io'", actual)



class PurePythonSuggestionFormattingTests(
    PurePythonExceptionFormattingMixin,
    SuggestionFormattingTestBase,
    __TestCase,
):
    """
    Same set of tests as above using the pure Python implementation of
    traceback printing in traceback.py.
    """


@cpython_only
class CPythonSuggestionFormattingTests(
    CAPIExceptionFormattingMixin,
    SuggestionFormattingTestBase,
    __TestCase,
):
    """
    Same set of tests as above but with Python's internal traceback printing.
    """


class MiscTest(__TestCase):

    def test_all(self):
        expected = set()
        denylist = {'print_list'}
        for name in dir(traceback):
            if name.startswith('_') or name in denylist:
                continue
            module_object = getattr(traceback, name)
            if getattr(module_object, '__module__', None) == 'traceback':
                expected.add(name)
        self.assertCountEqual(traceback.__all__, expected)

    def test_levenshtein_distance(self):
        # copied from _testinternalcapi.test_edit_cost
        # to also exercise the Python implementation

        def CHECK(a, b, expected):
            actual = traceback._levenshtein_distance(a, b, 4044)
            self.assertEqual(actual, expected)

        CHECK("", "", 0)
        CHECK("", "a", 2)
        CHECK("a", "A", 1)
        CHECK("Apple", "Aple", 2)
        CHECK("Banana", "B@n@n@", 6)
        CHECK("Cherry", "Cherry!", 2)
        CHECK("---0---", "------", 2)
        CHECK("abc", "y", 6)
        CHECK("aa", "bb", 4)
        CHECK("aaaaa", "AAAAA", 5)
        CHECK("wxyz", "wXyZ", 2)
        CHECK("wxyz", "wXyZ123", 8)
        CHECK("Python", "Java", 12)
        CHECK("Java", "C#", 8)
        CHECK("AbstractFoobarManager", "abstract_foobar_manager", 3+2*2)
        CHECK("CPython", "PyPy", 10)
        CHECK("CPython", "pypy", 11)
        CHECK("AttributeError", "AttributeErrop", 2)
        CHECK("AttributeError", "AttributeErrorTests", 10)
        CHECK("ABA", "AAB", 4)

    @unittest.expectedFailure
    @support.requires_resource('cpu')
    def test_levenshtein_distance_short_circuit(self):
        if not LEVENSHTEIN_DATA_FILE.is_file():
            self.fail(
                f"{LEVENSHTEIN_DATA_FILE} is missing."
                f" Run `make regen-test-levenshtein`"
            )

        with LEVENSHTEIN_DATA_FILE.open("r") as f:
            examples = json.load(f)
        for a, b, expected in examples:
            res1 = traceback._levenshtein_distance(a, b, 1000)
            self.assertEqual(res1, expected, msg=(a, b))

            for threshold in [expected, expected + 1, expected + 2]:
                # big enough thresholds shouldn't change the result
                res2 = traceback._levenshtein_distance(a, b, threshold)
                self.assertEqual(res2, expected, msg=(a, b, threshold))

            for threshold in range(expected):
                # for small thresholds, the only piece of information
                # we receive is "strings not close enough".
                res3 = traceback._levenshtein_distance(a, b, threshold)
                self.assertGreater(res3, threshold, msg=(a, b, threshold))

    @cpython_only
    def test_suggestions_extension(self):
        # Check that the C extension is available
        import _suggestions

        self.assertEqual(
            _suggestions._generate_suggestions(
                ["hello", "world"],
                "hell"
            ),
            "hello"
        )
        self.assertEqual(
            _suggestions._generate_suggestions(
                ["hovercraft"],
                "eels"
            ),
            None
        )

        with torch._dynamo.error_on_graph_break(False):
            # gh-131936: _generate_suggestions() doesn't accept list subclasses
            class MyList(list):
                pass

        with self.assertRaises(TypeError):
            _suggestions._generate_suggestions(MyList(), "")




class TestColorizedTraceback(__TestCase):
    def test_colorized_traceback(self):
        def foo(*args):
            x = {'a':{'b': None}}
            y = x['a']['b']['c']

        def baz2(*args):
            return (lambda *args: foo(*args))(1,2,3,4)

        def baz1(*args):
            return baz2(1,2,3,4)

        def bar():
            return baz1(1,
                    2,3
                    ,4)
        try:
            bar()
        except Exception as e:
            exc = traceback.TracebackException.from_exception(
                e, capture_locals=True
            )
        lines = "".join(exc.format(colorize=True))
        red = _colorize.ANSIColors.RED
        boldr = _colorize.ANSIColors.BOLD_RED
        reset = _colorize.ANSIColors.RESET
        self.assertIn("y = " + red + "x['a']['b']" + reset + boldr + "['c']" + reset, lines)
        self.assertIn("return " + red + "(lambda *args: foo(*args))" + reset + boldr + "(1,2,3,4)" + reset, lines)
        self.assertIn("return (lambda *args: " + red + "foo" + reset + boldr + "(*args)" + reset + ")(1,2,3,4)", lines)
        self.assertIn("return baz2(1,2,3,4)", lines)
        self.assertIn("return baz1(1,\n            2,3\n            ,4)", lines)
        self.assertIn(red + "bar" + reset + boldr + "()" + reset, lines)

    def test_colorized_syntax_error(self):
        try:
            compile("a $ b", "<string>", "exec")
        except SyntaxError as e:
            exc = traceback.TracebackException.from_exception(
                e, capture_locals=True
            )
        actual = "".join(exc.format(colorize=True))
        red = _colorize.ANSIColors.RED
        magenta = _colorize.ANSIColors.MAGENTA
        boldm = _colorize.ANSIColors.BOLD_MAGENTA
        boldr = _colorize.ANSIColors.BOLD_RED
        reset = _colorize.ANSIColors.RESET
        expected = "".join([
        f'  File {magenta}"<string>"{reset}, line {magenta}1{reset}\n',
        f'    a {boldr}${reset} b\n',
        f'      {boldr}^{reset}\n',
        f'{boldm}SyntaxError{reset}: {magenta}invalid syntax{reset}\n']
        )
        self.assertIn(expected, actual)

    def test_colorized_traceback_is_the_default(self):
        def foo():
            1/0

        from _testcapi import exception_print
        try:
            foo()
            self.fail("No exception thrown.")
        except Exception as e:
            with captured_output("stderr") as tbstderr:
                with unittest.mock.patch('_colorize.can_colorize', return_value=True):
                    exception_print(e)
            actual = tbstderr.getvalue().splitlines()

        red = _colorize.ANSIColors.RED
        boldr = _colorize.ANSIColors.BOLD_RED
        magenta = _colorize.ANSIColors.MAGENTA
        boldm = _colorize.ANSIColors.BOLD_MAGENTA
        reset = _colorize.ANSIColors.RESET
        lno_foo = foo.__code__.co_firstlineno
        expected = ['Traceback (most recent call last):',
            f'  File {magenta}"{__file__}"{reset}, '
            f'line {magenta}{lno_foo+5}{reset}, in {magenta}test_colorized_traceback_is_the_default{reset}',
            f'    {red}foo{reset+boldr}(){reset}',
            f'    {red}~~~{reset+boldr}^^{reset}',
            f'  File {magenta}"{__file__}"{reset}, '
            f'line {magenta}{lno_foo+1}{reset}, in {magenta}foo{reset}',
            f'    {red}1{reset+boldr}/{reset+red}0{reset}',
            f'    {red}~{reset+boldr}^{reset+red}~{reset}',
            f'{boldm}ZeroDivisionError{reset}: {magenta}division by zero{reset}']
        self.assertEqual(actual, expected)

    def test_colorized_traceback_from_exception_group(self):
        def foo():
            exceptions = []
            try:
                1 / 0
            except ZeroDivisionError as inner_exc:
                exceptions.append(inner_exc)
            raise ExceptionGroup("test", exceptions)

        try:
            foo()
        except Exception as e:
            exc = traceback.TracebackException.from_exception(
                e, capture_locals=True
            )

        red = _colorize.ANSIColors.RED
        boldr = _colorize.ANSIColors.BOLD_RED
        magenta = _colorize.ANSIColors.MAGENTA
        boldm = _colorize.ANSIColors.BOLD_MAGENTA
        reset = _colorize.ANSIColors.RESET
        lno_foo = foo.__code__.co_firstlineno
        actual = "".join(exc.format(colorize=True)).splitlines()
        expected = [f"  + Exception Group Traceback (most recent call last):",
                   f'  |   File {magenta}"{__file__}"{reset}, line {magenta}{lno_foo+9}{reset}, in {magenta}test_colorized_traceback_from_exception_group{reset}',
                   f'  |     {red}foo{reset}{boldr}(){reset}',
                   f'  |     {red}~~~{reset}{boldr}^^{reset}',
                   f"  |     e = ExceptionGroup('test', [ZeroDivisionError('division by zero')])",
                   f"  |     foo = {foo}",
                   f'  |     self = <{__name__}.TestColorizedTraceback testMethod=test_colorized_traceback_from_exception_group>',
                   f'  |   File {magenta}"{__file__}"{reset}, line {magenta}{lno_foo+6}{reset}, in {magenta}foo{reset}',
                   f'  |     raise ExceptionGroup("test", exceptions)',
                   f"  |     exceptions = [ZeroDivisionError('division by zero')]",
                   f'  | {boldm}ExceptionGroup{reset}: {magenta}test (1 sub-exception){reset}',
                   f'  +-+---------------- 1 ----------------',
                   f'    | Traceback (most recent call last):',
                   f'    |   File {magenta}"{__file__}"{reset}, line {magenta}{lno_foo+3}{reset}, in {magenta}foo{reset}',
                   f'    |     {red}1 {reset}{boldr}/{reset}{red} 0{reset}',
                   f'    |     {red}~~{reset}{boldr}^{reset}{red}~~{reset}',
                   f"    |     exceptions = [ZeroDivisionError('division by zero')]",
                   f'    | {boldm}ZeroDivisionError{reset}: {magenta}division by zero{reset}',
                   f'    +------------------------------------']
        self.assertEqual(actual, expected)

if __name__ == "__main__":
    run_tests()
