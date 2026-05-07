# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_builtin.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    skipIfTorchDynamo,
    run_tests,
)

# ======= END DYNAMO PATCH =======

# Python test set -- built-in functions

import ast
import asyncio
import builtins
import collections
import contextlib
import decimal
import fractions
import gc
import io
import locale
import math
import os
import pickle
import platform
import random
import re
import sys
import traceback
import types
import typing
import unittest
import warnings
from contextlib import ExitStack
from functools import partial
from inspect import CO_COROUTINE
from itertools import product
from textwrap import dedent
from types import AsyncGeneratorType, FunctionType, CellType
from operator import neg
from test import support
from test.support import (cpython_only, swap_attr, maybe_get_event_loop_policy)
from test.support.import_helper import import_module
from test.support.os_helper import (EnvironmentVarGuard, TESTFN, unlink)
from test.support.script_helper import assert_python_ok
from test.support.warnings_helper import check_warnings
from test.support import requires_IEEE_754
from unittest.mock import MagicMock, patch
try:
    import pty, signal
except ImportError:
    pty = signal = None


# Detect evidence of double-rounding: sum() does not always
# get improved accuracy on machines that suffer from double rounding.
x, y = 1e16, 2.9999 # use temporary values to defeat peephole optimizer
HAVE_DOUBLE_ROUNDING = (x + y == 1e16 + 4)

# used as proof of globals being used
A_GLOBAL_VALUE = 123

class Squares:

    def __init__(self, max):
        self.max = max
        self.sofar = []

    def __len__(self): return len(self.sofar)

    def __getitem__(self, i):
        if not 0 <= i < self.max: raise IndexError
        n = len(self.sofar)
        while n <= i:
            self.sofar.append(n*n)
            n += 1
        return self.sofar[i]

class StrSquares:

    def __init__(self, max):
        self.max = max
        self.sofar = []

    def __len__(self):
        return len(self.sofar)

    def __getitem__(self, i):
        if not 0 <= i < self.max:
            raise IndexError
        n = len(self.sofar)
        while n <= i:
            self.sofar.append(str(n*n))
            n += 1
        return self.sofar[i]

class BitBucket:
    def write(self, line):
        pass

test_conv_no_sign = [
        ('0', 0),
        ('1', 1),
        ('9', 9),
        ('10', 10),
        ('99', 99),
        ('100', 100),
        ('314', 314),
        (' 314', 314),
        ('314 ', 314),
        ('  \t\t  314  \t\t  ', 314),
        (repr(sys.maxsize), sys.maxsize),
        ('  1x', ValueError),
        ('  1  ', 1),
        ('  1\02  ', ValueError),
        ('', ValueError),
        (' ', ValueError),
        ('  \t\t  ', ValueError),
        (str(br'\u0663\u0661\u0664 ','raw-unicode-escape'), 314),
        (chr(0x200), ValueError),
]

test_conv_sign = [
        ('0', 0),
        ('1', 1),
        ('9', 9),
        ('10', 10),
        ('99', 99),
        ('100', 100),
        ('314', 314),
        (' 314', ValueError),
        ('314 ', 314),
        ('  \t\t  314  \t\t  ', ValueError),
        (repr(sys.maxsize), sys.maxsize),
        ('  1x', ValueError),
        ('  1  ', ValueError),
        ('  1\02  ', ValueError),
        ('', ValueError),
        (' ', ValueError),
        ('  \t\t  ', ValueError),
        (str(br'\u0663\u0661\u0664 ','raw-unicode-escape'), 314),
        (chr(0x200), ValueError),
]

class TestFailingBool:
    def __bool__(self):
        raise RuntimeError

class TestFailingIter:
    def __iter__(self):
        raise RuntimeError

def filter_char(arg):
    return ord(arg) > ord("d")

def map_char(arg):
    return chr(ord(arg)+1)

class BuiltinTest(CPythonTestCase):
    # Helper to check picklability
    def check_iter_pickle(self, it, seq, proto):
        itorg = it
        d = pickle.dumps(it, proto)
        it = pickle.loads(d)
        self.assertEqual(type(itorg), type(it))
        self.assertEqual(list(it), seq)

        #test the iterator after dropping one from it
        it = pickle.loads(d)
        try:
            next(it)
        except StopIteration:
            return
        d = pickle.dumps(it, proto)
        it = pickle.loads(d)
        self.assertEqual(list(it), seq[1:])

    def test_import(self):
        __import__('sys')
        __import__('time')
        __import__('string')
        __import__(name='sys')
        __import__(name='time', level=0)
        self.assertRaises(ModuleNotFoundError, __import__, 'spamspam')
        self.assertRaises(TypeError, __import__, 1, 2, 3, 4)
        self.assertRaises(ValueError, __import__, '')
        self.assertRaises(TypeError, __import__, 'sys', name='sys')
        # Relative import outside of a package with no __package__ or __spec__ (bpo-37409).
        with self.assertWarns(ImportWarning):
            self.assertRaises(ImportError, __import__, '',
                              {'__package__': None, '__spec__': None, '__name__': '__main__'},
                              locals={}, fromlist=('foo',), level=1)
        # embedded null character
        self.assertRaises(ModuleNotFoundError, __import__, 'string\x00')

    def test_abs(self):
        # int
        self.assertEqual(abs(0), 0)
        self.assertEqual(abs(1234), 1234)
        self.assertEqual(abs(-1234), 1234)
        self.assertTrue(abs(-sys.maxsize-1) > 0)
        # float
        self.assertEqual(abs(0.0), 0.0)
        self.assertEqual(abs(3.14), 3.14)
        self.assertEqual(abs(-3.14), 3.14)
        # str
        self.assertRaises(TypeError, abs, 'a')
        # bool
        self.assertEqual(abs(True), 1)
        self.assertEqual(abs(False), 0)
        # other
        self.assertRaises(TypeError, abs)
        self.assertRaises(TypeError, abs, None)
        with torch._dynamo.error_on_graph_break(False):
            class AbsClass(object):
                def __abs__(self):
                    return -5
        self.assertEqual(abs(AbsClass()), -5)

    def test_all(self):
        self.assertEqual(all([2, 4, 6]), True)
        self.assertEqual(all([2, None, 6]), False)
        self.assertRaises(RuntimeError, all, [2, TestFailingBool(), 6])
        self.assertRaises(RuntimeError, all, TestFailingIter())
        self.assertRaises(TypeError, all, 10)               # Non-iterable
        self.assertRaises(TypeError, all)                   # No args
        self.assertRaises(TypeError, all, [2, 4, 6], [])    # Too many args
        self.assertEqual(all([]), True)                     # Empty iterator
        self.assertEqual(all([0, TestFailingBool()]), False)# Short-circuit
        S = [50, 60]
        self.assertEqual(all(x > 42 for x in S), True)
        S = [50, 40, 60]
        self.assertEqual(all(x > 42 for x in S), False)

    def test_any(self):
        self.assertEqual(any([None, None, None]), False)
        self.assertEqual(any([None, 4, None]), True)
        self.assertRaises(RuntimeError, any, [None, TestFailingBool(), 6])
        self.assertRaises(RuntimeError, any, TestFailingIter())
        self.assertRaises(TypeError, any, 10)               # Non-iterable
        self.assertRaises(TypeError, any)                   # No args
        self.assertRaises(TypeError, any, [2, 4, 6], [])    # Too many args
        self.assertEqual(any([]), False)                    # Empty iterator
        self.assertEqual(any([1, TestFailingBool()]), True) # Short-circuit
        S = [40, 60, 30]
        self.assertEqual(any(x > 42 for x in S), True)
        S = [10, 20, 30]
        self.assertEqual(any(x > 42 for x in S), False)

    def test_ascii(self):
        self.assertEqual(ascii(''), '\'\'')
        self.assertEqual(ascii(0), '0')
        self.assertEqual(ascii(()), '()')
        self.assertEqual(ascii([]), '[]')
        self.assertEqual(ascii({}), '{}')
        a = []
        a.append(a)
        self.assertEqual(ascii(a), '[[...]]')
        a = {}
        a[0] = a
        self.assertEqual(ascii(a), '{0: {...}}')
        # Advanced checks for unicode strings
        def _check_uni(s):
            self.assertEqual(ascii(s), repr(s))
        _check_uni("'")
        _check_uni('"')
        _check_uni('"\'')
        _check_uni('\0')
        _check_uni('\r\n\t .')
        # Unprintable non-ASCII characters
        _check_uni('\x85')
        _check_uni('\u1fff')
        _check_uni('\U00012fff')
        # Lone surrogates
        _check_uni('\ud800')
        _check_uni('\udfff')
        # Issue #9804: surrogates should be joined even for printable
        # wide characters (UCS-2 builds).
        self.assertEqual(ascii('\U0001d121'), "'\\U0001d121'")
        # All together
        s = "'\0\"\n\r\t abcd\x85é\U00012fff\uD800\U0001D121xxx."
        self.assertEqual(ascii(s),
            r"""'\'\x00"\n\r\t abcd\x85\xe9\U00012fff\ud800\U0001d121xxx.'""")

    def test_neg(self):
        x = -sys.maxsize-1
        self.assertTrue(isinstance(x, int))
        self.assertEqual(-x, sys.maxsize+1)

    def test_callable(self):
        self.assertTrue(callable(len))
        self.assertFalse(callable("a"))
        self.assertTrue(callable(callable))
        self.assertTrue(callable(lambda x, y: x + y))
        self.assertFalse(callable(__builtins__))
        def f(): pass
        self.assertTrue(callable(f))

        with torch._dynamo.error_on_graph_break(False):
            class C1:
                def meth(self): pass
        self.assertTrue(callable(C1))
        c = C1()
        self.assertTrue(callable(c.meth))
        self.assertFalse(callable(c))

        # __call__ is looked up on the class, not the instance
        c.__call__ = None
        self.assertFalse(callable(c))
        c.__call__ = lambda self: 0
        self.assertFalse(callable(c))
        del c.__call__
        self.assertFalse(callable(c))

        with torch._dynamo.error_on_graph_break(False):
            class C2(object):
                def __call__(self): pass
        c2 = C2()
        self.assertTrue(callable(c2))
        c2.__call__ = None
        self.assertTrue(callable(c2))
        with torch._dynamo.error_on_graph_break(False):
            class C3(C2): pass
        c3 = C3()
        self.assertTrue(callable(c3))

    def test_chr(self):
        self.assertEqual(chr(0), '\0')
        self.assertEqual(chr(32), ' ')
        self.assertEqual(chr(65), 'A')
        self.assertEqual(chr(97), 'a')
        self.assertEqual(chr(0xff), '\xff')
        self.assertRaises(TypeError, chr)
        self.assertRaises(TypeError, chr, 65.0)
        self.assertEqual(chr(0x0000FFFF), "\U0000FFFF")
        self.assertEqual(chr(0x00010000), "\U00010000")
        self.assertEqual(chr(0x00010001), "\U00010001")
        self.assertEqual(chr(0x000FFFFE), "\U000FFFFE")
        self.assertEqual(chr(0x000FFFFF), "\U000FFFFF")
        self.assertEqual(chr(0x00100000), "\U00100000")
        self.assertEqual(chr(0x00100001), "\U00100001")
        self.assertEqual(chr(0x0010FFFE), "\U0010FFFE")
        self.assertEqual(chr(0x0010FFFF), "\U0010FFFF")
        self.assertRaises(ValueError, chr, -1)
        self.assertRaises(ValueError, chr, 0x00110000)
        self.assertRaises(ValueError, chr, 1<<24)
        self.assertRaises(ValueError, chr, 2**32-1)
        self.assertRaises(ValueError, chr, -2**32)
        self.assertRaises(ValueError, chr, 2**1000)
        self.assertRaises(ValueError, chr, -2**1000)

    def test_cmp(self):
        self.assertTrue(not hasattr(builtins, "cmp"))

    def test_compile(self):
        compile('print(1)\n', '', 'exec')
        bom = b'\xef\xbb\xbf'
        compile(bom + b'print(1)\n', '', 'exec')
        compile(source='pass', filename='?', mode='exec')
        compile(dont_inherit=False, filename='tmp', source='0', mode='eval')
        compile('pass', '?', dont_inherit=True, mode='exec')
        compile(memoryview(b"text"), "name", "exec")
        self.assertRaises(TypeError, compile)
        self.assertRaises(ValueError, compile, 'print(42)\n', '<string>', 'badmode')
        self.assertRaises(ValueError, compile, 'print(42)\n', '<string>', 'single', 0xff)
        self.assertRaises(TypeError, compile, 'pass', '?', 'exec',
                          mode='eval', source='0', filename='tmp')
        compile('print("\xe5")\n', '', 'exec')
        self.assertRaises(SyntaxError, compile, chr(0), 'f', 'exec')
        self.assertRaises(ValueError, compile, str('a = 1'), 'f', 'bad')

        # test the optimize argument

        codestr = '''def f():
        """doc"""
        debug_enabled = False
        if __debug__:
            debug_enabled = True
        try:
            assert False
        except AssertionError:
            return (True, f.__doc__, debug_enabled, __debug__)
        else:
            return (False, f.__doc__, debug_enabled, __debug__)
        '''
        def f(): """doc"""
        values = [(-1, __debug__, f.__doc__, __debug__, __debug__),
                  (0, True, 'doc', True, True),
                  (1, False, 'doc', False, False),
                  (2, False, None, False, False)]
        for optval, *expected in values:
            with self.subTest(optval=optval):
            # test both direct compilation and compilation via AST
                codeobjs = []
                codeobjs.append(compile(codestr, "<test>", "exec", optimize=optval))
                tree = ast.parse(codestr)
                codeobjs.append(compile(tree, "<test>", "exec", optimize=optval))
                for code in codeobjs:
                    ns = {}
                    exec(code, ns)
                    rv = ns['f']()
                    self.assertEqual(rv, tuple(expected))

    def test_compile_top_level_await_no_coro(self):
        """Make sure top level non-await codes get the correct coroutine flags"""
        modes = ('single', 'exec')
        code_samples = [
            '''def f():pass\n''',
            '''[x for x in l]''',
            '''{x for x in l}''',
            '''(x for x in l)''',
            '''{x:x for x in l}''',
        ]
        for mode, code_sample in product(modes, code_samples):
            source = dedent(code_sample)
            co = compile(source,
                            '?',
                            mode,
                            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

            self.assertNotEqual(co.co_flags & CO_COROUTINE, CO_COROUTINE,
                                msg=f"source={source} mode={mode}")


    @unittest.skipIf(
        support.is_emscripten or support.is_wasi,
        "socket.accept is broken"
    )
    def test_compile_top_level_await(self):
        """Test whether code some top level await can be compiled.

        Make sure it compiles only with the PyCF_ALLOW_TOP_LEVEL_AWAIT flag
        set, and make sure the generated code object has the CO_COROUTINE flag
        set in order to execute it with  `await eval(.....)` instead of exec,
        or via a FunctionType.
        """

        # helper function just to check we can run top=level async-for
        async def arange(n):
            for i in range(n):
                yield i

        modes = ('single', 'exec')
        code_samples = [
            '''a = await asyncio.sleep(0, result=1)''',
            '''async for i in arange(1):
                   a = 1''',
            '''async with asyncio.Lock() as l:
                   a = 1''',
            '''a = [x async for x in arange(2)][1]''',
            '''a = 1 in {x async for x in arange(2)}''',
            '''a = {x:1 async for x in arange(1)}[0]''',
            '''a = [x async for x in arange(2) async for x in arange(2)][1]''',
            '''a = [x async for x in (x async for x in arange(5))][1]''',
            '''a, = [1 for x in {x async for x in arange(1)}]''',
            '''a = [await asyncio.sleep(0, x) async for x in arange(2)][1]'''
        ]
        policy = maybe_get_event_loop_policy()
        try:
            for mode, code_sample in product(modes, code_samples):
                source = dedent(code_sample)
                with self.assertRaises(
                        SyntaxError, msg=f"source={source} mode={mode}"):
                    compile(source, '?', mode)

                co = compile(source,
                             '?',
                             mode,
                             flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

                self.assertEqual(co.co_flags & CO_COROUTINE, CO_COROUTINE,
                                 msg=f"source={source} mode={mode}")

                # test we can create and  advance a function type
                globals_ = {'asyncio': asyncio, 'a': 0, 'arange': arange}
                async_f = FunctionType(co, globals_)
                asyncio.run(async_f())
                self.assertEqual(globals_['a'], 1)

                # test we can await-eval,
                globals_ = {'asyncio': asyncio, 'a': 0, 'arange': arange}
                asyncio.run(eval(co, globals_))
                self.assertEqual(globals_['a'], 1)
        finally:
            asyncio.set_event_loop_policy(policy)

    def test_compile_top_level_await_invalid_cases(self):
         # helper function just to check we can run top=level async-for
        async def arange(n):
            for i in range(n):
                yield i

        modes = ('single', 'exec')
        code_samples = [
            '''def f():  await arange(10)\n''',
            '''def f():  [x async for x in arange(10)]\n''',
            '''def f():  [await x async for x in arange(10)]\n''',
            '''def f():
                   async for i in arange(1):
                       a = 1
            ''',
            '''def f():
                   async with asyncio.Lock() as l:
                       a = 1
            '''
        ]
        policy = maybe_get_event_loop_policy()
        try:
            for mode, code_sample in product(modes, code_samples):
                source = dedent(code_sample)
                with self.assertRaises(
                        SyntaxError, msg=f"source={source} mode={mode}"):
                    compile(source, '?', mode)

                with self.assertRaises(
                        SyntaxError, msg=f"source={source} mode={mode}"):
                    co = compile(source,
                             '?',
                             mode,
                             flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        finally:
            asyncio.set_event_loop_policy(policy)


    def test_compile_async_generator(self):
        """
        With the PyCF_ALLOW_TOP_LEVEL_AWAIT flag added in 3.8, we want to
        make sure AsyncGenerators are still properly not marked with the
        CO_COROUTINE flag.
        """
        code = dedent("""async def ticker():
                for i in range(10):
                    yield i
                    await asyncio.sleep(0)""")

        co = compile(code, '?', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        glob = {}
        exec(co, glob)
        self.assertEqual(type(glob['ticker']()), AsyncGeneratorType)

    def test_compile_ast(self):
        args = ("a*(1+2)", "f.py", "exec")
        raw = compile(*args, flags = ast.PyCF_ONLY_AST).body[0]
        opt1 = compile(*args, flags = ast.PyCF_OPTIMIZED_AST).body[0]
        opt2 = compile(ast.parse(args[0]), *args[1:], flags = ast.PyCF_OPTIMIZED_AST).body[0]

        for tree in (raw, opt1, opt2):
            self.assertIsInstance(tree.value, ast.BinOp)
            self.assertIsInstance(tree.value.op, ast.Mult)
            self.assertIsInstance(tree.value.left, ast.Name)
            self.assertEqual(tree.value.left.id, 'a')

        raw_right = raw.value.right  # expect BinOp(1, '+', 2)
        self.assertIsInstance(raw_right, ast.BinOp)
        self.assertIsInstance(raw_right.left, ast.Constant)
        self.assertEqual(raw_right.left.value, 1)
        self.assertIsInstance(raw_right.right, ast.Constant)
        self.assertEqual(raw_right.right.value, 2)

        for opt in [opt1, opt2]:
            opt_right = opt.value.right  # expect Constant(3)
            self.assertIsInstance(opt_right, ast.Constant)
            self.assertEqual(opt_right.value, 3)

    def test_delattr(self):
        sys.spam = 1
        delattr(sys, 'spam')
        self.assertRaises(TypeError, delattr)
        self.assertRaises(TypeError, delattr, sys)
        msg = r"^attribute name must be string, not 'int'$"
        self.assertRaisesRegex(TypeError, msg, delattr, sys, 1)

    def test_dir(self):
        # dir(wrong number of arguments)
        self.assertRaises(TypeError, dir, 42, 42)

        # dir() - local scope
        local_var = 1
        self.assertIn('local_var', dir())

        # dir(module)
        self.assertIn('exit', dir(sys))

        # dir(module_with_invalid__dict__)
        with torch._dynamo.error_on_graph_break(False):
            class Foo(types.ModuleType):
                __dict__ = 8
        f = Foo("foo")
        self.assertRaises(TypeError, dir, f)

        # dir(type)
        self.assertIn("strip", dir(str))
        self.assertNotIn("__mro__", dir(str))

        # dir(obj)
        with torch._dynamo.error_on_graph_break(False):
            class Foo(object):
                def __init__(self):
                    self.x = 7
                    self.y = 8
                    self.z = 9
        f = Foo()
        self.assertIn("y", dir(f))

        # dir(obj_no__dict__)
        with torch._dynamo.error_on_graph_break(False):
            class Foo(object):
                __slots__ = []
        f = Foo()
        self.assertIn("__repr__", dir(f))

        # dir(obj_no__class__with__dict__)
        # (an ugly trick to cause getattr(f, "__class__") to fail)
        with torch._dynamo.error_on_graph_break(False):
            class Foo(object):
                __slots__ = ["__class__", "__dict__"]
                def __init__(self):
                    self.bar = "wow"
        f = Foo()
        self.assertNotIn("__repr__", dir(f))
        self.assertIn("bar", dir(f))

        # dir(obj_using __dir__)
        with torch._dynamo.error_on_graph_break(False):
            class Foo(object):
                def __dir__(self):
                    return ["kan", "ga", "roo"]
        f = Foo()
        self.assertTrue(dir(f) == ["ga", "kan", "roo"])

        # dir(obj__dir__tuple)
        with torch._dynamo.error_on_graph_break(False):
            class Foo(object):
                def __dir__(self):
                    return ("b", "c", "a")
        res = dir(Foo())
        self.assertIsInstance(res, list)
        self.assertTrue(res == ["a", "b", "c"])

        # dir(obj__dir__iterable)
        with torch._dynamo.error_on_graph_break(False):
            class Foo(object):
                def __dir__(self):
                    return {"b", "c", "a"}
        res = dir(Foo())
        self.assertIsInstance(res, list)
        self.assertEqual(sorted(res), ["a", "b", "c"])

        # dir(obj__dir__not_sequence)
        with torch._dynamo.error_on_graph_break(False):
            class Foo(object):
                def __dir__(self):
                    return 7
        f = Foo()
        self.assertRaises(TypeError, dir, f)

        # dir(traceback)
        try:
            raise IndexError
        except IndexError as e:
            self.assertEqual(len(dir(e.__traceback__)), 4)

        # test that object has a __dir__()
        self.assertEqual(sorted([].__dir__()), dir([]))

    def test___ne__(self):
        self.assertFalse(None.__ne__(None))
        self.assertIs(None.__ne__(0), NotImplemented)
        self.assertIs(None.__ne__("abc"), NotImplemented)

    def test_divmod(self):
        self.assertEqual(divmod(12, 7), (1, 5))
        self.assertEqual(divmod(-12, 7), (-2, 2))
        self.assertEqual(divmod(12, -7), (-2, -2))
        self.assertEqual(divmod(-12, -7), (1, -5))

        self.assertEqual(divmod(-sys.maxsize-1, -1), (sys.maxsize+1, 0))

        for num, denom, exp_result in [ (3.25, 1.0, (3.0, 0.25)),
                                        (-3.25, 1.0, (-4.0, 0.75)),
                                        (3.25, -1.0, (-4.0, -0.75)),
                                        (-3.25, -1.0, (3.0, -0.25))]:
            result = divmod(num, denom)
            self.assertAlmostEqual(result[0], exp_result[0])
            self.assertAlmostEqual(result[1], exp_result[1])

        self.assertRaises(TypeError, divmod)

    def test_eval(self):
        self.assertEqual(eval('1+1'), 2)
        self.assertEqual(eval(' 1+1\n'), 2)
        globals = {'a': 1, 'b': 2}
        locals = {'b': 200, 'c': 300}
        self.assertEqual(eval('a', globals) , 1)
        self.assertEqual(eval('a', globals, locals), 1)
        self.assertEqual(eval('b', globals, locals), 200)
        self.assertEqual(eval('c', globals, locals), 300)
        globals = {'a': 1, 'b': 2}
        locals = {'b': 200, 'c': 300}
        bom = b'\xef\xbb\xbf'
        self.assertEqual(eval(bom + b'a', globals, locals), 1)
        self.assertEqual(eval('"\xe5"', globals), "\xe5")
        self.assertRaises(TypeError, eval)
        self.assertRaises(TypeError, eval, ())
        self.assertRaises(SyntaxError, eval, bom[:2] + b'a')

        with torch._dynamo.error_on_graph_break(False):
            class X:
                def __getitem__(self, key):
                    raise ValueError
        self.assertRaises(ValueError, eval, "foo", {}, X())

    def test_eval_kwargs(self):
        data = {"A_GLOBAL_VALUE": 456}
        self.assertEqual(eval("globals()['A_GLOBAL_VALUE']", globals=data), 456)
        self.assertEqual(eval("globals()['A_GLOBAL_VALUE']", locals=data), 123)

    def test_general_eval(self):
        # Tests that general mappings can be used for the locals argument

        with torch._dynamo.error_on_graph_break(False):
            class M:
                "Test mapping interface versus possible calls from eval()."
                def __getitem__(self, key):
                    if key == 'a':
                        return 12
                    raise KeyError
                def keys(self):
                    return list('xyz')

        m = M()
        g = globals()
        self.assertEqual(eval('a', g, m), 12)
        self.assertRaises(NameError, eval, 'b', g, m)
        self.assertEqual(eval('dir()', g, m), list('xyz'))
        self.assertEqual(eval('globals()', g, m), g)
        self.assertEqual(eval('locals()', g, m), m)
        self.assertRaises(TypeError, eval, 'a', m)
        with torch._dynamo.error_on_graph_break(False):
            class A:
                "Non-mapping"
                pass
        m = A()
        self.assertRaises(TypeError, eval, 'a', g, m)

        # Verify that dict subclasses work as well
        with torch._dynamo.error_on_graph_break(False):
            class D(dict):
                def __getitem__(self, key):
                    if key == 'a':
                        return 12
                    return dict.__getitem__(self, key)
                def keys(self):
                    return list('xyz')

        d = D()
        self.assertEqual(eval('a', g, d), 12)
        self.assertRaises(NameError, eval, 'b', g, d)
        self.assertEqual(eval('dir()', g, d), list('xyz'))
        self.assertEqual(eval('globals()', g, d), g)
        self.assertEqual(eval('locals()', g, d), d)

        # Verify locals stores (used by list comps)
        eval('[locals() for i in (2,3)]', g, d)
        eval('[locals() for i in (2,3)]', g, collections.UserDict())

        with torch._dynamo.error_on_graph_break(False):
            class SpreadSheet:
                "Sample application showing nested, calculated lookups."
                _cells = {}
                def __setitem__(self, key, formula):
                    self._cells[key] = formula
                def __getitem__(self, key):
                    return eval(self._cells[key], globals(), self)

        ss = SpreadSheet()
        ss['a1'] = '5'
        ss['a2'] = 'a1*6'
        ss['a3'] = 'a2*7'
        self.assertEqual(ss['a3'], 210)

        # Verify that dir() catches a non-list returned by eval
        # SF bug #1004669
        with torch._dynamo.error_on_graph_break(False):
            class C:
                def __getitem__(self, item):
                    raise KeyError(item)
                def keys(self):
                    return 1 # used to be 'a' but that's no longer an error
        self.assertRaises(TypeError, eval, 'dir()', globals(), C())

    def test_exec(self):
        g = {}
        exec('z = 1', g)
        if '__builtins__' in g:
            del g['__builtins__']
        self.assertEqual(g, {'z': 1})

        exec('z = 1+1', g)
        if '__builtins__' in g:
            del g['__builtins__']
        self.assertEqual(g, {'z': 2})
        g = {}
        l = {}

        with check_warnings():
            warnings.filterwarnings("ignore", "global statement",
                    module="<string>")
            exec('global a; a = 1; b = 2', g, l)
        if '__builtins__' in g:
            del g['__builtins__']
        if '__builtins__' in l:
            del l['__builtins__']
        self.assertEqual((g, l), ({'a': 1}, {'b': 2}))

    def test_exec_kwargs(self):
        g = {}
        exec('global z\nz = 1', globals=g)
        if '__builtins__' in g:
            del g['__builtins__']
        self.assertEqual(g, {'z': 1})

        # if we only set locals, the global assignment will not
        # reach this locals dictionary
        g = {}
        exec('global z\nz = 1', locals=g)
        self.assertEqual(g, {})

    def test_exec_globals(self):
        code = compile("print('Hello World!')", "", "exec")
        # no builtin function
        self.assertRaisesRegex(NameError, "name 'print' is not defined",
                               exec, code, {'__builtins__': {}})
        # __builtins__ must be a mapping type
        self.assertRaises(TypeError,
                          exec, code, {'__builtins__': 123})

    def test_exec_globals_frozen(self):
        with torch._dynamo.error_on_graph_break(False):
            class frozendict_error(Exception):
                pass

            class frozendict(dict):
                def __setitem__(self, key, value):
                    raise frozendict_error("frozendict is readonly")

        # read-only builtins
        if isinstance(__builtins__, types.ModuleType):
            frozen_builtins = frozendict(__builtins__.__dict__)
        else:
            frozen_builtins = frozendict(__builtins__)
        code = compile("__builtins__['superglobal']=2; print(superglobal)", "test", "exec")
        self.assertRaises(frozendict_error,
                          exec, code, {'__builtins__': frozen_builtins})

        # no __build_class__ function
        code = compile("class A: pass", "", "exec")
        self.assertRaisesRegex(NameError, "__build_class__ not found",
                               exec, code, {'__builtins__': {}})
        # __build_class__ in a custom __builtins__
        exec(code, {'__builtins__': frozen_builtins})
        self.assertRaisesRegex(NameError, "__build_class__ not found",
                               exec, code, {'__builtins__': frozendict()})

        # read-only globals
        namespace = frozendict({})
        code = compile("x=1", "test", "exec")
        self.assertRaises(frozendict_error,
                          exec, code, namespace)

    def test_exec_globals_error_on_get(self):
        # custom `globals` or `builtins` can raise errors on item access
        with torch._dynamo.error_on_graph_break(False):
            class setonlyerror(Exception):
                pass

            class setonlydict(dict):
                def __getitem__(self, key):
                    raise setonlyerror

        # globals' `__getitem__` raises
        code = compile("globalname", "test", "exec")
        self.assertRaises(setonlyerror,
                          exec, code, setonlydict({'globalname': 1}))

        # builtins' `__getitem__` raises
        code = compile("superglobal", "test", "exec")
        self.assertRaises(setonlyerror, exec, code,
                          {'__builtins__': setonlydict({'superglobal': 1})})

    def test_exec_globals_dict_subclass(self):
        with torch._dynamo.error_on_graph_break(False):
            class customdict(dict):  # this one should not do anything fancy
                pass

        code = compile("superglobal", "test", "exec")
        # works correctly
        exec(code, {'__builtins__': customdict({'superglobal': 1})})
        # custom builtins dict subclass is missing key
        self.assertRaisesRegex(NameError, "name 'superglobal' is not defined",
                               exec, code, {'__builtins__': customdict()})

    def test_eval_builtins_mapping(self):
        code = compile("superglobal", "test", "eval")
        # works correctly
        ns = {'__builtins__': types.MappingProxyType({'superglobal': 1})}
        self.assertEqual(eval(code, ns), 1)
        # custom builtins mapping is missing key
        ns = {'__builtins__': types.MappingProxyType({})}
        self.assertRaisesRegex(NameError, "name 'superglobal' is not defined",
                               eval, code, ns)

    def test_exec_builtins_mapping_import(self):
        code = compile("import foo.bar", "test", "exec")
        ns = {'__builtins__': types.MappingProxyType({})}
        self.assertRaisesRegex(ImportError, "__import__ not found", exec, code, ns)
        ns = {'__builtins__': types.MappingProxyType({'__import__': lambda *args: args})}
        exec(code, ns)
        self.assertEqual(ns['foo'], ('foo.bar', ns, ns, None, 0))

    def test_eval_builtins_mapping_reduce(self):
        # list_iterator.__reduce__() calls _PyEval_GetBuiltin("iter")
        code = compile("x.__reduce__()", "test", "eval")
        ns = {'__builtins__': types.MappingProxyType({}), 'x': iter([1, 2])}
        self.assertRaisesRegex(AttributeError, "iter", eval, code, ns)
        ns = {'__builtins__': types.MappingProxyType({'iter': iter}), 'x': iter([1, 2])}
        self.assertEqual(eval(code, ns), (iter, ([1, 2],), 0))

    def test_exec_redirected(self):
        savestdout = sys.stdout
        sys.stdout = None # Whatever that cannot flush()
        try:
            # Used to raise SystemError('error return without exception set')
            exec('a')
        except NameError:
            pass
        finally:
            sys.stdout = savestdout

    def test_exec_closure(self):
        def function_without_closures():
            return 3 * 5

        result = 0
        def make_closure_functions():
            a = 2
            b = 3
            c = 5
            def three_freevars():
                nonlocal result
                nonlocal a
                nonlocal b
                result = a*b
            def four_freevars():
                nonlocal result
                nonlocal a
                nonlocal b
                nonlocal c
                result = a*b*c
            return three_freevars, four_freevars
        three_freevars, four_freevars = make_closure_functions()

        # "smoke" test
        result = 0
        exec(three_freevars.__code__,
            three_freevars.__globals__,
            closure=three_freevars.__closure__)
        self.assertEqual(result, 6)

        # should also work with a manually created closure
        result = 0
        my_closure = (CellType(35), CellType(72), three_freevars.__closure__[2])
        exec(three_freevars.__code__,
            three_freevars.__globals__,
            closure=my_closure)
        self.assertEqual(result, 2520)

        # should fail: closure isn't allowed
        # for functions without free vars
        self.assertRaises(TypeError,
            exec,
            function_without_closures.__code__,
            function_without_closures.__globals__,
            closure=my_closure)

        # should fail: closure required but wasn't specified
        self.assertRaises(TypeError,
            exec,
            three_freevars.__code__,
            three_freevars.__globals__,
            closure=None)

        # should fail: closure of wrong length
        self.assertRaises(TypeError,
            exec,
            three_freevars.__code__,
            three_freevars.__globals__,
            closure=four_freevars.__closure__)

        # should fail: closure using a list instead of a tuple
        my_closure = list(my_closure)
        self.assertRaises(TypeError,
            exec,
            three_freevars.__code__,
            three_freevars.__globals__,
            closure=my_closure)
        my_closure = tuple(my_closure)

        # should fail: anything passed to closure= isn't allowed
        # when the source is a string
        self.assertRaises(TypeError,
            exec,
            "pass",
            closure=int)

        # should fail: correct closure= argument isn't allowed
        # when the source is a string
        self.assertRaises(TypeError,
            exec,
            "pass",
            closure=my_closure)

        # should fail: closure tuple with one non-cell-var
        my_closure = list(my_closure)
        my_closure[0] = int
        my_closure = tuple(my_closure)
        self.assertRaises(TypeError,
            exec,
            three_freevars.__code__,
            three_freevars.__globals__,
            closure=my_closure)


    def test_filter(self):
        self.assertEqual(list(filter(lambda c: 'a' <= c <= 'z', 'Hello World')), list('elloorld'))
        self.assertEqual(list(filter(None, [1, 'hello', [], [3], '', None, 9, 0])), [1, 'hello', [3], 9])
        self.assertEqual(list(filter(lambda x: x > 0, [1, -3, 9, 0, 2])), [1, 9, 2])
        self.assertEqual(list(filter(None, Squares(10))), [1, 4, 9, 16, 25, 36, 49, 64, 81])
        self.assertEqual(list(filter(lambda x: x%2, Squares(10))), [1, 9, 25, 49, 81])
        def identity(item):
            return 1
        filter(identity, Squares(5))
        self.assertRaises(TypeError, filter)
        with torch._dynamo.error_on_graph_break(False):
            class BadSeq(object):
                def __getitem__(self, index):
                    if index<4:
                        return 42
                    raise ValueError
        self.assertRaises(ValueError, list, filter(lambda x: x, BadSeq()))
        def badfunc():
            pass
        self.assertRaises(TypeError, list, filter(badfunc, range(5)))

        # test bltinmodule.c::filtertuple()
        self.assertEqual(list(filter(None, (1, 2))), [1, 2])
        self.assertEqual(list(filter(lambda x: x>=3, (1, 2, 3, 4))), [3, 4])
        self.assertRaises(TypeError, list, filter(42, (1, 2)))

    def test_filter_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            f1 = filter(filter_char, "abcdeabcde")
            f2 = filter(filter_char, "abcdeabcde")
            self.check_iter_pickle(f1, list(f2), proto)

    @skipIfTorchDynamo("gc.collect")
    @support.requires_resource('cpu')
    def test_filter_dealloc(self):
        # Tests recursive deallocation of nested filter objects using the
        # thrashcan mechanism. See gh-102356 for more details.
        max_iters = 1000000
        i = filter(bool, range(max_iters))
        for _ in range(max_iters):
            i = filter(bool, i)
        del i
        gc.collect()

    def test_getattr(self):
        self.assertTrue(getattr(sys, 'stdout') is sys.stdout)
        self.assertRaises(TypeError, getattr)
        self.assertRaises(TypeError, getattr, sys)
        msg = r"^attribute name must be string, not 'int'$"
        self.assertRaisesRegex(TypeError, msg, getattr, sys, 1)
        self.assertRaisesRegex(TypeError, msg, getattr, sys, 1, 'spam')
        self.assertRaises(AttributeError, getattr, sys, chr(sys.maxunicode))
        # unicode surrogates are not encodable to the default encoding (utf8)
        self.assertRaises(AttributeError, getattr, 1, "\uDAD1\uD51E")

    def test_hasattr(self):
        self.assertTrue(hasattr(sys, 'stdout'))
        self.assertRaises(TypeError, hasattr)
        self.assertRaises(TypeError, hasattr, sys)
        msg = r"^attribute name must be string, not 'int'$"
        self.assertRaisesRegex(TypeError, msg, hasattr, sys, 1)
        self.assertEqual(False, hasattr(sys, chr(sys.maxunicode)))

        # Check that hasattr propagates all exceptions outside of
        # AttributeError.
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __getattr__(self, what):
                    raise SystemExit
        self.assertRaises(SystemExit, hasattr, A(), "b")
        with torch._dynamo.error_on_graph_break(False):
            class B:
                def __getattr__(self, what):
                    raise ValueError
        self.assertRaises(ValueError, hasattr, B(), "b")

    def test_hash(self):
        hash(None)
        self.assertEqual(hash(1), hash(1))
        self.assertEqual(hash(1), hash(1.0))
        hash('spam')
        self.assertEqual(hash('spam'), hash(b'spam'))
        hash((0,1,2,3))
        def f(): pass
        hash(f)
        self.assertRaises(TypeError, hash, [])
        self.assertRaises(TypeError, hash, {})
        # Bug 1536021: Allow hash to return long objects
        with torch._dynamo.error_on_graph_break(False):
            class X:
                def __hash__(self):
                    return 2**100
        self.assertEqual(type(hash(X())), int)
        with torch._dynamo.error_on_graph_break(False):
            class Z(int):
                def __hash__(self):
                    return self
        self.assertEqual(hash(Z(42)), hash(42))

    def test_hex(self):
        self.assertEqual(hex(16), '0x10')
        self.assertEqual(hex(-16), '-0x10')
        self.assertRaises(TypeError, hex, {})

    def test_id(self):
        id(None)
        id(1)
        id(1.0)
        id('spam')
        id((0,1,2,3))
        id([0,1,2,3])
        id({'spam': 1, 'eggs': 2, 'ham': 3})

    # Test input() later, alphabetized as if it were raw_input

    def test_iter(self):
        self.assertRaises(TypeError, iter)
        self.assertRaises(TypeError, iter, 42, 42)
        lists = [("1", "2"), ["1", "2"], "12"]
        for l in lists:
            i = iter(l)
            self.assertEqual(next(i), '1')
            self.assertEqual(next(i), '2')
            self.assertRaises(StopIteration, next, i)

    def test_isinstance(self):
        with torch._dynamo.error_on_graph_break(False):
            class C:
                pass
            class D(C):
                pass
            class E:
                pass
        c = C()
        d = D()
        e = E()
        self.assertTrue(isinstance(c, C))
        self.assertTrue(isinstance(d, C))
        self.assertTrue(not isinstance(e, C))
        self.assertTrue(not isinstance(c, D))
        self.assertTrue(not isinstance('foo', E))
        self.assertRaises(TypeError, isinstance, E, 'foo')
        self.assertRaises(TypeError, isinstance)

    def test_issubclass(self):
        with torch._dynamo.error_on_graph_break(False):
            class C:
                pass
            class D(C):
                pass
            class E:
                pass
        c = C()
        d = D()
        e = E()
        self.assertTrue(issubclass(D, C))
        self.assertTrue(issubclass(C, C))
        self.assertTrue(not issubclass(C, D))
        self.assertRaises(TypeError, issubclass, 'foo', E)
        self.assertRaises(TypeError, issubclass, E, 'foo')
        self.assertRaises(TypeError, issubclass)

    def test_len(self):
        self.assertEqual(len('123'), 3)
        self.assertEqual(len(()), 0)
        self.assertEqual(len((1, 2, 3, 4)), 4)
        self.assertEqual(len([1, 2, 3, 4]), 4)
        self.assertEqual(len({}), 0)
        self.assertEqual(len({'a':1, 'b': 2}), 2)
        with torch._dynamo.error_on_graph_break(False):
            class BadSeq:
                def __len__(self):
                    raise ValueError
        self.assertRaises(ValueError, len, BadSeq())
        with torch._dynamo.error_on_graph_break(False):
            class InvalidLen:
                def __len__(self):
                    return None
        self.assertRaises(TypeError, len, InvalidLen())
        with torch._dynamo.error_on_graph_break(False):
            class FloatLen:
                def __len__(self):
                    return 4.5
        self.assertRaises(TypeError, len, FloatLen())
        with torch._dynamo.error_on_graph_break(False):
            class NegativeLen:
                def __len__(self):
                    return -10
        self.assertRaises(ValueError, len, NegativeLen())
        with torch._dynamo.error_on_graph_break(False):
            class HugeLen:
                def __len__(self):
                    return sys.maxsize + 1
        self.assertRaises(OverflowError, len, HugeLen())
        with torch._dynamo.error_on_graph_break(False):
            class HugeNegativeLen:
                def __len__(self):
                    return -sys.maxsize-10
        self.assertRaises(ValueError, len, HugeNegativeLen())
        class NoLenMethod(object): pass
        self.assertRaises(TypeError, len, NoLenMethod())

    def test_map(self):
        self.assertEqual(
            list(map(lambda x: x*x, range(1,4))),
            [1, 4, 9]
        )
        try:
            from math import sqrt
        except ImportError:
            def sqrt(x):
                return pow(x, 0.5)
        self.assertEqual(
            list(map(lambda x: list(map(sqrt, x)), [[16, 4], [81, 9]])),
            [[4.0, 2.0], [9.0, 3.0]]
        )
        self.assertEqual(
            list(map(lambda x, y: x+y, [1,3,2], [9,1,4])),
            [10, 4, 6]
        )

        def plus(*v):
            accu = 0
            for i in v: accu = accu + i
            return accu
        self.assertEqual(
            list(map(plus, [1, 3, 7])),
            [1, 3, 7]
        )
        self.assertEqual(
            list(map(plus, [1, 3, 7], [4, 9, 2])),
            [1+4, 3+9, 7+2]
        )
        self.assertEqual(
            list(map(plus, [1, 3, 7], [4, 9, 2], [1, 1, 0])),
            [1+4+1, 3+9+1, 7+2+0]
        )
        self.assertEqual(
            list(map(int, Squares(10))),
            [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        )
        def Max(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return max(a, b)
        self.assertEqual(
            list(map(Max, Squares(3), Squares(2))),
            [0, 1]
        )
        self.assertRaises(TypeError, map)
        self.assertRaises(TypeError, map, lambda x: x, 42)
        with torch._dynamo.error_on_graph_break(False):
            class BadSeq:
                def __iter__(self):
                    raise ValueError
                    yield None
        self.assertRaises(ValueError, list, map(lambda x: x, BadSeq()))
        def badfunc(x):
            raise RuntimeError
        self.assertRaises(RuntimeError, list, map(badfunc, range(5)))

    def test_map_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            m1 = map(map_char, "Is this the real life?")
            m2 = map(map_char, "Is this the real life?")
            self.check_iter_pickle(m1, list(m2), proto)

    def test_max(self):
        self.assertEqual(max('123123'), '3')
        self.assertEqual(max(1, 2, 3), 3)
        self.assertEqual(max((1, 2, 3, 1, 2, 3)), 3)
        self.assertEqual(max([1, 2, 3, 1, 2, 3]), 3)

        self.assertEqual(max(1, 2, 3.0), 3.0)
        self.assertEqual(max(1, 2.0, 3), 3)
        self.assertEqual(max(1.0, 2, 3), 3)

        with self.assertRaisesRegex(
            TypeError,
            'max expected at least 1 argument, got 0'
        ):
            max()

        self.assertRaises(TypeError, max, 42)
        with self.assertRaisesRegex(
            ValueError,
            r'max\(\) iterable argument is empty'
        ):
            max(())
        with torch._dynamo.error_on_graph_break(False):
            class BadSeq:
                def __getitem__(self, index):
                    raise ValueError
        self.assertRaises(ValueError, max, BadSeq())

        for stmt in (
            "max(key=int)",                 # no args
            "max(default=None)",
            "max(1, 2, default=None)",      # require container for default
            "max(default=None, key=int)",
            "max(1, key=int)",              # single arg not iterable
            "max(1, 2, keystone=int)",      # wrong keyword
            "max(1, 2, key=int, abc=int)",  # two many keywords
            "max(1, 2, key=1)",             # keyfunc is not callable
            ):
            try:
                exec(stmt, globals())
            except TypeError:
                pass
            else:
                self.fail(stmt)

        self.assertEqual(max((1,), key=neg), 1)     # one elem iterable
        self.assertEqual(max((1,2), key=neg), 1)    # two elem iterable
        self.assertEqual(max(1, 2, key=neg), 1)     # two elems

        self.assertEqual(max((), default=None), None)    # zero elem iterable
        self.assertEqual(max((1,), default=None), 1)     # one elem iterable
        self.assertEqual(max((1,2), default=None), 2)    # two elem iterable

        self.assertEqual(max((), default=1, key=neg), 1)
        self.assertEqual(max((1, 2), default=3, key=neg), 1)

        self.assertEqual(max((1, 2), key=None), 2)

        data = [random.randrange(200) for i in range(100)]
        keys = dict((elem, random.randrange(50)) for elem in data)
        f = keys.__getitem__
        self.assertEqual(max(data, key=f),
                         sorted(reversed(data), key=f)[-1])

    def test_min(self):
        self.assertEqual(min('123123'), '1')
        self.assertEqual(min(1, 2, 3), 1)
        self.assertEqual(min((1, 2, 3, 1, 2, 3)), 1)
        self.assertEqual(min([1, 2, 3, 1, 2, 3]), 1)

        self.assertEqual(min(1, 2, 3.0), 1)
        self.assertEqual(min(1, 2.0, 3), 1)
        self.assertEqual(min(1.0, 2, 3), 1.0)

        with self.assertRaisesRegex(
            TypeError,
            'min expected at least 1 argument, got 0'
        ):
            min()

        self.assertRaises(TypeError, min, 42)
        with self.assertRaisesRegex(
            ValueError,
            r'min\(\) iterable argument is empty'
        ):
            min(())
        with torch._dynamo.error_on_graph_break(False):
            class BadSeq:
                def __getitem__(self, index):
                    raise ValueError
        self.assertRaises(ValueError, min, BadSeq())

        for stmt in (
            "min(key=int)",                 # no args
            "min(default=None)",
            "min(1, 2, default=None)",      # require container for default
            "min(default=None, key=int)",
            "min(1, key=int)",              # single arg not iterable
            "min(1, 2, keystone=int)",      # wrong keyword
            "min(1, 2, key=int, abc=int)",  # two many keywords
            "min(1, 2, key=1)",             # keyfunc is not callable
            ):
            try:
                exec(stmt, globals())
            except TypeError:
                pass
            else:
                self.fail(stmt)

        self.assertEqual(min((1,), key=neg), 1)     # one elem iterable
        self.assertEqual(min((1,2), key=neg), 2)    # two elem iterable
        self.assertEqual(min(1, 2, key=neg), 2)     # two elems

        self.assertEqual(min((), default=None), None)    # zero elem iterable
        self.assertEqual(min((1,), default=None), 1)     # one elem iterable
        self.assertEqual(min((1,2), default=None), 1)    # two elem iterable

        self.assertEqual(min((), default=1, key=neg), 1)
        self.assertEqual(min((1, 2), default=1, key=neg), 2)

        self.assertEqual(min((1, 2), key=None), 1)

        data = [random.randrange(200) for i in range(100)]
        keys = dict((elem, random.randrange(50)) for elem in data)
        f = keys.__getitem__
        self.assertEqual(min(data, key=f),
                         sorted(data, key=f)[0])

    def test_next(self):
        it = iter(range(2))
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 1)
        self.assertRaises(StopIteration, next, it)
        self.assertRaises(StopIteration, next, it)
        self.assertEqual(next(it, 42), 42)

        with torch._dynamo.error_on_graph_break(False):
            class Iter(object):
                def __iter__(self):
                    return self
                def __next__(self):
                    raise StopIteration

        it = iter(Iter())
        self.assertEqual(next(it, 42), 42)
        self.assertRaises(StopIteration, next, it)

        def gen():
            yield 1
            return

        it = gen()
        self.assertEqual(next(it), 1)
        self.assertRaises(StopIteration, next, it)
        self.assertEqual(next(it, 42), 42)

    def test_oct(self):
        self.assertEqual(oct(100), '0o144')
        self.assertEqual(oct(-100), '-0o144')
        self.assertRaises(TypeError, oct, ())

    def write_testfile(self):
        # NB the first 4 lines are also used to test input, below
        fp = open(TESTFN, 'w', encoding="utf-8")
        self.addCleanup(unlink, TESTFN)
        with fp:
            fp.write('1+1\n')
            fp.write('The quick brown fox jumps over the lazy dog')
            fp.write('.\n')
            fp.write('Dear John\n')
            fp.write('XXX'*100)
            fp.write('YYY'*100)

    def test_open(self):
        self.write_testfile()
        fp = open(TESTFN, encoding="utf-8")
        with fp:
            self.assertEqual(fp.readline(4), '1+1\n')
            self.assertEqual(fp.readline(), 'The quick brown fox jumps over the lazy dog.\n')
            self.assertEqual(fp.readline(4), 'Dear')
            self.assertEqual(fp.readline(100), ' John\n')
            self.assertEqual(fp.read(300), 'XXX'*100)
            self.assertEqual(fp.read(1000), 'YYY'*100)

        # embedded null bytes and characters
        self.assertRaises(ValueError, open, 'a\x00b')
        self.assertRaises(ValueError, open, b'a\x00b')

    @unittest.skipIf(sys.flags.utf8_mode, "utf-8 mode is enabled")
    def test_open_default_encoding(self):
        old_environ = dict(os.environ)
        try:
            # try to get a user preferred encoding different than the current
            # locale encoding to check that open() uses the current locale
            # encoding and not the user preferred encoding
            for key in ('LC_ALL', 'LANG', 'LC_CTYPE'):
                if key in os.environ:
                    del os.environ[key]

            self.write_testfile()
            current_locale_encoding = locale.getencoding()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EncodingWarning)
                fp = open(TESTFN, 'w')
            with fp:
                self.assertEqual(fp.encoding, current_locale_encoding)
        finally:
            os.environ.clear()
            os.environ.update(old_environ)

    @support.requires_subprocess()
    def test_open_non_inheritable(self):
        fileobj = open(__file__, encoding="utf-8")
        with fileobj:
            self.assertFalse(os.get_inheritable(fileobj.fileno()))

    def test_ord(self):
        self.assertEqual(ord(' '), 32)
        self.assertEqual(ord('A'), 65)
        self.assertEqual(ord('a'), 97)
        self.assertEqual(ord('\x80'), 128)
        self.assertEqual(ord('\xff'), 255)

        self.assertEqual(ord(b' '), 32)
        self.assertEqual(ord(b'A'), 65)
        self.assertEqual(ord(b'a'), 97)
        self.assertEqual(ord(b'\x80'), 128)
        self.assertEqual(ord(b'\xff'), 255)

        self.assertEqual(ord(chr(sys.maxunicode)), sys.maxunicode)
        self.assertRaises(TypeError, ord, 42)

        self.assertEqual(ord(chr(0x10FFFF)), 0x10FFFF)
        self.assertEqual(ord("\U0000FFFF"), 0x0000FFFF)
        self.assertEqual(ord("\U00010000"), 0x00010000)
        self.assertEqual(ord("\U00010001"), 0x00010001)
        self.assertEqual(ord("\U000FFFFE"), 0x000FFFFE)
        self.assertEqual(ord("\U000FFFFF"), 0x000FFFFF)
        self.assertEqual(ord("\U00100000"), 0x00100000)
        self.assertEqual(ord("\U00100001"), 0x00100001)
        self.assertEqual(ord("\U0010FFFE"), 0x0010FFFE)
        self.assertEqual(ord("\U0010FFFF"), 0x0010FFFF)

    def test_pow(self):
        self.assertEqual(pow(0,0), 1)
        self.assertEqual(pow(0,1), 0)
        self.assertEqual(pow(1,0), 1)
        self.assertEqual(pow(1,1), 1)

        self.assertEqual(pow(2,0), 1)
        self.assertEqual(pow(2,10), 1024)
        self.assertEqual(pow(2,20), 1024*1024)
        self.assertEqual(pow(2,30), 1024*1024*1024)

        self.assertEqual(pow(-2,0), 1)
        self.assertEqual(pow(-2,1), -2)
        self.assertEqual(pow(-2,2), 4)
        self.assertEqual(pow(-2,3), -8)

        self.assertAlmostEqual(pow(0.,0), 1.)
        self.assertAlmostEqual(pow(0.,1), 0.)
        self.assertAlmostEqual(pow(1.,0), 1.)
        self.assertAlmostEqual(pow(1.,1), 1.)

        self.assertAlmostEqual(pow(2.,0), 1.)
        self.assertAlmostEqual(pow(2.,10), 1024.)
        self.assertAlmostEqual(pow(2.,20), 1024.*1024.)
        self.assertAlmostEqual(pow(2.,30), 1024.*1024.*1024.)

        self.assertAlmostEqual(pow(-2.,0), 1.)
        self.assertAlmostEqual(pow(-2.,1), -2.)
        self.assertAlmostEqual(pow(-2.,2), 4.)
        self.assertAlmostEqual(pow(-2.,3), -8.)

        for x in 2, 2.0:
            for y in 10, 10.0:
                for z in 1000, 1000.0:
                    if isinstance(x, float) or \
                       isinstance(y, float) or \
                       isinstance(z, float):
                        self.assertRaises(TypeError, pow, x, y, z)
                    else:
                        self.assertAlmostEqual(pow(x, y, z), 24.0)

        self.assertAlmostEqual(pow(-1, 0.5), 1j)
        self.assertAlmostEqual(pow(-1, 1/3), 0.5 + 0.8660254037844386j)

        # See test_pow for additional tests for three-argument pow.
        self.assertEqual(pow(-1, -2, 3), 1)
        self.assertRaises(ValueError, pow, 1, 2, 0)

        self.assertRaises(TypeError, pow)

        # Test passing in arguments as keywords.
        self.assertEqual(pow(0, exp=0), 1)
        self.assertEqual(pow(base=2, exp=4), 16)
        self.assertEqual(pow(base=5, exp=2, mod=14), 11)
        twopow = partial(pow, base=2)
        self.assertEqual(twopow(exp=5), 32)
        fifth_power = partial(pow, exp=5)
        self.assertEqual(fifth_power(2), 32)
        mod10 = partial(pow, mod=10)
        self.assertEqual(mod10(2, 6), 4)
        self.assertEqual(mod10(exp=6, base=2), 4)

    def test_input(self):
        self.write_testfile()
        fp = open(TESTFN, encoding="utf-8")
        savestdin = sys.stdin
        savestdout = sys.stdout # Eats the echo
        try:
            sys.stdin = fp
            sys.stdout = BitBucket()
            self.assertEqual(input(), "1+1")
            self.assertEqual(input(), 'The quick brown fox jumps over the lazy dog.')
            self.assertEqual(input('testing\n'), 'Dear John')

            # SF 1535165: don't segfault on closed stdin
            # sys.stdout must be a regular file for triggering
            sys.stdout = savestdout
            sys.stdin.close()
            self.assertRaises(ValueError, input)

            sys.stdout = BitBucket()
            sys.stdin = io.StringIO("NULL\0")
            self.assertRaises(TypeError, input, 42, 42)
            sys.stdin = io.StringIO("    'whitespace'")
            self.assertEqual(input(), "    'whitespace'")
            sys.stdin = io.StringIO()
            self.assertRaises(EOFError, input)

            del sys.stdout
            self.assertRaises(RuntimeError, input, 'prompt')
            del sys.stdin
            self.assertRaises(RuntimeError, input, 'prompt')
        finally:
            sys.stdin = savestdin
            sys.stdout = savestdout
            fp.close()

    def test_input_gh130163(self):
        with torch._dynamo.error_on_graph_break(False):
            class X(io.StringIO):
                def __getattribute__(self, name):
                    nonlocal patch
                    if patch:
                        patch = False
                        sys.stdout = X()
                        sys.stderr = X()
                        sys.stdin = X('input\n')
                        support.gc_collect()
                    return io.StringIO.__getattribute__(self, name)

        with (support.swap_attr(sys, 'stdout', None),
              support.swap_attr(sys, 'stderr', None),
              support.swap_attr(sys, 'stdin', None)):
            patch = False
            # the only references:
            sys.stdout = X()
            sys.stderr = X()
            sys.stdin = X('input\n')
            patch = True
            input()  # should not crash

    # test_int(): see test_int.py for tests of built-in function int().

    def test_repr(self):
        self.assertEqual(repr(''), '\'\'')
        self.assertEqual(repr(0), '0')
        self.assertEqual(repr(()), '()')
        self.assertEqual(repr([]), '[]')
        self.assertEqual(repr({}), '{}')
        a = []
        a.append(a)
        self.assertEqual(repr(a), '[[...]]')
        a = {}
        a[0] = a
        self.assertEqual(repr(a), '{0: {...}}')

    def test_round(self):
        self.assertEqual(round(0.0), 0.0)
        self.assertEqual(type(round(0.0)), int)
        self.assertEqual(round(1.0), 1.0)
        self.assertEqual(round(10.0), 10.0)
        self.assertEqual(round(1000000000.0), 1000000000.0)
        self.assertEqual(round(1e20), 1e20)

        self.assertEqual(round(-1.0), -1.0)
        self.assertEqual(round(-10.0), -10.0)
        self.assertEqual(round(-1000000000.0), -1000000000.0)
        self.assertEqual(round(-1e20), -1e20)

        self.assertEqual(round(0.1), 0.0)
        self.assertEqual(round(1.1), 1.0)
        self.assertEqual(round(10.1), 10.0)
        self.assertEqual(round(1000000000.1), 1000000000.0)

        self.assertEqual(round(-1.1), -1.0)
        self.assertEqual(round(-10.1), -10.0)
        self.assertEqual(round(-1000000000.1), -1000000000.0)

        self.assertEqual(round(0.9), 1.0)
        self.assertEqual(round(9.9), 10.0)
        self.assertEqual(round(999999999.9), 1000000000.0)

        self.assertEqual(round(-0.9), -1.0)
        self.assertEqual(round(-9.9), -10.0)
        self.assertEqual(round(-999999999.9), -1000000000.0)

        self.assertEqual(round(-8.0, -1), -10.0)
        self.assertEqual(type(round(-8.0, -1)), float)

        self.assertEqual(type(round(-8.0, 0)), float)
        self.assertEqual(type(round(-8.0, 1)), float)

        # Check even / odd rounding behaviour
        self.assertEqual(round(5.5), 6)
        self.assertEqual(round(6.5), 6)
        self.assertEqual(round(-5.5), -6)
        self.assertEqual(round(-6.5), -6)

        # Check behavior on ints
        self.assertEqual(round(0), 0)
        self.assertEqual(round(8), 8)
        self.assertEqual(round(-8), -8)
        self.assertEqual(type(round(0)), int)
        self.assertEqual(type(round(-8, -1)), int)
        self.assertEqual(type(round(-8, 0)), int)
        self.assertEqual(type(round(-8, 1)), int)

        # test new kwargs
        self.assertEqual(round(number=-8.0, ndigits=-1), -10.0)

        self.assertRaises(TypeError, round)

        # test generic rounding delegation for reals
        with torch._dynamo.error_on_graph_break(False):
            class TestRound:
                def __round__(self):
                    return 23

            class TestNoRound:
                pass

        self.assertEqual(round(TestRound()), 23)

        self.assertRaises(TypeError, round, 1, 2, 3)
        self.assertRaises(TypeError, round, TestNoRound())

        t = TestNoRound()
        t.__round__ = lambda *args: args
        self.assertRaises(TypeError, round, t)
        self.assertRaises(TypeError, round, t, 0)

    # Some versions of glibc for alpha have a bug that affects
    # float -> integer rounding (floor, ceil, rint, round) for
    # values in the range [2**52, 2**53).  See:
    #
    # We skip this test on Linux/alpha if it would fail.
    linux_alpha = (platform.system().startswith('Linux') and
                   platform.machine().startswith('alpha'))
    system_round_bug = round(5e15+1) != 5e15+1
    @unittest.skipIf(linux_alpha and system_round_bug,
                     "test will fail;  failure is probably due to a "
                     "buggy system round function")
    def test_round_large(self):
        # Issue #1869: integral floats should remain unchanged
        self.assertEqual(round(5e15-1), 5e15-1)
        self.assertEqual(round(5e15), 5e15)
        self.assertEqual(round(5e15+1), 5e15+1)
        self.assertEqual(round(5e15+2), 5e15+2)
        self.assertEqual(round(5e15+3), 5e15+3)

    def test_bug_27936(self):
        # Verify that ndigits=None means the same as passing in no argument
        for x in [1234,
                  1234.56,
                  decimal.Decimal('1234.56'),
                  fractions.Fraction(123456, 100)]:
            self.assertEqual(round(x, None), round(x))
            self.assertEqual(type(round(x, None)), type(round(x)))

    def test_setattr(self):
        setattr(sys, 'spam', 1)
        self.assertEqual(sys.spam, 1)
        self.assertRaises(TypeError, setattr)
        self.assertRaises(TypeError, setattr, sys)
        self.assertRaises(TypeError, setattr, sys, 'spam')
        msg = r"^attribute name must be string, not 'int'$"
        self.assertRaisesRegex(TypeError, msg, setattr, sys, 1, 'spam')

    # test_str(): see test_str.py and test_bytes.py for str() tests.

    def test_sum(self):
        self.assertEqual(sum([]), 0)
        self.assertEqual(sum(list(range(2,8))), 27)
        self.assertEqual(sum(iter(list(range(2,8)))), 27)
        self.assertEqual(sum(Squares(10)), 285)
        self.assertEqual(sum(iter(Squares(10))), 285)
        self.assertEqual(sum([[1], [2], [3]], []), [1, 2, 3])

        self.assertEqual(sum(range(10), 1000), 1045)
        self.assertEqual(sum(range(10), start=1000), 1045)
        self.assertEqual(sum(range(10), 2**31-5), 2**31+40)
        self.assertEqual(sum(range(10), 2**63-5), 2**63+40)

        self.assertEqual(sum(i % 2 != 0 for i in range(10)), 5)
        self.assertEqual(sum((i % 2 != 0 for i in range(10)), 2**31-3),
                         2**31+2)
        self.assertEqual(sum((i % 2 != 0 for i in range(10)), 2**63-3),
                         2**63+2)
        self.assertIs(sum([], False), False)

        self.assertEqual(sum(i / 2 for i in range(10)), 22.5)
        self.assertEqual(sum((i / 2 for i in range(10)), 1000), 1022.5)
        self.assertEqual(sum((i / 2 for i in range(10)), 1000.25), 1022.75)
        self.assertEqual(sum([0.5, 1]), 1.5)
        self.assertEqual(sum([1, 0.5]), 1.5)
        self.assertEqual(repr(sum([-0.0])), '0.0')
        self.assertEqual(repr(sum([-0.0], -0.0)), '-0.0')
        self.assertEqual(repr(sum([], -0.0)), '-0.0')
        self.assertTrue(math.isinf(sum([float("inf"), float("inf")])))
        self.assertTrue(math.isinf(sum([1e308, 1e308])))

        self.assertRaises(TypeError, sum)
        self.assertRaises(TypeError, sum, 42)
        self.assertRaises(TypeError, sum, ['a', 'b', 'c'])
        self.assertRaises(TypeError, sum, ['a', 'b', 'c'], '')
        self.assertRaises(TypeError, sum, [b'a', b'c'], b'')
        values = [bytearray(b'a'), bytearray(b'b')]
        self.assertRaises(TypeError, sum, values, bytearray(b''))
        self.assertRaises(TypeError, sum, [[1], [2], [3]])
        self.assertRaises(TypeError, sum, [{2:3}])
        self.assertRaises(TypeError, sum, [{2:3}]*2, {2:3})
        self.assertRaises(TypeError, sum, [], '')
        self.assertRaises(TypeError, sum, [], b'')
        self.assertRaises(TypeError, sum, [], bytearray())

        with torch._dynamo.error_on_graph_break(False):
            class BadSeq:
                def __getitem__(self, index):
                    raise ValueError
        self.assertRaises(ValueError, sum, BadSeq())

        empty = []
        sum(([x] for x in range(10)), empty)
        self.assertEqual(empty, [])

    @requires_IEEE_754
    @unittest.skipIf(HAVE_DOUBLE_ROUNDING,
                         "sum accuracy not guaranteed on machines with double rounding")
    @support.cpython_only    # Other implementations may choose a different algorithm
    def test_sum_accuracy(self):
        self.assertEqual(sum([0.1] * 10), 1.0)
        self.assertEqual(sum([1.0, 10E100, 1.0, -10E100]), 2.0)

    def test_type(self):
        self.assertEqual(type(''),  type('123'))
        self.assertNotEqual(type(''), type(()))

    # We don't want self in vars(), so these are static methods

    @staticmethod
    def get_vars_f0():
        return vars()

    @staticmethod
    def get_vars_f2():
        BuiltinTest.get_vars_f0()
        a = 1
        b = 2
        return vars()

    with torch._dynamo.error_on_graph_break(False):
        class C_get_vars(object):
            def getDict(self):
                return {'a':2}
            __dict__ = property(fget=getDict)

    def test_vars(self):
        self.assertEqual(set(vars()), set(dir()))
        self.assertEqual(set(vars(sys)), set(dir(sys)))
        self.assertEqual(self.get_vars_f0(), {})
        self.assertEqual(self.get_vars_f2(), {'a': 1, 'b': 2})
        self.assertRaises(TypeError, vars, 42, 42)
        self.assertRaises(TypeError, vars, 42)
        self.assertEqual(vars(self.C_get_vars()), {'a':2})

    def iter_error(self, iterable, error):
        """Collect `iterable` into a list, catching an expected `error`."""
        items = []
        with self.assertRaises(error):
            for item in iterable:
                items.append(item)
        return items

    def test_zip(self):
        a = (1, 2, 3)
        b = (4, 5, 6)
        t = [(1, 4), (2, 5), (3, 6)]
        self.assertEqual(list(zip(a, b)), t)
        b = [4, 5, 6]
        self.assertEqual(list(zip(a, b)), t)
        b = (4, 5, 6, 7)
        self.assertEqual(list(zip(a, b)), t)
        with torch._dynamo.error_on_graph_break(False):
            class I:
                def __getitem__(self, i):
                    if i < 0 or i > 2: raise IndexError
                    return i + 4
        self.assertEqual(list(zip(a, I())), t)
        self.assertEqual(list(zip()), [])
        self.assertEqual(list(zip(*[])), [])
        self.assertRaises(TypeError, zip, None)
        with torch._dynamo.error_on_graph_break(False):
            class G:
                pass
        self.assertRaises(TypeError, zip, a, G())
        self.assertRaises(RuntimeError, zip, a, TestFailingIter())

        # Make sure zip doesn't try to allocate a billion elements for the
        # result list when one of its arguments doesn't say how long it is.
        # A MemoryError is the most likely failure mode.
        with torch._dynamo.error_on_graph_break(False):
            class SequenceWithoutALength:
                def __getitem__(self, i):
                    if i == 5:
                        raise IndexError
                    else:
                        return i
        self.assertEqual(
            list(zip(SequenceWithoutALength(), range(2**30))),
            list(enumerate(range(5)))
        )

        with torch._dynamo.error_on_graph_break(False):
            class BadSeq:
                def __getitem__(self, i):
                    if i == 5:
                        raise ValueError
                    else:
                        return i
        self.assertRaises(ValueError, list, zip(BadSeq(), BadSeq()))

    def test_zip_pickle(self):
        a = (1, 2, 3)
        b = (4, 5, 6)
        t = [(1, 4), (2, 5), (3, 6)]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z1 = zip(a, b)
            self.check_iter_pickle(z1, t, proto)

    def test_zip_pickle_strict(self):
        a = (1, 2, 3)
        b = (4, 5, 6)
        t = [(1, 4), (2, 5), (3, 6)]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z1 = zip(a, b, strict=True)
            self.check_iter_pickle(z1, t, proto)

    def test_zip_pickle_strict_fail(self):
        a = (1, 2, 3)
        b = (4, 5, 6, 7)
        t = [(1, 4), (2, 5), (3, 6)]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z1 = zip(a, b, strict=True)
            z2 = pickle.loads(pickle.dumps(z1, proto))
            self.assertEqual(self.iter_error(z1, ValueError), t)
            self.assertEqual(self.iter_error(z2, ValueError), t)

    def test_zip_bad_iterable(self):
        exception = TypeError()

        with torch._dynamo.error_on_graph_break(False):
            class BadIterable:
                def __iter__(self):
                    raise exception

        with self.assertRaises(TypeError) as cm:
            zip(BadIterable())

        self.assertIs(cm.exception, exception)

    def test_zip_strict(self):
        self.assertEqual(tuple(zip((1, 2, 3), 'abc', strict=True)),
                         ((1, 'a'), (2, 'b'), (3, 'c')))
        self.assertRaises(ValueError, tuple,
                          zip((1, 2, 3, 4), 'abc', strict=True))
        self.assertRaises(ValueError, tuple,
                          zip((1, 2), 'abc', strict=True))
        self.assertRaises(ValueError, tuple,
                          zip((1, 2), (1, 2), 'abc', strict=True))

    def test_zip_strict_iterators(self):
        x = iter(range(5))
        y = [0]
        z = iter(range(5))
        self.assertRaises(ValueError, list,
                          (zip(x, y, z, strict=True)))
        self.assertEqual(next(x), 2)
        self.assertEqual(next(z), 1)

    def test_zip_strict_error_handling(self):
        with torch._dynamo.error_on_graph_break(False):
            class Error(Exception):
                pass

            class Iter:
                def __init__(self, size):
                    self.size = size
                def __iter__(self):
                    return self
                def __next__(self):
                    self.size -= 1
                    if self.size < 0:
                        raise Error
                    return self.size

        l1 = self.iter_error(zip("AB", Iter(1), strict=True), Error)
        self.assertEqual(l1, [("A", 0)])
        l2 = self.iter_error(zip("AB", Iter(2), "A", strict=True), ValueError)
        self.assertEqual(l2, [("A", 1, "A")])
        l3 = self.iter_error(zip("AB", Iter(2), "ABC", strict=True), Error)
        self.assertEqual(l3, [("A", 1, "A"), ("B", 0, "B")])
        l4 = self.iter_error(zip("AB", Iter(3), strict=True), ValueError)
        self.assertEqual(l4, [("A", 2), ("B", 1)])
        l5 = self.iter_error(zip(Iter(1), "AB", strict=True), Error)
        self.assertEqual(l5, [(0, "A")])
        l6 = self.iter_error(zip(Iter(2), "A", strict=True), ValueError)
        self.assertEqual(l6, [(1, "A")])
        l7 = self.iter_error(zip(Iter(2), "ABC", strict=True), Error)
        self.assertEqual(l7, [(1, "A"), (0, "B")])
        l8 = self.iter_error(zip(Iter(3), "AB", strict=True), ValueError)
        self.assertEqual(l8, [(2, "A"), (1, "B")])

    def test_zip_strict_error_handling_stopiteration(self):
        with torch._dynamo.error_on_graph_break(False):
            class Iter:
                def __init__(self, size):
                    self.size = size
                def __iter__(self):
                    return self
                def __next__(self):
                    self.size -= 1
                    if self.size < 0:
                        raise StopIteration
                    return self.size

        l1 = self.iter_error(zip("AB", Iter(1), strict=True), ValueError)
        self.assertEqual(l1, [("A", 0)])
        l2 = self.iter_error(zip("AB", Iter(2), "A", strict=True), ValueError)
        self.assertEqual(l2, [("A", 1, "A")])
        l3 = self.iter_error(zip("AB", Iter(2), "ABC", strict=True), ValueError)
        self.assertEqual(l3, [("A", 1, "A"), ("B", 0, "B")])
        l4 = self.iter_error(zip("AB", Iter(3), strict=True), ValueError)
        self.assertEqual(l4, [("A", 2), ("B", 1)])
        l5 = self.iter_error(zip(Iter(1), "AB", strict=True), ValueError)
        self.assertEqual(l5, [(0, "A")])
        l6 = self.iter_error(zip(Iter(2), "A", strict=True), ValueError)
        self.assertEqual(l6, [(1, "A")])
        l7 = self.iter_error(zip(Iter(2), "ABC", strict=True), ValueError)
        self.assertEqual(l7, [(1, "A"), (0, "B")])
        l8 = self.iter_error(zip(Iter(3), "AB", strict=True), ValueError)
        self.assertEqual(l8, [(2, "A"), (1, "B")])

    @support.cpython_only
    def test_zip_result_gc(self):
        # bpo-42536: zip's tuple-reuse speed trick breaks the GC's assumptions
        # about what can be untracked. Make sure we re-track result tuples
        # whenever we reuse them.
        it = zip([[]])
        gc.collect()
        # That GC collection probably untracked the recycled internal result
        # tuple, which is initialized to (None,). Make sure it's re-tracked when
        # it's mutated and returned from __next__:
        self.assertTrue(gc.is_tracked(next(it)))

    def test_format(self):
        # Test the basic machinery of the format() builtin.  Don't test
        #  the specifics of the various formatters
        self.assertEqual(format(3, ''), '3')

        # Returns some classes to use for various tests.  There's
        #  an old-style version, and a new-style version
        def classes_new():
            with torch._dynamo.error_on_graph_break(False):
                class A(object):
                    def __init__(self, x):
                        self.x = x
                    def __format__(self, format_spec):
                        return str(self.x) + format_spec
                class DerivedFromA(A):
                    pass

                class Simple(object): pass
                class DerivedFromSimple(Simple):
                    def __init__(self, x):
                        self.x = x
                    def __format__(self, format_spec):
                        return str(self.x) + format_spec
                class DerivedFromSimple2(DerivedFromSimple): pass
            return A, DerivedFromA, DerivedFromSimple, DerivedFromSimple2

        def class_test(A, DerivedFromA, DerivedFromSimple, DerivedFromSimple2):
            self.assertEqual(format(A(3), 'spec'), '3spec')
            self.assertEqual(format(DerivedFromA(4), 'spec'), '4spec')
            self.assertEqual(format(DerivedFromSimple(5), 'abc'), '5abc')
            self.assertEqual(format(DerivedFromSimple2(10), 'abcdef'),
                             '10abcdef')

        class_test(*classes_new())

        def empty_format_spec(value):
            # test that:
            #  format(x, '') == str(x)
            #  format(x) == str(x)
            self.assertEqual(format(value, ""), str(value))
            self.assertEqual(format(value), str(value))

        # for builtin types, format(x, "") == str(x)
        empty_format_spec(17**13)
        empty_format_spec(1.0)
        empty_format_spec(3.1415e104)
        empty_format_spec(-3.1415e104)
        empty_format_spec(3.1415e-104)
        empty_format_spec(-3.1415e-104)
        empty_format_spec(object)
        empty_format_spec(None)

        # TypeError because self.__format__ returns the wrong type
        with torch._dynamo.error_on_graph_break(False):
            class BadFormatResult:
                def __format__(self, format_spec):
                    return 1.0
        self.assertRaises(TypeError, format, BadFormatResult(), "")

        # TypeError because format_spec is not unicode or str
        self.assertRaises(TypeError, format, object(), 4)
        self.assertRaises(TypeError, format, object(), object())

        # tests for object.__format__ really belong elsewhere, but
        #  there's no good place to put them
        x = object().__format__('')
        self.assertTrue(x.startswith('<object object at'))

        # first argument to object.__format__ must be string
        self.assertRaises(TypeError, object().__format__, 3)
        self.assertRaises(TypeError, object().__format__, object())
        self.assertRaises(TypeError, object().__format__, None)

        # --------------------------------------------------------------------
        # Issue #7994: object.__format__ with a non-empty format string is
        # disallowed
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __format__(self, fmt_str):
                    return format('', fmt_str)

        self.assertEqual(format(A()), '')
        self.assertEqual(format(A(), ''), '')
        self.assertEqual(format(A(), 's'), '')

        with torch._dynamo.error_on_graph_break(False):
            class B:
                pass

            class C(object):
                pass

        for cls in [object, B, C]:
            obj = cls()
            self.assertEqual(format(obj), str(obj))
            self.assertEqual(format(obj, ''), str(obj))
            with self.assertRaisesRegex(TypeError,
                                        r'\b%s\b' % re.escape(cls.__name__)):
                format(obj, 's')
        # --------------------------------------------------------------------

        # make sure we can take a subclass of str as a format spec
        with torch._dynamo.error_on_graph_break(False):
            class DerivedFromStr(str): pass
        self.assertEqual(format(0, DerivedFromStr('10')), '         0')

    def test_bin(self):
        self.assertEqual(bin(0), '0b0')
        self.assertEqual(bin(1), '0b1')
        self.assertEqual(bin(-1), '-0b1')
        self.assertEqual(bin(2**65), '0b1' + '0' * 65)
        self.assertEqual(bin(2**65-1), '0b' + '1' * 65)
        self.assertEqual(bin(-(2**65)), '-0b1' + '0' * 65)
        self.assertEqual(bin(-(2**65-1)), '-0b' + '1' * 65)

    def test_bytearray_translate(self):
        x = bytearray(b"abc")
        self.assertRaises(ValueError, x.translate, b"1", 1)
        self.assertRaises(TypeError, x.translate, b"1"*256, 1)

    def test_bytearray_extend_error(self):
        array = bytearray()
        bad_iter = map(int, "X")
        self.assertRaises(ValueError, array.extend, bad_iter)

    def test_bytearray_join_with_misbehaving_iterator(self):
        # Issue #112625
        array = bytearray(b',')
        def iterator():
            array.clear()
            yield b'A'
            yield b'B'
        self.assertRaises(BufferError, array.join, iterator())

    def test_bytearray_join_with_custom_iterator(self):
        # Issue #112625
        array = bytearray(b',')
        def iterator():
            yield b'A'
            yield b'B'
        self.assertEqual(bytearray(b'A,B'), array.join(iterator()))

    def test_construct_singletons(self):
        for const in None, Ellipsis, NotImplemented:
            tp = type(const)
            self.assertIs(tp(), const)
            self.assertRaises(TypeError, tp, 1, 2)
            self.assertRaises(TypeError, tp, a=1, b=2)

    def test_warning_notimplemented(self):
        # Issue #35712: NotImplemented is a sentinel value that should never
        # be evaluated in a boolean context (virtually all such use cases
        # are a result of accidental misuse implementing rich comparison
        # operations in terms of one another).
        # For the time being, it will continue to evaluate as a true value, but
        # issue a deprecation warning (with the eventual intent to make it
        # a TypeError).
        self.assertWarns(DeprecationWarning, bool, NotImplemented)
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(NotImplemented)
        with self.assertWarns(DeprecationWarning):
            self.assertFalse(not NotImplemented)

    def test_singleton_attribute_access(self):
        for singleton in (NotImplemented, Ellipsis):
            with self.subTest(singleton):
                self.assertIs(type(singleton), singleton.__class__)
                self.assertIs(type(singleton).__class__, type)

                # Missing instance attributes:
                with self.assertRaises(AttributeError):
                    singleton.prop = 1
                with self.assertRaises(AttributeError):
                    singleton.prop

                # Missing class attributes:
                with self.assertRaises(TypeError):
                    type(singleton).prop = 1
                with self.assertRaises(AttributeError):
                    type(singleton).prop


class TestBreakpoint(CPythonTestCase):
    def setUp(self):
        # These tests require a clean slate environment.  For example, if the
        # test suite is run with $PYTHONBREAKPOINT set to something else, it
        # will mess up these tests.  Similarly for sys.breakpointhook.
        # Cleaning the slate here means you can't use breakpoint() to debug
        # these tests, but I think that's okay.  Just use pdb.set_trace() if
        # you must.
        super().setUp()
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.env = self.resources.enter_context(EnvironmentVarGuard())
        del self.env['PYTHONBREAKPOINT']
        self.resources.enter_context(
            swap_attr(sys, 'breakpointhook', sys.__breakpointhook__))

    def test_breakpoint(self):
        with patch('pdb.set_trace') as mock:
            breakpoint()
        mock.assert_called_once()

    def test_breakpoint_with_breakpointhook_set(self):
        my_breakpointhook = MagicMock()
        sys.breakpointhook = my_breakpointhook
        breakpoint()
        my_breakpointhook.assert_called_once_with()

    def test_breakpoint_with_breakpointhook_reset(self):
        my_breakpointhook = MagicMock()
        sys.breakpointhook = my_breakpointhook
        breakpoint()
        my_breakpointhook.assert_called_once_with()
        # Reset the hook and it will not be called again.
        sys.breakpointhook = sys.__breakpointhook__
        with patch('pdb.set_trace') as mock:
            breakpoint()
            mock.assert_called_once_with()
        my_breakpointhook.assert_called_once_with()

    def test_breakpoint_with_args_and_keywords(self):
        my_breakpointhook = MagicMock()
        sys.breakpointhook = my_breakpointhook
        breakpoint(1, 2, 3, four=4, five=5)
        my_breakpointhook.assert_called_once_with(1, 2, 3, four=4, five=5)

    def test_breakpoint_with_passthru_error(self):
        def my_breakpointhook():
            pass
        sys.breakpointhook = my_breakpointhook
        self.assertRaises(TypeError, breakpoint, 1, 2, 3, four=4, five=5)

    @unittest.skipIf(sys.flags.ignore_environment, '-E was given')
    def test_envar_good_path_builtin(self):
        self.env['PYTHONBREAKPOINT'] = 'int'
        with patch('builtins.int') as mock:
            breakpoint('7')
            mock.assert_called_once_with('7')

    @unittest.skipIf(sys.flags.ignore_environment, '-E was given')
    def test_envar_good_path_other(self):
        self.env['PYTHONBREAKPOINT'] = 'sys.exit'
        with patch('sys.exit') as mock:
            breakpoint()
            mock.assert_called_once_with()

    @unittest.skipIf(sys.flags.ignore_environment, '-E was given')
    def test_envar_good_path_noop_0(self):
        self.env['PYTHONBREAKPOINT'] = '0'
        with patch('pdb.set_trace') as mock:
            breakpoint()
            mock.assert_not_called()

    def test_envar_good_path_empty_string(self):
        # PYTHONBREAKPOINT='' is the same as it not being set.
        self.env['PYTHONBREAKPOINT'] = ''
        with patch('pdb.set_trace') as mock:
            breakpoint()
            mock.assert_called_once_with()

    @unittest.skipIf(sys.flags.ignore_environment, '-E was given')
    def test_envar_unimportable(self):
        for envar in (
                '.', '..', '.foo', 'foo.', '.int', 'int.',
                '.foo.bar', '..foo.bar', '/./',
                'nosuchbuiltin',
                'nosuchmodule.nosuchcallable',
                ):
            with self.subTest(envar=envar):
                self.env['PYTHONBREAKPOINT'] = envar
                mock = self.resources.enter_context(patch('pdb.set_trace'))
                w = self.resources.enter_context(check_warnings(quiet=True))
                breakpoint()
                self.assertEqual(
                    str(w.message),
                    f'Ignoring unimportable $PYTHONBREAKPOINT: "{envar}"')
                self.assertEqual(w.category, RuntimeWarning)
                mock.assert_not_called()

    def test_envar_ignored_when_hook_is_set(self):
        self.env['PYTHONBREAKPOINT'] = 'sys.exit'
        with patch('sys.exit') as mock:
            sys.breakpointhook = int
            breakpoint()
            mock.assert_not_called()

    def test_runtime_error_when_hook_is_lost(self):
        del sys.breakpointhook
        with self.assertRaises(RuntimeError):
            breakpoint()


@unittest.skipUnless(pty, "the pty and signal modules must be available")
class PtyTests(CPythonTestCase):
    """Tests that use a pseudo terminal to guarantee stdin and stdout are
    terminals in the test environment"""

    @staticmethod
    def handle_sighup(signum, frame):
        # bpo-40140: if the process is the session leader, os.close(fd)
        # of "pid, fd = pty.fork()" can raise SIGHUP signal:
        # just ignore the signal.
        pass

    def run_child(self, child, terminal_input):
        old_sighup = signal.signal(signal.SIGHUP, self.handle_sighup)
        try:
            return self._run_child(child, terminal_input)
        finally:
            signal.signal(signal.SIGHUP, old_sighup)

    def _run_child(self, child, terminal_input):
        r, w = os.pipe()  # Pipe test results from child back to parent
        try:
            pid, fd = pty.fork()
        except (OSError, AttributeError) as e:
            os.close(r)
            os.close(w)
            self.skipTest("pty.fork() raised {}".format(e))
            raise

        if pid == 0:
            # Child
            try:
                os.close(r)
                with open(w, "w") as wpipe:
                    child(wpipe)
            except:
                traceback.print_exc()
            finally:
                # We don't want to return to unittest...
                os._exit(0)

        # Parent
        os.close(w)
        os.write(fd, terminal_input)

        # Get results from the pipe
        with open(r, encoding="utf-8") as rpipe:
            lines = []
            while True:
                line = rpipe.readline().strip()
                if line == "":
                    # The other end was closed => the child exited
                    break
                lines.append(line)

        # Check the result was got and corresponds to the user's terminal input
        if len(lines) != 2:
            # Something went wrong, try to get at stderr
            # Beware of Linux raising EIO when the slave is closed
            child_output = bytearray()
            while True:
                try:
                    chunk = os.read(fd, 3000)
                except OSError:  # Assume EIO
                    break
                if not chunk:
                    break
                child_output.extend(chunk)
            os.close(fd)
            child_output = child_output.decode("ascii", "ignore")
            self.fail("got %d lines in pipe but expected 2, child output was:\n%s"
                      % (len(lines), child_output))

        # bpo-40155: Close the PTY before waiting for the child process
        # completion, otherwise the child process hangs on AIX.
        os.close(fd)

        support.wait_process(pid, exitcode=0)

        return lines

    def check_input_tty(self, prompt, terminal_input, stdio_encoding=None, *,
                        expected=None,
                        stdin_errors='surrogateescape',
                        stdout_errors='replace'):
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            self.skipTest("stdin and stdout must be ttys")
        def child(wpipe):
            # Check the error handlers are accounted for
            if stdio_encoding:
                sys.stdin = io.TextIOWrapper(sys.stdin.detach(),
                                             encoding=stdio_encoding,
                                             errors=stdin_errors)
                sys.stdout = io.TextIOWrapper(sys.stdout.detach(),
                                              encoding=stdio_encoding,
                                              errors=stdout_errors)
            print("tty =", sys.stdin.isatty() and sys.stdout.isatty(), file=wpipe)
            try:
                print(ascii(input(prompt)), file=wpipe)
            except BaseException as e:
                print(ascii(f'{e.__class__.__name__}: {e!s}'), file=wpipe)
        with self.detach_readline():
            lines = self.run_child(child, terminal_input + b"\r\n")
        # Check we did exercise the GNU readline path
        self.assertIn(lines[0], {'tty = True', 'tty = False'})
        if lines[0] != 'tty = True':
            self.skipTest("standard IO in should have been a tty")
        input_result = eval(lines[1])   # ascii() -> eval() roundtrip
        if expected is None:
            if stdio_encoding:
                expected = terminal_input.decode(stdio_encoding, 'surrogateescape')
            else:
                expected = terminal_input.decode(sys.stdin.encoding)  # what else?
        self.assertEqual(input_result, expected)

    @contextlib.contextmanager
    def detach_readline(self):
        # bpo-13886: When the readline module is loaded, PyOS_Readline() uses
        # the readline implementation. In some cases, the Python readline
        # callback rlhandler() is called by readline with a string without
        # non-ASCII characters.
        # Unlink readline temporarily from PyOS_Readline() for those tests,
        # since test_builtin is not intended to test
        # the readline module, but the builtins module.
        if "readline" in sys.modules:
            c = import_module("ctypes")
            fp_api = "PyOS_ReadlineFunctionPointer"
            prev_value = c.c_void_p.in_dll(c.pythonapi, fp_api).value
            c.c_void_p.in_dll(c.pythonapi, fp_api).value = None
            try:
                yield
            finally:
                c.c_void_p.in_dll(c.pythonapi, fp_api).value = prev_value
        else:
            yield

    def test_input_tty(self):
        # Test input() functionality when wired to a tty
        self.check_input_tty("prompt", b"quux")

    def test_input_tty_non_ascii(self):
        # Check stdin/stdout encoding is used when invoking PyOS_Readline()
        self.check_input_tty("prompté", b"quux\xc3\xa9", "utf-8")

    def test_input_tty_non_ascii_unicode_errors(self):
        # Check stdin/stdout error handler is used when invoking PyOS_Readline()
        self.check_input_tty("prompté", b"quux\xe9", "ascii")

    def test_input_tty_null_in_prompt(self):
        self.check_input_tty("prompt\0", b"",
                expected='ValueError: input: prompt string cannot contain '
                         'null characters')

    def test_input_tty_nonencodable_prompt(self):
        self.check_input_tty("prompté", b"quux", "ascii", stdout_errors='strict',
                expected="UnicodeEncodeError: 'ascii' codec can't encode "
                         "character '\\xe9' in position 6: ordinal not in "
                         "range(128)")

    def test_input_tty_nondecodable_input(self):
        self.check_input_tty("prompt", b"quux\xe9", "ascii", stdin_errors='strict',
                expected="UnicodeDecodeError: 'ascii' codec can't decode "
                         "byte 0xe9 in position 4: ordinal not in "
                         "range(128)")

    def test_input_no_stdout_fileno(self):
        # Issue #24402: If stdin is the original terminal but stdout.fileno()
        # fails, do not use the original stdout file descriptor
        def child(wpipe):
            print("stdin.isatty():", sys.stdin.isatty(), file=wpipe)
            sys.stdout = io.StringIO()  # Does not support fileno()
            input("prompt")
            print("captured:", ascii(sys.stdout.getvalue()), file=wpipe)
        lines = self.run_child(child, b"quux\r")
        expected = (
            "stdin.isatty(): True",
            "captured: 'prompt'",
        )
        self.assertSequenceEqual(lines, expected)

class TestSorted(CPythonTestCase):

    def test_basic(self):
        data = list(range(100))
        copy = data[:]
        random.shuffle(copy)
        self.assertEqual(data, sorted(copy))
        self.assertNotEqual(data, copy)

        data.reverse()
        random.shuffle(copy)
        self.assertEqual(data, sorted(copy, key=lambda x: -x))
        self.assertNotEqual(data, copy)
        random.shuffle(copy)
        self.assertEqual(data, sorted(copy, reverse=True))
        self.assertNotEqual(data, copy)

    def test_bad_arguments(self):
        # Issue #29327: The first argument is positional-only.
        sorted([])
        with self.assertRaises(TypeError):
            sorted(iterable=[])
        # Other arguments are keyword-only
        sorted([], key=None)
        with self.assertRaises(TypeError):
            sorted([], None)

    def test_inputtypes(self):
        s = 'abracadabra'
        types = [list, tuple, str]
        for T in types:
            self.assertEqual(sorted(s), sorted(T(s)))

        s = ''.join(set(s))  # unique letters only
        types = [str, set, frozenset, list, tuple, dict.fromkeys]
        for T in types:
            self.assertEqual(sorted(s), sorted(T(s)))

    def test_baddecorator(self):
        data = 'The quick Brown fox Jumped over The lazy Dog'.split()
        self.assertRaises(TypeError, sorted, data, None, lambda x,y: 0)


class ShutdownTest(CPythonTestCase):

    def test_cleanup(self):
        # Issue #19255: builtins are still available at shutdown
        code = """if 1:
            import builtins
            import sys

            class C:
                def __del__(self):
                    print("before")
                    # Check that builtins still exist
                    len(())
                    print("after")

            c = C()
            # Make this module survive until builtins and sys are cleaned
            builtins.here = sys.modules[__name__]
            sys.here = sys.modules[__name__]
            # Create a reference loop so that this module needs to go
            # through a GC phase.
            here = sys.modules[__name__]
            """
        # Issue #20599: Force ASCII encoding to get a codec implemented in C,
        # otherwise the codec may be unloaded before C.__del__() is called, and
        # so print("before") fails because the codec cannot be used to encode
        # "before" to sys.stdout.encoding. For example, on Windows,
        # sys.stdout.encoding is the OEM code page and these code pages are
        # implemented in Python
        rc, out, err = assert_python_ok("-c", code,
                                        PYTHONIOENCODING="ascii")
        self.assertEqual(["before", "after"], out.decode().splitlines())


@cpython_only
class ImmortalTests(CPythonTestCase):

    if sys.maxsize < (1 << 32):
        IMMORTAL_REFCOUNT = (1 << 30) - 1
    else:
        IMMORTAL_REFCOUNT = (1 << 32) - 1

    IMMORTALS = (None, True, False, Ellipsis, NotImplemented, *range(-5, 257))

    def assert_immortal(self, immortal):
        with self.subTest(immortal):
            self.assertEqual(sys.getrefcount(immortal), self.IMMORTAL_REFCOUNT)

    def test_immortals(self):
        for immortal in self.IMMORTALS:
            self.assert_immortal(immortal)

    def test_list_repeat_respect_immortality(self):
        refs = list(self.IMMORTALS) * 42
        for immortal in self.IMMORTALS:
            self.assert_immortal(immortal)

    def test_tuple_repeat_respect_immortality(self):
        refs = tuple(self.IMMORTALS) * 42
        for immortal in self.IMMORTALS:
            self.assert_immortal(immortal)


class TestType(CPythonTestCase):
    def test_new_type(self):
        A = type('A', (), {})
        self.assertEqual(A.__name__, 'A')
        self.assertEqual(A.__qualname__, 'A')
        self.assertEqual(A.__module__, __name__)
        self.assertEqual(A.__bases__, (object,))
        self.assertIs(A.__base__, object)
        self.assertNotIn('__firstlineno__', A.__dict__)
        x = A()
        self.assertIs(type(x), A)
        self.assertIs(x.__class__, A)

        with torch._dynamo.error_on_graph_break(False):
            class B:
                def ham(self):
                    return 'ham%d' % self
        C = type('C', (B, int), {'spam': lambda self: 'spam%s' % self})
        self.assertEqual(C.__name__, 'C')
        self.assertEqual(C.__qualname__, 'C')
        self.assertEqual(C.__module__, __name__)
        self.assertEqual(C.__bases__, (B, int))
        self.assertIs(C.__base__, int)
        self.assertIn('spam', C.__dict__)
        self.assertNotIn('ham', C.__dict__)
        x = C(42)
        self.assertEqual(x, 42)
        self.assertIs(type(x), C)
        self.assertIs(x.__class__, C)
        self.assertEqual(x.ham(), 'ham42')
        self.assertEqual(x.spam(), 'spam42')
        self.assertEqual(x.to_bytes(2, 'little'), b'\x2a\x00')

    def test_type_nokwargs(self):
        with self.assertRaises(TypeError):
            type('a', (), {}, x=5)
        with self.assertRaises(TypeError):
            type('a', (), dict={})

    def test_type_name(self):
        for name in 'A', '\xc4', '\U0001f40d', 'B.A', '42', '':
            with self.subTest(name=name):
                A = type(name, (), {})
                self.assertEqual(A.__name__, name)
                self.assertEqual(A.__qualname__, name)
                self.assertEqual(A.__module__, __name__)
        with self.assertRaises(ValueError):
            type('A\x00B', (), {})
        with self.assertRaises(UnicodeEncodeError):
            type('A\udcdcB', (), {})
        with self.assertRaises(TypeError):
            type(b'A', (), {})

        C = type('C', (), {})
        for name in 'A', '\xc4', '\U0001f40d', 'B.A', '42', '':
            with self.subTest(name=name):
                C.__name__ = name
                self.assertEqual(C.__name__, name)
                self.assertEqual(C.__qualname__, 'C')
                self.assertEqual(C.__module__, __name__)

        A = type('C', (), {})
        with self.assertRaises(ValueError):
            A.__name__ = 'A\x00B'
        self.assertEqual(A.__name__, 'C')
        with self.assertRaises(UnicodeEncodeError):
            A.__name__ = 'A\udcdcB'
        self.assertEqual(A.__name__, 'C')
        with self.assertRaises(TypeError):
            A.__name__ = b'A'
        self.assertEqual(A.__name__, 'C')

    def test_type_qualname(self):
        A = type('A', (), {'__qualname__': 'B.C'})
        self.assertEqual(A.__name__, 'A')
        self.assertEqual(A.__qualname__, 'B.C')
        self.assertEqual(A.__module__, __name__)
        with self.assertRaises(TypeError):
            type('A', (), {'__qualname__': b'B'})
        self.assertEqual(A.__qualname__, 'B.C')

        A.__qualname__ = 'D.E'
        self.assertEqual(A.__name__, 'A')
        self.assertEqual(A.__qualname__, 'D.E')
        with self.assertRaises(TypeError):
            A.__qualname__ = b'B'
        self.assertEqual(A.__qualname__, 'D.E')

    def test_type_firstlineno(self):
        A = type('A', (), {'__firstlineno__': 42})
        self.assertEqual(A.__name__, 'A')
        self.assertEqual(A.__module__, __name__)
        self.assertEqual(A.__dict__['__firstlineno__'], 42)
        A.__module__ = 'testmodule'
        self.assertEqual(A.__module__, 'testmodule')
        self.assertNotIn('__firstlineno__', A.__dict__)
        A.__firstlineno__ = 43
        self.assertEqual(A.__dict__['__firstlineno__'], 43)

    def test_type_typeparams(self):
        with torch._dynamo.error_on_graph_break(False):
            class A[T]:
                pass
        T, = A.__type_params__
        self.assertIsInstance(T, typing.TypeVar)
        A.__type_params__ = "whatever"
        self.assertEqual(A.__type_params__, "whatever")
        with self.assertRaises(TypeError):
            del A.__type_params__
        self.assertEqual(A.__type_params__, "whatever")

    def test_type_doc(self):
        for doc in 'x', '\xc4', '\U0001f40d', 'x\x00y', b'x', 42, None:
            A = type('A', (), {'__doc__': doc})
            self.assertEqual(A.__doc__, doc)
        with self.assertRaises(UnicodeEncodeError):
            type('A', (), {'__doc__': 'x\udcdcy'})

        A = type('A', (), {})
        self.assertEqual(A.__doc__, None)
        for doc in 'x', '\xc4', '\U0001f40d', 'x\x00y', 'x\udcdcy', b'x', 42, None:
            A.__doc__ = doc
            self.assertEqual(A.__doc__, doc)

    def test_bad_args(self):
        with self.assertRaises(TypeError):
            type()
        with self.assertRaises(TypeError):
            type('A', ())
        with self.assertRaises(TypeError):
            type('A', (), {}, ())
        with self.assertRaises(TypeError):
            type('A', (), dict={})
        with self.assertRaises(TypeError):
            type('A', [], {})
        with self.assertRaises(TypeError):
            type('A', (), types.MappingProxyType({}))
        with self.assertRaises(TypeError):
            type('A', (None,), {})
        with self.assertRaises(TypeError):
            type('A', (bool,), {})
        with self.assertRaises(TypeError):
            type('A', (int, str), {})

    def test_bad_slots(self):
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': b'x'})
        with self.assertRaises(TypeError):
            type('A', (int,), {'__slots__': 'x'})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': ''})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': '42'})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': 'x\x00y'})
        with self.assertRaises(ValueError):
            type('A', (), {'__slots__': 'x', 'x': 0})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': ('__dict__', '__dict__')})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': ('__weakref__', '__weakref__')})

        with torch._dynamo.error_on_graph_break(False):
            class B:
                pass
        with self.assertRaises(TypeError):
            type('A', (B,), {'__slots__': '__dict__'})
        with self.assertRaises(TypeError):
            type('A', (B,), {'__slots__': '__weakref__'})

    def test_namespace_order(self):
        # bpo-34320: namespace should preserve order
        od = collections.OrderedDict([('a', 1), ('b', 2)])
        od.move_to_end('a')
        expected = list(od.items())

        C = type('C', (), od)
        self.assertEqual(list(C.__dict__.items())[:2], [('b', 2), ('a', 1)])


if __name__ == "__main__":
    run_tests()
