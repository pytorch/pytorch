# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase

# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

import copy
import gc
import pickle
import sys
import doctest
import unittest
import weakref
import inspect
import types

from test import support

try:
    import _testcapi
except ImportError:
    _testcapi = None


# This tests to make sure that if a SIGINT arrives just before we send into a
# yield from chain, the KeyboardInterrupt is raised in the innermost
# generator (see bpo-30039).
@unittest.skipUnless(_testcapi is not None and
                     hasattr(_testcapi, "raise_SIGINT_then_send_None"),
                     "needs _testcapi.raise_SIGINT_then_send_None")
class SignalAndYieldFromTest(__TestCase):

    def generator1(self):
        return (yield from self.generator2())

    def generator2(self):
        try:
            yield
        except KeyboardInterrupt:
            return "PASSED"
        else:
            return "FAILED"

    def test_raise_and_yield_from(self):
        gen = self.generator1()
        gen.send(None)
        try:
            _testcapi.raise_SIGINT_then_send_None(gen)
        except BaseException as _exc:
            exc = _exc
        self.assertIs(type(exc), StopIteration)
        self.assertEqual(exc.value, "PASSED")


class FinalizationTest(__TestCase):

    def test_frame_resurrect(self):
        # A generator frame can be resurrected by a generator's finalization.
        def gen():
            nonlocal frame
            try:
                yield
            finally:
                frame = sys._getframe()

        g = gen()
        wr = weakref.ref(g)
        next(g)
        del g
        support.gc_collect()
        self.assertIs(wr(), None)
        self.assertTrue(frame)
        del frame
        support.gc_collect()

    def test_refcycle(self):
        # A generator caught in a refcycle gets finalized anyway.
        old_garbage = gc.garbage[:]
        finalized = False
        def gen():
            nonlocal finalized
            try:
                g = yield
                yield 1
            finally:
                finalized = True

        g = gen()
        next(g)
        g.send(g)
        self.assertGreater(sys.getrefcount(g), 2)
        self.assertFalse(finalized)
        del g
        support.gc_collect()
        self.assertTrue(finalized)
        self.assertEqual(gc.garbage, old_garbage)

    def test_lambda_generator(self):
        # bpo-23192, gh-119897: Test that a lambda returning a generator behaves
        # like the equivalent function
        f = lambda: (yield 1)
        self.assertIsInstance(f(), types.GeneratorType)
        self.assertEqual(next(f()), 1)

        def g(): return (yield 1)

        # test 'yield from'
        f2 = lambda: (yield from g())
        def g2(): return (yield from g())

        f3 = lambda: (yield from f())
        def g3(): return (yield from f())

        for gen_fun in (f, g, f2, g2, f3, g3):
            gen = gen_fun()
            self.assertEqual(next(gen), 1)
            with self.assertRaises(StopIteration) as cm:
                gen.send(2)
            self.assertEqual(cm.exception.value, 2)


class GeneratorTest(__TestCase):

    def test_name(self):
        def func():
            yield 1

        # check generator names
        gen = func()
        self.assertEqual(gen.__name__, "func")
        self.assertEqual(gen.__qualname__,
                         "GeneratorTest.test_name.<locals>.func")

        # modify generator names
        gen.__name__ = "name"
        gen.__qualname__ = "qualname"
        self.assertEqual(gen.__name__, "name")
        self.assertEqual(gen.__qualname__, "qualname")

        # generator names must be a string and cannot be deleted
        self.assertRaises(TypeError, setattr, gen, '__name__', 123)
        self.assertRaises(TypeError, setattr, gen, '__qualname__', 123)
        self.assertRaises(TypeError, delattr, gen, '__name__')
        self.assertRaises(TypeError, delattr, gen, '__qualname__')

        # modify names of the function creating the generator
        func.__qualname__ = "func_qualname"
        func.__name__ = "func_name"
        gen = func()
        self.assertEqual(gen.__name__, "func_name")
        self.assertEqual(gen.__qualname__, "func_qualname")

        # unnamed generator
        gen = (x for x in range(10))
        self.assertEqual(gen.__name__,
                         "<genexpr>")
        self.assertEqual(gen.__qualname__,
                         "GeneratorTest.test_name.<locals>.<genexpr>")

    def test_copy(self):
        def f():
            yield 1
        g = f()
        with self.assertRaises(TypeError):
            copy.copy(g)

    def test_pickle(self):
        def f():
            yield 1
        g = f()
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.assertRaises((TypeError, pickle.PicklingError)):
                pickle.dumps(g, proto)

    def test_send_non_none_to_new_gen(self):
        def f():
            yield 1
        g = f()
        with self.assertRaises(TypeError):
            g.send(0)
        self.assertEqual(next(g), 1)

    def test_handle_frame_object_in_creation(self):

        #Attempt to expose partially constructed frames
        #See https://github.com/python/cpython/issues/94262

        def cb(*args):
            inspect.stack()

        def gen():
            yield 1

        thresholds = gc.get_threshold()

        gc.callbacks.append(cb)
        gc.set_threshold(1, 0, 0)
        try:
            gen()
        finally:
            gc.set_threshold(*thresholds)
            gc.callbacks.pop()

        class Sneaky:
            def __del__(self):
                inspect.stack()

        sneaky = Sneaky()
        sneaky._s = Sneaky()
        sneaky._s._s = sneaky

        gc.set_threshold(1, 0, 0)
        try:
            del sneaky
            gen()
        finally:
            gc.set_threshold(*thresholds)

    def test_ag_frame_f_back(self):
        async def f():
            yield
        ag = f()
        self.assertIsNone(ag.ag_frame.f_back)

    def test_cr_frame_f_back(self):
        async def f():
            pass
        cr = f()
        self.assertIsNone(cr.cr_frame.f_back)
        cr.close()  # Suppress RuntimeWarning.

    def test_gi_frame_f_back(self):
        def f():
            yield
        gi = f()
        self.assertIsNone(gi.gi_frame.f_back)

    def test_issue103488(self):

        def gen_raises():
            yield
            raise ValueError()

        def loop():
            try:
                for _ in gen_raises():
                    if True is False:
                        return
            except ValueError:
                pass

        #This should not raise
        loop()

    @unittest.expectedFailure
    def test_genexpr_only_calls_dunder_iter_once(self):

        class Iterator:

            def __init__(self):
                self.val = 0

            def __next__(self):
                if self.val == 2:
                    raise StopIteration
                self.val += 1
                return self.val

            # No __iter__ method

        class C:

            def __iter__(self):
                return Iterator()

        self.assertEqual([1,2], list(i for i in C()))


class ModifyUnderlyingIterableTest(__TestCase):
    iterables = [
        range(0),
        range(20),
        [1, 2, 3],
        (2,),
        {13, 48, 211},
        frozenset((15, 8, 6)),
        {1: 2, 3: 4},
    ]

    non_iterables = [
        None,
        42,
        3.0,
        2j,
    ]

    def genexpr(self):
        return (x for x in range(10))

    def genfunc(self):
        def gen(it):
            for x in it:
                yield x
        return gen(range(10))

    def process_tests(self, get_generator):
        for obj in self.iterables:
            g_obj = get_generator(obj)
            with self.subTest(g_obj=g_obj, obj=obj):
                self.assertListEqual(list(g_obj), list(obj))

            g_iter = get_generator(iter(obj))
            with self.subTest(g_iter=g_iter, obj=obj):
                self.assertListEqual(list(g_iter), list(obj))

        err_regex = "'.*' object is not iterable"
        for obj in self.non_iterables:
            g_obj = get_generator(obj)
            with self.subTest(g_obj=g_obj):
                self.assertRaisesRegex(TypeError, err_regex, list, g_obj)

    def test_modify_f_locals(self):
        def modify_f_locals(g, local, obj):
            g.gi_frame.f_locals[local] = obj
            return g

        def get_generator_genexpr(obj):
            return modify_f_locals(self.genexpr(), '.0', obj)

        def get_generator_genfunc(obj):
            return modify_f_locals(self.genfunc(), 'it', obj)

        self.process_tests(get_generator_genexpr)
        self.process_tests(get_generator_genfunc)

    def test_new_gen_from_gi_code(self):
        def new_gen_from_gi_code(g, obj):
            generator_func = types.FunctionType(g.gi_code, {})
            return generator_func(obj)

        def get_generator_genexpr(obj):
            return new_gen_from_gi_code(self.genexpr(), obj)

        def get_generator_genfunc(obj):
            return new_gen_from_gi_code(self.genfunc(), obj)

        self.process_tests(get_generator_genexpr)
        self.process_tests(get_generator_genfunc)


class ExceptionTest(__TestCase):
    # Tests for the issue #23353: check that the currently handled exception
    # is correctly saved/restored in PyEval_EvalFrameEx().

    def test_except_throw(self):
        def store_raise_exc_generator():
            try:
                self.assertIsNone(sys.exception())
                yield
            except Exception as exc:
                # exception raised by gen.throw(exc)
                self.assertIsInstance(sys.exception(), ValueError)
                self.assertIsNone(exc.__context__)
                yield

                # ensure that the exception is not lost
                self.assertIsInstance(sys.exception(), ValueError)
                yield

                # we should be able to raise back the ValueError
                raise

        make = store_raise_exc_generator()
        next(make)

        try:
            raise ValueError()
        except Exception as exc:
            try:
                make.throw(exc)
            except Exception:
                pass

        next(make)
        with self.assertRaises(ValueError) as cm:
            next(make)
        self.assertIsNone(cm.exception.__context__)

        self.assertIsNone(sys.exception())

    def test_except_next(self):
        def gen():
            self.assertIsInstance(sys.exception(), ValueError)
            yield "done"

        g = gen()
        try:
            raise ValueError
        except Exception:
            self.assertEqual(next(g), "done")
        self.assertIsNone(sys.exception())

    def test_except_gen_except(self):
        def gen():
            try:
                self.assertIsNone(sys.exception())
                yield
                # we are called from "except ValueError:", TypeError must
                # inherit ValueError in its context
                raise TypeError()
            except TypeError as exc:
                self.assertIsInstance(sys.exception(), TypeError)
                self.assertEqual(type(exc.__context__), ValueError)
            # here we are still called from the "except ValueError:"
            self.assertIsInstance(sys.exception(), ValueError)
            yield
            self.assertIsNone(sys.exception())
            yield "done"

        g = gen()
        next(g)
        try:
            raise ValueError
        except Exception:
            next(g)

        self.assertEqual(next(g), "done")
        self.assertIsNone(sys.exception())

    def test_nested_gen_except_loop(self):
        def gen():
            for i in range(100):
                self.assertIsInstance(sys.exception(), TypeError)
                yield "doing"

        def outer():
            try:
                raise TypeError
            except:
                for x in gen():
                    yield x

        try:
            raise ValueError
        except Exception:
            for x in outer():
                self.assertEqual(x, "doing")
        self.assertEqual(sys.exception(), None)

    def test_except_throw_exception_context(self):
        def gen():
            try:
                try:
                    self.assertIsNone(sys.exception())
                    yield
                except ValueError:
                    # we are called from "except ValueError:"
                    self.assertIsInstance(sys.exception(), ValueError)
                    raise TypeError()
            except Exception as exc:
                self.assertIsInstance(sys.exception(), TypeError)
                self.assertEqual(type(exc.__context__), ValueError)
            # we are still called from "except ValueError:"
            self.assertIsInstance(sys.exception(), ValueError)
            yield
            self.assertIsNone(sys.exception())
            yield "done"

        g = gen()
        next(g)
        try:
            raise ValueError
        except Exception as exc:
            g.throw(exc)

        self.assertEqual(next(g), "done")
        self.assertIsNone(sys.exception())

    def test_except_throw_bad_exception(self):
        class E(Exception):
            def __new__(cls, *args, **kwargs):
                return cls

        def boring_generator():
            yield

        gen = boring_generator()

        err_msg = 'should have returned an instance of BaseException'

        with self.assertRaisesRegex(TypeError, err_msg):
            gen.throw(E)

        self.assertRaises(StopIteration, next, gen)

        def generator():
            with self.assertRaisesRegex(TypeError, err_msg):
                yield

        gen = generator()
        next(gen)
        with self.assertRaises(StopIteration):
            gen.throw(E)

    def test_gen_3_arg_deprecation_warning(self):
        def g():
            yield 42

        gen = g()
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(TypeError):
                gen.throw(TypeError, TypeError(24), None)

    def test_stopiteration_error(self):
        # See also PEP 479.

        def gen():
            raise StopIteration
            yield

        with self.assertRaisesRegex(RuntimeError, 'raised StopIteration'):
            next(gen())

    def test_tutorial_stopiteration(self):
        # Raise StopIteration" stops the generator too:

        def f():
            yield 1
            raise StopIteration
            yield 2 # never reached

        g = f()
        self.assertEqual(next(g), 1)

        with self.assertRaisesRegex(RuntimeError, 'raised StopIteration'):
            next(g)

    def test_return_tuple(self):
        def g():
            return (yield 1)

        gen = g()
        self.assertEqual(next(gen), 1)
        with self.assertRaises(StopIteration) as cm:
            gen.send((2,))
        self.assertEqual(cm.exception.value, (2,))

    def test_return_stopiteration(self):
        def g():
            return (yield 1)

        gen = g()
        self.assertEqual(next(gen), 1)
        with self.assertRaises(StopIteration) as cm:
            gen.send(StopIteration(2))
        self.assertIsInstance(cm.exception.value, StopIteration)
        self.assertEqual(cm.exception.value.value, 2)


class GeneratorCloseTest(__TestCase):

    def test_close_no_return_value(self):
        def f():
            yield

        gen = f()
        gen.send(None)
        self.assertIsNone(gen.close())

    def test_close_return_value(self):
        def f():
            try:
                yield
                # close() raises GeneratorExit here, which is caught
            except GeneratorExit:
                return 0

        gen = f()
        gen.send(None)
        self.assertEqual(gen.close(), 0)

    def test_close_not_catching_exit(self):
        def f():
            yield
            # close() raises GeneratorExit here, which isn't caught and
            # therefore propagates -- no return value
            return 0

        gen = f()
        gen.send(None)
        self.assertIsNone(gen.close())

    def test_close_not_started(self):
        def f():
            try:
                yield
            except GeneratorExit:
                return 0

        gen = f()
        self.assertIsNone(gen.close())

    def test_close_exhausted(self):
        def f():
            try:
                yield
            except GeneratorExit:
                return 0

        gen = f()
        next(gen)
        with self.assertRaises(StopIteration):
            next(gen)
        self.assertIsNone(gen.close())

    def test_close_closed(self):
        def f():
            try:
                yield
            except GeneratorExit:
                return 0

        gen = f()
        gen.send(None)
        self.assertEqual(gen.close(), 0)
        self.assertIsNone(gen.close())

    def test_close_raises(self):
        def f():
            try:
                yield
            except GeneratorExit:
                pass
            raise RuntimeError

        gen = f()
        gen.send(None)
        with self.assertRaises(RuntimeError):
            gen.close()

    def test_close_releases_frame_locals(self):
        # See gh-118272

        class Foo:
            pass

        f = Foo()
        f_wr = weakref.ref(f)

        def genfn():
            a = f
            yield

        g = genfn()
        next(g)
        del f
        g.close()
        support.gc_collect()
        self.assertIsNone(f_wr())


class GeneratorThrowTest(__TestCase):

    def test_exception_context_with_yield(self):
        def f():
            try:
                raise KeyError('a')
            except Exception:
                yield

        gen = f()
        gen.send(None)
        with self.assertRaises(ValueError) as cm:
            gen.throw(ValueError)
        context = cm.exception.__context__
        self.assertEqual((type(context), context.args), (KeyError, ('a',)))

    def test_exception_context_with_yield_inside_generator(self):
        # Check that the context is also available from inside the generator
        # with yield, as opposed to outside.
        def f():
            try:
                raise KeyError('a')
            except Exception:
                try:
                    yield
                except Exception as exc:
                    self.assertEqual(type(exc), ValueError)
                    context = exc.__context__
                    self.assertEqual((type(context), context.args),
                        (KeyError, ('a',)))
                    yield 'b'

        gen = f()
        gen.send(None)
        actual = gen.throw(ValueError)
        # This ensures that the assertions inside were executed.
        self.assertEqual(actual, 'b')

    def test_exception_context_with_yield_from(self):
        def f():
            yield

        def g():
            try:
                raise KeyError('a')
            except Exception:
                yield from f()

        gen = g()
        gen.send(None)
        with self.assertRaises(ValueError) as cm:
            gen.throw(ValueError)
        context = cm.exception.__context__
        self.assertEqual((type(context), context.args), (KeyError, ('a',)))

    def test_exception_context_with_yield_from_with_context_cycle(self):
        # Check trying to create an exception context cycle:
        # https://bugs.python.org/issue40696
        has_cycle = None

        def f():
            yield

        def g(exc):
            nonlocal has_cycle
            try:
                raise exc
            except Exception:
                try:
                    yield from f()
                except Exception as exc:
                    has_cycle = (exc is exc.__context__)
            yield

        exc = KeyError('a')
        gen = g(exc)
        gen.send(None)
        gen.throw(exc)
        # This also distinguishes from the initial has_cycle=None.
        self.assertEqual(has_cycle, False)

    def test_throw_after_none_exc_type(self):
        def g():
            try:
                raise KeyError
            except KeyError:
                pass

            try:
                yield
            except Exception:
                raise RuntimeError

        gen = g()
        gen.send(None)
        with self.assertRaises(RuntimeError) as cm:
            gen.throw(ValueError)


class GeneratorStackTraceTest(__TestCase):

    def check_stack_names(self, frame, expected):
        names = []
        while frame:
            name = frame.f_code.co_name
            # Stop checking frames when we get to our test helper.
            if name.startswith('check_') or name.startswith('call_'):
                break

            names.append(name)
            frame = frame.f_back

        self.assertEqual(names, expected)

    def check_yield_from_example(self, call_method):
        def f():
            self.check_stack_names(sys._getframe(), ['f', 'g'])
            try:
                yield
            except Exception:
                pass
            self.check_stack_names(sys._getframe(), ['f', 'g'])

        def g():
            self.check_stack_names(sys._getframe(), ['g'])
            yield from f()
            self.check_stack_names(sys._getframe(), ['g'])

        gen = g()
        gen.send(None)
        try:
            call_method(gen)
        except StopIteration:
            pass

    def test_send_with_yield_from(self):
        def call_send(gen):
            gen.send(None)

        self.check_yield_from_example(call_send)

    def test_throw_with_yield_from(self):
        def call_throw(gen):
            gen.throw(RuntimeError)

        self.check_yield_from_example(call_throw)


class YieldFromTests(__TestCase):
    def test_generator_gi_yieldfrom(self):
        def a():
            self.assertEqual(inspect.getgeneratorstate(gen_b), inspect.GEN_RUNNING)
            self.assertIsNone(gen_b.gi_yieldfrom)
            yield
            self.assertEqual(inspect.getgeneratorstate(gen_b), inspect.GEN_RUNNING)
            self.assertIsNone(gen_b.gi_yieldfrom)

        def b():
            self.assertIsNone(gen_b.gi_yieldfrom)
            yield from a()
            self.assertIsNone(gen_b.gi_yieldfrom)
            yield
            self.assertIsNone(gen_b.gi_yieldfrom)

        gen_b = b()
        self.assertEqual(inspect.getgeneratorstate(gen_b), inspect.GEN_CREATED)
        self.assertIsNone(gen_b.gi_yieldfrom)

        gen_b.send(None)
        self.assertEqual(inspect.getgeneratorstate(gen_b), inspect.GEN_SUSPENDED)
        self.assertEqual(gen_b.gi_yieldfrom.gi_code.co_name, 'a')

        gen_b.send(None)
        self.assertEqual(inspect.getgeneratorstate(gen_b), inspect.GEN_SUSPENDED)
        self.assertIsNone(gen_b.gi_yieldfrom)

        [] = gen_b  # Exhaust generator
        self.assertEqual(inspect.getgeneratorstate(gen_b), inspect.GEN_CLOSED)
        self.assertIsNone(gen_b.gi_yieldfrom)


tutorial_tests = """
Let's try a simple generator:

    >>> def f():
    ...    yield 1
    ...    yield 2

    >>> for i in f():
    ...     print(i)
    1
    2
    >>> g = f()
    >>> next(g)
    1
    >>> next(g)
    2

"Falling off the end" stops the generator:

    >>> next(g)
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
      File "<stdin>", line 2, in g
    StopIteration

"return" also stops the generator:

    >>> def f():
    ...     yield 1
    ...     return
    ...     yield 2 # never reached
    ...
    >>> g = f()
    >>> next(g)
    1
    >>> next(g)
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
      File "<stdin>", line 3, in f
    StopIteration
    >>> next(g) # once stopped, can't be resumed
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    StopIteration

However, "return" and StopIteration are not exactly equivalent:

    >>> def g1():
    ...     try:
    ...         return
    ...     except:
    ...         yield 1
    ...
    >>> list(g1())
    []

    >>> def g2():
    ...     try:
    ...         raise StopIteration
    ...     except:
    ...         yield 42
    >>> print(list(g2()))
    [42]

This may be surprising at first:

    >>> def g3():
    ...     try:
    ...         return
    ...     finally:
    ...         yield 1
    ...
    >>> list(g3())
    [1]

Let's create an alternate range() function implemented as a generator:

    >>> def yrange(n):
    ...     for i in range(n):
    ...         yield i
    ...
    >>> list(yrange(5))
    [0, 1, 2, 3, 4]

Generators always return to the most recent caller:

    >>> def creator():
    ...     r = yrange(5)
    ...     print("creator", next(r))
    ...     return r
    ...
    >>> def caller():
    ...     r = creator()
    ...     for i in r:
    ...             print("caller", i)
    ...
    >>> caller()
    creator 0
    caller 1
    caller 2
    caller 3
    caller 4

Generators can call other generators:

    >>> def zrange(n):
    ...     for i in yrange(n):
    ...         yield i
    ...
    >>> list(zrange(5))
    [0, 1, 2, 3, 4]

"""

# The examples from PEP 255.

pep_tests = """

Specification:  Yield

    Restriction:  A generator cannot be resumed while it is actively
    running:

    >>> def g():
    ...     i = next(me)
    ...     yield i
    >>> me = g()
    >>> next(me)
    Traceback (most recent call last):
     ...
      File "<string>", line 2, in g
    ValueError: generator already executing

Specification: Return

    Note that return isn't always equivalent to raising StopIteration:  the
    difference lies in how enclosing try/except constructs are treated.
    For example,

        >>> def f1():
        ...     try:
        ...         return
        ...     except:
        ...        yield 1
        >>> print(list(f1()))
        []

    because, as in any function, return simply exits, but

        >>> def f2():
        ...     try:
        ...         raise StopIteration
        ...     except:
        ...         yield 42
        >>> print(list(f2()))
        [42]

    because StopIteration is captured by a bare "except", as is any
    exception.

Specification: Generators and Exception Propagation

    >>> def f():
    ...     return 1//0
    >>> def g():
    ...     yield f()  # the zero division exception propagates
    ...     yield 42   # and we'll never get here
    >>> k = g()
    >>> next(k)
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
      File "<stdin>", line 2, in g
      File "<stdin>", line 2, in f
    ZeroDivisionError: integer division or modulo by zero
    >>> next(k)  # and the generator cannot be resumed
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    StopIteration
    >>>

Specification: Try/Except/Finally

    >>> def f():
    ...     try:
    ...         yield 1
    ...         try:
    ...             yield 2
    ...             1//0
    ...             yield 3  # never get here
    ...         except ZeroDivisionError:
    ...             yield 4
    ...             yield 5
    ...             raise
    ...         except:
    ...             yield 6
    ...         yield 7     # the "raise" above stops this
    ...     except:
    ...         yield 8
    ...     yield 9
    ...     try:
    ...         x = 12
    ...     finally:
    ...         yield 10
    ...     yield 11
    >>> print(list(f()))
    [1, 2, 4, 5, 8, 9, 10, 11]
    >>>

Guido's binary tree example.

    >>> # A binary tree class.
    >>> class Tree:
    ...
    ...     def __init__(self, label, left=None, right=None):
    ...         self.label = label
    ...         self.left = left
    ...         self.right = right
    ...
    ...     def __repr__(self, level=0, indent="    "):
    ...         s = level*indent + repr(self.label)
    ...         if self.left:
    ...             s = s + "\\n" + self.left.__repr__(level+1, indent)
    ...         if self.right:
    ...             s = s + "\\n" + self.right.__repr__(level+1, indent)
    ...         return s
    ...
    ...     def __iter__(self):
    ...         return inorder(self)

    >>> # Create a Tree from a list.
    >>> def tree(list):
    ...     n = len(list)
    ...     if n == 0:
    ...         return []
    ...     i = n // 2
    ...     return Tree(list[i], tree(list[:i]), tree(list[i+1:]))

    >>> # Show it off: create a tree.
    >>> t = tree("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    >>> # A recursive generator that generates Tree labels in in-order.
    >>> def inorder(t):
    ...     if t:
    ...         for x in inorder(t.left):
    ...             yield x
    ...         yield t.label
    ...         for x in inorder(t.right):
    ...             yield x

    >>> # Show it off: create a tree.
    >>> t = tree("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    >>> # Print the nodes of the tree in in-order.
    >>> for x in t:
    ...     print(' '+x, end='')
     A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

    >>> # A non-recursive generator.
    >>> def inorder(node):
    ...     stack = []
    ...     while node:
    ...         while node.left:
    ...             stack.append(node)
    ...             node = node.left
    ...         yield node.label
    ...         while not node.right:
    ...             try:
    ...                 node = stack.pop()
    ...             except IndexError:
    ...                 return
    ...             yield node.label
    ...         node = node.right

    >>> # Exercise the non-recursive generator.
    >>> for x in t:
    ...     print(' '+x, end='')
     A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

"""

# Examples from Iterator-List and Python-Dev and c.l.py.

email_tests = """

The difference between yielding None and returning it.

>>> def g():
...     for i in range(3):
...         yield None
...     yield None
...     return
>>> list(g())
[None, None, None, None]

Ensure that explicitly raising StopIteration acts like any other exception
in try/except, not like a return.

>>> def g():
...     yield 1
...     try:
...         raise StopIteration
...     except:
...         yield 2
...     yield 3
>>> list(g())
[1, 2, 3]

Next one was posted to c.l.py.

>>> def gcomb(x, k):
...     "Generate all combinations of k elements from list x."
...
...     if k > len(x):
...         return
...     if k == 0:
...         yield []
...     else:
...         first, rest = x[0], x[1:]
...         # A combination does or doesn't contain first.
...         # If it does, the remainder is a k-1 comb of rest.
...         for c in gcomb(rest, k-1):
...             c.insert(0, first)
...             yield c
...         # If it doesn't contain first, it's a k comb of rest.
...         for c in gcomb(rest, k):
...             yield c

>>> seq = list(range(1, 5))
>>> for k in range(len(seq) + 2):
...     print("%d-combs of %s:" % (k, seq))
...     for c in gcomb(seq, k):
...         print("   ", c)
0-combs of [1, 2, 3, 4]:
    []
1-combs of [1, 2, 3, 4]:
    [1]
    [2]
    [3]
    [4]
2-combs of [1, 2, 3, 4]:
    [1, 2]
    [1, 3]
    [1, 4]
    [2, 3]
    [2, 4]
    [3, 4]
3-combs of [1, 2, 3, 4]:
    [1, 2, 3]
    [1, 2, 4]
    [1, 3, 4]
    [2, 3, 4]
4-combs of [1, 2, 3, 4]:
    [1, 2, 3, 4]
5-combs of [1, 2, 3, 4]:

From the Iterators list, about the types of these things.

>>> def g():
...     yield 1
...
>>> type(g)
<class 'function'>
>>> i = g()
>>> type(i)
<class 'generator'>
>>> [s for s in dir(i) if not s.startswith('_')]
['close', 'gi_code', 'gi_frame', 'gi_running', 'gi_suspended', 'gi_yieldfrom', 'send', 'throw']
>>> from test.support import HAVE_DOCSTRINGS
>>> print(i.__next__.__doc__ if HAVE_DOCSTRINGS else 'Implement next(self).')
Implement next(self).
>>> iter(i) is i
True
>>> import types
>>> isinstance(i, types.GeneratorType)
True

And more, added later.

>>> i.gi_running
0
>>> type(i.gi_frame)
<class 'frame'>
>>> i.gi_running = 42
Traceback (most recent call last):
  ...
AttributeError: attribute 'gi_running' of 'generator' objects is not writable
>>> def g():
...     yield me.gi_running
>>> me = g()
>>> me.gi_running
0
>>> next(me)
1
>>> me.gi_running
0

A clever union-find implementation from c.l.py, due to David Eppstein.
Sent: Friday, June 29, 2001 12:16 PM
To: python-list@python.org
Subject: Re: PEP 255: Simple Generators

>>> class disjointSet:
...     def __init__(self, name):
...         self.name = name
...         self.parent = None
...         self.generator = self.generate()
...
...     def generate(self):
...         while not self.parent:
...             yield self
...         for x in self.parent.generator:
...             yield x
...
...     def find(self):
...         return next(self.generator)
...
...     def union(self, parent):
...         if self.parent:
...             raise ValueError("Sorry, I'm not a root!")
...         self.parent = parent
...
...     def __str__(self):
...         return self.name

>>> names = "ABCDEFGHIJKLM"
>>> sets = [disjointSet(name) for name in names]
>>> roots = sets[:]

>>> import random
>>> gen = random.Random(42)
>>> while 1:
...     for s in sets:
...         print(" %s->%s" % (s, s.find()), end='')
...     print()
...     if len(roots) > 1:
...         s1 = gen.choice(roots)
...         roots.remove(s1)
...         s2 = gen.choice(roots)
...         s1.union(s2)
...         print("merged", s1, "into", s2)
...     else:
...         break
 A->A B->B C->C D->D E->E F->F G->G H->H I->I J->J K->K L->L M->M
merged K into B
 A->A B->B C->C D->D E->E F->F G->G H->H I->I J->J K->B L->L M->M
merged A into F
 A->F B->B C->C D->D E->E F->F G->G H->H I->I J->J K->B L->L M->M
merged E into F
 A->F B->B C->C D->D E->F F->F G->G H->H I->I J->J K->B L->L M->M
merged D into C
 A->F B->B C->C D->C E->F F->F G->G H->H I->I J->J K->B L->L M->M
merged M into C
 A->F B->B C->C D->C E->F F->F G->G H->H I->I J->J K->B L->L M->C
merged J into B
 A->F B->B C->C D->C E->F F->F G->G H->H I->I J->B K->B L->L M->C
merged B into C
 A->F B->C C->C D->C E->F F->F G->G H->H I->I J->C K->C L->L M->C
merged F into G
 A->G B->C C->C D->C E->G F->G G->G H->H I->I J->C K->C L->L M->C
merged L into C
 A->G B->C C->C D->C E->G F->G G->G H->H I->I J->C K->C L->C M->C
merged G into I
 A->I B->C C->C D->C E->I F->I G->I H->H I->I J->C K->C L->C M->C
merged I into H
 A->H B->C C->C D->C E->H F->H G->H H->H I->H J->C K->C L->C M->C
merged C into H
 A->H B->H C->H D->H E->H F->H G->H H->H I->H J->H K->H L->H M->H

"""
# Emacs turd '

# Fun tests (for sufficiently warped notions of "fun").

fun_tests = """

Build up to a recursive Sieve of Eratosthenes generator.

>>> def firstn(g, n):
...     return [next(g) for i in range(n)]

>>> def intsfrom(i):
...     while 1:
...         yield i
...         i += 1

>>> firstn(intsfrom(5), 7)
[5, 6, 7, 8, 9, 10, 11]

>>> def exclude_multiples(n, ints):
...     for i in ints:
...         if i % n:
...             yield i

>>> firstn(exclude_multiples(3, intsfrom(1)), 6)
[1, 2, 4, 5, 7, 8]

>>> def sieve(ints):
...     prime = next(ints)
...     yield prime
...     not_divisible_by_prime = exclude_multiples(prime, ints)
...     for p in sieve(not_divisible_by_prime):
...         yield p

>>> primes = sieve(intsfrom(2))
>>> firstn(primes, 20)
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]


Another famous problem:  generate all integers of the form
    2**i * 3**j  * 5**k
in increasing order, where i,j,k >= 0.  Trickier than it may look at first!
Try writing it without generators, and correctly, and without generating
3 internal results for each result output.

>>> def times(n, g):
...     for i in g:
...         yield n * i
>>> firstn(times(10, intsfrom(1)), 10)
[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

>>> def merge(g, h):
...     ng = next(g)
...     nh = next(h)
...     while 1:
...         if ng < nh:
...             yield ng
...             ng = next(g)
...         elif ng > nh:
...             yield nh
...             nh = next(h)
...         else:
...             yield ng
...             ng = next(g)
...             nh = next(h)

The following works, but is doing a whale of a lot of redundant work --
it's not clear how to get the internal uses of m235 to share a single
generator.  Note that me_times2 (etc) each need to see every element in the
result sequence.  So this is an example where lazy lists are more natural
(you can look at the head of a lazy list any number of times).

>>> def m235():
...     yield 1
...     me_times2 = times(2, m235())
...     me_times3 = times(3, m235())
...     me_times5 = times(5, m235())
...     for i in merge(merge(me_times2,
...                          me_times3),
...                    me_times5):
...         yield i

Don't print "too many" of these -- the implementation above is extremely
inefficient:  each call of m235() leads to 3 recursive calls, and in
turn each of those 3 more, and so on, and so on, until we've descended
enough levels to satisfy the print stmts.  Very odd:  when I printed 5
lines of results below, this managed to screw up Win98's malloc in "the
usual" way, i.e. the heap grew over 4Mb so Win98 started fragmenting
address space, and it *looked* like a very slow leak.

>>> result = m235()
>>> for i in range(3):
...     print(firstn(result, 15))
[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
[25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80]
[81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192]

Heh.  Here's one way to get a shared list, complete with an excruciating
namespace renaming trick.  The *pretty* part is that the times() and merge()
functions can be reused as-is, because they only assume their stream
arguments are iterable -- a LazyList is the same as a generator to times().

>>> class LazyList:
...     def __init__(self, g):
...         self.sofar = []
...         self.fetch = g.__next__
...
...     def __getitem__(self, i):
...         sofar, fetch = self.sofar, self.fetch
...         while i >= len(sofar):
...             sofar.append(fetch())
...         return sofar[i]

>>> def m235():
...     yield 1
...     # Gack:  m235 below actually refers to a LazyList.
...     me_times2 = times(2, m235)
...     me_times3 = times(3, m235)
...     me_times5 = times(5, m235)
...     for i in merge(merge(me_times2,
...                          me_times3),
...                    me_times5):
...         yield i

Print as many of these as you like -- *this* implementation is memory-
efficient.

>>> m235 = LazyList(m235())
>>> for i in range(5):
...     print([m235[j] for j in range(15*i, 15*(i+1))])
[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
[25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80]
[81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192]
[200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384]
[400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675]

Ye olde Fibonacci generator, LazyList style.

>>> def fibgen(a, b):
...
...     def sum(g, h):
...         while 1:
...             yield next(g) + next(h)
...
...     def tail(g):
...         next(g)    # throw first away
...         for x in g:
...             yield x
...
...     yield a
...     yield b
...     for s in sum(iter(fib),
...                  tail(iter(fib))):
...         yield s

>>> fib = LazyList(fibgen(1, 2))
>>> firstn(iter(fib), 17)
[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]


Running after your tail with itertools.tee (new in version 2.4)

The algorithms "m235" (Hamming) and Fibonacci presented above are both
examples of a whole family of FP (functional programming) algorithms
where a function produces and returns a list while the production algorithm
suppose the list as already produced by recursively calling itself.
For these algorithms to work, they must:

- produce at least a first element without presupposing the existence of
  the rest of the list
- produce their elements in a lazy manner

To work efficiently, the beginning of the list must not be recomputed over
and over again. This is ensured in most FP languages as a built-in feature.
In python, we have to explicitly maintain a list of already computed results
and abandon genuine recursivity.

This is what had been attempted above with the LazyList class. One problem
with that class is that it keeps a list of all of the generated results and
therefore continually grows. This partially defeats the goal of the generator
concept, viz. produce the results only as needed instead of producing them
all and thereby wasting memory.

Thanks to itertools.tee, it is now clear "how to get the internal uses of
m235 to share a single generator".

>>> from itertools import tee
>>> def m235():
...     def _m235():
...         yield 1
...         for n in merge(times(2, m2),
...                        merge(times(3, m3),
...                              times(5, m5))):
...             yield n
...     m1 = _m235()
...     m2, m3, m5, mRes = tee(m1, 4)
...     return mRes

>>> it = m235()
>>> for i in range(5):
...     print(firstn(it, 15))
[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
[25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80]
[81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192]
[200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384]
[400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675]

The "tee" function does just what we want. It internally keeps a generated
result for as long as it has not been "consumed" from all of the duplicated
iterators, whereupon it is deleted. You can therefore print the hamming
sequence during hours without increasing memory usage, or very little.

The beauty of it is that recursive running-after-their-tail FP algorithms
are quite straightforwardly expressed with this Python idiom.

Ye olde Fibonacci generator, tee style.

>>> def fib():
...
...     def _isum(g, h):
...         while 1:
...             yield next(g) + next(h)
...
...     def _fib():
...         yield 1
...         yield 2
...         next(fibTail) # throw first away
...         for res in _isum(fibHead, fibTail):
...             yield res
...
...     realfib = _fib()
...     fibHead, fibTail, fibRes = tee(realfib, 3)
...     return fibRes

>>> firstn(fib(), 17)
[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]

"""

# syntax_tests mostly provokes SyntaxErrors.  Also fiddling with #if 0
# hackery.

syntax_tests = """

These are fine:

>>> def f():
...     yield 1
...     return

>>> def f():
...     try:
...         yield 1
...     finally:
...         pass

>>> def f():
...     try:
...         try:
...             1//0
...         except ZeroDivisionError:
...             yield 666
...         except:
...             pass
...     finally:
...         pass

>>> def f():
...     try:
...         try:
...             yield 12
...             1//0
...         except ZeroDivisionError:
...             yield 666
...         except:
...             try:
...                 x = 12
...             finally:
...                 yield 12
...     except:
...         return
>>> list(f())
[12, 666]

>>> def f():
...    yield
>>> type(f())
<class 'generator'>


>>> def f():
...    if 0:
...        yield
>>> type(f())
<class 'generator'>


>>> def f():
...     if 0:
...         yield 1
>>> type(f())
<class 'generator'>

>>> def f():
...    if "":
...        yield None
>>> type(f())
<class 'generator'>

>>> def f():
...     return
...     try:
...         if x==4:
...             pass
...         elif 0:
...             try:
...                 1//0
...             except SyntaxError:
...                 pass
...             else:
...                 if 0:
...                     while 12:
...                         x += 1
...                         yield 2 # don't blink
...                         f(a, b, c, d, e)
...         else:
...             pass
...     except:
...         x = 1
...     return
>>> type(f())
<class 'generator'>

>>> def f():
...     if 0:
...         def g():
...             yield 1
...
>>> type(f())
<class 'NoneType'>

>>> def f():
...     if 0:
...         class C:
...             def __init__(self):
...                 yield 1
...             def f(self):
...                 yield 2
>>> type(f())
<class 'NoneType'>

>>> def f():
...     if 0:
...         return
...     if 0:
...         yield 2
>>> type(f())
<class 'generator'>

This one caused a crash (see SF bug 567538):

>>> def f():
...     for i in range(3):
...         try:
...             continue
...         finally:
...             yield i
...
>>> g = f()
>>> print(next(g))
0
>>> print(next(g))
1
>>> print(next(g))
2
>>> print(next(g))
Traceback (most recent call last):
StopIteration


Test the gi_code attribute

>>> def f():
...     yield 5
...
>>> g = f()
>>> g.gi_code is f.__code__
True
>>> next(g)
5
>>> next(g)
Traceback (most recent call last):
StopIteration
>>> g.gi_code is f.__code__
True


Test the __name__ attribute and the repr()

>>> def f():
...    yield 5
...
>>> g = f()
>>> g.__name__
'f'
>>> repr(g)  # doctest: +ELLIPSIS
'<generator object f at ...>'

Lambdas shouldn't have their usual return behavior.

>>> x = lambda: (yield 1)
>>> list(x())
[1]

>>> x = lambda: ((yield 1), (yield 2))
>>> list(x())
[1, 2]
"""

# conjoin is a simple backtracking generator, named in honor of Icon's
# "conjunction" control structure.  Pass a list of no-argument functions
# that return iterable objects.  Easiest to explain by example:  assume the
# function list [x, y, z] is passed.  Then conjoin acts like:
#
# def g():
#     values = [None] * 3
#     for values[0] in x():
#         for values[1] in y():
#             for values[2] in z():
#                 yield values
#
# So some 3-lists of values *may* be generated, each time we successfully
# get into the innermost loop.  If an iterator fails (is exhausted) before
# then, it "backtracks" to get the next value from the nearest enclosing
# iterator (the one "to the left"), and starts all over again at the next
# slot (pumps a fresh iterator).  Of course this is most useful when the
# iterators have side-effects, so that which values *can* be generated at
# each slot depend on the values iterated at previous slots.

def simple_conjoin(gs):

    values = [None] * len(gs)

    def gen(i):
        if i >= len(gs):
            yield values
        else:
            for values[i] in gs[i]():
                for x in gen(i+1):
                    yield x

    for x in gen(0):
        yield x

# That works fine, but recursing a level and checking i against len(gs) for
# each item produced is inefficient.  By doing manual loop unrolling across
# generator boundaries, it's possible to eliminate most of that overhead.
# This isn't worth the bother *in general* for generators, but conjoin() is
# a core building block for some CPU-intensive generator applications.

def conjoin(gs):

    n = len(gs)
    values = [None] * n

    # Do one loop nest at time recursively, until the # of loop nests
    # remaining is divisible by 3.

    def gen(i):
        if i >= n:
            yield values

        elif (n-i) % 3:
            ip1 = i+1
            for values[i] in gs[i]():
                for x in gen(ip1):
                    yield x

        else:
            for x in _gen3(i):
                yield x

    # Do three loop nests at a time, recursing only if at least three more
    # remain.  Don't call directly:  this is an internal optimization for
    # gen's use.

    def _gen3(i):
        assert i < n and (n-i) % 3 == 0
        ip1, ip2, ip3 = i+1, i+2, i+3
        g, g1, g2 = gs[i : ip3]

        if ip3 >= n:
            # These are the last three, so we can yield values directly.
            for values[i] in g():
                for values[ip1] in g1():
                    for values[ip2] in g2():
                        yield values

        else:
            # At least 6 loop nests remain; peel off 3 and recurse for the
            # rest.
            for values[i] in g():
                for values[ip1] in g1():
                    for values[ip2] in g2():
                        for x in _gen3(ip3):
                            yield x

    for x in gen(0):
        yield x

# And one more approach:  For backtracking apps like the Knight's Tour
# solver below, the number of backtracking levels can be enormous (one
# level per square, for the Knight's Tour, so that e.g. a 100x100 board
# needs 10,000 levels).  In such cases Python is likely to run out of
# stack space due to recursion.  So here's a recursion-free version of
# conjoin too.
# NOTE WELL:  This allows large problems to be solved with only trivial
# demands on stack space.  Without explicitly resumable generators, this is
# much harder to achieve.  OTOH, this is much slower (up to a factor of 2)
# than the fancy unrolled recursive conjoin.

def flat_conjoin(gs):  # rename to conjoin to run tests with this instead
    n = len(gs)
    values = [None] * n
    iters  = [None] * n
    _StopIteration = StopIteration  # make local because caught a *lot*
    i = 0
    while 1:
        # Descend.
        try:
            while i < n:
                it = iters[i] = gs[i]().__next__
                values[i] = it()
                i += 1
        except _StopIteration:
            pass
        else:
            assert i == n
            yield values

        # Backtrack until an older iterator can be resumed.
        i -= 1
        while i >= 0:
            try:
                values[i] = iters[i]()
                # Success!  Start fresh at next level.
                i += 1
                break
            except _StopIteration:
                # Continue backtracking.
                i -= 1
        else:
            assert i < 0
            break

# A conjoin-based N-Queens solver.

class Queens:
    def __init__(self, n):
        self.n = n
        rangen = range(n)

        # Assign a unique int to each column and diagonal.
        # columns:  n of those, range(n).
        # NW-SE diagonals: 2n-1 of these, i-j unique and invariant along
        # each, smallest i-j is 0-(n-1) = 1-n, so add n-1 to shift to 0-
        # based.
        # NE-SW diagonals: 2n-1 of these, i+j unique and invariant along
        # each, smallest i+j is 0, largest is 2n-2.

        # For each square, compute a bit vector of the columns and
        # diagonals it covers, and for each row compute a function that
        # generates the possibilities for the columns in that row.
        self.rowgenerators = []
        for i in rangen:
            rowuses = [(1 << j) |                  # column ordinal
                       (1 << (n + i-j + n-1)) |    # NW-SE ordinal
                       (1 << (n + 2*n-1 + i+j))    # NE-SW ordinal
                            for j in rangen]

            def rowgen(rowuses=rowuses):
                for j in rangen:
                    uses = rowuses[j]
                    if uses & self.used == 0:
                        self.used |= uses
                        yield j
                        self.used &= ~uses

            self.rowgenerators.append(rowgen)

    # Generate solutions.
    def solve(self):
        self.used = 0
        for row2col in conjoin(self.rowgenerators):
            yield row2col

    def printsolution(self, row2col):
        n = self.n
        assert n == len(row2col)
        sep = "+" + "-+" * n
        print(sep)
        for i in range(n):
            squares = [" " for j in range(n)]
            squares[row2col[i]] = "Q"
            print("|" + "|".join(squares) + "|")
            print(sep)

# A conjoin-based Knight's Tour solver.  This is pretty sophisticated
# (e.g., when used with flat_conjoin above, and passing hard=1 to the
# constructor, a 200x200 Knight's Tour was found quickly -- note that we're
# creating 10s of thousands of generators then!), and is lengthy.

class Knights:
    def __init__(self, m, n, hard=0):
        self.m, self.n = m, n

        # solve() will set up succs[i] to be a list of square #i's
        # successors.
        succs = self.succs = []

        # Remove i0 from each of its successor's successor lists, i.e.
        # successors can't go back to i0 again.  Return 0 if we can
        # detect this makes a solution impossible, else return 1.

        def remove_from_successors(i0, len=len):
            # If we remove all exits from a free square, we're dead:
            # even if we move to it next, we can't leave it again.
            # If we create a square with one exit, we must visit it next;
            # else somebody else will have to visit it, and since there's
            # only one adjacent, there won't be a way to leave it again.
            # Finally, if we create more than one free square with a
            # single exit, we can only move to one of them next, leaving
            # the other one a dead end.
            ne0 = ne1 = 0
            for i in succs[i0]:
                s = succs[i]
                s.remove(i0)
                e = len(s)
                if e == 0:
                    ne0 += 1
                elif e == 1:
                    ne1 += 1
            return ne0 == 0 and ne1 < 2

        # Put i0 back in each of its successor's successor lists.

        def add_to_successors(i0):
            for i in succs[i0]:
                succs[i].append(i0)

        # Generate the first move.
        def first():
            if m < 1 or n < 1:
                return

            # Since we're looking for a cycle, it doesn't matter where we
            # start.  Starting in a corner makes the 2nd move easy.
            corner = self.coords2index(0, 0)
            remove_from_successors(corner)
            self.lastij = corner
            yield corner
            add_to_successors(corner)

        # Generate the second moves.
        def second():
            corner = self.coords2index(0, 0)
            assert self.lastij == corner  # i.e., we started in the corner
            if m < 3 or n < 3:
                return
            assert len(succs[corner]) == 2
            assert self.coords2index(1, 2) in succs[corner]
            assert self.coords2index(2, 1) in succs[corner]
            # Only two choices.  Whichever we pick, the other must be the
            # square picked on move m*n, as it's the only way to get back
            # to (0, 0).  Save its index in self.final so that moves before
            # the last know it must be kept free.
            for i, j in (1, 2), (2, 1):
                this  = self.coords2index(i, j)
                final = self.coords2index(3-i, 3-j)
                self.final = final

                remove_from_successors(this)
                succs[final].append(corner)
                self.lastij = this
                yield this
                succs[final].remove(corner)
                add_to_successors(this)

        # Generate moves 3 through m*n-1.
        def advance(len=len):
            # If some successor has only one exit, must take it.
            # Else favor successors with fewer exits.
            candidates = []
            for i in succs[self.lastij]:
                e = len(succs[i])
                assert e > 0, "else remove_from_successors() pruning flawed"
                if e == 1:
                    candidates = [(e, i)]
                    break
                candidates.append((e, i))
            else:
                candidates.sort()

            for e, i in candidates:
                if i != self.final:
                    if remove_from_successors(i):
                        self.lastij = i
                        yield i
                    add_to_successors(i)

        # Generate moves 3 through m*n-1.  Alternative version using a
        # stronger (but more expensive) heuristic to order successors.
        # Since the # of backtracking levels is m*n, a poor move early on
        # can take eons to undo.  Smallest square board for which this
        # matters a lot is 52x52.
        def advance_hard(vmid=(m-1)/2.0, hmid=(n-1)/2.0, len=len):
            # If some successor has only one exit, must take it.
            # Else favor successors with fewer exits.
            # Break ties via max distance from board centerpoint (favor
            # corners and edges whenever possible).
            candidates = []
            for i in succs[self.lastij]:
                e = len(succs[i])
                assert e > 0, "else remove_from_successors() pruning flawed"
                if e == 1:
                    candidates = [(e, 0, i)]
                    break
                i1, j1 = self.index2coords(i)
                d = (i1 - vmid)**2 + (j1 - hmid)**2
                candidates.append((e, -d, i))
            else:
                candidates.sort()

            for e, d, i in candidates:
                if i != self.final:
                    if remove_from_successors(i):
                        self.lastij = i
                        yield i
                    add_to_successors(i)

        # Generate the last move.
        def last():
            assert self.final in succs[self.lastij]
            yield self.final

        if m*n < 4:
            self.squaregenerators = [first]
        else:
            self.squaregenerators = [first, second] + \
                [hard and advance_hard or advance] * (m*n - 3) + \
                [last]

    def coords2index(self, i, j):
        assert 0 <= i < self.m
        assert 0 <= j < self.n
        return i * self.n + j

    def index2coords(self, index):
        assert 0 <= index < self.m * self.n
        return divmod(index, self.n)

    def _init_board(self):
        succs = self.succs
        del succs[:]
        m, n = self.m, self.n
        c2i = self.coords2index

        offsets = [( 1,  2), ( 2,  1), ( 2, -1), ( 1, -2),
                   (-1, -2), (-2, -1), (-2,  1), (-1,  2)]
        rangen = range(n)
        for i in range(m):
            for j in rangen:
                s = [c2i(i+io, j+jo) for io, jo in offsets
                                     if 0 <= i+io < m and
                                        0 <= j+jo < n]
                succs.append(s)

    # Generate solutions.
    def solve(self):
        self._init_board()
        for x in conjoin(self.squaregenerators):
            yield x

    def printsolution(self, x):
        m, n = self.m, self.n
        assert len(x) == m*n
        w = len(str(m*n))
        format = "%" + str(w) + "d"

        squares = [[None] * n for i in range(m)]
        k = 1
        for i in x:
            i1, j1 = self.index2coords(i)
            squares[i1][j1] = format % k
            k += 1

        sep = "+" + ("-" * w + "+") * n
        print(sep)
        for i in range(m):
            row = squares[i]
            print("|" + "|".join(row) + "|")
            print(sep)

conjoin_tests = """

Generate the 3-bit binary numbers in order.  This illustrates dumbest-
possible use of conjoin, just to generate the full cross-product.

>>> for c in conjoin([lambda: iter((0, 1))] * 3):
...     print(c)
[0, 0, 0]
[0, 0, 1]
[0, 1, 0]
[0, 1, 1]
[1, 0, 0]
[1, 0, 1]
[1, 1, 0]
[1, 1, 1]

For efficiency in typical backtracking apps, conjoin() yields the same list
object each time.  So if you want to save away a full account of its
generated sequence, you need to copy its results.

>>> def gencopy(iterator):
...     for x in iterator:
...         yield x[:]

>>> for n in range(10):
...     all = list(gencopy(conjoin([lambda: iter((0, 1))] * n)))
...     print(n, len(all), all[0] == [0] * n, all[-1] == [1] * n)
0 1 True True
1 2 True True
2 4 True True
3 8 True True
4 16 True True
5 32 True True
6 64 True True
7 128 True True
8 256 True True
9 512 True True

And run an 8-queens solver.

>>> q = Queens(8)
>>> LIMIT = 2
>>> count = 0
>>> for row2col in q.solve():
...     count += 1
...     if count <= LIMIT:
...         print("Solution", count)
...         q.printsolution(row2col)
Solution 1
+-+-+-+-+-+-+-+-+
|Q| | | | | | | |
+-+-+-+-+-+-+-+-+
| | | | |Q| | | |
+-+-+-+-+-+-+-+-+
| | | | | | | |Q|
+-+-+-+-+-+-+-+-+
| | | | | |Q| | |
+-+-+-+-+-+-+-+-+
| | |Q| | | | | |
+-+-+-+-+-+-+-+-+
| | | | | | |Q| |
+-+-+-+-+-+-+-+-+
| |Q| | | | | | |
+-+-+-+-+-+-+-+-+
| | | |Q| | | | |
+-+-+-+-+-+-+-+-+
Solution 2
+-+-+-+-+-+-+-+-+
|Q| | | | | | | |
+-+-+-+-+-+-+-+-+
| | | | | |Q| | |
+-+-+-+-+-+-+-+-+
| | | | | | | |Q|
+-+-+-+-+-+-+-+-+
| | |Q| | | | | |
+-+-+-+-+-+-+-+-+
| | | | | | |Q| |
+-+-+-+-+-+-+-+-+
| | | |Q| | | | |
+-+-+-+-+-+-+-+-+
| |Q| | | | | | |
+-+-+-+-+-+-+-+-+
| | | | |Q| | | |
+-+-+-+-+-+-+-+-+

>>> print(count, "solutions in all.")
92 solutions in all.

And run a Knight's Tour on a 10x10 board.  Note that there are about
20,000 solutions even on a 6x6 board, so don't dare run this to exhaustion.

>>> k = Knights(10, 10)
>>> LIMIT = 2
>>> count = 0
>>> for x in k.solve():
...     count += 1
...     if count <= LIMIT:
...         print("Solution", count)
...         k.printsolution(x)
...     else:
...         break
Solution 1
+---+---+---+---+---+---+---+---+---+---+
|  1| 58| 27| 34|  3| 40| 29| 10|  5|  8|
+---+---+---+---+---+---+---+---+---+---+
| 26| 35|  2| 57| 28| 33|  4|  7| 30| 11|
+---+---+---+---+---+---+---+---+---+---+
| 59|100| 73| 36| 41| 56| 39| 32|  9|  6|
+---+---+---+---+---+---+---+---+---+---+
| 74| 25| 60| 55| 72| 37| 42| 49| 12| 31|
+---+---+---+---+---+---+---+---+---+---+
| 61| 86| 99| 76| 63| 52| 47| 38| 43| 50|
+---+---+---+---+---+---+---+---+---+---+
| 24| 75| 62| 85| 54| 71| 64| 51| 48| 13|
+---+---+---+---+---+---+---+---+---+---+
| 87| 98| 91| 80| 77| 84| 53| 46| 65| 44|
+---+---+---+---+---+---+---+---+---+---+
| 90| 23| 88| 95| 70| 79| 68| 83| 14| 17|
+---+---+---+---+---+---+---+---+---+---+
| 97| 92| 21| 78| 81| 94| 19| 16| 45| 66|
+---+---+---+---+---+---+---+---+---+---+
| 22| 89| 96| 93| 20| 69| 82| 67| 18| 15|
+---+---+---+---+---+---+---+---+---+---+
Solution 2
+---+---+---+---+---+---+---+---+---+---+
|  1| 58| 27| 34|  3| 40| 29| 10|  5|  8|
+---+---+---+---+---+---+---+---+---+---+
| 26| 35|  2| 57| 28| 33|  4|  7| 30| 11|
+---+---+---+---+---+---+---+---+---+---+
| 59|100| 73| 36| 41| 56| 39| 32|  9|  6|
+---+---+---+---+---+---+---+---+---+---+
| 74| 25| 60| 55| 72| 37| 42| 49| 12| 31|
+---+---+---+---+---+---+---+---+---+---+
| 61| 86| 99| 76| 63| 52| 47| 38| 43| 50|
+---+---+---+---+---+---+---+---+---+---+
| 24| 75| 62| 85| 54| 71| 64| 51| 48| 13|
+---+---+---+---+---+---+---+---+---+---+
| 87| 98| 89| 80| 77| 84| 53| 46| 65| 44|
+---+---+---+---+---+---+---+---+---+---+
| 90| 23| 92| 95| 70| 79| 68| 83| 14| 17|
+---+---+---+---+---+---+---+---+---+---+
| 97| 88| 21| 78| 81| 94| 19| 16| 45| 66|
+---+---+---+---+---+---+---+---+---+---+
| 22| 91| 96| 93| 20| 69| 82| 67| 18| 15|
+---+---+---+---+---+---+---+---+---+---+
"""

weakref_tests = """\
Generators are weakly referencable:

>>> import weakref
>>> def gen():
...     yield 'foo!'
...
>>> wr = weakref.ref(gen)
>>> wr() is gen
True
>>> p = weakref.proxy(gen)

Generator-iterators are weakly referencable as well:

>>> gi = gen()
>>> wr = weakref.ref(gi)
>>> wr() is gi
True
>>> p = weakref.proxy(gi)
>>> list(p)
['foo!']

"""

coroutine_tests = """\
>>> from test.support import gc_collect

Sending a value into a started generator:

>>> def f():
...     print((yield 1))
...     yield 2
>>> g = f()
>>> next(g)
1
>>> g.send(42)
42
2

Sending a value into a new generator produces a TypeError:

>>> f().send("foo")
Traceback (most recent call last):
...
TypeError: can't send non-None value to a just-started generator


Yield by itself yields None:

>>> def f(): yield
>>> list(f())
[None]


Yield is allowed only in the outermost iterable in generator expression:

>>> def f(): list(i for i in [(yield 26)])
>>> type(f())
<class 'generator'>


A yield expression with augmented assignment.

>>> def coroutine(seq):
...     count = 0
...     while count < 200:
...         count += yield
...         seq.append(count)
>>> seq = []
>>> c = coroutine(seq)
>>> next(c)
>>> print(seq)
[]
>>> c.send(10)
>>> print(seq)
[10]
>>> c.send(10)
>>> print(seq)
[10, 20]
>>> c.send(10)
>>> print(seq)
[10, 20, 30]


Check some syntax errors for yield expressions:

>>> f=lambda: (yield 1),(yield 2)
Traceback (most recent call last):
  ...
SyntaxError: 'yield' outside function

>>> f=lambda: (yield from (1,2)), (yield from (3,4))
Traceback (most recent call last):
  ...
SyntaxError: 'yield from' outside function

>>> yield from [1,2]
Traceback (most recent call last):
  ...
SyntaxError: 'yield from' outside function

>>> def f(): x = yield = y
Traceback (most recent call last):
  ...
SyntaxError: assignment to yield expression not possible

>>> def f(): (yield bar) = y
Traceback (most recent call last):
  ...
SyntaxError: cannot assign to yield expression here. Maybe you meant '==' instead of '='?

>>> def f(): (yield bar) += y
Traceback (most recent call last):
  ...
SyntaxError: 'yield expression' is an illegal expression for augmented assignment


Now check some throw() conditions:

>>> def f():
...     while True:
...         try:
...             print((yield))
...         except ValueError as v:
...             print("caught ValueError (%s)" % (v))
>>> import sys
>>> g = f()
>>> next(g)

>>> g.throw(ValueError) # type only
caught ValueError ()

>>> g.throw(ValueError("xyz"))  # value only
caught ValueError (xyz)

>>> import warnings
>>> old_filters = warnings.filters.copy()
>>> warnings.filterwarnings("ignore", category=DeprecationWarning)

# Filter DeprecationWarning: regarding the (type, val, tb) signature of throw().
# Deprecation warnings are re-enabled below.

>>> g.throw(ValueError, ValueError(1))   # value+matching type
caught ValueError (1)

>>> g.throw(ValueError, TypeError(1))  # mismatched type, rewrapped
caught ValueError (1)

>>> g.throw(ValueError, ValueError(1), None)   # explicit None traceback
caught ValueError (1)

>>> g.throw(ValueError(1), "foo")       # bad args
Traceback (most recent call last):
  ...
TypeError: instance exception may not have a separate value

>>> g.throw(ValueError, "foo", 23)      # bad args
Traceback (most recent call last):
  ...
TypeError: throw() third argument must be a traceback object

>>> g.throw("abc")
Traceback (most recent call last):
  ...
TypeError: exceptions must be classes or instances deriving from BaseException, not str

>>> g.throw(0)
Traceback (most recent call last):
  ...
TypeError: exceptions must be classes or instances deriving from BaseException, not int

>>> g.throw(list)
Traceback (most recent call last):
  ...
TypeError: exceptions must be classes or instances deriving from BaseException, not type

>>> def throw(g,exc):
...     try:
...         raise exc
...     except:
...         g.throw(*sys.exc_info())
>>> throw(g,ValueError) # do it with traceback included
caught ValueError ()

>>> g.send(1)
1

>>> throw(g,TypeError)  # terminate the generator
Traceback (most recent call last):
  ...
TypeError

>>> print(g.gi_frame)
None

>>> g.send(2)
Traceback (most recent call last):
  ...
StopIteration

>>> g.throw(ValueError,6)       # throw on closed generator
Traceback (most recent call last):
  ...
ValueError: 6

>>> f().throw(ValueError,7)     # throw on just-opened generator
Traceback (most recent call last):
  ...
ValueError: 7

>>> warnings.filters[:] = old_filters

# Re-enable DeprecationWarning: the (type, val, tb) exception representation is deprecated,
#                               and may be removed in a future version of Python.

Plain "raise" inside a generator should preserve the traceback (#13188).
The traceback should have 3 levels:
- g.throw()
- f()
- 1/0

>>> def f():
...     try:
...         yield
...     except:
...         raise
>>> g = f()
>>> try:
...     1/0
... except ZeroDivisionError as v:
...     try:
...         g.throw(v)
...     except Exception as w:
...         tb = w.__traceback__
>>> levels = 0
>>> while tb:
...     levels += 1
...     tb = tb.tb_next
>>> levels
3

Now let's try closing a generator:

>>> def f():
...     try: yield
...     except GeneratorExit:
...         print("exiting")

>>> g = f()
>>> next(g)
>>> g.close()
exiting
>>> g.close()  # should be no-op now

>>> f().close()  # close on just-opened generator should be fine

>>> def f(): yield      # an even simpler generator
>>> f().close()         # close before opening
>>> g = f()
>>> next(g)
>>> g.close()           # close normally

And finalization:

>>> def f():
...     try: yield
...     finally:
...         print("exiting")

>>> g = f()
>>> next(g)
>>> del g; gc_collect()  # For PyPy or other GCs.
exiting


GeneratorExit is not caught by except Exception:

>>> def f():
...     try: yield
...     except Exception:
...         print('except')
...     finally:
...         print('finally')

>>> g = f()
>>> next(g)
>>> del g; gc_collect()  # For PyPy or other GCs.
finally


Now let's try some ill-behaved generators:

>>> def f():
...     try: yield
...     except GeneratorExit:
...         yield "foo!"
>>> g = f()
>>> next(g)
>>> g.close()
Traceback (most recent call last):
  ...
RuntimeError: generator ignored GeneratorExit
>>> g.close()


Our ill-behaved code should be invoked during GC:

>>> with support.catch_unraisable_exception() as cm:
...     g = f()
...     next(g)
...     del g
...
...     cm.unraisable.exc_type == RuntimeError
...     "generator ignored GeneratorExit" in str(cm.unraisable.exc_value)
...     cm.unraisable.exc_traceback is not None
True
True
True

And errors thrown during closing should propagate:

>>> def f():
...     try: yield
...     except GeneratorExit:
...         raise TypeError("fie!")
>>> g = f()
>>> next(g)
>>> g.close()
Traceback (most recent call last):
  ...
TypeError: fie!


Ensure that various yield expression constructs make their
enclosing function a generator:

>>> def f(): x += yield
>>> type(f())
<class 'generator'>

>>> def f(): x = yield
>>> type(f())
<class 'generator'>

>>> def f(): lambda x=(yield): 1
>>> type(f())
<class 'generator'>

>>> def f(d): d[(yield "a")] = d[(yield "b")] = 27
>>> data = [1,2]
>>> g = f(data)
>>> type(g)
<class 'generator'>
>>> g.send(None)
'a'
>>> data
[1, 2]
>>> g.send(0)
'b'
>>> data
[27, 2]
>>> try: g.send(1)
... except StopIteration: pass
>>> data
[27, 27]

"""

refleaks_tests = """
Prior to adding cycle-GC support to itertools.tee, this code would leak
references. We add it to the standard suite so the routine refleak-tests
would trigger if it starts being uncleanable again.

>>> import itertools
>>> def leak():
...     class gen:
...         def __iter__(self):
...             return self
...         def __next__(self):
...             return self.item
...     g = gen()
...     head, tail = itertools.tee(g)
...     g.item = head
...     return head
>>> it = leak()

Make sure to also test the involvement of the tee-internal teedataobject,
which stores returned items.

>>> item = next(it)



This test leaked at one point due to generator finalization/destruction.
It was copied from Lib/test/leakers/test_generator_cycle.py before the file
was removed.

>>> def leak():
...    def gen():
...        while True:
...            yield g
...    g = gen()

>>> leak()



This test isn't really generator related, but rather exception-in-cleanup
related. The coroutine tests (above) just happen to cause an exception in
the generator's __del__ (tp_del) method. We can also test for this
explicitly, without generators. We do have to redirect stderr to avoid
printing warnings and to doublecheck that we actually tested what we wanted
to test.

>>> from test import support
>>> class Leaker:
...     def __del__(self):
...         def invoke(message):
...             raise RuntimeError(message)
...         invoke("del failed")
...
>>> with support.catch_unraisable_exception() as cm:
...     l = Leaker()
...     del l
...
...     cm.unraisable.object == Leaker.__del__
...     cm.unraisable.exc_type == RuntimeError
...     str(cm.unraisable.exc_value) == "del failed"
...     cm.unraisable.exc_traceback is not None
True
True
True
True


These refleak tests should perhaps be in a testfile of their own,
test_generators just happened to be the test that drew these out.

"""

# __test__ = {"tut":      tutorial_tests,
#             "pep":      pep_tests,
#             "email":    email_tests,
#             "fun":      fun_tests,
#             "syntax":   syntax_tests,
#             "conjoin":  conjoin_tests,
#             "weakref":  weakref_tests,
#             "coroutine":  coroutine_tests,
#             "refleaks": refleaks_tests,
#             }

# def load_tests(loader, tests, pattern):
#     # ======= BEGIN Dynamo patch =======
#     suite = doctest.DocTestSuite()
#     for test in suite:
#         # Dynamically change base class
#         test.__class__ = type(test.__class__.__name__, (__TestCase, test.__class__), {})
#     tests.addTests(suite)
#     # ======= END DYNAMO PATCH =======
#     return tests


if __name__ == "__main__":
    run_tests()
