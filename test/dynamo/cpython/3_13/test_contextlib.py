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

"""Unit tests for contextlib.py, and other context managers."""

import io
import os
import sys
import tempfile
import threading
import traceback
import unittest
from contextlib import *  # Tests __all__
from test import support
from test.support import os_helper
from test.support.testcase import ExceptionIsLikeMixin
import weakref


class TestAbstractContextManager(__TestCase):

    def test_enter(self):
        class DefaultEnter(AbstractContextManager):
            def __exit__(self, *args):
                super().__exit__(*args)

        manager = DefaultEnter()
        self.assertIs(manager.__enter__(), manager)

    def test_slots(self):
        class DefaultContextManager(AbstractContextManager):
            __slots__ = ()

            def __exit__(self, *args):
                super().__exit__(*args)

        with self.assertRaises(AttributeError):
            DefaultContextManager().var = 42

    def test_exit_is_abstract(self):
        class MissingExit(AbstractContextManager):
            pass

        with self.assertRaises(TypeError):
            MissingExit()

    def test_structural_subclassing(self):
        class ManagerFromScratch:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_value, traceback):
                return None

        self.assertTrue(issubclass(ManagerFromScratch, AbstractContextManager))

        class DefaultEnter(AbstractContextManager):
            def __exit__(self, *args):
                super().__exit__(*args)

        self.assertTrue(issubclass(DefaultEnter, AbstractContextManager))

        class NoEnter(ManagerFromScratch):
            __enter__ = None

        self.assertFalse(issubclass(NoEnter, AbstractContextManager))

        class NoExit(ManagerFromScratch):
            __exit__ = None

        self.assertFalse(issubclass(NoExit, AbstractContextManager))


class ContextManagerTestCase(__TestCase):

    def test_contextmanager_plain(self):
        state = []
        @contextmanager
        def woohoo():
            state.append(1)
            yield 42
            state.append(999)
        with woohoo() as x:
            self.assertEqual(state, [1])
            self.assertEqual(x, 42)
            state.append(x)
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_finally(self):
        state = []
        @contextmanager
        def woohoo():
            state.append(1)
            try:
                yield 42
            finally:
                state.append(999)
        with self.assertRaises(ZeroDivisionError):
            with woohoo() as x:
                self.assertEqual(state, [1])
                self.assertEqual(x, 42)
                state.append(x)
                raise ZeroDivisionError()
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_traceback(self):
        @contextmanager
        def f():
            yield

        try:
            with f():
                1/0
        except ZeroDivisionError as e:
            frames = traceback.extract_tb(e.__traceback__)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].name, 'test_contextmanager_traceback')
        self.assertEqual(frames[0].line, '1/0')

        # Repeat with RuntimeError (which goes through a different code path)
        class RuntimeErrorSubclass(RuntimeError):
            pass

        try:
            with f():
                raise RuntimeErrorSubclass(42)
        except RuntimeErrorSubclass as e:
            frames = traceback.extract_tb(e.__traceback__)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].name, 'test_contextmanager_traceback')
        self.assertEqual(frames[0].line, 'raise RuntimeErrorSubclass(42)')

        class StopIterationSubclass(StopIteration):
            pass

        for stop_exc in (
            StopIteration('spam'),
            StopIterationSubclass('spam'),
        ):
            with self.subTest(type=type(stop_exc)):
                try:
                    with f():
                        raise stop_exc
                except type(stop_exc) as e:
                    self.assertIs(e, stop_exc)
                    frames = traceback.extract_tb(e.__traceback__)
                else:
                    self.fail(f'{stop_exc} was suppressed')

                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0].name, 'test_contextmanager_traceback')
                self.assertEqual(frames[0].line, 'raise stop_exc')

    def test_contextmanager_no_reraise(self):
        @contextmanager
        def whee():
            yield
        ctx = whee()
        ctx.__enter__()
        # Calling __exit__ should not result in an exception
        self.assertFalse(ctx.__exit__(TypeError, TypeError("foo"), None))

    def test_contextmanager_trap_yield_after_throw(self):
        @contextmanager
        def whoo():
            try:
                yield
            except:
                yield
        ctx = whoo()
        ctx.__enter__()
        with self.assertRaises(RuntimeError):
            ctx.__exit__(TypeError, TypeError("foo"), None)
        if support.check_impl_detail(cpython=True):
            # The "gen" attribute is an implementation detail.
            self.assertFalse(ctx.gen.gi_suspended)

    def test_contextmanager_trap_no_yield(self):
        @contextmanager
        def whoo():
            if False:
                yield
        ctx = whoo()
        with self.assertRaises(RuntimeError):
            ctx.__enter__()

    def test_contextmanager_trap_second_yield(self):
        @contextmanager
        def whoo():
            yield
            yield
        ctx = whoo()
        ctx.__enter__()
        with self.assertRaises(RuntimeError):
            ctx.__exit__(None, None, None)
        if support.check_impl_detail(cpython=True):
            # The "gen" attribute is an implementation detail.
            self.assertFalse(ctx.gen.gi_suspended)

    def test_contextmanager_non_normalised(self):
        @contextmanager
        def whoo():
            try:
                yield
            except RuntimeError:
                raise SyntaxError

        ctx = whoo()
        ctx.__enter__()
        with self.assertRaises(SyntaxError):
            ctx.__exit__(RuntimeError, None, None)

    def test_contextmanager_except(self):
        state = []
        @contextmanager
        def woohoo():
            state.append(1)
            try:
                yield 42
            except ZeroDivisionError as e:
                state.append(e.args[0])
                self.assertEqual(state, [1, 42, 999])
        with woohoo() as x:
            self.assertEqual(state, [1])
            self.assertEqual(x, 42)
            state.append(x)
            raise ZeroDivisionError(999)
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_except_stopiter(self):
        @contextmanager
        def woohoo():
            yield

        class StopIterationSubclass(StopIteration):
            pass

        for stop_exc in (StopIteration('spam'), StopIterationSubclass('spam')):
            with self.subTest(type=type(stop_exc)):
                try:
                    with woohoo():
                        raise stop_exc
                except Exception as ex:
                    self.assertIs(ex, stop_exc)
                else:
                    self.fail(f'{stop_exc} was suppressed')

    def test_contextmanager_except_pep479(self):
        code = """\
from __future__ import generator_stop
from contextlib import contextmanager
@contextmanager
def woohoo():
    yield
"""
        locals = {}
        exec(code, locals, locals)
        woohoo = locals['woohoo']

        stop_exc = StopIteration('spam')
        try:
            with woohoo():
                raise stop_exc
        except Exception as ex:
            self.assertIs(ex, stop_exc)
        else:
            self.fail('StopIteration was suppressed')

    def test_contextmanager_do_not_unchain_non_stopiteration_exceptions(self):
        @contextmanager
        def test_issue29692():
            try:
                yield
            except Exception as exc:
                raise RuntimeError('issue29692:Chained') from exc
        try:
            with test_issue29692():
                raise ZeroDivisionError
        except Exception as ex:
            self.assertIs(type(ex), RuntimeError)
            self.assertEqual(ex.args[0], 'issue29692:Chained')
            self.assertIsInstance(ex.__cause__, ZeroDivisionError)

        try:
            with test_issue29692():
                raise StopIteration('issue29692:Unchained')
        except Exception as ex:
            self.assertIs(type(ex), StopIteration)
            self.assertEqual(ex.args[0], 'issue29692:Unchained')
            self.assertIsNone(ex.__cause__)

    def test_contextmanager_wrap_runtimeerror(self):
        @contextmanager
        def woohoo():
            try:
                yield
            except Exception as exc:
                raise RuntimeError(f'caught {exc}') from exc

        with self.assertRaises(RuntimeError):
            with woohoo():
                1 / 0

        # If the context manager wrapped StopIteration in a RuntimeError,
        # we also unwrap it, because we can't tell whether the wrapping was
        # done by the generator machinery or by the generator itself.
        with self.assertRaises(StopIteration):
            with woohoo():
                raise StopIteration

    def _create_contextmanager_attribs(self):
        def attribs(**kw):
            def decorate(func):
                for k,v in kw.items():
                    setattr(func,k,v)
                return func
            return decorate
        @contextmanager
        @attribs(foo='bar')
        def baz(spam):
            """Whee!"""
            yield
        return baz

    def test_contextmanager_attribs(self):
        baz = self._create_contextmanager_attribs()
        self.assertEqual(baz.__name__,'baz')
        self.assertEqual(baz.foo, 'bar')

    @support.requires_docstrings
    def test_contextmanager_doc_attrib(self):
        baz = self._create_contextmanager_attribs()
        self.assertEqual(baz.__doc__, "Whee!")

    @support.requires_docstrings
    def test_instance_docstring_given_cm_docstring(self):
        baz = self._create_contextmanager_attribs()(None)
        self.assertEqual(baz.__doc__, "Whee!")

    def test_keywords(self):
        # Ensure no keyword arguments are inhibited
        @contextmanager
        def woohoo(self, func, args, kwds):
            yield (self, func, args, kwds)
        with woohoo(self=11, func=22, args=33, kwds=44) as target:
            self.assertEqual(target, (11, 22, 33, 44))

    def test_nokeepref(self):
        class A:
            pass

        @contextmanager
        def woohoo(a, b):
            a = weakref.ref(a)
            b = weakref.ref(b)
            # Allow test to work with a non-refcounted GC
            support.gc_collect()
            self.assertIsNone(a())
            self.assertIsNone(b())
            yield

        with woohoo(A(), b=A()):
            pass

    def test_param_errors(self):
        @contextmanager
        def woohoo(a, *, b):
            yield

        with self.assertRaises(TypeError):
            woohoo()
        with self.assertRaises(TypeError):
            woohoo(3, 5)
        with self.assertRaises(TypeError):
            woohoo(b=3)

    def test_recursive(self):
        depth = 0
        ncols = 0
        @contextmanager
        def woohoo():
            nonlocal ncols
            ncols += 1
            nonlocal depth
            before = depth
            depth += 1
            yield
            depth -= 1
            self.assertEqual(depth, before)

        @woohoo()
        def recursive():
            if depth < 10:
                recursive()

        recursive()
        self.assertEqual(ncols, 10)
        self.assertEqual(depth, 0)


class ClosingTestCase(__TestCase):

    @support.requires_docstrings
    def test_instance_docs(self):
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = closing.__doc__
        obj = closing(None)
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_closing(self):
        state = []
        class C:
            def close(self):
                state.append(1)
        x = C()
        self.assertEqual(state, [])
        with closing(x) as y:
            self.assertEqual(x, y)
        self.assertEqual(state, [1])

    def test_closing_error(self):
        state = []
        class C:
            def close(self):
                state.append(1)
        x = C()
        self.assertEqual(state, [])
        with self.assertRaises(ZeroDivisionError):
            with closing(x) as y:
                self.assertEqual(x, y)
                1 / 0
        self.assertEqual(state, [1])


class NullcontextTestCase(__TestCase):
    def test_nullcontext(self):
        class C:
            pass
        c = C()
        with nullcontext(c) as c_in:
            self.assertIs(c_in, c)


class FileContextTestCase(__TestCase):

    def testWithOpen(self):
        tfn = tempfile.mktemp()
        try:
            with open(tfn, "w", encoding="utf-8") as f:
                self.assertFalse(f.closed)
                f.write("Booh\n")
            self.assertTrue(f.closed)
            with self.assertRaises(ZeroDivisionError):
                with open(tfn, "r", encoding="utf-8") as f:
                    self.assertFalse(f.closed)
                    self.assertEqual(f.read(), "Booh\n")
                    1 / 0
            self.assertTrue(f.closed)
        finally:
            os_helper.unlink(tfn)

class LockContextTestCase(__TestCase):

    def boilerPlate(self, lock, locked):
        self.assertFalse(locked())
        with lock:
            self.assertTrue(locked())
        self.assertFalse(locked())
        with self.assertRaises(ZeroDivisionError):
            with lock:
                self.assertTrue(locked())
                1 / 0
        self.assertFalse(locked())

    def testWithLock(self):
        lock = threading.Lock()
        self.boilerPlate(lock, lock.locked)

    def testWithRLock(self):
        lock = threading.RLock()
        self.boilerPlate(lock, lock._is_owned)

    def testWithCondition(self):
        lock = threading.Condition()
        def locked():
            return lock._is_owned()
        self.boilerPlate(lock, locked)

    def testWithSemaphore(self):
        lock = threading.Semaphore()
        def locked():
            if lock.acquire(False):
                lock.release()
                return False
            else:
                return True
        self.boilerPlate(lock, locked)

    def testWithBoundedSemaphore(self):
        lock = threading.BoundedSemaphore()
        def locked():
            if lock.acquire(False):
                lock.release()
                return False
            else:
                return True
        self.boilerPlate(lock, locked)


class mycontext(ContextDecorator):
    """Example decoration-compatible context manager for testing"""
    started = False
    exc = None
    catch = False

    def __enter__(self):
        self.started = True
        return self

    def __exit__(self, *exc):
        self.exc = exc
        return self.catch


class TestContextDecorator(__TestCase):

    @support.requires_docstrings
    def test_instance_docs(self):
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = mycontext.__doc__
        obj = mycontext()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_contextdecorator(self):
        context = mycontext()
        with context as result:
            self.assertIs(result, context)
            self.assertTrue(context.started)

        self.assertEqual(context.exc, (None, None, None))


    def test_contextdecorator_with_exception(self):
        context = mycontext()

        with self.assertRaisesRegex(NameError, 'foo'):
            with context:
                raise NameError('foo')
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)

        context = mycontext()
        context.catch = True
        with context:
            raise NameError('foo')
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)


    def test_decorator(self):
        context = mycontext()

        @context
        def test():
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
        test()
        self.assertEqual(context.exc, (None, None, None))


    def test_decorator_with_exception(self):
        context = mycontext()

        @context
        def test():
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
            raise NameError('foo')

        with self.assertRaisesRegex(NameError, 'foo'):
            test()
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)


    def test_decorating_method(self):
        context = mycontext()

        class Test(object):

            @context
            def method(self, a, b, c=None):
                self.a = a
                self.b = b
                self.c = c

        # these tests are for argument passing when used as a decorator
        test = Test()
        test.method(1, 2)
        self.assertEqual(test.a, 1)
        self.assertEqual(test.b, 2)
        self.assertEqual(test.c, None)

        test = Test()
        test.method('a', 'b', 'c')
        self.assertEqual(test.a, 'a')
        self.assertEqual(test.b, 'b')
        self.assertEqual(test.c, 'c')

        test = Test()
        test.method(a=1, b=2)
        self.assertEqual(test.a, 1)
        self.assertEqual(test.b, 2)


    def test_typo_enter(self):
        class mycontext(ContextDecorator):
            def __unter__(self):
                pass
            def __exit__(self, *exc):
                pass

        with self.assertRaisesRegex(TypeError, 'the context manager'):
            with mycontext():
                pass


    def test_typo_exit(self):
        class mycontext(ContextDecorator):
            def __enter__(self):
                pass
            def __uxit__(self, *exc):
                pass

        with self.assertRaisesRegex(TypeError, 'the context manager.*__exit__'):
            with mycontext():
                pass


    def test_contextdecorator_as_mixin(self):
        class somecontext(object):
            started = False
            exc = None

            def __enter__(self):
                self.started = True
                return self

            def __exit__(self, *exc):
                self.exc = exc

        class mycontext(somecontext, ContextDecorator):
            pass

        context = mycontext()
        @context
        def test():
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
        test()
        self.assertEqual(context.exc, (None, None, None))


    def test_contextmanager_as_decorator(self):
        @contextmanager
        def woohoo(y):
            state.append(y)
            yield
            state.append(999)

        state = []
        @woohoo(1)
        def test(x):
            self.assertEqual(state, [1])
            state.append(x)
        test('something')
        self.assertEqual(state, [1, 'something', 999])

        # Issue #11647: Ensure the decorated function is 'reusable'
        state = []
        test('something else')
        self.assertEqual(state, [1, 'something else', 999])


class _TestBaseExitStack:
    exit_stack = None

    @support.requires_docstrings
    def test_instance_docs(self):
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = self.exit_stack.__doc__
        obj = self.exit_stack()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_resources(self):
        with self.exit_stack():
            pass

    def test_callback(self):
        expected = [
            ((), {}),
            ((1,), {}),
            ((1,2), {}),
            ((), dict(example=1)),
            ((1,), dict(example=1)),
            ((1,2), dict(example=1)),
            ((1,2), dict(self=3, callback=4)),
        ]
        result = []
        def _exit(*args, **kwds):
            """Test metadata propagation"""
            result.append((args, kwds))
        with self.exit_stack() as stack:
            for args, kwds in reversed(expected):
                if args and kwds:
                    f = stack.callback(_exit, *args, **kwds)
                elif args:
                    f = stack.callback(_exit, *args)
                elif kwds:
                    f = stack.callback(_exit, **kwds)
                else:
                    f = stack.callback(_exit)
                self.assertIs(f, _exit)
            for wrapper in stack._exit_callbacks:
                self.assertIs(wrapper[1].__wrapped__, _exit)
                self.assertNotEqual(wrapper[1].__name__, _exit.__name__)
                self.assertIsNone(wrapper[1].__doc__, _exit.__doc__)
        self.assertEqual(result, expected)

        result = []
        with self.exit_stack() as stack:
            with self.assertRaises(TypeError):
                stack.callback(arg=1)
            with self.assertRaises(TypeError):
                self.exit_stack.callback(arg=2)
            with self.assertRaises(TypeError):
                stack.callback(callback=_exit, arg=3)
        self.assertEqual(result, [])

    def test_push(self):
        exc_raised = ZeroDivisionError
        def _expect_exc(exc_type, exc, exc_tb):
            self.assertIs(exc_type, exc_raised)
        def _suppress_exc(*exc_details):
            return True
        def _expect_ok(exc_type, exc, exc_tb):
            self.assertIsNone(exc_type)
            self.assertIsNone(exc)
            self.assertIsNone(exc_tb)
        class ExitCM(object):
            def __init__(self, check_exc):
                self.check_exc = check_exc
            def __enter__(self):
                self.fail("Should not be called!")
            def __exit__(self, *exc_details):
                self.check_exc(*exc_details)
        with self.exit_stack() as stack:
            stack.push(_expect_ok)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_ok)
            cm = ExitCM(_expect_ok)
            stack.push(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            stack.push(_suppress_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _suppress_exc)
            cm = ExitCM(_expect_exc)
            stack.push(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            stack.push(_expect_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_exc)
            stack.push(_expect_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_exc)
            1/0

    def test_enter_context(self):
        class TestCM(object):
            def __enter__(self):
                result.append(1)
            def __exit__(self, *exc_details):
                result.append(3)

        result = []
        cm = TestCM()
        with self.exit_stack() as stack:
            @stack.callback  # Registered first => cleaned up last
            def _exit():
                result.append(4)
            self.assertIsNotNone(_exit)
            stack.enter_context(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            result.append(2)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_enter_context_errors(self):
        class LacksEnterAndExit:
            pass
        class LacksEnter:
            def __exit__(self, *exc_info):
                pass
        class LacksExit:
            def __enter__(self):
                pass

        with self.exit_stack() as stack:
            with self.assertRaisesRegex(TypeError, 'the context manager'):
                stack.enter_context(LacksEnterAndExit())
            with self.assertRaisesRegex(TypeError, 'the context manager'):
                stack.enter_context(LacksEnter())
            with self.assertRaisesRegex(TypeError, 'the context manager'):
                stack.enter_context(LacksExit())
            self.assertFalse(stack._exit_callbacks)

    def test_close(self):
        result = []
        with self.exit_stack() as stack:
            @stack.callback
            def _exit():
                result.append(1)
            self.assertIsNotNone(_exit)
            stack.close()
            result.append(2)
        self.assertEqual(result, [1, 2])

    def test_pop_all(self):
        result = []
        with self.exit_stack() as stack:
            @stack.callback
            def _exit():
                result.append(3)
            self.assertIsNotNone(_exit)
            new_stack = stack.pop_all()
            result.append(1)
        result.append(2)
        new_stack.close()
        self.assertEqual(result, [1, 2, 3])

    def test_exit_raise(self):
        with self.assertRaises(ZeroDivisionError):
            with self.exit_stack() as stack:
                stack.push(lambda *exc: False)
                1/0

    def test_exit_suppress(self):
        with self.exit_stack() as stack:
            stack.push(lambda *exc: True)
            1/0

    def test_exit_exception_traceback(self):
        # This test captures the current behavior of ExitStack so that we know
        # if we ever unintendedly change it. It is not a statement of what the
        # desired behavior is (for instance, we may want to remove some of the
        # internal contextlib frames).

        def raise_exc(exc):
            raise exc

        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, ValueError)
                1/0
        except ValueError as e:
            exc = e

        self.assertIsInstance(exc, ValueError)
        ve_frames = traceback.extract_tb(exc.__traceback__)
        expected = \
            [('test_exit_exception_traceback', 'with self.exit_stack() as stack:')] + \
            self.callback_error_internal_frames + \
            [('_exit_wrapper', 'callback(*args, **kwds)'),
             ('raise_exc', 'raise exc')]

        self.assertEqual(
            [(f.name, f.line) for f in ve_frames], expected)

        self.assertIsInstance(exc.__context__, ZeroDivisionError)
        zde_frames = traceback.extract_tb(exc.__context__.__traceback__)
        self.assertEqual([(f.name, f.line) for f in zde_frames],
                         [('test_exit_exception_traceback', '1/0')])

    def test_exit_exception_chaining_reference(self):
        # Sanity check to make sure that ExitStack chaining matches
        # actual nested with statements
        class RaiseExc:
            def __init__(self, exc):
                self.exc = exc
            def __enter__(self):
                return self
            def __exit__(self, *exc_details):
                raise self.exc

        class RaiseExcWithContext:
            def __init__(self, outer, inner):
                self.outer = outer
                self.inner = inner
            def __enter__(self):
                return self
            def __exit__(self, *exc_details):
                try:
                    raise self.inner
                except:
                    raise self.outer

        class SuppressExc:
            def __enter__(self):
                return self
            def __exit__(self, *exc_details):
                type(self).saved_details = exc_details
                return True

        try:
            with RaiseExc(IndexError):
                with RaiseExcWithContext(KeyError, AttributeError):
                    with SuppressExc():
                        with RaiseExc(ValueError):
                            1 / 0
        except IndexError as exc:
            self.assertIsInstance(exc.__context__, KeyError)
            self.assertIsInstance(exc.__context__.__context__, AttributeError)
            # Inner exceptions were suppressed
            self.assertIsNone(exc.__context__.__context__.__context__)
        else:
            self.fail("Expected IndexError, but no exception was raised")
        # Check the inner exceptions
        inner_exc = SuppressExc.saved_details[1]
        self.assertIsInstance(inner_exc, ValueError)
        self.assertIsInstance(inner_exc.__context__, ZeroDivisionError)

    def test_exit_exception_chaining(self):
        # Ensure exception chaining matches the reference behaviour
        def raise_exc(exc):
            raise exc

        saved_details = None
        def suppress_exc(*exc_details):
            nonlocal saved_details
            saved_details = exc_details
            return True

        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, IndexError)
                stack.callback(raise_exc, KeyError)
                stack.callback(raise_exc, AttributeError)
                stack.push(suppress_exc)
                stack.callback(raise_exc, ValueError)
                1 / 0
        except IndexError as exc:
            self.assertIsInstance(exc.__context__, KeyError)
            self.assertIsInstance(exc.__context__.__context__, AttributeError)
            # Inner exceptions were suppressed
            self.assertIsNone(exc.__context__.__context__.__context__)
        else:
            self.fail("Expected IndexError, but no exception was raised")
        # Check the inner exceptions
        inner_exc = saved_details[1]
        self.assertIsInstance(inner_exc, ValueError)
        self.assertIsInstance(inner_exc.__context__, ZeroDivisionError)

    def test_exit_exception_explicit_none_context(self):
        # Ensure ExitStack chaining matches actual nested `with` statements
        # regarding explicit __context__ = None.

        class MyException(Exception):
            pass

        @contextmanager
        def my_cm():
            try:
                yield
            except BaseException:
                exc = MyException()
                try:
                    raise exc
                finally:
                    exc.__context__ = None

        @contextmanager
        def my_cm_with_exit_stack():
            with self.exit_stack() as stack:
                stack.enter_context(my_cm())
                yield stack

        for cm in (my_cm, my_cm_with_exit_stack):
            with self.subTest():
                try:
                    with cm():
                        raise IndexError()
                except MyException as exc:
                    self.assertIsNone(exc.__context__)
                else:
                    self.fail("Expected IndexError, but no exception was raised")

    def test_exit_exception_non_suppressing(self):
        # http://bugs.python.org/issue19092
        def raise_exc(exc):
            raise exc

        def suppress_exc(*exc_details):
            return True

        try:
            with self.exit_stack() as stack:
                stack.callback(lambda: None)
                stack.callback(raise_exc, IndexError)
        except Exception as exc:
            self.assertIsInstance(exc, IndexError)
        else:
            self.fail("Expected IndexError, but no exception was raised")

        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, KeyError)
                stack.push(suppress_exc)
                stack.callback(raise_exc, IndexError)
        except Exception as exc:
            self.assertIsInstance(exc, KeyError)
        else:
            self.fail("Expected KeyError, but no exception was raised")

    def test_exit_exception_with_correct_context(self):
        # http://bugs.python.org/issue20317
        @contextmanager
        def gets_the_context_right(exc):
            try:
                yield
            finally:
                raise exc

        exc1 = Exception(1)
        exc2 = Exception(2)
        exc3 = Exception(3)
        exc4 = Exception(4)

        # The contextmanager already fixes the context, so prior to the
        # fix, ExitStack would try to fix it *again* and get into an
        # infinite self-referential loop
        try:
            with self.exit_stack() as stack:
                stack.enter_context(gets_the_context_right(exc4))
                stack.enter_context(gets_the_context_right(exc3))
                stack.enter_context(gets_the_context_right(exc2))
                raise exc1
        except Exception as exc:
            self.assertIs(exc, exc4)
            self.assertIs(exc.__context__, exc3)
            self.assertIs(exc.__context__.__context__, exc2)
            self.assertIs(exc.__context__.__context__.__context__, exc1)
            self.assertIsNone(
                       exc.__context__.__context__.__context__.__context__)

    def test_exit_exception_with_existing_context(self):
        # Addresses a lack of test coverage discovered after checking in a
        # fix for issue 20317 that still contained debugging code.
        def raise_nested(inner_exc, outer_exc):
            try:
                raise inner_exc
            finally:
                raise outer_exc
        exc1 = Exception(1)
        exc2 = Exception(2)
        exc3 = Exception(3)
        exc4 = Exception(4)
        exc5 = Exception(5)
        try:
            with self.exit_stack() as stack:
                stack.callback(raise_nested, exc4, exc5)
                stack.callback(raise_nested, exc2, exc3)
                raise exc1
        except Exception as exc:
            self.assertIs(exc, exc5)
            self.assertIs(exc.__context__, exc4)
            self.assertIs(exc.__context__.__context__, exc3)
            self.assertIs(exc.__context__.__context__.__context__, exc2)
            self.assertIs(
                 exc.__context__.__context__.__context__.__context__, exc1)
            self.assertIsNone(
                exc.__context__.__context__.__context__.__context__.__context__)

    def test_body_exception_suppress(self):
        def suppress_exc(*exc_details):
            return True
        try:
            with self.exit_stack() as stack:
                stack.push(suppress_exc)
                1/0
        except IndexError as exc:
            self.fail("Expected no exception, got IndexError")

    def test_exit_exception_chaining_suppress(self):
        with self.exit_stack() as stack:
            stack.push(lambda *exc: True)
            stack.push(lambda *exc: 1/0)
            stack.push(lambda *exc: {}[1])

    def test_excessive_nesting(self):
        # The original implementation would die with RecursionError here
        with self.exit_stack() as stack:
            for i in range(10000):
                stack.callback(int)

    def test_instance_bypass(self):
        class Example(object): pass
        cm = Example()
        cm.__enter__ = object()
        cm.__exit__ = object()
        stack = self.exit_stack()
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            stack.enter_context(cm)
        stack.push(cm)
        self.assertIs(stack._exit_callbacks[-1][1], cm)

    def test_dont_reraise_RuntimeError(self):
        # https://bugs.python.org/issue27122
        class UniqueException(Exception): pass
        class UniqueRuntimeError(RuntimeError): pass

        @contextmanager
        def second():
            try:
                yield 1
            except Exception as exc:
                raise UniqueException("new exception") from exc

        @contextmanager
        def first():
            try:
                yield 1
            except Exception as exc:
                raise exc

        # The UniqueRuntimeError should be caught by second()'s exception
        # handler which chain raised a new UniqueException.
        with self.assertRaises(UniqueException) as err_ctx:
            with self.exit_stack() as es_ctx:
                es_ctx.enter_context(second())
                es_ctx.enter_context(first())
                raise UniqueRuntimeError("please no infinite loop.")

        exc = err_ctx.exception
        self.assertIsInstance(exc, UniqueException)
        self.assertIsInstance(exc.__context__, UniqueRuntimeError)
        self.assertIsNone(exc.__context__.__context__)
        self.assertIsNone(exc.__context__.__cause__)
        self.assertIs(exc.__cause__, exc.__context__)


class TestExitStack(_TestBaseExitStack, __TestCase):
    exit_stack = ExitStack
    callback_error_internal_frames = [
        ('__exit__', 'raise exc'),
        ('__exit__', 'if cb(*exc_details):'),
    ]


class _TestRedirectStream:

    redirect_stream = None
    orig_stream = None

    @support.requires_docstrings
    def test_instance_docs(self):
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = self.redirect_stream.__doc__
        obj = self.redirect_stream(None)
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_redirect_in_init(self):
        orig_stdout = getattr(sys, self.orig_stream)
        self.redirect_stream(None)
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)

    def test_redirect_to_string_io(self):
        f = io.StringIO()
        msg = "Consider an API like help(), which prints directly to stdout"
        orig_stdout = getattr(sys, self.orig_stream)
        with self.redirect_stream(f):
            print(msg, file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue().strip()
        self.assertEqual(s, msg)

    def test_enter_result_is_target(self):
        f = io.StringIO()
        with self.redirect_stream(f) as enter_result:
            self.assertIs(enter_result, f)

    def test_cm_is_reusable(self):
        f = io.StringIO()
        write_to_f = self.redirect_stream(f)
        orig_stdout = getattr(sys, self.orig_stream)
        with write_to_f:
            print("Hello", end=" ", file=getattr(sys, self.orig_stream))
        with write_to_f:
            print("World!", file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue()
        self.assertEqual(s, "Hello World!\n")

    def test_cm_is_reentrant(self):
        f = io.StringIO()
        write_to_f = self.redirect_stream(f)
        orig_stdout = getattr(sys, self.orig_stream)
        with write_to_f:
            print("Hello", end=" ", file=getattr(sys, self.orig_stream))
            with write_to_f:
                print("World!", file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue()
        self.assertEqual(s, "Hello World!\n")


class TestRedirectStdout(_TestRedirectStream, __TestCase):

    redirect_stream = redirect_stdout
    orig_stream = "stdout"


class TestRedirectStderr(_TestRedirectStream, __TestCase):

    redirect_stream = redirect_stderr
    orig_stream = "stderr"


class TestSuppress(ExceptionIsLikeMixin, __TestCase):

    @support.requires_docstrings
    def test_instance_docs(self):
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = suppress.__doc__
        obj = suppress()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_result_from_enter(self):
        with suppress(ValueError) as enter_result:
            self.assertIsNone(enter_result)

    def test_no_exception(self):
        with suppress(ValueError):
            self.assertEqual(pow(2, 5), 32)

    def test_exact_exception(self):
        with suppress(TypeError):
            len(5)

    def test_exception_hierarchy(self):
        with suppress(LookupError):
            'Hello'[50]

    def test_other_exception(self):
        with self.assertRaises(ZeroDivisionError):
            with suppress(TypeError):
                1/0

    def test_no_args(self):
        with self.assertRaises(ZeroDivisionError):
            with suppress():
                1/0

    def test_multiple_exception_args(self):
        with suppress(ZeroDivisionError, TypeError):
            1/0
        with suppress(ZeroDivisionError, TypeError):
            len(5)

    def test_cm_is_reentrant(self):
        ignore_exceptions = suppress(Exception)
        with ignore_exceptions:
            pass
        with ignore_exceptions:
            len(5)
        with ignore_exceptions:
            with ignore_exceptions: # Check nested usage
                len(5)
            outer_continued = True
            1/0
        self.assertTrue(outer_continued)

    def test_exception_groups(self):
        eg_ve = lambda: ExceptionGroup(
            "EG with ValueErrors only",
            [ValueError("ve1"), ValueError("ve2"), ValueError("ve3")],
        )
        eg_all = lambda: ExceptionGroup(
            "EG with many types of exceptions",
            [ValueError("ve1"), KeyError("ke1"), ValueError("ve2"), KeyError("ke2")],
        )
        with suppress(ValueError):
            raise eg_ve()
        with suppress(ValueError, KeyError):
            raise eg_all()
        with self.assertRaises(ExceptionGroup) as eg1:
            with suppress(ValueError):
                raise eg_all()
        self.assertExceptionIsLike(
            eg1.exception,
            ExceptionGroup(
                "EG with many types of exceptions",
                [KeyError("ke1"), KeyError("ke2")],
            ),
        )
        # Check handling of BaseExceptionGroup, using GeneratorExit so that
        # we don't accidentally discard a ctrl-c with KeyboardInterrupt.
        with suppress(GeneratorExit):
            raise BaseExceptionGroup("message", [GeneratorExit()])
        # If we raise a BaseException group, we can still suppress parts
        with self.assertRaises(BaseExceptionGroup) as eg1:
            with suppress(KeyError):
                raise BaseExceptionGroup("message", [GeneratorExit("g"), KeyError("k")])
        self.assertExceptionIsLike(
            eg1.exception, BaseExceptionGroup("message", [GeneratorExit("g")]),
        )
        # If we suppress all the leaf BaseExceptions, we get a non-base ExceptionGroup
        with self.assertRaises(ExceptionGroup) as eg1:
            with suppress(GeneratorExit):
                raise BaseExceptionGroup("message", [GeneratorExit("g"), KeyError("k")])
        self.assertExceptionIsLike(
            eg1.exception, ExceptionGroup("message", [KeyError("k")]),
        )


class TestChdir(__TestCase):
    def make_relative_path(self, *parts):
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            *parts,
        )

    def test_simple(self):
        old_cwd = os.getcwd()
        target = self.make_relative_path('data')
        self.assertNotEqual(old_cwd, target)

        with chdir(target):
            self.assertEqual(os.getcwd(), target)
        self.assertEqual(os.getcwd(), old_cwd)

    @unittest.skip("Missing archivetestdata")
    def test_reentrant(self):
        old_cwd = os.getcwd()
        target1 = self.make_relative_path('data')
        target2 = self.make_relative_path('archivetestdata')
        self.assertNotIn(old_cwd, (target1, target2))
        chdir1, chdir2 = chdir(target1), chdir(target2)

        with chdir1:
            self.assertEqual(os.getcwd(), target1)
            with chdir2:
                self.assertEqual(os.getcwd(), target2)
                with chdir1:
                    self.assertEqual(os.getcwd(), target1)
                self.assertEqual(os.getcwd(), target2)
            self.assertEqual(os.getcwd(), target1)
        self.assertEqual(os.getcwd(), old_cwd)

    def test_exception(self):
        old_cwd = os.getcwd()
        target = self.make_relative_path('data')
        self.assertNotEqual(old_cwd, target)

        try:
            with chdir(target):
                self.assertEqual(os.getcwd(), target)
                raise RuntimeError("boom")
        except RuntimeError as re:
            self.assertEqual(str(re), "boom")
        self.assertEqual(os.getcwd(), old_cwd)


if __name__ == "__main__":
    run_tests()
