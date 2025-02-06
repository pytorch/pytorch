# Owner(s): ["module: dynamo"]
import contextlib
import sys
import traceback
import unittest
from contextlib import contextmanager, ExitStack

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@contextlib.contextmanager
def set_default_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class TestExitStack(torch._dynamo.test_case.TestCase):
    def setUp(self):
        if sys.version_info < (3, 11):
            self.skipTest(
                "Tracing the unittest module needs exception table (Python 3.11+) to work"
            )
        self._old = torch._dynamo.config.enable_trace_contextlib
        torch._dynamo.config.enable_trace_contextlib = True

    def tearDown(self):
        torch._dynamo.config.enable_trace_contextlib = self._old

    def test_exitstack(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            with contextlib.ExitStack() as stack:
                stack.enter_context(set_default_dtype(torch.float64))
                return t.sin()

        t = torch.randn(2, dtype=torch.float64)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(y.dtype, torch.float64)


class CPythonTestBaseExitStack:
    exit_stack = None

    @make_dynamo_test
    def test_no_resources(self):
        with self.exit_stack():
            pass

    @unittest.expectedFailure
    @make_dynamo_test
    def test_callback(self):
        expected = [
            ((), {}),
            ((1,), {}),
            ((1, 2), {}),
            ((), dict(example=1)),
            ((1,), dict(example=1)),
            ((1, 2), dict(example=1)),
            ((1, 2), dict(self=3, callback=4)),
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

    @unittest.expectedFailure
    @make_dynamo_test
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

        class ExitCM:
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
            1 / 0

    @unittest.expectedFailure
    @make_dynamo_test
    def test_enter_context(self):
        class TestCM:
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

    @unittest.expectedFailure
    @unittest.skipIf(sys.version_info < (3, 11), "Python 3.11+")
    @make_dynamo_test
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
            with self.assertRaisesRegex(TypeError, "the context manager"):
                stack.enter_context(LacksEnterAndExit())
            with self.assertRaisesRegex(TypeError, "the context manager"):
                stack.enter_context(LacksEnter())
            with self.assertRaisesRegex(TypeError, "the context manager"):
                stack.enter_context(LacksExit())
            self.assertFalse(stack._exit_callbacks)

    @make_dynamo_test
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

    @unittest.expectedFailure
    @make_dynamo_test
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

    @make_dynamo_test
    def test_exit_raise(self):
        with self.assertRaises(ZeroDivisionError):
            with self.exit_stack() as stack:
                stack.push(lambda *exc: False)
                1 / 0

    @make_dynamo_test
    def test_exit_suppress(self):
        with self.exit_stack() as stack:
            stack.push(lambda *exc: True)
            1 / 0

    @unittest.expectedFailure
    @unittest.skipIf(sys.version_info < (3, 12), "Python 3.12+")
    @make_dynamo_test
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
                1 / 0
        except ValueError as e:
            exc = e

        self.assertIsInstance(exc, ValueError)
        ve_frames = traceback.extract_tb(exc.__traceback__)
        expected = (
            [("test_exit_exception_traceback", "with self.exit_stack() as stack:")]
            + self.callback_error_internal_frames
            + [("_exit_wrapper", "callback(*args, **kwds)"), ("raise_exc", "raise exc")]
        )

        self.assertEqual([(f.name, f.line) for f in ve_frames], expected)

        self.assertIsInstance(exc.__context__, ZeroDivisionError)
        zde_frames = traceback.extract_tb(exc.__context__.__traceback__)
        self.assertEqual(
            [(f.name, f.line) for f in zde_frames],
            [("test_exit_exception_traceback", "1/0")],
        )

    @unittest.expectedFailure
    @make_dynamo_test
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
                except Exception:
                    raise self.outer  # noqa: B904

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

    @make_dynamo_test
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

    @unittest.expectedFailure
    @make_dynamo_test
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
                        raise IndexError
                except MyException as exc:
                    self.assertIsNone(exc.__context__)
                else:
                    self.fail("Expected IndexError, but no exception was raised")

    @make_dynamo_test
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

    @make_dynamo_test
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
            self.assertIsNone(exc.__context__.__context__.__context__.__context__)

    @make_dynamo_test
    def test_exit_exception_with_existing_context(self):
        # Addresses a lack of test coverage discovered after checking in a
        # fix for issue 20317 that still contained debugging code.
        def raise_nested(inner_exc, outer_exc):
            try:
                raise inner_exc
            finally:
                raise outer_exc

        exc1 = AttributeError(1)
        exc2 = BytesWarning(2)
        exc3 = ConnectionError(3)
        exc4 = DeprecationWarning(4)
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
            self.assertIs(exc.__context__.__context__.__context__.__context__, exc1)
            self.assertIsNone(
                exc.__context__.__context__.__context__.__context__.__context__
            )

    @make_dynamo_test
    def test_body_exception_suppress(self):
        def suppress_exc(*exc_details):
            return True

        try:
            with self.exit_stack() as stack:
                stack.push(suppress_exc)
                1 / 0
        except IndexError:
            self.fail("Expected no exception, got IndexError")

    @make_dynamo_test
    def test_exit_exception_chaining_suppress(self):
        with self.exit_stack() as stack:
            stack.push(lambda *exc: True)
            stack.push(lambda *exc: 1 / 0)
            stack.push(lambda *exc: {}[1])

    @make_dynamo_test
    def test_excessive_nesting(self):
        # The original implementation would die with RecursionError here
        with self.exit_stack() as stack:
            # Original test uses 10.000 but that takes too long to finish
            for i in range(100):
                stack.callback(int)

    @unittest.expectedFailure
    @unittest.skipIf(sys.version_info < (3, 12), "Python 3.12+")
    @make_dynamo_test
    def test_instance_bypass(self):
        class Example:
            pass

        cm = Example()
        cm.__enter__ = object()
        cm.__exit__ = object()
        stack = self.exit_stack()
        with self.assertRaisesRegex(TypeError, "the context manager"):
            stack.enter_context(cm)
        stack.push(cm)
        self.assertIs(stack._exit_callbacks[-1][1], cm)

    @unittest.expectedFailure
    @make_dynamo_test
    def test_dont_reraise_RuntimeError(self):
        # https://bugs.python.org/issue27122
        class UniqueException(Exception):
            pass

        class UniqueRuntimeError(RuntimeError):
            pass

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
            except Exception as exc:  # noqa: TRY203
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


class CPythonTestExitStack(CPythonTestBaseExitStack, torch._dynamo.test_case.TestCase):
    # Tests taken from CPython source code in cpython/Lib/test/test_contextlib.py
    # https://github.com/python/cpython/blob/v3.13.1/Lib/test/test_contextlib.py
    exit_stack = ExitStack
    callback_error_internal_frames = [
        ("__exit__", "raise exc"),
        ("__exit__", "if cb(*exc_details):"),
    ]

    def setUp(self):
        if sys.version_info < (3, 11):
            self.skipTest(
                "Tracing the unittest module needs exception table (Python 3.11+) to work"
            )
        return super().setUp()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
