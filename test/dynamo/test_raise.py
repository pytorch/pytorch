# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

import sys
import types
import unittest

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn
import torch.utils.checkpoint
from torch.testing._internal.common_utils import make_dynamo_test


def get_tb():
    try:
        raise OSError()
    except:
        return sys.exc_info()[2]


class Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return True


class MyException(Exception):
    def __init__(self):
        raise RuntimeError()


class ContextManager:
    def __enter__(self):
        pass

    def __exit__(self, t, v, tb):
        raise NameError


class TestRaise(torch._dynamo.test_case.CPythonTestCase):
    # Tests taken from CPython source code in cpython/Lib/test/test_raise.py
    # https://github.com/python/cpython/blob/v3.13.1/Lib/test/test_raise.py
    @make_dynamo_test
    def test_invalid_reraise(self):
        try:
            raise
        except RuntimeError as e:
            self.assertIn("No active exception", str(e))
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_reraise(self):
        try:
            try:
                raise IndexError
            except IndexError as e:
                exc1 = e
                raise
        except IndexError as exc2:
            self.assertIs(exc1, exc2)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_except_reraise(self):
        def reraise():
            try:
                raise TypeError("foo")
            except:
                try:
                    raise KeyError("caught")
                except KeyError:
                    pass
                raise

        self.assertRaises(TypeError, reraise)

    @make_dynamo_test
    def test_finally_reraise(self):
        def reraise():
            try:
                raise TypeError("foo")
            except:
                try:
                    raise KeyError("caught")
                finally:
                    raise

        self.assertRaises(KeyError, reraise)

    @make_dynamo_test
    def test_nested_reraise(self):
        def nested_reraise():
            raise

        def reraise():
            try:
                raise TypeError("foo")
            except:
                nested_reraise()

        self.assertRaises(TypeError, reraise)

    @make_dynamo_test
    def test_raise_from_None(self):
        try:
            try:
                raise TypeError("foo")
            except:
                raise ValueError() from None
        except ValueError as e:
            self.assertIsInstance(e.__context__, TypeError)
            self.assertIsNone(e.__cause__)

    @make_dynamo_test
    def test_with_reraise1(self):
        def reraise():
            try:
                raise TypeError("foo")
            except:
                with Context():
                    pass
                raise

        self.assertRaises(TypeError, reraise)

    @make_dynamo_test
    def test_with_reraise2(self):
        def reraise():
            try:
                raise TypeError("foo")
            except:
                with Context():
                    raise KeyError("caught")
                raise

        self.assertRaises(TypeError, reraise)

    @make_dynamo_test
    def test_yield_reraise(self):
        def reraise():
            try:
                raise TypeError("foo")
            except:
                yield 1
                raise

        g = reraise()
        next(g)
        self.assertRaises(TypeError, lambda: next(g))
        self.assertRaises(StopIteration, lambda: next(g))

    @make_dynamo_test
    def test_erroneous_exception(self):
        try:
            raise MyException
        except RuntimeError:
            pass
        else:
            self.fail("No exception raised")

    @unittest.expectedFailure  # object
    @make_dynamo_test
    def test_new_returns_invalid_instance(self):
        # See issue #11627.
        class MyException2(Exception):
            def __new__(cls, *args):
                return object()

        with self.assertRaises(TypeError):
            raise MyException2

    @unittest.expectedFailure  # Assertion with non-string message
    @make_dynamo_test
    def test_assert_with_tuple_arg(self):
        try:
            assert False, (3,)
        except AssertionError as e:
            self.assertEqual(str(e), "(3,)")


class TestCause(torch._dynamo.test_case.TestCase):
    # Tests taken from CPython source code in cpython/Lib/test/test_raise.py
    # https://github.com/python/cpython/blob/v3.13.1/Lib/test/test_raise.py
    def setUp(self):
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self._prev

    @make_dynamo_test
    def testCauseSyntax(self):
        try:
            try:
                try:
                    raise TypeError
                except Exception:
                    raise ValueError from None
            except ValueError as exc:
                self.assertIsNone(exc.__cause__)
                self.assertTrue(exc.__suppress_context__)
                exc.__suppress_context__ = False
                raise exc
        except ValueError as exc:
            e = exc

        self.assertIsNone(e.__cause__)
        self.assertFalse(e.__suppress_context__)
        self.assertIsInstance(e.__context__, TypeError)

    @make_dynamo_test
    def test_invalid_cause(self):
        try:
            raise IndexError from 5
        except TypeError as e:
            self.assertIn("exception cause", str(e))
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_class_cause(self):
        try:
            raise IndexError from KeyError
        except IndexError as e:
            self.assertIsInstance(e.__cause__, KeyError)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_instance_cause(self):
        cause = KeyError()
        try:
            raise IndexError from cause
        except IndexError as e:
            self.assertIs(e.__cause__, cause)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_erroneous_cause(self):
        try:
            raise IndexError from MyException
        except RuntimeError:
            pass
        else:
            self.fail("No exception raised")


class TestTraceback(torch._dynamo.test_case.TestCase):
    # Tests taken from CPython source code in cpython/Lib/test/test_raise.py
    # https://github.com/python/cpython/blob/v3.13.1/Lib/test/test_raise.py
    def setUp(self):
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self._prev

    @unittest.expectedFailure  # Dynamo doesn't track traceback
    @make_dynamo_test
    def test_sets_traceback(self):
        try:
            raise IndexError()
        except IndexError as e:
            self.assertIsInstance(e.__traceback__, types.TracebackType)
        else:
            self.fail("No exception raised")

    @unittest.expectedFailure  # Dynamo doesn't track traceback
    @make_dynamo_test
    def test_accepts_traceback(self):
        tb = get_tb()
        try:
            raise IndexError().with_traceback(tb)
        except IndexError as e:
            self.assertNotEqual(e.__traceback__, tb)
            self.assertEqual(e.__traceback__.tb_next, tb)
        else:
            self.fail("No exception raised")


class TestTracebackType(torch._dynamo.test_case.TestCase):
    # Tests taken from CPython source code in cpython/Lib/test/test_raise.py
    # https://github.com/python/cpython/blob/v3.13.1/Lib/test/test_raise.py
    def setUp(self):
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self._prev

    def raiser(self):
        raise ValueError

    @unittest.expectedFailure  # Dynamo doesn't track traceback
    @make_dynamo_test
    def test_attrs(self):
        try:
            self.raiser()
        except Exception as exc:
            tb = exc.__traceback__

        self.assertIsInstance(tb.tb_next, types.TracebackType)
        self.assertIs(tb.tb_frame, sys._getframe())
        self.assertIsInstance(tb.tb_lasti, int)
        self.assertIsInstance(tb.tb_lineno, int)

        self.assertIs(tb.tb_next.tb_next, None)

        # Invalid assignments
        with self.assertRaises(TypeError):
            del tb.tb_next

        with self.assertRaises(TypeError):
            tb.tb_next = "asdf"

        # Loops
        with self.assertRaises(ValueError):
            tb.tb_next = tb

        with self.assertRaises(ValueError):
            tb.tb_next.tb_next = tb

        # Valid assignments
        tb.tb_next = None
        self.assertIs(tb.tb_next, None)

        new_tb = get_tb()
        tb.tb_next = new_tb
        self.assertIs(tb.tb_next, new_tb)

    @unittest.expectedFailure  # Dynamo doesn't track traceback
    @make_dynamo_test
    def test_constructor(self):
        other_tb = get_tb()
        frame = sys._getframe()

        tb = types.TracebackType(other_tb, frame, 1, 2)
        self.assertEqual(tb.tb_next, other_tb)
        self.assertEqual(tb.tb_frame, frame)
        self.assertEqual(tb.tb_lasti, 1)
        self.assertEqual(tb.tb_lineno, 2)

        tb = types.TracebackType(None, frame, 1, 2)
        self.assertEqual(tb.tb_next, None)

        with self.assertRaises(TypeError):
            types.TracebackType("no", frame, 1, 2)

        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, "no", 1, 2)

        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, frame, "no", 2)

        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, frame, 1, "nuh-uh")


class TestContext(torch._dynamo.test_case.TestCase):
    # Tests taken from CPython source code in cpython/Lib/test/test_raise.py
    # https://github.com/python/cpython/blob/v3.13.1/Lib/test/test_raise.py
    def setUp(self):
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self._prev

    @unittest.expectedFailure  # missing Exception.__eq__
    @make_dynamo_test
    def test_instance_context_instance_raise(self):
        context = IndexError()
        try:
            try:
                raise context
            except:
                raise OSError()
        except OSError as e:
            self.assertEqual(e.__context__, context)
        else:
            self.fail("No exception raised")

    @unittest.expectedFailure  # missing Exception.__eq__ and Exception.__repr__
    @make_dynamo_test
    def test_class_context_instance_raise(self):
        context = IndexError
        try:
            try:
                raise context
            except:
                raise OSError()
        except OSError as e:
            self.assertNotEqual(e.__context__, context)
            self.assertIsInstance(e.__context__, context)
        else:
            self.fail("No exception raised")

    @unittest.expectedFailure  # missing Exception.__eq__ and Exception.__repr__
    @make_dynamo_test
    def test_class_context_class_raise(self):
        context = IndexError
        try:
            try:
                raise context
            except:
                raise OSError
        except OSError as e:
            self.assertNotEqual(e.__context__, context)
            self.assertIsInstance(e.__context__, context)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_c_exception_context(self):
        try:
            try:
                raise ZeroDivisionError
            except:
                raise OSError
        except OSError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_c_exception_raise(self):
        try:
            try:
                raise ZeroDivisionError
            except:
                raise NameError
        except NameError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_noraise_finally(self):
        try:
            try:
                pass
            finally:
                raise OSError
        except OSError as e:
            self.assertIsNone(e.__context__)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_raise_finally(self):
        try:
            try:
                raise ZeroDivisionError
            finally:
                raise OSError
        except OSError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_context_manager(self):
        try:
            with ContextManager():
                raise ZeroDivisionError
        except NameError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail("No exception raised")

    @make_dynamo_test
    def test_cycle_broken(self):
        # Self-cycles (when re-raising a caught exception) are broken
        try:
            try:
                raise ZeroDivisionError
            except ZeroDivisionError as e:
                raise e
        except ZeroDivisionError as e:
            self.assertIsNone(e.__context__)

    @make_dynamo_test
    def test_reraise_cycle_broken(self):
        # Non-trivial context cycles (through re-raising a previous exception)
        # are broken too.
        try:
            try:
                raise NameError
            except NameError as a:
                try:
                    raise ZeroDivisionError
                except ZeroDivisionError:
                    raise a
        except NameError as e:
            self.assertIsNone(e.__context__.__context__)

    @make_dynamo_test
    def test_3118(self):
        # deleting the generator caused the __context__ to be cleared
        def gen():
            try:
                yield 1
            finally:
                pass

        def f():
            g = gen()
            next(g)
            try:
                try:
                    raise ValueError
                except:
                    del g
                    raise KeyError
            except Exception as e:
                self.assertIsInstance(e.__context__, ValueError)

        f()

    @unittest.expectedFailure  # too CPython specific(?)
    @make_dynamo_test
    def test_3611(self):
        # A re-raised exception in a __del__ caused the __context__
        # to be cleared
        class C:
            def __del__(self):
                try:
                    raise ZeroDivisionError
                except:
                    raise

        def f():
            x = C()
            try:
                try:
                    x.x
                except AttributeError:
                    del x
                    raise TypeError
            except Exception as e:
                self.assertNotEqual(e.__context__, None)
                self.assertIsInstance(e.__context__, AttributeError)

        with support.catch_unraisable_exception() as cm:
            f()

            self.assertEqual(ZeroDivisionError, cm.unraisable.exc_type)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
