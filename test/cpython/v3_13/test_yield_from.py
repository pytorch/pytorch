# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_yield_from.py

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

# -*- coding: utf-8 -*-

"""
Test suite for PEP 380 implementation

adapted from original tests written by Greg Ewing
see <http://www.cosc.canterbury.ac.nz/greg.ewing/python/yield-from/YieldFrom-Python3.1.2-rev5.zip>
"""

import unittest
import inspect

from test.support import captured_stderr, disable_gc, gc_collect
from test import support

class TestPEP380Operation(CPythonTestCase):
    """
    Test semantics.
    """

    def test_delegation_of_initial_next_to_subgenerator(self):
        """
        Test delegation of initial next() call to subgenerator
        """
        trace = []
        def g1():
            trace.append("Starting g1")
            yield from g2()
            trace.append("Finishing g1")
        def g2():
            trace.append("Starting g2")
            yield 42
            trace.append("Finishing g2")
        for x in g1():
            trace.append("Yielded %s" % (x,))
        self.assertEqual(trace,[
            "Starting g1",
            "Starting g2",
            "Yielded 42",
            "Finishing g2",
            "Finishing g1",
        ])

    def test_raising_exception_in_initial_next_call(self):
        """
        Test raising exception in initial next() call
        """
        trace = []
        def g1():
            try:
                trace.append("Starting g1")
                yield from g2()
            finally:
                trace.append("Finishing g1")
        def g2():
            try:
                trace.append("Starting g2")
                raise ValueError("spanish inquisition occurred")
            finally:
                trace.append("Finishing g2")
        try:
            for x in g1():
                trace.append("Yielded %s" % (x,))
        except ValueError as e:
            self.assertEqual(e.args[0], "spanish inquisition occurred")
        else:
            self.fail("subgenerator failed to raise ValueError")
        self.assertEqual(trace,[
            "Starting g1",
            "Starting g2",
            "Finishing g2",
            "Finishing g1",
        ])

    def test_delegation_of_next_call_to_subgenerator(self):
        """
        Test delegation of next() call to subgenerator
        """
        trace = []
        def g1():
            trace.append("Starting g1")
            yield "g1 ham"
            yield from g2()
            yield "g1 eggs"
            trace.append("Finishing g1")
        def g2():
            trace.append("Starting g2")
            yield "g2 spam"
            yield "g2 more spam"
            trace.append("Finishing g2")
        for x in g1():
            trace.append("Yielded %s" % (x,))
        self.assertEqual(trace,[
            "Starting g1",
            "Yielded g1 ham",
            "Starting g2",
            "Yielded g2 spam",
            "Yielded g2 more spam",
            "Finishing g2",
            "Yielded g1 eggs",
            "Finishing g1",
        ])

    def test_raising_exception_in_delegated_next_call(self):
        """
        Test raising exception in delegated next() call
        """
        trace = []
        def g1():
            try:
                trace.append("Starting g1")
                yield "g1 ham"
                yield from g2()
                yield "g1 eggs"
            finally:
                trace.append("Finishing g1")
        def g2():
            try:
                trace.append("Starting g2")
                yield "g2 spam"
                raise ValueError("hovercraft is full of eels")
                yield "g2 more spam"
            finally:
                trace.append("Finishing g2")
        try:
            for x in g1():
                trace.append("Yielded %s" % (x,))
        except ValueError as e:
            self.assertEqual(e.args[0], "hovercraft is full of eels")
        else:
            self.fail("subgenerator failed to raise ValueError")
        self.assertEqual(trace,[
            "Starting g1",
            "Yielded g1 ham",
            "Starting g2",
            "Yielded g2 spam",
            "Finishing g2",
            "Finishing g1",
        ])

    def test_delegation_of_send(self):
        """
        Test delegation of send()
        """
        trace = []
        def g1():
            trace.append("Starting g1")
            x = yield "g1 ham"
            trace.append("g1 received %s" % (x,))
            yield from g2()
            x = yield "g1 eggs"
            trace.append("g1 received %s" % (x,))
            trace.append("Finishing g1")
        def g2():
            trace.append("Starting g2")
            x = yield "g2 spam"
            trace.append("g2 received %s" % (x,))
            x = yield "g2 more spam"
            trace.append("g2 received %s" % (x,))
            trace.append("Finishing g2")
        g = g1()
        y = next(g)
        x = 1
        try:
            while 1:
                y = g.send(x)
                trace.append("Yielded %s" % (y,))
                x += 1
        except StopIteration:
            pass
        self.assertEqual(trace,[
            "Starting g1",
            "g1 received 1",
            "Starting g2",
            "Yielded g2 spam",
            "g2 received 2",
            "Yielded g2 more spam",
            "g2 received 3",
            "Finishing g2",
            "Yielded g1 eggs",
            "g1 received 4",
            "Finishing g1",
        ])

    def test_handling_exception_while_delegating_send(self):
        """
        Test handling exception while delegating 'send'
        """
        trace = []
        def g1():
            trace.append("Starting g1")
            x = yield "g1 ham"
            trace.append("g1 received %s" % (x,))
            yield from g2()
            x = yield "g1 eggs"
            trace.append("g1 received %s" % (x,))
            trace.append("Finishing g1")
        def g2():
            trace.append("Starting g2")
            x = yield "g2 spam"
            trace.append("g2 received %s" % (x,))
            raise ValueError("hovercraft is full of eels")
            x = yield "g2 more spam"
            trace.append("g2 received %s" % (x,))
            trace.append("Finishing g2")
        def run():
            g = g1()
            y = next(g)
            x = 1
            try:
                while 1:
                    y = g.send(x)
                    trace.append("Yielded %s" % (y,))
                    x += 1
            except StopIteration:
                trace.append("StopIteration")
        self.assertRaises(ValueError,run)
        self.assertEqual(trace,[
            "Starting g1",
            "g1 received 1",
            "Starting g2",
            "Yielded g2 spam",
            "g2 received 2",
        ])

    def test_delegating_close(self):
        """
        Test delegating 'close'
        """
        trace = []
        def g1():
            try:
                trace.append("Starting g1")
                yield "g1 ham"
                yield from g2()
                yield "g1 eggs"
            finally:
                trace.append("Finishing g1")
        def g2():
            try:
                trace.append("Starting g2")
                yield "g2 spam"
                yield "g2 more spam"
            finally:
                trace.append("Finishing g2")
        g = g1()
        for i in range(2):
            x = next(g)
            trace.append("Yielded %s" % (x,))
        g.close()
        self.assertEqual(trace,[
            "Starting g1",
            "Yielded g1 ham",
            "Starting g2",
            "Yielded g2 spam",
            "Finishing g2",
            "Finishing g1"
        ])

    def test_handing_exception_while_delegating_close(self):
        """
        Test handling exception while delegating 'close'
        """
        trace = []
        def g1():
            try:
                trace.append("Starting g1")
                yield "g1 ham"
                yield from g2()
                yield "g1 eggs"
            finally:
                trace.append("Finishing g1")
        def g2():
            try:
                trace.append("Starting g2")
                yield "g2 spam"
                yield "g2 more spam"
            finally:
                trace.append("Finishing g2")
                raise ValueError("nybbles have exploded with delight")
        try:
            g = g1()
            for i in range(2):
                x = next(g)
                trace.append("Yielded %s" % (x,))
            g.close()
        except ValueError as e:
            self.assertEqual(e.args[0], "nybbles have exploded with delight")
            self.assertIsInstance(e.__context__, GeneratorExit)
        else:
            self.fail("subgenerator failed to raise ValueError")
        self.assertEqual(trace,[
            "Starting g1",
            "Yielded g1 ham",
            "Starting g2",
            "Yielded g2 spam",
            "Finishing g2",
            "Finishing g1",
        ])

    def test_delegating_throw(self):
        """
        Test delegating 'throw'
        """
        trace = []
        def g1():
            try:
                trace.append("Starting g1")
                yield "g1 ham"
                yield from g2()
                yield "g1 eggs"
            finally:
                trace.append("Finishing g1")
        def g2():
            try:
                trace.append("Starting g2")
                yield "g2 spam"
                yield "g2 more spam"
            finally:
                trace.append("Finishing g2")
        try:
            g = g1()
            for i in range(2):
                x = next(g)
                trace.append("Yielded %s" % (x,))
            e = ValueError("tomato ejected")
            g.throw(e)
        except ValueError as e:
            self.assertEqual(e.args[0], "tomato ejected")
        else:
            self.fail("subgenerator failed to raise ValueError")
        self.assertEqual(trace,[
            "Starting g1",
            "Yielded g1 ham",
            "Starting g2",
            "Yielded g2 spam",
            "Finishing g2",
            "Finishing g1",
        ])

    def test_value_attribute_of_StopIteration_exception(self):
        """
        Test 'value' attribute of StopIteration exception
        """
        trace = []
        def pex(e):
            trace.append("%s: %s" % (e.__class__.__name__, e))
            trace.append("value = %s" % (e.value,))
        e = StopIteration()
        pex(e)
        e = StopIteration("spam")
        pex(e)
        e.value = "eggs"
        pex(e)
        self.assertEqual(trace,[
            "StopIteration: ",
            "value = None",
            "StopIteration: spam",
            "value = spam",
            "StopIteration: spam",
            "value = eggs",
        ])


    def test_exception_value_crash(self):
        # There used to be a refcount error when the return value
        # stored in the StopIteration has a refcount of 1.
        def g1():
            yield from g2()
        def g2():
            yield "g2"
            return [42]
        self.assertEqual(list(g1()), ["g2"])


    def test_generator_return_value(self):
        """
        Test generator return value
        """
        trace = []
        def g1():
            trace.append("Starting g1")
            yield "g1 ham"
            ret = yield from g2()
            trace.append("g2 returned %r" % (ret,))
            for v in 1, (2,), StopIteration(3):
                ret = yield from g2(v)
                trace.append("g2 returned %r" % (ret,))
            yield "g1 eggs"
            trace.append("Finishing g1")
        def g2(v = None):
            trace.append("Starting g2")
            yield "g2 spam"
            yield "g2 more spam"
            trace.append("Finishing g2")
            if v:
                return v
        for x in g1():
            trace.append("Yielded %s" % (x,))
        self.assertEqual(trace,[
            "Starting g1",
            "Yielded g1 ham",
            "Starting g2",
            "Yielded g2 spam",
            "Yielded g2 more spam",
            "Finishing g2",
            "g2 returned None",
            "Starting g2",
            "Yielded g2 spam",
            "Yielded g2 more spam",
            "Finishing g2",
            "g2 returned 1",
            "Starting g2",
            "Yielded g2 spam",
            "Yielded g2 more spam",
            "Finishing g2",
            "g2 returned (2,)",
            "Starting g2",
            "Yielded g2 spam",
            "Yielded g2 more spam",
            "Finishing g2",
            "g2 returned StopIteration(3)",
            "Yielded g1 eggs",
            "Finishing g1",
        ])

    def test_delegation_of_next_to_non_generator(self):
        """
        Test delegation of next() to non-generator
        """
        trace = []
        def g():
            yield from range(3)
        for x in g():
            trace.append("Yielded %s" % (x,))
        self.assertEqual(trace,[
            "Yielded 0",
            "Yielded 1",
            "Yielded 2",
        ])


    def test_conversion_of_sendNone_to_next(self):
        """
        Test conversion of send(None) to next()
        """
        trace = []
        def g():
            yield from range(3)
        gi = g()
        for x in range(3):
            y = gi.send(None)
            trace.append("Yielded: %s" % (y,))
        self.assertEqual(trace,[
            "Yielded: 0",
            "Yielded: 1",
            "Yielded: 2",
        ])

    def test_delegation_of_close_to_non_generator(self):
        """
        Test delegation of close() to non-generator
        """
        trace = []
        def g():
            try:
                trace.append("starting g")
                yield from range(3)
                trace.append("g should not be here")
            finally:
                trace.append("finishing g")
        gi = g()
        next(gi)
        with captured_stderr() as output:
            gi.close()
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(trace,[
            "starting g",
            "finishing g",
        ])

    def test_delegating_throw_to_non_generator(self):
        """
        Test delegating 'throw' to non-generator
        """
        trace = []
        def g():
            try:
                trace.append("Starting g")
                yield from range(10)
            finally:
                trace.append("Finishing g")
        try:
            gi = g()
            for i in range(5):
                x = next(gi)
                trace.append("Yielded %s" % (x,))
            e = ValueError("tomato ejected")
            gi.throw(e)
        except ValueError as e:
            self.assertEqual(e.args[0],"tomato ejected")
        else:
            self.fail("subgenerator failed to raise ValueError")
        self.assertEqual(trace,[
            "Starting g",
            "Yielded 0",
            "Yielded 1",
            "Yielded 2",
            "Yielded 3",
            "Yielded 4",
            "Finishing g",
        ])

    def test_attempting_to_send_to_non_generator(self):
        """
        Test attempting to send to non-generator
        """
        trace = []
        def g():
            try:
                trace.append("starting g")
                yield from range(3)
                trace.append("g should not be here")
            finally:
                trace.append("finishing g")
        try:
            gi = g()
            next(gi)
            for x in range(3):
                y = gi.send(42)
                trace.append("Should not have yielded: %s" % (y,))
        except AttributeError as e:
            self.assertIn("send", e.args[0])
        else:
            self.fail("was able to send into non-generator")
        self.assertEqual(trace,[
            "starting g",
            "finishing g",
        ])

    def test_broken_getattr_handling(self):
        """
        Test subiterator with a broken getattr implementation
        """
        class Broken:
            def __iter__(self):
                return self
            def __next__(self):
                return 1
            def __getattr__(self, attr):
                1/0

        def g():
            yield from Broken()

        with self.assertRaises(ZeroDivisionError):
            gi = g()
            self.assertEqual(next(gi), 1)
            gi.send(1)

        with self.assertRaises(ZeroDivisionError):
            gi = g()
            self.assertEqual(next(gi), 1)
            gi.throw(AttributeError)

        with support.catch_unraisable_exception() as cm:
            gi = g()
            self.assertEqual(next(gi), 1)
            gi.close()

            self.assertEqual(ZeroDivisionError, cm.unraisable.exc_type)

    def test_exception_in_initial_next_call(self):
        """
        Test exception in initial next() call
        """
        trace = []
        def g1():
            trace.append("g1 about to yield from g2")
            yield from g2()
            trace.append("g1 should not be here")
        def g2():
            yield 1/0
        def run():
            gi = g1()
            next(gi)
        self.assertRaises(ZeroDivisionError,run)
        self.assertEqual(trace,[
            "g1 about to yield from g2"
        ])

    def test_attempted_yield_from_loop(self):
        """
        Test attempted yield-from loop
        """
        trace = []
        def g1():
            trace.append("g1: starting")
            yield "y1"
            trace.append("g1: about to yield from g2")
            yield from g2()
            trace.append("g1 should not be here")

        def g2():
            trace.append("g2: starting")
            yield "y2"
            trace.append("g2: about to yield from g1")
            yield from gi
            trace.append("g2 should not be here")
        try:
            gi = g1()
            for y in gi:
                trace.append("Yielded: %s" % (y,))
        except ValueError as e:
            self.assertEqual(e.args[0],"generator already executing")
        else:
            self.fail("subgenerator didn't raise ValueError")
        self.assertEqual(trace,[
            "g1: starting",
            "Yielded: y1",
            "g1: about to yield from g2",
            "g2: starting",
            "Yielded: y2",
            "g2: about to yield from g1",
        ])

    def test_returning_value_from_delegated_throw(self):
        """
        Test returning value from delegated 'throw'
        """
        trace = []
        def g1():
            try:
                trace.append("Starting g1")
                yield "g1 ham"
                yield from g2()
                yield "g1 eggs"
            finally:
                trace.append("Finishing g1")
        def g2():
            try:
                trace.append("Starting g2")
                yield "g2 spam"
                yield "g2 more spam"
            except LunchError:
                trace.append("Caught LunchError in g2")
                yield "g2 lunch saved"
                yield "g2 yet more spam"
        class LunchError(Exception):
            pass
        g = g1()
        for i in range(2):
            x = next(g)
            trace.append("Yielded %s" % (x,))
        e = LunchError("tomato ejected")
        g.throw(e)
        for x in g:
            trace.append("Yielded %s" % (x,))
        self.assertEqual(trace,[
            "Starting g1",
            "Yielded g1 ham",
            "Starting g2",
            "Yielded g2 spam",
            "Caught LunchError in g2",
            "Yielded g2 yet more spam",
            "Yielded g1 eggs",
            "Finishing g1",
        ])

    def test_next_and_return_with_value(self):
        """
        Test next and return with value
        """
        trace = []
        def f(r):
            gi = g(r)
            next(gi)
            try:
                trace.append("f resuming g")
                next(gi)
                trace.append("f SHOULD NOT BE HERE")
            except StopIteration as e:
                trace.append("f caught %r" % (e,))
        def g(r):
            trace.append("g starting")
            yield
            trace.append("g returning %r" % (r,))
            return r
        f(None)
        f(1)
        f((2,))
        f(StopIteration(3))
        self.assertEqual(trace,[
            "g starting",
            "f resuming g",
            "g returning None",
            "f caught StopIteration()",
            "g starting",
            "f resuming g",
            "g returning 1",
            "f caught StopIteration(1)",
            "g starting",
            "f resuming g",
            "g returning (2,)",
            "f caught StopIteration((2,))",
            "g starting",
            "f resuming g",
            "g returning StopIteration(3)",
            "f caught StopIteration(StopIteration(3))",
        ])

    def test_send_and_return_with_value(self):
        """
        Test send and return with value
        """
        trace = []
        def f(r):
            gi = g(r)
            next(gi)
            try:
                trace.append("f sending spam to g")
                gi.send("spam")
                trace.append("f SHOULD NOT BE HERE")
            except StopIteration as e:
                trace.append("f caught %r" % (e,))
        def g(r):
            trace.append("g starting")
            x = yield
            trace.append("g received %r" % (x,))
            trace.append("g returning %r" % (r,))
            return r
        f(None)
        f(1)
        f((2,))
        f(StopIteration(3))
        self.assertEqual(trace, [
            "g starting",
            "f sending spam to g",
            "g received 'spam'",
            "g returning None",
            "f caught StopIteration()",
            "g starting",
            "f sending spam to g",
            "g received 'spam'",
            "g returning 1",
            'f caught StopIteration(1)',
            'g starting',
            'f sending spam to g',
            "g received 'spam'",
            'g returning (2,)',
            'f caught StopIteration((2,))',
            'g starting',
            'f sending spam to g',
            "g received 'spam'",
            'g returning StopIteration(3)',
            'f caught StopIteration(StopIteration(3))'
        ])

    def test_catching_exception_from_subgen_and_returning(self):
        """
        Test catching an exception thrown into a
        subgenerator and returning a value
        """
        def inner():
            try:
                yield 1
            except ValueError:
                trace.append("inner caught ValueError")
            return value

        def outer():
            v = yield from inner()
            trace.append("inner returned %r to outer" % (v,))
            yield v

        for value in 2, (2,), StopIteration(2):
            trace = []
            g = outer()
            trace.append(next(g))
            trace.append(repr(g.throw(ValueError)))
            self.assertEqual(trace, [
                1,
                "inner caught ValueError",
                "inner returned %r to outer" % (value,),
                repr(value),
            ])

    def test_throwing_GeneratorExit_into_subgen_that_returns(self):
        """
        Test throwing GeneratorExit into a subgenerator that
        catches it and returns normally.
        """
        trace = []
        def f():
            try:
                trace.append("Enter f")
                yield
                trace.append("Exit f")
            except GeneratorExit:
                return
        def g():
            trace.append("Enter g")
            yield from f()
            trace.append("Exit g")
        try:
            gi = g()
            next(gi)
            gi.throw(GeneratorExit)
        except GeneratorExit:
            pass
        else:
            self.fail("subgenerator failed to raise GeneratorExit")
        self.assertEqual(trace,[
            "Enter g",
            "Enter f",
        ])

    def test_throwing_GeneratorExit_into_subgenerator_that_yields(self):
        """
        Test throwing GeneratorExit into a subgenerator that
        catches it and yields.
        """
        trace = []
        def f():
            try:
                trace.append("Enter f")
                yield
                trace.append("Exit f")
            except GeneratorExit:
                yield
        def g():
            trace.append("Enter g")
            yield from f()
            trace.append("Exit g")
        try:
            gi = g()
            next(gi)
            gi.throw(GeneratorExit)
        except RuntimeError as e:
            self.assertEqual(e.args[0], "generator ignored GeneratorExit")
        else:
            self.fail("subgenerator failed to raise GeneratorExit")
        self.assertEqual(trace,[
            "Enter g",
            "Enter f",
        ])

    def test_throwing_GeneratorExit_into_subgen_that_raises(self):
        """
        Test throwing GeneratorExit into a subgenerator that
        catches it and raises a different exception.
        """
        trace = []
        def f():
            try:
                trace.append("Enter f")
                yield
                trace.append("Exit f")
            except GeneratorExit:
                raise ValueError("Vorpal bunny encountered")
        def g():
            trace.append("Enter g")
            yield from f()
            trace.append("Exit g")
        try:
            gi = g()
            next(gi)
            gi.throw(GeneratorExit)
        except ValueError as e:
            self.assertEqual(e.args[0], "Vorpal bunny encountered")
            self.assertIsInstance(e.__context__, GeneratorExit)
        else:
            self.fail("subgenerator failed to raise ValueError")
        self.assertEqual(trace,[
            "Enter g",
            "Enter f",
        ])

    def test_yield_from_empty(self):
        def g():
            yield from ()
        self.assertRaises(StopIteration, next, g())

    def test_delegating_generators_claim_to_be_running(self):
        # Check with basic iteration
        def one():
            yield 0
            yield from two()
            yield 3
        def two():
            yield 1
            try:
                yield from g1
            except ValueError:
                pass
            yield 2
        g1 = one()
        self.assertEqual(list(g1), [0, 1, 2, 3])

        # Check with send
        g1 = one()
        res = [next(g1)]
        try:
            while True:
                res.append(g1.send(42))
        except StopIteration:
            pass
        self.assertEqual(res, [0, 1, 2, 3])

    def test_delegating_generators_claim_to_be_running_with_throw(self):
        # Check with throw
        class MyErr(Exception):
            pass
        def one():
            try:
                yield 0
            except MyErr:
                pass
            yield from two()
            try:
                yield 3
            except MyErr:
                pass
        def two():
            try:
                yield 1
            except MyErr:
                pass
            try:
                yield from g1
            except ValueError:
                pass
            try:
                yield 2
            except MyErr:
                pass
        g1 = one()
        res = [next(g1)]
        try:
            while True:
                res.append(g1.throw(MyErr))
        except StopIteration:
            pass
        except:
            self.assertEqual(res, [0, 1, 2, 3])
            raise

    def test_delegating_generators_claim_to_be_running_with_close(self):
        # Check with close
        class MyIt:
            def __iter__(self):
                return self
            def __next__(self):
                return 42
            def close(self_):
                self.assertTrue(g1.gi_running)
                self.assertRaises(ValueError, next, g1)
        def one():
            yield from MyIt()
        g1 = one()
        next(g1)
        g1.close()

    def test_delegator_is_visible_to_debugger(self):
        def call_stack():
            return [f[3] for f in inspect.stack()]

        def gen():
            yield call_stack()
            yield call_stack()
            yield call_stack()

        def spam(g):
            yield from g

        def eggs(g):
            yield from g

        for stack in spam(gen()):
            self.assertTrue('spam' in stack)

        for stack in spam(eggs(gen())):
            self.assertTrue('spam' in stack and 'eggs' in stack)

    def test_custom_iterator_return(self):
        # See issue #15568
        class MyIter:
            def __iter__(self):
                return self
            def __next__(self):
                raise StopIteration(42)
        def gen():
            nonlocal ret
            ret = yield from MyIter()
        ret = None
        list(gen())
        self.assertEqual(ret, 42)

    def test_close_with_cleared_frame(self):
        # See issue #17669.
        #
        # Create a stack of generators: outer() delegating to inner()
        # delegating to innermost(). The key point is that the instance of
        # inner is created first: this ensures that its frame appears before
        # the instance of outer in the GC linked list.
        #
        # At the gc.collect call:
        #   - frame_clear is called on the inner_gen frame.
        #   - gen_dealloc is called on the outer_gen generator (the only
        #     reference is in the frame's locals).
        #   - gen_close is called on the outer_gen generator.
        #   - gen_close_iter is called to close the inner_gen generator, which
        #     in turn calls gen_close, and gen_yf.
        #
        # Previously, gen_yf would crash since inner_gen's frame had been
        # cleared (and in particular f_stacktop was NULL).

        def innermost():
            yield
        def inner():
            outer_gen = yield
            yield from innermost()
        def outer():
            inner_gen = yield
            yield from inner_gen

        with disable_gc():
            inner_gen = inner()
            outer_gen = outer()
            outer_gen.send(None)
            outer_gen.send(inner_gen)
            outer_gen.send(outer_gen)

            del outer_gen
            del inner_gen
            gc_collect()

    def test_send_tuple_with_custom_generator(self):
        # See issue #21209.
        class MyGen:
            def __iter__(self):
                return self
            def __next__(self):
                return 42
            def send(self, what):
                nonlocal v
                v = what
                return None
        def outer():
            v = yield from MyGen()
        g = outer()
        next(g)
        v = None
        g.send((1, 2, 3, 4))
        self.assertEqual(v, (1, 2, 3, 4))

class TestInterestingEdgeCases(CPythonTestCase):

    def assert_stop_iteration(self, iterator):
        with self.assertRaises(StopIteration) as caught:
            next(iterator)
        self.assertIsNone(caught.exception.value)
        self.assertIsNone(caught.exception.__context__)

    def assert_generator_raised_stop_iteration(self):
        return self.assertRaisesRegex(RuntimeError, r"^generator raised StopIteration$")

    def assert_generator_ignored_generator_exit(self):
        return self.assertRaisesRegex(RuntimeError, r"^generator ignored GeneratorExit$")

    def test_close_and_throw_work(self):

        yielded_first = object()
        yielded_second = object()
        returned = object()

        def inner():
            yield yielded_first
            yield yielded_second
            return returned

        def outer():
            return (yield from inner())

        with self.subTest("close"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            g.close()
            self.assert_stop_iteration(g)

        with self.subTest("throw GeneratorExit"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = GeneratorExit()
            with self.assertRaises(GeneratorExit) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, thrown)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw StopIteration"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = StopIteration()
            # PEP 479:
            with self.assert_generator_raised_stop_iteration() as caught:
                g.throw(thrown)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw BaseException"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = BaseException()
            with self.assertRaises(BaseException) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, thrown)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw Exception"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = Exception()
            with self.assertRaises(Exception) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, thrown)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

    def test_close_and_throw_raise_generator_exit(self):

        yielded_first = object()
        yielded_second = object()
        returned = object()

        def inner():
            try:
                yield yielded_first
                yield yielded_second
                return returned
            finally:
                raise raised

        def outer():
            return (yield from inner())

        with self.subTest("close"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = GeneratorExit()
            # GeneratorExit is suppressed. This is consistent with PEP 342:
            # https://peps.python.org/pep-0342/#new-generator-method-close
            g.close()
            self.assert_stop_iteration(g)

        with self.subTest("throw GeneratorExit"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = GeneratorExit()
            thrown = GeneratorExit()
            with self.assertRaises(GeneratorExit) as caught:
                g.throw(thrown)
            # The raised GeneratorExit is suppressed, but the thrown one
            # propagates. This is consistent with PEP 380:
            # https://peps.python.org/pep-0380/#proposal
            self.assertIs(caught.exception, thrown)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw StopIteration"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = GeneratorExit()
            thrown = StopIteration()
            with self.assertRaises(GeneratorExit) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw BaseException"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = GeneratorExit()
            thrown = BaseException()
            with self.assertRaises(GeneratorExit) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw Exception"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = GeneratorExit()
            thrown = Exception()
            with self.assertRaises(GeneratorExit) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

    def test_close_and_throw_raise_stop_iteration(self):

        yielded_first = object()
        yielded_second = object()
        returned = object()

        def inner():
            try:
                yield yielded_first
                yield yielded_second
                return returned
            finally:
                raise raised

        def outer():
            return (yield from inner())

        with self.subTest("close"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = StopIteration()
            # PEP 479:
            with self.assert_generator_raised_stop_iteration() as caught:
                g.close()
            self.assertIs(caught.exception.__context__, raised)
            self.assertIsInstance(caught.exception.__context__.__context__, GeneratorExit)
            self.assertIsNone(caught.exception.__context__.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw GeneratorExit"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = StopIteration()
            thrown = GeneratorExit()
            # PEP 479:
            with self.assert_generator_raised_stop_iteration() as caught:
                g.throw(thrown)
            self.assertIs(caught.exception.__context__, raised)
            # This isn't the same GeneratorExit as thrown! It's the one created
            # by calling inner.close():
            self.assertIsInstance(caught.exception.__context__.__context__, GeneratorExit)
            self.assertIsNone(caught.exception.__context__.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw StopIteration"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = StopIteration()
            thrown = StopIteration()
            # PEP 479:
            with self.assert_generator_raised_stop_iteration() as caught:
                g.throw(thrown)
            self.assertIs(caught.exception.__context__, raised)
            self.assertIs(caught.exception.__context__.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw BaseException"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = StopIteration()
            thrown = BaseException()
            # PEP 479:
            with self.assert_generator_raised_stop_iteration() as caught:
                g.throw(thrown)
            self.assertIs(caught.exception.__context__, raised)
            self.assertIs(caught.exception.__context__.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw Exception"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = StopIteration()
            thrown = Exception()
            # PEP 479:
            with self.assert_generator_raised_stop_iteration() as caught:
                g.throw(thrown)
            self.assertIs(caught.exception.__context__, raised)
            self.assertIs(caught.exception.__context__.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__.__context__)
            self.assert_stop_iteration(g)

    def test_close_and_throw_raise_base_exception(self):

        yielded_first = object()
        yielded_second = object()
        returned = object()

        def inner():
            try:
                yield yielded_first
                yield yielded_second
                return returned
            finally:
                raise raised

        def outer():
            return (yield from inner())

        with self.subTest("close"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = BaseException()
            with self.assertRaises(BaseException) as caught:
                g.close()
            self.assertIs(caught.exception, raised)
            self.assertIsInstance(caught.exception.__context__, GeneratorExit)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw GeneratorExit"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = BaseException()
            thrown = GeneratorExit()
            with self.assertRaises(BaseException) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            # This isn't the same GeneratorExit as thrown! It's the one created
            # by calling inner.close():
            self.assertIsInstance(caught.exception.__context__, GeneratorExit)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw StopIteration"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = BaseException()
            thrown = StopIteration()
            with self.assertRaises(BaseException) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw BaseException"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = BaseException()
            thrown = BaseException()
            with self.assertRaises(BaseException) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw Exception"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = BaseException()
            thrown = Exception()
            with self.assertRaises(BaseException) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

    def test_close_and_throw_raise_exception(self):

        yielded_first = object()
        yielded_second = object()
        returned = object()

        def inner():
            try:
                yield yielded_first
                yield yielded_second
                return returned
            finally:
                raise raised

        def outer():
            return (yield from inner())

        with self.subTest("close"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = Exception()
            with self.assertRaises(Exception) as caught:
                g.close()
            self.assertIs(caught.exception, raised)
            self.assertIsInstance(caught.exception.__context__, GeneratorExit)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw GeneratorExit"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = Exception()
            thrown = GeneratorExit()
            with self.assertRaises(Exception) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            # This isn't the same GeneratorExit as thrown! It's the one created
            # by calling inner.close():
            self.assertIsInstance(caught.exception.__context__, GeneratorExit)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw StopIteration"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = Exception()
            thrown = StopIteration()
            with self.assertRaises(Exception) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw BaseException"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = Exception()
            thrown = BaseException()
            with self.assertRaises(Exception) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw Exception"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            raised = Exception()
            thrown = Exception()
            with self.assertRaises(Exception) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, raised)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

    def test_close_and_throw_yield(self):

        yielded_first = object()
        yielded_second = object()
        returned = object()

        def inner():
            try:
                yield yielded_first
            finally:
                yield yielded_second
            return returned

        def outer():
            return (yield from inner())

        with self.subTest("close"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            # No chaining happens. This is consistent with PEP 342:
            # https://peps.python.org/pep-0342/#new-generator-method-close
            with self.assert_generator_ignored_generator_exit() as caught:
                g.close()
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw GeneratorExit"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = GeneratorExit()
            # No chaining happens. This is consistent with PEP 342:
            # https://peps.python.org/pep-0342/#new-generator-method-close
            with self.assert_generator_ignored_generator_exit() as caught:
                g.throw(thrown)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw StopIteration"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = StopIteration()
            self.assertEqual(g.throw(thrown), yielded_second)
            # PEP 479:
            with self.assert_generator_raised_stop_iteration() as caught:
                next(g)
            self.assertIs(caught.exception.__context__, thrown)
            self.assertIsNone(caught.exception.__context__.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw BaseException"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = BaseException()
            self.assertEqual(g.throw(thrown), yielded_second)
            with self.assertRaises(BaseException) as caught:
                next(g)
            self.assertIs(caught.exception, thrown)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw Exception"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = Exception()
            self.assertEqual(g.throw(thrown), yielded_second)
            with self.assertRaises(Exception) as caught:
                next(g)
            self.assertIs(caught.exception, thrown)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

    def test_close_and_throw_return(self):

        yielded_first = object()
        yielded_second = object()
        returned = object()

        def inner():
            try:
                yield yielded_first
                yield yielded_second
            finally:
                return returned

        def outer():
            return (yield from inner())

        with self.subTest("close"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            # StopIteration is suppressed. This is consistent with PEP 342:
            # https://peps.python.org/pep-0342/#new-generator-method-close
            g.close()
            self.assert_stop_iteration(g)

        with self.subTest("throw GeneratorExit"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = GeneratorExit()
            # StopIteration is suppressed. This is consistent with PEP 342:
            # https://peps.python.org/pep-0342/#new-generator-method-close
            with self.assertRaises(GeneratorExit) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception, thrown)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw StopIteration"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = StopIteration()
            with self.assertRaises(StopIteration) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception.value, returned)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw BaseException"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = BaseException()
            with self.assertRaises(StopIteration) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception.value, returned)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)

        with self.subTest("throw Exception"):
            g = outer()
            self.assertIs(next(g), yielded_first)
            thrown = Exception()
            with self.assertRaises(StopIteration) as caught:
                g.throw(thrown)
            self.assertIs(caught.exception.value, returned)
            self.assertIsNone(caught.exception.__context__)
            self.assert_stop_iteration(g)


if __name__ == "__main__":
    run_tests()
