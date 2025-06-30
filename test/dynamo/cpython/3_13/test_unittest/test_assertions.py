# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch.testing._internal.common_utils import run_tests


__TestCase = torch._dynamo.test_case.CPythonTestCase


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

import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product


class Test_Assertions(__TestCase):
    def test_AlmostEqual(self):
        self.assertAlmostEqual(1.00000001, 1.0)
        self.assertNotAlmostEqual(1.0000001, 1.0)
        self.assertRaises(self.failureException,
                          self.assertAlmostEqual, 1.0000001, 1.0)
        self.assertRaises(self.failureException,
                          self.assertNotAlmostEqual, 1.00000001, 1.0)

        self.assertAlmostEqual(1.1, 1.0, places=0)
        self.assertRaises(self.failureException,
                          self.assertAlmostEqual, 1.1, 1.0, places=1)

        self.assertAlmostEqual(0, .1+.1j, places=0)
        self.assertNotAlmostEqual(0, .1+.1j, places=1)
        self.assertRaises(self.failureException,
                          self.assertAlmostEqual, 0, .1+.1j, places=1)
        self.assertRaises(self.failureException,
                          self.assertNotAlmostEqual, 0, .1+.1j, places=0)

        self.assertAlmostEqual(float('inf'), float('inf'))
        self.assertRaises(self.failureException, self.assertNotAlmostEqual,
                          float('inf'), float('inf'))

    def test_AmostEqualWithDelta(self):
        self.assertAlmostEqual(1.1, 1.0, delta=0.5)
        self.assertAlmostEqual(1.0, 1.1, delta=0.5)
        self.assertNotAlmostEqual(1.1, 1.0, delta=0.05)
        self.assertNotAlmostEqual(1.0, 1.1, delta=0.05)

        self.assertAlmostEqual(1.0, 1.0, delta=0.5)
        self.assertRaises(self.failureException, self.assertNotAlmostEqual,
                          1.0, 1.0, delta=0.5)

        self.assertRaises(self.failureException, self.assertAlmostEqual,
                          1.1, 1.0, delta=0.05)
        self.assertRaises(self.failureException, self.assertNotAlmostEqual,
                          1.1, 1.0, delta=0.5)

        self.assertRaises(TypeError, self.assertAlmostEqual,
                          1.1, 1.0, places=2, delta=2)
        self.assertRaises(TypeError, self.assertNotAlmostEqual,
                          1.1, 1.0, places=2, delta=2)

        first = datetime.datetime.now()
        second = first + datetime.timedelta(seconds=10)
        self.assertAlmostEqual(first, second,
                               delta=datetime.timedelta(seconds=20))
        self.assertNotAlmostEqual(first, second,
                                  delta=datetime.timedelta(seconds=5))

    def test_assertRaises(self):
        def _raise(e):
            raise e
        self.assertRaises(KeyError, _raise, KeyError)
        self.assertRaises(KeyError, _raise, KeyError("key"))
        try:
            self.assertRaises(KeyError, lambda: None)
        except self.failureException as e:
            self.assertIn("KeyError not raised", str(e))
        else:
            self.fail("assertRaises() didn't fail")
        try:
            self.assertRaises(KeyError, _raise, ValueError)
        except ValueError:
            pass
        else:
            self.fail("assertRaises() didn't let exception pass through")
        with self.assertRaises(KeyError) as cm:
            try:
                raise KeyError
            except Exception as e:
                exc = e
                raise
        self.assertIs(cm.exception, exc)

        with self.assertRaises(KeyError):
            raise KeyError("key")
        try:
            with self.assertRaises(KeyError):
                pass
        except self.failureException as e:
            self.assertIn("KeyError not raised", str(e))
        else:
            self.fail("assertRaises() didn't fail")
        try:
            with self.assertRaises(KeyError):
                raise ValueError
        except ValueError:
            pass
        else:
            self.fail("assertRaises() didn't let exception pass through")

    def test_assertRaises_frames_survival(self):
        # Issue #9815: assertRaises should avoid keeping local variables
        # in a traceback alive.
        class A:
            pass
        wr = None

        class Foo(unittest.TestCase):

            def foo(self):
                nonlocal wr
                a = A()
                wr = weakref.ref(a)
                try:
                    raise OSError
                except OSError:
                    raise ValueError

            def test_functional(self):
                self.assertRaises(ValueError, self.foo)

            def test_with(self):
                with self.assertRaises(ValueError):
                    self.foo()

        Foo("test_functional").run()
        gc_collect()  # For PyPy or other GCs.
        self.assertIsNone(wr())
        Foo("test_with").run()
        gc_collect()  # For PyPy or other GCs.
        self.assertIsNone(wr())

    def testAssertNotRegex(self):
        self.assertNotRegex('Ala ma kota', r'r+')
        try:
            self.assertNotRegex('Ala ma kota', r'k.t', 'Message')
        except self.failureException as e:
            self.assertIn('Message', e.args[0])
        else:
            self.fail('assertNotRegex should have failed.')


class TestLongMessage(__TestCase):
    """Test that the individual asserts honour longMessage.
    This actually tests all the message behaviour for
    asserts that use longMessage."""

    def setUp(self):
        super().setUp()
        class TestableTestFalse(unittest.TestCase):
            longMessage = False
            failureException = self.failureException

            def testTest(self):
                pass

        class TestableTestTrue(unittest.TestCase):
            longMessage = True
            failureException = self.failureException

            def testTest(self):
                pass

        self.testableTrue = TestableTestTrue('testTest')
        self.testableFalse = TestableTestFalse('testTest')

    def testDefault(self):
        self.assertTrue(unittest.TestCase.longMessage)

    def test_formatMsg(self):
        self.assertEqual(self.testableFalse._formatMessage(None, "foo"), "foo")
        self.assertEqual(self.testableFalse._formatMessage("foo", "bar"), "foo")

        self.assertEqual(self.testableTrue._formatMessage(None, "foo"), "foo")
        self.assertEqual(self.testableTrue._formatMessage("foo", "bar"), "bar : foo")

        # This blows up if _formatMessage uses string concatenation
        self.testableTrue._formatMessage(object(), 'foo')

    def test_formatMessage_unicode_error(self):
        one = ''.join(chr(i) for i in range(255))
        # this used to cause a UnicodeDecodeError constructing msg
        self.testableTrue._formatMessage(one, '\uFFFD')

    def assertMessages(self, methodName, args, errors):
        """
        Check that methodName(*args) raises the correct error messages.
        errors should be a list of 4 regex that match the error when:
          1) longMessage = False and no msg passed;
          2) longMessage = False and msg passed;
          3) longMessage = True and no msg passed;
          4) longMessage = True and msg passed;
        """
        def getMethod(i):
            useTestableFalse  = i < 2
            if useTestableFalse:
                test = self.testableFalse
            else:
                test = self.testableTrue
            return getattr(test, methodName)

        for i, expected_regex in enumerate(errors):
            testMethod = getMethod(i)
            kwargs = {}
            withMsg = i % 2
            if withMsg:
                kwargs = {"msg": "oops"}

            with self.assertRaisesRegex(self.failureException,
                                        expected_regex=expected_regex):
                testMethod(*args, **kwargs)

    def testAssertTrue(self):
        self.assertMessages('assertTrue', (False,),
                            ["^False is not true$", "^oops$", "^False is not true$",
                             "^False is not true : oops$"])

    def testAssertFalse(self):
        self.assertMessages('assertFalse', (True,),
                            ["^True is not false$", "^oops$", "^True is not false$",
                             "^True is not false : oops$"])

    def testNotEqual(self):
        self.assertMessages('assertNotEqual', (1, 1),
                            ["^1 == 1$", "^oops$", "^1 == 1$",
                             "^1 == 1 : oops$"])

    def testAlmostEqual(self):
        self.assertMessages(
            'assertAlmostEqual', (1, 2),
            [r"^1 != 2 within 7 places \(1 difference\)$", "^oops$",
             r"^1 != 2 within 7 places \(1 difference\)$",
             r"^1 != 2 within 7 places \(1 difference\) : oops$"])

    def testNotAlmostEqual(self):
        self.assertMessages('assertNotAlmostEqual', (1, 1),
                            ["^1 == 1 within 7 places$", "^oops$",
                             "^1 == 1 within 7 places$", "^1 == 1 within 7 places : oops$"])

    def test_baseAssertEqual(self):
        self.assertMessages('_baseAssertEqual', (1, 2),
                            ["^1 != 2$", "^oops$", "^1 != 2$", "^1 != 2 : oops$"])

    def testAssertSequenceEqual(self):
        # Error messages are multiline so not testing on full message
        # assertTupleEqual and assertListEqual delegate to this method
        self.assertMessages('assertSequenceEqual', ([], [None]),
                            [r"\+ \[None\]$", "^oops$", r"\+ \[None\]$",
                             r"\+ \[None\] : oops$"])

    def testAssertSetEqual(self):
        self.assertMessages('assertSetEqual', (set(), set([None])),
                            ["None$", "^oops$", "None$",
                             "None : oops$"])

    def testAssertIn(self):
        self.assertMessages('assertIn', (None, []),
                            [r'^None not found in \[\]$', "^oops$",
                             r'^None not found in \[\]$',
                             r'^None not found in \[\] : oops$'])

    def testAssertNotIn(self):
        self.assertMessages('assertNotIn', (None, [None]),
                            [r'^None unexpectedly found in \[None\]$', "^oops$",
                             r'^None unexpectedly found in \[None\]$',
                             r'^None unexpectedly found in \[None\] : oops$'])

    def testAssertDictEqual(self):
        self.assertMessages('assertDictEqual', ({}, {'key': 'value'}),
                            [r"\+ \{'key': 'value'\}$", "^oops$",
                             r"\+ \{'key': 'value'\}$",
                             r"\+ \{'key': 'value'\} : oops$"])

    def testAssertMultiLineEqual(self):
        self.assertMessages('assertMultiLineEqual', ("", "foo"),
                            [r"\+ foo\n$", "^oops$",
                             r"\+ foo\n$",
                             r"\+ foo\n : oops$"])

    def testAssertLess(self):
        self.assertMessages('assertLess', (2, 1),
                            ["^2 not less than 1$", "^oops$",
                             "^2 not less than 1$", "^2 not less than 1 : oops$"])

    def testAssertLessEqual(self):
        self.assertMessages('assertLessEqual', (2, 1),
                            ["^2 not less than or equal to 1$", "^oops$",
                             "^2 not less than or equal to 1$",
                             "^2 not less than or equal to 1 : oops$"])

    def testAssertGreater(self):
        self.assertMessages('assertGreater', (1, 2),
                            ["^1 not greater than 2$", "^oops$",
                             "^1 not greater than 2$",
                             "^1 not greater than 2 : oops$"])

    def testAssertGreaterEqual(self):
        self.assertMessages('assertGreaterEqual', (1, 2),
                            ["^1 not greater than or equal to 2$", "^oops$",
                             "^1 not greater than or equal to 2$",
                             "^1 not greater than or equal to 2 : oops$"])

    def testAssertIsNone(self):
        self.assertMessages('assertIsNone', ('not None',),
                            ["^'not None' is not None$", "^oops$",
                             "^'not None' is not None$",
                             "^'not None' is not None : oops$"])

    def testAssertIsNotNone(self):
        self.assertMessages('assertIsNotNone', (None,),
                            ["^unexpectedly None$", "^oops$",
                             "^unexpectedly None$",
                             "^unexpectedly None : oops$"])

    def testAssertIs(self):
        self.assertMessages('assertIs', (None, 'foo'),
                            ["^None is not 'foo'$", "^oops$",
                             "^None is not 'foo'$",
                             "^None is not 'foo' : oops$"])

    def testAssertIsNot(self):
        self.assertMessages('assertIsNot', (None, None),
                            ["^unexpectedly identical: None$", "^oops$",
                             "^unexpectedly identical: None$",
                             "^unexpectedly identical: None : oops$"])

    def testAssertRegex(self):
        self.assertMessages('assertRegex', ('foo', 'bar'),
                            ["^Regex didn't match:",
                             "^oops$",
                             "^Regex didn't match:",
                             "^Regex didn't match: (.*) : oops$"])

    def testAssertNotRegex(self):
        self.assertMessages('assertNotRegex', ('foo', 'foo'),
                            ["^Regex matched:",
                             "^oops$",
                             "^Regex matched:",
                             "^Regex matched: (.*) : oops$"])


    def assertMessagesCM(self, methodName, args, func, errors):
        """
        Check that the correct error messages are raised while executing:
          with method(*args):
              func()
        *errors* should be a list of 4 regex that match the error when:
          1) longMessage = False and no msg passed;
          2) longMessage = False and msg passed;
          3) longMessage = True and no msg passed;
          4) longMessage = True and msg passed;
        """
        p = product((self.testableFalse, self.testableTrue),
                    ({}, {"msg": "oops"}))
        for (cls, kwargs), err in zip(p, errors):
            method = getattr(cls, methodName)
            with self.assertRaisesRegex(cls.failureException, err):
                with method(*args, **kwargs) as cm:
                    func()

    def testAssertRaises(self):
        self.assertMessagesCM('assertRaises', (TypeError,), lambda: None,
                              ['^TypeError not raised$', '^oops$',
                               '^TypeError not raised$',
                               '^TypeError not raised : oops$'])

    def testAssertRaisesRegex(self):
        # test error not raised
        self.assertMessagesCM('assertRaisesRegex', (TypeError, 'unused regex'),
                              lambda: None,
                              ['^TypeError not raised$', '^oops$',
                               '^TypeError not raised$',
                               '^TypeError not raised : oops$'])
        # test error raised but with wrong message
        def raise_wrong_message():
            raise TypeError('foo')
        self.assertMessagesCM('assertRaisesRegex', (TypeError, 'regex'),
                              raise_wrong_message,
                              ['^"regex" does not match "foo"$', '^oops$',
                               '^"regex" does not match "foo"$',
                               '^"regex" does not match "foo" : oops$'])

    def testAssertWarns(self):
        self.assertMessagesCM('assertWarns', (UserWarning,), lambda: None,
                              ['^UserWarning not triggered$', '^oops$',
                               '^UserWarning not triggered$',
                               '^UserWarning not triggered : oops$'])

    def test_assertNotWarns(self):
        def warn_future():
            warnings.warn('xyz', FutureWarning, stacklevel=2)
        self.assertMessagesCM('_assertNotWarns', (FutureWarning,),
                              warn_future,
                              ['^FutureWarning triggered$',
                               '^oops$',
                               '^FutureWarning triggered$',
                               '^FutureWarning triggered : oops$'])

    def testAssertWarnsRegex(self):
        # test error not raised
        self.assertMessagesCM('assertWarnsRegex', (UserWarning, 'unused regex'),
                              lambda: None,
                              ['^UserWarning not triggered$', '^oops$',
                               '^UserWarning not triggered$',
                               '^UserWarning not triggered : oops$'])
        # test warning raised but with wrong message
        def raise_wrong_message():
            warnings.warn('foo')
        self.assertMessagesCM('assertWarnsRegex', (UserWarning, 'regex'),
                              raise_wrong_message,
                              ['^"regex" does not match "foo"$', '^oops$',
                               '^"regex" does not match "foo"$',
                               '^"regex" does not match "foo" : oops$'])


if __name__ == "__main__":
    run_tests()
