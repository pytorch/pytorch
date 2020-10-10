"""
Test the decorators from ``testing.decorators``.

"""
import warnings
import pytest

from numpy.testing import (
    assert_, assert_raises, dec, SkipTest, KnownFailureException,
    )


try:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        import nose  # noqa: F401
except ImportError:
    HAVE_NOSE = False
else:
    HAVE_NOSE = True


@pytest.mark.skipif(not HAVE_NOSE, reason="Needs nose")
class TestNoseDecorators:
    # These tests are run in a class for simplicity while still
    # getting a report on each, skipped or success.

    class DidntSkipException(Exception):
        pass

    def test_slow(self):
        @dec.slow
        def slow_func(x, y, z):
            pass

        assert_(slow_func.slow)

    def test_setastest(self):
        @dec.setastest()
        def f_default(a):
            pass

        @dec.setastest(True)
        def f_istest(a):
            pass

        @dec.setastest(False)
        def f_isnottest(a):
            pass

        assert_(f_default.__test__)
        assert_(f_istest.__test__)
        assert_(not f_isnottest.__test__)

    def test_skip_functions_hardcoded(self):
        @dec.skipif(True)
        def f1(x):
            raise self.DidntSkipException

        try:
            f1('a')
        except self.DidntSkipException:
            raise Exception('Failed to skip')
        except SkipTest().__class__:
            pass

        @dec.skipif(False)
        def f2(x):
            raise self.DidntSkipException

        try:
            f2('a')
        except self.DidntSkipException:
            pass
        except SkipTest().__class__:
            raise Exception('Skipped when not expected to')

    def test_skip_functions_callable(self):
        def skip_tester():
            return skip_flag == 'skip me!'

        @dec.skipif(skip_tester)
        def f1(x):
            raise self.DidntSkipException

        try:
            skip_flag = 'skip me!'
            f1('a')
        except self.DidntSkipException:
            raise Exception('Failed to skip')
        except SkipTest().__class__:
            pass

        @dec.skipif(skip_tester)
        def f2(x):
            raise self.DidntSkipException

        try:
            skip_flag = 'five is right out!'
            f2('a')
        except self.DidntSkipException:
            pass
        except SkipTest().__class__:
            raise Exception('Skipped when not expected to')

    def test_skip_generators_hardcoded(self):
        @dec.knownfailureif(True, "This test is known to fail")
        def g1(x):
            yield from range(x)

        try:
            for j in g1(10):
                pass
        except KnownFailureException().__class__:
            pass
        else:
            raise Exception('Failed to mark as known failure')

        @dec.knownfailureif(False, "This test is NOT known to fail")
        def g2(x):
            yield from range(x)
            raise self.DidntSkipException('FAIL')

        try:
            for j in g2(10):
                pass
        except KnownFailureException().__class__:
            raise Exception('Marked incorrectly as known failure')
        except self.DidntSkipException:
            pass

    def test_skip_generators_callable(self):
        def skip_tester():
            return skip_flag == 'skip me!'

        @dec.knownfailureif(skip_tester, "This test is known to fail")
        def g1(x):
            yield from range(x)

        try:
            skip_flag = 'skip me!'
            for j in g1(10):
                pass
        except KnownFailureException().__class__:
            pass
        else:
            raise Exception('Failed to mark as known failure')

        @dec.knownfailureif(skip_tester, "This test is NOT known to fail")
        def g2(x):
            yield from range(x)
            raise self.DidntSkipException('FAIL')

        try:
            skip_flag = 'do not skip'
            for j in g2(10):
                pass
        except KnownFailureException().__class__:
            raise Exception('Marked incorrectly as known failure')
        except self.DidntSkipException:
            pass

    def test_deprecated(self):
        @dec.deprecated(True)
        def non_deprecated_func():
            pass

        @dec.deprecated()
        def deprecated_func():
            import warnings
            warnings.warn("TEST: deprecated func", DeprecationWarning)

        @dec.deprecated()
        def deprecated_func2():
            import warnings
            warnings.warn("AHHHH")
            raise ValueError

        @dec.deprecated()
        def deprecated_func3():
            import warnings
            warnings.warn("AHHHH")

        # marked as deprecated, but does not raise DeprecationWarning
        assert_raises(AssertionError, non_deprecated_func)
        # should be silent
        deprecated_func()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # do not propagate unrelated warnings
            # fails if deprecated decorator just disables test. See #1453.
            assert_raises(ValueError, deprecated_func2)
            # warning is not a DeprecationWarning
            assert_raises(AssertionError, deprecated_func3)

    def test_parametrize(self):
        # dec.parametrize assumes that it is being run by nose. Because
        # we are running under pytest, we need to explicitly check the
        # results.
        @dec.parametrize('base, power, expected',
                [(1, 1, 1),
                 (2, 1, 2),
                 (2, 2, 4)])
        def check_parametrize(base, power, expected):
            assert_(base**power == expected)

        count = 0
        for test in check_parametrize():
            test[0](*test[1:])
            count += 1
        assert_(count == 3)
