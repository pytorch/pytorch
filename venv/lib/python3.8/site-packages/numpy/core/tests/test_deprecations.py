"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
import datetime
import operator
import warnings
import pytest
import tempfile
import re

import numpy as np
from numpy.testing import (
    assert_raises, assert_warns, assert_, assert_array_equal
    )

from numpy.core._multiarray_tests import fromstring_null_term_c_api

try:
    import pytz
    _has_pytz = True
except ImportError:
    _has_pytz = False


class _DeprecationTestCase:
    # Just as warning: warnings uses re.match, so the start of this message
    # must match.
    message = ''
    warning_cls = DeprecationWarning

    def setup(self):
        self.warn_ctx = warnings.catch_warnings(record=True)
        self.log = self.warn_ctx.__enter__()

        # Do *not* ignore other DeprecationWarnings. Ignoring warnings
        # can give very confusing results because of
        # https://bugs.python.org/issue4180 and it is probably simplest to
        # try to keep the tests cleanly giving only the right warning type.
        # (While checking them set to "error" those are ignored anyway)
        # We still have them show up, because otherwise they would be raised
        warnings.filterwarnings("always", category=self.warning_cls)
        warnings.filterwarnings("always", message=self.message,
                                category=self.warning_cls)

    def teardown(self):
        self.warn_ctx.__exit__()

    def assert_deprecated(self, function, num=1, ignore_others=False,
                          function_fails=False,
                          exceptions=np._NoValue,
                          args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.

        This first checks if the function when called gives `num`
        DeprecationWarnings, after that it tries to raise these
        DeprecationWarnings and compares them with `exceptions`.
        The exceptions can be different for cases where this code path
        is simply not anticipated and the exception is replaced.

        Parameters
        ----------
        function : callable
            The function to test
        num : int
            Number of DeprecationWarnings to expect. This should normally be 1.
        ignore_others : bool
            Whether warnings of the wrong type should be ignored (note that
            the message is not checked)
        function_fails : bool
            If the function would normally fail, setting this will check for
            warnings inside a try/except block.
        exceptions : Exception or tuple of Exceptions
            Exception to expect when turning the warnings into an error.
            The default checks for DeprecationWarnings. If exceptions is
            empty the function is expected to run successfully.
        args : tuple
            Arguments for `function`
        kwargs : dict
            Keyword arguments for `function`
        """
        # reset the log
        self.log[:] = []

        if exceptions is np._NoValue:
            exceptions = (self.warning_cls,)

        try:
            function(*args, **kwargs)
        except (Exception if function_fails else tuple()):
            pass

        # just in case, clear the registry
        num_found = 0
        for warning in self.log:
            if warning.category is self.warning_cls:
                num_found += 1
            elif not ignore_others:
                raise AssertionError(
                        "expected %s but got: %s" %
                        (self.warning_cls.__name__, warning.category))
        if num is not None and num_found != num:
            msg = "%i warnings found but %i expected." % (len(self.log), num)
            lst = [str(w) for w in self.log]
            raise AssertionError("\n".join([msg] + lst))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=self.message,
                                    category=self.warning_cls)
            try:
                function(*args, **kwargs)
                if exceptions != tuple():
                    raise AssertionError(
                            "No error raised during function call")
            except exceptions:
                if exceptions == tuple():
                    raise AssertionError(
                            "Error raised during function call")

    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test that warnings are not raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)


class _VisibleDeprecationTestCase(_DeprecationTestCase):
    warning_cls = np.VisibleDeprecationWarning


class TestNonTupleNDIndexDeprecation:
    def test_basic(self):
        a = np.zeros((5, 5))
        with warnings.catch_warnings():
            warnings.filterwarnings('always')
            assert_warns(FutureWarning, a.__getitem__, [[0, 1], [0, 1]])
            assert_warns(FutureWarning, a.__getitem__, [slice(None)])

            warnings.filterwarnings('error')
            assert_raises(FutureWarning, a.__getitem__, [[0, 1], [0, 1]])
            assert_raises(FutureWarning, a.__getitem__, [slice(None)])

            # a a[[0, 1]] always was advanced indexing, so no error/warning
            a[[0, 1]]


class TestComparisonDeprecations(_DeprecationTestCase):
    """This tests the deprecation, for non-element-wise comparison logic.
    This used to mean that when an error occurred during element-wise comparison
    (i.e. broadcasting) NotImplemented was returned, but also in the comparison
    itself, False was given instead of the error.

    Also test FutureWarning for the None comparison.
    """

    message = "elementwise.* comparison failed; .*"

    def test_normal_types(self):
        for op in (operator.eq, operator.ne):
            # Broadcasting errors:
            self.assert_deprecated(op, args=(np.zeros(3), []))
            a = np.zeros(3, dtype='i,i')
            # (warning is issued a couple of times here)
            self.assert_deprecated(op, args=(a, a[:-1]), num=None)

            # ragged array comparison returns True/False
            a = np.array([1, np.array([1,2,3])], dtype=object)
            b = np.array([1, np.array([1,2,3])], dtype=object)
            self.assert_deprecated(op, args=(a, b), num=None)

    def test_string(self):
        # For two string arrays, strings always raised the broadcasting error:
        a = np.array(['a', 'b'])
        b = np.array(['a', 'b', 'c'])
        assert_raises(ValueError, lambda x, y: x == y, a, b)

        # The empty list is not cast to string, and this used to pass due
        # to dtype mismatch; now (2018-06-21) it correctly leads to a
        # FutureWarning.
        assert_warns(FutureWarning, lambda: a == [])

    def test_void_dtype_equality_failures(self):
        class NotArray:
            def __array__(self):
                raise TypeError

            # Needed so Python 3 does not raise DeprecationWarning twice.
            def __ne__(self, other):
                return NotImplemented

        self.assert_deprecated(lambda: np.arange(2) == NotArray())
        self.assert_deprecated(lambda: np.arange(2) != NotArray())

        struct1 = np.zeros(2, dtype="i4,i4")
        struct2 = np.zeros(2, dtype="i4,i4,i4")

        assert_warns(FutureWarning, lambda: struct1 == 1)
        assert_warns(FutureWarning, lambda: struct1 == struct2)
        assert_warns(FutureWarning, lambda: struct1 != 1)
        assert_warns(FutureWarning, lambda: struct1 != struct2)

    def test_array_richcompare_legacy_weirdness(self):
        # It doesn't really work to use assert_deprecated here, b/c part of
        # the point of assert_deprecated is to check that when warnings are
        # set to "error" mode then the error is propagated -- which is good!
        # But here we are testing a bunch of code that is deprecated *because*
        # it has the habit of swallowing up errors and converting them into
        # different warnings. So assert_warns will have to be sufficient.
        assert_warns(FutureWarning, lambda: np.arange(2) == "a")
        assert_warns(FutureWarning, lambda: np.arange(2) != "a")
        # No warning for scalar comparisons
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            assert_(not (np.array(0) == "a"))
            assert_(np.array(0) != "a")
            assert_(not (np.int16(0) == "a"))
            assert_(np.int16(0) != "a")

        for arg1 in [np.asarray(0), np.int16(0)]:
            struct = np.zeros(2, dtype="i4,i4")
            for arg2 in [struct, "a"]:
                for f in [operator.lt, operator.le, operator.gt, operator.ge]:
                    with warnings.catch_warnings() as l:
                        warnings.filterwarnings("always")
                        assert_raises(TypeError, f, arg1, arg2)
                        assert_(not l)


class TestDatetime64Timezone(_DeprecationTestCase):
    """Parsing of datetime64 with timezones deprecated in 1.11.0, because
    datetime64 is now timezone naive rather than UTC only.

    It will be quite a while before we can remove this, because, at the very
    least, a lot of existing code uses the 'Z' modifier to avoid conversion
    from local time to UTC, even if otherwise it handles time in a timezone
    naive fashion.
    """
    def test_string(self):
        self.assert_deprecated(np.datetime64, args=('2000-01-01T00+01',))
        self.assert_deprecated(np.datetime64, args=('2000-01-01T00Z',))

    @pytest.mark.skipif(not _has_pytz,
                        reason="The pytz module is not available.")
    def test_datetime(self):
        tz = pytz.timezone('US/Eastern')
        dt = datetime.datetime(2000, 1, 1, 0, 0, tzinfo=tz)
        self.assert_deprecated(np.datetime64, args=(dt,))


class TestNonCContiguousViewDeprecation(_DeprecationTestCase):
    """View of non-C-contiguous arrays deprecated in 1.11.0.

    The deprecation will not be raised for arrays that are both C and F
    contiguous, as C contiguous is dominant. There are more such arrays
    with relaxed stride checking than without so the deprecation is not
    as visible with relaxed stride checking in force.
    """

    def test_fortran_contiguous(self):
        self.assert_deprecated(np.ones((2,2)).T.view, args=(complex,))
        self.assert_deprecated(np.ones((2,2)).T.view, args=(np.int8,))


class TestArrayDataAttributeAssignmentDeprecation(_DeprecationTestCase):
    """Assigning the 'data' attribute of an ndarray is unsafe as pointed
     out in gh-7093. Eventually, such assignment should NOT be allowed, but
     in the interests of maintaining backwards compatibility, only a Deprecation-
     Warning will be raised instead for the time being to give developers time to
     refactor relevant code.
    """

    def test_data_attr_assignment(self):
        a = np.arange(10)
        b = np.linspace(0, 1, 10)

        self.message = ("Assigning the 'data' attribute is an "
                        "inherently unsafe operation and will "
                        "be removed in the future.")
        self.assert_deprecated(a.__setattr__, args=('data', b.data))


class TestBinaryReprInsufficientWidthParameterForRepresentation(_DeprecationTestCase):
    """
    If a 'width' parameter is passed into ``binary_repr`` that is insufficient to
    represent the number in base 2 (positive) or 2's complement (negative) form,
    the function used to silently ignore the parameter and return a representation
    using the minimal number of bits needed for the form in question. Such behavior
    is now considered unsafe from a user perspective and will raise an error in the future.
    """

    def test_insufficient_width_positive(self):
        args = (10,)
        kwargs = {'width': 2}

        self.message = ("Insufficient bit width provided. This behavior "
                        "will raise an error in the future.")
        self.assert_deprecated(np.binary_repr, args=args, kwargs=kwargs)

    def test_insufficient_width_negative(self):
        args = (-5,)
        kwargs = {'width': 2}

        self.message = ("Insufficient bit width provided. This behavior "
                        "will raise an error in the future.")
        self.assert_deprecated(np.binary_repr, args=args, kwargs=kwargs)


class TestNumericStyleTypecodes(_DeprecationTestCase):
    """
    Deprecate the old numeric-style dtypes, which are especially
    confusing for complex types, e.g. Complex32 -> complex64. When the
    deprecation cycle is complete, the check for the strings should be
    removed from PyArray_DescrConverter in descriptor.c, and the
    deprecated keys should not be added as capitalized aliases in
    _add_aliases in numerictypes.py.
    """
    def test_all_dtypes(self):
        deprecated_types = [
            'Bool', 'Complex32', 'Complex64', 'Float16', 'Float32', 'Float64',
            'Int8', 'Int16', 'Int32', 'Int64', 'Object0', 'Timedelta64',
            'UInt8', 'UInt16', 'UInt32', 'UInt64', 'Void0'
            ]
        for dt in deprecated_types:
            self.assert_deprecated(np.dtype, exceptions=(TypeError,),
                                   args=(dt,))


class TestTestDeprecated:
    def test_assert_deprecated(self):
        test_case_instance = _DeprecationTestCase()
        test_case_instance.setup()
        assert_raises(AssertionError,
                      test_case_instance.assert_deprecated,
                      lambda: None)

        def foo():
            warnings.warn("foo", category=DeprecationWarning, stacklevel=2)

        test_case_instance.assert_deprecated(foo)
        test_case_instance.teardown()


class TestNonNumericConjugate(_DeprecationTestCase):
    """
    Deprecate no-op behavior of ndarray.conjugate on non-numeric dtypes,
    which conflicts with the error behavior of np.conjugate.
    """
    def test_conjugate(self):
        for a in np.array(5), np.array(5j):
            self.assert_not_deprecated(a.conjugate)
        for a in (np.array('s'), np.array('2016', 'M'),
                np.array((1, 2), [('a', int), ('b', int)])):
            self.assert_deprecated(a.conjugate)


class TestNPY_CHAR(_DeprecationTestCase):
    # 2017-05-03, 1.13.0
    def test_npy_char_deprecation(self):
        from numpy.core._multiarray_tests import npy_char_deprecation
        self.assert_deprecated(npy_char_deprecation)
        assert_(npy_char_deprecation() == 'S1')


class TestPyArray_AS1D(_DeprecationTestCase):
    def test_npy_pyarrayas1d_deprecation(self):
        from numpy.core._multiarray_tests import npy_pyarrayas1d_deprecation
        assert_raises(NotImplementedError, npy_pyarrayas1d_deprecation)


class TestPyArray_AS2D(_DeprecationTestCase):
    def test_npy_pyarrayas2d_deprecation(self):
        from numpy.core._multiarray_tests import npy_pyarrayas2d_deprecation
        assert_raises(NotImplementedError, npy_pyarrayas2d_deprecation)


class Test_UPDATEIFCOPY(_DeprecationTestCase):
    """
    v1.14 deprecates creating an array with the UPDATEIFCOPY flag, use
    WRITEBACKIFCOPY instead
    """
    def test_npy_updateifcopy_deprecation(self):
        from numpy.core._multiarray_tests import npy_updateifcopy_deprecation
        arr = np.arange(9).reshape(3, 3)
        v = arr.T
        self.assert_deprecated(npy_updateifcopy_deprecation, args=(v,))


class TestDatetimeEvent(_DeprecationTestCase):
    # 2017-08-11, 1.14.0
    def test_3_tuple(self):
        for cls in (np.datetime64, np.timedelta64):
            # two valid uses - (unit, num) and (unit, num, den, None)
            self.assert_not_deprecated(cls, args=(1, ('ms', 2)))
            self.assert_not_deprecated(cls, args=(1, ('ms', 2, 1, None)))

            # trying to use the event argument, removed in 1.7.0, is deprecated
            # it used to be a uint8
            self.assert_deprecated(cls, args=(1, ('ms', 2, 'event')))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 63)))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 1, 'event')))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 1, 63)))


class TestTruthTestingEmptyArrays(_DeprecationTestCase):
    # 2017-09-25, 1.14.0
    message = '.*truth value of an empty array is ambiguous.*'

    def test_1d(self):
        self.assert_deprecated(bool, args=(np.array([]),))

    def test_2d(self):
        self.assert_deprecated(bool, args=(np.zeros((1, 0)),))
        self.assert_deprecated(bool, args=(np.zeros((0, 1)),))
        self.assert_deprecated(bool, args=(np.zeros((0, 0)),))


class TestBincount(_DeprecationTestCase):
    # 2017-06-01, 1.14.0
    def test_bincount_minlength(self):
        self.assert_deprecated(lambda: np.bincount([1, 2, 3], minlength=None))


class TestAlen(_DeprecationTestCase):
    # 2019-08-02, 1.18.0
    def test_alen(self):
        self.assert_deprecated(lambda: np.alen(np.array([1, 2, 3])))


class TestGeneratorSum(_DeprecationTestCase):
    # 2018-02-25, 1.15.0
    def test_generator_sum(self):
        self.assert_deprecated(np.sum, args=((i for i in range(5)),))


class TestSctypeNA(_VisibleDeprecationTestCase):
    # 2018-06-24, 1.16
    def test_sctypeNA(self):
        self.assert_deprecated(lambda: np.sctypeNA['?'])
        self.assert_deprecated(lambda: np.typeNA['?'])
        self.assert_deprecated(lambda: np.typeNA.get('?'))


class TestPositiveOnNonNumerical(_DeprecationTestCase):
    # 2018-06-28, 1.16.0
    def test_positive_on_non_number(self):
        self.assert_deprecated(operator.pos, args=(np.array('foo'),))


class TestFromstring(_DeprecationTestCase):
    # 2017-10-19, 1.14
    def test_fromstring(self):
        self.assert_deprecated(np.fromstring, args=('\x00'*80,))


class TestFromStringAndFileInvalidData(_DeprecationTestCase):
    # 2019-06-08, 1.17.0
    # Tests should be moved to real tests when deprecation is done.
    message = "string or file could not be read to its end"

    @pytest.mark.parametrize("invalid_str", [",invalid_data", "invalid_sep"])
    def test_deprecate_unparsable_data_file(self, invalid_str):
        x = np.array([1.51, 2, 3.51, 4], dtype=float)

        with tempfile.TemporaryFile(mode="w") as f:
            x.tofile(f, sep=',', format='%.2f')
            f.write(invalid_str)

            f.seek(0)
            self.assert_deprecated(lambda: np.fromfile(f, sep=","))
            f.seek(0)
            self.assert_deprecated(lambda: np.fromfile(f, sep=",", count=5))
            # Should not raise:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                f.seek(0)
                res = np.fromfile(f, sep=",", count=4)
                assert_array_equal(res, x)

    @pytest.mark.parametrize("invalid_str", [",invalid_data", "invalid_sep"])
    def test_deprecate_unparsable_string(self, invalid_str):
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        x_str = "1.51,2,3.51,4{}".format(invalid_str)

        self.assert_deprecated(lambda: np.fromstring(x_str, sep=","))
        self.assert_deprecated(lambda: np.fromstring(x_str, sep=",", count=5))

        # The C-level API can use not fixed size, but 0 terminated strings,
        # so test that as well:
        bytestr = x_str.encode("ascii")
        self.assert_deprecated(lambda: fromstring_null_term_c_api(bytestr))

        with assert_warns(DeprecationWarning):
            # this is slightly strange, in that fromstring leaves data
            # potentially uninitialized (would be good to error when all is
            # read, but count is larger then actual data maybe).
            res = np.fromstring(x_str, sep=",", count=5)
            assert_array_equal(res[:-1], x)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # Should not raise:
            res = np.fromstring(x_str, sep=",", count=4)
            assert_array_equal(res, x)


class Test_GetSet_NumericOps(_DeprecationTestCase):
    # 2018-09-20, 1.16.0
    def test_get_numeric_ops(self):
        from numpy.core._multiarray_tests import getset_numericops
        self.assert_deprecated(getset_numericops, num=2)

        # empty kwargs prevents any state actually changing which would break
        # other tests.
        self.assert_deprecated(np.set_numeric_ops, kwargs={})
        assert_raises(ValueError, np.set_numeric_ops, add='abc')


class TestShape1Fields(_DeprecationTestCase):
    warning_cls = FutureWarning

    # 2019-05-20, 1.17.0
    def test_shape_1_fields(self):
        self.assert_deprecated(np.dtype, args=([('a', int, 1)],))


class TestNonZero(_DeprecationTestCase):
    # 2019-05-26, 1.17.0
    def test_zerod(self):
        self.assert_deprecated(lambda: np.nonzero(np.array(0)))
        self.assert_deprecated(lambda: np.nonzero(np.array(1)))


def test_deprecate_ragged_arrays():
    # 2019-11-29 1.19.0
    #
    # NEP 34 deprecated automatic object dtype when creating ragged
    # arrays. Also see the "ragged" tests in `test_multiarray`
    #
    # emits a VisibleDeprecationWarning
    arg = [1, [2, 3]]
    with assert_warns(np.VisibleDeprecationWarning):
        np.array(arg)


class TestToString(_DeprecationTestCase):
    # 2020-03-06 1.19.0
    message = re.escape("tostring() is deprecated. Use tobytes() instead.")

    def test_tostring(self):
        arr = np.array(list(b"test\xFF"), dtype=np.uint8)
        self.assert_deprecated(arr.tostring)

    def test_tostring_matches_tobytes(self):
        arr = np.array(list(b"test\xFF"), dtype=np.uint8)
        b = arr.tobytes()
        with assert_warns(DeprecationWarning):
            s = arr.tostring()
        assert s == b


class TestDTypeCoercion(_DeprecationTestCase):
    # 2020-02-06 1.19.0
    message = "Converting .* to a dtype .*is deprecated"
    deprecated_types = [
        # The builtin scalar super types:
        np.generic, np.flexible, np.number,
        np.inexact, np.floating, np.complexfloating,
        np.integer, np.unsignedinteger, np.signedinteger,
        # character is a deprecated S1 special case:
        np.character,
    ]

    def test_dtype_coercion(self):
        for scalar_type in self.deprecated_types:
            self.assert_deprecated(np.dtype, args=(scalar_type,))

    def test_array_construction(self):
        for scalar_type in self.deprecated_types:
            self.assert_deprecated(np.array, args=([], scalar_type,))

    def test_not_deprecated(self):
        # All specific types are not deprecated:
        for group in np.sctypes.values():
            for scalar_type in group:
                self.assert_not_deprecated(np.dtype, args=(scalar_type,))

        for scalar_type in [type, dict, list, tuple]:
            # Typical python types are coerced to object currently:
            self.assert_not_deprecated(np.dtype, args=(scalar_type,))


class BuiltInRoundComplexDType(_DeprecationTestCase):
    # 2020-03-31 1.19.0
    deprecated_types = [np.csingle, np.cdouble, np.clongdouble]
    not_deprecated_types = [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    ]

    def test_deprecated(self):
        for scalar_type in self.deprecated_types:
            scalar = scalar_type(0)
            self.assert_deprecated(round, args=(scalar,))
            self.assert_deprecated(round, args=(scalar, 0))
            self.assert_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})
    
    def test_not_deprecated(self):
        for scalar_type in self.not_deprecated_types:
            scalar = scalar_type(0)
            self.assert_not_deprecated(round, args=(scalar,))
            self.assert_not_deprecated(round, args=(scalar, 0))
            self.assert_not_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})
