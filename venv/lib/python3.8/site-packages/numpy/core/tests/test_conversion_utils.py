"""
Tests for numpy/core/src/multiarray/conversion_utils.c
"""
import re

import pytest

import numpy as np
import numpy.core._multiarray_tests as mt


class StringConverterTestCase:
    allow_bytes = True
    case_insensitive = True
    exact_match = False

    def _check_value_error(self, val):
        pattern = r'\(got {}\)'.format(re.escape(repr(val)))
        with pytest.raises(ValueError, match=pattern) as exc:
            self.conv(val)

    def _check(self, val, expected):
        assert self.conv(val) == expected

        if self.allow_bytes:
            assert self.conv(val.encode('ascii')) == expected
        else:
            with pytest.raises(TypeError):
                self.conv(val.encode('ascii'))

        if len(val) != 1:
            if self.exact_match:
                self._check_value_error(val[:1])
                self._check_value_error(val + '\0')
            else:
                assert self.conv(val[:1]) == expected

        if self.case_insensitive:
            if val != val.lower():
                assert self.conv(val.lower()) == expected
            if val != val.upper():
                assert self.conv(val.upper()) == expected
        else:
            if val != val.lower():
                self._check_value_error(val.lower())
            if val != val.upper():
                self._check_value_error(val.upper())

    def test_wrong_type(self):
        # common cases which apply to all the below
        with pytest.raises(TypeError):
            self.conv({})
        with pytest.raises(TypeError):
            self.conv([])

    def test_wrong_value(self):
        # nonsense strings
        self._check_value_error('')
        self._check_value_error('\N{greek small letter pi}')

        if self.allow_bytes:
            self._check_value_error(b'')
            # bytes which can't be converted to strings via utf8
            self._check_value_error(b"\xFF")
        if self.exact_match:
            self._check_value_error("there's no way this is supported")


class TestByteorderConverter(StringConverterTestCase):
    """ Tests of PyArray_ByteorderConverter """
    conv = mt.run_byteorder_converter
    def test_valid(self):
        for s in ['big', '>']:
            self._check(s, 'NPY_BIG')
        for s in ['little', '<']:
            self._check(s, 'NPY_LITTLE')
        for s in ['native', '=']:
            self._check(s, 'NPY_NATIVE')
        for s in ['ignore', '|']:
            self._check(s, 'NPY_IGNORE')
        for s in ['swap']:
            self._check(s, 'NPY_SWAP')


class TestSortkindConverter(StringConverterTestCase):
    """ Tests of PyArray_SortkindConverter """
    conv = mt.run_sortkind_converter
    def test_valid(self):
        self._check('quick', 'NPY_QUICKSORT')
        self._check('heap', 'NPY_HEAPSORT')
        self._check('merge', 'NPY_STABLESORT')  # alias
        self._check('stable', 'NPY_STABLESORT')


class TestSelectkindConverter(StringConverterTestCase):
    """ Tests of PyArray_SelectkindConverter """
    conv = mt.run_selectkind_converter
    case_insensitive = False
    exact_match = True

    def test_valid(self):
        self._check('introselect', 'NPY_INTROSELECT')


class TestSearchsideConverter(StringConverterTestCase):
    """ Tests of PyArray_SearchsideConverter """
    conv = mt.run_searchside_converter
    def test_valid(self):
        self._check('left', 'NPY_SEARCHLEFT')
        self._check('right', 'NPY_SEARCHRIGHT')


class TestOrderConverter(StringConverterTestCase):
    """ Tests of PyArray_OrderConverter """
    conv = mt.run_order_converter
    def test_valid(self):
        self._check('c', 'NPY_CORDER')
        self._check('f', 'NPY_FORTRANORDER')
        self._check('a', 'NPY_ANYORDER')
        self._check('k', 'NPY_KEEPORDER')

    def test_flatten_invalid_order(self):
        # invalid after gh-14596
        with pytest.raises(ValueError):
            self.conv('Z')
        for order in [False, True, 0, 8]:
            with pytest.raises(TypeError):
                self.conv(order)


class TestClipmodeConverter(StringConverterTestCase):
    """ Tests of PyArray_ClipmodeConverter """
    conv = mt.run_clipmode_converter
    def test_valid(self):
        self._check('clip', 'NPY_CLIP')
        self._check('wrap', 'NPY_WRAP')
        self._check('raise', 'NPY_RAISE')

        # integer values allowed here
        assert self.conv(np.CLIP) == 'NPY_CLIP'
        assert self.conv(np.WRAP) == 'NPY_WRAP'
        assert self.conv(np.RAISE) == 'NPY_RAISE'


class TestCastingConverter(StringConverterTestCase):
    """ Tests of PyArray_CastingConverter """
    conv = mt.run_casting_converter
    case_insensitive = False
    exact_match = True

    def test_valid(self):
        self._check("no", "NPY_NO_CASTING")
        self._check("equiv", "NPY_EQUIV_CASTING")
        self._check("safe", "NPY_SAFE_CASTING")
        self._check("same_kind", "NPY_SAME_KIND_CASTING")
        self._check("unsafe", "NPY_UNSAFE_CASTING")
