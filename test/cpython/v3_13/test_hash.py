# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_hash.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# test the invariant that
#   iff a==b then hash(a)==hash(b)
#
# Also test that hash implementations are inherited as expected

import datetime
import os
import sys
import unittest
from test.support.script_helper import assert_python_ok
from collections.abc import Hashable

IS_64BIT = sys.maxsize > 2**32

def lcg(x, length=16):
    """Linear congruential generator"""
    if x == 0:
        return bytes(length)
    out = bytearray(length)
    for i in range(length):
        x = (214013 * x + 2531011) & 0x7fffffff
        out[i] = (x >> 16) & 0xff
    return bytes(out)

def pysiphash(uint64):
    """Convert SipHash24 output to Py_hash_t
    """
    assert 0 <= uint64 < (1 << 64)
    # simple unsigned to signed int64
    if uint64 > (1 << 63) - 1:
        int64 = uint64 - (1 << 64)
    else:
        int64 = uint64
    # mangle uint64 to uint32
    uint32 = (uint64 ^ uint64 >> 32) & 0xffffffff
    # simple unsigned to signed int32
    if uint32 > (1 << 31) - 1:
        int32 = uint32 - (1 << 32)
    else:
        int32 = uint32
    return int32, int64

def skip_unless_internalhash(test):
    """Skip decorator for tests that depend on SipHash24 or FNV"""
    ok = sys.hash_info.algorithm in {"fnv", "siphash13", "siphash24"}
    msg = "Requires SipHash13, SipHash24 or FNV"
    return test if ok else unittest.skip(msg)(test)


class HashEqualityTestCase(CPythonTestCase):

    def same_hash(self, *objlist):
        # Hash each object given and fail if
        # the hash values are not all the same.
        hashed = list(map(hash, objlist))
        for h in hashed[1:]:
            if h != hashed[0]:
                self.fail("hashed values differ: %r" % (objlist,))

    def test_numeric_literals(self):
        self.same_hash(1, 1, 1.0, 1.0+0.0j)
        self.same_hash(0, 0.0, 0.0+0.0j)
        self.same_hash(-1, -1.0, -1.0+0.0j)
        self.same_hash(-2, -2.0, -2.0+0.0j)

    def test_coerced_integers(self):
        self.same_hash(int(1), int(1), float(1), complex(1),
                       int('1'), float('1.0'))
        self.same_hash(int(-2**31), float(-2**31))
        self.same_hash(int(1-2**31), float(1-2**31))
        self.same_hash(int(2**31-1), float(2**31-1))
        # for 64-bit platforms
        self.same_hash(int(2**31), float(2**31))
        self.same_hash(int(-2**63), float(-2**63))
        self.same_hash(int(2**63), float(2**63))

    def test_coerced_floats(self):
        self.same_hash(int(1.23e300), float(1.23e300))
        self.same_hash(float(0.5), complex(0.5, 0.0))

    def test_unaligned_buffers(self):
        # The hash function for bytes-like objects shouldn't have
        # alignment-dependent results (example in issue #16427).
        b = b"123456789abcdefghijklmnopqrstuvwxyz" * 128
        for i in range(16):
            for j in range(16):
                aligned = b[i:128+j]
                unaligned = memoryview(b)[i:128+j]
                self.assertEqual(hash(aligned), hash(unaligned))


_default_hash = object.__hash__
class DefaultHash(object): pass

_FIXED_HASH_VALUE = 42
class FixedHash(object):
    def __hash__(self):
        return _FIXED_HASH_VALUE

class OnlyEquality(object):
    def __eq__(self, other):
        return self is other

class OnlyInequality(object):
    def __ne__(self, other):
        return self is not other

class InheritedHashWithEquality(FixedHash, OnlyEquality): pass
class InheritedHashWithInequality(FixedHash, OnlyInequality): pass

class NoHash(object):
    __hash__ = None

class HashInheritanceTestCase(CPythonTestCase):
    default_expected = [object(),
                        DefaultHash(),
                        OnlyInequality(),
                       ]
    fixed_expected = [FixedHash(),
                      InheritedHashWithEquality(),
                      InheritedHashWithInequality(),
                      ]
    error_expected = [NoHash(),
                      OnlyEquality(),
                      ]

    def test_default_hash(self):
        for obj in self.default_expected:
            self.assertEqual(hash(obj), _default_hash(obj))

    def test_fixed_hash(self):
        for obj in self.fixed_expected:
            self.assertEqual(hash(obj), _FIXED_HASH_VALUE)

    def test_error_hash(self):
        for obj in self.error_expected:
            self.assertRaises(TypeError, hash, obj)

    def test_hashable(self):
        objects = (self.default_expected +
                   self.fixed_expected)
        for obj in objects:
            self.assertIsInstance(obj, Hashable)

    def test_not_hashable(self):
        for obj in self.error_expected:
            self.assertNotIsInstance(obj, Hashable)


# Issue #4701: Check that some builtin types are correctly hashable
class DefaultIterSeq(object):
    seq = range(10)
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, index):
        return self.seq[index]

class HashBuiltinsTestCase(CPythonTestCase):
    hashes_to_check = [enumerate(range(10)),
                       iter(DefaultIterSeq()),
                       iter(lambda: 0, 0),
                      ]

    def test_hashes(self):
        _default_hash = object.__hash__
        for obj in self.hashes_to_check:
            self.assertEqual(hash(obj), _default_hash(obj))

class HashRandomizationTests:

    # Each subclass should define a field "repr_", containing the repr() of
    # an object to be tested

    def get_hash_command(self, repr_):
        return 'print(hash(eval(%a)))' % repr_

    def get_hash(self, repr_, seed=None):
        env = os.environ.copy()
        env['__cleanenv'] = True  # signal to assert_python not to do a copy
                                  # of os.environ on its own
        if seed is not None:
            env['PYTHONHASHSEED'] = str(seed)
        else:
            env.pop('PYTHONHASHSEED', None)
        out = assert_python_ok(
            '-c', self.get_hash_command(repr_),
            **env)
        stdout = out[1].strip()
        return int(stdout)

    def test_randomized_hash(self):
        # two runs should return different hashes
        run1 = self.get_hash(self.repr_, seed='random')
        run2 = self.get_hash(self.repr_, seed='random')
        self.assertNotEqual(run1, run2)

class StringlikeHashRandomizationTests(HashRandomizationTests):
    repr_ = None
    repr_long = None

    # 32bit little, 64bit little, 32bit big, 64bit big
    known_hashes = {
        'djba33x': [ # only used for small strings
            # seed 0, 'abc'
            [193485960, 193485960,  193485960, 193485960],
            # seed 42, 'abc'
            [-678966196, 573763426263223372, -820489388, -4282905804826039665],
            ],
        'siphash13': [
            # NOTE: PyUCS2 layout depends on endianness
            # seed 0, 'abc'
            [69611762, -4594863902769663758, 69611762, -4594863902769663758],
            # seed 42, 'abc'
            [-975800855, 3869580338025362921, -975800855, 3869580338025362921],
            # seed 42, 'abcdefghijk'
            [-595844228, 7764564197781545852, -595844228, 7764564197781545852],
            # seed 0, 'äú∑ℇ'
            [-1093288643, -2810468059467891395, -1041341092, 4925090034378237276],
            # seed 42, 'äú∑ℇ'
            [-585999602, -2845126246016066802, -817336969, -2219421378907968137],
        ],
        'siphash24': [
            # NOTE: PyUCS2 layout depends on endianness
            # seed 0, 'abc'
            [1198583518, 4596069200710135518, 1198583518, 4596069200710135518],
            # seed 42, 'abc'
            [273876886, -4501618152524544106, 273876886, -4501618152524544106],
            # seed 42, 'abcdefghijk'
            [-1745215313, 4436719588892876975, -1745215313, 4436719588892876975],
            # seed 0, 'äú∑ℇ'
            [493570806, 5749986484189612790, -1006381564, -5915111450199468540],
            # seed 42, 'äú∑ℇ'
            [-1677110816, -2947981342227738144, -1860207793, -4296699217652516017],
        ],
        'fnv': [
            # seed 0, 'abc'
            [-1600925533, 1453079729188098211, -1600925533,
             1453079729188098211],
            # seed 42, 'abc'
            [-206076799, -4410911502303878509, -1024014457,
             -3570150969479994130],
            # seed 42, 'abcdefghijk'
            [811136751, -5046230049376118746, -77208053 ,
             -4779029615281019666],
            # seed 0, 'äú∑ℇ'
            [44402817, 8998297579845987431, -1956240331,
             -782697888614047887],
            # seed 42, 'äú∑ℇ'
            [-283066365, -4576729883824601543, -271871407,
             -3927695501187247084],
        ]
    }

    def get_expected_hash(self, position, length):
        if length < sys.hash_info.cutoff:
            algorithm = "djba33x"
        else:
            algorithm = sys.hash_info.algorithm
        if sys.byteorder == 'little':
            platform = 1 if IS_64BIT else 0
        else:
            assert(sys.byteorder == 'big')
            platform = 3 if IS_64BIT else 2
        return self.known_hashes[algorithm][position][platform]

    def test_null_hash(self):
        # PYTHONHASHSEED=0 disables the randomized hash
        known_hash_of_obj = self.get_expected_hash(0, 3)

        # Randomization is enabled by default:
        self.assertNotEqual(self.get_hash(self.repr_), known_hash_of_obj)

        # It can also be disabled by setting the seed to 0:
        self.assertEqual(self.get_hash(self.repr_, seed=0), known_hash_of_obj)

    @skip_unless_internalhash
    def test_fixed_hash(self):
        # test a fixed seed for the randomized hash
        # Note that all types share the same values:
        h = self.get_expected_hash(1, 3)
        self.assertEqual(self.get_hash(self.repr_, seed=42), h)

    @skip_unless_internalhash
    def test_long_fixed_hash(self):
        if self.repr_long is None:
            return
        h = self.get_expected_hash(2, 11)
        self.assertEqual(self.get_hash(self.repr_long, seed=42), h)


class StrHashRandomizationTests(StringlikeHashRandomizationTests,
                                CPythonTestCase):
    repr_ = repr('abc')
    repr_long = repr('abcdefghijk')
    repr_ucs2 = repr('äú∑ℇ')

    @skip_unless_internalhash
    def test_empty_string(self):
        self.assertEqual(hash(""), 0)

    @skip_unless_internalhash
    def test_ucs2_string(self):
        h = self.get_expected_hash(3, 6)
        self.assertEqual(self.get_hash(self.repr_ucs2, seed=0), h)
        h = self.get_expected_hash(4, 6)
        self.assertEqual(self.get_hash(self.repr_ucs2, seed=42), h)

class BytesHashRandomizationTests(StringlikeHashRandomizationTests,
                                  CPythonTestCase):
    repr_ = repr(b'abc')
    repr_long = repr(b'abcdefghijk')

    @skip_unless_internalhash
    def test_empty_string(self):
        self.assertEqual(hash(b""), 0)

class MemoryviewHashRandomizationTests(StringlikeHashRandomizationTests,
                                       CPythonTestCase):
    repr_ = "memoryview(b'abc')"
    repr_long = "memoryview(b'abcdefghijk')"

    @skip_unless_internalhash
    def test_empty_string(self):
        self.assertEqual(hash(memoryview(b"")), 0)

class DatetimeTests(HashRandomizationTests):
    def get_hash_command(self, repr_):
        return 'import datetime; print(hash(%s))' % repr_

class DatetimeDateTests(DatetimeTests, CPythonTestCase):
    repr_ = repr(datetime.date(1066, 10, 14))

class DatetimeDatetimeTests(DatetimeTests, CPythonTestCase):
    repr_ = repr(datetime.datetime(1, 2, 3, 4, 5, 6, 7))

class DatetimeTimeTests(DatetimeTests, CPythonTestCase):
    repr_ = repr(datetime.time(0))


class HashDistributionTestCase(CPythonTestCase):

    def test_hash_distribution(self):
        # check for hash collision
        base = "abcdefghabcdefg"
        for i in range(1, len(base)):
            prefix = base[:i]
            with self.subTest(prefix=prefix):
                s15 = set()
                s255 = set()
                for c in range(256):
                    h = hash(prefix + chr(c))
                    s15.add(h & 0xf)
                    s255.add(h & 0xff)
                # SipHash24 distribution depends on key, usually > 60%
                self.assertGreater(len(s15), 8, prefix)
                self.assertGreater(len(s255), 128, prefix)

if __name__ == "__main__":
    run_tests()
