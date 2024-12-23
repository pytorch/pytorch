import os
from ctypes import c_long, sizeof
from functools import reduce
from typing import Tuple as tTuple, Type
from warnings import warn

from sympy.external import import_module

from .pythonmpq import PythonMPQ

from .ntheory import (
    bit_scan1 as python_bit_scan1,
    bit_scan0 as python_bit_scan0,
    remove as python_remove,
    factorial as python_factorial,
    sqrt as python_sqrt,
    sqrtrem as python_sqrtrem,
    gcd as python_gcd,
    lcm as python_lcm,
    gcdext as python_gcdext,
    is_square as python_is_square,
    invert as python_invert,
    legendre as python_legendre,
    jacobi as python_jacobi,
    kronecker as python_kronecker,
    iroot as python_iroot,
    is_fermat_prp as python_is_fermat_prp,
    is_euler_prp as python_is_euler_prp,
    is_strong_prp as python_is_strong_prp,
    is_fibonacci_prp as python_is_fibonacci_prp,
    is_lucas_prp as python_is_lucas_prp,
    is_selfridge_prp as python_is_selfridge_prp,
    is_strong_lucas_prp as python_is_strong_lucas_prp,
    is_strong_selfridge_prp as python_is_strong_selfridge_prp,
    is_bpsw_prp as python_is_bpsw_prp,
    is_strong_bpsw_prp as python_is_strong_bpsw_prp,
)


__all__ = [
    # GROUND_TYPES is either 'gmpy' or 'python' depending on which is used. If
    # gmpy is installed then it will be used unless the environment variable
    # SYMPY_GROUND_TYPES is set to something other than 'auto', 'gmpy', or
    # 'gmpy2'.
    'GROUND_TYPES',

    # If HAS_GMPY is 0, no supported version of gmpy is available. Otherwise,
    # HAS_GMPY will be 2 for gmpy2 if GROUND_TYPES is 'gmpy'. It used to be
    # possible for HAS_GMPY to be 1 for gmpy but gmpy is no longer supported.
    'HAS_GMPY',

    # SYMPY_INTS is a tuple containing the base types for valid integer types.
    # This is either (int,) or (int, type(mpz(0))) depending on GROUND_TYPES.
    'SYMPY_INTS',

    # MPQ is either gmpy.mpq or the Python equivalent from
    # sympy.external.pythonmpq
    'MPQ',

    # MPZ is either gmpy.mpz or int.
    'MPZ',

    'bit_scan1',
    'bit_scan0',
    'remove',
    'factorial',
    'sqrt',
    'is_square',
    'sqrtrem',
    'gcd',
    'lcm',
    'gcdext',
    'invert',
    'legendre',
    'jacobi',
    'kronecker',
    'iroot',
    'is_fermat_prp',
    'is_euler_prp',
    'is_strong_prp',
    'is_fibonacci_prp',
    'is_lucas_prp',
    'is_selfridge_prp',
    'is_strong_lucas_prp',
    'is_strong_selfridge_prp',
    'is_bpsw_prp',
    'is_strong_bpsw_prp',
]


#
# Tested python-flint version. Future versions might work but we will only use
# them if explicitly requested by SYMPY_GROUND_TYPES=flint.
#
_PYTHON_FLINT_VERSION_NEEDED = ["0.6", "0.7", "0.8", "0.9"]


def _flint_version_okay(flint_version):
    major, minor = flint_version.split('.')[:2]
    flint_ver = f'{major}.{minor}'
    return flint_ver in _PYTHON_FLINT_VERSION_NEEDED

#
# We will only use gmpy2 >= 2.0.0
#
_GMPY2_MIN_VERSION = '2.0.0'


def _get_flint(sympy_ground_types):
    if sympy_ground_types not in ('auto', 'flint'):
        return None

    try:
        import flint
        # Earlier versions of python-flint may not have __version__.
        from flint import __version__ as _flint_version
    except ImportError:
        if sympy_ground_types == 'flint':
            warn("SYMPY_GROUND_TYPES was set to flint but python-flint is not "
                 "installed. Falling back to other ground types.")
        return None

    if _flint_version_okay(_flint_version):
        return flint
    elif sympy_ground_types == 'auto':
        return None
    else:
        warn(f"Using python-flint {_flint_version} because SYMPY_GROUND_TYPES "
             f"is set to flint but this version of SymPy is only tested "
             f"with python-flint versions {_PYTHON_FLINT_VERSION_NEEDED}.")
        return flint


def _get_gmpy2(sympy_ground_types):
    if sympy_ground_types not in ('auto', 'gmpy', 'gmpy2'):
        return None

    gmpy = import_module('gmpy2', min_module_version=_GMPY2_MIN_VERSION,
            module_version_attr='version', module_version_attr_call_args=())

    if sympy_ground_types != 'auto' and gmpy is None:
        warn("gmpy2 library is not installed, switching to 'python' ground types")

    return gmpy


#
# SYMPY_GROUND_TYPES can be flint, gmpy, gmpy2, python or auto (default)
#
_SYMPY_GROUND_TYPES = os.environ.get('SYMPY_GROUND_TYPES', 'auto').lower()
_flint = None
_gmpy = None

#
# First handle auto-detection of flint/gmpy2. We will prefer flint if available
# or otherwise gmpy2 if available and then lastly the python types.
#
if _SYMPY_GROUND_TYPES in ('auto', 'flint'):
    _flint = _get_flint(_SYMPY_GROUND_TYPES)
    if _flint is not None:
        _SYMPY_GROUND_TYPES = 'flint'
    else:
        _SYMPY_GROUND_TYPES = 'auto'

if _SYMPY_GROUND_TYPES in ('auto', 'gmpy', 'gmpy2'):
    _gmpy = _get_gmpy2(_SYMPY_GROUND_TYPES)
    if _gmpy is not None:
        _SYMPY_GROUND_TYPES = 'gmpy'
    else:
        _SYMPY_GROUND_TYPES = 'python'

if _SYMPY_GROUND_TYPES not in ('flint', 'gmpy', 'python'):
    warn("SYMPY_GROUND_TYPES environment variable unrecognised. "
         "Should be 'auto', 'flint', 'gmpy', 'gmpy2' or 'python'.")
    _SYMPY_GROUND_TYPES = 'python'

#
# At this point _SYMPY_GROUND_TYPES is either flint, gmpy or python. The blocks
# below define the values exported by this module in each case.
#

#
# In gmpy2 and flint, there are functions that take a long (or unsigned long)
# argument. That is, it is not possible to input a value larger than that.
#
LONG_MAX = (1 << (8*sizeof(c_long) - 1)) - 1

#
# Type checkers are confused by what SYMPY_INTS is. There may be a better type
# hint for this like Type[Integral] or something.
#
SYMPY_INTS: tTuple[Type, ...]

if _SYMPY_GROUND_TYPES == 'gmpy':

    assert _gmpy is not None

    flint = None
    gmpy = _gmpy

    HAS_GMPY = 2
    GROUND_TYPES = 'gmpy'
    SYMPY_INTS = (int, type(gmpy.mpz(0)))
    MPZ = gmpy.mpz
    MPQ = gmpy.mpq

    bit_scan1 = gmpy.bit_scan1
    bit_scan0 = gmpy.bit_scan0
    remove = gmpy.remove
    factorial = gmpy.fac
    sqrt = gmpy.isqrt
    is_square = gmpy.is_square
    sqrtrem = gmpy.isqrt_rem
    gcd = gmpy.gcd
    lcm = gmpy.lcm
    gcdext = gmpy.gcdext
    invert = gmpy.invert
    legendre = gmpy.legendre
    jacobi = gmpy.jacobi
    kronecker = gmpy.kronecker

    def iroot(x, n):
        # In the latest gmpy2, the threshold for n is ULONG_MAX,
        # but adjust to the older one.
        if n <= LONG_MAX:
            return gmpy.iroot(x, n)
        return python_iroot(x, n)

    is_fermat_prp = gmpy.is_fermat_prp
    is_euler_prp = gmpy.is_euler_prp
    is_strong_prp = gmpy.is_strong_prp
    is_fibonacci_prp = gmpy.is_fibonacci_prp
    is_lucas_prp = gmpy.is_lucas_prp
    is_selfridge_prp = gmpy.is_selfridge_prp
    is_strong_lucas_prp = gmpy.is_strong_lucas_prp
    is_strong_selfridge_prp = gmpy.is_strong_selfridge_prp
    is_bpsw_prp = gmpy.is_bpsw_prp
    is_strong_bpsw_prp = gmpy.is_strong_bpsw_prp

elif _SYMPY_GROUND_TYPES == 'flint':

    assert _flint is not None

    flint = _flint
    gmpy = None

    HAS_GMPY = 0
    GROUND_TYPES = 'flint'
    SYMPY_INTS = (int, flint.fmpz) # type: ignore
    MPZ = flint.fmpz # type: ignore
    MPQ = flint.fmpq # type: ignore

    bit_scan1 = python_bit_scan1
    bit_scan0 = python_bit_scan0
    remove = python_remove
    factorial = python_factorial

    def sqrt(x):
        return flint.fmpz(x).isqrt()

    def is_square(x):
        if x < 0:
            return False
        return flint.fmpz(x).sqrtrem()[1] == 0

    def sqrtrem(x):
        return flint.fmpz(x).sqrtrem()

    def gcd(*args):
        return reduce(flint.fmpz.gcd, args, flint.fmpz(0))

    def lcm(*args):
        return reduce(flint.fmpz.lcm, args, flint.fmpz(1))

    gcdext = python_gcdext
    invert = python_invert
    legendre = python_legendre

    def jacobi(x, y):
        if y <= 0 or not y % 2:
            raise ValueError("y should be an odd positive integer")
        return flint.fmpz(x).jacobi(y)

    kronecker = python_kronecker

    def iroot(x, n):
        if n <= LONG_MAX:
            y = flint.fmpz(x).root(n)
            return y, y**n == x
        return python_iroot(x, n)

    is_fermat_prp = python_is_fermat_prp
    is_euler_prp = python_is_euler_prp
    is_strong_prp = python_is_strong_prp
    is_fibonacci_prp = python_is_fibonacci_prp
    is_lucas_prp = python_is_lucas_prp
    is_selfridge_prp = python_is_selfridge_prp
    is_strong_lucas_prp = python_is_strong_lucas_prp
    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    is_bpsw_prp = python_is_bpsw_prp
    is_strong_bpsw_prp = python_is_strong_bpsw_prp

elif _SYMPY_GROUND_TYPES == 'python':

    flint = None
    gmpy = None

    HAS_GMPY = 0
    GROUND_TYPES = 'python'
    SYMPY_INTS = (int,)
    MPZ = int
    MPQ = PythonMPQ

    bit_scan1 = python_bit_scan1
    bit_scan0 = python_bit_scan0
    remove = python_remove
    factorial = python_factorial
    sqrt = python_sqrt
    is_square = python_is_square
    sqrtrem = python_sqrtrem
    gcd = python_gcd
    lcm = python_lcm
    gcdext = python_gcdext
    invert = python_invert
    legendre = python_legendre
    jacobi = python_jacobi
    kronecker = python_kronecker
    iroot = python_iroot
    is_fermat_prp = python_is_fermat_prp
    is_euler_prp = python_is_euler_prp
    is_strong_prp = python_is_strong_prp
    is_fibonacci_prp = python_is_fibonacci_prp
    is_lucas_prp = python_is_lucas_prp
    is_selfridge_prp = python_is_selfridge_prp
    is_strong_lucas_prp = python_is_strong_lucas_prp
    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    is_bpsw_prp = python_is_bpsw_prp
    is_strong_bpsw_prp = python_is_strong_bpsw_prp

else:
    assert False
