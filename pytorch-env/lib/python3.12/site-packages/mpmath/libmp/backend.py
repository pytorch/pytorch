import os
import sys

#----------------------------------------------------------------------------#
# Support GMPY for high-speed large integer arithmetic.                      #
#                                                                            #
# To allow an external module to handle arithmetic, we need to make sure     #
# that all high-precision variables are declared of the correct type. MPZ    #
# is the constructor for the high-precision type. It defaults to Python's    #
# long type but can be assinged another type, typically gmpy.mpz.            #
#                                                                            #
# MPZ must be used for the mantissa component of an mpf and must be used     #
# for internal fixed-point operations.                                       #
#                                                                            #
# Side-effects                                                               #
# 1) "is" cannot be used to test for special values. Must use "==".          #
# 2) There are bugs in GMPY prior to v1.02 so we must use v1.03 or later.    #
#----------------------------------------------------------------------------#

# So we can import it from this module
gmpy = None
sage = None
sage_utils = None

if sys.version_info[0] < 3:
    python3 = False
else:
    python3 = True

BACKEND = 'python'

if not python3:
    MPZ = long
    xrange = xrange
    basestring = basestring

    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")
else:
    MPZ = int
    xrange = range
    basestring = str

    import builtins
    exec_ = getattr(builtins, "exec")

# Define constants for calculating hash on Python 3.2.
if sys.version_info >= (3, 2):
    HASH_MODULUS = sys.hash_info.modulus
    if sys.hash_info.width == 32:
        HASH_BITS = 31
    else:
        HASH_BITS = 61
else:
    HASH_MODULUS = None
    HASH_BITS = None

if 'MPMATH_NOGMPY' not in os.environ:
    try:
        try:
            import gmpy2 as gmpy
        except ImportError:
            try:
                import gmpy
            except ImportError:
                raise ImportError
        if gmpy.version() >= '1.03':
            BACKEND = 'gmpy'
            MPZ = gmpy.mpz
    except:
        pass

if ('MPMATH_NOSAGE' not in os.environ and 'SAGE_ROOT' in os.environ or
        'MPMATH_SAGE' in os.environ):
    try:
        import sage.all
        import sage.libs.mpmath.utils as _sage_utils
        sage = sage.all
        sage_utils = _sage_utils
        BACKEND = 'sage'
        MPZ = sage.Integer
    except:
        pass

if 'MPMATH_STRICT' in os.environ:
    STRICT = True
else:
    STRICT = False

MPZ_TYPE = type(MPZ(0))
MPZ_ZERO = MPZ(0)
MPZ_ONE = MPZ(1)
MPZ_TWO = MPZ(2)
MPZ_THREE = MPZ(3)
MPZ_FIVE = MPZ(5)

try:
    if BACKEND == 'python':
        int_types = (int, long)
    else:
        int_types = (int, long, MPZ_TYPE)
except NameError:
    if BACKEND == 'python':
        int_types = (int,)
    else:
        int_types = (int, MPZ_TYPE)
