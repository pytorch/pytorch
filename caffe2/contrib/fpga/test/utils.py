from __future__ import absolute_import, division, print_function, unicode_literals
from ctypes import cast, pointer, POINTER, c_float, c_int32


def bfloat_conversion(x_float, nbits):
    shift = 32 - nbits
    bits = cast(pointer(c_float(x_float)), POINTER(c_int32)).contents.value
    # TODO: add rounding
    # bits += (1 << (shift - 1))
    bits = ((bits >> shift) << shift)
    x_bfp = cast(pointer(c_int32(bits)), POINTER(c_float)).contents.value
    return x_bfp
