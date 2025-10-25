"""Constants for annotations in the mapping.

The constants defined here are used to annotate the mapping tuples in cuda_to_hip_mappings.py.
They are based on
https://github.com/ROCm/HIPIFY/blob/master/src/Statistics.h
and fall in three categories: 1) type of mapping, 2) API of mapping, 3) unsupported
mapping.
"""

import warnings
warnings.warn("hipify's constants.py is no longer used as of version 2.0.0", FutureWarning)

CONV_VERSION = 0,
CONV_INIT = 1
CONV_DEVICE = 2
CONV_MEM = 3
CONV_KERN = 4
CONV_COORD_FUNC = 5
CONV_MATH_FUNC = 6
CONV_DEVICE_FUNC = 7
CONV_SPECIAL_FUNC = 8
CONV_STREAM = 9
CONV_EVENT = 10
CONV_OCCUPANCY = 11
CONV_CONTEXT = 12
CONV_PEER = 13
CONV_MODULE = 14
CONV_CACHE = 15
CONV_EXEC = 16
CONV_ERROR = 17
CONV_DEF = 18
CONV_TEX = 19
CONV_GL = 20
CONV_GRAPHICS = 21
CONV_SURFACE = 22
CONV_JIT = 23
CONV_D3D9 = 24
CONV_D3D10 = 25
CONV_D3D11 = 26
CONV_VDPAU = 27
CONV_EGL = 28
CONV_THREAD = 29
CONV_OTHER = 30
CONV_INCLUDE = 31
CONV_INCLUDE_CUDA_MAIN_H = 32
CONV_TYPE = 33
CONV_LITERAL = 34
CONV_NUMERIC_LITERAL = 35
CONV_LAST = 36

API_DRIVER = 37
API_RUNTIME = 38
API_BLAS = 39
API_SPECIAL = 40
API_RAND = 41
API_LAST = 42
API_FFT = 43
API_RTC = 44
API_ROCTX = 45
API_PYT_EXT = 46

HIP_UNSUPPORTED = 47
API_PYTORCH = 1337
API_CAFFE2 = 1338
API_C10 = 1339
API_ROCMSMI = 1340
