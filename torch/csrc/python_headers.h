#pragma once
#include <math.h>
// workaround for Python 2 issue: https://bugs.python.org/issue17120
// NOTE: It looks like this affects Python 3 as well.
#pragma push_macro("_XOPEN_SOURCE")
#pragma push_macro("_POSIX_C_SOURCE")
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE

#include <Python.h>
#include <structseq.h>

#pragma pop_macro("_XOPEN_SOURCE")
#pragma pop_macro("_POSIX_C_SOURCE")

#if PY_MAJOR_VERSION < 3
#error "Python 2 has reached end-of-life and is no longer supported by PyTorch."
#endif

// Check for Python APIs that interact with long types on Windows.
// As those usages will lead to potential errors since `sizeof(long) == 4`.
#ifdef _WIN32
#undef PyLong_FromLong
#undef PyLong_AsLong
#undef PyLong_FromUnsignedLong
#undef PyLong_AsUnsignedLong

#define PyLong_FromLong(...)         static_assert(false, "Usage of PyLong_FromLong may cause problems on Windows. Please use THPUtils_packInt64 in torch/csrc/utils/python_numbers.h instead.")
#define PyLong_AsLong(...)           static_assert(false, "Usage of PyLong_AsLong may cause problems on Windows. Please use THPUtils_unpackLong in torch/csrc/utils/python_numbers.h instead.")
#define PyLong_FromUnsignedLong(...) static_assert(false, "Usage of PyLong_FromUnsignedLong may cause problems on Windows. Please use THPUtils_packUInt64 in torch/csrc/utils/python_numbers.h instead.")
#define PyLong_AsUnsignedLong(...)   static_assert(false, "Usage of PyLong_AsUnsignedLong may cause problems on Windows. Please use THPUtils_unpackUInt64 in torch/csrc/utils/python_numbers.h instead.")
#endif
