#pragma once
// workaround for https://github.com/python/cpython/pull/23326
#include <cmath>
#include <complex>
// workaround for Python 2 issue: https://bugs.python.org/issue17120
// NOTE: It looks like this affects Python 3 as well.
#pragma push_macro("_XOPEN_SOURCE")
#pragma push_macro("_POSIX_C_SOURCE")
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE

#include <Python.h>
#include <frameobject.h>
#include <structseq.h>

#pragma pop_macro("_XOPEN_SOURCE")
#pragma pop_macro("_POSIX_C_SOURCE")

#ifdef copysign
#undef copysign
#endif

#if PY_MAJOR_VERSION < 3
#error "Python 2 has reached end-of-life and is no longer supported by PyTorch."
#endif
