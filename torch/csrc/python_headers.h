#pragma once

// workaround for Python 2 issue: https://bugs.python.org/issue17120
#pragma push_macro("_XOPEN_SOURCE")
#pragma push_macro("_POSIX_C_SOURCE")
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE

// remove _DEBUG for MSVC
#if defined(_MSC_VER)
#  if defined(_DEBUG) && !defined(Py_DEBUG)
#    undef _DEBUG
#  endif
#endif

#include <Python.h>
#include <structseq.h>

#pragma pop_macro("_XOPEN_SOURCE")
#pragma pop_macro("_POSIX_C_SOURCE")
