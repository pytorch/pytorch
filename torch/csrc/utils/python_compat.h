#ifndef PYTHON_COMPAT
#define PYTHON_COMPAT

#include <torch/csrc/utils/pythoncapi_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

// PyTorch-only compat functions

PYCAPI_COMPAT_STATIC_INLINE(int)
PyCode_GetNCellvars(PyCodeObject *code)
{
    // gh-26364 added co_ncellvars to Python 3.11.0rc1
    #if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
    return PyTuple_GET_SIZE(code->co_cellvars);
    #else
    return code->co_ncellvars;
    #endif
}

PYCAPI_COMPAT_STATIC_INLINE(int)
PyCode_GetNFreevars(PyCodeObject *code)
{
    // gh-26364 added co_nfreevars to Python 3.11.0rc1
    #if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
    return PyTuple_GET_SIZE(code->co_freevars);
    #else
    return code->co_nfreevars;
    #endif
}


#ifdef __cplusplus
}
#endif
#endif  // PYTHON_COMPAT
