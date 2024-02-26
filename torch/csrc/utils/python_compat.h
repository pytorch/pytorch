#ifndef PYTHON_COMPAT
#define PYTHON_COMPAT

#include <torch/csrc/utils/pythoncapi_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

// PyTorch-only compat functions

#define IS_PYTHON_3_11_PLUS PY_VERSION_HEX >= 0x030B00C1
#define IS_PYTHON_3_12_PLUS PY_VERSION_HEX >= 0x030C0000

PYCAPI_COMPAT_STATIC_INLINE(int)
PyCode_GetNCellvars(PyCodeObject* code) {
// gh-26364 added co_ncellvars to Python 3.11.0rc1
#if IS_PYTHON_3_11_PLUS
  return code->co_ncellvars;
#else
  return PyTuple_GET_SIZE(code->co_cellvars);
#endif
}

PYCAPI_COMPAT_STATIC_INLINE(int)
PyCode_GetNFreevars(PyCodeObject* code) {
// gh-26364 added co_nfreevars to Python 3.11.0rc1
#if IS_PYTHON_3_11_PLUS
  return code->co_nfreevars;
#else
  return PyTuple_GET_SIZE(code->co_freevars);
#endif
}

#ifdef __cplusplus
}
#endif
#endif // PYTHON_COMPAT
