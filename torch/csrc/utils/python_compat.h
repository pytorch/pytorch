#ifndef PYTHON_COMPAT
#define PYTHON_COMPAT

#include <torch/csrc/utils/pythoncapi_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

// PyTorch-only compat functions

#define IS_PYTHON_3_11_PLUS PY_VERSION_HEX >= 0x030B00C1
#define IS_PYTHON_3_12_PLUS PY_VERSION_HEX >= 0x030C0000
#define IS_PYTHON_3_13_PLUS PY_VERSION_HEX >= 0x030D0000
#define IS_PYTHON_3_14_PLUS PY_VERSION_HEX >= 0x030E0000

static inline int PyCode_GetNCellvars(PyCodeObject* code) {
// gh-26364 added co_ncellvars to Python 3.11.0rc1
#if IS_PYTHON_3_11_PLUS
  return code->co_ncellvars;
#else
  return PyTuple_GET_SIZE(code->co_cellvars);
#endif
}

static inline int PyCode_GetNFreevars(PyCodeObject* code) {
// gh-26364 added co_nfreevars to Python 3.11.0rc1
#if IS_PYTHON_3_11_PLUS
  return code->co_nfreevars;
#else
  return PyTuple_GET_SIZE(code->co_freevars);
#endif
}

// Provided by CPython but getting the header for them is very hard
#if IS_PYTHON_3_11_PLUS
// NOLINTNEXTLINE(readability-redundant-declaration)
PyAPI_FUNC(void) _PyWeakref_ClearRef(PyWeakReference* self);
#else
extern void _PyWeakref_ClearRef(PyWeakReference* self);
#endif

#ifdef __cplusplus
}
#endif
#endif // PYTHON_COMPAT
