#pragma once

#include <torch/csrc/utils/python_compat.h>

#if IS_PYTHON_3_14_PLUS

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Use a void* to avoid exposing the internal _PyStackRef union on this
// translation unit
PyObject* THP_PyStackRef_AsPyObjectBorrow(void* stackref);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // IS_PYTHON_3_14_PLUS
