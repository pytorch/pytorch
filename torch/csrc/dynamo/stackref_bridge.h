#pragma once

#include <torch/csrc/utils/python_compat.h>

#if IS_PYTHON_3_14_PLUS && defined(_WIN32)

#ifdef __cplusplus
extern "C" {
#endif

// Use a void* to avoid exposing the internal _PyStackRef union on this
// translation unit
PyObject* Torch_PyStackRef_AsPyObjectBorrow(void* stackref);

#ifdef __cplusplus
}
#endif

#endif
