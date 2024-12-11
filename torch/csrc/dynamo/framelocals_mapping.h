#pragma once

#include <torch/csrc/utils/python_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

#if IS_PYTHON_3_11_PLUS
typedef struct _PyInterpreterFrame _PyInterpreterFrame;
PyObject* get_framelocals_mapping(_PyInterpreterFrame* frame);
#else
PyObject* get_framelocals_mapping(PyFrameObject* frame);
#endif

#ifdef __cplusplus
} // extern "C"
#endif
