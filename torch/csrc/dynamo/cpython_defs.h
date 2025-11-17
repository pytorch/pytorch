#pragma once

#include <torch/csrc/utils/python_compat.h>

// Functions that need to be copied from the CPython source
// should go in cpython_defs.c. Copying is required when, e.g.,
// we need to call internal CPython functions that are not exposed.

#if IS_PYTHON_3_11_PLUS

typedef struct _PyInterpreterFrame _PyInterpreterFrame;

PyFunctionObject* _PyFunction_CopyWithNewCode(
    PyFunctionObject* o,
    PyCodeObject* code);

void THP_PyFrame_Clear(_PyInterpreterFrame* frame);

_PyInterpreterFrame* THP_PyThreadState_BumpFramePointerSlow(
    PyThreadState* tstate,
    size_t size);

void THP_PyThreadState_PopFrame(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame);

#endif

// pointers to _PyOpcode_Caches for C++
#ifdef __cplusplus

extern "C" const uint8_t* THP_PyOpcode_Caches;
extern "C" const int THP_PyOpcode_Caches_size;

#else

extern const uint8_t* THP_PyOpcode_Caches;
extern const int THP_PyOpcode_Caches_size;

#endif
