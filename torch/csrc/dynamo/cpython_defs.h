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

#ifdef STATIC_LIBPYTHON
// When libpython is statically linked, CPython's internal functions are
// available at link time. Redirect THP_ wrappers to CPython originals.
extern void _PyFrame_ClearExceptCode(_PyInterpreterFrame* frame);
extern void _PyThreadState_PopFrame(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame);
#if IS_PYTHON_3_12_PLUS
extern _PyInterpreterFrame* _PyThreadState_PushFrame(
    PyThreadState* tstate,
    size_t size);
#else
extern _PyInterpreterFrame* _PyThreadState_BumpFramePointerSlow(
    PyThreadState* tstate,
    size_t size);
#endif

#define THP_PyFrame_Clear _PyFrame_ClearExceptCode
#define THP_PyThreadState_PopFrame _PyThreadState_PopFrame
#if IS_PYTHON_3_12_PLUS
#define THP_PyThreadState_BumpFramePointerSlow _PyThreadState_PushFrame
#else
#define THP_PyThreadState_BumpFramePointerSlow _PyThreadState_BumpFramePointerSlow
#endif
#else // !STATIC_LIBPYTHON
void THP_PyFrame_Clear(_PyInterpreterFrame* frame);

_PyInterpreterFrame* THP_PyThreadState_BumpFramePointerSlow(
    PyThreadState* tstate,
    size_t size);

void THP_PyThreadState_PopFrame(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame);
#endif // STATIC_LIBPYTHON

#endif

// pointers to _PyOpcode_Caches for C++
#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t* THP_PyOpcode_Caches;
extern int THP_PyOpcode_Caches_size;
void init_THPCaches();

#ifdef __cplusplus
} // extern "C"
#endif
