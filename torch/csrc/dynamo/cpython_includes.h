#pragma once

#include <torch/csrc/utils/python_compat.h>

// Problem in CPython includes when mixing core and non-core build
// The fix was not backported to 3.12 so this is needed here
// https://github.com/python/cpython/issues/105268
#if IS_PYTHON_3_12_PLUS
#undef _PyGC_FINALIZED
#endif

// see https://bugs.python.org/issue35886
#define Py_BUILD_CORE

#ifndef __cplusplus
// C-only headers
#include <internal/pycore_pystate.h>

#endif // __cplusplus

#if IS_PYTHON_3_11_PLUS
#include <internal/pycore_frame.h>

#include <torch/csrc/dynamo/stackref_bridge.h>
#if IS_PYTHON_3_14_PLUS && !defined(_WIN32)
#include <internal/pycore_code.h>
#include <internal/pycore_genobject.h>
#include <internal/pycore_interpframe.h>
#include <internal/pycore_stackref.h>
#elif IS_PYTHON_3_14_PLUS && defined(_WIN32)
#include <internal/pycore_interpframe_structs.h> // _PyInterpreterFrame
#endif

#endif

#undef Py_BUILD_CORE

#ifdef __cplusplus
extern "C" {
#endif

#if IS_PYTHON_3_14_PLUS

#define F_CODE(x) \
  ((PyCodeObject*)THP_PyStackRef_AsPyObjectBorrow(&(x)->f_executable))
#define PREV_INSTR(x) (x)->instr_ptr

#else

#if IS_PYTHON_3_13_PLUS
#define F_CODE(x) ((PyCodeObject*)(x)->f_executable)
#define PREV_INSTR(x) (x)->instr_ptr
#else
#define F_CODE(x) ((PyCodeObject*)(x)->f_code)
#define PREV_INSTR(x) (x)->prev_instr
#endif // IS_PYTHON_3_13_PLUS

#endif // IS_PYTHON_3_14_PLUS

#if IS_PYTHON_3_14_PLUS
#define FUNC(x) \
  ((PyFunctionObject*)THP_PyStackRef_AsPyObjectBorrow(&(x)->f_funcobj))
#elif IS_PYTHON_3_12_PLUS
#define FUNC(x) ((PyFunctionObject*)(x)->f_funcobj)
#else
#define FUNC(x) ((PyFunctionObject*)(x)->f_func)
#endif

#ifdef __cplusplus
} // extern "C"
#endif
