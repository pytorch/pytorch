#pragma once

#include <torch/csrc/utils/python_compat.h>

// Problem in CPython includes when mixing core and non-core build
// The fix was not backported to 3.12 so this is needed here
// https://github.com/python/cpython/issues/105268
#if IS_PYTHON_3_12_PLUS
#undef _PyGC_FINALIZED
#endif

// see https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
#define Py_BUILD_CORE

#ifndef __cplusplus
// C-only headers
#include <internal/pycore_pystate.h>

#endif // __cplusplus

#if IS_PYTHON_3_11_PLUS
#include <internal/pycore_frame.h>
#endif

#undef Py_BUILD_CORE
#endif // PY_VERSION_HEX >= 0x03080000

#ifdef __cplusplus
extern "C" {
#endif

#if IS_PYTHON_3_13_PLUS
#define F_CODE(x) ((PyCodeObject*)(x)->f_executable)
#define PREV_INSTR(x) (x)->instr_ptr
#else
#define F_CODE(x) ((PyCodeObject*)(x)->f_code)
#define PREV_INSTR(x) (x)->prev_instr
#endif

#if IS_PYTHON_3_12_PLUS
#define FUNC(x) ((x)->f_funcobj)
#else
#define FUNC(x) ((x)->f_func)
#endif

#ifdef __cplusplus
} // extern "C"
#endif
