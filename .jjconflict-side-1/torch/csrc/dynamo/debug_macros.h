#pragma once

#include <torch/csrc/utils/python_compat.h>

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define unlikely(x) (x)
#else
#define unlikely(x) __builtin_expect((x), 0)
#endif

#define NULL_CHECK(val)                                         \
  if (unlikely((val) == NULL)) {                                \
    fprintf(stderr, "NULL ERROR: %s:%d\n", __FILE__, __LINE__); \
    PyErr_Print();                                              \
    abort();                                                    \
  } else {                                                      \
  }

// CHECK might be previously declared
#undef CHECK
#define CHECK(cond)                                                     \
  if (unlikely(!(cond))) {                                              \
    fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__); \
    abort();                                                            \
  } else {                                                              \
  }

// Uncomment next line to print debug message
// #define TORCHDYNAMO_DEBUG 1
#ifdef TORCHDYNAMO_DEBUG

#define DEBUG_CHECK(cond) CHECK(cond)
#define DEBUG_NULL_CHECK(val) NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...) \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__, __VA_ARGS__)
#define DEBUG_TRACE0(msg) \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__)

#else

#define DEBUG_CHECK(cond)
#define DEBUG_NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...)
#define DEBUG_TRACE0(msg)

#endif

inline _PyFrameEvalFunction _debug_set_eval_frame(
    PyThreadState* tstate,
    _PyFrameEvalFunction eval_frame) {
  _PyFrameEvalFunction prev =
      _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
  _PyInterpreterState_SetEvalFrameFunc(tstate->interp, eval_frame);
  return prev;
}

// Inspect PyObject*'s from C/C++ at the Python level, in pdb.
// e.g.
//
// PyObject* obj1 = PyList_New(...);
// PyObject* obj2 = PyObject_CallFunction(...);
// INSPECT(obj1, obj2);
// (pdb) p args[0]
// # list
// (pdb) p args[1]
// # some object
// (pdb) p args[1].some_attr
// # etc.
//
// Implementation: set eval frame callback to default, call
// torch._dynamo.utils._breakpoint_for_c_dynamo, reset eval frame callback.
#define INSPECT(...)                                                  \
  {                                                                   \
    PyThreadState* cur_tstate = PyThreadState_Get();                  \
    _PyFrameEvalFunction prev_eval_frame =                            \
        _debug_set_eval_frame(cur_tstate, &_PyEval_EvalFrameDefault); \
    PyObject* torch__dynamo_utils_module =                            \
        PyImport_ImportModule("torch._dynamo.utils");                 \
    NULL_CHECK(torch__dynamo_utils_module);                           \
    PyObject* breakpoint_for_c_dynamo_fn = PyObject_GetAttrString(    \
        torch__dynamo_utils_module, "_breakpoint_for_c_dynamo");      \
    NULL_CHECK(breakpoint_for_c_dynamo_fn);                           \
    PyObject_CallFunctionObjArgs(                                     \
        breakpoint_for_c_dynamo_fn, __VA_ARGS__, NULL);               \
    _debug_set_eval_frame(cur_tstate, prev_eval_frame);               \
    Py_DECREF(breakpoint_for_c_dynamo_fn);                            \
    Py_DECREF(torch__dynamo_utils_module);                            \
  }

#ifdef __cplusplus
} // extern "C"
#endif
