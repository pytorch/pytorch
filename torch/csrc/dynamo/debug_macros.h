#pragma once

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

// e.g. INSPECT(obj1, obj2)
// in order to inspect obj1, obj2 at the Python level, in pdb.
// Alias for torch._dynamo.utils._breakpoint_for_c_dynamo(...)
// WARNING: makes a Python function call! Make sure eval frame callback is unset!
#define INSPECT(...) \
  { \
    PyObject *torch__dynamo_utils_module = PyImport_ImportModule("torch._dynamo.utils"); \
    NULL_CHECK(torch__dynamo_utils_module); \
    PyObject *breakpoint_for_c_dynamo_fn = PyObject_GetAttrString(torch__dynamo_utils_module, "_breakpoint_for_c_dynamo"); \
    NULL_CHECK(breakpoint_for_c_dynamo_fn); \
    PyObject_CallFunctionObjArgs(breakpoint_for_c_dynamo_fn, __VA_ARGS__, NULL); \
    Py_DECREF(breakpoint_for_c_dynamo_fn); \
    Py_DECREF(torch__dynamo_utils_module); \
  }

#ifdef __cplusplus
} // extern "C"
#endif
