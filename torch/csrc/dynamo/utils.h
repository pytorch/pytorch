#pragma once

#ifdef __cplusplus

#include <torch/csrc/python_headers.h>
// C2039 MSVC
#include <pybind11/complex.h>
#include <torch/csrc/utils/pybind.h>

#endif // __cplusplus

#include <Python.h>

#ifdef __cplusplus

// The visibility attribute is to avoid a warning about storing a field in the
// struct that has a different visibility (from pybind) than the struct.
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

namespace torch::dynamo {

extern "C" {

#endif // __cplusplus

// random.getstate()
// returns new reference
PyObject* system_random_getstate();

// random.setstate(state)
// state: borrowed references
// no return value
void system_random_setstate(PyObject* state);

#ifdef __cplusplus

} // extern "C"

PyObject* torch_c_dynamo_utils_init();
} // namespace torch::dynamo

#endif // __cplusplus
