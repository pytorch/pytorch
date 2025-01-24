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

// reference to random module
// returns borrowed reference
PyObject* random_module();

// rng.getstate()
// rng can be random module or random.Random object
// rng: borrowed reference
// returns new reference
PyObject* random_getstate(PyObject* rng);

// rng.setstate(state)
// rng can be random module or random.Random object
// rng, state: borrowed references
// no return value
void random_setstate(PyObject* rng, PyObject* state);

#ifdef __cplusplus

} // extern "C"

PyObject* torch_c_dynamo_utils_init();
} // namespace torch::dynamo

#endif // __cplusplus
