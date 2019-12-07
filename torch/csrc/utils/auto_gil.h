#pragma once

// RAII structs to acquire and release Python's global interpreter lock (GIL)

#include <torch/csrc/python_headers.h>

// TODO: Deprecate these structs after we land this diff
// (to avoid -Werror failures)

// Acquires the GIL on construction
struct /* [[deprecated(
    "Use pybind11::gil_scoped_acquire instead")]] */ AutoGIL {
  AutoGIL() : gstate(PyGILState_Ensure()) {}
  ~AutoGIL() {
    PyGILState_Release(gstate);
  }

  PyGILState_STATE gstate;
};

// Releases the GIL on construction
struct /* [[deprecated(
    "Use pybind11::gil_scoped_release instead")]] */ AutoNoGIL {
  AutoNoGIL() : save(PyEval_SaveThread()) {}
  ~AutoNoGIL() {
    PyEval_RestoreThread(save);
  }

  PyThreadState* save;
};

// Runs the function without the GIL
template <typename F>
/* [[deprecated]] */ inline void with_no_gil(F f) {
  // TODO: The deprecation here triggers a deprecated use warning
  // on some versions of compilers; need to avoid this
  AutoNoGIL no_gil;
  f();
}
